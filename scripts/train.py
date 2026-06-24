#!/usr/bin/env python3
"""
train.py -- Train the ZHead1D redshift estimator on high-resolution spectra.

Usage:
    python scripts/train.py --train_data data/train_DR4.npz --eval_data data/eval_DR4.npz
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml
import matplotlib.pyplot as plt

from zestimatr.dataset import SpectraForZHead
from zestimatr.model import ZHead1D, heteroscedastic_nll
from zestimatr.metrics import compute_metrics, compute_calibration_metrics


# ---- config helpers ----

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping/dict.")
    return cfg


def parse_wandb_config(cfg: dict) -> dict:
    flat = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and "value" in v:
            flat[k] = v["value"]
        else:
            flat[k] = v
    return flat


def set_defaults_from_config(parser: argparse.ArgumentParser, cfg: dict):
    valid_dests = {a.dest for a in parser._actions}
    filtered = {k: v for k, v in cfg.items() if k in valid_dests}
    parser.set_defaults(**filtered)



# ---- debug helper ----

def finfo(name, a: np.ndarray):
    a = np.asarray(a)
    return {
        f"{name}/finite_frac": float(np.isfinite(a).mean()),
        f"{name}/nan_count": int(np.isnan(a).sum()),
        f"{name}/inf_count": int(np.isinf(a).sum()),
        f"{name}/min": float(np.nanmin(a)) if np.isfinite(a).any() else np.nan,
        f"{name}/max": float(np.nanmax(a)) if np.isfinite(a).any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--train_data", type=str, default="data/train_DR4.npz")
    ap.add_argument("--eval_data", type=str, default="data/eval_DR4.npz")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_blocks", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    ap.add_argument("--z_var_floor", type=float, default=1e-6)

    ap.add_argument("--wandb_project", type=str, default="super_resolution")
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])

    if os.path.exists("config.yaml"):
        cfg = load_yaml_config("config.yaml")
        cfg = parse_wandb_config(cfg)
        set_defaults_from_config(ap, cfg)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset (pre-split: no data leakage) ----
    train_ds = SpectraForZHead(args.train_data)
    val_ds = SpectraForZHead(args.eval_data)
    print(f"Loaded: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # ---- Z normalization ----
    z_train = [train_ds[i][1] for i in range(len(train_ds))]
    z_mean, z_std = float(np.mean(z_train)), float(np.std(z_train))
    z_min, z_max = float(np.min(z_train)), float(np.max(z_train))
    z_min_n = (z_min - z_mean) / z_std
    z_max_n = (z_max - z_mean) / z_std
    print(f"z_mean={z_mean:.4f}, z_std={z_std:.4f}, z_range=[{z_min:.4f},{z_max:.4f}]")

    # ---- Model ----
    zhead = ZHead1D(
        in_channels=1,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in zhead.parameters() if p.requires_grad)
    print(f"Z-head: hidden_dim={args.hidden_dim}, num_blocks={args.num_blocks}, "
          f"dropout={args.dropout}, params={n_params:,}")

    opt = torch.optim.AdamW(zhead.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # ---- W&B ----
    wandb.init(project=args.wandb_project, name=args.wandb_name,
               config=vars(args), mode=args.wandb_mode)
    wandb.define_metric("epoch")
    wandb.define_metric("train_*", step_metric="epoch")
    wandb.define_metric("val_*", step_metric="epoch")

    best_val = 1e30

    for epoch in range(args.epochs):
        E = epoch + 1

        # ---- train ----
        zhead.train()
        tr_loss = 0.0

        for x_high, z in tqdm(train_loader, desc=f"Epoch {E}/{args.epochs} [train]"):
            x_high = x_high.to(device).unsqueeze(1)
            z = z.to(device)

            z_n = (z - z_mean) / z_std
            mu_raw, logvar_n = zhead(x_high)

            logvar_n = torch.clamp(logvar_n, min=-12.0, max=12.0)
            mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)

            loss = heteroscedastic_nll(mu_n, logvar_n, z_n,
                                       var_floor=args.z_var_floor)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(zhead.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item()

        tr_loss /= max(1, len(train_loader))

        # ---- val ----
        zhead.eval()
        va_loss = 0.0
        all_mu, all_z, all_sig = [], [], []

        with torch.no_grad():
            for x_high, z in tqdm(val_loader, desc=f"Epoch {E}/{args.epochs} [val]"):
                x_high = x_high.to(device).unsqueeze(1)
                z = z.to(device)

                z_n = (z - z_mean) / z_std
                mu_raw, logvar_n = zhead(x_high)

                logvar_n = torch.clamp(logvar_n, min=-12.0, max=12.0)
                mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)

                loss = heteroscedastic_nll(mu_n, logvar_n, z_n,
                                           var_floor=args.z_var_floor)
                va_loss += loss.item()

                mu = mu_n * z_std + z_mean
                sig = torch.sqrt(torch.exp(logvar_n).clamp_min(
                    args.z_var_floor)) * z_std

                all_mu.append(mu.cpu())
                all_z.append(z.cpu())
                all_sig.append(sig.cpu())

        va_loss /= max(1, len(val_loader))

        z_pred_t = torch.cat(all_mu)
        z_true_t = torch.cat(all_z)
        z_sig_t = torch.cat(all_sig)

        z_pred_np = z_pred_t.numpy()
        z_true_np = z_true_t.numpy()
        z_sig_np = z_sig_t.numpy()

        abs_dz = np.abs(z_pred_np - z_true_np)
        abs_dz_over_1pz = abs_dz / (1.0 + np.abs(z_true_np))
        absres_over_sig = abs_dz / (z_sig_np + 1e-12)
        dz = z_pred_np - z_true_np

        dbg = {}
        dbg.update(finfo("val_dbg/z_true", z_true_np))
        dbg.update(finfo("val_dbg/z_pred", z_pred_np))
        dbg.update(finfo("val_dbg/z_sig", z_sig_np))
        dbg.update(finfo("val_dbg/abs_dz_over_1pz", abs_dz_over_1pz))

        m = np.isfinite(z_true_np) & np.isfinite(z_pred_np)
        n_finite = int(m.sum())
        dbg["val_dbg/n_finite_pred_true"] = n_finite

        # Library metrics (numpy-based)
        metrics = compute_metrics(z_pred_np, z_true_np)
        cal_metrics = compute_calibration_metrics(z_pred_np, z_true_np, z_sig_np)

        # W&B table
        table = wandb.Table(
            data=list(zip(z_true_np, z_pred_np, z_sig_np,
                          abs_dz_over_1pz, absres_over_sig)),
            columns=["z_true", "z_pred", "z_sig",
                      "abs_dz_over_1pz", "absres_over_sig"],
        )

        # ---- Matplotlib plots ----
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        if n_finite == 0:
            ax1.text(0.5, 0.5, "No finite points", ha="center",
                     va="center", transform=ax1.transAxes)
        else:
            x, y = z_true_np[m], z_pred_np[m]
            sig = z_sig_np[m]
            c = abs_dz_over_1pz[m]
            vmax = max(float(np.nanpercentile(c, 99)), 1e-6)
            sc1 = ax1.scatter(x, y, c=c, s=5, alpha=0.5,
                              vmin=0.0, vmax=vmax, cmap='viridis')
            zmin = float(min(x.min(), y.min()))
            zmax = float(max(x.max(), y.max()))
            ax1.plot([zmin, zmax], [zmin, zmax], "r--", lw=1.5,
                     label="Perfect prediction")
            ax1.set_xlim(zmin, zmax)
            ax1.set_ylim(zmin, zmax)
            ax1.set_aspect("equal", adjustable="box")
            cb = fig1.colorbar(sc1, ax=ax1)
            cb.set_label("|dz|/(1+z)")
            ax1.legend(loc="upper left")
            metrics_text = (
                f"MAE: {metrics['mae']:.4f}\n"
                f"RMSE: {metrics['rmse']:.4f}\n"
                f"Med |dz|/(1+z): {metrics['median_rel_error']:.4f}\n"
                f"Outlier rate: {metrics['outlier_rate']:.1%}"
            )
            ax1.text(0.95, 0.05, metrics_text, transform=ax1.transAxes,
                     ha="right", va="bottom",
                     bbox=dict(boxstyle="round", facecolor="wheat",
                               alpha=0.7),
                     fontsize=8)
        ax1.set_xlabel("True redshift")
        ax1.set_ylabel("Predicted redshift")
        ax1.set_title(f"Epoch {E}: Predicted vs True Redshift")
        ax1.grid(True, alpha=0.2)

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        m2 = m & np.isfinite(abs_dz_over_1pz)
        if int(m2.sum()) > 0:
            x, y, c = z_true_np[m2], z_pred_np[m2], abs_dz_over_1pz[m2]
            vmax = max(float(np.nanpercentile(c, 99)), 1e-6)
            sc = ax2.scatter(x, y, c=c, s=6, alpha=0.55, vmin=0.0, vmax=vmax)
            zmin = float(min(x.min(), y.min()))
            zmax = float(max(x.max(), y.max()))
            ax2.plot([zmin, zmax], [zmin, zmax], "k--", lw=1)
            ax2.set_xlim(zmin, zmax)
            ax2.set_ylim(zmin, zmax)
            ax2.set_aspect("equal", adjustable="box")
            cb = fig2.colorbar(sc, ax=ax2)
            cb.set_label("|dz|/(1+z)")
        ax2.set_xlabel("True redshift")
        ax2.set_ylabel("Predicted redshift")
        ax2.set_title("Colored by |dz|/(1+z)")

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        m3 = np.isfinite(z_true_np) & np.isfinite(dz)
        if int(m3.sum()) > 0:
            ax3.scatter(z_true_np[m3], dz[m3], s=6, alpha=0.35)
            ax3.axhline(0.0, linestyle="--", linewidth=1)
        ax3.set_xlabel("True redshift")
        ax3.set_ylabel("z_pred - z_true")
        ax3.set_title("Residual vs true z")

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        norm_res = dz / (z_sig_np + 1e-12)
        m4 = np.isfinite(norm_res)
        if int(m4.sum()) > 0:
            ax4.hist(norm_res[m4], bins=50, alpha=0.7, edgecolor='black',
                     density=True)
            x_range = np.linspace(-4, 4, 100)
            ax4.plot(x_range,
                     1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x_range ** 2),
                     'r--', lw=2, label='N(0,1) ideal')
            ax4.axvline(0, color='k', linestyle='--', alpha=0.3)
            ax4.legend()
        ax4.set_xlabel("Normalized residuals")
        ax4.set_ylabel("Density")
        ax4.set_title(
            f"Calibration (std={cal_metrics['calibration_std']:.2f}, target=1.0)")

        resid = np.abs(z_pred_np - z_true_np)
        cal_med = float(np.median(resid / (z_sig_np + 1e-12)))

        # ---- W&B log ----
        wandb.log({
            "epoch": E,
            "val/z_pred_vs_true_table": table,
            "val/z_pred_vs_true": wandb.plot.scatter(
                table, x="z_true", y="z_pred",
                title="Predicted vs True Redshift",
            ),
            "val/z_pred_vs_true_fig": wandb.Image(fig1),
            "val/z_pred_vs_true_fig_dz1pz": wandb.Image(fig2),
            "val/z_residual_vs_true_fig": wandb.Image(fig3),
            "val/calibration_hist": wandb.Image(fig4),
            "train_loss_nll": float(tr_loss),
            "val_loss_nll": float(va_loss),
            "val_mae_z": metrics['mae'],
            "val_rmse_z": metrics['rmse'],
            "val_med_abs_dz_over_1pz": metrics['median_rel_error'],
            "val_p90_abs_dz_over_1pz": metrics['p90_rel_error'],
            "val_cal_med_absres_over_sig": cal_med,
            "val_calibration_std": cal_metrics['calibration_std'],
            "val_calibration_mean": cal_metrics['calibration_mean'],
            "val_frac_1sigma": cal_metrics['frac_1sigma'],
            "val_frac_2sigma": cal_metrics['frac_2sigma'],
            "val_frac_3sigma": cal_metrics['frac_3sigma'],
            "val_outlier_rate": cal_metrics['outlier_rate'],
            "val_median_uncertainty": cal_metrics['median_uncertainty'],
            **dbg,
        }, step=E)

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)

        wandb.run.summary["val_loss_nll"] = float(va_loss)
        wandb.run.summary["val_med_abs_dz_over_1pz"] = metrics['median_rel_error']
        wandb.run.summary["val_calibration_std"] = cal_metrics['calibration_std']

        print(f"\nEpoch {E} Summary:")
        print(f"   Train Loss: {tr_loss:.6f}")
        print(f"   Val Loss: {va_loss:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}, "
              f"Med |dz|/(1+z): {metrics['median_rel_error']:.6f}")
        print(f"   Calibration: std={cal_metrics['calibration_std']:.3f} "
              f"(target: 1.0)")
        print(f"   1-sigma coverage: {cal_metrics['frac_1sigma']:.1%} "
              f"(target: 68%)")
        print(f"   Outlier rate: {cal_metrics['outlier_rate']:.1%}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "zhead_state_dict": zhead.state_dict(),
                "z_mean": z_mean,
                "z_std": z_std,
                "config": vars(args),
            }, "best_zhead_hires.pth")
            wandb.save("best_zhead_hires.pth")
            print(f"   Saved best_zhead_hires.pth "
                  f"(val_loss_nll={best_val:.6f})")

    # ---- Final summary: evaluate best checkpoint on val set ----
    print("\nEvaluating best checkpoint on validation set...")
    best_ckpt = torch.load("best_zhead_hires.pth", map_location=device)
    zhead.load_state_dict(best_ckpt["zhead_state_dict"])
    zhead.eval()

    all_mu, all_z, all_sig = [], [], []
    with torch.no_grad():
        for x_high, z in val_loader:
            x_high = x_high.to(device).unsqueeze(1)
            z = z.to(device)
            mu_raw, logvar_n = zhead(x_high)
            logvar_n = torch.clamp(logvar_n, min=-12.0, max=12.0)
            mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
            mu = mu_n * z_std + z_mean
            sig = torch.sqrt(torch.exp(logvar_n).clamp_min(
                args.z_var_floor)) * z_std
            all_mu.append(mu.cpu())
            all_z.append(z.cpu())
            all_sig.append(sig.cpu())

    z_pred_np = torch.cat(all_mu).numpy()
    z_true_np = torch.cat(all_z).numpy()
    z_sig_np = torch.cat(all_sig).numpy()

    best_metrics = compute_metrics(z_pred_np, z_true_np)
    best_cal = compute_calibration_metrics(z_pred_np, z_true_np, z_sig_np)
    abs_dz_1pz = np.abs(z_pred_np - z_true_np) / (1.0 + np.abs(z_true_np))

    fig_final, ax_final = plt.subplots(figsize=(7, 7))
    vmax = max(float(np.nanpercentile(abs_dz_1pz, 99)), 1e-6)
    sc = ax_final.scatter(z_true_np, z_pred_np, c=abs_dz_1pz, s=5,
                          alpha=0.5, vmin=0.0, vmax=vmax, cmap='viridis')
    zmin = float(min(z_true_np.min(), z_pred_np.min()))
    zmax = float(max(z_true_np.max(), z_pred_np.max()))
    ax_final.plot([zmin, zmax], [zmin, zmax], "r--", lw=2,
                  label="Perfect prediction")
    ax_final.set_xlim(zmin, zmax)
    ax_final.set_ylim(zmin, zmax)
    ax_final.set_aspect("equal", adjustable="box")
    cb = fig_final.colorbar(sc, ax=ax_final)
    cb.set_label("|dz|/(1+z)")
    ax_final.legend(loc="upper left")
    ax_final.grid(True, alpha=0.2)
    ax_final.set_xlabel("True Redshift", fontsize=12)
    ax_final.set_ylabel("Predicted Redshift", fontsize=12)
    ax_final.set_title("Best Model: Predicted vs True Redshift (Validation)",
                       fontsize=13)
    metrics_text = (
        f"MAE: {best_metrics['mae']:.4f}\n"
        f"RMSE: {best_metrics['rmse']:.4f}\n"
        f"NMAD: {best_metrics['nmad']:.4f}\n"
        f"Med |dz|/(1+z): {best_metrics['median_rel_error']:.4f}\n"
        f"Outlier rate: {best_metrics['outlier_rate']:.1%}\n"
        f"Cal. std: {best_cal['calibration_std']:.3f}\n"
        f"1-sigma cov: {best_cal['frac_1sigma']:.1%}"
    )
    ax_final.text(0.95, 0.05, metrics_text, transform=ax_final.transAxes,
                  ha="right", va="bottom",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                  fontsize=9)
    fig_final.tight_layout()

    wandb.log({"val/best_z_pred_vs_true": wandb.Image(fig_final)})
    os.makedirs("plots", exist_ok=True)
    fig_final.savefig("plots/best_predictions.png", dpi=300, bbox_inches="tight")
    plt.close(fig_final)

    print(f"\nBest model metrics:")
    print(f"   MAE: {best_metrics['mae']:.6f}")
    print(f"   RMSE: {best_metrics['rmse']:.6f}")
    print(f"   Med |dz|/(1+z): {best_metrics['median_rel_error']:.6f}")
    print(f"   Outlier rate: {best_metrics['outlier_rate']:.1%}")
    print(f"   Calibration std: {best_cal['calibration_std']:.3f}")
    print(f"Plot saved to plots/best_predictions.png")

    print("\nTraining complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
