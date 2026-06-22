#!/usr/bin/env python3
"""
train_z_head.py — High-res oracle: predict redshift directly from high-resolution spectra.

Same architecture and training procedure as the low-res baseline,
but using flux_high. This serves as the upper bound on redshift
prediction performance (best possible with the available spectral info).
"""
import os
import time
import argparse
import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import yaml
import matplotlib.pyplot as plt

from dataset_z_head import SpectraForZHead
from model_z_head import ZHead1D, heteroscedastic_nll


# ---------------- config helpers ----------------
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


# ---------------- split helper ----------------
def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_or_make_split(dataset_path, N, train_frac=0.8, seed=42, split_dir="splits"):
    os.makedirs(split_dir, exist_ok=True)
    ds_hash = hash_file(dataset_path)
    split_path = os.path.join(split_dir, f"split_{ds_hash}.npz")

    if os.path.exists(split_path):
        arr = np.load(split_path)
        train_idx = arr["train_idx"]; test_idx = arr["test_idx"]
        return train_idx, test_idx, split_path

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(train_frac * N)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    np.savez(split_path, train_idx=train_idx, test_idx=test_idx, dataset_hash=ds_hash, N=N)
    return train_idx, test_idx, split_path


# ---------------- metrics ----------------
@torch.no_grad()
def compute_metrics(z_pred, z_true):
    abs_err = (z_pred - z_true).abs()
    mae = abs_err.mean().item()
    rmse = torch.sqrt(((z_pred - z_true) ** 2).mean()).item()
    med_dz1pz = (abs_err / (1.0 + z_true.abs())).median().item()
    p90_dz1pz = (abs_err / (1.0 + z_true.abs())).quantile(0.9).item()
    return mae, rmse, med_dz1pz, p90_dz1pz


@torch.no_grad()
def compute_calibration_metrics(z_pred, z_true, z_uncertainties):
    norm_residuals = (z_pred - z_true) / (z_uncertainties + 1e-12)

    calibration_std = norm_residuals.std().item()
    calibration_mean = norm_residuals.mean().item()

    abs_norm_res = norm_residuals.abs()
    frac_1sigma = (abs_norm_res < 1.0).float().mean().item()
    frac_2sigma = (abs_norm_res < 2.0).float().mean().item()
    frac_3sigma = (abs_norm_res < 3.0).float().mean().item()

    abs_err = (z_pred - z_true).abs()
    relative_errors = abs_err / (1.0 + z_true.abs())
    outlier_rate = (relative_errors > 0.15).float().mean().item()

    return {
        'calibration_std': calibration_std,
        'calibration_mean': calibration_mean,
        'frac_1sigma': frac_1sigma,
        'frac_2sigma': frac_2sigma,
        'frac_3sigma': frac_3sigma,
        'outlier_rate': outlier_rate,
        'median_uncertainty': z_uncertainties.median().item(),
    }


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

    ap.add_argument("--config", type=str, default="config.yaml",
                    help="Path to YAML config (CLI overrides YAML).")

    # dataset + splits
    ap.add_argument("--data", type=str, default="../../data/spectra_dataset_2500.npz",
                    help="Path to dataset .npz with flux_high and z")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)

    # z-head arch
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    # training
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    # regularization
    ap.add_argument("--z_var_floor", type=float, default=1e-6)

    # wandb
    ap.add_argument("--wandb_project", type=str, default="super_resolution")
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])

    # Load config if exists
    if os.path.exists("config.yaml"):
        cfg = load_yaml_config("config.yaml")
        cfg = parse_wandb_config(cfg)
        set_defaults_from_config(ap, cfg)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: HIGH-RES oracle (upper bound)")

    # ---- Dataset ----
    full_ds = SpectraForZHead(args.data)
    N = len(full_ds)
    train_idx, test_idx, split_path = get_or_make_split(
        args.data, N, train_frac=args.train_frac, seed=args.seed
    )
    print(f"Split: {len(train_idx)} train, {len(test_idx)} val (from {split_path})")

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ---- Z normalization ----
    z_train = [full_ds[i][1] for i in train_idx]
    z_mean, z_std = float(np.mean(z_train)), float(np.std(z_train))
    z_min, z_max = float(np.min(z_train)), float(np.max(z_train))
    z_min_n = (z_min - z_mean) / z_std
    z_max_n = (z_max - z_mean) / z_std
    print(f"z_mean={z_mean:.4f}, z_std={z_std:.4f}, z_range=[{z_min:.4f},{z_max:.4f}]")

    # ---- Z-head model ----
    zhead = ZHead1D(
        in_channels=1,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in zhead.parameters() if p.requires_grad)
    print(f"Z-head: hidden_dim={args.hidden_dim}, num_blocks={args.num_blocks}, "
          f"dropout={args.dropout}, params={n_params:,}")

    opt = torch.optim.AdamW(zhead.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- W&B ----
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        mode=args.wandb_mode
    )
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
            x_high = x_high.to(device).unsqueeze(1)  # (B, 1, L_high)
            z = z.to(device)

            z_n = (z - z_mean) / z_std
            mu_raw, logvar_n = zhead(x_high)

            logvar_n = torch.clamp(logvar_n, min=-12.0, max=12.0)
            mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)

            loss = heteroscedastic_nll(mu_n, logvar_n, z_n, var_floor=args.z_var_floor)

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

                loss = heteroscedastic_nll(mu_n, logvar_n, z_n, var_floor=args.z_var_floor)
                va_loss += loss.item()

                mu = mu_n * z_std + z_mean
                sig = torch.sqrt(torch.exp(logvar_n).clamp_min(args.z_var_floor)) * z_std

                all_mu.append(mu.cpu())
                all_z.append(z.cpu())
                all_sig.append(sig.cpu())

        va_loss /= max(1, len(val_loader))

        z_pred = torch.cat(all_mu)
        z_true = torch.cat(all_z)
        z_sig  = torch.cat(all_sig)

        z_pred_np = z_pred.numpy()
        z_true_np = z_true.numpy()
        z_sig_np  = z_sig.numpy()

        abs_dz = np.abs(z_pred_np - z_true_np)
        abs_dz_over_1pz = abs_dz / (1.0 + np.abs(z_true_np))
        absres_over_sig = abs_dz / (z_sig_np + 1e-12)
        dz = z_pred_np - z_true_np

        # Debug stats
        dbg = {}
        dbg.update(finfo("val_dbg/z_true", z_true_np))
        dbg.update(finfo("val_dbg/z_pred", z_pred_np))
        dbg.update(finfo("val_dbg/z_sig",  z_sig_np))
        dbg.update(finfo("val_dbg/abs_dz_over_1pz", abs_dz_over_1pz))

        m = np.isfinite(z_true_np) & np.isfinite(z_pred_np)
        n_finite = int(m.sum())
        dbg["val_dbg/n_finite_pred_true"] = n_finite

        # Calibration metrics
        cal_metrics = compute_calibration_metrics(z_pred, z_true, z_sig)

        # W&B table
        table = wandb.Table(
            data=list(zip(z_true_np, z_pred_np, z_sig_np, abs_dz_over_1pz, absres_over_sig)),
            columns=["z_true", "z_pred", "z_sig", "abs_dz_over_1pz", "absres_over_sig"]
        )

        # ---- Matplotlib plots ----
        # (A) z_pred vs z_true
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        if n_finite == 0:
            ax1.text(0.5, 0.5, "No finite points", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        else:
            x = z_true_np[m]; y = z_pred_np[m]
            ax1.scatter(x, y, s=6, alpha=0.35)
            zmin = float(min(x.min(), y.min()))
            zmax = float(max(x.max(), y.max()))
            ax1.plot([zmin, zmax], [zmin, zmax], "k--", lw=1)
            ax1.set_xlim(zmin, zmax); ax1.set_ylim(zmin, zmax)
            ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("True redshift")
        ax1.set_ylabel("Predicted redshift")
        ax1.set_title("High-res oracle: z_pred vs z_true")

        # (B) colored by |dz|/(1+z)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        m2 = m & np.isfinite(abs_dz_over_1pz)
        if int(m2.sum()) == 0:
            ax2.text(0.5, 0.5, "No finite points", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        else:
            x = z_true_np[m2]; y = z_pred_np[m2]; c = abs_dz_over_1pz[m2]
            vmax = float(np.nanpercentile(c, 99)) if np.isfinite(c).any() else 1.0
            vmax = max(vmax, 1e-6)
            sc = ax2.scatter(x, y, c=c, s=6, alpha=0.55, vmin=0.0, vmax=vmax)
            zmin = float(min(x.min(), y.min()))
            zmax = float(max(x.max(), y.max()))
            ax2.plot([zmin, zmax], [zmin, zmax], "k--", lw=1)
            ax2.set_xlim(zmin, zmax); ax2.set_ylim(zmin, zmax)
            ax2.set_aspect("equal", adjustable="box")
            cb = fig2.colorbar(sc, ax=ax2)
            cb.set_label("|dz|/(1+z) (clipped @ p99)")
        ax2.set_xlabel("True redshift")
        ax2.set_ylabel("Predicted redshift")
        ax2.set_title("High-res oracle: colored by |dz|/(1+z)")

        # (C) residual vs true z
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        m3 = np.isfinite(z_true_np) & np.isfinite(dz)
        if int(m3.sum()) == 0:
            ax3.text(0.5, 0.5, "No finite points", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_xlim(0, 1); ax3.set_ylim(-1, 1)
        else:
            ax3.scatter(z_true_np[m3], dz[m3], s=6, alpha=0.35)
            ax3.axhline(0.0, linestyle="--", linewidth=1)
        ax3.set_xlabel("True redshift")
        ax3.set_ylabel("z_pred - z_true")
        ax3.set_title("High-res oracle: residual vs true z")

        # (D) Calibration histogram
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        norm_res = (z_pred_np - z_true_np) / (z_sig_np + 1e-12)
        m4 = np.isfinite(norm_res)
        if int(m4.sum()) > 0:
            ax4.hist(norm_res[m4], bins=50, alpha=0.7, edgecolor='black', density=True)
            x_range = np.linspace(-4, 4, 100)
            ax4.plot(x_range, 1/np.sqrt(2*np.pi) * np.exp(-0.5*x_range**2),
                    'r--', lw=2, label='N(0,1) ideal')
            ax4.axvline(0, color='k', linestyle='--', alpha=0.3)
            ax4.legend()
        ax4.set_xlabel("Normalized residuals: (z_pred - z_true) / sigma_z")
        ax4.set_ylabel("Density")
        ax4.set_title(f"Calibration (std={cal_metrics['calibration_std']:.2f}, target=1.0)")

        # ---- scalar metrics ----
        mae, rmse, med_dz1pz, p90_dz1pz = compute_metrics(z_pred, z_true)
        resid_t = (z_pred - z_true).abs()
        cal_med = (resid_t / (z_sig + 1e-12)).median().item()

        # ---- W&B log ----
        wandb.log({
            "epoch": E,

            "val/z_pred_vs_true_table": table,
            "val/z_pred_vs_true": wandb.plot.scatter(
                table, x="z_true", y="z_pred",
                title="High-res oracle: Predicted vs True Redshift"
            ),

            "val/z_pred_vs_true_fig": wandb.Image(fig1),
            "val/z_pred_vs_true_fig_dz1pz": wandb.Image(fig2),
            "val/z_residual_vs_true_fig": wandb.Image(fig3),
            "val/calibration_hist": wandb.Image(fig4),

            "train_loss_nll": float(tr_loss),
            "val_loss_nll": float(va_loss),
            "val_mae_z": float(mae),
            "val_rmse_z": float(rmse),
            "val_med_abs_dz_over_1pz": float(med_dz1pz),
            "val_p90_abs_dz_over_1pz": float(p90_dz1pz),
            "val_cal_med_absres_over_sig": float(cal_med),

            "val_calibration_std": cal_metrics['calibration_std'],
            "val_calibration_mean": cal_metrics['calibration_mean'],
            "val_frac_1sigma": cal_metrics['frac_1sigma'],
            "val_frac_2sigma": cal_metrics['frac_2sigma'],
            "val_frac_3sigma": cal_metrics['frac_3sigma'],
            "val_outlier_rate": cal_metrics['outlier_rate'],
            "val_median_uncertainty": cal_metrics['median_uncertainty'],

            **dbg,
        }, step=E)

        plt.close(fig1); plt.close(fig2); plt.close(fig3); plt.close(fig4)

        wandb.run.summary["val_loss_nll"] = float(va_loss)
        wandb.run.summary["val_med_abs_dz_over_1pz"] = float(med_dz1pz)
        wandb.run.summary["val_calibration_std"] = cal_metrics['calibration_std']

        print(f"\nEpoch {E} Summary:")
        print(f"   Train Loss: {tr_loss:.6f}")
        print(f"   Val Loss: {va_loss:.6f}")
        print(f"   MAE: {mae:.6f}, Med |dz|/(1+z): {med_dz1pz:.6f}")
        print(f"   Calibration: std={cal_metrics['calibration_std']:.3f} (target: 1.0)")
        print(f"   1-sigma coverage: {cal_metrics['frac_1sigma']:.1%} (target: 68%)")
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
            print(f"   Saved best_zhead_hires.pth (val_loss_nll={best_val:.6f})")

    print("\nTraining complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
