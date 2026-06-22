#!/usr/bin/env python3
"""
infer_zhead.py — Inference for high-res oracle redshift prediction

Usage:

python infer_zhead.py --zhead_ckpt best_zhead_hires.pth --data ../../data/spectra_dataset_2500.npz --output predictions.npz --plot

This script:
1. Loads trained Z-head model
2. Runs inference on high-resolution spectra directly
3. Outputs redshift predictions with uncertainties
4. Optionally creates validation plots if ground truth is available
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from model_z_head import ZHead1D


# ---------------- Dataset ----------------
class InferenceDataset(Dataset):
    """
    Simple dataset for inference.
    Expects .npz file with 'flux_high' and optionally redshift keys.
    Normalizes flux per-spectrum to match training data preprocessing.
    """
    def __init__(self, npz_path, redshift_key=None):
        data = np.load(npz_path)

        print(f"\nLoading dataset from {npz_path}")
        print(f"   Available keys: {list(data.keys())}")

        # Load flux
        if 'flux_high' in data:
            flux_raw = data['flux_high'].astype(np.float32)
        elif 'flux_hi' in data:
            flux_raw = data['flux_hi'].astype(np.float32)
        else:
            raise KeyError(f"Could not find high-resolution flux. Available keys: {list(data.keys())}")

        # Normalize each spectrum individually (same as training)
        self.flux = []
        for i in range(len(flux_raw)):
            f = flux_raw[i]
            mean = np.nanmean(f)
            std = np.nanstd(f)
            if std < 1e-25:
                std = 1e-25
            f_norm = (f - mean) / std
            self.flux.append(f_norm)
        self.flux = np.array(self.flux, dtype=np.float32)
        print(f"   Normalized {len(self.flux)} spectra (per-spectrum mean=0, std=1)")

        # Try to find redshift with common key names
        if redshift_key is not None:
            if redshift_key in data:
                self.redshift = data[redshift_key].astype(np.float32)
                self.has_ground_truth = True
                print(f"   Using '{redshift_key}' for ground truth redshifts")
            else:
                raise KeyError(f"Specified redshift key '{redshift_key}' not found in dataset")
        else:
            possible_keys = ['redshift', 'z', 'z_true', 'redshifts', 'z_spec']
            self.has_ground_truth = False

            for key in possible_keys:
                if key in data:
                    self.redshift = data[key].astype(np.float32)
                    self.has_ground_truth = True
                    print(f"   Found ground truth redshifts under key: '{key}'")
                    break

            if not self.has_ground_truth:
                self.redshift = None
                print(f"   No ground truth redshifts found. Tried: {possible_keys}")

        print(f"   Total spectra: {len(self.flux)}")
        print(f"   Flux shape: {self.flux.shape}")
        if self.has_ground_truth:
            print(f"   Redshift range: [{self.redshift.min():.3f}, {self.redshift.max():.3f}]")

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        flux = torch.from_numpy(self.flux[idx]).float()
        if self.has_ground_truth:
            z = float(self.redshift[idx])
            return flux, z
        else:
            return flux, -1.0


# ---------------- Model loading ----------------
def load_model(zhead_ckpt_path, device, fallback_data_path=None):
    """
    Load Z-head model from checkpoint.

    Returns:
        zhead, normalization_params
    """
    print(f"\nLoading model from {zhead_ckpt_path}")

    ckpt = torch.load(zhead_ckpt_path, map_location="cpu")

    saved_config = ckpt['config']
    z_mean = ckpt['z_mean']
    z_std = ckpt['z_std']

    z_min = ckpt.get('z_min', None)
    z_max = ckpt.get('z_max', None)

    if z_min is None or z_max is None:
        print("z_min/z_max not in checkpoint, attempting to compute from training data...")
        candidate_paths = []
        cfg_data_path = saved_config.get('data')
        if cfg_data_path:
            candidate_paths.append(cfg_data_path)
            candidate_paths.append(os.path.join(os.path.dirname(zhead_ckpt_path), cfg_data_path))
        if fallback_data_path:
            candidate_paths.append(fallback_data_path)

        tried = set()
        for data_path in candidate_paths:
            if not data_path:
                continue
            resolved = os.path.abspath(data_path)
            if resolved in tried:
                continue
            tried.add(resolved)
            if not os.path.exists(resolved):
                continue
            try:
                data = np.load(resolved, allow_pickle=True)
                for key in ['z', 'redshift', 'z_true', 'redshifts']:
                    if key in data:
                        z_all = data[key]
                        z_min = float(z_all.min())
                        z_max = float(z_all.max())
                        print(f"Computed z_min={z_min:.4f}, z_max={z_max:.4f} from {resolved}")
                        break
                if z_min is not None and z_max is not None:
                    break
            except Exception as e:
                print(f"Could not load training data from {resolved}: {e}")

        if z_min is None or z_max is None:
            raise ValueError(
                "Could not determine z_min and z_max!\n"
                "  - Not found in checkpoint\n"
                "  - Training data not accessible\n"
                "Please ensure training data is available or retrain."
            )

    # Build Z-head
    zhead = ZHead1D(
        in_channels=1,
        hidden_dim=saved_config.get('hidden_dim', 64),
        num_blocks=saved_config.get('num_blocks', 4),
        dropout=saved_config.get('dropout', 0.1)
    ).to(device)

    zhead.load_state_dict(ckpt['zhead_state_dict'])
    print(f"Loaded Z-head: hidden_dim={saved_config.get('hidden_dim', 64)}, "
          f"num_blocks={saved_config.get('num_blocks', 4)}")

    zhead.eval()
    for p in zhead.parameters():
        p.requires_grad = False

    norm_params = {
        'z_mean': z_mean,
        'z_std': z_std,
        'z_min': z_min,
        'z_max': z_max,
    }

    print(f"z_mean={z_mean:.4f}, z_std={z_std:.4f}, z_range=[{z_min:.4f},{z_max:.4f}]")

    return zhead, norm_params


@torch.no_grad()
def predict_redshifts(zhead, dataloader, norm_params, device):
    """Run inference on dataset."""
    z_mean = norm_params['z_mean']
    z_std = norm_params['z_std']
    z_min = norm_params['z_min']
    z_max = norm_params['z_max']

    z_min_n = (z_min - z_mean) / z_std
    z_max_n = (z_max - z_mean) / z_std

    all_z_pred = []
    all_z_std = []
    all_z_true = []
    has_gt = False

    print("\nRunning inference...")
    for flux, z_true in tqdm(dataloader, desc="Predicting"):
        flux = flux.to(device).unsqueeze(1)  # (B, 1, L_high)

        mu_raw, logvar_n = zhead(flux)

        # Denormalize
        mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
        z_pred = mu_n * z_std + z_mean

        # Uncertainty
        z_var = torch.exp(logvar_n.clamp(min=-12, max=12)) * (z_std ** 2)
        z_uncertainty = torch.sqrt(z_var)

        all_z_pred.append(z_pred.cpu())
        all_z_std.append(z_uncertainty.cpu())

        if z_true[0] != -1.0:
            has_gt = True
            all_z_true.append(z_true)

    predictions = {
        'z_pred': torch.cat(all_z_pred).numpy(),
        'z_uncertainty': torch.cat(all_z_std).numpy(),
    }

    if has_gt:
        predictions['z_true'] = torch.cat(all_z_true).numpy()

    return predictions


def compute_metrics(z_pred, z_true):
    abs_err = np.abs(z_pred - z_true)
    mae = abs_err.mean()
    rmse = np.sqrt(((z_pred - z_true) ** 2).mean())

    med_dz = np.median(abs_err)
    nmad = 1.4826 * med_dz

    rel_err = abs_err / (1.0 + np.abs(z_true))
    med_rel_err = np.median(rel_err)
    outlier_rate = (rel_err > 0.15).mean()

    return {
        'mae': mae,
        'rmse': rmse,
        'median_abs_error': med_dz,
        'nmad': nmad,
        'median_rel_error': med_rel_err,
        'outlier_rate': outlier_rate
    }


def plot_predictions(predictions, output_path=None):
    if 'z_true' not in predictions:
        print("No ground truth available, skipping plots")
        return

    z_pred = predictions['z_pred']
    z_true = predictions['z_true']
    z_std = predictions['z_uncertainty']

    metrics = compute_metrics(z_pred, z_true)

    dz_over_1pz = (z_pred - z_true) / (1.0 + np.abs(z_true))
    abs_dz_over_1pz = np.abs(dz_over_1pz)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Predicted vs True
    vmax = np.percentile(abs_dz_over_1pz, 99)
    sc = ax1.scatter(z_true, z_pred, c=abs_dz_over_1pz, s=10, alpha=0.6,
                     vmin=0, vmax=vmax, cmap='viridis')

    z_min, z_max = z_true.min(), z_true.max()
    ax1.plot([z_min, z_max], [z_min, z_max], 'r--', lw=2, label='Perfect prediction')

    ax1.set_xlabel('True Redshift', fontsize=12)
    ax1.set_ylabel('Predicted Redshift', fontsize=12)
    ax1.set_title('High-res oracle: Predicted vs True', fontsize=14)
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('|dz|/(1+z)', fontsize=10)

    metrics_text = f"MAE: {metrics['mae']:.4f}\n"
    metrics_text += f"RMSE: {metrics['rmse']:.4f}\n"
    metrics_text += f"NMAD: {metrics['nmad']:.4f}\n"
    metrics_text += f"Median |dz|/(1+z): {metrics['median_rel_error']:.4f}\n"
    metrics_text += f"Outlier rate: {metrics['outlier_rate']:.2%}"
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

    # Plot 2: Histogram of relative errors
    ax2.hist(dz_over_1pz, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax2.axvline(0.15, color='orange', linestyle=':', lw=2, label='Outlier threshold')
    ax2.axvline(-0.15, color='orange', linestyle=':', lw=2)

    ax2.set_xlabel('dz/(1+z)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Relative Errors', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    stat_text = f"Mean: {dz_over_1pz.mean():.4f}\n"
    stat_text += f"Median: {np.median(dz_over_1pz):.4f}\n"
    stat_text += f"Std: {dz_over_1pz.std():.4f}"
    ax2.text(0.95, 0.95, stat_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="High-res oracle redshift inference")

    parser.add_argument("--zhead_ckpt", type=str, required=True,
                        help="Path to best_zhead_hires.pth checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .npz file with 'flux_high' and optionally 'redshift'")
    parser.add_argument("--redshift_key", type=str, default=None,
                        help="Name of redshift key in .npz file (auto-detects if not specified)")
    parser.add_argument("--output", type=str, default="predictions.npz",
                        help="Output path for predictions")
    parser.add_argument("--plot", action="store_true",
                        help="Create validation plots (requires ground truth)")
    parser.add_argument("--plot_path", type=str, default="predictions.png",
                        help="Path to save plot")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    zhead, norm_params = load_model(args.zhead_ckpt, device, fallback_data_path=args.data)

    # Load data
    dataset = InferenceDataset(args.data, redshift_key=args.redshift_key)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)

    # Run inference
    predictions = predict_redshifts(zhead, dataloader, norm_params, device)

    # Save predictions
    np.savez(args.output, **predictions)
    print(f"\nPredictions saved to {args.output}")
    print(f"   - z_pred: shape {predictions['z_pred'].shape}")
    print(f"   - z_uncertainty: shape {predictions['z_uncertainty'].shape}")
    if 'z_true' in predictions:
        print(f"   - z_true: shape {predictions['z_true'].shape}")

    # Summary
    print(f"\nPrediction summary:")
    print(f"   Mean predicted z: {predictions['z_pred'].mean():.4f} +/- {predictions['z_pred'].std():.4f}")
    print(f"   Range: [{predictions['z_pred'].min():.4f}, {predictions['z_pred'].max():.4f}]")
    print(f"   Median uncertainty: {np.median(predictions['z_uncertainty']):.4f}")

    # Metrics if ground truth available
    if 'z_true' in predictions:
        metrics = compute_metrics(predictions['z_pred'], predictions['z_true'])
        print(f"\nEvaluation metrics:")
        print(f"   MAE: {metrics['mae']:.6f}")
        print(f"   RMSE: {metrics['rmse']:.6f}")
        print(f"   NMAD: {metrics['nmad']:.6f}")
        print(f"   Median |dz|/(1+z): {metrics['median_rel_error']:.6f}")
        print(f"   Outlier rate (|dz|/(1+z) > 0.15): {metrics['outlier_rate']:.2%}")

    # Plots
    if args.plot:
        plot_predictions(predictions, args.plot_path if 'z_true' in predictions else None)

    print("\nInference complete!")


if __name__ == "__main__":
    main()
