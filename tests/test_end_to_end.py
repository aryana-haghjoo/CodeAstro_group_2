import numpy as np
import torch
from torch.utils.data import DataLoader
import pytest
import os

from zestimatr.model import ZHead1D, heteroscedastic_nll
from zestimatr.dataset import SpectraForZHead
from zestimatr.inference import load_model, predict, predict_redshifts
from zestimatr.metrics import compute_metrics, compute_calibration_metrics


N_TRAIN = 200
N_VAL = 50
SEQ_LEN = 500
N_EPOCHS = 15
HIDDEN_DIM = 16
NUM_BLOCKS = 2


@pytest.fixture
def synthetic_data(tmp_path):
    rng = np.random.default_rng(42)
    wl = np.linspace(1.0, 5.0, SEQ_LEN)

    def make_spectra(n):
        z = rng.uniform(0.5, 3.0, size=n).astype(np.float32)
        flux = np.empty((n, SEQ_LEN), dtype=np.float32)
        for i in range(n):
            base = np.sin(2 * np.pi * wl / (1 + z[i])) + 0.5
            flux[i] = base + rng.normal(0, 0.1, SEQ_LEN).astype(np.float32)
        return flux, z

    train_flux, train_z = make_spectra(N_TRAIN)
    val_flux, val_z = make_spectra(N_VAL)

    train_path = str(tmp_path / "train.npz")
    val_path = str(tmp_path / "val.npz")
    np.savez(train_path, flux_high=train_flux, z=train_z, wavelength_high=wl)
    np.savez(val_path, flux_high=val_flux, z=val_z, wavelength_high=wl)

    return train_path, val_path, wl


@pytest.fixture
def trained_checkpoint(synthetic_data, tmp_path):
    train_path, val_path, wl = synthetic_data

    train_ds = SpectraForZHead(train_path)
    val_ds = SpectraForZHead(val_path)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    z_train = [train_ds[i][1].item() for i in range(len(train_ds))]
    z_mean = float(np.mean(z_train))
    z_std = float(np.std(z_train))
    z_min = float(np.min(z_train))
    z_max = float(np.max(z_train))
    z_min_n = (z_min - z_mean) / z_std
    z_max_n = (z_max - z_mean) / z_std

    model = ZHead1D(in_channels=1, hidden_dim=HIDDEN_DIM,
                    num_blocks=NUM_BLOCKS, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(N_EPOCHS):
        model.train()
        for x, z in train_loader:
            x = x.unsqueeze(1)
            z_n = (z - z_mean) / z_std
            mu_raw, logvar_n = model(x)
            logvar_n = torch.clamp(logvar_n, min=-12.0, max=12.0)
            mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
            loss = heteroscedastic_nll(mu_n, logvar_n, z_n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    ckpt_path = str(tmp_path / "best_zhead.pth")
    torch.save({
        "zhead_state_dict": model.state_dict(),
        "z_mean": z_mean,
        "z_std": z_std,
        "z_min": z_min,
        "z_max": z_max,
        "wavelength_grid": wl,
        "config": {
            "hidden_dim": HIDDEN_DIM,
            "num_blocks": NUM_BLOCKS,
            "dropout": 0.1,
        },
    }, ckpt_path)

    return ckpt_path, val_path, wl


class TestEndToEnd:

    def test_train_reduces_loss(self, synthetic_data):
        train_path, _, _ = synthetic_data
        ds = SpectraForZHead(train_path)
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        z_all = [ds[i][1].item() for i in range(len(ds))]
        z_mean, z_std = float(np.mean(z_all)), float(np.std(z_all))
        z_min_n = (float(np.min(z_all)) - z_mean) / z_std
        z_max_n = (float(np.max(z_all)) - z_mean) / z_std

        model = ZHead1D(in_channels=1, hidden_dim=HIDDEN_DIM,
                        num_blocks=NUM_BLOCKS, dropout=0.0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for epoch in range(N_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for x, z in loader:
                x = x.unsqueeze(1)
                z_n = (z - z_mean) / z_std
                mu_raw, logvar_n = model(x)
                logvar_n = torch.clamp(logvar_n, min=-12.0, max=12.0)
                mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
                loss = heteroscedastic_nll(mu_n, logvar_n, z_n)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        assert losses[-1] < losses[0], "Training loss should decrease"

    def test_checkpoint_save_load_predict(self, trained_checkpoint):
        ckpt_path, val_path, wl = trained_checkpoint

        model, norm_params = load_model(ckpt_path, device="cpu")

        assert not any(p.requires_grad for p in model.parameters())
        assert norm_params["wavelength_grid"] is not None

        val_ds = SpectraForZHead(val_path)
        flux = np.stack([val_ds[i][0].numpy() for i in range(len(val_ds))])
        result = predict(flux, model, norm_params, device="cpu")

        assert "z_pred" in result
        assert "z_uncertainty" in result
        assert result["z_pred"].shape == (N_VAL,)
        assert np.all(np.isfinite(result["z_pred"]))
        assert np.all(result["z_uncertainty"] > 0)

    def test_predictions_correlate_with_truth(self, trained_checkpoint):
        ckpt_path, val_path, wl = trained_checkpoint

        model, norm_params = load_model(ckpt_path, device="cpu")

        val_ds = SpectraForZHead(val_path)
        flux = np.stack([val_ds[i][0].numpy() for i in range(len(val_ds))])
        z_true = np.array([val_ds[i][1].item() for i in range(len(val_ds))])

        result = predict(flux, model, norm_params, device="cpu")
        corr = np.corrcoef(result["z_pred"], z_true)[0, 1]
        assert corr > 0.0, f"Predictions should positively correlate with truth (r={corr:.3f})"

    def test_metrics_are_finite(self, trained_checkpoint):
        ckpt_path, val_path, wl = trained_checkpoint

        model, norm_params = load_model(ckpt_path, device="cpu")
        val_ds = SpectraForZHead(val_path)
        flux = np.stack([val_ds[i][0].numpy() for i in range(len(val_ds))])
        z_true = np.array([val_ds[i][1].item() for i in range(len(val_ds))])

        result = predict(flux, model, norm_params, device="cpu")
        metrics = compute_metrics(result["z_pred"], z_true)
        cal = compute_calibration_metrics(
            result["z_pred"], z_true, result["z_uncertainty"])

        for key in ("mae", "rmse", "median_rel_error", "outlier_rate"):
            assert np.isfinite(metrics[key]), f"{key} is not finite"
        for key in ("calibration_std", "calibration_mean", "frac_1sigma"):
            assert np.isfinite(cal[key]), f"{key} is not finite"

    def test_wavelength_resampling_in_pipeline(self, trained_checkpoint):
        ckpt_path, val_path, wl = trained_checkpoint

        model, norm_params = load_model(ckpt_path, device="cpu")

        rng = np.random.default_rng(99)
        different_wl = np.linspace(0.8, 5.2, 600)
        flux_on_different_grid = rng.normal(0, 1, (5, 600)).astype(np.float32)

        result = predict(flux_on_different_grid, model, norm_params,
                         wavelength=different_wl, device="cpu")

        assert result["z_pred"].shape == (5,)
        assert np.all(np.isfinite(result["z_pred"]))
        assert np.all(result["z_uncertainty"] > 0)
