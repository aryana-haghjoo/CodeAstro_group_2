import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from zestimatr.model import ZHead1D
from zestimatr.inference import predict_redshifts, predict, load_model, resample_flux


@pytest.fixture
def trained_model_and_params():
    model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2, dropout=0.0)
    model.eval()
    norm_params = {
        "z_mean": 2.0,
        "z_std": 1.0,
        "z_min": 0.0,
        "z_max": 4.0,
        "wavelength_grid": np.linspace(1.0, 5.0, 500),
    }
    return model, norm_params


@pytest.fixture
def dummy_dataloader():
    flux = torch.randn(20, 500)
    z_true = torch.rand(20) * 4.0
    ds = TensorDataset(flux, z_true)
    return DataLoader(ds, batch_size=8)


class TestPredictRedshifts:

    def test_output_keys(self, trained_model_and_params, dummy_dataloader):
        model, params = trained_model_and_params
        result = predict_redshifts(model, dummy_dataloader, params,
                                   device="cpu")
        assert "z_pred" in result
        assert "z_uncertainty" in result
        assert "z_true" in result

    def test_output_shapes(self, trained_model_and_params, dummy_dataloader):
        model, params = trained_model_and_params
        result = predict_redshifts(model, dummy_dataloader, params,
                                   device="cpu")
        assert result["z_pred"].shape == (20,)
        assert result["z_uncertainty"].shape == (20,)
        assert result["z_true"].shape == (20,)

    def test_predictions_finite(self, trained_model_and_params,
                                dummy_dataloader):
        model, params = trained_model_and_params
        result = predict_redshifts(model, dummy_dataloader, params,
                                   device="cpu")
        assert np.all(np.isfinite(result["z_pred"]))
        assert np.all(np.isfinite(result["z_uncertainty"]))

    def test_uncertainties_positive(self, trained_model_and_params,
                                   dummy_dataloader):
        model, params = trained_model_and_params
        result = predict_redshifts(model, dummy_dataloader, params,
                                   device="cpu")
        assert np.all(result["z_uncertainty"] > 0)

    def test_no_ground_truth(self, trained_model_and_params):
        model, params = trained_model_and_params
        flux = torch.randn(5, 500)
        z_dummy = torch.full((5,), -1.0)
        ds = TensorDataset(flux, z_dummy)
        loader = DataLoader(ds, batch_size=5)
        result = predict_redshifts(model, loader, params, device="cpu")
        assert "z_pred" in result
        assert "z_true" not in result


class TestLoadModel:

    def test_roundtrip_save_load(self, tmp_path):
        model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2)
        config = {"hidden_dim": 16, "num_blocks": 2, "dropout": 0.1}
        ckpt_path = str(tmp_path / "test_ckpt.pth")
        torch.save({
            "zhead_state_dict": model.state_dict(),
            "z_mean": 2.0,
            "z_std": 1.0,
            "z_min": 0.0,
            "z_max": 4.0,
            "config": config,
        }, ckpt_path)

        loaded_model, norm_params = load_model(ckpt_path, device="cpu")
        assert isinstance(loaded_model, ZHead1D)
        assert norm_params["z_mean"] == 2.0
        assert norm_params["z_std"] == 1.0
        assert norm_params["z_min"] == 0.0
        assert norm_params["z_max"] == 4.0

    def test_loaded_model_is_eval(self, tmp_path):
        model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2)
        ckpt_path = str(tmp_path / "test_ckpt.pth")
        torch.save({
            "zhead_state_dict": model.state_dict(),
            "z_mean": 1.0, "z_std": 1.0,
            "z_min": 0.0, "z_max": 3.0,
            "config": {"hidden_dim": 16, "num_blocks": 2, "dropout": 0.1},
        }, ckpt_path)

        loaded, _ = load_model(ckpt_path, device="cpu")
        assert not loaded.training
        for p in loaded.parameters():
            assert not p.requires_grad

    def test_wavelength_grid_from_checkpoint(self, tmp_path):
        model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2)
        wl = np.linspace(0.5, 3.0, 1000)
        ckpt_path = str(tmp_path / "test_ckpt.pth")
        torch.save({
            "zhead_state_dict": model.state_dict(),
            "z_mean": 1.0, "z_std": 1.0,
            "z_min": 0.0, "z_max": 3.0,
            "wavelength_grid": wl,
            "config": {"hidden_dim": 16, "num_blocks": 2, "dropout": 0.1},
        }, ckpt_path)

        _, params = load_model(ckpt_path, device="cpu")
        np.testing.assert_array_equal(params["wavelength_grid"], wl)

    def test_wavelength_grid_default_fallback(self, tmp_path):
        model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2)
        ckpt_path = str(tmp_path / "test_ckpt.pth")
        torch.save({
            "zhead_state_dict": model.state_dict(),
            "z_mean": 1.0, "z_std": 1.0,
            "z_min": 0.0, "z_max": 3.0,
            "config": {"hidden_dim": 16, "num_blocks": 2, "dropout": 0.1},
        }, ckpt_path)

        _, params = load_model(ckpt_path, device="cpu")
        expected = np.linspace(1.0, 5.0, 2500)
        np.testing.assert_array_almost_equal(params["wavelength_grid"], expected)


class TestResampleFlux:

    def test_identity_resample(self):
        wl = np.linspace(1.0, 5.0, 100)
        flux = np.sin(wl)
        resampled = resample_flux(wl, flux, wl)
        np.testing.assert_allclose(resampled, flux, atol=1e-5)

    def test_upsample(self):
        wl_coarse = np.linspace(1.0, 5.0, 50)
        flux_coarse = np.sin(wl_coarse)
        wl_fine = np.linspace(1.0, 5.0, 200)
        resampled = resample_flux(wl_coarse, flux_coarse, wl_fine)
        assert resampled.shape == (200,)
        expected = np.sin(wl_fine)
        np.testing.assert_allclose(resampled, expected, atol=0.01)

    def test_downsample(self):
        wl_fine = np.linspace(1.0, 5.0, 500)
        flux_fine = np.cos(wl_fine)
        wl_coarse = np.linspace(1.0, 5.0, 50)
        resampled = resample_flux(wl_fine, flux_fine, wl_coarse)
        assert resampled.shape == (50,)
        np.testing.assert_allclose(resampled, np.cos(wl_coarse), atol=1e-4)

    def test_batch(self):
        wl = np.linspace(1.0, 5.0, 100)
        flux = np.stack([np.sin(wl), np.cos(wl)])
        target_wl = np.linspace(1.0, 5.0, 200)
        resampled = resample_flux(wl, flux, target_wl)
        assert resampled.shape == (2, 200)

    def test_different_range(self):
        wl_in = np.linspace(0.5, 3.0, 100)
        flux = np.ones(100)
        target = np.linspace(1.0, 5.0, 200)
        resampled = resample_flux(wl_in, flux, target)
        assert resampled.shape == (200,)
        assert np.all(np.isfinite(resampled))


class TestPredict:

    def test_single_spectrum(self, trained_model_and_params):
        model, params = trained_model_and_params
        flux = np.random.randn(500)
        result = predict(flux, model, params, device="cpu")
        assert isinstance(result["z_pred"], (float, np.floating))
        assert isinstance(result["z_uncertainty"], (float, np.floating))

    def test_batch_spectra(self, trained_model_and_params):
        model, params = trained_model_and_params
        flux = np.random.randn(10, 500)
        result = predict(flux, model, params, device="cpu")
        assert result["z_pred"].shape == (10,)
        assert result["z_uncertainty"].shape == (10,)

    def test_with_wavelength_resampling(self, trained_model_and_params):
        model, params = trained_model_and_params
        user_wl = np.linspace(0.8, 5.5, 300)
        flux = np.random.randn(300)
        result = predict(flux, model, params, wavelength=user_wl,
                         device="cpu")
        assert np.isfinite(result["z_pred"])
        assert np.isfinite(result["z_uncertainty"])

    def test_with_matching_wavelength(self, trained_model_and_params):
        model, params = trained_model_and_params
        flux = np.random.randn(500)
        result = predict(flux, model, params,
                         wavelength=params["wavelength_grid"],
                         device="cpu")
        assert np.isfinite(result["z_pred"])

    def test_wavelength_length_mismatch_raises(self, trained_model_and_params):
        model, params = trained_model_and_params
        flux = np.random.randn(500)
        bad_wl = np.linspace(1.0, 5.0, 300)
        with pytest.raises(ValueError, match="wavelength length"):
            predict(flux, model, params, wavelength=bad_wl, device="cpu")
