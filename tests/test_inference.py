import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from zestimatr.model import ZHead1D
from zestimatr.inference import predict_redshifts, load_model


@pytest.fixture
def trained_model_and_params():
    model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2, dropout=0.0)
    model.eval()
    norm_params = {
        "z_mean": 2.0,
        "z_std": 1.0,
        "z_min": 0.0,
        "z_max": 4.0,
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
