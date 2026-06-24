import numpy as np
import torch
import pytest
import os


from zestimatr.dataset import SpectraForZHead


@pytest.fixture
def sample_npz(tmp_path):
    n, length = 20, 500
    rng = np.random.default_rng(42)
    flux = rng.normal(1e-20, 1e-21, size=(n, length))
    z = rng.uniform(0.5, 3.0, size=n)
    path = tmp_path / "test_spectra.npz"
    np.savez(path, flux_high=flux, z=z)
    return str(path), n, length


class TestSpectraForZHead:

    def test_length(self, sample_npz):
        path, n, _ = sample_npz
        ds = SpectraForZHead(path)
        assert len(ds) == n

    def test_item_types_and_shapes(self, sample_npz):
        path, _, length = sample_npz
        ds = SpectraForZHead(path)
        x, z = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(z, torch.Tensor)
        assert x.shape == (length,)
        assert z.shape == ()

    def test_normalization_zero_mean_unit_std(self, sample_npz):
        path, _, _ = sample_npz
        ds = SpectraForZHead(path, normalize_flux=True)
        x, _ = ds[0]
        assert abs(x.mean().item()) < 0.1
        assert abs(x.std().item() - 1.0) < 0.1

    def test_no_normalization_preserves_scale(self, sample_npz):
        path, _, _ = sample_npz
        ds_raw = SpectraForZHead(path, normalize_flux=False)
        ds_norm = SpectraForZHead(path, normalize_flux=True)
        x_raw, _ = ds_raw[0]
        x_norm, _ = ds_norm[0]
        assert x_raw.std().item() != pytest.approx(x_norm.std().item(), rel=0.1)

    def test_all_items_finite(self, sample_npz):
        path, n, _ = sample_npz
        ds = SpectraForZHead(path)
        for i in range(n):
            x, z = ds[i]
            assert torch.isfinite(x).all()
            assert torch.isfinite(z)

    def test_constant_spectrum_handled(self, tmp_path):
        flux = np.ones((5, 100)) * 1e-20
        z = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        path = tmp_path / "const.npz"
        np.savez(path, flux_high=flux, z=z)
        ds = SpectraForZHead(str(path))
        x, _ = ds[0]
        assert torch.isfinite(x).all()
