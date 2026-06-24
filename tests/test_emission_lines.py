import numpy as np
import pandas as pd
import pytest

from zestimatr.emission_lines import (
    EMISSION_LINES,
    detect_emission_lines,
    plot_spectrum,
)


def _make_spectrum_with_line(rest_lam, z, n_points=5000):
    """Create a synthetic spectrum with a Gaussian emission line."""
    obs_lam = rest_lam * (1.0 + z)
    wavelength = np.linspace(0.6, 5.3, n_points)
    flux = np.random.default_rng(42).normal(0, 0.1, n_points)
    sigma_w = 0.005
    flux += 50.0 * np.exp(-0.5 * ((wavelength - obs_lam) / sigma_w) ** 2)
    return wavelength, flux


class TestEmissionLineCatalog:

    def test_catalog_not_empty(self):
        assert len(EMISSION_LINES) > 0

    def test_all_wavelengths_positive(self):
        for name, lam in EMISSION_LINES.items():
            assert lam > 0, f"{name} has non-positive wavelength"

    def test_halpha_present(self):
        assert any("alpha" in name.lower() or "H" in name
                    for name in EMISSION_LINES)


class TestDetectEmissionLines:

    def test_detects_injected_line(self):
        ha_rest = EMISSION_LINES[r"H$\alpha$"]
        z = 1.0
        wl, flux = _make_spectrum_with_line(ha_rest, z)
        result = detect_emission_lines(wl, flux, z, sigma_thresh=3.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert r"H$\alpha$" in result["line"].values

    def test_returns_empty_for_flat_spectrum(self):
        wl = np.linspace(1.0, 5.0, 2500)
        flux = np.zeros_like(wl)
        result = detect_emission_lines(wl, flux, z=1.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_output_columns(self):
        ha_rest = EMISSION_LINES[r"H$\alpha$"]
        wl, flux = _make_spectrum_with_line(ha_rest, z=1.5)
        result = detect_emission_lines(wl, flux, z=1.5)
        expected_cols = {"line", "rest_wavelength", "obs_wavelength",
                         "peak_flux", "snr"}
        assert set(result.columns) == expected_cols

    def test_high_sigma_thresh_reduces_detections(self):
        ha_rest = EMISSION_LINES[r"H$\alpha$"]
        wl, flux = _make_spectrum_with_line(ha_rest, z=1.0)
        low = detect_emission_lines(wl, flux, z=1.0, sigma_thresh=2.0)
        high = detect_emission_lines(wl, flux, z=1.0, sigma_thresh=50.0)
        assert len(high) <= len(low)

    def test_save_csv(self, tmp_path):
        ha_rest = EMISSION_LINES[r"H$\alpha$"]
        wl, flux = _make_spectrum_with_line(ha_rest, z=1.0)
        csv_path = str(tmp_path / "lines.csv")
        detect_emission_lines(wl, flux, z=1.0, save_path=csv_path)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) > 0

    def test_obs_wavelength_is_redshifted(self):
        ha_rest = EMISSION_LINES[r"H$\alpha$"]
        z = 2.0
        wl, flux = _make_spectrum_with_line(ha_rest, z)
        result = detect_emission_lines(wl, flux, z=z)
        if len(result) > 0:
            ha_row = result[result["line"] == r"H$\alpha$"]
            if len(ha_row) > 0:
                expected_obs = ha_rest * (1.0 + z)
                assert ha_row.iloc[0]["obs_wavelength"] == pytest.approx(
                    expected_obs, rel=1e-3)


class TestPlotSpectrum:

    def test_returns_dataframe_with_redshift(self):
        import matplotlib
        matplotlib.use("Agg")
        ha_rest = EMISSION_LINES[r"H$\alpha$"]
        wl, flux = _make_spectrum_with_line(ha_rest, z=1.0)
        result = plot_spectrum(wl, flux, z=1.0)
        assert isinstance(result, pd.DataFrame)

    def test_returns_none_without_redshift(self):
        import matplotlib
        matplotlib.use("Agg")
        wl = np.linspace(1.0, 5.0, 100)
        flux = np.random.randn(100)
        result = plot_spectrum(wl, flux)
        assert result is None

    def test_saves_to_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        wl = np.linspace(1.0, 5.0, 100)
        flux = np.random.randn(100)
        out = str(tmp_path / "spectrum.png")
        plot_spectrum(wl, flux, output_path=out)
        assert os.path.exists(out)


import os
