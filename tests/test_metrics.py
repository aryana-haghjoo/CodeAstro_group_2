import numpy as np
import pytest

from zestimatr.metrics import compute_metrics, compute_calibration_metrics


class TestComputeMetrics:

    def test_perfect_predictions(self):
        z = np.array([0.1, 0.5, 1.0, 2.0])
        m = compute_metrics(z, z)
        assert m['mae'] == pytest.approx(0.0)
        assert m['rmse'] == pytest.approx(0.0)
        assert m['outlier_rate'] == pytest.approx(0.0)

    def test_known_values(self):
        z_true = np.array([1.0, 2.0, 3.0])
        z_pred = np.array([1.1, 2.2, 3.3])
        m = compute_metrics(z_pred, z_true)

        expected_abs_err = np.array([0.1, 0.2, 0.3])
        assert m['mae'] == pytest.approx(expected_abs_err.mean())
        assert m['rmse'] == pytest.approx(
            np.sqrt((expected_abs_err ** 2).mean()))

        rel_err = expected_abs_err / (1.0 + z_true)
        assert m['median_rel_error'] == pytest.approx(np.median(rel_err))

    def test_outlier_detection(self):
        z_true = np.array([0.0, 0.0])
        z_pred = np.array([0.0, 0.5])  # second has |dz|/(1+z)=0.5 > 0.15
        m = compute_metrics(z_pred, z_true)
        assert m['outlier_rate'] == pytest.approx(0.5)

    def test_single_element(self):
        m = compute_metrics(np.array([1.0]), np.array([1.0]))
        assert m['mae'] == pytest.approx(0.0)


class TestComputeCalibrationMetrics:

    def test_perfect_calibration(self):
        rng = np.random.default_rng(42)
        sigma = 0.01
        z_true = rng.uniform(0.5, 2.5, size=10000)
        z_pred = z_true + rng.normal(0, sigma, size=10000)
        z_unc = np.full_like(z_true, sigma)

        cal = compute_calibration_metrics(z_pred, z_true, z_unc)
        assert cal['calibration_std'] == pytest.approx(1.0, abs=0.05)
        assert cal['calibration_mean'] == pytest.approx(0.0, abs=0.05)
        assert cal['frac_1sigma'] == pytest.approx(0.6827, abs=0.03)

    def test_overconfident_uncertainties(self):
        z_true = np.array([1.0, 2.0, 3.0])
        z_pred = np.array([1.5, 2.3, 3.8])
        z_unc = np.array([0.01, 0.01, 0.01])  # way too small

        cal = compute_calibration_metrics(z_pred, z_true, z_unc)
        assert cal['calibration_std'] > 1.0

    def test_keys_present(self):
        z = np.array([1.0, 2.0])
        cal = compute_calibration_metrics(z, z, np.array([0.1, 0.1]))
        expected_keys = {
            'calibration_std', 'calibration_mean', 'frac_1sigma',
            'frac_2sigma', 'frac_3sigma', 'outlier_rate',
            'median_uncertainty',
        }
        assert set(cal.keys()) == expected_keys
