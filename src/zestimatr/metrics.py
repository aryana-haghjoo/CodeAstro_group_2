import numpy as np


def compute_metrics(z_pred, z_true):
    """
    Compute redshift prediction accuracy metrics.

    Parameters
    ----------
    z_pred : array_like
        Predicted redshifts.
    z_true : array_like
        True redshifts.

    Returns
    -------
    dict
        Keys: mae, rmse, median_abs_error, nmad, median_rel_error,
        p90_rel_error, outlier_rate.
    """
    z_pred = np.asarray(z_pred, dtype=np.float64)
    z_true = np.asarray(z_true, dtype=np.float64)

    abs_err = np.abs(z_pred - z_true)
    rel_err = abs_err / (1.0 + np.abs(z_true))

    return {
        'mae': float(abs_err.mean()),
        'rmse': float(np.sqrt(((z_pred - z_true) ** 2).mean())),
        'median_abs_error': float(np.median(abs_err)),
        'nmad': float(1.4826 * np.median(abs_err)),
        'median_rel_error': float(np.median(rel_err)),
        'p90_rel_error': float(np.percentile(rel_err, 90)),
        'outlier_rate': float((rel_err > 0.15).mean()),
    }


def compute_calibration_metrics(z_pred, z_true, z_uncertainties):
    """
    Compute uncertainty calibration metrics.

    Parameters
    ----------
    z_pred : array_like
        Predicted redshifts.
    z_true : array_like
        True redshifts.
    z_uncertainties : array_like
        Predicted 1-sigma uncertainties.

    Returns
    -------
    dict
        Keys: calibration_std, calibration_mean, frac_1sigma,
        frac_2sigma, frac_3sigma, outlier_rate, median_uncertainty.
    """
    z_pred = np.asarray(z_pred, dtype=np.float64)
    z_true = np.asarray(z_true, dtype=np.float64)
    z_uncertainties = np.asarray(z_uncertainties, dtype=np.float64)

    norm_residuals = (z_pred - z_true) / (z_uncertainties + 1e-12)
    abs_norm_res = np.abs(norm_residuals)

    abs_err = np.abs(z_pred - z_true)
    relative_errors = abs_err / (1.0 + np.abs(z_true))

    return {
        'calibration_std': float(np.std(norm_residuals)),
        'calibration_mean': float(np.mean(norm_residuals)),
        'frac_1sigma': float((abs_norm_res < 1.0).mean()),
        'frac_2sigma': float((abs_norm_res < 2.0).mean()),
        'frac_3sigma': float((abs_norm_res < 3.0).mean()),
        'outlier_rate': float((relative_errors > 0.15).mean()),
        'median_uncertainty': float(np.median(z_uncertainties)),
    }
