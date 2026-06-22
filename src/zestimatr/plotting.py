import numpy as np
import matplotlib.pyplot as plt

from .metrics import compute_metrics


def plot_predictions(predictions, output_path=None):
    """
    Create a two-panel validation plot: predicted vs true redshift, and
    a histogram of relative errors.

    Parameters
    ----------
    predictions : dict
        Must contain 'z_pred', 'z_uncertainty', and 'z_true'.
    output_path : str, optional
        If given, save the figure to this path. Otherwise call plt.show().

    Returns
    -------
    dict or None
        Metrics dict if z_true is present, else None.
    """
    if 'z_true' not in predictions:
        print("No ground truth available, skipping plots")
        return None

    z_pred = predictions['z_pred']
    z_true = predictions['z_true']

    metrics = compute_metrics(z_pred, z_true)

    dz_over_1pz = (z_pred - z_true) / (1.0 + np.abs(z_true))
    abs_dz_over_1pz = np.abs(dz_over_1pz)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: predicted vs true, coloured by |dz|/(1+z)
    vmax = np.percentile(abs_dz_over_1pz, 99)
    sc = ax1.scatter(z_true, z_pred, c=abs_dz_over_1pz, s=10, alpha=0.6,
                     vmin=0, vmax=vmax, cmap='viridis')

    z_lo, z_hi = z_true.min(), z_true.max()
    ax1.plot([z_lo, z_hi], [z_lo, z_hi], 'r--', lw=2, label='Perfect prediction')

    ax1.set_xlabel('True Redshift')
    ax1.set_ylabel('Predicted Redshift')
    ax1.set_title('Predicted vs True Redshift')
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('|dz|/(1+z)')

    metrics_text = (
        f"MAE: {metrics['mae']:.4f}\n"
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"NMAD: {metrics['nmad']:.4f}\n"
        f"Median |dz|/(1+z): {metrics['median_rel_error']:.4f}\n"
        f"Outlier rate: {metrics['outlier_rate']:.2%}"
    )
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

    # Panel 2: histogram of relative errors
    ax2.hist(dz_over_1pz, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax2.axvline(0.15, color='orange', linestyle=':', lw=2, label='Outlier threshold')
    ax2.axvline(-0.15, color='orange', linestyle=':', lw=2)

    ax2.set_xlabel('dz/(1+z)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Relative Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    stat_text = (
        f"Mean: {dz_over_1pz.mean():.4f}\n"
        f"Median: {np.median(dz_over_1pz):.4f}\n"
        f"Std: {dz_over_1pz.std():.4f}"
    )
    ax2.text(0.95, 0.95, stat_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
    return metrics
