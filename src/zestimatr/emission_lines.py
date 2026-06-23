import numpy as np
import matplotlib.pyplot as plt


# Rest-frame vacuum wavelengths in microns
EMISSION_LINES = {
    r"Ly$\alpha$": 0.12157,
    "C IV": 0.15490,
    "C III]": 0.19090,
    "Mg II": 0.27965,
    "[O II]": 0.37270,
    "[Ne III]": 0.38690,
    r"H$\delta$": 0.41017,
    r"H$\gamma$": 0.43405,
    r"H$\beta$": 0.48613,
    "[O III] 4959": 0.49590,
    "[O III] 5007": 0.50070,
    "[N II] 6548": 0.65480,
    r"H$\alpha$": 0.65628,
    "[N II] 6583": 0.65830,
    "[S II] 6716": 0.67160,
    "[S II] 6731": 0.67310,
}


def detect_emission_lines(wavelength, flux, z, sigma_thresh=2.5, window_size=30):
    """
    Detect emission lines in a spectrum given a known redshift.

    Shifts rest-frame emission line wavelengths to the observed frame,
    checks which fall within the observed wavelength range, and looks
    for flux peaks near the expected positions.

    Parameters
    ----------
    wavelength : array-like
        Observed wavelength array in microns.
    flux : array-like
        Observed flux array (same length as wavelength).
    z : float
        Redshift of the source (e.g. from ``predict_redshifts``).
    sigma_thresh : float, optional
        A line is considered detected if the peak flux in its window
        exceeds ``median + sigma_thresh * mad`` of the local region.
    window_size : int, optional
        Half-width in pixels of the search window around each expected
        line position.

    Returns
    -------
    list of dict
        Each detected line is a dict with keys:

        - ``"name"`` — line label (str)
        - ``"rest_wavelength"`` — rest-frame wavelength in microns (float)
        - ``"obs_wavelength"`` — observed wavelength in microns (float)
        - ``"peak_flux"`` — peak flux value at the detected position (float)
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    detected = []

    for name, lam_rest in EMISSION_LINES.items():
        lam_obs = lam_rest * (1.0 + z)

        if lam_obs < wavelength.min() or lam_obs > wavelength.max():
            continue

        idx_center = np.argmin(np.abs(wavelength - lam_obs))

        lo = max(0, idx_center - window_size)
        hi = min(len(flux), idx_center + window_size + 1)

        region = flux[lo:hi]
        finite = region[np.isfinite(region)]
        if len(finite) < 5:
            continue

        median_val = np.median(finite)
        mad = np.median(np.abs(finite - median_val))
        if mad == 0:
            continue

        peak_val = np.nanmax(region)
        if peak_val > median_val + sigma_thresh * mad * 1.4826:
            detected.append({
                "name": name,
                "rest_wavelength": lam_rest,
                "obs_wavelength": lam_obs,
                "peak_flux": float(peak_val),
            })

    return detected


def plot_spectrum(wavelength, flux, z=None, detected_lines=None,
                  flux_err=None, output_path=None, ax=None):
    """
    Plot a spectrum with optional emission-line markers.

    Parameters
    ----------
    wavelength : array-like
        Observed wavelength in microns.
    flux : array-like
        Observed flux (same length as wavelength).
    z : float, optional
        Redshift. If provided and ``detected_lines`` is None,
        ``detect_emission_lines`` is called automatically.
    detected_lines : list of dict, optional
        Output of ``detect_emission_lines``. Overrides auto-detection
        when given.
    flux_err : array-like, optional
        Flux uncertainties for a shaded error band.
    output_path : str, optional
        If given, save the figure to this path instead of showing it.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when *None*.

    Returns
    -------
    list of dict or None
        The detected lines list, or None if no redshift was provided.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(wavelength, flux, color="k", lw=0.8)

    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        ax.fill_between(wavelength, flux - flux_err, flux + flux_err,
                        color="gray", alpha=0.25)

    if z is not None:
        if detected_lines is None:
            detected_lines = detect_emission_lines(wavelength, flux, z)

        colors = plt.cm.tab10.colors
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin

        label_positions = [0.97, 0.82, 0.67]
        prev_obs_wl = -np.inf
        slot = 0

        for i, line in enumerate(detected_lines):
            color = colors[i % len(colors)]
            ax.axvline(line["obs_wavelength"], color=color,
                       linestyle="--", alpha=0.7, lw=1.2)

            if line["obs_wavelength"] - prev_obs_wl < 0.08:
                slot += 1
            else:
                slot = 0
            y_frac = label_positions[slot % len(label_positions)]
            prev_obs_wl = line["obs_wavelength"]

            ax.text(line["obs_wavelength"], ymin + y_range * y_frac,
                    f' {line["name"]}',
                    rotation=90, va="top", ha="right",
                    fontsize=8, color=color)

        ax.set_title(f"Spectrum  (z = {z:.4f},  {len(detected_lines)} lines detected)")
    else:
        ax.set_title("Spectrum")

    ax.set_xlabel("Wavelength (microns)")
    ax.set_ylabel("Flux")
    ax.grid(True, alpha=0.3)

    if own_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    return detected_lines
