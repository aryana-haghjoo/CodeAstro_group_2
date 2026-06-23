import numpy as np
import pandas as pd
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


def detect_emission_lines(wavelength, flux, z, sigma_thresh=3.0,
                          line_halfwidth=8, background_halfwidth=80,
                          save_path=None):
    """
    Detect emission lines in a spectrum given a known redshift.

    Shifts rest-frame emission line wavelengths to the observed frame,
    checks which fall within the observed wavelength range, and looks
    for flux peaks near the expected positions.  Noise is estimated
    from a background annulus *around* the line so that the line
    itself does not inflate the noise estimate.  The signal is taken
    as the peak of a 3-pixel running mean inside the line window,
    rejecting isolated single-pixel noise spikes while still
    capturing narrow emission features.

    Parameters
    ----------
    wavelength : array-like
        Observed wavelength array in microns.
    flux : array-like
        Observed flux array (same length as wavelength).
    z : float
        Redshift of the source (e.g. from ``predict_redshifts``).
    sigma_thresh : float, optional
        Minimum SNR (peak 3-pixel mean above continuum, divided by
        background noise) for a line to count as detected.
    line_halfwidth : int, optional
        Half-width in pixels of the window centred on the expected
        line position in which the peak is searched.
    background_halfwidth : int, optional
        Half-width of the wider region used to estimate the continuum
        level and noise.  Pixels inside the line window are excluded.
    save_path : str, optional
        If given, save the results to this path as a ``.csv`` file.

    Returns
    -------
    pandas.DataFrame
        One row per detected line with columns: ``line``,
        ``rest_wavelength``, ``obs_wavelength``, ``peak_flux``,
        ``snr``.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    rows = []

    for name, lam_rest in EMISSION_LINES.items():
        lam_obs = lam_rest * (1.0 + z)

        if lam_obs < wavelength.min() or lam_obs > wavelength.max():
            continue

        idx_center = np.argmin(np.abs(wavelength - lam_obs))

        # Narrow line window
        l_lo = max(0, idx_center - line_halfwidth)
        l_hi = min(len(flux), idx_center + line_halfwidth + 1)
        line_region = flux[l_lo:l_hi]
        if np.sum(np.isfinite(line_region)) < 3:
            continue

        # Wider background annulus (excluding the line window)
        bg_lo = max(0, idx_center - background_halfwidth)
        bg_hi = min(len(flux), idx_center + background_halfwidth + 1)
        bg_mask = np.ones(bg_hi - bg_lo, dtype=bool)
        bg_mask[l_lo - bg_lo : l_hi - bg_lo] = False
        bg_region = flux[bg_lo:bg_hi][bg_mask]
        bg_finite = bg_region[np.isfinite(bg_region)]
        if len(bg_finite) < 10:
            continue

        continuum = np.median(bg_finite)
        noise = np.median(np.abs(bg_finite - continuum)) * 1.4826
        if noise == 0:
            continue

        # 3-pixel running mean to smooth single-pixel spikes
        kernel = np.ones(3) / 3.0
        smoothed = np.convolve(np.nan_to_num(line_region, nan=continuum),
                               kernel, mode="valid")
        peak_smooth = np.max(smoothed)
        snr = (peak_smooth - continuum) / noise

        if snr >= sigma_thresh:
            rows.append({
                "line": name,
                "rest_wavelength": lam_rest,
                "obs_wavelength": float(lam_obs),
                "peak_flux": float(np.nanmax(line_region)),
                "snr": round(float(snr), 2),
            })

    df = pd.DataFrame(rows, columns=[
        "line", "rest_wavelength", "obs_wavelength", "peak_flux", "snr",
    ])

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


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
    detected_lines : pandas.DataFrame, optional
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
    pandas.DataFrame or None
        The detected lines DataFrame, or None if no redshift was
        provided.
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

        for i, row in detected_lines.iterrows():
            color = colors[i % len(colors)]
            ax.axvline(row["obs_wavelength"], color=color,
                       linestyle="--", alpha=0.7, lw=1.2)

            if row["obs_wavelength"] - prev_obs_wl < 0.08:
                slot += 1
            else:
                slot = 0
            y_frac = label_positions[slot % len(label_positions)]
            prev_obs_wl = row["obs_wavelength"]

            ax.text(row["obs_wavelength"], ymin + y_range * y_frac,
                    f' {row["line"]}',
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
