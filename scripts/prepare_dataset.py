#!/usr/bin/env python3
"""
prepare_dataset.py -- Build train/eval .npz datasets from JADES DR4 FITS files.

Corrected pipeline: split at the object level BEFORE augmentation to prevent
data leakage between train and evaluation sets.

Steps:
  1. Extract spectra from prism + three medium-resolution gratings
  2. Match objects across the four configurations by (field, RA, Dec)
  3. Combine three high-res gratings onto a common binned grid
  4. Cross-match with the DR4 catalog and apply quality cuts (S/N > 5
     on at least one of Hα, [OII]3727, [OIII]5007)
  5. Split at the object level (80/20 train/eval)
  6. Augment ONLY the training set (redshift-shift + noise)
  7. Clean NaNs, trim to 1–5 μm, interpolate to a fixed 2500-point grid
  8. Save separate train and eval .npz files

Usage:
    python scripts/prepare_dataset.py \
        --jades_dir /home/aryana/Documents/GitHub/JADES_data/DR4 \
        --out_dir data
"""

import os
import argparse
import glob
import warnings

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from tqdm import tqdm

warnings.filterwarnings("ignore", category=VerifyWarning)


# ═══════════════════════════════════════════════════════════════════════
#  1. FITS extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_spectra(fits_dir, grating_subdir):
    """Extract wavelength/flux/flux_err + RA/DEC/FIELD from _x1d FITS files."""
    pattern = os.path.join(fits_dir, "**", grating_subdir, "**", "*_x1d.fits")
    fits_files = sorted(glob.glob(pattern, recursive=True))
    data_list = []

    for fpath in tqdm(fits_files, desc=f"  {grating_subdir}"):
        with fits.open(fpath) as hdul:
            extract_hdu = None
            for hdu in hdul:
                extname = hdu.header.get("EXTNAME", "").strip().upper()
                if extname in ("EXTRACT5PIX1D", "EXTRACT3PIX1D",
                               "EXTRACTOPT1D", "EXTRACT1D"):
                    extract_hdu = hdu
                    break
            if extract_hdu is None or not hasattr(extract_hdu, "columns"):
                continue

            col_names = [c.upper() for c in extract_hdu.columns.names]
            if not all(c in col_names for c in ("WAVELENGTH", "FLUX", "FLUX_ERR")):
                continue

            tbl = extract_hdu.data
            wavelength = np.array(tbl["WAVELENGTH"], dtype=np.float64)
            flux = np.array(tbl["FLUX"], dtype=np.float64)
            flux_err = np.array(tbl["FLUX_ERR"], dtype=np.float64)

            ph = hdul[0].header
            avg_ra = np.float64(ph.get("RA", np.nan))
            avg_dec = np.float64(ph.get("DEC", np.nan))
            field = ph.get("HLSPTARG", "UNKNOWN")

            data_list.append({
                "file_name": os.path.basename(fpath),
                "FIELD": field,
                "WAVELENGTH": wavelength,
                "FLUX": flux,
                "FLUX_ERR": flux_err,
                "RA": avg_ra,
                "DEC": avg_dec,
            })

    df = pd.DataFrame(data_list)
    for col in ("WAVELENGTH", "FLUX", "FLUX_ERR", "RA", "DEC"):
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float64))
    return df


# ═══════════════════════════════════════════════════════════════════════
#  2. Cross-match the four grating catalogues
# ═══════════════════════════════════════════════════════════════════════

def match_spectra(df_list, match_radius=0.01):
    ref_df = df_list[0]
    matched_indices = [[] for _ in df_list]

    for j in range(len(ref_df)):
        field_val = ref_df.at[j, "FIELD"]
        ref_coord = SkyCoord(
            ra=ref_df.at[j, "RA"] * u.deg,
            dec=ref_df.at[j, "DEC"] * u.deg,
        )
        current_match = [j]
        valid = True

        for i in range(1, len(df_list)):
            candidates = df_list[i][df_list[i]["FIELD"] == field_val]
            if candidates.empty:
                valid = False
                break
            cat_coords = SkyCoord(
                ra=candidates["RA"].values * u.deg,
                dec=candidates["DEC"].values * u.deg,
            )
            seps = ref_coord.separation(cat_coords)
            if seps.min().arcsec < match_radius:
                current_match.append(candidates.index[seps.argmin()])
            else:
                valid = False
                break

        if valid and len(current_match) == len(df_list):
            for i in range(len(df_list)):
                matched_indices[i].append(current_match[i])

    if not matched_indices[0]:
        return pd.DataFrame()

    matched_dfs = []
    for i, df in enumerate(df_list):
        sub = df.loc[matched_indices[i]].reset_index(drop=True)
        sub = sub.add_suffix(f"_{i + 1}")
        matched_dfs.append(sub)

    return pd.concat(matched_dfs, axis=1, join="inner")


# ═══════════════════════════════════════════════════════════════════════
#  3. Combine three high-res gratings
# ═══════════════════════════════════════════════════════════════════════

def _fill_nans_interp(x, y):
    valid = ~np.isnan(y)
    if np.sum(valid) < 2:
        if np.sum(valid) == 1:
            return np.full_like(y, y[valid][0])
        return y
    out = y.copy()
    out[~valid] = np.interp(x[~valid], x[valid], y[valid])
    return out


def combine_high_res(matched_df, grid_resolution=0.001, sigma_smooth=0):
    from scipy.ndimage import gaussian_filter1d
    from scipy import interpolate as interp_mod

    rows = []
    for index, row in matched_df.iterrows():
        w_lo = np.array(row["WAVELENGTH_1"], dtype=float)
        f_lo = np.array(row["FLUX_1"], dtype=float)
        f_lo_err = np.array(row["FLUX_ERR_1"], dtype=float)

        hr_wl, hr_fl, hr_err = [], [], []
        for i in range(2, 5):
            wk = f"WAVELENGTH_{i}"
            fk = f"FLUX_{i}"
            ek = f"FLUX_ERR_{i}"
            if wk not in matched_df.columns:
                continue
            wl = np.array(row[wk], dtype=float)
            fl = np.array(row[fk], dtype=float)
            er = np.array(row[ek], dtype=float)
            ok = ~np.isnan(wl) & ~np.isnan(fl) & ~np.isnan(er)
            if ok.any():
                hr_wl.append(wl[ok])
                hr_fl.append(fl[ok])
                hr_err.append(er[ok])

        if not hr_wl:
            continue

        cw = np.concatenate(hr_wl)
        cf = np.concatenate(hr_fl)
        ce = np.concatenate(hr_err)
        order = np.argsort(cw)
        cw, cf, ce = cw[order], cf[order], ce[order]

        bins = np.arange(cw.min(), cw.max() + grid_resolution, grid_resolution)
        centres = 0.5 * (bins[1:] + bins[:-1])

        bf, _, _ = binned_statistic(cw, cf, statistic="mean", bins=bins)
        be, _, _ = binned_statistic(cw, ce, statistic="mean", bins=bins)
        bf = _fill_nans_interp(centres, bf)
        be = _fill_nans_interp(centres, be)

        lo_min, hi_max = max(w_lo.min(), centres[0]), min(w_lo.max(), centres[-1])
        mask = (centres >= lo_min) & (centres <= hi_max)
        centres = centres[mask]
        bf, be = bf[mask], be[mask]

        f_lo_i = _fill_nans_interp(
            centres,
            interp_mod.interp1d(w_lo, f_lo, kind="linear",
                                bounds_error=False, fill_value="extrapolate")(centres),
        )
        e_lo_i = _fill_nans_interp(
            centres,
            interp_mod.interp1d(w_lo, f_lo_err, kind="linear",
                                bounds_error=False, fill_value="extrapolate")(centres),
        )
        wl_lo_i = _fill_nans_interp(
            centres,
            interp_mod.interp1d(w_lo, w_lo, kind="linear",
                                bounds_error=False, fill_value="extrapolate")(centres),
        )

        if sigma_smooth > 0:
            bf_s = _fill_nans_interp(centres, gaussian_filter1d(bf, sigma=sigma_smooth))
            be_s = _fill_nans_interp(centres, gaussian_filter1d(be, sigma=sigma_smooth))
        else:
            bf_s, be_s = bf.copy(), be.copy()

        rows.append({
            "id": index,
            "ra": row["RA_1"],
            "dec": row["DEC_1"],
            "field": row["FIELD_1"],
            "wavelength_high": centres,
            "flux_high": bf,
            "flux_high_err": be,
            "flux_high_smoothed": bf_s,
            "flux_high_smoothed_err": be_s,
            "wavelength_low": wl_lo_i,
            "flux_low": f_lo_i,
            "flux_low_err": e_lo_i,
            "filename1": row.get("file_name_1"),
            "filename2": row.get("file_name_2"),
            "filename3": row.get("file_name_3"),
            "filename4": row.get("file_name_4"),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  4. Quality cuts  (S/N > 5 on Hα, [OII], or [OIII])
# ═══════════════════════════════════════════════════════════════════════

def _to_native(arr):
    arr = np.asarray(arr)
    if arr.dtype.byteorder in ("=", "|"):
        return arr
    return arr.byteswap().view(arr.dtype.newbyteorder("="))


def _fits_table_to_df(hdu, columns):
    data = hdu.data
    out = {}
    for col in columns:
        if col not in data.names:
            continue
        arr = _to_native(data[col])
        if arr.ndim == 1:
            out[col] = arr
        elif arr.ndim == 2 and arr.shape[1] == 1:
            out[col] = _to_native(arr[:, 0])
        else:
            out[col] = [_to_native(x) for x in arr]
    return pd.DataFrame(out)


def load_catalog_and_cut(catalog_path, sn_thresh=5):
    obs_cols = [
        "Unique_ID", "PID", "TIER", "NIRSpec_ID",
        "NIRCam_DR5_ID", "NIRCam_DR3_ID",
        "RA_TARG", "Dec_TARG", "Field", "z_Spec", "z_PRISM",
    ]
    r1000_cols = [
        "Unique_ID", "PID", "TIER", "NIRSpec_ID",
        "NIRCam_DR5_ID", "NIRCam_DR3_ID",
        "HBaA_6563_flux", "HBaA_6563_flux_err",
        "O2_3727_flux", "O2_3727_flux_err",
        "O3_5007_flux", "O3_5007_flux_err",
    ]
    merge_keys = [
        "Unique_ID", "PID", "TIER", "NIRSpec_ID",
        "NIRCam_DR5_ID", "NIRCam_DR3_ID",
    ]

    with fits.open(catalog_path) as hdul:
        obs_df = _fits_table_to_df(hdul["Obs_info"], obs_cols)
        r1000_df = _fits_table_to_df(hdul["R1000_5pix"], r1000_cols)

    for df in (obs_df, r1000_df):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )

    merged = obs_df.merge(r1000_df, on=merge_keys, how="inner")

    merged["z"] = merged.apply(
        lambda r: r["z_Spec"]
        if pd.notna(r["z_Spec"]) and r["z_Spec"] != -1
        else r["z_PRISM"],
        axis=1,
    )
    merged = merged[merged["z"].notna()].reset_index(drop=True)
    n_before = len(merged)

    merged["HA_SN"] = merged["HBaA_6563_flux"] / merged["HBaA_6563_flux_err"]
    merged["OII_SN"] = merged["O2_3727_flux"] / merged["O2_3727_flux_err"]
    merged["OIII_SN"] = merged["O3_5007_flux"] / merged["O3_5007_flux_err"]

    merged = merged[
        (merged["HA_SN"] > sn_thresh)
        | (merged["OII_SN"] > sn_thresh)
        | (merged["OIII_SN"] > sn_thresh)
    ].reset_index(drop=True)

    keep = [c for c in [
        "RA_TARG", "Dec_TARG", "Field", "z",
        "HBaA_6563_flux", "HBaA_6563_flux_err",
        "O2_3727_flux", "O2_3727_flux_err",
        "O3_5007_flux", "O3_5007_flux_err",
    ] if c in merged.columns]
    merged = merged[keep].copy()

    print(f"  Quality cut: {len(merged)}/{n_before} objects kept "
          f"(S/N > {sn_thresh} on Hα, [OII], or [OIII])")
    return merged


def apply_redshift_match(combined_df, df_redshifts, tol=0.01):
    def short(f):
        return "GS" if "South" in f else "GN" if "North" in f else f

    df = combined_df.copy()
    df["short_field"] = df["field"].apply(short)
    df["orig_index"] = df.index

    uniq = df[["short_field", "ra", "dec"]].drop_duplicates().reset_index(drop=True)
    uniq["z"] = np.nan

    for fld in uniq["short_field"].unique():
        red = df_redshifts[df_redshifts["Field"] == fld]
        if red.empty:
            continue
        sub = uniq[uniq["short_field"] == fld]
        cu = SkyCoord(ra=sub["ra"].values * u.deg, dec=sub["dec"].values * u.deg)
        cr = SkyCoord(ra=red["RA_TARG"].values * u.deg,
                      dec=red["Dec_TARG"].values * u.deg)
        idx, d2d, _ = cu.match_to_catalog_sky(cr)
        sel = d2d < tol * u.arcsec
        uniq.loc[sub.index[sel], "z"] = red.iloc[idx[sel]]["z"].values

    out = df.merge(uniq[["short_field", "ra", "dec", "z"]],
                   on=["short_field", "ra", "dec"], how="left")
    out = out[out["z"].notna()].copy()
    out.sort_values("orig_index", inplace=True)
    out.drop(columns=["short_field", "orig_index"], inplace=True)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  5. Train / eval split at the OBJECT level
# ═══════════════════════════════════════════════════════════════════════

def split_objects(df, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    eval_df = df.iloc[eval_idx].reset_index(drop=True)
    print(f"  Split: {len(train_df)} train, {len(eval_df)} eval "
          f"(from {n} unique objects)")
    return train_df, eval_df


# ═══════════════════════════════════════════════════════════════════════
#  6. Augmentation  (redshift-shift + noise) — train only
# ═══════════════════════════════════════════════════════════════════════

def augment_spectra_redshift(df, sigma=0.3, copies_per_row=20,
                             seed=None, add_noise=True, noise_frac=0.1):
    if seed is not None:
        np.random.seed(seed)

    aug_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="  Augmenting training set"):
        w_lo = np.array(row["wavelength_low"])
        f_lo = np.array(row["flux_low"])
        f_lo_err = np.array(row["flux_low_err"])

        w_hi = np.array(row["wavelength_high"])
        f_hi = np.array(row["flux_high"])
        f_hi_sm = np.array(row["flux_high_smoothed"])
        f_hi_err = np.array(row["flux_high_err"])
        f_hi_sm_err = np.array(row["flux_high_smoothed_err"])

        z_orig = row["z"]

        for i in range(copies_per_row):
            dz = np.random.normal(0, sigma)
            z_new = z_orig + dz

            w_rest_lo = w_lo / (1 + z_orig)
            w_rest_hi = w_hi / (1 + z_orig)
            w_lo_z = w_rest_lo * (1 + z_new)
            w_hi_z = w_rest_hi * (1 + z_new)

            f_lo_z = f_lo.copy()
            f_hi_z = f_hi.copy()
            f_hi_sm_z = f_hi_sm.copy()
            f_lo_err_z = f_lo_err.copy()
            f_hi_err_z = f_hi_err.copy()
            f_hi_sm_err_z = f_hi_sm_err.copy()

            if add_noise:
                sig_lo = noise_frac * np.abs(f_lo_z) + 1e-23
                sig_hi = noise_frac * np.abs(f_hi_z) + 1e-23
                sig_sm = noise_frac * np.abs(f_hi_sm_z) + 1e-23

                f_lo_z += np.random.normal(0, sig_lo)
                f_hi_z += np.random.normal(0, sig_hi)
                f_hi_sm_z += np.random.normal(0, sig_sm)

                f_lo_err_z = np.sqrt(f_lo_err_z**2 + sig_lo**2)
                f_hi_err_z = np.sqrt(f_hi_err_z**2 + sig_hi**2)
                f_hi_sm_err_z = np.sqrt(f_hi_sm_err_z**2 + sig_sm**2)

            aug_rows.append({
                "id": f"{row['id']}_zshift{i}_dz{dz:+.4f}",
                "ra": row["ra"],
                "dec": row["dec"],
                "field": row["field"],
                "z": z_new,
                "wavelength_low": w_lo_z,
                "flux_low": f_lo_z,
                "flux_low_err": f_lo_err_z,
                "wavelength_high": w_hi_z,
                "flux_high": f_hi_z,
                "flux_high_err": f_hi_err_z,
                "flux_high_smoothed": f_hi_sm_z,
                "flux_high_smoothed_err": f_hi_sm_err_z,
            })

    aug_df = pd.DataFrame(aug_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    print(f"  Augmented: {len(df)} → {len(combined)} "
          f"({copies_per_row} copies/row + originals)")
    return combined


# ═══════════════════════════════════════════════════════════════════════
#  7. Clean → Trim → Interpolate
# ═══════════════════════════════════════════════════════════════════════

def clean_nans(df):
    def _fill(arr):
        arr = np.array(arr, dtype=float)
        nans = np.isnan(arr)
        if not nans.any():
            return arr
        valid = arr[~nans]
        med = np.median(valid) if len(valid) > 0 else 0.0
        arr[nans] = med
        return arr

    flux_cols = [
        "wavelength_low", "flux_low", "flux_low_err",
        "wavelength_high", "flux_high", "flux_high_err",
        "flux_high_smoothed", "flux_high_smoothed_err",
    ]
    for col in flux_cols:
        if col in df.columns:
            df[col] = df[col].apply(_fill)
    return df


def trim_to_range(df, wl_min=1.0, wl_max=5.0):
    rows = []
    lo_cols = ["wavelength_low", "flux_low", "flux_low_err"]
    hi_cols = ["wavelength_high", "flux_high", "flux_high_err",
               "flux_high_smoothed", "flux_high_smoothed_err"]

    for _, row in df.iterrows():
        r = dict(row)

        w_lo = np.array(r["wavelength_low"])
        mask_lo = (w_lo >= wl_min) & (w_lo <= wl_max)
        for c in lo_cols:
            r[c] = np.array(r[c])[mask_lo]

        w_hi = np.array(r["wavelength_high"])
        mask_hi = (w_hi >= wl_min) & (w_hi <= wl_max)
        for c in hi_cols:
            r[c] = np.array(r[c])[mask_hi]

        rows.append(r)
    return pd.DataFrame(rows)


def interpolate_to_grid(df, wl_min=1.0, wl_max=5.0, n_points=2500):
    grid = np.linspace(wl_min, wl_max, n_points)
    rows = []

    def _interp_fill(x, y):
        if len(x) < 2:
            return np.full_like(grid, np.nanmedian(y) if len(y) > 0 else 0.0)
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan)
        yi = f(grid)
        nans = np.isnan(yi)
        if nans.any():
            med = np.nanmedian(yi)
            yi[nans] = med if np.isfinite(med) else 0.0
        return yi

    for _, row in df.iterrows():
        w_lo = np.array(row["wavelength_low"])
        w_hi = np.array(row["wavelength_high"])

        rows.append({
            "id": row["id"],
            "ra": row["ra"],
            "dec": row["dec"],
            "field": row["field"],
            "z": row["z"],
            "wavelength_low": grid,
            "flux_low": _interp_fill(w_lo, np.array(row["flux_low"])),
            "flux_low_err": _interp_fill(w_lo, np.array(row["flux_low_err"])),
            "wavelength_high": grid,
            "flux_high": _interp_fill(w_hi, np.array(row["flux_high"])),
            "flux_high_err": _interp_fill(w_hi, np.array(row["flux_high_err"])),
            "flux_high_smoothed": _interp_fill(
                w_hi, np.array(row["flux_high_smoothed"])),
            "flux_high_smoothed_err": _interp_fill(
                w_hi, np.array(row["flux_high_smoothed_err"])),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  8. Save to .npz
# ═══════════════════════════════════════════════════════════════════════

def save_npz(df, path):
    np.savez_compressed(
        path,
        flux_low=np.stack(df["flux_low"].values),
        flux_low_err=np.stack(df["flux_low_err"].values),
        flux_high=np.stack(df["flux_high"].values),
        flux_high_err=np.stack(df["flux_high_err"].values),
        flux_high_smoothed=np.stack(df["flux_high_smoothed"].values),
        flux_high_smoothed_err=np.stack(df["flux_high_smoothed_err"].values),
        id=np.array(df["id"].values),
        ra=np.array(df["ra"].values, dtype=np.float64),
        dec=np.array(df["dec"].values, dtype=np.float64),
        field=np.array(df["field"].values),
        z=np.array(df["z"].values, dtype=np.float64),
        wavelength_low=df["wavelength_low"].iloc[0],
        wavelength_high=df["wavelength_high"].iloc[0],
    )
    print(f"  Saved {path}  ({len(df)} spectra)")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jades_dir", type=str,
                    default="/home/aryana/Documents/GitHub/JADES_data/DR4")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug_copies", type=int, default=20)
    ap.add_argument("--aug_sigma", type=float, default=0.3)
    ap.add_argument("--noise_frac", type=float, default=0.1)
    ap.add_argument("--sn_thresh", type=float, default=5.0)
    ap.add_argument("--match_radius", type=float, default=0.01)
    ap.add_argument("--n_points", type=int, default=2500)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    catalog_path = os.path.join(args.jades_dir, "Combined_DR4_external_v1.2.1.fits")

    # --- Step 1: Extract spectra ---
    print("Step 1: Extracting spectra from FITS files...")
    df_prism = extract_spectra(args.jades_dir, "clear-prism")
    df_g140m = extract_spectra(args.jades_dir, "f070lp-g140m")
    df_g235m = extract_spectra(args.jades_dir, "f170lp-g235m")
    df_g395m = extract_spectra(args.jades_dir, "f290lp-g395m")
    print(f"  Extracted: prism={len(df_prism)}, g140m={len(df_g140m)}, "
          f"g235m={len(df_g235m)}, g395m={len(df_g395m)}")

    # --- Step 2: Match across gratings ---
    print("\nStep 2: Matching objects across gratings...")
    matched_df = match_spectra(
        [df_prism, df_g140m, df_g235m, df_g395m],
        match_radius=args.match_radius,
    )
    print(f"  Matched: {len(matched_df)} objects")

    # --- Step 3: Combine high-res ---
    print("\nStep 3: Combining high-resolution gratings...")
    combined_df = combine_high_res(matched_df, grid_resolution=0.001,
                                   sigma_smooth=0)
    print(f"  Combined: {len(combined_df)} objects")

    # --- Step 4: Quality cuts ---
    print("\nStep 4: Applying quality cuts from DR4 catalog...")
    df_redshifts = load_catalog_and_cut(catalog_path, sn_thresh=args.sn_thresh)
    combined_df = apply_redshift_match(combined_df, df_redshifts,
                                       tol=args.match_radius)
    # Drop filename columns no longer needed
    for c in ("filename1", "filename2", "filename3", "filename4"):
        if c in combined_df.columns:
            combined_df.drop(columns=[c], inplace=True)
    print(f"  After quality cuts + redshift match: {len(combined_df)} objects")

    # --- Step 5: Split BEFORE augmentation ---
    print("\nStep 5: Splitting at object level...")
    train_df, eval_df = split_objects(combined_df, train_frac=args.train_frac,
                                      seed=args.seed)

    # --- Step 6: Augment ONLY training set ---
    print("\nStep 6: Augmenting training set...")
    train_aug_df = augment_spectra_redshift(
        train_df,
        sigma=args.aug_sigma,
        copies_per_row=args.aug_copies,
        seed=args.seed,
        add_noise=True,
        noise_frac=args.noise_frac,
    )

    # --- Step 7: Clean + Trim + Interpolate (both sets) ---
    print("\nStep 7: Cleaning, trimming, interpolating...")
    for label, df in [("train", train_aug_df), ("eval", eval_df)]:
        print(f"  Processing {label} set ({len(df)} spectra)...")
        df = clean_nans(df)
        df = trim_to_range(df, wl_min=1.0, wl_max=5.0)
        df = interpolate_to_grid(df, wl_min=1.0, wl_max=5.0,
                                 n_points=args.n_points)
        if label == "train":
            train_final = df
        else:
            eval_final = df

    # --- Step 8: Save ---
    print("\nStep 8: Saving datasets...")
    save_npz(train_final, os.path.join(args.out_dir, "train_DR4.npz"))
    save_npz(eval_final, os.path.join(args.out_dir, "eval_DR4.npz"))

    print(f"\nDone! Train: {len(train_final)}, Eval: {len(eval_final)}")
    print("No data leakage: augmentation applied only to training objects.")


if __name__ == "__main__":
    main()
