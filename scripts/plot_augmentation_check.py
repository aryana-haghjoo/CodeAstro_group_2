#!/usr/bin/env python3
"""
Plot an original spectrum alongside its augmented copies from the
train_DR4.npz file, similar to the final plot in Augmentation.ipynb.

Usage:
    python scripts/plot_augmentation_check.py [--obj_id 42]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_DR4.npz")
    ap.add_argument("--obj_id", type=int, default=None,
                    help="Original object ID to plot (random if omitted)")
    ap.add_argument("--out", default="plots/augmentation_after_split.png")
    args = ap.parse_args()

    data = np.load(args.train, allow_pickle=True)
    ids = data["id"]
    flux_low = data["flux_low"]
    flux_high = data["flux_high"]
    wl = data["wavelength_high"]
    wl_lo = data["wavelength_low"]
    z = data["z"]

    # Separate originals from augmented (augmented IDs contain "_zshift")
    is_original = np.array([
        "_zshift" not in str(i) for i in ids
    ])
    orig_indices = np.where(is_original)[0]

    if args.obj_id is not None:
        orig_idx = None
        for idx in orig_indices:
            if int(ids[idx]) == args.obj_id:
                orig_idx = idx
                break
        if orig_idx is None:
            raise ValueError(f"Object ID {args.obj_id} not found in originals")
    else:
        orig_idx = np.random.choice(orig_indices)

    base_id = str(ids[orig_idx])
    z_orig = z[orig_idx]

    # Find all augmented copies
    prefix = base_id + "_zshift"
    aug_mask = np.array([str(i).startswith(prefix) for i in ids])
    aug_indices = np.where(aug_mask)[0]

    # Parse dz values for coloring
    def parse_dz(id_str):
        try:
            return float(str(id_str).split("_dz")[-1])
        except Exception:
            return np.nan

    dz_vals = np.array([parse_dz(ids[i]) for i in aug_indices])
    order = np.argsort(dz_vals)
    aug_indices = aug_indices[order]
    dz_vals = dz_vals[order]

    # ---- Plot style (matching notebook) ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    cmap = plt.get_cmap("plasma")
    n_aug = len(aug_indices)
    colors = [cmap(i / max(n_aug, 1)) for i in range(n_aug)]

    orig_label = f"original (z = {z_orig:.4f})"

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    fig.suptitle(
        f"ID = {base_id}  —  Augmented training spectrum",
        fontsize=11, fontweight="regular", y=0.9,
    )

    # ---- Low-res ----
    axs[0].plot(wl_lo, flux_low[orig_idx], color="k", lw=1.2,
                label=orig_label)
    for i, (ai, dz) in enumerate(zip(aug_indices, dz_vals)):
        axs[0].plot(wl_lo, flux_low[ai], color=colors[i], linestyle="--",
                    alpha=0.6, label=f"dz = {dz:+.4f}")
    axs[0].set_title("Low-Resolution")
    axs[0].set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    axs[0].legend(ncol=2, fontsize="small", loc="upper right")

    # ---- High-res ----
    axs[1].plot(wl, flux_high[orig_idx], color="k", lw=1.2,
                label=orig_label)
    for i, (ai, dz) in enumerate(zip(aug_indices, dz_vals)):
        axs[1].plot(wl, flux_high[ai], color=colors[i], linestyle="--",
                    alpha=0.6, label=f"dz = {dz:+.4f}")
    axs[1].set_title("High-Resolution")
    axs[1].set_xlabel("Wavelength (μm)")
    axs[1].set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    axs[1].legend(ncol=2, fontsize="small", loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=450, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
