# zestimatr

[![PyPI](https://img.shields.io/pypi/v/zestimatr)](https://pypi.org/project/zestimatr/)
[![Python](https://img.shields.io/pypi/pyversions/zestimatr)](https://pypi.org/project/zestimatr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-orange)](https://huggingface.co/aryana-haghjoo/zestimatr)
[![arXiv](https://img.shields.io/badge/arXiv-2603.18357-b31b1b)](https://arxiv.org/abs/2603.18357)

Spectroscopic redshift estimation from high-resolution galaxy spectra, with uncertainty quantification.

`zestimatr` is a Python package that uses a residual 1D convolutional neural network with an MLP head to predict redshifts directly from high-resolution spectral flux. The model outputs both a point estimate and a calibrated uncertainty (predicted standard deviation) for each spectrum.

## Installation

```bash
pip install zestimatr
```

Or install from source:

```bash
git clone https://github.com/aryana-haghjoo/CodeAstro_group_2.git
cd CodeAstro_group_2
pip install -e .
```

## Quick Start

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import zestimatr

# Load a spectrum
data = np.load("galaxy300_spectrum.npz")
flux = data["flux_high"]
z_true = float(data["z"])

# Normalize (zero mean, unit variance)
flux_norm = (flux - np.nanmean(flux)) / max(np.nanstd(flux), 1e-25)
flux_tensor = torch.tensor(flux_norm, dtype=torch.float32)

# Download and load pretrained model from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = zestimatr.download_pretrained()
zhead, norm_params = zestimatr.load_model(checkpoint_path, device=device)

# Predict
dataset = TensorDataset(flux_tensor.unsqueeze(0), torch.tensor([z_true]))
dataloader = DataLoader(dataset, batch_size=1)
predictions = zestimatr.predict_redshifts(zhead, dataloader, norm_params, device)

print(f"Predicted: z = {predictions['z_pred'][0]:.4f} +/- {predictions['z_uncertainty'][0]:.4f}")
print(f"True:      z = {z_true:.4f}")

# Evaluate
metrics = zestimatr.compute_metrics(predictions["z_pred"], predictions["z_true"])
print(f"MAE: {metrics['mae']:.4f}, Outlier rate: {metrics['outlier_rate']:.1%}")
```

## Emission Line Detection

After estimating a redshift, you can detect and visualize emission lines in the spectrum:

```python
wavelength = data["wavelength_high"]
flux = data["flux_high"]

# Detect lines — returns a pandas DataFrame
lines = zestimatr.detect_emission_lines(wavelength, flux, predictions["z_pred"][0])
print(lines)

# Optionally save to CSV
lines = zestimatr.detect_emission_lines(wavelength, flux, predictions["z_pred"][0],
                                        save_path="detected_lines.csv")

# Plot spectrum with emission lines marked as dashed vertical lines
zestimatr.plot_spectrum(wavelength, flux, z=predictions["z_pred"][0])
```

The built-in catalog includes 16 common rest-frame lines (Ly-alpha, H-alpha, H-beta, [O II], [O III], [N II], [S II], and more). Detection uses a local peak-finding approach with a configurable `sigma_thresh` (default 3.0).

## Metrics

`zestimatr` provides two evaluation functions:

- **`compute_metrics(z_pred, z_true)`** -- accuracy metrics:
  - MAE, RMSE, NMAD
  - Median |dz|/(1+z)
  - Outlier rate (fraction with |dz|/(1+z) > 0.15)

- **`compute_calibration_metrics(z_pred, z_true, z_uncertainty)`** -- uncertainty calibration:
  - Calibration std and mean of normalized residuals
  - 1/2/3-sigma coverage fractions
  - Median predicted uncertainty

## For Developers

Most users only need the base install above. The following is for retraining or rebuilding the dataset from raw JADES data.

### Extra dependencies

```bash
# Training only
pip install -e ".[train]"

# Data preprocessing only (astropy, scipy)
pip install -e ".[preprocess]"

# Everything
pip install -e ".[train,preprocess]"
```

### Data preprocessing

Prepare train/eval datasets from JADES DR4 FITS files. The pipeline splits at the object level before augmentation to prevent data leakage:

```bash
python scripts/prepare_dataset.py --jades_dir /path/to/JADES_data/DR4
```

This produces `data/train_DR4.npz` (augmented training set) and `data/eval_DR4.npz` (held-out evaluation set).

### Training

```bash
python scripts/train.py --train_data data/train_DR4.npz --eval_data data/eval_DR4.npz --wandb_mode online
```

Key training options:

| Flag | Default | Description |
|------|---------|-------------|
| `--hidden_dim` | 128 | Conv block hidden dimension |
| `--num_blocks` | 6 | Number of residual conv blocks |
| `--dropout` | 0.2 | Dropout rate |
| `--epochs` | 200 | Training epochs |
| `--lr` | 3e-4 | Learning rate |
| `--batch_size` | 32 | Batch size |
| `--wandb_mode` | online | `online`, `offline`, or `disabled` |

Training logs and plots are synced to [Weights & Biases](https://wandb.ai/).

## Pretrained Model

A pretrained checkpoint trained on JADES DR4 (52,647 spectra) is available on [Hugging Face](https://huggingface.co/aryana-haghjoo/zestimatr).

Download it automatically:

```python
import zestimatr

path = zestimatr.download_pretrained()
zhead, norm_params = zestimatr.load_model(path)
```

| Metric | Value |
|--------|-------|
| MAE | 0.141 |
| RMSE | 0.323 |
| Median \|dz\|/(1+z) | 0.012 |
| Outlier rate | 5.7% |
| Calibration std | 0.84 |

## Data Format

Input `.npz` files should contain:

- `flux_high` -- high-resolution spectral flux, shape `(N, L)` for datasets or `(L,)` for a single spectrum
- `z` -- ground truth redshifts, shape `(N,)` or scalar (also accepts keys: `redshift`, `z_true`, `z_spec`)

Optional keys: `wavelength_high`, `flux_high_err`, `id`, `ra`, `dec`.

## Project Structure

```
zestimatr/
├── src/zestimatr/          # Package source
│   ├── model.py             # ZHead1D network + loss function
│   ├── dataset.py           # PyTorch dataset for training
│   ├── metrics.py           # Accuracy and calibration metrics
│   ├── inference.py         # Model loading + prediction pipeline
│   ├── plotting.py          # Validation plots
│   └── emission_lines.py    # Emission line detection + spectrum plotting
├── scripts/
│   ├── prepare_dataset.py   # Data extraction, quality cuts, split, augmentation
│   └── train.py             # Training CLI (not part of the package)
├── tests/
│   └── test_metrics.py      # Unit tests
├── tutorials/
│   ├── quickstart_single_spectrum.ipynb  # Single spectrum tutorial
│   ├── quickstart_batch_spectra.ipynb    # Batch (200 spectra) tutorial
│   ├── galaxy300_spectrum.npz            # Example single spectrum
│   └── sample_200_spectra.npz            # Example 200-spectrum sample
├── pyproject.toml           # Package metadata
└── README.md
```

## License

MIT

## Authors

Code/Astro 2026 Group 2: Aryana Haghjoo - Lau, Marie Wingyee - Michele Woodland
