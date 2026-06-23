# zestimatr

Spectroscopic redshift estimation from high-resolution galaxy spectra, with uncertainty quantification.

`zestimatr` is a Python package that uses a 1D convolutional neural network to predict redshifts directly from high-resolution spectral flux. The model outputs both a point estimate and a calibrated uncertainty (predicted standard deviation) for each spectrum.

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

To install with training dependencies (wandb, pyyaml):

```bash
pip install -e ".[train]"
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

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zhead, norm_params = zestimatr.load_model("best_zhead_hires.pth", device=device)

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

## Training

To train the model from scratch:

```bash
pip install -e ".[train]"
python scripts/train.py --data data/spectra_dataset_2500_unaugmented.npz --wandb_mode online
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
│   └── plotting.py          # Validation plots
├── scripts/
│   └── train.py             # Training CLI (not part of the package)
├── tests/
│   └── test_metrics.py      # Unit tests
├── tutorials/
│   └── quickstart.ipynb     # Tutorial notebook
├── pyproject.toml           # Package metadata
└── README.md
```

## License

MIT

## Authors

Code/Astro 2026 Group 2
