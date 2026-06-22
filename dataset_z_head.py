import numpy as np
import torch
from torch.utils.data import Dataset


class SpectraForZHead(Dataset):
    """
    Dataset for high-res redshift prediction (oracle upper bound).

    Uses the *original* dataset NPZ:
      - flux_high (HR spectra)
      - z          (ground truth redshifts)

    Returns:
      x_high_norm: (L_high,) float32  — per-spectrum normalized
      z:           float32
    """
    def __init__(self, dataset_npz_path, normalize_flux=True):
        data = np.load(dataset_npz_path, allow_pickle=True)
        self.flux_high = data["flux_high"]
        self.z = data["z"].astype(np.float32)

        self.normalize_flux = normalize_flux

        if normalize_flux:
            hi_normed = []
            for i in range(len(self.flux_high)):
                hi_n, _, _ = self._normalize(self.flux_high[i])
                hi_normed.append(hi_n)
            self.flux_high = np.asarray(hi_normed, dtype=np.float32)
        else:
            self.flux_high = self.flux_high.astype(np.float32)

    @staticmethod
    def _normalize(x, eps=1e-25):
        m = np.nanmean(x)
        s = np.nanstd(x)
        if s < eps:
            s = eps
        return (x - m) / s, m, s

    def __len__(self):
        return len(self.flux_high)

    def __getitem__(self, idx):
        x_high = torch.tensor(self.flux_high[idx], dtype=torch.float32)  # (L_high,)
        z = torch.tensor(self.z[idx], dtype=torch.float32)               # scalar
        return x_high, z
