import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import ZHead1D


class InferenceDataset(Dataset):
    """
    Dataset for inference on high-resolution spectra.

    Expects .npz file with 'flux_high' (or 'flux_hi') and optionally
    a redshift column ('z', 'redshift', 'z_true', 'redshifts', 'z_spec').
    """
    def __init__(self, npz_path, redshift_key=None):
        data = np.load(npz_path)

        if 'flux_high' in data:
            flux_raw = data['flux_high'].astype(np.float32)
        elif 'flux_hi' in data:
            flux_raw = data['flux_hi'].astype(np.float32)
        else:
            raise KeyError(f"Could not find high-resolution flux. Available keys: {list(data.keys())}")

        self.flux = []
        for i in range(len(flux_raw)):
            f = flux_raw[i]
            mean = np.nanmean(f)
            std = np.nanstd(f)
            if std < 1e-25:
                std = 1e-25
            self.flux.append((f - mean) / std)
        self.flux = np.array(self.flux, dtype=np.float32)

        if redshift_key is not None:
            if redshift_key in data:
                self.redshift = data[redshift_key].astype(np.float32)
                self.has_ground_truth = True
            else:
                raise KeyError(f"Specified redshift key '{redshift_key}' not found in dataset")
        else:
            possible_keys = ['redshift', 'z', 'z_true', 'redshifts', 'z_spec']
            self.has_ground_truth = False
            for key in possible_keys:
                if key in data:
                    self.redshift = data[key].astype(np.float32)
                    self.has_ground_truth = True
                    break
            if not self.has_ground_truth:
                self.redshift = None

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        flux = torch.from_numpy(self.flux[idx]).float()
        if self.has_ground_truth:
            return flux, float(self.redshift[idx])
        return flux, -1.0


def load_model(zhead_ckpt_path, device=None, fallback_data_path=None):
    """
    Load a trained ZHead1D model from a checkpoint.

    Parameters
    ----------
    zhead_ckpt_path : str
        Path to the .pth checkpoint file.
    device : torch.device, optional
        Device to load the model onto. Defaults to CUDA if available.
    fallback_data_path : str, optional
        Path to the dataset .npz, used to compute z_min/z_max if not
        stored in the checkpoint.

    Returns
    -------
    zhead : ZHead1D
        The model in eval mode with frozen parameters.
    norm_params : dict
        Keys: z_mean, z_std, z_min, z_max.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(zhead_ckpt_path, map_location="cpu")

    saved_config = ckpt['config']
    z_mean = ckpt['z_mean']
    z_std = ckpt['z_std']
    z_min = ckpt.get('z_min', None)
    z_max = ckpt.get('z_max', None)

    if z_min is None or z_max is None:
        candidate_paths = []
        cfg_data_path = saved_config.get('data')
        if cfg_data_path:
            candidate_paths.append(cfg_data_path)
            candidate_paths.append(os.path.join(os.path.dirname(zhead_ckpt_path), cfg_data_path))
        if fallback_data_path:
            candidate_paths.append(fallback_data_path)

        tried = set()
        for data_path in candidate_paths:
            if not data_path:
                continue
            resolved = os.path.abspath(data_path)
            if resolved in tried:
                continue
            tried.add(resolved)
            if not os.path.exists(resolved):
                continue
            try:
                data = np.load(resolved, allow_pickle=True)
                for key in ['z', 'redshift', 'z_true', 'redshifts']:
                    if key in data:
                        z_all = data[key]
                        z_min = float(z_all.min())
                        z_max = float(z_all.max())
                        break
                if z_min is not None and z_max is not None:
                    break
            except Exception:
                continue

        if z_min is None or z_max is None:
            raise ValueError(
                "Could not determine z_min and z_max. "
                "Not found in checkpoint and training data not accessible. "
                "Please provide fallback_data_path or retrain with a newer checkpoint."
            )

    zhead = ZHead1D(
        in_channels=1,
        hidden_dim=saved_config.get('hidden_dim', 64),
        num_blocks=saved_config.get('num_blocks', 4),
        dropout=saved_config.get('dropout', 0.1),
    ).to(device)

    zhead.load_state_dict(ckpt['zhead_state_dict'])
    zhead.eval()
    for p in zhead.parameters():
        p.requires_grad = False

    norm_params = {
        'z_mean': z_mean,
        'z_std': z_std,
        'z_min': z_min,
        'z_max': z_max,
    }

    return zhead, norm_params


@torch.no_grad()
def predict_redshifts(zhead, dataloader, norm_params, device=None):
    """
    Run redshift inference on a dataloader.

    Parameters
    ----------
    zhead : ZHead1D
        Trained model in eval mode.
    dataloader : DataLoader
        Yields (flux, z_true) batches. z_true = -1.0 when unknown.
    norm_params : dict
        From ``load_model``.
    device : torch.device, optional

    Returns
    -------
    dict
        Keys: z_pred, z_uncertainty, and optionally z_true (numpy arrays).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_mean = norm_params['z_mean']
    z_std = norm_params['z_std']
    z_min_n = (norm_params['z_min'] - z_mean) / z_std
    z_max_n = (norm_params['z_max'] - z_mean) / z_std

    all_z_pred, all_z_std, all_z_true = [], [], []
    has_gt = False

    for flux, z_true in tqdm(dataloader, desc="Predicting"):
        flux = flux.to(device).unsqueeze(1)
        mu_raw, logvar_n = zhead(flux)

        mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
        z_pred = mu_n * z_std + z_mean

        z_var = torch.exp(logvar_n.clamp(min=-12, max=12)) * (z_std ** 2)
        z_uncertainty = torch.sqrt(z_var)

        all_z_pred.append(z_pred.cpu())
        all_z_std.append(z_uncertainty.cpu())

        if z_true[0] != -1.0:
            has_gt = True
            all_z_true.append(z_true)

    predictions = {
        'z_pred': torch.cat(all_z_pred).numpy(),
        'z_uncertainty': torch.cat(all_z_std).numpy(),
    }
    if has_gt:
        predictions['z_true'] = torch.cat(all_z_true).numpy()

    return predictions
