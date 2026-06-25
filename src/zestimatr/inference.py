import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interp1d
from tqdm import tqdm

from .model import ZHead1D

HF_REPO_ID = "aryana-haghjoo/zestimatr"
HF_FILENAME = "best_zhead_hires.pth"

DEFAULT_WL_MIN = 1.0
DEFAULT_WL_MAX = 5.0
DEFAULT_WL_NPOINTS = 2500


def download_pretrained(repo_id=HF_REPO_ID, filename=HF_FILENAME):
    """
    Download the pretrained checkpoint from Hugging Face Hub.

    Returns the local path to the downloaded file.
    Requires ``huggingface_hub`` to be installed (``pip install huggingface_hub``).
    """
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename)


def resample_flux(wavelength, flux, target_wavelength):
    """
    Resample a spectrum onto a target wavelength grid via linear interpolation.

    Parameters
    ----------
    wavelength : array-like, shape (M,)
        Input wavelength array (must be monotonically increasing).
    flux : array-like, shape (M,) or (N, M)
        Flux values corresponding to ``wavelength``.
        Can be a single spectrum (1-D) or a batch (2-D).
    target_wavelength : array-like, shape (K,)
        Target wavelength grid to resample onto.

    Returns
    -------
    resampled : ndarray, shape (K,) or (N, K)
        Interpolated flux on the target grid.  Points outside the input
        range are filled with the nearest boundary value.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    target_wavelength = np.asarray(target_wavelength, dtype=np.float64)

    single = flux.ndim == 1
    if single:
        flux = flux[np.newaxis, :]

    resampled = np.empty((flux.shape[0], len(target_wavelength)),
                         dtype=np.float32)
    for i in range(flux.shape[0]):
        valid = np.isfinite(flux[i])
        if valid.sum() < 2:
            resampled[i] = 0.0
            continue
        f = interp1d(wavelength[valid], flux[i][valid], kind="linear",
                     bounds_error=False, fill_value=(flux[i][valid][0],
                                                     flux[i][valid][-1]))
        resampled[i] = f(target_wavelength)

    return resampled[0] if single else resampled


def _normalize_flux(flux):
    """Per-spectrum zero-mean unit-variance normalization."""
    flux = np.asarray(flux, dtype=np.float32)
    single = flux.ndim == 1
    if single:
        flux = flux[np.newaxis, :]

    normed = np.empty_like(flux)
    for i in range(len(flux)):
        mean = np.nanmean(flux[i])
        std = np.nanstd(flux[i])
        if std < 1e-25:
            std = 1e-25
        normed[i] = (flux[i] - mean) / std

    return normed[0] if single else normed


class InferenceDataset(Dataset):
    """
    Dataset for inference on high-resolution spectra.

    Expects .npz file with 'flux_high' (or 'flux_hi') and optionally
    a redshift column ('z', 'redshift', 'z_true', 'redshifts', 'z_spec').

    If the spectra live on a different wavelength grid than the model's
    training grid, pass ``target_wavelength`` to resample automatically.
    The input wavelength grid is read from the 'wavelength_high' (or
    'wavelength') key in the .npz file.
    """
    def __init__(self, npz_path, redshift_key=None, target_wavelength=None):
        data = np.load(npz_path)

        if 'flux_high' in data:
            flux_raw = data['flux_high'].astype(np.float32)
        elif 'flux_hi' in data:
            flux_raw = data['flux_hi'].astype(np.float32)
        else:
            raise KeyError(f"Could not find high-resolution flux. Available keys: {list(data.keys())}")

        if target_wavelength is not None:
            wl_key = None
            for k in ('wavelength_high', 'wavelength', 'wavelength_hi'):
                if k in data:
                    wl_key = k
                    break
            if wl_key is None:
                raise KeyError(
                    "Cannot resample: no wavelength array found in .npz. "
                    f"Available keys: {list(data.keys())}")
            input_wl = data[wl_key].astype(np.float64)
            flux_raw = resample_flux(input_wl, flux_raw, target_wavelength)

        self.flux = _normalize_flux(flux_raw)

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

    ckpt = torch.load(zhead_ckpt_path, map_location="cpu", weights_only=False)

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

    wl_grid = ckpt.get('wavelength_grid', None)
    if wl_grid is None:
        wl_min = ckpt.get('wl_min', DEFAULT_WL_MIN)
        wl_max = ckpt.get('wl_max', DEFAULT_WL_MAX)
        wl_npoints = ckpt.get('wl_npoints', DEFAULT_WL_NPOINTS)
        wl_grid = np.linspace(wl_min, wl_max, wl_npoints)

    norm_params = {
        'z_mean': z_mean,
        'z_std': z_std,
        'z_min': z_min,
        'z_max': z_max,
        'wavelength_grid': wl_grid,
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


@torch.no_grad()
def predict(flux, zhead, norm_params, wavelength=None, device=None,
            batch_size=64):
    """
    Predict redshifts from flux arrays, with optional wavelength resampling.

    This is a convenience wrapper around ``predict_redshifts`` that accepts
    raw numpy arrays instead of a DataLoader.  If ``wavelength`` is provided
    and differs from the training grid, the spectra are automatically
    resampled.

    Parameters
    ----------
    flux : array-like, shape (L,) or (N, L)
        Flux values.  A single spectrum (1-D) or a batch (2-D).
    zhead : ZHead1D
        Trained model in eval mode.
    norm_params : dict
        From ``load_model`` (must include ``wavelength_grid``).
    wavelength : array-like, shape (L,), optional
        Wavelength array corresponding to ``flux``.  If provided, the
        spectra are resampled onto the model's training wavelength grid.
        If ``None``, the flux is assumed to already be on the training grid.
    device : torch.device, optional
    batch_size : int
        Batch size for inference.

    Returns
    -------
    dict
        Keys: ``z_pred``, ``z_uncertainty`` (numpy arrays).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flux = np.asarray(flux, dtype=np.float64)
    single = flux.ndim == 1
    if single:
        flux = flux[np.newaxis, :]

    target_wl = norm_params['wavelength_grid']

    if wavelength is not None:
        wavelength = np.asarray(wavelength, dtype=np.float64)
        if wavelength.shape[0] != flux.shape[1]:
            raise ValueError(
                f"wavelength length ({wavelength.shape[0]}) does not match "
                f"flux length ({flux.shape[1]})")
        if not np.array_equal(wavelength, target_wl):
            flux = resample_flux(wavelength, flux, target_wl)

    flux = _normalize_flux(flux)

    flux_tensor = torch.from_numpy(flux).float()
    z_dummy = torch.full((len(flux_tensor),), -1.0)
    from torch.utils.data import TensorDataset
    ds = TensorDataset(flux_tensor, z_dummy)
    loader = DataLoader(ds, batch_size=batch_size)

    result = predict_redshifts(zhead, loader, norm_params, device=device)
    if single:
        result = {
            'z_pred': result['z_pred'][0],
            'z_uncertainty': result['z_uncertainty'][0],
        }
    return result
