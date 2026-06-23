import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import zestimatr

# Load a spectrum
data = np.load("tutorials/galaxy300_spectrum.npz")
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