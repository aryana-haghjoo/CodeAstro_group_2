import numpy as np
import zestimatr
import argparse

parser = argparse.ArgumentParser(description='Predict redshift from a spectrum file')
parser.add_argument('data_path', type=str, help='Path to the .npz spectrum file')
args = parser.parse_args()

# Load a spectrum
data = np.load(args.data_path)
flux = data["flux_high"]
z_true = float(data["z"])

# Load wavelength array if available (for resampling to training grid)
wavelength = None
for key in ("wavelength_high", "wavelength", "wavelength_hi"):
    if key in data:
        wavelength = data[key]
        break

# Download and load pretrained model from Hugging Face
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
checkpoint_path = zestimatr.download_pretrained()
zhead, norm_params = zestimatr.load_model(checkpoint_path, device=device)

# Predict — wavelength resampling is handled automatically
predictions = zestimatr.predict(flux, zhead, norm_params,
                                wavelength=wavelength, device=device)

print(f"Predicted: z = {predictions['z_pred']:.4f} +/- {predictions['z_uncertainty']:.4f}")
print(f"True:      z = {z_true:.4f}")