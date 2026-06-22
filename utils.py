import numpy as np
import scipy
import matplotlib.pyplot as plt
import astropy.io.fits as fits

def open_spectrum(file):
    """
    Opens a spectrum file and returns the wavelength and intensity arrays.
    
    Returns:
        wavelength (numpy.ndarray): Array of wavelengths.
        intensity (numpy.ndarray): Array of intensities.
    """
    spec = fits.open(file)
    wavelength = spec[0].data['WAVELENGTH']
    intensity = spec[0].data['FLUX']
    
    return wavelength, intensity

def plot_spectrum(wavelength, intensity):
    """
    Plots the spectrum given wavelength and intensity arrays.
    
    Args:
        wavelength (numpy.ndarray): Array of wavelengths.
        intensity (numpy.ndarray): Array of intensities.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, intensity, color='blue')
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Intensity')
    plt.title('Spectrum')
    plt.grid()
    plt.show()