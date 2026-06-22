import numpy as np
import scipy
import matplotlib.pyplot as plt
import astropy.io.fits as fits

def open_spectrum(file):
    """
    Opens a spectrum file and returns the wavelength and intensity arrays.
    i (int): int of galaxy you want to look at from JADES file
    
    Returns:
        wavelength (numpy.ndarray): Array of wavelengths.
        flux (numpy.ndarray): Array of intensities.
    """
    spec = np.load(file)
    wavelength = spec['wavelength_high']
    flux = spec['flux_high']
    return wavelength, flux

def plot_spectrum(wavelength, flux):
    """
    Plots the spectrum given wavelength and intensity arrays.
    
    Args:
        wavelength (numpy.ndarray): Array of wavelengths.
        intensity (numpy.ndarray): Array of intensities.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux, color='blue')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Flux')
    plt.title('Spectrum')
    plt.grid()
    plt.show()