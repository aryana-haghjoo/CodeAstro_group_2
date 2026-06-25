.. _inference:

Inference
=========

Functions for loading a trained model and predicting redshifts.

The recommended entry point is :func:`~zestimatr.inference.predict`, which
accepts raw numpy flux arrays and an optional wavelength array.  When a
wavelength array is provided, spectra are automatically resampled onto the
model's training grid before inference — no manual preprocessing needed.

For lower-level control, :func:`~zestimatr.inference.predict_redshifts`
accepts a PyTorch ``DataLoader`` directly, and
:func:`~zestimatr.inference.resample_flux` can be used standalone to
interpolate spectra between arbitrary wavelength grids.

.. automodule:: zestimatr.inference
   :members:
