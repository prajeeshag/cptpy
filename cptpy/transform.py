"""
transform.py
------------
CPT standardises both X and Y at each grid point before EOF decomposition.
Means and stds are returned separately so LOO folds can recompute them
on training-only data (excluding the target year).
"""
import numpy as np
import xarray as xr


def standardize_field(
    da: xr.DataArray,
    eps: float = 1e-8,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Zero-mean, unit-variance standardisation along T.

    Returns
    -------
    da_std : standardised DataArray
    mu     : per-point mean
    sigma  : per-point std  (clipped to eps)
    """
    mu    = da.mean("T")
    sigma = da.std("T").clip(min=eps)
    return (da - mu) / sigma, mu, sigma


def standardize_all(
    Y2d: xr.DataArray,
    X2d: dict[str, xr.DataArray],
    eps: float = 1e-8,
) -> tuple[
    xr.DataArray, dict[str, xr.DataArray],   # standardised fields
    xr.DataArray, xr.DataArray,               # Y mu, Y sigma
    dict[str, xr.DataArray], dict[str, xr.DataArray],  # X mu, X sigma
]:
    """
    CPT-style: standardise Y and all X fields per grid point.
    Returns standardised arrays + their statistics (needed for un-standardisation
    and LOO recomputation).
    """
    Y_std, Y_mu, Y_sigma = standardize_field(Y2d, eps)

    X_std  = {}
    X_mu   = {}
    X_sigma = {}
    for m, da in X2d.items():
        X_std[m], X_mu[m], X_sigma[m] = standardize_field(da, eps)

    return Y_std, X_std, Y_mu, Y_sigma, X_mu, X_sigma
