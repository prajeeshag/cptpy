import logging
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


def align_time(
    Y: xr.DataArray,
    X_models: dict[str, xr.DataArray],
    config: dict,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Trim all fields to configured year range and common years."""
    ymin, ymax = config["hindcast_start_year"], config["hindcast_end_year"]

    def _years(da):
        yrs = da["T"].dt.year.values
        return set(yrs[(yrs >= ymin) & (yrs <= ymax)])

    common = _years(Y)
    for da in X_models.values():
        common &= _years(da)
    common = np.array(sorted(common))

    def _sel(da):
        return da.sel(T=da["T"].dt.year.isin(common))

    return _sel(Y), {m: _sel(X_models[m]) for m in X_models}


def regrid(
    Y: xr.DataArray,
    X_models: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """Bilinearly interpolate each model onto the obs grid."""
    return {m: X_models[m].interp(Y=Y["Y"], X=Y["X"]) for m in X_models}


def filter_sparse_models(
    Y: xr.DataArray,
    X_models: dict[str, xr.DataArray],
    min_frac: float = 0.10,
) -> dict[str, xr.DataArray]:
    """
    Drop models whose regridded field covers fewer than min_frac of the
    obs valid points.  Catches models that are all-NaN for a given season
    (e.g. METEOFRANCE9 in dry months) which would otherwise zero-out the
    spatial intersection in flatten().
    """
    Y2d      = Y.stack(space=("Y", "X"))
    obs_valid = int((~np.isnan(Y2d).any("T")).sum())
    if obs_valid == 0:
        return X_models

    kept = {}
    for m, da in X_models.items():
        da2d      = da.stack(space=("Y", "X"))
        mod_valid = int((~np.isnan(da2d).any("T")).sum())
        frac      = mod_valid / obs_valid
        if frac >= min_frac:
            kept[m] = da
        else:
            log.warning(
                "Dropping model %s: only %.1f%% valid spatial coverage "
                "(threshold %.0f%%) — likely all-NaN for this season.",
                m, 100 * frac, 100 * min_frac,
            )
    return kept


def apply_mask(
    Y: xr.DataArray,
    X_models: dict[str, xr.DataArray],
    config: dict,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Mask always-NaN, zero-variance, or too-dry grid points."""
    valid = Y.notnull().all("T") & (Y.std("T") > 0)
    if config.get("drymask_threshold") is not None:
        valid &= Y.mean("T") > config["drymask_threshold"]
    Y = Y.where(valid)
    X_models = {m: X_models[m].where(valid) for m in X_models}
    return Y, X_models


def flatten(
    Y: xr.DataArray,
    X_models: dict[str, xr.DataArray],
    valid_space=None,
) -> tuple[xr.DataArray, dict[str, xr.DataArray], any]:
    """
    Stack (Y,X) → space and drop any location with NaNs in Y or any model.

    Pass `valid_space` from the hindcast call to guarantee the forecast
    uses exactly the same spatial mask.
    """
    Y2d = Y.stack(space=("Y", "X"))
    X2d = {m: X_models[m].stack(space=("Y", "X")) for m in X_models}

    if valid_space is None:
        valid_space = ~np.isnan(Y2d).any("T")
        for da in X2d.values():
            valid_space = valid_space & ~np.isnan(da).any("T")

    Y2d = Y2d.sel(space=valid_space)
    X2d = {m: da.sel(space=valid_space) for m, da in X2d.items()}
    return Y2d, X2d, valid_space


def remove_low_variance(
    Y2d: xr.DataArray,
    X2d: dict[str, xr.DataArray],
    eps: float = 1e-8,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Drop spatial locations with near-zero obs variance."""
    mask = Y2d.std("T") > eps
    Y2d  = Y2d.sel(space=mask)
    X2d  = {m: da.sel(space=mask) for m, da in X2d.items()}
    return Y2d, X2d
