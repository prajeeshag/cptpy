import numpy as np
import xarray as xr

from .probabilistic import loo_tercile_thresholds
from .skill import compute_rpss


def save_skill(
    prob: dict[int, np.ndarray],
    Y2d: xr.DataArray,
    config: dict,
    path: str,
    dry_mask: xr.DataArray | None = None,
) -> None:
    """Compute LOO RPSS and write to NetCDF."""
    w = config["crossvalidation_window"]
    q33, q67 = loo_tercile_thresholds(Y2d.values, w)

    rpss = compute_rpss(prob, Y2d.values, q33, q67, w)

    da = xr.DataArray(rpss, dims=("space",), coords={"space": Y2d["space"]}).unstack(
        "space"
    )
    if dry_mask is not None:
        da = da.where(dry_mask)

    xr.Dataset({"rank_probability_skill_score": da}).to_netcdf(path)
    print(f"  Skill saved → {path}")
    print(
        f"  RPSS  mean={np.nanmean(rpss):+.3f}  "
        f"pos={100 * np.mean(rpss > 0):.0f}%  "
        f"max={np.nanmax(rpss):+.3f}"
    )


def save_mme_forecast(
    Y2d: xr.DataArray,
    prob_f: dict[int, np.ndarray],
    path: str,
    dry_mask: xr.DataArray | None = None,
) -> None:
    """Write MME probabilistic forecast to NetCDF."""
    p_below = prob_f[33]
    p_above = 1 - prob_f[67]
    p_normal = np.clip(1 - p_below - p_above, 0, 1)
    prob_mme = np.stack([p_below, p_normal, p_above], axis=0)  # (3, n_f, space)

    # coords_t = {"T": [config["fdate"]], "space": Y2d["space"]}
    coords_t = {"space": Y2d["space"]}

    prob_da = xr.DataArray(
        prob_mme,
        dims=("C", "T", "space"),
        coords={"C": ["below", "normal", "above"], **coords_t},
    ).unstack("space")

    poe10_da = xr.DataArray(prob_f[10], dims=("T", "space"), coords=coords_t).unstack(
        "space"
    )

    poe90_da = xr.DataArray(
        1 - prob_f[90], dims=("T", "space"), coords=coords_t
    ).unstack("space")

    print("n points in prob_da before masking:", prob_da.notnull().sum().values)

    if dry_mask is not None:
        prob_da = prob_da.where(dry_mask)
        poe10_da = poe10_da.where(dry_mask)
        poe90_da = poe90_da.where(dry_mask)

    print("n points in prob_da after masking:", prob_da.notnull().sum().values)

    problematic = (prob_da[2].values - poe90_da.values) < 0.0
    nprob_points = problematic.sum()
    print(f"Problematic grid points: {nprob_points}")
    if nprob_points > 0:
        raise RuntimeError("Problematic grid points detected; p90 > p_above")

    problematic = (prob_da[0].values - poe10_da.values) < 0.0
    nprob_points = problematic.sum()
    print(f"Problematic grid points: {nprob_points}")
    if nprob_points > 0:
        raise RuntimeError("Problematic grid points detected; p10 > p_below")

    xr.Dataset(
        {
            "probabilistic": prob_da,
            "prob_below_p10": poe10_da,
            "prob_above_p90": poe90_da,
        }
    ).to_netcdf(path)
    print(f"  Forecast saved → {path}")
