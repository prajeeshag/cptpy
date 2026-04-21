import numpy as np
import xarray as xr
from skill import compute_rpss
from probabilistic import loo_tercile_thresholds


def save_skill(
    prob:   dict[int, np.ndarray],
    Y2d:    xr.DataArray,
    config: dict,
    path:   str,
) -> None:
    """Compute LOO RPSS and write to NetCDF."""
    w   = config["crossvalidation_window"]
    q33, q67 = loo_tercile_thresholds(Y2d.values, w)

    rpss = compute_rpss(prob, Y2d.values, q33, q67, w)

    da = (
        xr.DataArray(rpss, dims=("space",), coords={"space": Y2d["space"]})
        .unstack("space")
    )
    xr.Dataset({"rank_probability_skill_score": da}).to_netcdf(path)
    print(f"  Skill saved → {path}")
    print(f"  RPSS  mean={np.nanmean(rpss):+.3f}  "
          f"pos={100*np.mean(rpss>0):.0f}%  "
          f"max={np.nanmax(rpss):+.3f}")


def save_mme_forecast(
    Y_f:    dict[str, np.ndarray],
    var:    dict[str, np.ndarray],
    Y2d:    xr.DataArray,
    prob_f: dict[int, np.ndarray],
    config: dict,
    path:   str,
) -> None:
    """Write MME probabilistic forecast to NetCDF."""
    p_below  = prob_f[33]
    p_above  = 1 - prob_f[67]
    p_normal = np.clip(1 - p_below - p_above, 0, 1)
    prob_mme = np.stack([p_below, p_normal, p_above], axis=0)   # (3, n_f, space)

    coords_t = {"T": [config["fdate"]], "space": Y2d["space"]}

    prob_da  = xr.DataArray(
        prob_mme,
        dims=("C", "T", "space"),
        coords={"C": ["below", "normal", "above"], **coords_t},
    ).unstack("space")

    poe10_da = xr.DataArray(
        prob_f[10], dims=("T", "space"), coords=coords_t
    ).unstack("space")

    poe90_da = xr.DataArray(
        prob_f[90], dims=("T", "space"), coords=coords_t
    ).unstack("space")

    xr.Dataset({
        "probabilistic":  prob_da,
        "prob_exceed_10": poe10_da,
        "prob_exceed_90": poe90_da,
    }).to_netcdf(path)
    print(f"  Forecast saved → {path}")
