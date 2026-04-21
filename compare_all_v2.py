import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from run_all import (
    PYCPT_ROOT, REGION, LEAD_DIRS, VARIABLES,
    SEASON_TARGETS, OUTPUT_ROOT, enumerate_combos,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Season order ─────────────────────────────────────────────────────────────
SEASON_ORDER = [
    "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA",
    "JAS", "ASO", "SON", "OND", "NDJ", "DJF"
]

# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_pair(lead, season_tag, year, var):
    new_nc = OUTPUT_ROOT / lead / f"{season_tag}{year}_{var}" / "MME_skill_scores.nc"

    lead_dir  = LEAD_DIRS[lead]
    pycpt_nc  = (lead_dir / f"{season_tag}{year}_{var}_KMME"
                 / REGION / "output" / "MME_skill_scores.nc")

    if not new_nc.exists() or not pycpt_nc.exists():
        return None, None

    new_da   = xr.open_dataset(new_nc)["rank_probability_skill_score"]
    pycpt_da = xr.open_dataset(pycpt_nc)["rank_probability_skill_score"] * 0.01

    return new_da, pycpt_da


def _mean(da):
    v = da.values.ravel()
    v = v[np.isfinite(v)]
    return float(np.nanmean(v)) if len(v) > 0 else np.nan


# ─── Multi-season map ─────────────────────────────────────────────────────────

def _save_multi_map(entries, lead, var):
    entries = sorted(
        entries,
        key=lambda x: (
            SEASON_ORDER.index(x[0]) if x[0] in SEASON_ORDER else 999,
            x[1]
        )
    )

    n = len(entries)
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        n, 3, figsize=(12, 4*n),
        subplot_kw={"projection": proj}
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (season, year, new_da, pycpt_da) in enumerate(entries):

        new_interp = new_da.interp(X=pycpt_da.X, Y=pycpt_da.Y, method="linear")
        diff = new_interp - pycpt_da

        title = f"{season} {year}"

        def _panel(ax, da, ttl, vmin=-0.5, vmax=0.5):
            im = ax.pcolormesh(
                da.X, da.Y, da,
                cmap="bwr", vmin=vmin, vmax=vmax,
                transform=proj
            )
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_title(ttl, fontsize=8)
            return im

        im = _panel(axes[i, 0], pycpt_da,   f"{title} pycpt")
        _panel(axes[i, 1], new_interp, f"{title} new")
        _panel(axes[i, 2], diff,       f"{title} diff", vmin=-0.3, vmax=0.3)

    # Shared colorbars
    #fig.colorbar(im, ax=axes[:, :2], orientation="horizontal", pad=2, label="RPSS")
    #fig.colorbar(im, ax=axes[:, 2], orientation="horizontal", pad=2, label="Δ RPSS")

    plt.subplots_adjust(hspace=0.2, wspace=0.05)

    out = Path("comparison_maps") / f"{lead}_{var}_ALL.png"
    out.parent.mkdir(exist_ok=True)

    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    log.info("Saved: %s", out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", action="store_true")
    args = parser.parse_args()

    combos = enumerate_combos(None, None)
    results = {}
    maps = {}

    for lead, season, year, var, _ in combos:
        new_da, pycpt_da = _load_pair(lead, season, year, var)
        if new_da is None:
            continue

        results[(lead, season, year, var)] = dict(
            new=_mean(new_da),
            pycpt=_mean(pycpt_da),
        )

        if args.maps:
            key = (lead, var)
            maps.setdefault(key, []).append((season, year, new_da, pycpt_da))

    if args.maps:
        for (lead, var), entries in maps.items():
            _save_multi_map(entries, lead, var)



if __name__ == "__main__":
    main()
