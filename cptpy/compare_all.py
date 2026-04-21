"""
compare_all.py
--------------
Compares new-system RPSS skill scores (produced by run_all.py) against
pycpt output for every (lead × season × variable) combination.

Reads from
----------
new_system/{LEAD}/{SEASON}{YEAR}_{VAR}/MME_skill_scores.nc   ← run_all.py output
pycpt/[2.LEAD2/][3.LEAD3/]{SEASON}{YEAR}_{VAR}_KMME/
    60W-30E_to_5S-35N/output/MME_skill_scores.nc             ← pycpt benchmark

Outputs
-------
comparison_summary.png                                        ← scatter: new vs pycpt
comparison_maps/{LEAD}_{SEASON}{YEAR}_{VAR}.png               ← 3-panel maps (--maps)

Usage
-----
    python compare_all.py             # summary scatter plot
    python compare_all.py --maps      # also save per-combination map PNGs
    python compare_all.py --lead LEAD1 --var PRCP
"""

import argparse
import logging
import re
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


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_pair(
    lead: str, season_tag: str, year: int, var: str,
) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """Return (new_da, pycpt_da) or (None, None) if either file is missing."""
    new_nc = OUTPUT_ROOT / lead / f"{season_tag}{year}_{var}" / "MME_skill_scores.nc"

    lead_dir  = LEAD_DIRS[lead]
    pycpt_nc  = (lead_dir / f"{season_tag}{year}_{var}_KMME"
                 / REGION / "output" / "MME_skill_scores.nc")

    if not new_nc.exists():
        log.warning("New-system skill not found: %s  (run run_all.py first)", new_nc)
        return None, None
    if not pycpt_nc.exists():
        log.warning("pycpt skill not found: %s", pycpt_nc)
        return None, None

    new_da   = xr.open_dataset(new_nc)["rank_probability_skill_score"]
    pycpt_da = xr.open_dataset(pycpt_nc)["rank_probability_skill_score"] * 0.01
    return new_da, pycpt_da


def _mean(da: xr.DataArray) -> float:
    v = da.values.ravel()
    return float(np.nanmean(v[np.isfinite(v)]))


# ─── Per-combination map ───────────────────────────────────────────────────────

def _save_map(
    new_da: xr.DataArray,
    pycpt_da: xr.DataArray,
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    proj       = ccrs.PlateCarree()
    new_interp = new_da.interp(X=pycpt_da.X, Y=pycpt_da.Y, method="linear")
    diff       = new_interp - pycpt_da

    def _stat(da):
        v = da.values.ravel()
        v = v[np.isfinite(v)]
        return f"mean={v.mean():.3f}  pos={100*np.mean(v>0):.0f}%"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             subplot_kw={"projection": proj})
    fig.suptitle(title, fontsize=12, fontweight="bold")

    def _panel(ax, da, ttl, vmin=-0.5, vmax=0.5, cmap="bwr"):
        im = ax.pcolormesh(da.X.values, da.Y.values, da.values,
                           cmap=cmap, vmin=vmin, vmax=vmax, transform=proj)
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4, linewidth=0.5)
        gl.top_labels   = False
        gl.right_labels = False
        ax.set_title(ttl, fontsize=9)
        return im

    im1 = _panel(axes[0], pycpt_da,   f"pycpt\n{_stat(pycpt_da)}")
    im2 = _panel(axes[1], new_interp, f"New system\n{_stat(new_interp)}")
    im3 = _panel(axes[2], diff,       f"Difference (New − pycpt)\n{_stat(diff)}",
                 vmin=-0.3, vmax=0.3)

    fig.colorbar(im1, ax=axes[:2], orientation="horizontal",
                 pad=0.06, label="RPSS", shrink=0.8)
    fig.colorbar(im3, ax=axes[2], orientation="horizontal",
                 pad=0.06, label="Δ RPSS", shrink=0.8)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved map: %s", path)


# ─── Summary scatter plot ─────────────────────────────────────────────────────

def _plot_summary(results: dict) -> None:
    leads = list(LEAD_DIRS.keys())

    fig, axes = plt.subplots(
        len(VARIABLES), len(leads),
        figsize=(5 * len(leads), 5 * len(VARIABLES)),
        squeeze=False,
    )
    fig.suptitle("Mean RPSS: New System vs pycpt\n(each dot = one season)",
                 fontsize=13, fontweight="bold")

    markers = ["o", "s", "^", "v", "D", "P", "*", "X", "h", "8", "<", ">"]
    season_marker = {s: markers[i % len(markers)]
                     for i, s in enumerate(SEASON_TARGETS)}
    lim = (-0.20, 0.45)

    for ri, var in enumerate(VARIABLES):
        for ci, lead in enumerate(leads):
            ax    = axes[ri][ci]
            pts   = [(s, y, r) for (ld, s, y, v), r in results.items()
                     if ld == lead and v == var]

            if not pts:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{var} — {lead}", fontsize=10)
                continue

            for season, year, r in pts:
                ax.scatter(r["pycpt"], r["new"],
                           s=80, marker=season_marker.get(season, "o"), zorder=3,
                           label=f"{season}{year}")
                ax.annotate(season, (r["pycpt"], r["new"]),
                            textcoords="offset points", xytext=(0, 5), fontsize=6)

            lo, hi = lim
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6)
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
            ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("pycpt mean RPSS", fontsize=9)
            ax.set_ylabel("New system mean RPSS", fontsize=9)
            ax.set_title(f"{var} — {lead}", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3, linewidth=0.5)

            diffs    = [r["new"] - r["pycpt"] for (_, _, r) in pts]
            n_better = sum(d > 0 for d in diffs)
            ax.text(0.03, 0.97,
                    f"N={len(diffs)}  better={n_better}/{len(diffs)}\n"
                    f"mean Δ={np.mean(diffs):+.3f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    out = Path("comparison_summary.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


# ─── Print table ──────────────────────────────────────────────────────────────

def _print_table(results: dict) -> None:
    print(f"\n{'Lead':<7} {'Season':>9} {'Var':<5} {'pycpt':>8} {'New':>8} {'Δ':>8}")
    print("-" * 50)
    for (lead, season, year, var), r in sorted(results.items()):
        delta = r["new"] - r["pycpt"]
        flag  = "↑" if delta > 0 else "↓"
        print(f"{lead:<7} {season+str(year):>9} {var:<5} "
              f"{r['pycpt']:>8.3f} {r['new']:>8.3f} {delta:>+7.3f} {flag}")
    if results:
        all_d = [r["new"] - r["pycpt"] for r in results.values()]
        print(f"\nOverall: mean Δ = {np.mean(all_d):+.3f}  "
              f"better in {sum(d>0 for d in all_d)}/{len(all_d)} cases")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lead",  help="Filter by lead (e.g. LEAD1)")
    parser.add_argument("--var",   help="Filter by variable (PRCP or T2M)")
    parser.add_argument("--maps",  action="store_true",
                        help="Save per-combination 3-panel map PNGs")
    args = parser.parse_args()

    combos  = enumerate_combos(args.lead, args.var)
    results = {}

    for lead, season, year, var, _ in combos:
        new_da, pycpt_da = _load_pair(lead, season, year, var)
        if new_da is None or pycpt_da is None:
            continue

        results[(lead, season, year, var)] = dict(
            new=_mean(new_da),
            pycpt=_mean(pycpt_da),
        )

        if args.maps:
            _save_map(
                new_da, pycpt_da,
                title=f"RPSS  |  {lead}  {season}{year}  {var}",
                path=Path("comparison_maps") / f"{lead}_{season}{year}_{var}.png",
            )

    if not results:
        log.error("No results — run run_all.py first to produce new-system outputs.")
        sys.exit(1)

    _print_table(results)
    _plot_summary(results)


if __name__ == "__main__":
    main()
