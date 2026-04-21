"""
plot_comparison.py
------------------
Standalone plot script: reads pre-computed skill scores from skill_output/
(produced by compare_all.py) and pycpt output, then generates comparison figures.

Usage
-----
    python plot_comparison.py                        # all leads, vars, seasons
    python plot_comparison.py --lead LEAD1           # one lead only
    python plot_comparison.py --var PRCP             # one variable only
    python plot_comparison.py --season MAM2025 PRCP  # single combination map
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PYCPT_ROOT  = Path("pycpt")
SKILL_ROOT  = Path("skill_output")
REGION      = "60W-30E_to_5S-35N"
LEAD_DIRS   = {
    "LEAD1": PYCPT_ROOT,
    "LEAD2": PYCPT_ROOT / "2.LEAD2",
    "LEAD3": PYCPT_ROOT / "3.LEAD3",
}
VARIABLES = ["PRCP", "T2M"]


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_pair(lead: str, season_tag: str, year: int, var: str):
    """Return (new_da, pycpt_da) or (None, None) if files are missing."""
    new_nc   = SKILL_ROOT / lead / f"{season_tag}{year}_{var}.nc"
    lead_dir = LEAD_DIRS[lead]
    pycpt_nc = lead_dir / f"{season_tag}{year}_{var}_KMME" / REGION / "output" / "MME_skill_scores.nc"

    if not new_nc.exists() or not pycpt_nc.exists():
        return None, None

    new_da   = xr.open_dataset(new_nc)["rank_probability_skill_score"]
    pycpt_da = xr.open_dataset(pycpt_nc)["rank_probability_skill_score"]
    return new_da, pycpt_da


def _discover(lead_filter=None, var_filter=None):
    """Yield (lead, season_tag, year, var) for all available skill files."""
    for lead in LEAD_DIRS:
        if lead_filter and lead != lead_filter:
            continue
        lead_skill = SKILL_ROOT / lead
        if not lead_skill.exists():
            continue
        for f in sorted(lead_skill.glob("*.nc")):
            m = re.match(r"([A-Z]{3})(\d{4})_([A-Z0-9]+)\.nc$", f.name)
            if not m:
                continue
            season_tag, year, var = m.group(1), int(m.group(2)), m.group(3)
            if var_filter and var != var_filter:
                continue
            yield lead, season_tag, year, var


def _mean(da):
    v = da.values.ravel()
    return float(np.nanmean(v[np.isfinite(v)]))


# ─── Single-combination map ────────────────────────────────────────────────────

def plot_one(lead: str, season_tag: str, year: int, var: str,
             out_path: Path | None = None) -> None:
    new_da, pycpt_da = _load_pair(lead, season_tag, year, var)
    if new_da is None:
        print(f"Files not found for {lead} {season_tag}{year} {var}")
        return

    proj       = ccrs.PlateCarree()
    new_interp = new_da.interp(X=pycpt_da.X, Y=pycpt_da.Y, method="linear")
    diff       = new_interp - pycpt_da

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             subplot_kw={"projection": proj})
    fig.suptitle(f"RPSS  |  {lead}  {season_tag}{year}  {var}", fontsize=12, fontweight="bold")

    def _stat(da):
        v = da.values.ravel()
        v = v[np.isfinite(v)]
        return f"mean={v.mean():.3f}  pos={100*np.mean(v>0):.0f}%"

    def _panel(ax, da, title, vmin=-0.5, vmax=0.5, cmap="bwr"):
        im = ax.pcolormesh(da.X.values, da.Y.values, da.values,
                           cmap=cmap, vmin=vmin, vmax=vmax, transform=proj)
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4, linewidth=0.5)
        gl.top_labels   = False
        gl.right_labels = False
        ax.set_title(title, fontsize=9)
        return im

    im1 = _panel(axes[0], pycpt_da,   f"pycpt\n{_stat(pycpt_da)}")
    im2 = _panel(axes[1], new_interp, f"New system\n{_stat(new_interp)}")
    im3 = _panel(axes[2], diff,       f"Difference (New − pycpt)\n{_stat(diff)}",
                 vmin=-0.3, vmax=0.3)

    fig.colorbar(im1, ax=axes[:2], orientation="horizontal",
                 pad=0.06, label="RPSS", shrink=0.8)
    fig.colorbar(im3, ax=axes[2], orientation="horizontal",
                 pad=0.06, label="Δ RPSS", shrink=0.8)

    if out_path is None:
        out_path = Path(f"rpss_{lead}_{season_tag}{year}_{var}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─── Summary scatter (all combinations) ───────────────────────────────────────

def plot_summary(lead_filter=None, var_filter=None) -> None:
    leads = [l for l in LEAD_DIRS if not lead_filter or l == lead_filter]
    vars_ = [v for v in VARIABLES if not var_filter or v == var_filter]

    fig, axes = plt.subplots(
        len(vars_), len(leads),
        figsize=(5 * len(leads), 5 * len(vars_)),
        squeeze=False,
    )
    fig.suptitle("Mean RPSS: New System vs pycpt", fontsize=13, fontweight="bold")

    lim = (-0.20, 0.45)

    for ri, var in enumerate(vars_):
        for ci, lead in enumerate(leads):
            ax = axes[ri][ci]
            points = []

            for (ld, season_tag, year, v) in _discover(lead, var):
                new_da, pycpt_da = _load_pair(ld, season_tag, year, v)
                if new_da is None:
                    continue
                points.append((season_tag, year, _mean(new_da), _mean(pycpt_da)))

            if not points:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{var} — {lead}", fontsize=10)
                continue

            markers = ["o","s","^","v","D","P","*","X","h","8","<",">"]
            seen    = {}
            for i, (season, year, new_m, pycpt_m) in enumerate(points):
                key = season
                if key not in seen:
                    seen[key] = markers[len(seen) % len(markers)]
                ax.scatter(pycpt_m, new_m, s=80, marker=seen[key], zorder=3,
                           label=f"{season}{year}")
                ax.annotate(f"{season}", (pycpt_m, new_m),
                            textcoords="offset points", xytext=(0, 5), fontsize=6)

            lo, hi = lim
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6)
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("pycpt mean RPSS", fontsize=9)
            ax.set_ylabel("New system mean RPSS", fontsize=9)
            ax.set_title(f"{var} — {lead}", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3, linewidth=0.5)

            diffs    = [n - p for (_, _, n, p) in points]
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
    print(f"Saved: {out}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lead",   help="Filter by lead (e.g. LEAD1)")
    parser.add_argument("--var",    help="Filter by variable (PRCP or T2M)")
    parser.add_argument("--season", nargs=2, metavar=("SEASON_YEAR", "VAR"),
                        help="Single-combination map, e.g. --season MAM2025 PRCP")
    args = parser.parse_args()

    if args.season:
        raw, var = args.season
        m = re.match(r"([A-Z]{3})(\d{4})", raw)
        if not m:
            parser.error("--season expects SEASON_YEAR like MAM2025")
        season_tag, year = m.group(1), int(m.group(2))
        # try all leads
        for lead in LEAD_DIRS:
            plot_one(lead, season_tag, year, var)
    else:
        plot_summary(lead_filter=args.lead, var_filter=args.var)


if __name__ == "__main__":
    main()
