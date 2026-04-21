"""
run_all.py
----------
Runs the full new-system pipeline (hindcast skill + realtime forecast) for
every (lead × season × variable) combination found under ./pycpt.

Input layout (pycpt symlink)
----------------------------
pycpt/
    {SEASON}{YEAR}_{VAR}_KMME/60W-30E_to_5S-35N/data/   ← LEAD 1
    2.LEAD2/{SEASON}{YEAR}_{VAR}_KMME/...                 ← LEAD 2
    3.LEAD3/{SEASON}{YEAR}_{VAR}_KMME/...                 ← LEAD 3

Output layout
-------------
new_system/{LEAD}/{SEASON}{YEAR}_{VAR}/
    MME_skill_scores.nc    ← cross-validated RPSS (hindcast)
    MME_forecast.nc        ← real-time probabilistic forecast

Usage
-----
    python run_all.py                      # all combos, sequential
    python run_all.py --jobs 5             # 5 combos in parallel (recommended: cpu_count // 10)
    python run_all.py --lead LEAD1         # one lead only
    python run_all.py --var PRCP           # one variable only
    python run_all.py --dry-run            # print combinations, don't compute
    python run_all.py --force              # recompute even if output exists

Parallelism notes
-----------------
Each combination runs a CCA mode search using one worker per model (~10 workers).
Set --jobs so that  jobs × 10 ≈ total CPU cores.
Example with 52 cores:  --jobs 5  (5 × 10 = 50 workers)
"""

import argparse
import datetime as dt
import logging
import multiprocessing as mp
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from main  import run_hindcast, run_forecast_stage, run_plots, InsufficientDataError
from utils import Timer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

PYCPT_ROOT = Path("pycpt")
REGION     = "60W-30E_to_5S-35N"
OUTPUT_ROOT = Path("new_system")

LEAD_DIRS = {
    "LEAD1": PYCPT_ROOT,
    "LEAD2": PYCPT_ROOT / "2.LEAD2",
    "LEAD3": PYCPT_ROOT / "3.LEAD3",
}

VARIABLES = ["PRCP", "T2M"]

# 3-letter tag → CPT target string (for display / config["target"])
SEASON_TARGETS = {
    "AMJ": "Apr-Jun",
    "ASO": "Aug-Oct",
    "DJF": "Dec-Feb",
    "FMA": "Feb-Apr",
    "JAS": "Jul-Sep",
    "JFM": "Jan-Mar",
    "JJA": "Jun-Aug",
    "MAM": "Mar-May",
    "MJJ": "May-Jul",
    "NDJ": "Nov-Jan",
    "OND": "Oct-Dec",
    "SON": "Sep-Nov",
}

# Models to include — must match config.yaml; controls which files are loaded
# from the pycpt data directories (any extra files found there are ignored).
MODELS = [
    "CanSIPSIC4",
    "CFSv2",
    "GEOSS2S",
    "GLOSEA6",
    "SEAS51c",
    "SPSv3p5",
    "CCSM4",
    "GCFS2p1",
    "METEOFRANCE8",
    "SPEAR",
]

# EOF / CCA hyper-parameters — same for all combinations
BASE_CONFIG = dict(
    models                 = MODELS,
    target_first_year      = 1991,
    target_final_year      = 2020,
    y_eof_modes            = 6,
    x_eof_modes            = 8,
    cca_modes              = 3,
    drymask_threshold      = None,
    crossvalidation_window = 3,
    percentiles            = [10, 33, 67, 90],
    plot                   = {"skill": False, "forecast": False},
)


# ─── Discovery ────────────────────────────────────────────────────────────────

def _scan_lead(lead_dir: Path, var: str) -> list[tuple[str, int, Path]]:
    """
    Return (season_tag, year, region_dir) for every {SEASON}{YEAR}_{VAR}_KMME
    folder in lead_dir.
    """
    pat = re.compile(r"^([A-Z]{3})(\d{4})_" + var + r"_KMME$")
    hits = []
    if not lead_dir.exists():
        log.warning("Lead dir not found: %s", lead_dir)
        return hits
    for d in sorted(lead_dir.iterdir()):
        m = pat.match(d.name)
        if m and d.is_dir():
            hits.append((m.group(1), int(m.group(2)), d / REGION))
    return hits


def enumerate_combos(
    lead_filter: str | None = None,
    var_filter:  str | None = None,
) -> list[tuple[str, str, int, str, Path]]:
    """Return list of (lead, season_tag, year, var, region_dir)."""
    combos = []
    for lead, lead_dir in LEAD_DIRS.items():
        if lead_filter and lead != lead_filter:
            continue
        for var in VARIABLES:
            if var_filter and var != var_filter:
                continue
            for season_tag, year, region_dir in _scan_lead(lead_dir, var):
                combos.append((lead, season_tag, year, var, region_dir))
    return combos


# ─── Single-combination runner ────────────────────────────────────────────────

def run_one(
    lead:       str,
    season_tag: str,
    year:       int,
    var:        str,
    region_dir: Path,
    force:      bool = False,
    cca_n_jobs: int  = -1,
) -> tuple[str, str, int, str, bool | None]:
    """
    Run hindcast + forecast for one combination.
    Returns (lead, season_tag, year, var, status) where status is
    True=success, None=skipped, False=error.
    cca_n_jobs controls inner CCA parallelism per combination.
    """
    # Re-attach logging in worker processes (lost after fork)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    data_dir   = region_dir / "data"
    output_dir = OUTPUT_ROOT / lead / f"{season_tag}{year}_{var}"

    skill_nc = output_dir / "MME_skill_scores.nc"
    if skill_nc.exists() and not force:
        log.info("%-7s %s%d %-4s — cached, skipping",
                 lead, season_tag, year, var)
        return lead, season_tag, year, var, True

    label = f"{lead} | {season_tag}{year} | {var}"
    log.info("Running: %s", label)

    config = {
        **BASE_CONFIG,
        "variable":   var,
        "target":     SEASON_TARGETS.get(season_tag, season_tag),
        "fdate":      dt.datetime(year, 1, 1),
        "base_dir":   str(data_dir),
        "output_dir": str(output_dir),
        "cca_n_jobs": cca_n_jobs,
    }

    timer = Timer()
    try:
        hindcast = run_hindcast(config, timer)
        run_forecast_stage(hindcast, config, timer)
        run_plots(config, timer)
        timer.total()
        log.info("Done: %s → %s", label, output_dir)
        return lead, season_tag, year, var, True
    except InsufficientDataError as e:
        log.warning("SKIPPED (insufficient data): %s — %s", label, e)
        return lead, season_tag, year, var, None
    except Exception:
        log.error("FAILED: %s\n%s", label, traceback.format_exc())
        return lead, season_tag, year, var, False


# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_all(
    lead_filter: str | None = None,
    var_filter:  str | None = None,
    dry_run:     bool = False,
    force:       bool = False,
    jobs:        int  = 1,
) -> None:
    combos = enumerate_combos(lead_filter, var_filter)

    if not combos:
        log.error("No combinations found — check pycpt directory structure.")
        sys.exit(1)

    log.info("Found %d combinations to process.", len(combos))

    if dry_run:
        print(f"\n{'Lead':<7} {'Season':>9} {'Var':<5}  Data dir")
        print("-" * 70)
        for lead, season, year, var, region_dir in combos:
            data_dir = region_dir / "data"
            status   = "OK" if data_dir.exists() else "MISSING"
            print(f"{lead:<7} {season+str(year):>9} {var:<5}  [{status}] {data_dir}")
        print(f"\nTotal: {len(combos)}")
        return

    # Divide cores: outer jobs × inner CCA workers ≈ total CPUs
    total_cpus  = os.cpu_count() or 1
    cca_n_jobs  = max(1, total_cpus // jobs)
    log.info("Parallelism: %d outer jobs × %d CCA workers = %d cores",
             jobs, cca_n_jobs, jobs * cca_n_jobs)

    n_ok = n_skip = n_err = 0

    if jobs == 1:
        # Sequential — simpler output, easier to read logs
        for lead, season, year, var, region_dir in combos:
            _, _, _, _, status = run_one(
                lead, season, year, var, region_dir, force, cca_n_jobs
            )
            if status is True:   n_ok   += 1
            elif status is None: n_skip += 1
            else:                n_err  += 1
    else:
        # Parallel — use spawn context to avoid issues with nested ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as exe:
            futures = {
                exe.submit(run_one, lead, season, year, var, region_dir, force, cca_n_jobs): (lead, season, year, var)
                for lead, season, year, var, region_dir in combos
            }
            for fut in as_completed(futures):
                try:
                    _, _, _, _, status = fut.result()
                except Exception:
                    label = "{}|{}{}|{}".format(*futures[fut])
                    log.error("FAILED (executor): %s\n%s", label, traceback.format_exc())
                    status = False
                if status is True:   n_ok   += 1
                elif status is None: n_skip += 1
                else:                n_err  += 1

    print(f"\n{'─'*50}")
    print(f"Completed {n_ok}/{len(combos)}  "
          f"(skipped {n_skip} dry, {n_err} failed)")
    if n_err:
        print("Check logs above for pipeline errors.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lead",    help="Run only this lead (e.g. LEAD1)")
    parser.add_argument("--var",     help="Run only this variable (PRCP or T2M)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print combinations and data paths, don't compute")
    parser.add_argument("--force",   action="store_true",
                        help="Recompute even when output files already exist")
    parser.add_argument("--jobs",    type=int, default=1,
                        help="Combinations to run in parallel. "
                             "Rule of thumb: cpu_count // 10  (e.g. --jobs 5 for 52 cores). "
                             "Inner CCA workers are set to cpu_count // jobs automatically.")
    args = parser.parse_args()

    run_all(
        lead_filter = args.lead,
        var_filter  = args.var,
        dry_run     = args.dry_run,
        force       = args.force,
        jobs        = args.jobs,
    )


if __name__ == "__main__":
    main()
