"""
Seasonal_v2 — CPT-faithful MME EOF/CCA seasonal forecast pipeline
=================================================================
Run:  python main.py [config.yaml]
"""

from pathlib import Path

import typer
import yaml

from .cca import forecast_cca, run_cca_cv
from .data_io import (
    load_forecast_models,
    load_hindcast_models,
    load_obs,
)
from .eof import compute_eof
from .output import save_mme_forecast, save_skill
from .preprocess import (
    align_time,
    apply_mask,
    filter_sparse_models,
    flatten,
    regrid,
    remove_low_variance,
)
from .probabilistic import (
    compute_error_variance,
    forecast_probabilities,
    hindcast_probabilities,
)
from .transform import standardize_all
from .utils import Timer

app = typer.Typer()


class InsufficientDataError(ValueError):
    """Raised when too few valid grid points survive masking (e.g. dry season)."""


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


# ─── Hindcast ─────────────────────────────────────────────────────────────────


def run_hindcast(
    base_dir: Path,
    obs_file: Path,
    config: dict,
    varname: str,
    timer: Timer,
) -> dict:
    models = config["models"]
    Y = load_obs(obs_file)
    X_models = load_hindcast_models(config, base_dir, models, varname)
    timer.step("Load")

    # 2. Preprocess
    Y, X_models = align_time(Y, X_models, config)
    X_models = regrid(Y, X_models)
    X_models = filter_sparse_models(Y, X_models)
    Y, X_models = apply_mask(Y, X_models, config)
    Y2d, X2d, valid_space = flatten(Y, X_models)
    Y2d, X2d = remove_low_variance(Y2d, X2d)
    timer.step("Preprocess")

    n_pts = Y2d.sizes["space"]
    if n_pts < config.get("min_valid_points", 10):
        raise InsufficientDataError(
            f"Only {n_pts} valid grid points after masking and model filtering."
        )

    # 3. CPT standardisation (both X and Y, per grid point)
    Y_std, X_std, Y_mu, Y_sigma, X_mu, X_sigma = standardize_all(Y2d, X2d)
    timer.step("Standardise")

    # 4. EOF (on standardised fields)
    PC_y, PC_x, EOF_y, EOF_x = compute_eof(Y_std, X_std, config)
    timer.step("EOF")

    # 5. LOO CCA with variance inflation
    PC_y_pred, cca_models, best_modes = run_cca_cv(
        PC_x, PC_y, config, n_jobs=config.get("cca_n_jobs", -1)
    )
    timer.step("CCA (LOO CV)")

    # 6. Back-project CV predictions to obs grid using per-model optimal n_y
    #    Un-standardise: Y_pred = (PC_pred @ EOF_y[:n_y]) * Y_sigma + Y_mu
    Y_pred = {}
    for m in PC_y_pred:
        n_y = best_modes[m][1]
        Y_pred[m] = PC_y_pred[m] @ EOF_y[:n_y, :]  # (T, space) in std units
        Y_pred[m] = Y_pred[m] * Y_sigma.values + Y_mu.values  # un-standardise
    timer.step("Reconstruct")

    # 7. Error variance and df (use mean optimal n_cca across models for df)
    mean_ncca = int(round(sum(best_modes[m][2] for m in best_modes) / len(best_modes)))
    var, df = compute_error_variance(Y2d.values, Y_pred, mean_ncca)
    print(f"  df = {df}")

    # 8. LOO probabilistic hindcast (t-distribution, LOO thresholds)
    w = config["crossvalidation_window"]
    prob = hindcast_probabilities(Y_pred, var, Y2d.values, config, df, w)
    timer.step("Probabilistic")

    # 9. Skill
    out_dir = Path(config.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_skill(prob, Y2d, config, str(out_dir / "MME_skill_scores.nc"))
    timer.step("Skill")

    _diagnostics(prob)

    return dict(
        Y=Y,
        Y2d=Y2d,
        X2d=X2d,
        valid_space=valid_space,
        Y_mu=Y_mu,
        Y_sigma=Y_sigma,
        EOF_y=EOF_y,
        EOF_x=EOF_x,
        cca_models=cca_models,
        best_modes=best_modes,
        var=var,
        df=df,
        base_dir=base_dir,
        models=models,
    )


# ─── Forecast ─────────────────────────────────────────────────────────────────


def run_forecast_stage(hindcast: dict, config: dict, var: str, timer: Timer) -> None:

    # Load & preprocess forecast fields (skip models with missing _f files)
    Xf_models = load_forecast_models(hindcast["base_dir"], hindcast["models"], var)
    Xf_models = regrid(hindcast["Y"], Xf_models)
    _, Xf2d, _ = flatten(hindcast["Y"], Xf_models, valid_space=hindcast["valid_space"])
    timer.step("Forecast load + preprocess")

    # Project forecast anomalies onto hindcast EOFs
    EOF_x = hindcast["EOF_x"]
    X2d = hindcast["X2d"]
    PC_x_f = {}
    for m in Xf2d:
        # Anomaly w.r.t. hindcast climatology, then standardise by hindcast sigma
        Xf_anom = (Xf2d[m] - X2d[m].mean("T")).fillna(0).values
        X_sigma_np = hindcast["X2d"][m].std("T").clip(min=1e-8).values
        Xf_std = Xf_anom / X_sigma_np
        PC_x_f[m] = Xf_std @ EOF_x[m].T  # (n_f, n_x_modes)
    timer.step("Forecast EOF projection")

    # CCA + back-project to obs grid
    Y_f = forecast_cca(hindcast["cca_models"], PC_x_f, hindcast["EOF_y"])

    # Un-standardise
    Y_mu = hindcast["Y_mu"].values
    Y_sigma = hindcast["Y_sigma"].values
    Y_f = {m: Y_f[m] * Y_sigma + Y_mu for m in Y_f}
    timer.step("Forecast CCA + reconstruct")

    # Forecast probabilities (full-period thresholds)
    prob_f = forecast_probabilities(
        Y_f, hindcast["var"], hindcast["Y2d"].values, config, hindcast["df"]
    )

    out_dir = Path(config.get("output_dir", "."))
    save_mme_forecast(
        Y_f,
        hindcast["var"],
        hindcast["Y2d"],
        prob_f,
        config,
        str(out_dir / "MME_forecast.nc"),
    )
    timer.step("Forecast output")


# ─── Plots ────────────────────────────────────────────────────────────────────


def run_plots(config: dict, timer: Timer) -> None:
    cfg = config.get("plot", {})
    if cfg.get("skill") or cfg.get("forecast"):
        try:
            from .plot import plot_extremes, plot_forecast, plot_skill
        except ImportError:
            print("  [Plot] plot.py not found — skipping plots")
            timer.step("Plot")
            return
    out_dir = Path(config.get("output_dir", "."))
    if cfg.get("skill"):
        plot_skill(str(out_dir / "MME_skill_scores.nc"))
    if cfg.get("forecast"):
        plot_forecast(str(out_dir / "MME_forecast.nc"))
        plot_extremes(str(out_dir / "MME_forecast.nc"))
    timer.step("Plot")


# ─── Entry point ──────────────────────────────────────────────────────────────


def _diagnostics(prob: dict) -> None:
    for p in [33, 67]:
        if p in prob:
            print(
                f"  Diag — mean prob{p}: {prob[p].mean():.3f}  "
                f"(should differ from {p / 100:.2f} if forecast has skill)"
            )


# ─── Entry point ──────────────────────────────────────────────────────────────


def _diagnostics(prob: dict) -> None:
    for p in [33, 67]:
        if p in prob:
            print(
                f"  Diag — mean prob{p}: {prob[p].mean():.3f}  "
                f"(should differ from {p / 100:.2f} if forecast has skill)"
            )


@app.command()
def main(
    base_dir: Path, obs_file: Path, var: str, config_path: str = "config.yaml"
) -> None:
    config = _load_config(config_path)
    timer = Timer()
    timer = Timer()

    print("\n=== HINDCAST ===")
    hindcast = run_hindcast(base_dir, obs_file, config, var, timer)

    print("\n=== FORECAST ===")
    run_forecast_stage(hindcast, config, var, timer)

    print("\n=== PLOTS ===")
    run_plots(config, timer)

    print()
    timer.total()


if __name__ == "__main__":
    app()
