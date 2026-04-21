"""
probabilistic.py
----------------
CPT-faithful probability estimation.

CPT uses a t-distribution centred on the CCA prediction, with:
  - error variance computed from LOO residuals
  - degrees of freedom = T - n_cca - 1
  - LOO climatological thresholds (exclude year t when computing terciles)

This avoids the RPSS inflation that occurs when in-sample climatology
is used as both the threshold and the reference.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import t as tdist


# ─── Error variance ───────────────────────────────────────────────────────────

def compute_error_variance(
    Y_true: np.ndarray,
    Y_pred: dict[str, np.ndarray],
    n_cca:  int = 1,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Per-model LOO error variance and degrees of freedom.

    df = T - n_cca - 1  (CPT convention: lose one df per CCA component)

    Returns
    -------
    var : {model: (space,)}
    df  : int
    """
    T   = Y_true.shape[0]
    df  = max(T - n_cca - 1, 1)
    var = {
        m: np.var(Y_true - Y_pred[m], axis=0, ddof=1)
        for m in Y_pred
    }
    return var, df


# ─── LOO climatological thresholds ────────────────────────────────────────────

def loo_tercile_thresholds(
    Y_true: np.ndarray,
    w:      int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each year t, compute q33 and q67 from all years except [t-w, t+w].

    CPT excludes the target year from the climatology when verifying it,
    which gives an honest out-of-sample skill estimate.

    Returns
    -------
    q33 : (T, space)
    q67 : (T, space)
    """
    T = Y_true.shape[0]
    q33 = np.zeros_like(Y_true)
    q67 = np.zeros_like(Y_true)

    def _year(t):
        exclude = np.arange(max(0, t - w), min(T, t + w + 1))
        train   = np.setdiff1d(np.arange(T), exclude)
        return t, np.percentile(Y_true[train], 33, axis=0), \
                  np.percentile(Y_true[train], 67, axis=0)

    with ThreadPoolExecutor(max_workers=T) as exe:
        for t, q33_t, q67_t in exe.map(_year, range(T)):
            q33[t] = q33_t
            q67[t] = q67_t

    return q33, q67


# ─── Probability estimation ───────────────────────────────────────────────────

def hindcast_probabilities(
    Y_pred: dict[str, np.ndarray],
    var:    dict[str, np.ndarray],
    Y_true: np.ndarray,
    config: dict,
    df:     int,
    w:      int = 0,
) -> dict[int, np.ndarray]:
    """
    CPT-faithful hindcast probabilities using t-distribution.

    For each year t:
      - threshold = LOO tercile from training years (excludes ±w around t)
      - P(Y ≤ thresh) = t-CDF((thresh - mu) / sigma, df)
      - Average across models (equal weights)

    Returns {percentile: (T, space)}
    """
    percentiles = config["percentiles"]

    # LOO thresholds: shape (T, space)
    q33, q67 = loo_tercile_thresholds(Y_true, w)

    # Also compute extreme thresholds LOO-style
    T   = Y_true.shape[0]
    q10 = np.zeros_like(Y_true)
    q90 = np.zeros_like(Y_true)

    def _extreme(t):
        exclude = np.arange(max(0, t - w), min(T, t + w + 1))
        train   = np.setdiff1d(np.arange(T), exclude)
        return t, np.percentile(Y_true[train], 10, axis=0), \
                  np.percentile(Y_true[train], 90, axis=0)

    with ThreadPoolExecutor(max_workers=T) as exe:
        for t, q10_t, q90_t in exe.map(_extreme, range(T)):
            q10[t] = q10_t
            q90[t] = q90_t

    thresh_map = {10: q10, 33: q33, 67: q67, 90: q90}

    def _one_model(m):
        mu    = Y_pred[m]                          # (T, space)
        sigma = np.sqrt(var[m]).clip(min=1e-8)     # (space,) broadcast
        return {p: tdist.cdf((thresh_map[p] - mu) / sigma, df)
                for p in percentiles}

    # scipy t.cdf releases the GIL — threads give real parallelism here
    results: dict[int, list] = {p: [] for p in percentiles}
    with ThreadPoolExecutor(max_workers=len(Y_pred)) as exe:
        for model_probs in exe.map(_one_model, Y_pred):
            for p in percentiles:
                results[p].append(model_probs[p])

    return {p: np.mean(results[p], axis=0) for p in percentiles}


def forecast_probabilities(
    Y_f:    dict[str, np.ndarray],
    var:    dict[str, np.ndarray],
    Y_true: np.ndarray,
    config: dict,
    df:     int,
) -> dict[int, np.ndarray]:
    """
    Real forecast probabilities using full-period climatological thresholds.

    Y_f values have shape (1, space) or (n_f, space).
    Returns {percentile: (n_f, space)}.
    """
    percentiles = config["percentiles"]
    thresh_map  = {
        p: np.percentile(Y_true, p, axis=0)
        for p in percentiles
    }

    results: dict[int, list] = {p: [] for p in percentiles}

    for m in Y_f:
        mu    = Y_f[m]
        sigma = np.sqrt(var[m]).clip(min=1e-8)

        for p in percentiles:
            z = (thresh_map[p] - mu) / sigma
            results[p].append(tdist.cdf(z, df))

    return {p: np.mean(results[p], axis=0) for p in percentiles}
