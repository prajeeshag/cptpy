"""
cca.py
------
CPT-faithful CCA with per-model mode optimization.

Mode search (CPT style):
  For each model, exhaustively search all valid (n_x, n_y, n_cca) combinations,
  evaluate cross-validated RPSS for each, and select the combination that
  maximizes mean RPSS across the domain.  The full-period model is then fit
  with those optimal modes.

Search space (from config):
  n_y   in [1 .. y_eof_modes]   -- obs EOF modes
  n_x   in [1 .. x_eof_modes]   -- predictor EOF modes
  n_cca in [1 .. cca_modes],  constrained to <= min(n_x, n_y)

Each combination runs a full LOO pass, so runtime scales as:
  n_combinations x T x CCA_fit_time
  ~40 combos x 30 years x ~1ms = ~1-2s per model -> ~15s total for 9 models.

Parallelism: model searches run in parallel via multiprocessing.
"""
import os
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cross_decomposition import CCA


# --------------------------------------------------------------------------- #
#  Low-level helpers  (module-level for multiprocessing pickling)             #
# --------------------------------------------------------------------------- #

def _loo_indices(T, t, w):
    exclude = np.arange(max(0, t - w), min(T, t + w + 1))
    return np.setdiff1d(np.arange(T), exclude)


def _standardize_np(X_train, X_test, eps=1e-8):
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=1).clip(min=eps)
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def _cca_predict(cca, X_train, Y_train, X_test):
    """Manual CPT-style CCA reconstruction via canonical variates."""
    Xc_tr, Yc_tr = cca.transform(X_train, Y_train)
    r = np.array([
        np.corrcoef(Xc_tr[:, k], Yc_tr[:, k])[0, 1]
        for k in range(cca.n_components)
    ])
    Xc_te = cca.transform(X_test)
    return (Xc_te * r) @ cca.y_rotations_.T   # (n_test, n_y)


def _inflate_variance(Y_pred, Y_true, eps=1e-8):
    """Barnett & Preisendorfer (1987) variance inflation."""
    mu    = Y_pred.mean(axis=0)
    scale = Y_true.std(axis=0, ddof=1) / Y_pred.std(axis=0, ddof=1).clip(min=eps)
    return mu + (Y_pred - mu) * scale


# --------------------------------------------------------------------------- #
#  Single LOO pass for fixed (n_x, n_y, n_cca)                               #
# --------------------------------------------------------------------------- #

def _loo_pass(X, PC_y, n_x, n_y, n_cca, w):
    """Full LOO cross-validation. Returns variance-inflated Y_pred (T, n_y)."""
    T      = PC_y.shape[0]
    Xk     = X[:, :n_x]
    Yk     = PC_y[:, :n_y]
    Y_pred = np.zeros_like(Yk)

    for t in range(T):
        train = _loo_indices(T, t, w)
        if len(train) < n_cca * 2:
            continue
        X_tr_std, X_te_std = _standardize_np(Xk[train], Xk[[t]])
        Y_tr_std, _        = _standardize_np(Yk[train], Yk[[t]])
        cca = CCA(n_components=n_cca, max_iter=1000)
        cca.fit(X_tr_std, Y_tr_std)
        Y_pred[t] = _cca_predict(cca, X_tr_std, Y_tr_std, X_te_std)

    return _inflate_variance(Y_pred, Yk)


# --------------------------------------------------------------------------- #
#  RPSS scorer in PC space (fast; used for mode comparison only)             #
# --------------------------------------------------------------------------- #

def _rpss_pc(Y_pred, PC_y, n_y, w, n_cca):
    """
    Mean RPSS across PC dimensions using LOO tercile thresholds + t-CDF.
    Scores in PC space for speed -- relative ranking across combos is valid.
    """
    from scipy.stats import t as tdist

    Yk   = PC_y[:, :n_y]
    T    = Yk.shape[0]
    df   = max(T - n_cca - 1, 1)
    resid = Yk - Y_pred
    sigma = resid.std(axis=0, ddof=1).clip(min=1e-8)

    rpss_dims = []
    for d in range(n_y):
        mu_d  = Y_pred[:, d]
        obs_d = Yk[:, d]
        sig_d = sigma[d]
        rps_f = rps_c = 0.0

        for t in range(T):
            excl  = np.arange(max(0, t - w), min(T, t + w + 1))
            tr    = np.setdiff1d(np.arange(T), excl)
            q33   = np.percentile(obs_d[tr], 33)
            q67   = np.percentile(obs_d[tr], 67)

            pb = tdist.cdf((q33 - mu_d[t]) / sig_d, df)
            pa = 1 - tdist.cdf((q67 - mu_d[t]) / sig_d, df)
            pn = np.clip(1 - pb - pa, 0, 1)
            fc = np.cumsum([pb, pn, pa])

            cat = 0 if obs_d[t] <= q33 else (1 if obs_d[t] <= q67 else 2)
            oc  = np.array([float(cat == 0), float(cat <= 1), 1.0])
            cc  = np.array([1/3, 2/3, 1.0])

            rps_f += np.sum((fc - oc) ** 2)
            rps_c += np.sum((cc - oc) ** 2)

        rps_f /= T;  rps_c /= T
        rpss_dims.append(0.0 if rps_c == 0 else 1 - rps_f / rps_c)

    return float(np.mean(rpss_dims))


# --------------------------------------------------------------------------- #
#  Per-model search worker  (top-level for pickling)                         #
# --------------------------------------------------------------------------- #

def _search_model(args):
    """
    Exhaustive mode search for one model.
    Returns (model_name, best_modes, best_rpss, best_Y_pred, score_table).
    """
    m, X, PC_y, config = args
    max_nx   = config["x_eof_modes"]
    max_ny   = config["y_eof_modes"]
    max_ncca = config["cca_modes"]
    w        = config["crossvalidation_window"]

    best_rpss  = -np.inf
    best_modes = (1, 1, 1)
    best_pred  = None
    scores     = []

    for n_x, n_y, n_cca in product(
        range(1, max_nx + 1),
        range(1, max_ny + 1),
        range(1, max_ncca + 1),
    ):
        if n_cca > min(n_x, n_y):
            continue

        Y_pred = _loo_pass(X, PC_y, n_x, n_y, n_cca, w)
        rpss   = _rpss_pc(Y_pred, PC_y, n_y, w, n_cca)
        scores.append((n_x, n_y, n_cca, rpss))

        if rpss > best_rpss:
            best_rpss  = rpss
            best_modes = (n_x, n_y, n_cca)
            best_pred  = Y_pred

    return m, best_modes, best_rpss, best_pred, scores


# --------------------------------------------------------------------------- #
#  Full-period model fit with optimal modes                                   #
# --------------------------------------------------------------------------- #

def _fit_full_model(X, PC_y, n_x, n_y, n_cca):
    Xk = X[:, :n_x]
    Yk = PC_y[:, :n_y]
    X_std, _ = _standardize_np(Xk, Xk)
    Y_std, _ = _standardize_np(Yk, Yk)

    cca = CCA(n_components=n_cca, max_iter=1000)
    cca.fit(X_std, Y_std)

    Xc, Yc = cca.transform(X_std, Y_std)
    cca._r       = np.array([np.corrcoef(Xc[:, k], Yc[:, k])[0, 1]
                              for k in range(n_cca)])
    cca._X_mu    = Xk.mean(axis=0)
    cca._X_sigma = Xk.std(axis=0, ddof=1).clip(min=1e-8)
    cca._Y_mu    = Yk.mean(axis=0)
    cca._Y_sigma = Yk.std(axis=0, ddof=1).clip(min=1e-8)
    cca._n_x     = n_x
    cca._n_y     = n_y
    cca._n_cca   = n_cca
    return cca


def _count_combos(config):
    return sum(
        1 for nx, ny, nc in product(
            range(1, config["x_eof_modes"] + 1),
            range(1, config["y_eof_modes"] + 1),
            range(1, config["cca_modes"] + 1),
        ) if nc <= min(nx, ny)
    )


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #

def run_cca_cv(PC_x, PC_y, config, n_jobs=-1):
    """
    Per-model mode-optimized LOO CCA (CPT style).

    Parameters
    ----------
    PC_x   : {model: (T, x_eof_modes_max)}
    PC_y   : (T, y_eof_modes_max)
    config : x_eof_modes, y_eof_modes, cca_modes, crossvalidation_window
    n_jobs : parallel workers (-1 = all CPUs)

    Returns
    -------
    Y_pred_pc  : {model: (T, n_y_opt)}   CV hindcast in Y-PC space
    cca_models : {model: CCA}            full-period models
    best_modes : {model: (n_x, n_y, n_cca)}
    """
    n_combos  = _count_combos(config)
    n_workers = min(os.cpu_count() if n_jobs == -1 else n_jobs, len(PC_x))
    print(f"  [Mode search] {len(PC_x)} models x {n_combos} combos, "
          f"{n_workers} parallel workers")

    args_list = [(m, PC_x[m], PC_y, config) for m in PC_x]
    raw = {}

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_search_model, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            m, modes, rpss, pred, scores = fut.result()
            raw[m] = (modes, pred)
            nx, ny, nc = modes
            print(f"  [Mode search] {m:20s}  "
                  f"n_x={nx} n_y={ny} n_cca={nc}  RPSS={rpss:+.4f}")

    Y_pred_pc  = {}
    cca_models = {}
    best_modes = {}

    for m in PC_x:
        (n_x, n_y, n_cca), pred = raw[m]
        Y_pred_pc[m]  = pred
        cca_models[m] = _fit_full_model(PC_x[m], PC_y, n_x, n_y, n_cca)
        best_modes[m] = (n_x, n_y, n_cca)

    return Y_pred_pc, cca_models, best_modes


def forecast_cca(cca_models, PC_x_f, EOF_y):
    """
    Real forecast: apply full-period CCA with each model's optimal modes.
    Models in cca_models that have no entry in PC_x_f (missing forecast file)
    are silently skipped — the MME average is computed over available models only.
    Returns {model: (n_f, space)}.
    """
    import logging
    log = logging.getLogger(__name__)
    Y_f = {}
    for m, cca in cca_models.items():
        if m not in PC_x_f:
            log.warning("forecast_cca: no forecast PC for %s — skipping from MME", m)
            continue
        n_x, n_y = cca._n_x, cca._n_y
        X_f      = PC_x_f[m][:, :n_x]
        X_f_std  = (X_f - cca._X_mu) / cca._X_sigma
        Xc_f     = cca.transform(X_f_std)
        Y_pred_std = (Xc_f * cca._r) @ cca.y_rotations_.T   # (n_f, n_y)
        Y_pred_pc  = Y_pred_std * cca._Y_sigma + cca._Y_mu
        Y_f[m]     = Y_pred_pc @ EOF_y[:n_y, :]               # (n_f, space)
    return Y_f
