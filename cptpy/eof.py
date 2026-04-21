"""
eof.py
------
CPT uses fixed EOF truncation chosen by the user — NOT variance-threshold
based.  For PRCP the obs truncation acts as a deliberate signal filter;
the leading modes capture large-scale predictable variance and the rest
(mostly local noise) is discarded.

config["y_eof_modes"] : int  — number of obs EOF modes
config["x_eof_modes"] : int  — number of predictor EOF modes (per model)
"""
import numpy as np
import xarray as xr


def _svd_modes(mat: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Thin SVD → return (PC, EOF).
    PC  shape: (T, n)   — time scores (U * s)
    EOF shape: (n, space) — spatial patterns (rows of Vt)
    """
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return (U * s)[:, :n], Vt[:n, :]


def _pct_var(s: np.ndarray, n: int) -> float:
    return 100.0 * np.sum(s[:n] ** 2) / np.sum(s ** 2)


def compute_eof(
    Y_std: xr.DataArray,
    X_std: dict[str, xr.DataArray],
    config: dict,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """
    EOF decomposition of standardised obs and model fields.

    Returns
    -------
    PC_y  : (T, n_y)        obs time scores
    PC_x  : {m: (T, n_x)}  model time scores
    EOF_y : (n_y, space)    obs spatial patterns
    EOF_x : {m: (n_x, space)}
    """
    n_y = config["y_eof_modes"]
    n_x = config["x_eof_modes"]

    # --- obs ---
    Ymat = Y_std.fillna(0).values
    _, sy, _ = np.linalg.svd(Ymat, full_matrices=False)
    PC_y, EOF_y = _svd_modes(Ymat, n_y)
    print(f"  [EOF] obs: {n_y} modes ({_pct_var(sy, n_y):.1f}% variance)")

    # --- models ---
    PC_x:  dict[str, np.ndarray] = {}
    EOF_x: dict[str, np.ndarray] = {}
    for m, da in X_std.items():
        Xmat = da.fillna(0).values
        _, sx, _ = np.linalg.svd(Xmat, full_matrices=False)
        PC_x[m], EOF_x[m] = _svd_modes(Xmat, n_x)
        print(f"  [EOF] {m}: {n_x} modes ({_pct_var(sx, n_x):.1f}% variance)")

    return PC_y, PC_x, EOF_y, EOF_x
