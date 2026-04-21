"""
skill.py
--------
RPSS computed with LOO climatological thresholds, matching CPT's
out-of-sample verification standard.
"""
import numpy as np

_Q_LOW  = 33
_Q_HIGH = 67


def _obs_onehot(
    Y_true: np.ndarray,
    q33:    np.ndarray,
    q67:    np.ndarray,
) -> np.ndarray:
    """Observed category one-hot: (T, 3, space) — [below, normal, above]."""
    obs = np.zeros((Y_true.shape[0], 3, Y_true.shape[1]))
    obs[:, 0, :] = Y_true <= q33
    obs[:, 1, :] = (Y_true > q33) & (Y_true <= q67)
    obs[:, 2, :] = Y_true > q67
    return obs


def _fcst_stack(prob: dict[int, np.ndarray]) -> np.ndarray:
    """Stack forecast probs into (T, 3, space) — [below, normal, above]."""
    p_below  = prob[_Q_LOW]
    p_above  = 1 - prob[_Q_HIGH]
    p_normal = np.clip(1 - p_below - p_above, 0, 1)
    return np.stack([p_below, p_normal, p_above], axis=1)


def compute_rpss(
    prob:   dict[int, np.ndarray],
    Y_true: np.ndarray,
    q33:    np.ndarray,
    q67:    np.ndarray,
    w:      int = 0,
) -> np.ndarray:
    """
    RPSS against climatology reference, using LOO thresholds.

    Parameters
    ----------
    prob   : {33: (T, space), 67: (T, space)}  forecast probs
    Y_true : (T, space)
    q33    : (T, space)  LOO lower tercile threshold
    q67    : (T, space)  LOO upper tercile threshold
    w      : LOO exclusion window (for climatology reference)

    Returns
    -------
    rpss : (space,)
    """
    T = Y_true.shape[0]

    obs_oh   = _obs_onehot(Y_true, q33, q67)            # (T, 3, space)
    fcst_cum = np.cumsum(_fcst_stack(prob), axis=1)      # (T, 3, space)
    obs_cum  = np.cumsum(obs_oh,            axis=1)

    rps_fcst = np.mean(np.sum((fcst_cum - obs_cum) ** 2, axis=1), axis=0)

    # Climatology reference: equal tercile probs (1/3 each), LOO-aware
    # For each year t, the clim forecast uses 1/3 for each category
    # (this is exact regardless of LOO since equal probs are always 1/3)
    clim_prob = {
        _Q_LOW:  np.full_like(prob[_Q_LOW],  1 / 3),
        _Q_HIGH: np.full_like(prob[_Q_HIGH], 2 / 3),
    }
    clim_cum = np.cumsum(_fcst_stack(clim_prob), axis=1)
    rps_clim = np.mean(np.sum((clim_cum - obs_cum) ** 2, axis=1), axis=0)

    rps_clim = np.where(rps_clim == 0, np.nan, rps_clim)
    return 1 - rps_fcst / rps_clim
