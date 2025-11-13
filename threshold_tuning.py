# ============================================
# threshold_tuning.py
# ============================================
"""
Functions for class-wise threshold adjustment and application
after calibrating model probabilities.

Includes:
- apply_thresholds
- tune_thresholds_by_class
- coordinate_ascent_thresholds
"""

import numpy as np


def apply_thresholds(proba: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """
    Rule: if any class surpasses its threshold, pick the one with the largest (proba_k - thr_k);
    if none surpass, fallback to argmax.
    """
    proba = np.asarray(proba)
    thr = np.asarray(thr)
    y_argmax = proba.argmax(axis=1)

    # margin matrix (proba - thr); <0 means "did not surpass threshold"
    margins = proba - thr.reshape(1, -1)
    best_cls = margins.argmax(axis=1)
    best_margin = margins[np.arange(len(proba)), best_cls]

    y_pred = y_argmax.copy()
    mask = best_margin >= 0.0
    y_pred[mask] = best_cls[mask]
    return y_pred


def tune_thresholds_by_class(y_true: np.ndarray, proba: np.ndarray, metric_fn) -> np.ndarray:
    """
    Searches independent thresholds per class to maximize 'metric_fn' (e.g., macro-F1).
    Sweeps a grid of thresholds from 0.2 to 0.9 in ~0.02 increments.
    """
    K = proba.shape[1]
    thr = np.full(K, 1/3, dtype=float)
    for k in range(K):
        best_s, best_t = -1.0, thr[k]
        for t in np.linspace(0.2, 0.9, 36):
            y_pred = proba.argmax(1)
            mask = proba[:, k] >= t
            y_pred[mask] = k
            s = metric_fn(y_true, y_pred)
            if s > best_s:
                best_s, best_t = s, t
        thr[k] = best_t
    return thr


def coordinate_ascent_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    thr0: np.ndarray,
    metric_fn,
    rounds: int = 2
) -> tuple[np.ndarray, float]:
    """
    Iterative refinement (coordinate ascent) of class-wise thresholds.

    Starts from an initial threshold vector thr0 and,
    during 'rounds' iterations, adjusts each class to
    maximize the global metric (macro-F1 or other).
    """
    thr = np.array(thr0, dtype=float).copy()
    K = proba.shape[1]
    best = metric_fn(y_true, apply_thresholds(proba, thr))
    for _ in range(rounds):
        for k in range(K):
            cur_best, cur_t = best, thr[k]
            lo = max(0.20, thr[k] - 0.15)
            hi = min(0.95, thr[k] + 0.25)
            for t in np.linspace(lo, hi, 21):
                thr_try = thr.copy()
                thr_try[k] = t
                y_pred_try = apply_thresholds(proba, thr_try)
                s = metric_fn(y_true, y_pred_try)
                if s > cur_best:
                    cur_best, cur_t = s, t
            thr[k] = cur_t
            best = cur_best
    return thr, best
