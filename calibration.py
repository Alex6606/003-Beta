# ============================================
# calibration.py
# ============================================
"""
Functions for temperature scaling calibration of model logits.
Temperature Scaling improves probability calibration without
modifying the underlying model accuracy.
"""

import numpy as np


def softmax_T(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    Applies softmax with temperature T > 0.
    T > 1 → smoother distributions (less confident)
    T < 1 → sharper distributions (more confident)
    """
    z = logits / T
    z -= z.max(axis=1, keepdims=True)  # numerical stability
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def nll_from_logits_T(logits: np.ndarray, y_true: np.ndarray, T: float) -> float:
    """
    Computes the Negative Log-Likelihood (NLL) for logits
    calibrated using temperature T.
    """
    z = logits / T
    z -= z.max(axis=1, keepdims=True)  # avoid overflow
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    # log_probs[np.arange(len(y_true)), y_true] contains log(prob of true class)
    return -log_probs[np.arange(len(y_true)), y_true].mean()


def find_temperature(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    Ts: np.ndarray = np.linspace(0.8, 3.0, 23)
) -> float:
    """
    Searches for the optimal temperature T that minimizes the NLL
    on the validation set. Returns the best T value found.
    """
    bestT, bestNLL = 1.0, 1e9
    for T in Ts:
        nll = nll_from_logits_T(logits_val, y_val, T)
        if nll < bestNLL:
            bestNLL, bestT = nll, T
    return bestT
