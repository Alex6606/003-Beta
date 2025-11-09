# ============================================
# threshold_tuning.py
# ============================================
"""
Funciones para ajuste y aplicación de umbrales por clase
después de calibrar las probabilidades del modelo.
Incluye:
- apply_thresholds
- tune_thresholds_by_class
- coordinate_ascent_thresholds
"""

import numpy as np


def apply_thresholds(proba: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """
    Regla: si alguna clase supera su umbral, elige la de mayor (proba_k - thr_k);
    si ninguna supera, usa argmax.
    """
    proba = np.asarray(proba)
    thr = np.asarray(thr)
    y_argmax = proba.argmax(axis=1)

    # matriz "margen" (proba - thr); <0 significa "no superó umbral"
    margins = proba - thr.reshape(1, -1)
    best_cls = margins.argmax(axis=1)
    best_margin = margins[np.arange(len(proba)), best_cls]

    y_pred = y_argmax.copy()
    mask = best_margin >= 0.0
    y_pred[mask] = best_cls[mask]
    return y_pred


def tune_thresholds_by_class(y_true: np.ndarray, proba: np.ndarray, metric_fn) -> np.ndarray:
    """
    Busca umbrales independientes por clase maximizando 'metric_fn' (p. ej. macro-F1).
    Recorre una cuadrícula de thresholds de 0.2 a 0.9 en pasos de ~0.02.
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
    Refinamiento iterativo (coordinate ascent) de umbrales por clase.

    Parte de un vector inicial thr0 y, durante 'rounds' iteraciones,
    ajusta cada clase para maximizar la métrica global (macro-F1 u otra).
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
