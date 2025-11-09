# ============================================
# calibration.py
# ============================================
"""
Funciones para calibración de temperatura (Temperature Scaling)
de los logits del modelo, usadas para mejorar la calibración
de probabilidades sin alterar la precisión base.
"""

import numpy as np


def softmax_T(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    Aplica softmax con temperatura T > 0.
    T > 1 → distribuciones más suaves (menos confianza)
    T < 1 → distribuciones más agudas (más confianza)
    """
    z = logits / T
    z -= z.max(axis=1, keepdims=True)  # estabilidad numérica
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def nll_from_logits_T(logits: np.ndarray, y_true: np.ndarray, T: float) -> float:
    """
    Calcula la Negative Log-Likelihood (NLL) para logits calibrados con temperatura T.
    """
    z = logits / T
    z -= z.max(axis=1, keepdims=True)  # evita overflow
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    # log_probs[np.arange(len(y_true)), y_true] = log(probabilidad de clase verdadera)
    return -log_probs[np.arange(len(y_true)), y_true].mean()


def find_temperature(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    Ts: np.ndarray = np.linspace(0.8, 3.0, 23)
) -> float:
    """
    Busca la mejor temperatura (T óptima) que minimiza el NLL en un conjunto de validación.
    Devuelve el mejor valor de T encontrado.
    """
    bestT, bestNLL = 1.0, 1e9
    for T in Ts:
        nll = nll_from_logits_T(logits_val, y_val, T)
        if nll < bestNLL:
            bestNLL, bestT = nll, T
    return bestT
