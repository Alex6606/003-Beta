# ============================================
# prior_correction.py
# ============================================
"""
Módulo para corrección de cambio de distribución de clases (prior shift)
y ajuste de sesgos (biases) en los logits.
Incluye:
- project_simplex
- bbse_prior_shift_soft
- search_logit_biases
"""

import numpy as np
from threshold_tuning import apply_thresholds
from calibration import softmax_T  # asumiendo que softmax_T está en calibration.py
from metrics import macro_f1       # asumimos que macro_f1 se define ahí


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Proyección al simplex {x >= 0, sum(x) = 1}.
    Asegura que el vector sea una distribución válida de probabilidades.
    """
    v = np.asarray(v, float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def bbse_prior_shift_soft(pi_train: np.ndarray,
                          y_true_val: np.ndarray,
                          proba_val: np.ndarray,
                          proba_test: np.ndarray,
                          lam_ridge: float = 1e-2,
                          eps: float = 1e-6):
    """
    Estima la distribución de clases en test (pi_test) usando BBSE suave:
      min ||C^T pi - q||² + lam * ||pi - pi_train||²

    Donde:
      - C[i,j] = E[ P(ŷ=j | y=i) ] (promedio de probabilidades predichas por clase)
      - q[j]   = E[ P(ŷ=j) ] sobre test
    """
    K = proba_val.shape[1]
    C = np.zeros((K, K), dtype=float)
    for i in range(K):
        mask = (y_true_val == i)
        if mask.any():
            C[i] = proba_val[mask].mean(axis=0)
        else:
            C[i] = np.full(K, 1.0 / K)

    q = proba_test.mean(axis=0)

    # Resuelve (CᵀC + λI)π = Cᵀq + λπ_train
    A = C.T @ C + lam_ridge * np.eye(K)
    b = C.T @ q + lam_ridge * pi_train
    pi_hat = np.linalg.solve(A, b)

    # Proyección al simplex
    pi_hat = np.clip(pi_hat, eps, 1.0)
    pi_hat = project_simplex(pi_hat)
    return pi_hat, C


def search_logit_biases(logits_val: np.ndarray,
                        y_val: np.ndarray,
                        thr_refined: np.ndarray,
                        T: float = 1.0,
                        grid_d0=np.linspace(-0.2, 0.2, 9),
                        grid_d1=np.linspace(0.0, 0.8, 9),
                        grid_d2=np.linspace(-0.2, 0.2, 9)):
    """
    Búsqueda exhaustiva de sesgos (deltas) por clase que maximizan macro-F1.
    Permite ajustar los logits post-calibración (bias tilting).
    """
    best = (-1.0, (0.0, 0.0, 0.0))
    for d0 in grid_d0:
        for d1 in grid_d1:
            for d2 in grid_d2:
                L = logits_val + np.array([d0, d1, d2])[None, :]
                P = softmax_T(L, T)
                y_pred = apply_thresholds(P, thr_refined)
                s = macro_f1(y_val, y_pred)
                if s > best[0]:
                    best = (s, (d0, d1, d2))
    return best
