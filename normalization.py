# ============================================
# normalization.py
# ============================================
"""
Módulo de normalización de features según su tipo.
Permite ajustar escalas heterogéneas (precio, volumen, indicadores bounded, etc.)
para alimentar modelos neuronales o ML.
"""

import numpy as np
import pandas as pd


# ===================== Funciones internas =====================
def _robust_stats(series: pd.Series):
    """Calcula mediana e IQR robustos, con fallback si el IQR es 0 o NaN."""
    med = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or np.isnan(iqr):
        alt = series.mad() if hasattr(series, "mad") else None
        iqr = alt if (alt and alt != 0) else (series.std(ddof=0) or 1.0)

    return float(med), float(iqr)


# ===================== Función principal =====================
def fit_apply_normalizer(
    features: pd.DataFrame,
    feature_types: dict,
    ref_df: pd.DataFrame | None = None
):
    """
    Ajusta y aplica la normalización por tipo de variable.

    Tipos soportados:
      - 'asset_scale'    → robust z-score con (mediana, IQR)
      - 'bounded_0_100'  → divide entre 100
      - 'bounded_0_1'    → no cambia
      - 'bounded_-1_1'   → no cambia

    Parámetros
    ----------
    features : DataFrame
        Conjunto de features a normalizar.
    feature_types : dict
        Mapeo {columna: tipo} según make_features_auto().
    ref_df : DataFrame, opcional
        DataFrame de referencia para calcular estadísticas (por ejemplo, TRAIN).

    Retorna
    -------
    (features_norm, norm_stats)
        features_norm : DataFrame normalizado
        norm_stats : dict con estadísticas por columna
    """
    if ref_df is None:
        ref_df = features

    norm = features.copy()
    stats = {}

    for col in features.columns:
        t = feature_types.get(col, "asset_scale")

        if t == "asset_scale":
            med, iqr = _robust_stats(ref_df[col].dropna())
            stats[col] = {"type": t, "median": med, "iqr": iqr}
            denom = iqr if iqr != 0 else 1.0
            norm[col] = (norm[col] - med) / denom

        elif t == "bounded_0_100":
            stats[col] = {"type": t, "scale": 100.0}
            norm[col] = norm[col] / 100.0

        elif t in ("bounded_0_1", "bounded_-1_1"):
            stats[col] = {"type": t}  # no transformación

        else:
            # fallback prudente
            med, iqr = _robust_stats(ref_df[col].dropna())
            stats[col] = {"type": "asset_scale(default)", "median": med, "iqr": iqr}
            denom = iqr if iqr != 0 else 1.0
            norm[col] = (norm[col] - med) / denom

    norm = norm.dropna(how="any").copy()
    return norm, stats


def apply_normalizer_from_stats(
    features: pd.DataFrame,
    feature_types: dict,
    norm_stats: dict
) -> pd.DataFrame:
    """
    Aplica normalización a un nuevo conjunto (ej. TEST o VAL)
    usando las estadísticas calculadas con fit_apply_normalizer().
    """
    norm = features.copy()
    for col in features.columns:
        t = feature_types.get(col, "asset_scale")

        if t == "asset_scale":
            st = norm_stats[col]
            med = st.get("median", 0.0)
            iqr = st.get("iqr", 1.0)
            denom = iqr if iqr != 0 else 1.0
            norm[col] = (norm[col] - med) / denom

        elif t == "bounded_0_100":
            norm[col] = norm[col] / 100.0

        # bounded_0_1 y bounded_-1_1 → no transformación

    return norm.dropna(how="any").copy()
