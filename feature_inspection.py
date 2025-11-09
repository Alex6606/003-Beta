# ============================================
# feature_inspection.py
# ============================================
"""
Funciones auxiliares para inspeccionar y auditar conjuntos de features.
Permiten entender la composición por tipo, familia e indicadores.
"""

from collections import Counter, defaultdict
import pandas as pd


def summarize_features(features_df: pd.DataFrame, feature_types: dict):
    """
    Imprime un resumen rápido de las features generadas:
      - Conteo total
      - Distribución por familia (Momentum, Volatilidad, Volumen)
      - Distribución por tipo de normalización
      - Ejemplo de columnas por familia

    Parámetros
    ----------
    features_df : pd.DataFrame
        DataFrame con las features generadas.
    feature_types : dict
        Diccionario que mapea cada columna con su tipo ('asset_scale', 'bounded_0_1', etc.)
    """
    def family_from_name(col: str) -> str:
        """Identifica la familia de indicadores según el prefijo del nombre."""
        if col.startswith(("ROC_", "RSI_", "STOCHK_", "STOCHD_", "WPR100_", "CCI_", "TRIX_", "DIST_SMA_", "DIST_EMA_", "MACD_")):
            return "Momentum"
        if col.startswith(("LOGSTD_", "ATR_", "PCTB_", "BBWIDTH_", "KELT_BW_", "PLUS_DI_", "MINUS_DI_", "ADX_")):
            return "Volatilidad"
        if col.startswith(("OBV", "VROC_", "MFI_", "CMF_")):
            return "Volumen"
        return "Otro"

    # === Conteos por familia y tipo ===
    fam_counts = Counter(family_from_name(c) for c in features_df.columns)
    type_counts = Counter(feature_types.get(c, "asset_scale") for c in features_df.columns)

    print(f"Total de features: {features_df.shape[1]}")
    print("Distribución por familia:", dict(fam_counts))
    print("Distribución por tipo:", dict(type_counts))

    # === Muestras representativas ===
    buckets = defaultdict(list)
    for c in features_df.columns:
        buckets[family_from_name(c)].append(c)

    for fam, cols in buckets.items():
        print(f"\n{fam} (muestra): {cols[:8]}")
