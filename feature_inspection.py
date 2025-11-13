# ============================================
# feature_inspection.py
# ============================================
"""
Utility functions for inspecting and auditing feature sets.
Allows understanding composition by family, type, and indicator category.
"""

from collections import Counter, defaultdict
import pandas as pd


def summarize_features(features_df: pd.DataFrame, feature_types: dict):
    """
    Prints a quick summary of the generated features:
      - Total feature count
      - Distribution by family (Momentum, Volatility, Volume, Other)
      - Distribution by normalization type
      - Sample columns per family

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing the generated features.
    feature_types : dict
        Dictionary mapping each feature column to its type
        ('asset_scale', 'bounded_0_1', etc.).
    """
    def family_from_name(col: str) -> str:
        """Identifies the indicator family based on feature name prefix."""
        if col.startswith((
            "ROC_", "RSI_", "STOCHK_", "STOCHD_", "WPR100_", "CCI_", "TRIX_",
            "DIST_SMA_", "DIST_EMA_", "MACD_"
        )):
            return "Momentum"

        if col.startswith((
            "LOGSTD_", "ATR_", "PCTB_", "BBWIDTH_", "KELT_BW_",
            "PLUS_DI_", "MINUS_DI_", "ADX_"
        )):
            return "Volatility"

        if col.startswith(("OBV", "VROC_", "MFI_", "CMF_")):
            return "Volume"

        return "Other"

    # === Counts by family and type ===
    fam_counts = Counter(family_from_name(c) for c in features_df.columns)
    type_counts = Counter(feature_types.get(c, "asset_scale") for c in features_df.columns)

    print(f"Total features: {features_df.shape[1]}")
    print("Family distribution:", dict(fam_counts))
    print("Normalization type distribution:", dict(type_counts))

    # === Representative samples ===
    buckets = defaultdict(list)
    for c in features_df.columns:
        buckets[family_from_name(c)].append(c)

    for fam, cols in buckets.items():
        print(f"\n{fam} (sample): {cols[:8]}")
