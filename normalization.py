# ============================================
# normalization.py
# ============================================
"""
Feature normalization module based on feature type.
Allows handling heterogeneous scales (price, volume, bounded indicators, etc.)
before feeding them to neural networks or ML models.
"""

import numpy as np
import pandas as pd


# ===================== Internal helpers =====================
def _robust_stats(series: pd.Series):
    """
    Computes robust median and IQR, with fallback if IQR is zero or NaN.
    """
    med = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or np.isnan(iqr):
        alt = series.mad() if hasattr(series, "mad") else None
        iqr = alt if (alt and alt != 0) else (series.std(ddof=0) or 1.0)

    return float(med), float(iqr)


# ===================== Main normalization function =====================
def fit_apply_normalizer(
    features: pd.DataFrame,
    feature_types: dict,
    ref_df: pd.DataFrame | None = None
):
    """
    Fits and applies normalization based on feature type.

    Supported types:
      - 'asset_scale'    → robust z-score using (median, IQR)
      - 'bounded_0_100'  → divided by 100
      - 'bounded_0_1'    → unchanged
      - 'bounded_-1_1'   → unchanged

    Parameters
    ----------
    features : DataFrame
        Feature set to normalize.
    feature_types : dict
        Mapping {column: type} produced by make_features_auto().
    ref_df : DataFrame, optional
        Reference DataFrame for computing normalization stats (typically TRAIN).

    Returns
    -------
    (features_norm, norm_stats)
        features_norm : normalized DataFrame
        norm_stats : dict with per-column normalization statistics
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
            stats[col] = {"type": t}  # no transformation

        else:
            # conservative fallback
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
    Applies normalization to a new dataset (e.g., TEST or VAL)
    using the statistics computed via fit_apply_normalizer().
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

        # bounded_0_1 and bounded_-1_1 → no transformation

    return norm.dropna(how="any").copy()
