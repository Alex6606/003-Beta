# ============================================
# feature_pipeline.py
# ============================================
"""
Orchestrator module for feature generation and normalization.
Integrates:
  - Chronological 60/20/20 split (no leakage)
  - Technical indicator generation (make_features_auto)
  - Normalization based on TRAIN (robust per type)
"""

import pandas as pd
from data_split import split_60_20_20
from features_auto import make_features_auto
from normalization import fit_apply_normalizer, apply_normalizer_from_stats


def build_and_normalize_features_per_split(
    data: pd.DataFrame,
    windows,
    warmup: int = 0  # e.g., 200 if using large windows; 0 = no warmup
):
    """
    Pipeline:
    ---------
    1) Chronological 60/20/20 split on 'data'.
    2) make_features_auto for each split (optional warmup for TEST/VAL).
    3) Align columns to TRAIN set.
    4) Normalize: fit on TRAIN and apply to TEST/VAL using identical stats (no leakage).

    Returns:
    --------
    A dictionary:
        {
          "raw":  {"train": feat_train, "test": feat_test, "val": feat_val},
          "norm": {"train": feat_train_norm, "test": feat_test_norm, "val": feat_val_norm},
          "feature_types": ftypes,
          "norm_stats": norm_stats
        }
    """

    # === 1) Temporal split ===
    train_df, test_df, val_df = split_60_20_20(
        data, cols_required=["Close", "High", "Low", "Volume"], verbose=True
    )

    # === 2) Optional warmup for test/val ===
    def _with_warmup(base_df, hist_df, warmup):
        """Adds historical context to compute indicators without leakage."""
        if warmup and len(hist_df) > 0:
            pad = hist_df.iloc[-min(len(hist_df), warmup):]
            return pd.concat([pad, base_df], axis=0)
        return base_df

    train_full = train_df
    test_full = _with_warmup(test_df, train_df, warmup)
    val_full = _with_warmup(val_df, pd.concat([train_df, test_df], axis=0), warmup)

    # === 3) Generate features ===
    feat_train, ftypes, _ = make_features_auto(train_full, windows=windows)
    feat_test, _, _ = make_features_auto(test_full, windows=windows)
    feat_val, _, _ = make_features_auto(val_full, windows=windows)

    # Trim warmup → keep only actual split dates
    feat_test = feat_test.loc[test_df.index.intersection(feat_test.index)]
    feat_val = feat_val.loc[val_df.index.intersection(feat_val.index)]

    # === 4) Align columns to TRAIN ===
    cols = feat_train.columns.tolist()
    feat_test = feat_test.reindex(columns=cols).dropna(how="any")
    feat_val = feat_val.reindex(columns=cols).dropna(how="any")

    # === 5) No-leakage normalization ===
    feat_train_norm, norm_stats = fit_apply_normalizer(
        feat_train, ftypes, ref_df=feat_train
    )
    feat_test_norm = apply_normalizer_from_stats(feat_test, ftypes, norm_stats)
    feat_val_norm = apply_normalizer_from_stats(feat_val, ftypes, norm_stats)

    return {
        "raw": {
            "train": feat_train,
            "test": feat_test,
            "val": feat_val,
        },
        "norm": {
            "train": feat_train_norm,
            "test": feat_test_norm,
            "val": feat_val_norm,
        },
        "feature_types": ftypes,
        "norm_stats": norm_stats,
    }


def export_features_for_drift(bundle):
    import os
    os.makedirs("./data/features", exist_ok=True)
    bundle["norm"]["train"].to_csv("./data/features/feat_train_n.csv")
    bundle["norm"]["val"].to_csv("./data/features/feat_val_n.csv")
    bundle["norm"]["test"].to_csv("./data/features/feat_test_n.csv")
    print("✅ Normalized features exported to ./data/features/ (feat_*_n.csv)")

