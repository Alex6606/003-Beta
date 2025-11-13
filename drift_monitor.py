# ============================================================
# drift_monitor.py — KS-test computation and CSV export
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def ks_table(train_df: pd.DataFrame, other_df: pd.DataFrame):
    """
    Computes a feature-level KS-test between two normalized feature sets.

    Parameters
    ----------
    train_df : pd.DataFrame
        Reference dataset (typically train split).
    other_df : pd.DataFrame
        Comparison dataset (val or test split).

    Returns
    -------
    pd.DataFrame
        Sorted table with KS statistic, p-value and drift flag.
    """
    rows = []
    for col in train_df.columns:
        # Skip non-numeric features
        if not np.issubdtype(train_df[col].dtype, np.number):
            continue

        a = train_df[col].dropna().values
        b = other_df[col].dropna().values

        # Require sufficient sample size
        if len(a) < 30 or len(b) < 30:
            continue

        stat, p = ks_2samp(a, b, alternative="two-sided", mode="auto")
        rows.append({
            "feature": col,
            "ks_stat": float(stat),
            "p_value": float(p),
            "drift": bool(p < 0.05)
        })

    return pd.DataFrame(rows).sort_values("p_value")


# Save 3 drift tables: train vs val, train vs test, val vs test
def build_drift_csvs(train_df, val_df, test_df, out_dir="drift_reports"):
    """
    Generates three CSV reports with KS-test drift statistics:
        - train vs val
        - train vs test
        - val vs test

    Parameters
    ----------
    train_df : pd.DataFrame
    val_df   : pd.DataFrame
    test_df  : pd.DataFrame
    out_dir  : str
        Output directory to store CSV files.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    ks_table(train_df, val_df).to_csv(f"{out_dir}/ks_train_vs_val.csv", index=False)
    ks_table(train_df, test_df).to_csv(f"{out_dir}/ks_train_vs_test.csv", index=False)
    ks_table(val_df, test_df).to_csv(f"{out_dir}/ks_val_vs_test.csv", index=False)
