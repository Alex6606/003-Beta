# ============================================================
# drift_monitor.py — cálculo de KS test y CSV
# ============================================================
import pandas as pd
import numpy as np
from scipy.stats import ks_2sample

def ks_table(train_df: pd.DataFrame, other_df: pd.DataFrame):
    rows = []
    for col in train_df.columns:
        if not np.issubdtype(train_df[col].dtype, np.number):
            continue
        a = train_df[col].dropna().values
        b = other_df[col].dropna().values
        if len(a) < 30 or len(b) < 30:
            continue
        stat, p = ks_2sample(a, b, alternative="two-sided", mode="auto")
        rows.append({"feature": col, "ks_stat": float(stat), "p_value": float(p), "drift": bool(p < 0.05)})
    return pd.DataFrame(rows).sort_values("p_value")

# Guarda 3 tablas: train vs val, train vs test, val vs test
def build_drift_csvs(train_df, val_df, test_df, out_dir="drift_reports"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    ks_table(train_df, val_df).to_csv(f"{out_dir}/ks_train_vs_val.csv", index=False)
    ks_table(train_df, test_df).to_csv(f"{out_dir}/ks_train_vs_test.csv", index=False)
    ks_table(val_df, test_df).to_csv(f"{out_dir}/ks_val_vs_test.csv", index=False)
