# ============================================================
# drift_dashboard.py — Data Drift Modeling Dashboard (Streamlit)
# ============================================================
"""
Interactive dashboard to analyze *data drift* between
training, validation, and test periods.

Includes:
 - Distribution comparison via KS-test
 - Timeline of mean/standard deviation evolution
 - Summary table with p-values and drift detection
 - Top-5 most drifted features with interpretation

To run:
    streamlit run drift_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 1️⃣ — FEATURE LOADING
# ============================================================

@st.cache_data
def load_features():
    """Loads normalized features for train/val/test from exported CSV files."""
    base_path = "./data/features/"
    train_path = os.path.join(base_path, "feat_train_n.csv")
    val_path   = os.path.join(base_path, "feat_val_n.csv")
    test_path  = os.path.join(base_path, "feat_test_n.csv")

    if not os.path.exists(train_path):
        st.error(
            "❌ Feature CSV files not found. "
            "Run build_and_normalize_features_per_split() to export them."
        )
        st.stop()

    feat_train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    feat_val   = pd.read_csv(val_path, index_col=0, parse_dates=True)
    feat_test  = pd.read_csv(test_path, index_col=0, parse_dates=True)

    return feat_train, feat_val, feat_test


# ============================================================
# 2️⃣ — KS-TEST
# ============================================================

def compute_drift_table(train_df, test_df, val_df):
    """Applies KS-test between train–test and train–val for each feature."""
    results = []
    for col in train_df.columns:
        tr = train_df[col].dropna()
        te = test_df[col].dropna()
        va = val_df[col].dropna()

        # KS-test
        p_test = ks_2samp(tr, te).pvalue if len(te) > 0 else np.nan
        p_val  = ks_2samp(tr, va).pvalue if len(va) > 0 else np.nan

        results.append({
            "Feature": col,
            "p_value_test": p_test,
            "Drift_Test": p_test < 0.05 if not np.isnan(p_test) else False,
            "p_value_val": p_val,
            "Drift_Val": p_val < 0.05 if not np.isnan(p_val) else False,
            "Mean_Train": np.mean(tr),
            "Mean_Test": np.mean(te),
            "Mean_Val": np.mean(va),
            "Std_Train": np.std(tr),
            "Std_Test": np.std(te),
            "Std_Val": np.std(va),
        })

    df = pd.DataFrame(results)
    df["Drift_Global"] = df["Drift_Test"] | df["Drift_Val"]
    return df.sort_values("p_value_test", ascending=True)


# ============================================================
# 3️⃣ — TIMELINE VISUALIZATION
# ============================================================

def plot_feature_timeline(train_df, val_df, test_df, feature):
    """Plots time evolution of the selected feature across splits."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_df.index, train_df[feature], label="Train", color="tab:blue", alpha=0.7)
    plt.plot(val_df.index, val_df[feature], label="Val", color="tab:orange", alpha=0.7)
    plt.plot(test_df.index, test_df[feature], label="Test", color="tab:green", alpha=0.7)
    plt.title(f"Temporal Evolution of '{feature}'")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def plot_feature_distributions(train_df, val_df, test_df, feature):
    """Plots kernel density distributions for train/val/test."""
    plt.figure(figsize=(8, 4))
    sns.kdeplot(train_df[feature], label="Train", color="tab:blue", fill=True)
    sns.kdeplot(val_df[feature], label="Val", color="tab:orange", fill=True)
    sns.kdeplot(test_df[feature], label="Test", color="tab:green", fill=True)
    plt.title(f"Distributions of '{feature}'")
    plt.xlabel("Normalized Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


# ============================================================
# 4️⃣ — STREAMLIT MAIN INTERFACE
# ============================================================

def main():
    st.set_page_config(page_title="Data Drift Dashboard", layout="wide")
    st.title("📊 Data Drift Modeling Dashboard")

    st.markdown(
        "This dashboard analyzes **data drift** between the *train*, *validation*, "
        "and *test* datasets using the **KS-test**."
    )

    # --- Load features ---
    feat_train, feat_val, feat_test = load_features()

    # --- Compute KS-test ---
    st.subheader("🔍 Drift Computation")
    drift_df = compute_drift_table(feat_train, feat_test, feat_val)

    st.dataframe(
        drift_df[["Feature", "p_value_test", "Drift_Test", "p_value_val", "Drift_Val", "Drift_Global"]]
        .style.background_gradient(subset=["p_value_test", "p_value_val"], cmap="Reds_r")
        .applymap(lambda x: "background-color: #fdd" if x is True else "", subset=["Drift_Global"])
    )

    # --- Top 5 drifted features ---
    st.subheader("🔥 Top 5 Most Drifted Features (Lowest KS p-values)")
    top5 = drift_df.head(5)
    st.table(top5[["Feature", "p_value_test", "p_value_val", "Drift_Global"]])

    st.markdown(
        "**Interpretation:** These features show significant changes in their "
        "distribution across periods. They may indicate **market regime changes**, "
        "**volatility shifts**, or **structural variations** in the time series."
    )

    # --- Feature selection for visualization ---
    st.subheader("📈 Feature Analysis")
    selected_feature = st.selectbox("Select a feature to visualize:", feat_train.columns)

    col1, col2 = st.columns(2)
    with col1:
        plot_feature_distributions(feat_train, feat_val, feat_test, selected_feature)
    with col2:
        plot_feature_timeline(feat_train, feat_val, feat_test, selected_feature)


# ============================================================
# 5️⃣ — ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
