# ============================================================
# streamlit_app.py — Drift dashboard
# ============================================================
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Drift Monitor", layout="wide")

st.title("Data Drift Monitoring (KS test)")
folder = "drift_reports"

for name in ["ks_train_vs_val.csv", "ks_train_vs_test.csv", "ks_val_vs_test.csv"]:
    path = f"{folder}/{name}"
    st.subheader(name)
    try:
        df = pd.read_csv(path)
        st.dataframe(df)
        top = df.nsmallest(5, "p_value")
        st.write("Top 5 most-drifted:")
        st.dataframe(top)
    except Exception as e:
        st.info(f"No file found: {path} ({e})")
