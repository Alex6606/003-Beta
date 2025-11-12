# ============================================================
# drift_dashboard.py — Data Drift Modeling Dashboard (Streamlit)
# ============================================================
"""
Dashboard interactivo para analizar el *data drift* entre los periodos
de entrenamiento, validación y prueba.

Incluye:
 - Comparación de distribuciones (KS-test)
 - Timeline de evolución de medias/desviaciones
 - Tabla resumen con p-values y detección de drift
 - Top-5 features más afectados con interpretación

Para ejecutar:
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
# 1️⃣ — CARGA DE DATOS
# ============================================================

@st.cache_data
def load_features():
    """Carga los features normalizados de train/val/test desde archivos CSV."""
    base_path = "./data/features/"
    train_path = os.path.join(base_path, "feat_train_n.csv")
    val_path   = os.path.join(base_path, "feat_val_n.csv")
    test_path  = os.path.join(base_path, "feat_test_n.csv")

    if not os.path.exists(train_path):
        st.error("❌ No se encontraron los archivos CSV de features. "
                 "Ejecuta build_and_normalize_features_per_split() y exporta los DataFrames.")
        st.stop()

    feat_train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    feat_val   = pd.read_csv(val_path, index_col=0, parse_dates=True)
    feat_test  = pd.read_csv(test_path, index_col=0, parse_dates=True)

    return feat_train, feat_val, feat_test


# ============================================================
# 2️⃣ — FUNCIÓN DE KS-TEST
# ============================================================

def compute_drift_table(train_df, test_df, val_df):
    """Aplica KS-test entre train–test y train–val para cada feature."""
    results = []
    for col in train_df.columns:
        tr = train_df[col].dropna()
        te = test_df[col].dropna()
        va = val_df[col].dropna()

        # KS-test comparando distribuciones
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
# 3️⃣ — VISUALIZACIÓN DEL TIMELINE
# ============================================================

def plot_feature_timeline(train_df, val_df, test_df, feature):
    """Grafica la evolución temporal del feature seleccionado."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_df.index, train_df[feature], label="Train", color="tab:blue", alpha=0.7)
    plt.plot(val_df.index, val_df[feature], label="Val", color="tab:orange", alpha=0.7)
    plt.plot(test_df.index, test_df[feature], label="Test", color="tab:green", alpha=0.7)
    plt.title(f"Evolución temporal de '{feature}'")
    plt.xlabel("Fecha")
    plt.ylabel("Valor normalizado")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def plot_feature_distributions(train_df, val_df, test_df, feature):
    """Grafica las distribuciones por split."""
    plt.figure(figsize=(8, 4))
    sns.kdeplot(train_df[feature], label="Train", color="tab:blue", fill=True)
    sns.kdeplot(val_df[feature], label="Val", color="tab:orange", fill=True)
    sns.kdeplot(test_df[feature], label="Test", color="tab:green", fill=True)
    plt.title(f"Distribuciones de '{feature}'")
    plt.xlabel("Valor normalizado")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


# ============================================================
# 4️⃣ — INTERFAZ PRINCIPAL STREAMLIT
# ============================================================

def main():
    st.set_page_config(page_title="Data Drift Dashboard", layout="wide")
    st.title("📊 Data Drift Modeling Dashboard")
    st.markdown(
        "Este panel permite analizar el **data drift** entre los conjuntos "
        "de *train*, *validation* y *test* usando el **KS-test**."
    )

    # --- Cargar datos ---
    feat_train, feat_val, feat_test = load_features()

    # --- Calcular KS-test ---
    st.subheader("🔍 Cálculo de Drift")
    drift_df = compute_drift_table(feat_train, feat_test, feat_val)

    st.dataframe(
        drift_df[["Feature", "p_value_test", "Drift_Test", "p_value_val", "Drift_Val", "Drift_Global"]]
        .style.background_gradient(subset=["p_value_test", "p_value_val"], cmap="Reds_r")
        .applymap(lambda x: "background-color: #fdd" if x is True else "", subset=["Drift_Global"])
    )

    # --- Top 5 más afectados ---
    st.subheader("🔥 Top 5 Features con mayor Drift (KS más bajo)")
    top5 = drift_df.head(5)
    st.table(top5[["Feature", "p_value_test", "p_value_val", "Drift_Global"]])

    st.markdown("**Interpretación:** Estos *features* muestran cambios significativos en su distribución "
                "entre periodos. Pueden indicar **cambios de régimen de mercado**, "
                "**shifts de volatilidad**, o **alteraciones estructurales** en la serie temporal.")

    # --- Seleccionar feature para analizar ---
    st.subheader("📈 Análisis de un Feature")
    selected_feature = st.selectbox("Selecciona un feature para visualizar:", feat_train.columns)

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
