# ============================================================
# main.py — Deep Learning Trading with CNN + Backtesting
# ============================================================

from data_utils import get_data
from features_pipeline import build_and_normalize_features_per_split
from indicators import WINDOWS
from labeling import build_labels_for_feature_splits
from training_pipeline import train_eval_from_raw, summarize_run, sanity_check_from_res
from visualization import plot_equity_curves, plot_confusion_matrix, plot_f1_history

# === CONFIGURACIÓN CENTRAL ===
CONFIG = {
    "seq_window": 90,        # longitud de la ventana temporal (B)
    "seq_step": 2,           # salto entre ventanas
    "dropout": 0.4,          # regularización moderada
    "l2": 1e-3,              # penalización L2
    "threshold": 0.015,      # sensibilidad del label
    "gamma": 1.5,            # parámetro Focal Loss
    "shrink_lambda": 0.877,  # mezcla thresholds base / tilt
}

# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    print("=== Descargando datos ===")
    data = get_data("MSFT")  # incluye columna 'Close'

    print("\n=== Construyendo features y normalizando ===")
    bundle = build_and_normalize_features_per_split(data, windows=WINDOWS, warmup=200)
    feat_train_n = bundle["norm"]["train"]
    feat_test_n  = bundle["norm"]["test"]
    feat_val_n   = bundle["norm"]["val"]

    print("\n=== Construyendo etiquetas ===")
    labels_bundle = build_labels_for_feature_splits(
        data_ohlcv=data,
        feat_train_n=feat_train_n,
        feat_test_n=feat_test_n,
        feat_val_n=feat_val_n,
        price_col="Close",
        horizon=3,
        threshold=CONFIG["threshold"],
    )

    X_train, y_train = labels_bundle["X"]["train"], labels_bundle["y"]["train"]
    X_test,  y_test  = labels_bundle["X"]["test"],  labels_bundle["y"]["test"]
    X_val,   y_val   = labels_bundle["X"]["val"],   labels_bundle["y"]["val"]

    # === Índices y precios para backtesting ===
    idx_train = feat_train_n.index
    idx_test  = feat_test_n.index
    idx_val   = feat_val_n.index
    close_series = data["Close"]

    print("\n=== Entrenando modelo CNN-1D ===")
    res = train_eval_from_raw(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        cw_train_cb={0: 1.83, 1: 0.968, 2: 1.012},
        gamma=CONFIG["gamma"],
        seq_window=CONFIG["seq_window"],
        seq_step=CONFIG["seq_step"],
        shrink_lambda=CONFIG["shrink_lambda"],
        verbose=1,
        # parámetros nuevos para backtesting
        train_idx=idx_train,
        val_idx=idx_val,
        test_idx=idx_test,
        close_series=close_series,
    )

    print("\n=== Resultados ===")
    summarize_run(res)
    sanity_check_from_res(res)

    # === Métricas de Backtest ===
    if "backtest" in res:
        print("\n=== Métricas de Backtest ===")
        for split, bt in res["backtest"].items():
            print(f"[{split.upper()}] Return={bt['final_return']:.3%}, Sharpe={bt['sharpe']:.2f}")

    # === Visualizaciones ===
    print("\n=== Visualizaciones ===")
    if "backtest" in res:
        plot_equity_curves(res["backtest"])

    if "y_true_pred" in res and "test" in res["y_true_pred"]:
        y_true, y_pred = res["y_true_pred"]["test"]
        plot_confusion_matrix(y_true, y_pred, classes=(0, 1, 2))

    if "history" in res:
        plot_f1_history(res["history"])
