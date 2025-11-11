# ============================================================
# main.py — Deep Learning Trading with CNN + Backtesting Avanzado
# ============================================================

from data_utils import get_data
from features_pipeline import build_and_normalize_features_per_split
from indicators import WINDOWS
from labeling import build_labels_for_feature_splits
from training_pipeline import train_eval_from_raw, summarize_run, sanity_check_from_res
from backtest import backtest_advanced, y_to_signal
from visualization import plot_equity_curves, plot_confusion_matrix, plot_f1_history

# === CONFIGURACIÓN CENTRAL ===
CONFIG = {
    "seq_window": 60,    # longitud de la ventana temporal
    "seq_step": 1,       # salto entre ventanas
    "dropout": 0.3,      # regularización moderada
    "l2": 5e-4,          # penalización L2
    "threshold": 0.014,  # sensibilidad del label
    "gamma": 1.1,        # parámetro Focal Loss
    "shrink_lambda": 0.62,   # mezcla thresholds base / tilt
    "fee": 0.00125,          # comisión de trading
    "borrow_rate": 0.0025,   # tasa anual de préstamo para shorts
    "sl": 0.015,         # stop-loss %
    "tp": 0.04,          # take-profit %
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
        cw_train_cb={0: 1.15, 1: 1.10, 2: 1.05},
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

    # === DISTRIBUCIONES DE PREDICCIONES / VERDAD / SEÑALES ===
    import numpy as np
    from collections import Counter

    def _dist_fmt(cnt, total, labels, label_names=None):
        parts = []
        for i, lab in enumerate(labels):
            name = f"{lab}" if label_names is None else label_names[i]
            c = cnt.get(lab, 0)
            p = 100.0 * c / total if total > 0 else 0.0
            parts.append(f"{name}: {c:4d} ({p:5.1f}%)")
        return " | ".join(parts)

    def print_pred_distributions(res_obj):
        if "y_true_pred" not in res_obj:
            print("[pred_props] No hay y_true_pred en res.")
            return

        res_obj.setdefault("diagnostics", {})
        res_obj["diagnostics"].setdefault("pred_distributions", {})

        for split in ("train", "val", "test"):
            if split not in res_obj["y_true_pred"]:
                continue

            y_true_s, y_pred_s = res_obj["y_true_pred"][split]
            y_true_s = np.asarray(y_true_s).ravel().astype(int)
            y_pred_s = np.asarray(y_pred_s).ravel().astype(int)
            n = y_pred_s.size

            cnt_true = Counter(y_true_s)
            cnt_pred = Counter(y_pred_s)

            # Mapeo a señales (-1,0,1) a partir de lo PREDICHO
            sig = y_to_signal(y_pred_s)
            cnt_sig = Counter(sig)

            print(f"\n=== {split.upper()} — Distribuciones ===")
            print(f"Total muestras: {n}")
            print("Real     (clases 0/1/2):   " + _dist_fmt(cnt_true, n, [0,1,2]))
            print("Predicho (clases 0/1/2):   " + _dist_fmt(cnt_pred, n, [0,1,2]))
            print("Señales  (-1/0/1) de ŷ:    " + _dist_fmt(cnt_sig,  n, [-1,0,1], ["SHORT(-1)","HOLD(0)","LONG(1)"]))

            # Guarda también en res si luego quieres usarlo
            res_obj["diagnostics"]["pred_distributions"][split] = {
                "n": int(n),
                "true_counts": {int(k): int(v) for k, v in cnt_true.items()},
                "pred_counts": {int(k): int(v) for k, v in cnt_pred.items()},
                "signal_counts": {int(k): int(v) for k, v in cnt_sig.items()},
            }

    # Llamada a las distribuciones (TRAIN/VAL/TEST)
    print_pred_distributions(res)

    # Fallback: si no hay ŷ_train guardado en res, al menos muestra distribución REAL de TRAIN
    if not ("y_true_pred" in res and "train" in res["y_true_pred"]):
        ytr_true = np.asarray(y_train).ravel().astype(int)
        ntr = ytr_true.size
        cnt_true = Counter(ytr_true)
        print("\n=== TRAIN — Distribución REAL (no hay ŷ_train en res) ===")
        print(f"Total muestras: {ntr}")
        print("Real (0/1/2): " + _dist_fmt(cnt_true, ntr, [0,1,2]))

    # === Backtest avanzado (TEST) ===
    print("\n=== Ejecutando Backtest Avanzado ===")
    if "y_true_pred" in res and "test" in res["y_true_pred"]:
        y_true, y_pred = res["y_true_pred"]["test"]
        signals = y_to_signal(y_pred)

        # Ajustar longitud del índice al número de señales
        aligned_idx = idx_test[-len(signals):]

        bt_adv = backtest_advanced(
            close=close_series,
            idx=aligned_idx,
            signal=signals,
            fee=CONFIG["fee"],
            borrow_rate_annual=CONFIG["borrow_rate"],
            sl_pct=CONFIG["sl"],
            tp_pct=CONFIG["tp"],
        )

        print("\n=== Métricas del Backtest Avanzado ===")
        for k, v in bt_adv["metrics"].items():
            print(f" - {k}: {v:.4f}")

    # ============================================================
    # VISUALIZACIONES
    # ============================================================
    print("\n=== Visualizaciones ===")

    # 1) Curvas de equity (train/val/test en orden cronológico)
    if "backtest" in res:
        plot_equity_curves(res["backtest"])

    # 2) Curva del backtest avanzado (TEST)
    if 'bt_adv' in locals():
        import matplotlib.pyplot as plt

        eq = bt_adv["series"]["equity"]
        dd = bt_adv["series"]["drawdown"]

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(eq, label="Equity (TEST)", color="tab:blue")
        plt.title("Curva de Equity - Backtest Avanzado (TEST)")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(dd, label="Drawdown", color="tab:red")
        plt.title("Drawdown (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 3) Matriz de confusión (TEST)
    if "y_true_pred" in res and "test" in res["y_true_pred"]:
        y_true, y_pred = res["y_true_pred"]["test"]
        plot_confusion_matrix(y_true, y_pred, classes=(0, 1, 2))

    # 4) Evolución del F1/Loss
    if "history" in res:
        plot_f1_history(res["history"])
#Nuevos
