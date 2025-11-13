# ============================================================
# main.py — Deep Learning Trading with CNN + Advanced Backtesting (Final Refactor)
# ============================================================

from data_utils import get_data
from features_pipeline import build_and_normalize_features_per_split, export_features_for_drift
from indicators import WINDOWS
from labeling import build_labels_for_feature_splits
from training_pipeline import train_eval_from_raw, summarize_run, sanity_check_from_res
from backtest import y_to_signal
from visualization import plot_equity_curves, plot_confusion_matrix, plot_f1_history

import numpy as np
from collections import Counter


# === CENTRAL CONFIGURATION ===
CONFIG = {
    "seq_window": 60,
    "seq_step": 1,
    "dropout": 0.3,
    "l2": 5e-4,
    "threshold": 0.014,
    "gamma": 1.1,
    "shrink_lambda": 0.62,
    "fee": 0.00125,
    "borrow_rate": 0.0025,
    "sl": 0.015,
    "tp": 0.04,
}


# ============================================================
# Internal reporting utilities
# ============================================================

def _dist_fmt(cnt, total, labels, label_names=None):
    parts = []
    for i, lab in enumerate(labels):
        name = f"{lab}" if label_names is None else label_names[i]
        c = cnt.get(lab, 0)
        p = 100.0 * c / total if total > 0 else 0.0
        parts.append(f"{name}: {c:4d} ({p:5.1f}%)")
    return " | ".join(parts)


def print_pred_distributions(res_obj):
    """
    Prints class distributions (0/1/2) and signal distributions (-1/0/1)
    for train, val, and test splits.
    """
    if "y_true_pred" not in res_obj:
        print("[pred_props] No y_true_pred found in results.")
        return

    print("\n========== Class & Signal Distributions ==========\n")

    for split in ("train", "val", "test"):
        if split not in res_obj["y_true_pred"]:
            continue

        y_true_s, y_pred_s = res_obj["y_true_pred"][split]
        y_true_s = np.asarray(y_true_s).ravel().astype(int)
        y_pred_s = np.asarray(y_pred_s).ravel().astype(int)
        n = y_pred_s.size

        cnt_true = Counter(y_true_s)
        cnt_pred = Counter(y_pred_s)

        sig = y_to_signal(y_pred_s)
        cnt_sig = Counter(sig)

        print(f"\n=== {split.upper()} ===")
        print(f"Total samples: {n}")
        print("True     (0/1/2): " + _dist_fmt(cnt_true, n, [0, 1, 2]))
        print("Predicted (0/1/2): " + _dist_fmt(cnt_pred, n, [0, 1, 2]))
        print("Signals (-1/0/1): " +
              _dist_fmt(cnt_sig, n, [-1, 0, 1], ["SHORT(-1)", "HOLD(0)", "LONG(1)"]))


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":

    # ---------------------------
    # 1) DATA
    # ---------------------------
    print("=== Downloading data ===")
    data = get_data("MSFT")

    print("\n=== Building features and normalizing ===")
    bundle = build_and_normalize_features_per_split(data, windows=WINDOWS, warmup=200)
    export_features_for_drift(bundle)

    feat_train_n = bundle["norm"]["train"]
    feat_test_n = bundle["norm"]["test"]
    feat_val_n = bundle["norm"]["val"]

    # LABELS
    print("\n=== Building labels ===")
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
    X_test, y_test = labels_bundle["X"]["test"], labels_bundle["y"]["test"]
    X_val, y_val = labels_bundle["X"]["val"], labels_bundle["y"]["val"]

    idx_train = feat_train_n.index
    idx_test = feat_test_n.index
    idx_val = feat_val_n.index
    close_series = data["Close"]

    # ---------------------------
    # 2) TRAINING + EVALUATION
    # ---------------------------
    print("\n=== Training CNN-1D model ===")
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
        train_idx=idx_train,
        val_idx=idx_val,
        test_idx=idx_test,
        close_series=close_series,
        feature_names=list(feat_train_n.columns),
    )

    # ---------------------------
    # 3) REPORTS
    # ---------------------------
    print("\n========== RUN SUMMARY ==========")
    summarize_run(res)
    sanity_check_from_res(res)

    # CLASS/SIGNAL DISTRIBUTIONS
    print_pred_distributions(res)

    # ===========================================================
    # 4) BACKTEST (Train → Test → Val)
    # ===========================================================
    print("\n========== ADVANCED BACKTESTS PER SPLIT ==========\n")

    ORDER = ["train", "test", "val"]

    if "backtest" in res:
        for split in ORDER:
            if split not in res["backtest"]:
                continue
            bt = res["backtest"][split]
            print(f"--- {split.upper()} ---")
            print(f"Final Return: {bt.get('final_return', 0):.4f}")
            for k, v in bt["metrics"].items():
                print(f" - {k}: {v:.4f}")
            print()

    # ===========================================================
    # Global Metrics (average Train/Test/Val)
    # ===========================================================
    print("\n=== Global Metrics (Average Train/Test/Val) ===")

    if "backtest" in res:
        def avg(metric):
            vals = [
                res["backtest"][s]["metrics"].get(metric, 0)
                for s in ORDER if s in res["backtest"]
            ]
            return sum(vals) / len(vals)

        print(f"Global CAGR:       {avg('CAGR'):.4f}")
        print(f"Global Sharpe:     {avg('Sharpe'):.4f}")
        print(f"Global Sortino:    {avg('Sortino'):.4f}")
        print(f"Global Calmar:     {avg('Calmar'):.4f}")
        print(f"Global WinRate:    {avg('WinRate'):.4f}")
        print(f"Global MaxDD:      {avg('MaxDrawdown'):.4f}")

    # ===========================================================
    # 5) VISUALIZATIONS
    # ===========================================================
    print("\n========== Visualizations ==========\n")

    # Equity curves (Train → Test → Val)
    if "backtest" in res:
        plot_equity_curves(res["backtest"], show_drawdown=True)

    # Confusion Matrix (TEST)
    if "y_true_pred" in res and "test" in res["y_true_pred"]:
        y_true, y_pred = res["y_true_pred"]["test"]
        plot_confusion_matrix(y_true, y_pred, classes=(0, 1, 2))

    # F1/Loss evolution
    if "history" in res:
        plot_f1_history(res["history"])

    print("\n=== Pipeline completed. ===")
