# ================================================
# visualization.py — ordered and cleaned version
# ================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. Equity curves per split (order: train → test → val)
# ============================================================
def plot_equity_curves(backtest_results, show_drawdown=True):
    """
    Displays accumulated equity curves for each split (train/test/val)
    in a fixed clean order. Each plot is shown separately.
    """
    splits_order = ["train", "test", "val"]
    colors = {"train": "tab:blue", "test": "tab:green", "val": "tab:orange"}

    # === FIGURE 1: Equity curves ===
    plt.figure(figsize=(10, 5))

    offset = 0
    for split in splits_order:
        if split not in backtest_results:
            continue

        bt = backtest_results[split]
        equity = np.array(bt.get("equity", []))
        metrics = bt.get("metrics", {})

        if equity.size == 0:
            continue

        x = np.arange(offset, offset + len(equity))
        sharpe = metrics.get("Sharpe", np.nan)
        mdd = metrics.get("MaxDrawdown", np.nan)

        plt.plot(
            x, equity,
            color=colors[split],
            label=f"{split.upper()} | Sharpe={sharpe:.2f}, MDD={mdd*100:.1f}%"
        )

        offset += len(equity)

    plt.title("Equity Curves per Split (Train → Test → Val)")
    plt.xlabel("Unified Timestep")
    plt.ylabel("Accumulated Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === FIGURE 2: Drawdowns (only if requested) ===
    if show_drawdown:
        plt.figure(figsize=(10, 4))

        offset = 0
        for split in splits_order:
            if split not in backtest_results:
                continue

            equity = np.array(backtest_results[split].get("equity", []))
            if equity.size == 0:
                continue

            eq_series = equity
            drawdown = eq_series / np.maximum.accumulate(eq_series) - 1
            x = np.arange(offset, offset + len(drawdown))

            plt.plot(
                x, drawdown,
                color=colors[split],
                label=f"{split.upper()}"
            )

            offset += len(drawdown)

        plt.title("Drawdowns per Split (Train → Test → Val)")
        plt.xlabel("Unified Timestep")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================
# 2. Confusion Matrix
# ============================================================
def plot_confusion_matrix(y_true, y_pred, classes=(0, 1, 2)):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Prediction")
    plt.ylabel("True Value")
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. Loss / F1-score History
# ============================================================
def plot_f1_history(history):
    """
    Plots the loss curves during Warmup and Finetune,
    following your previous visual style.
    """
    plt.figure(figsize=(8, 5))

    for phase in ["warmup", "finetune"]:
        if phase in history:
            if "loss" in history[phase]:
                plt.plot(history[phase]["loss"], label=f"{phase}_loss")
            if "val_loss" in history[phase]:
                plt.plot(history[phase]["val_loss"], label=f"{phase}_val_loss")

    plt.title("Loss Evolution During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

