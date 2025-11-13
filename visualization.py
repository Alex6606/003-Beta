# ================================================
# visualization.py — versión ordenada y corregida
# ================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. Curvas de equity por split (orden: train → test → val)
# ============================================================
def plot_equity_curves(backtest_results, show_drawdown=True):
    """
    Muestra las curvas de equity acumuladas de cada split (train/test/val)
    en orden fijo y limpio. Cada gráfico va separado.
    """
    splits_order = ["train", "test", "val"]
    colors = {"train": "tab:blue", "test": "tab:green", "val": "tab:orange"}

    # === FIGURA 1: Curvas de equity ===
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

    plt.title("Curvas de Equity por Split (Train → Test → Val)")
    plt.xlabel("Timestep unificado")
    plt.ylabel("Equity acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === FIGURA 2: Curvas de drawdown (solo si se pide) ===
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

        plt.title("Drawdowns por Split (Train → Test → Val)")
        plt.xlabel("Timestep unificado")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================
# 2. Matriz de confusión
# ============================================================
def plot_confusion_matrix(y_true, y_pred, classes=(0, 1, 2)):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
    plt.title("Matriz de Confusión (Normalizada)")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. Historial del Loss / F1-score
# ============================================================
def plot_f1_history(history):
    """
    Gráfica el loss durante Warmup y Finetune respetando tu estilo previo.
    """
    plt.figure(figsize=(8, 5))

    for phase in ["warmup", "finetune"]:
        if phase in history:
            if "loss" in history[phase]:
                plt.plot(history[phase]["loss"], label=f"{phase}_loss")
            if "val_loss" in history[phase]:
                plt.plot(history[phase]["val_loss"], label=f"{phase}_val_loss")

    plt.title("Evolución del Loss durante Entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
