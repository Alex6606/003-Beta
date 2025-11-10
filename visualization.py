# ================================================
# visualization.py
# ================================================
"""
Visualizaciones para el proyecto Deep Learning Trading:
 - Equity curves de backtest
 - Matriz de confusión
 - Evolución del F1-score durante entrenamiento
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === 1. Curvas de equity (backtest) ===
def plot_equity_curves(backtest_results):
    """
    Muestra las curvas de equity acumuladas de cada split (train/val/test)
    en orden cronológico (una continua tras otra).
    """
    plt.figure(figsize=(10, 5))

    offset = 0
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}

    for split in ["train", "val", "test"]:
        if split not in backtest_results:
            print(f"[plot_equity_curves] No equity data for {split}.")
            continue

        metrics = backtest_results[split]
        equity = np.array(metrics.get("equity", []))
        if equity.size == 0:
            print(f"[plot_equity_curves] Empty equity for {split}.")
            continue

        # eje x desplazado para continuidad visual
        x = np.arange(offset, offset + len(equity))
        plt.plot(x, equity, label=f"{split.upper()} (Ret={metrics['final_return']*100:.2f}%)",
                 color=colors.get(split, None))
        offset += len(equity)

    plt.title("Curvas de Equity por Split (Orden Cronológico)")
    plt.xlabel("Timestep (unificado)")
    plt.ylabel("Equity acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === 2. Matriz de confusión ===
def plot_confusion_matrix(y_true, y_pred, classes=(0, 1, 2)):
    """
    Dibuja una matriz de confusión normalizada.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de confusión (normalizada)")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()
    plt.show()


# === 3. Historial de F1 durante entrenamiento ===
def plot_f1_history(history):
    """
    Dibuja la evolución del loss o métricas durante entrenamiento two-phase.
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
