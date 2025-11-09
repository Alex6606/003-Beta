# ============================================
# metrics.py
# ============================================
"""
Módulo de métricas y reportes de evaluación.
Incluye funciones de F1 macro, reportes detallados y matriz de confusión 3x3.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def macro_f1(y_true, y_pred, labels=(0, 1, 2)) -> float:
    """
    Calcula Macro-F1 promediando exactamente sobre las clases {0,1,2}.
    Retorna un float.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=list(labels),
            zero_division=0
        )
    )


def print_macro_reports(tag: str, y_true, y_pred):
    """
    Imprime el F1-macro y un reporte de clasificación detallado.
    """
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"{tag} macroF1={f1m:.4f}")
    print(classification_report(y_true, y_pred, digits=3))


def evaluate_split_3x3(tag: str, model, X, y):
    """
    Evalúa el modelo CNN en un split (train, val, test):
      - Accuracy
      - Macro-F1
      - Distribución de clases predichas
      - Matriz de confusión normalizada
    """
    logits = model.predict(X, verbose=0)
    pred = logits.argmax(axis=1)
    acc = (pred == y).mean()
    f1m = f1_score(y, pred, average="macro", zero_division=0)

    dist = (
        pd.Series(pred)
        .value_counts(normalize=True)
        .reindex([0, 1, 2])
        .fillna(0.0)
        .round(3)
        .to_dict()
    )

    print(f"{tag} | acc={acc:.4f}  macroF1={f1m:.4f}\n  y_pred dist: {dist}")

    cm = confusion_matrix(y, pred, labels=[0, 1, 2], normalize='true')
    print("  Matriz 3x3:\n", pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2]).round(3))

    return acc, f1m, dist, cm


def quick_report(name: str, pred, y):
    """
    Reporte rápido de exactitud, distribución y matriz normalizada.
    """
    acc = (pred == y).mean()
    dist = (
        pd.Series(pred)
        .value_counts(normalize=True)
        .sort_index()
        .round(3)
        .to_dict()
    )

    print(f"{name} acc={acc:.4f} pred_dist={dist}")
    print(
        pd.crosstab(
            pd.Series(y, name="y_true"),
            pd.Series(pred, name="y_pred"),
            normalize="index"
        ).round(3)
    )
