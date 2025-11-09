# ============================================
# sequence_builder.py
# ============================================
"""
Módulo para transformar conjuntos tabulares (X, y)
en secuencias 3D para modelos CNN o RNN.
"""

import numpy as np
import pandas as pd


def make_sequences(
    X_df: pd.DataFrame,
    y_ser: pd.Series,
    window: int = 30,
    step: int = 1
):
    """
    Convierte (X_df, y_ser) → (X_seq, y_seq, idx)
    para entrenamiento de modelos CNN/LSTM.

    Parámetros
    ----------
    X_df : pd.DataFrame
        Features normalizados con índice temporal.
    y_ser : pd.Series
        Etiquetas alineadas (clases 0/1/2).
    window : int
        Longitud de la ventana temporal.
    step : int
        Desplazamiento entre muestras consecutivas.

    Retorna
    -------
    X_seq : np.ndarray (n_samples, window, n_features)
    y_seq : np.ndarray (n_samples,)
    idx_seq : pd.Index de las fechas correspondientes
    """
    # --- Validaciones ---
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df debe ser DataFrame")
    if not isinstance(y_ser, pd.Series):
        raise TypeError("y_ser debe ser Series")
    if window < 1 or step < 1:
        raise ValueError("window y step deben ser >= 1")

    # --- Alineación por índice ---
    y_aligned = y_ser.reindex(X_df.index)
    mask = y_aligned.notna()

    if mask.sum() < window:
        raise ValueError(
            f"No hay suficientes muestras tras alinear (n={mask.sum()}) para window={window}"
        )

    X_base = X_df.loc[mask]
    y_base = y_aligned.loc[mask].astype("int64")
    idx_base = X_base.index

    X = X_base.values
    y = y_base.values
    n, n_feat = X.shape

    # --- Construcción de secuencias ---
    X_seq, y_seq, idx_seq = [], [], []
    start = window - 1
    for i in range(start, n, step):
        Xi = X[i - window + 1 : i + 1]
        if Xi.shape[0] != window:
            continue
        X_seq.append(Xi)
        y_seq.append(y[i])
        idx_seq.append(idx_base[i])

    if len(X_seq) == 0:
        raise ValueError("No se generaron secuencias; revisa window/step y tamaño del dataset.")

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.int64)
    idx_seq = pd.Index(idx_seq, name="date")

    return X_seq, y_seq, idx_seq


def build_cnn_sequences_for_splits(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
    X_val:   pd.DataFrame, y_val:  pd.Series,
    window: int = 30,
    step: int = 1
):
    """
    Aplica make_sequences a cada split (train/test/val)
    garantizando alineación temporal sin fugas.
    """
    Xtr_seq, ytr_seq, itrg = make_sequences(X_train, y_train, window=window, step=step)
    Xte_seq, yte_seq, iteg = make_sequences(X_test,  y_test,  window=window, step=step)
    Xva_seq, yva_seq, ivag = make_sequences(X_val,   y_val,   window=window, step=step)

    return {
        "train": {"X": Xtr_seq, "y": ytr_seq, "idx": itrg},
        "test":  {"X": Xte_seq, "y": yte_seq, "idx": iteg},
        "val":   {"X": Xva_seq, "y": yva_seq, "idx": ivag},
    }
