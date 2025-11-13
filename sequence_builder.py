# ============================================
# sequence_builder.py
# ============================================
"""
Module for transforming tabular datasets (X, y)
into 3D sequences for CNN or RNN models.
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
    Converts (X_df, y_ser) → (X_seq, y_seq, idx)
    for training CNN/LSTM models.

    Parameters
    ----------
    X_df : pd.DataFrame
        Normalized feature matrix with a time index.
    y_ser : pd.Series
        Aligned labels (classes 0/1/2).
    window : int
        Length of the rolling time window.
    step : int
        Step size between consecutive samples.

    Returns
    -------
    X_seq : np.ndarray (n_samples, window, n_features)
    y_seq : np.ndarray (n_samples,)
    idx_seq : pd.Index containing the corresponding dates
    """
    # --- Validations ---
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df must be a DataFrame")
    if not isinstance(y_ser, pd.Series):
        raise TypeError("y_ser must be a Series")
    if window < 1 or step < 1:
        raise ValueError("window and step must be >= 1")

    # --- Index alignment ---
    y_aligned = y_ser.reindex(X_df.index)
    mask = y_aligned.notna()

    if mask.sum() < window:
        raise ValueError(
            f"Not enough samples after alignment (n={mask.sum()}) for window={window}"
        )

    X_base = X_df.loc[mask]
    y_base = y_aligned.loc[mask].astype("int64")
    idx_base = X_base.index

    X = X_base.values
    y = y_base.values
    n, n_feat = X.shape

    # --- Build sequences ---
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
        raise ValueError("No sequences generated; check window/step and dataset size.")

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
    Applies make_sequences to each split (train/test/val),
    ensuring correct temporal alignment without leakage.
    """
    Xtr_seq, ytr_seq, itrg = make_sequences(X_train, y_train, window=window, step=step)
    Xte_seq, yte_seq, iteg = make_sequences(X_test,  y_test,  window=window, step=step)
    Xva_seq, yva_seq, ivag = make_sequences(X_val,   y_val,   window=window, step=step)

    return {
        "train": {"X": Xtr_seq, "y": ytr_seq, "idx": itrg},
        "test":  {"X": Xte_seq, "y": yte_seq, "idx": iteg},
        "val":   {"X": Xva_seq, "y": yva_seq, "idx": ivag},
    }
