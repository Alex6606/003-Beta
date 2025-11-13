# ============================================
# labeling.py
# ============================================
"""
Label generation module for financial ML models.
Includes simplified triple-barrier labeling, class weights,
alignment with feature matrices, and exploration utilities.
"""

import numpy as np
import pandas as pd
from IPython.display import display  # optional for debugging


# ======================================================
# 1. Triple-barrier labeling
# ======================================================

def make_labels_from_prices(prices: pd.Series, horizon: int = 3, threshold: float = 0.01) -> pd.DataFrame:
    """
    Lightweight Triple-Barrier labeling: picks the first barrier reached in [t+1 .. t+h].

    Labels (y3):
        y3 = 2 → UP    (first hit is +threshold)
        y3 = 1 → HOLD  (no barrier reached)
        y3 = 0 → DOWN  (first hit is -threshold)

    Returns:
        DataFrame with:
            - 'y3': multiclass label (int8)
            - 'ret_fwd': forward return at horizon h
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("Expected a price Series (pd.Series)")

    # Future returns matrix
    R = pd.concat([prices.shift(-i) / prices - 1.0 for i in range(1, horizon + 1)], axis=1)
    R.columns = range(1, horizon + 1)

    Rv = R.values
    up_mask = (Rv >= threshold)
    dn_mask = (Rv <= -threshold)

    up_any = up_mask.any(axis=1)
    dn_any = dn_mask.any(axis=1)

    first_up = np.where(up_any, np.argmax(up_mask, axis=1), -1)
    first_dn = np.where(dn_any, np.argmax(dn_mask, axis=1), -1)

    # Default label = HOLD
    y = np.full(len(R), 1, dtype=np.int8)

    # Only UP or only DOWN
    only_up = (first_up >= 0) & (first_dn < 0)
    only_dn = (first_dn >= 0) & (first_up < 0)
    y[only_up] = 2
    y[only_dn] = 0

    # Both barriers reached → choose first hit
    both = (first_up >= 0) & (first_dn >= 0)
    y[both & (first_up < first_dn)] = 2
    y[both & (first_up > first_dn)] = 0

    # Valid rows = no NaN forward returns
    valid = R.notna().all(axis=1).values

    y = pd.Series(y, index=prices.index)
    ret_fwd = (prices.shift(-horizon) / prices) - 1.0

    out = pd.DataFrame({"y3": y, "ret_fwd": ret_fwd}, index=prices.index)
    out = out[valid].copy()
    return out


# ======================================================
# 2. Class weights and statistics
# ======================================================

def compute_class_weights_triple(y: pd.Series) -> dict:
    """
    Balanced weights for classes {0, 1, 2}:
        w_c = N / (K * N_c)
    """
    classes = [0, 1, 2]
    present = [c for c in classes if (y == c).any()]
    K = len(present)
    N = len(y)
    return {
        c: (float(N) / (K * int((y == c).sum())))
        if int((y == c).sum()) > 0 else 0.0
        for c in present
    }


def compute_class_stats(y):
    classes, counts = np.unique(y, return_counts=True)
    freq = counts / counts.sum()
    return dict(zip(classes, counts)), dict(zip(classes, freq))


def class_weights_from_train(ytr):
    counts, _ = compute_class_stats(ytr)
    K = len(counts)
    N = sum(counts.values())
    return {k: N / (K * counts[k]) for k in counts}


def focal_alphas_from_train(ytr):
    _, freq = compute_class_stats(ytr)
    inv = {k: 1.0 / (freq[k] + 1e-9) for k in freq}
    s = sum(inv.values())
    return {k: inv[k] / s for k in inv}  # normalized α


# ======================================================
# 3. Alignment with features
# ======================================================

def build_labels_for_feature_splits(
    data_ohlcv: pd.DataFrame,
    feat_train_n: pd.DataFrame,
    feat_test_n: pd.DataFrame,
    feat_val_n: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 3,
    threshold: float = 0.01
):
    """
    Generates triple-barrier labels and aligns them with the normalized
    feature matrices. Returns a dict containing X, y, forward returns,
    and class_weights for each split.
    """
    labels_df = make_labels_from_prices(
        data_ohlcv[price_col], horizon=horizon, threshold=threshold
    )

    y_all = labels_df["y3"].astype("int64")
    ret_all = labels_df["ret_fwd"]

    def _align(feat_n: pd.DataFrame):
        idx = feat_n.index.intersection(labels_df.index)
        X = feat_n.loc[idx]
        y = y_all.loc[idx]
        r = ret_all.loc[idx]
        return X, y, r

    Xtr, ytr, rtr = _align(feat_train_n)
    Xte, yte, rte = _align(feat_test_n)
    Xva, yva, rva = _align(feat_val_n)

    cw_tr = compute_class_weights_triple(ytr)
    cw_te = compute_class_weights_triple(yte)
    cw_va = compute_class_weights_triple(yva)

    return {
        "X": {"train": Xtr, "test": Xte, "val": Xva},
        "y": {"train": ytr, "test": yte, "val": yva},
        "ret": {"train": rtr, "test": rte, "val": rva},
        "class_weights": {"train": cw_tr, "test": cw_te, "val": cw_va},
    }


# ======================================================
# 4. Exploration and debugging
# ======================================================

def evaluate_label_distributions_df(
    data_ohlcv: pd.DataFrame,
    feat_train_n: pd.DataFrame,
    thresholds,
    horizons,
    price_col="Close"
) -> pd.DataFrame:
    """
    Explores TRAIN-only label distributions for multiple (threshold, horizon)
    combinations using triple-barrier labeling.
    """
    rows = []
    for thr in thresholds:
        for h in horizons:
            labels_df = make_labels_from_prices(
                data_ohlcv[price_col], horizon=h, threshold=thr
            )
            y = labels_df["y3"].reindex(feat_train_n.index).dropna().astype("int64")
            dist = y.value_counts().reindex([0, 1, 2], fill_value=0)
            total = int(dist.sum())
            pct0, pct1, pct2 = (dist / total * 100).round(2).tolist()
            cw = compute_class_weights_triple(y)
            rows.append({
                "threshold": thr,
                "horizon": h,
                "n_train_labels": total,
                "pct_down_0": pct0,
                "pct_hold_1": pct1,
                "pct_up_2": pct2,
                "w0": round(cw.get(0, 0.0), 3),
                "w1": round(cw.get(1, 0.0), 3),
                "w2": round(cw.get(2, 0.0), 3),
            })
    return pd.DataFrame(rows).sort_values(["horizon", "threshold"]).reset_index(drop=True)


def _debug_triple_barrier_view(prices: pd.Series, horizon: int = 3, thr: float = 0.01, n=8):
    """
    Debugging utility: prints the matrix of forward returns with UP/DOWN hits.
    """
    R = pd.concat(
        [prices.shift(-i) / prices - 1.0 for i in range(1, horizon + 1)],
        axis=1
    )
    R.columns = [f"+{i}" for i in range(1, horizon + 1)]

    up_mask = (R.values >= thr)
    dn_mask = (R.values <= -thr)

    up_any = int(up_mask.any(axis=1).sum())
    dn_any = int(dn_mask.any(axis=1).sum())

    print(f"[DEBUG] horizon={horizon} thr={thr} | rows with UP-hit: {up_any} | DOWN-hit: {dn_any} | total: {len(R)}")
    display(R.head(n))
    return R
