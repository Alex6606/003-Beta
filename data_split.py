# ============================================
# data_split.py
# ============================================
"""
Module for splitting financial datasets into chronological 60/20/20 splits.
Prevents temporal leakage and ensures ordered DatetimeIndex formatting.
"""

import pandas as pd


def split_60_20_20(
    data: pd.DataFrame,
    cols_required=None,
    coerce_datetime=True,
    verbose=True
):
    """
    Chronologically splits a DataFrame into 3 non-overlapping subsets:
    train (60%), test (20%), validation (20%).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with temporal index (DatetimeIndex preferred).
    cols_required : list[str], optional
        Columns that must exist and contain no NaNs.
    coerce_datetime : bool
        If True, automatically converts the index to DatetimeIndex.
    verbose : bool
        If True, prints split sizes and date ranges.

    Returns
    -------
    (train, test, val) : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three chronological subsets with no overlap and no look-ahead.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a DataFrame for 'data'.")

    df = data.copy()

    # --- 1) Validate datetime index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        if coerce_datetime:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError("Could not convert index to DatetimeIndex.") from e
        else:
            raise ValueError("Index must be a DatetimeIndex.")

    # --- 2) Sort and remove duplicates ---
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # --- 3) Drop NaN based on required columns ---
    if cols_required:
        missing = [c for c in cols_required if c not in df.columns]
        if missing:
            raise ValueError(f"Required columns not found in 'data': {missing}")
        df = df.dropna(subset=cols_required)

    # --- 4) Compute split boundaries ---
    n = len(df)
    if n < 50:
        raise ValueError(f"Too few rows ({n}) for a robust 60/20/20 split.")

    i_train_end = (60 * n) // 100
    i_test_end = (80 * n) // 100

    train = df.iloc[:i_train_end].copy()
    test  = df.iloc[i_train_end:i_test_end].copy()
    val   = df.iloc[i_test_end:].copy()

    # --- 5) Sanity checks ---
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing
    assert val.index.is_monotonic_increasing
    assert len(train) + len(test) + len(val) == n

    # --- 6) Optional logs ---
    if verbose:
        print("Sizes → train:", train.shape, "| test:", test.shape, "| val:", val.shape)
        print("Date ranges:")
        print("  train:", train.index.min().date(), "→", train.index.max().date())
        print("  test :", test.index.min().date(),  "→", test.index.max().date())
        print("  val  :", val.index.min().date(),   "→", val.index.max().date())

    return train, test, val
