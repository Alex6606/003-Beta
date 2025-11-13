# ============================================
# data_utils.py
# ============================================
"""
Module for OHLCV data acquisition and cleaning using Yahoo Finance.
"""

import pandas as pd
import yfinance as yf


def get_data(ticker: str = "MSFT") -> pd.DataFrame:
    """
    Downloads 15 years of daily data from yfinance for a single ticker.
    Returns a DataFrame with a DatetimeIndex and the columns:
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    Processing steps:
      - Sorts index chronologically
      - Removes duplicate index entries
      - Forward-fills missing values
      - Drops remaining nulls
      - Ensures numeric types and non-negative volume
    """
    # --- 1) Time range ---
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=15)

    # --- 2) Download ---
    df_raw = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df_raw.empty:
        raise RuntimeError(
            f"Download returned empty data for {ticker}. "
            "Check the symbol or your internet connection."
        )

    # --- 3) Flatten MultiIndex columns if needed ---
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    # --- 4) Sort and remove duplicates ---
    df_raw = df_raw[~df_raw.index.duplicated(keep='last')].sort_index()

    # --- 5) Validate expected columns ---
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    missing = [c for c in expected_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns in download: {missing}")

    # --- 6) Numeric conversion ---
    for c in expected_cols:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    # --- 7) Cleaning ---
    df_clean = df_raw[expected_cols].ffill().dropna(how="any").copy()
    df_clean['Volume'] = df_clean['Volume'].clip(lower=0)

    # --- 8) Datetime index ---
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean.index = pd.to_datetime(df_clean.index)

    assert df_clean.index.is_monotonic_increasing, "Index is not sorted in ascending order."

    return df_clean




