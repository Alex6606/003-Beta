# ============================================
# data_utils.py
# ============================================
"""
Módulo de adquisición y limpieza de datos OHLCV desde Yahoo Finance.
"""

import pandas as pd
import yfinance as yf


def get_data(ticker: str = "MSFT") -> pd.DataFrame:
    """
    Descarga 15 años de datos diarios desde yfinance para un solo ticker,
    devuelve un DataFrame 'data' con índice de fechas y columnas:
      ['Open','High','Low','Close','Adj Close','Volume']

    Procesamiento:
      - Ordena el índice cronológicamente
      - Elimina duplicados de índice
      - Rellena faltantes por forward-fill
      - Elimina remanentes nulos
      - Asegura tipos numéricos y volumen no-negativo
    """
    # --- 1) Rango temporal ---
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=15)

    # --- 2) Descarga ---
    df_raw = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df_raw.empty:
        raise RuntimeError(f"La descarga regresó vacía para {ticker}. Verifica el símbolo o la conexión.")

    # --- 3) Si viene MultiIndex, aplanar columnas ---
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    # --- 4) Orden y duplicados ---
    df_raw = df_raw[~df_raw.index.duplicated(keep="last")].sort_index()

    # --- 5) Validar columnas ---
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    missing = [c for c in expected_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas en la descarga: {missing}")

    # --- 6) Conversión numérica ---
    for c in expected_cols:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    # --- 7) Limpieza ---
    df_clean = df_raw[expected_cols].ffill().dropna(how="any").copy()
    df_clean['Volume'] = df_clean['Volume'].clip(lower=0)

    # --- 8) Índice datetime ---
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean.index = pd.to_datetime(df_clean.index)

    assert df_clean.index.is_monotonic_increasing, "El índice no está ordenado ascendentemente."

    return df_clean




