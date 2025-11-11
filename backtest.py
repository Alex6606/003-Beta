# ==========================================================
# backtest.py
# ==========================================================
"""
Backtest avanzado (vectorizado) para estrategias basadas en señales discretas {-1, 0, +1}.
Incluye:
 - Stop-loss / Take-profit (porcentuales)
 - Costos de trading y borrow dinámico
 - Métricas de desempeño (Sharpe, CAGR, MDD, WinRate)
 - Curvas de equity y drawdown

"""

import numpy as np
import pandas as pd


# === Conversión de etiquetas CNN → señales de trading ===
def y_to_signal(y_pred):
    """
    Mapea las clases de la CNN {0, 1, 2} → {-1, 0, +1}
      0 = short, 1 = hold, 2 = long
    """
    y_pred = np.asarray(y_pred).ravel()
    mapping = {0: -1, 1: 0, 2: +1}
    return np.vectorize(mapping.get)(y_pred)


# === Backtest avanzado ===
def backtest_advanced(
    close: pd.Series,
    idx: pd.Index,
    signal: np.ndarray,
    fee: float = 0.00125,
    borrow_rate_annual: float = 0.0025,
    freq: int = 252,
    sl_pct: float = 0.02,
    tp_pct: float = 0.03,
):
    """
    Backtest vectorizado avanzado con SL/TP aproximado.

    Parámetros:
    -----------
    close : pd.Series
        Serie de precios (Close).
    idx : pd.Index
        Índices del conjunto (train/val/test).
    signal : np.ndarray
        Señales del modelo {-1, 0, +1}.
    fee : float
        Comisión por operación (por unidad de capital).
    borrow_rate_annual : float
        Costo anual por mantener posiciones cortas.
    freq : int
        Frecuencia anualizada (252 = días bursátiles).
    sl_pct, tp_pct : float
        Porcentajes de stop-loss y take-profit (en decimal).

    Retorna:
    --------
    dict con series (equity, retornos, drawdown) y métricas.
    """

    sig = pd.Series(signal, index=idx).astype(float)
    sig = sig.reindex(close.index).fillna(0.0)
    r = close.pct_change().fillna(0.0)
    pos = sig.shift(1).fillna(0.0)

    # --- Costos ---
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    trading_cost = turnover * fee
    borrow_daily = borrow_rate_annual / freq
    borrow_cost = borrow_daily * (pos < 0).astype(float).abs()

    # --- Rendimiento base ---
    strat_ret = pos * r

    # --- Stop-loss / Take-profit aproximado ---
    sl_hit = (r < -sl_pct) & (pos > 0) | (r > sl_pct) & (pos < 0)
    tp_hit = (r > tp_pct) & (pos > 0) | (r < -tp_pct) & (pos < 0)

    # penalización SL y bonus TP (simplificada)
    strat_ret = np.where(sl_hit, -sl_pct, strat_ret)
    strat_ret = np.where(tp_hit, tp_pct, strat_ret)

    # --- Agregar costos ---
    strat_ret = strat_ret - trading_cost - borrow_cost
    eq = (1.0 + strat_ret).cumprod()

    # --- Drawdown ---
    dd = eq / eq.cummax() - 1.0

    # --- Métricas ---
    mu = strat_ret.mean() * freq
    sigma = strat_ret.std(ddof=1) * np.sqrt(freq)
    sharpe = mu / sigma if sigma > 0 else 0.0
    mdd = dd.min()
    cagr = eq.iloc[-1] ** (freq / len(eq)) - 1.0 if len(eq) > 0 else 0.0

    # tasa de acierto condicional
    win_rate = float((strat_ret > 0).mean())

    return {
        "series": {
            "returns": pd.Series(strat_ret, index=close.index),
            "equity": eq,
            "drawdown": dd,
            "signals": sig,
        },
        "metrics": {
            "CAGR": float(cagr),
            "Sharpe": float(sharpe),
            "MaxDrawdown": float(mdd),
            "AnnualVol": float(sigma),
            "WinRate": win_rate,
            "SL_hits": int(sl_hit.sum()),
            "TP_hits": int(tp_hit.sum()),
        },
    }

