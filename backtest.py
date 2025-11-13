# ==========================================================
# backtest.py
# ==========================================================
"""
Advanced vectorized backtesting for strategies based on
discrete signals {-1, 0, +1}.

Includes:
 - Percentual Stop-loss / Take-profit
 - Trading and borrow costs
 - Performance metrics (Sharpe, Sortino, Calmar, MDD, WinRate)
 - Equity & drawdown curves
"""

import numpy as np
import pandas as pd


# === Mapping CNN labels → trading signals ===
def y_to_signal(y_pred):
    """
    Maps CNN classes {0, 1, 2} → {-1, 0, +1}
      0 = short, 1 = hold, 2 = long
    """
    y_pred = np.asarray(y_pred).ravel()
    mapping = {0: -1, 1: 0, 2: +1}
    return np.vectorize(mapping.get)(y_pred)


# === Advanced Backtest ===
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
    Advanced vectorized backtest with approximate SL/TP modeling.
    Includes extended metrics: Sharpe, Sortino, Calmar, MDD, WinRate.
    """

    # Align signals to price index
    sig = pd.Series(signal, index=idx).astype(float)
    sig = sig.reindex(close.index).fillna(0.0)

    # Daily returns of the asset
    r = close.pct_change().fillna(0.0)

    # Effective position
    pos = sig.shift(1).fillna(0.0)

    # =============================
    # Trading & Borrow Costs
    # =============================
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    trading_cost = turnover * fee

    borrow_daily = borrow_rate_annual / freq
    borrow_cost = borrow_daily * (pos < 0).astype(float).abs()

    # =============================
    # Base Strategy Return
    # =============================
    strat_ret = pos * r

    # =============================
    # Stop-loss / Take-profit
    # =============================
    sl_hit = ((r < -sl_pct) & (pos > 0)) | ((r > sl_pct) & (pos < 0))
    tp_hit = ((r > tp_pct) & (pos > 0)) | ((r < -tp_pct) & (pos < 0))

    # Apply SL/TP
    strat_ret = np.where(sl_hit, -sl_pct, strat_ret)
    strat_ret = np.where(tp_hit, tp_pct, strat_ret)

    # =============================
    # Final Costs
    # =============================
    strat_ret = strat_ret - trading_cost - borrow_cost

    # =============================
    # Equity Curve
    # =============================
    eq = (1.0 + strat_ret).cumprod()

    # Drawdown series
    dd = eq / eq.cummax() - 1.0

    # =============================
    # Core Performance Metrics
    # =============================
    mu = strat_ret.mean() * freq
    sigma = strat_ret.std(ddof=1) * np.sqrt(freq)

    sharpe = mu / sigma if sigma > 0 else 0.0
    mdd = float(dd.min())
    cagr = float(eq.iloc[-1] ** (freq / len(eq)) - 1.0) if len(eq) > 0 else 0.0
    win_rate = float((strat_ret > 0).mean())

    # =============================
    # Extended Metrics
    # =============================

    # --- Sortino Ratio ---
    downside = strat_ret[strat_ret < 0]
    downside_std = downside.std(ddof=1) * np.sqrt(freq)
    sortino = mu / downside_std if downside_std > 0 else 0.0

    # --- Calmar Ratio ---
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0

    return {
        "series": {
            "returns": pd.Series(strat_ret, index=close.index),
            "equity": eq,
            "drawdown": dd,
            "signals": sig,
        },
        "metrics": {
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "MaxDrawdown": mdd,
            "AnnualVol": sigma,
            "WinRate": win_rate,
            "SL_hits": int(sl_hit.sum()),
            "TP_hits": int(tp_hit.sum()),
        },
    }
