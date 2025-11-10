# ==========================================================
# backtest.py
# ==========================================================
import numpy as np
import pandas as pd

def y_to_signal(y_pred):
    # mapea clases {0,1,2} -> {-1, 0, +1}
    # 0 = DOWN -> short (-1), 1 = HOLD -> 0, 2 = UP -> long (+1)
    y_pred = np.asarray(y_pred).ravel()
    mapping = {0: -1, 1: 0, 2: +1}
    return np.vectorize(mapping.get)(y_pred)

def backtest_simple(close: "pd.Series", idx: "pd.Index", signal: np.ndarray,
                    fee=0.00125, borrow_rate_annual=0.0025, freq=252):
    """
    Backtest vectorizado ultrabásico:
      - Retorno diario = pos_{t-1} * r_t
      - Coste por cambio de posición (turnover * fee)
      - Borrow cost si pos < 0 (prorrateado por día)
    """
    import pandas as pd
    sig = pd.Series(signal, index=idx).astype(float)
    sig = sig.reindex(close.index).fillna(0.0)

    r = close.pct_change().fillna(0.0)
    pos = sig.shift(1).fillna(0.0)

    # costes
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    trading_cost = turnover * fee
    borrow_daily = borrow_rate_annual / freq
    borrow_cost = borrow_daily * (pos < 0).astype(float).abs()

    strat_ret = pos * r - trading_cost - borrow_cost
    eq = (1.0 + strat_ret).cumprod()

    # métricas rápidas
    mu = strat_ret.mean() * freq
    sigma = strat_ret.std(ddof=1) * np.sqrt(freq)
    sharpe = mu / sigma if sigma > 0 else 0.0
    mdd = (eq / eq.cummax() - 1.0).min()

    return {
        "series": {
            "returns": strat_ret,
            "equity": eq
        },
        "metrics": {
            "CAGR": float(eq.iloc[-1] ** (freq / max(len(eq), 1)) - 1.0),
            "Sharpe": float(sharpe),
            "MaxDrawdown": float(mdd),
            "AnnualVol": float(sigma),
            "WinRate": float((strat_ret > 0).mean())
        }
    }
