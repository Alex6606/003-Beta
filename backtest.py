# ==========================================================
# backtest.py
# ==========================================================
import numpy as np
import pandas as pd

# Mapea {0: short, 1: hold, 2: long} -> {-1, 0, +1}
def y_to_signal(y_pred: np.ndarray) -> pd.Series:
    mapping = {0: -1, 1: 0, 2: 1}
    sig = pd.Series([mapping[int(v)] for v in y_pred], dtype=int)
    return sig

def backtest_simple(
    close: pd.Series,
    idx_exec: pd.Index,     # Índices (fechas) de tus predicciones (TEST), idealmente seq_bundle["test"]["idx"]
    signals: pd.Series,     # Serie {-1,0,1} indexada igual que idx_exec
    commission=0.00125,     # 0.125%
    borrow_rate_annual=0.0025,  # 0.25% anual
    tz_days=252
) -> dict:
    """
    Backtest long/short/flat:
      - Entra al SIGUIENTE bar (shift(1)) para evitar look-ahead.
      - Costos de comisión cuando hay cambio de posición (turnover).
      - Borrow-fee diario cuando estás en short.
    """
    signals = signals.astype(int)
    signals.index = idx_exec

    # Alinear precios y calcular rendimientos diarios
    px = close.reindex(idx_exec).astype(float)
    ret = px.pct_change().fillna(0.0)

    # Sin look-ahead: ejecutas con la señal previa
    sig_exec = signals.shift(1).reindex(ret.index).fillna(0).astype(int)

    # Rendimiento bruto de la estrategia
    strat_gross = sig_exec * ret

    # Comisión cuando cambias de posición (|Δpos|)
    turns = sig_exec.diff().abs().fillna(0)
    # Costo proporcional ~ comisión * |Δpos| (asume 1x nocional)
    cost_comm = commission * turns

    # Borrow fee solo en posiciones short
    borrow_daily = borrow_rate_annual / tz_days
    cost_borrow = borrow_daily * (sig_exec == -1).astype(float)

    # Rendimiento neto
    strat_net = strat_gross - cost_comm - cost_borrow

    # Equity
    equity = (1.0 + strat_net).cumprod()

    # Métricas
    mu = strat_net.mean() * tz_days
    sd = strat_net.std(ddof=0) * np.sqrt(tz_days)
    sharpe = mu / (sd + 1e-12)

    downside = strat_net.copy()
    downside[downside > 0] = 0.0
    dd_std = downside.std(ddof=0) * np.sqrt(tz_days)
    sortino = mu / (dd_std + 1e-12)

    # Max Drawdown / Calmar
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    mdd = drawdown.min()
    # Asume retorno anualizado ~ mu (aprox), Calmar = mu / |mdd|
    calmar = (mu / abs(mdd)) if mdd < 0 else np.nan

    # Stats de trades (cambios de régimen)
    trades = int(turns.sum())
    win_rate = float((strat_net[strat_net != 0] > 0).mean()) if (strat_net != 0).any() else np.nan

    return {
        "series": {
            "ret_net": strat_net,
            "equity": equity,
            "signals_exec": sig_exec,
        },
        "metrics": {
            "ann_return": mu,
            "ann_vol": sd,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": float(mdd),
            "calmar": float(calmar),
            "trades": trades,
            "win_rate": win_rate,
        }
    }
