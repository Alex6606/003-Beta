# ============================================
# indicators.py
# ============================================
"""
Módulo de indicadores técnicos y funciones auxiliares para generación de features.
Contiene funciones rolling basadas en precios OHLCV.
"""

import numpy as np
import pandas as pd


# === Ventanas estándar (puedes cambiar según tus pruebas) ===
WINDOWS = [5, 10, 20, 30, 50, 100, 150, 200]


# === Helpers de rolling ===
def _rolling_min_max(series, window):
    ll = series.rolling(window=window, min_periods=window).min()
    hh = series.rolling(window=window, min_periods=window).max()
    return ll, hh


# === Indicadores Momentum ===
def _rsi(close, window):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _roc(series, window):
    return series.pct_change(periods=window)


def _trix(close, window):
    ema1 = close.ewm(span=window, adjust=False, min_periods=window).mean()
    ema2 = ema1.ewm(span=window, adjust=False, min_periods=window).mean()
    ema3 = ema2.ewm(span=window, adjust=False, min_periods=window).mean()
    return ema3.pct_change()


# === Volatilidad ===
def _rma(series, window):
    return series.ewm(alpha=1.0 / window, adjust=False).mean()


def _logret_std(close, window):
    lr = np.log(close).diff()
    return lr.rolling(window, min_periods=window).std(ddof=0)


def _true_range(high, low, close):
    prev_close = close.shift(1)
    m1 = high - low
    m2 = (high - prev_close).abs()
    m3 = (low - prev_close).abs()
    return pd.concat([m1, m2, m3], axis=1).max(axis=1)


def _atr_with_method(high, low, close, window, method="SMA"):
    tr = _true_range(high, low, close)
    if method.upper() == "RMA":
        return _rma(tr, window)
    return tr.rolling(window, min_periods=window).mean()


def _bollinger(close, window, k=2.0):
    ma = close.rolling(window, window).mean()
    sd = close.rolling(window, window).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / ma.replace(0, np.nan)
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return pctb, width


def _keltner_bandwidth(close, high, low, window, mult=1.5):
    ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
    atr = _atr_with_method(high, low, close, window, method="SMA")
    upper = ema + mult * atr
    lower = ema - mult * atr
    return (upper - lower) / ema.replace(0, np.nan)


# === Volumen ===
def _obv(close, volume):
    delta = close.diff()
    dirn = np.sign(delta).fillna(0)
    return (dirn * volume.fillna(0)).cumsum()


def _vol_roc(volume, window):
    return volume.pct_change(periods=window)


def _mfi(high, low, close, volume, window):
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    up = (tp > tp.shift(1))
    dn = (tp < tp.shift(1))
    pos_mf = rmf.where(up, 0.0).rolling(window, window).sum()
    neg_mf = rmf.where(dn, 0.0).rolling(window, window).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def _cmf(high, low, close, volume, window):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    return mfv.rolling(window, window).sum() / volume.rolling(window, window).sum().replace(0, np.nan)


# === Tendencia y fuerza ===
def _adx_plus_minus_di(high, low, close, window):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = _true_range(high, low, close)
    tr_rma = _rma(tr, window)
    plus_dm_rma = _rma(plus_dm, window)
    minus_dm_rma = _rma(minus_dm, window)
    plus_di = 100 * (plus_dm_rma / tr_rma.replace(0, np.nan))
    minus_di = 100 * (minus_dm_rma / tr_rma.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = _rma(dx, window)
    return plus_di, minus_di, adx


def _cci(high, low, close, window):
    tp = (high + low + close) / 3.0
    ma = tp.rolling(window, window).mean()
    md = (tp - ma).abs().rolling(window, window).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


# === Osciladores ===
def _stoch_k_smooth(close, high, low, k_window, smooth_k=1, d_window=None):
    ll = low.rolling(k_window, k_window).min()
    hh = high.rolling(k_window, k_window).max()
    base_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k_sm = base_k.rolling(smooth_k, smooth_k).mean() if smooth_k and smooth_k > 1 else base_k
    if d_window and d_window > 1:
        d_sm = k_sm.rolling(d_window, d_window).mean()
        return k_sm, d_sm
    return k_sm, None


def _williams_r_0_100(close, high, low, window):
    ll = low.rolling(window, window).min()
    hh = high.rolling(window, window).max()
    rng = (hh - ll).replace(0, np.nan)
    willr = -100 * (hh - close) / rng
    return willr + 100  # [0,100]


def _dist_to_ma(close, window, kind="sma"):
    if kind == "sma":
        ma = close.rolling(window, window).mean()
    else:
        ma = close.ewm(span=window, adjust=False, min_periods=window).mean()
    return (close - ma) / ma.replace(0, np.nan)
