# ============================================
# features_auto.py
# ============================================
"""
Automatic generator of technical indicators.
Combines the indicator functions from indicators.py applied to multiple windows.
"""

import numpy as np
import pandas as pd
from indicators import *  # import all base functions (rsi, atr, bollinger, etc.)
from indicators import (
    _roc, _rma, _logret_std, _rsi, _stoch_k_smooth, _williams_r_0_100, _cci, _trix,
    _dist_to_ma, _atr_with_method, _bollinger, _keltner_bandwidth,
    _adx_plus_minus_di, _obv, _vol_roc, _mfi, _cmf
)


# ===================== AUTO generator per window =====================
def make_features_auto(
        data: pd.DataFrame,
        windows=WINDOWS,
        max_per_indicator_per_window=None,
        rsi_methods=("RMA", "SMA"),
        stoch_smooth_k=(1, 3),
        stoch_d_windows=(3, 5),
        macd_triplets=((12, 26, 9), (8, 24, 9), (5, 35, 5)),
        atr_methods=("SMA", "RMA"),
        boll_k_values=(1.5, 2.0, 2.5),
        keltner_mults=(1.5, 2.0)
):
    """
    Automatically generates a feature set of technical indicators
    applied to different time windows.

    Returns:
        - features_df : DataFrame containing all generated feature columns.
        - feature_types : dict mapping each feature column to its scale type.
        - combos_usados : dict listing indicators grouped by family.
    """
    need = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not need.issubset(data.columns):
        raise ValueError(f"Missing required columns: {sorted(need - set(data.columns))}")

    close = data['Close']
    high = data['High']
    low = data['Low']
    vol = data['Volume']

    feats, ftype, combos_usados = {}, {}, {}

    def _register(name, series, typ, family_key):
        if series is None:
            return
        if name in feats:
            return
        feats[name] = series
        ftype[name] = typ
        combos_usados.setdefault(family_key, []).append(name)

    # -------- Momentum (per window) --------
    for w in windows:
        _register(f"ROC_{w}", _roc(close, w), "asset_scale", f"ROC@{w}")

        # RSI
        count = 0
        for method in rsi_methods:
            if method.upper() == "RMA":
                delta = close.diff()
                gain = delta.clip(lower=0.0)
                loss = -delta.clip(upper=0.0)
                avg_gain = _rma(gain, w)
                avg_loss = _rma(loss, w)
                rs = avg_gain / (avg_loss.replace(0, np.nan))
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = _rsi(close, w)

            _register(f"RSI_{method.upper()}_{w}", rsi, "bounded_0_100", f"RSI@{w}")
            count += 1
            if max_per_indicator_per_window and count >= max_per_indicator_per_window:
                break

        # Stochastic (K & D)
        count = 0
        for sk in stoch_smooth_k:
            for dw in stoch_d_windows:
                k_sm, d_sm = _stoch_k_smooth(close, high, low, w, smooth_k=sk, d_window=dw)
                _register(f"STOCHK_{w}_S{sk}", k_sm, "bounded_0_100", f"STOCH@{w}")
                _register(f"STOCHD_{w}_S{sk}_D{dw}", d_sm, "bounded_0_100", f"STOCH@{w}")
                count += 2
                if max_per_indicator_per_window and count >= max_per_indicator_per_window:
                    break
            if max_per_indicator_per_window and count >= max_per_indicator_per_window:
                break

        _register(f"WPR100_{w}", _williams_r_0_100(close, high, low, w), "bounded_0_100", f"WPR@{w}")
        _register(f"CCI_{w}", _cci(high, low, close, w), "asset_scale", f"CCI@{w}")
        _register(f"TRIX_{w}", _trix(close, w), "asset_scale", f"TRIX@{w}")
        _register(f"DIST_SMA_{w}", _dist_to_ma(close, w, "sma"), "asset_scale", f"DIST@{w}")
        _register(f"DIST_EMA_{w}", _dist_to_ma(close, w, "ema"), "asset_scale", f"DIST@{w}")

    # -------- MACD (does not depend on w) --------
    for (fast, slow, signal) in macd_triplets:
        ema_f = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_f - ema_s
        macd_sig = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        macd_hist = macd_line - macd_sig

        base = f"MACD_{fast}_{slow}_{signal}"
        _register(f"{base}_LINE", macd_line, "asset_scale", "MACD")
        _register(f"{base}_SIGNAL", macd_sig, "asset_scale", "MACD")
        _register(f"{base}_HIST", macd_hist, "asset_scale", "MACD")

    # -------- Volatility (per window) --------
    for w in windows:
        _register(f"LOGSTD_{w}", _logret_std(close, w), "asset_scale", f"LOGSTD@{w}")

        count = 0
        for m in atr_methods:
            _register(f"ATR_{m.upper()}_{w}", _atr_with_method(high, low, close, w, method=m),
                      "asset_scale", f"ATR@{w}")
            count += 1
            if max_per_indicator_per_window and count >= max_per_indicator_per_window:
                break

        count = 0
        for k in boll_k_values:
            pctb, width = _bollinger(close, w, k=k)
            _register(f"PCTB_{w}_K{k}", pctb, "bounded_0_1", f"BOLL@{w}")
            _register(f"BBWIDTH_{w}_K{k}", width, "asset_scale", f"BOLL@{w}")
            count += 2
            if max_per_indicator_per_window and count >= max_per_indicator_per_window:
                break

        count = 0
        for mult in keltner_mults:
            _register(f"KELT_BW_{w}_M{mult}", _keltner_bandwidth(close, high, low, w, mult=mult),
                      "asset_scale", f"KELT@{w}")
            count += 1
            if max_per_indicator_per_window and count >= max_per_indicator_per_window:
                break

        plus_di, minus_di, adx = _adx_plus_minus_di(high, low, close, w)
        _register(f"PLUS_DI_{w}", plus_di, "bounded_0_100", f"DMI@{w}")
        _register(f"MINUS_DI_{w}", minus_di, "bounded_0_100", f"DMI@{w}")
        _register(f"ADX_{w}", adx, "bounded_0_100", f"DMI@{w}")

    # -------- Volume --------
    _register("OBV", _obv(close, vol), "asset_scale", "OBV")
    for w in windows:
        _register(f"VROC_{w}", _vol_roc(vol, w), "asset_scale", f"VROC@{w}")
        _register(f"MFI_{w}", _mfi(high, low, close, vol, w), "bounded_0_100", f"MFI@{w}")
        _register(f"CMF_{w}", _cmf(high, low, close, vol, w), "bounded_-1_1", f"CMF@{w}")

    # -------- Final output --------
    features_df = pd.DataFrame(feats, index=data.index).dropna(how="any").copy()
    return features_df, ftype, combos_usados
