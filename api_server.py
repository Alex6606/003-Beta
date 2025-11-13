# ============================================================
# api_server.py — DeepCNN_Trading (OHLCV → features → predict)
# Loads model from Registry or runs:/ and loads feature_stats.json
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict

import os, re, json, glob
import numpy as np
import pandas as pd
import datetime as dt

import yfinance as yf
import tensorflow as tf
import mlflow
from mlflow.tracking import MlflowClient

# === User utilities ===
from features_auto import make_features_auto
from features_pipeline import apply_normalizer_from_stats
from indicators import WINDOWS
from backtest import y_to_signal

# ------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------
app = FastAPI(title="DeepCNN Trading API")

MIN_LOOKBACK_DAYS = 1000
SEQ_WINDOW = 60

MODEL_NAME  = os.getenv("MODEL_NAME", "CNN1D")
MODEL_REF   = os.getenv("MODEL_STAGE", "1")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

LATEST_URI_TXT = "latest_model_uri.txt"

MODEL = None
MODEL_SOURCE = None

FEATURE_NAMES = None
FEATURE_TYPES = None
NORM_STATS = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _softmax_T(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / max(T, 1e-6)
    z -= z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


def _ensure_batch_last_window(x_df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    X = x_df.reindex(columns=feature_names).dropna()
    if len(X) < SEQ_WINDOW:
        raise ValueError(f"At least {SEQ_WINDOW} normalized rows are required; received {len(X)}.")
    X_last = X.iloc[-SEQ_WINDOW:].values.astype(np.float32)
    return X_last[None, ...]


def _download_ohlcv_yf(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str]   = None,
    lookback_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Downloads daily OHLCV with a minimum of MIN_LOOKBACK_DAYS.
    Normalizes column names.
    """

    if start or end:
        end_date = dt.date.fromisoformat(end) if end else dt.date.today()
        start_date = dt.date.fromisoformat(start) if start else (end_date - dt.timedelta(days=MIN_LOOKBACK_DAYS))
    else:
        requested = max(lookback_days or MIN_LOOKBACK_DAYS, MIN_LOOKBACK_DAYS)
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=requested)

    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)

    if df is None or df.empty:
        raise ValueError(f"Could not download OHLCV for {ticker} in range {start_date} → {end_date}")

    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if low == "open": rename_map[col] = "Open"
        elif low == "high": rename_map[col] = "High"
        elif low == "low": rename_map[col] = "Low"
        elif low in ("close", "adj close", "adj_close"): rename_map[col] = "Close"
        elif low == "volume": rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    return df[required]


# ------------------------------------------------------------
# feature_stats.json loader
# ------------------------------------------------------------
def _load_feature_stats_from_path(path: str) -> bool:
    global FEATURE_NAMES, FEATURE_TYPES, NORM_STATS
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        FEATURE_NAMES = data.get("feature_names")
        FEATURE_TYPES = data.get("feature_types")
        NORM_STATS    = data.get("norm_stats")
        return all([FEATURE_NAMES, FEATURE_TYPES, NORM_STATS])
    except:
        return False


def _load_feature_stats_from_run(run_id: str) -> bool:
    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="feature_stats.json")
        if _load_feature_stats_from_path(path):
            print(f"[API] feature_stats.json loaded (run_id={run_id})")
            return True
    except Exception as e:
        print(f"[API][WARN] No feature_stats.json via MLflow: {e}")

    for pat in [
        os.path.join("mlruns", "*", run_id, "artifacts", "feature_stats.json"),
        os.path.join("mlruns", "*", "*", run_id, "artifacts", "feature_stats.json"),
    ]:
        for p in glob.glob(pat):
            if _load_feature_stats_from_path(p):
                print(f"[API] feature_stats.json loaded (filesystem): {p}")
                return True

    return False


# ------------------------------------------------------------
# Model loading (Registry / runs / local)
# ------------------------------------------------------------
def _resolve_registry_model(name: str, ref: str):
    client = MlflowClient()
    if ref.isdigit():
        mv = client.get_model_version(name, ref)
    else:
        vers = client.get_latest_versions(name, [ref])
        if not vers:
            raise RuntimeError(f"models:/{name}/{ref} does not exist")
        mv = vers[0]
    return mv.source, mv.run_id


def _resolve_model_paths(local_dir: str):
    for cand in [os.path.join(local_dir, "data", "model"), os.path.join(local_dir, "model"), local_dir]:
        if os.path.isdir(cand) and (
            os.path.exists(os.path.join(cand, "saved_model.pb")) or
            os.path.exists(os.path.join(cand, "saved_model.pbtxt"))
        ):
            return cand, None
    for root, _, files in os.walk(local_dir):
        for fn in files:
            if fn.endswith(".keras") or fn.endswith(".h5"):
                return None, os.path.join(root, fn)
    return None, None


def _try_load_from_registry() -> bool:
    global MODEL, MODEL_SOURCE
    try:
        source_uri, run_id = _resolve_registry_model(MODEL_NAME, MODEL_REF)
        local_dir = mlflow.artifacts.download_artifacts(source_uri)

        sm_dir, keras_file = _resolve_model_paths(local_dir)

        if sm_dir:
            MODEL = tf.saved_model.load(sm_dir)
        elif keras_file:
            MODEL = tf.keras.models.load_model(keras_file, compile=False)
        else:
            raise RuntimeError("Artifact does not contain SavedModel or .keras/.h5")

        MODEL_SOURCE = f"models:/{MODEL_NAME}/{MODEL_REF}"
        _load_feature_stats_from_run(run_id)
        print(f"[API] Model loaded from Registry: {MODEL_SOURCE}")
        return True

    except Exception as e:
        print(f"[API][WARN] Registry load failed: {e}")
        return False


def _try_load_from_runs_txt() -> bool:
    global MODEL, MODEL_SOURCE
    if not os.path.exists(LATEST_URI_TXT):
        return False

    try:
        with open(LATEST_URI_TXT, "r") as f:
            runs_uri = f.read().strip()

        local_dir = mlflow.artifacts.download_artifacts(runs_uri)
        sm_dir, keras_file = _resolve_model_paths(local_dir)

        if sm_dir:
            MODEL = tf.saved_model.load(sm_dir)
        else:
            MODEL = tf.keras.models.load_model(keras_file, compile=False)

        MODEL_SOURCE = runs_uri

        m = re.match(r"^runs:/([^/]+)/model/?$", runs_uri)
        if m:
            _load_feature_stats_from_run(m.group(1))

        print(f"[API] Model loaded from runs URI: {MODEL_SOURCE}")
        return True

    except Exception as e:
        print(f"[API][WARN] runs:/ load failed: {e}")
        return False


def _load_model():
    global MODEL, MODEL_SOURCE, FEATURE_NAMES, FEATURE_TYPES, NORM_STATS
    MODEL = MODEL_SOURCE = FEATURE_NAMES = FEATURE_TYPES = NORM_STATS = None

    if MLFLOW_TRACKING_URI.startswith(("file://", "http://", "https://")):
        if _try_load_from_registry():
            return

    if _try_load_from_runs_txt():
        return

    for candidate in ("best_model.keras", "best_cnn.h5"):
        if os.path.exists(candidate):
            try:
                MODEL = tf.keras.models.load_model(candidate, compile=False)
                MODEL_SOURCE = candidate
                print(f"[API] Local model loaded: {candidate}")
                return
            except Exception as e:
                print(f"[API][WARN] Local {candidate} failed: {e}")

    print("[API][ERROR] Could not load any model.")

_load_model()


# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
class PredictTickerRequest(BaseModel):
    ticker: str
    lookback_days: Optional[int] = None
    start: Optional[str] = None
    end:   Optional[str] = None


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "DeepCNN Trading API",
        "endpoints": {
            "GET /health": "model status",
            "GET /schema": "feature metadata",
            "POST /reload": "reload model",
            "POST /predict-ticker": "predict trading signal using yfinance OHLCV",
        },
        "min_lookback_days": MIN_LOOKBACK_DAYS
    }


@app.get("/health")
def health():
    return {
        "ok": MODEL is not None,
        "model_source": MODEL_SOURCE,
        "has_feature_stats": FEATURE_NAMES is not None,
        "seq_window": SEQ_WINDOW,
        "n_features_expected": len(FEATURE_NAMES) if FEATURE_NAMES else None,
        "min_lookback_days": MIN_LOOKBACK_DAYS,
    }


@app.get("/schema")
def schema():
    return {
        "feature_names": FEATURE_NAMES,
        "feature_types": FEATURE_TYPES,
        "seq_window": SEQ_WINDOW,
        "min_lookback_days": MIN_LOOKBACK_DAYS,
        "n_features_expected": len(FEATURE_NAMES) if FEATURE_NAMES else None,
    }


@app.post("/reload")
def reload_model():
    _load_model()
    return {
        "ok": MODEL is not None,
        "model_source": MODEL_SOURCE,
        "loaded_feature_stats": FEATURE_NAMES is not None,
    }


# ------------------------------------------------------------
# MAIN PREDICTION
# ------------------------------------------------------------
@app.post("/predict-ticker")
def predict_for_ticker(req: PredictTickerRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if not (FEATURE_NAMES and FEATURE_TYPES and NORM_STATS):
        raise HTTPException(status_code=500, detail="Missing feature_stats from training.")

    # Final lookback used
    lookback_used = max(req.lookback_days or MIN_LOOKBACK_DAYS, MIN_LOOKBACK_DAYS)

    # 1) Download OHLCV
    try:
        ohlcv = _download_ohlcv_yf(
            req.ticker, req.start, req.end,
            lookback_days=lookback_used
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OHLCV download error: {e}")

    # 2) Features
    try:
        feats_df, _, _ = make_features_auto(ohlcv, windows=WINDOWS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature construction error: {e}")

    # 3) Normalize
    try:
        feats_df = feats_df.reindex(columns=FEATURE_NAMES).dropna()
        norm_df  = apply_normalizer_from_stats(feats_df, FEATURE_TYPES, NORM_STATS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normalization error: {e}")

    # 4) Last window
    try:
        X_last = _ensure_batch_last_window(norm_df, FEATURE_NAMES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Window formation error: {e}")

    # 5) Inference
    try:
        if hasattr(MODEL, "signatures"):
            sig = MODEL.signatures.get("serving_default") or next(iter(MODEL.signatures.values()))
            in_name = list(sig.structured_input_signature[1].keys())[0]
            logits = next(iter(sig(**{
                in_name: tf.convert_to_tensor(X_last, dtype=tf.float32)
            }).values())).numpy()
        else:
            logits = MODEL.predict(X_last, verbose=0)
    except Exception as e:
        return {"ok": False, "error": f"Inference error: {e}"}

    # 6) Output
    proba = _softmax_T(logits)[0]
    yhat = int(np.argmax(proba))
    signal = int(y_to_signal(np.array([yhat]))[0])
    class_labels = ["SHORT", "HOLD", "LONG"]

    return {
        "ok": True,
        "ticker": req.ticker,
        "lookback_used": lookback_used,
        "class_index": yhat,
        "class_label": class_labels[yhat],
        "signal": signal,
        "proba": {
            "SHORT": float(proba[0]),
            "HOLD":  float(proba[1]),
            "LONG":  float(proba[2]),
        },
        "window_len": SEQ_WINDOW,
        "n_features": X_last.shape[-1],
    }


# ========= How to run =========
# .\venv\Scripts\Activate.ps1
# uvicorn api_server:app --host 127.0.0.1 --port 9999 --reload
