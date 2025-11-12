# ============================================================
# api_server.py — DeepCNN_Trading (OHLCV → features → predict)
# Carga modelo desde Registry o runs:/ y levanta feature_stats.json del mismo run
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

# === Utilidades de features/normalización (tuyas) ===
from features_auto import make_features_auto              # genera indicadores desde OHLCV
from features_pipeline import apply_normalizer_from_stats # aplica stats de TRAIN
from indicators import WINDOWS                            # mismo set de ventanas del training

# ------------------------------------------------------------
# Config / paths / globals
# ------------------------------------------------------------
app = FastAPI(title="DeepCNN Trading API")

# Si usas Registry local, define en PowerShell:
#   $env:MLFLOW_TRACKING_URI = "file:///C:/Users/lenin/OneDrive/Escritorio/004/mlruns"
#   $env:MODEL_NAME = "CNN1D"
#   $env:MODEL_STAGE = "1"      # o "Production"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME  = os.getenv("MODEL_NAME", "CNN1D")
MODEL_REF   = os.getenv("MODEL_STAGE", "1")  # "1" (número) o "Production"/"Staging"

LATEST_URI_TXT = "latest_model_uri.txt"   # escrito por tu training_pipeline
SEQ_WINDOW = 60                           # ajusta si cambias tu ventana

MODEL: Optional[object] = None            # SavedModel (signature) o tf.keras.Model
MODEL_SOURCE: Optional[str] = None

FEATURE_NAMES: Optional[List[str]] = None
FEATURE_TYPES: Optional[Dict[str, str]] = None
NORM_STATS: Optional[dict] = None


# ------------------------------------------------------------
# Helpers numéricos / datos
# ------------------------------------------------------------
def _softmax_T(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / max(T, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)

def _ensure_batch_last_window(x_df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """
    Reordena columnas a feature_names, elimina NAs y devuelve la última
    ventana [SEQ_WINDOW, F] como batch [1, SEQ_WINDOW, F].
    """
    X = x_df.reindex(columns=feature_names).dropna(how="any")
    if len(X) < SEQ_WINDOW:
        raise ValueError(f"Se requieren al menos {SEQ_WINDOW} filas normalizadas; llegaron {len(X)}.")
    X_last = X.iloc[-SEQ_WINDOW:].values.astype(np.float32)
    return X_last[None, ...]  # [1, T, F]

def _download_ohlcv_yf(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str]   = None,
    lookback_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Descarga OHLCV diario con yfinance. Devuelve columnas ['Open','High','Low','Close','Volume'].
    """
    if lookback_days and not start and not end:
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=int(lookback_days))
    else:
        start_date = dt.date.fromisoformat(start) if start else (dt.date.today() - dt.timedelta(days=420))
        end_date   = dt.date.fromisoformat(end)   if end   else dt.date.today()

    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    if df is None or df.empty:
        raise ValueError(f"No se pudo descargar OHLCV para {ticker} en el rango {start_date}→{end_date}.")

    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' en los datos descargados.")
    return df[cols]


# ------------------------------------------------------------
# feature_stats.json (del mismo run)
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
    except Exception:
        return False

def _load_feature_stats_from_run(run_id: str) -> bool:
    # 1) descarga directa vía MLflow
    try:
        path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/feature_stats.json")
        if _load_feature_stats_from_path(path):
            print(f"[API] feature_stats.json cargado (MLflow, run_id={run_id})")
            return True
    except Exception:
        pass
    # 2) fallback: búsqueda en filesystem por si estás con file://
    for pat in [
        os.path.join("mlruns", "*", run_id, "artifacts", "feature_stats.json"),
        os.path.join("mlruns", "*", "*", run_id, "artifacts", "feature_stats.json"),
    ]:
        for p in glob.glob(pat):
            if _load_feature_stats_from_path(p):
                print(f"[API] feature_stats.json cargado (filesystem): {p}")
                return True
    return False


# ------------------------------------------------------------
# Resolver/cargar modelo
# ------------------------------------------------------------
def _resolve_model_paths(local_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Dado el artifact descargado, devuelve (savedmodel_dir, keras_file)
    """
    for cand in [os.path.join(local_dir, "data", "model"),
                 os.path.join(local_dir, "model"),
                 local_dir]:
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

def _resolve_registry_model(name: str, ref: str) -> Tuple[str, str]:
    """
    Devuelve (artifact_uri, run_id) para models:/name/ref
    ref puede ser número de versión ("1") o stage ("Production")
    """
    client = MlflowClient()
    if ref.isdigit():
        mv = client.get_model_version(name, ref)
    else:
        vers = client.get_latest_versions(name, [ref])
        if not vers:
            raise RuntimeError(f"No existe models:/{name}/{ref}")
        mv = vers[0]
    return mv.source, mv.run_id

def _try_load_from_registry() -> bool:
    global MODEL, MODEL_SOURCE
    try:
        source_uri, run_id = _resolve_registry_model(MODEL_NAME, MODEL_REF)
        local_dir = mlflow.artifacts.download_artifacts(source_uri)
        sm_dir, keras_file = _resolve_model_paths(local_dir)

        if sm_dir:
            MODEL = tf.saved_model.load(sm_dir)
            MODEL_SOURCE = f"models:/{MODEL_NAME}/{MODEL_REF}"
        elif keras_file:
            MODEL = tf.keras.models.load_model(keras_file, compile=False)
            MODEL_SOURCE = f"models:/{MODEL_NAME}/{MODEL_REF}"
        else:
            raise RuntimeError("Artifact del Registry no contiene SavedModel ni .keras/.h5")

        # feature_stats del run
        if not _load_feature_stats_from_run(run_id):
            print(f"[API][WARN] No se encontró feature_stats.json (run_id={run_id})")
        print(f"[API] Modelo cargado desde Registry: {MODEL_SOURCE}")
        return True
    except Exception as e:
        print(f"[API][WARN] Registry falló: {e}")
        return False

def _try_load_from_runs_txt() -> bool:
    global MODEL, MODEL_SOURCE
    if not os.path.exists(LATEST_URI_TXT):
        return False
    try:
        with open(LATEST_URI_TXT, "r", encoding="utf-8") as f:
            runs_uri = f.read().strip()
        local_dir = mlflow.artifacts.download_artifacts(runs_uri)
        sm_dir, keras_file = _resolve_model_paths(local_dir)

        if sm_dir:
            MODEL = tf.saved_model.load(sm_dir)
            MODEL_SOURCE = runs_uri
        elif keras_file:
            MODEL = tf.keras.models.load_model(keras_file, compile=False)
            MODEL_SOURCE = runs_uri
        else:
            raise RuntimeError("Artifact del run no contiene SavedModel ni .keras/.h5")

        # intenta deducir run_id de la URI runs:/
        m = re.match(r"^runs:/([^/]+)/model/?$", runs_uri)
        if m:
            _load_feature_stats_from_run(m.group(1))

        print(f"[API] Modelo cargado desde runs URI: {MODEL_SOURCE}")
        return True
    except Exception as e:
        print(f"[API][WARN] runs:/ falló: {e}")
        return False

def _load_model():
    """
    Orden:
      1) Registry (si MLFLOW_TRACKING_URI está definido)
      2) runs:/.../model (latest_model_uri.txt)
      3) fallback local .keras/.h5
    """
    global MODEL, MODEL_SOURCE, FEATURE_NAMES, FEATURE_TYPES, NORM_STATS
    MODEL = None
    MODEL_SOURCE = None
    FEATURE_NAMES = FEATURE_TYPES = NORM_STATS = None

    use_registry = MLFLOW_TRACKING_URI.startswith(("file://", "http://", "https://"))
    if use_registry and _try_load_from_registry():
        return
    if _try_load_from_runs_txt():
        return

    # Fallback local
    for candidate in ("best_model.keras", "best_cnn.h5"):
        if os.path.exists(candidate):
            try:
                MODEL = tf.keras.models.load_model(candidate, compile=False)
                MODEL_SOURCE = candidate
                print(f"[API] Cargado modelo local: {MODEL_SOURCE}")
                return
            except Exception as e:
                print(f"[API][WARN] Local {candidate} falló: {e}")

    print("[API][ERROR] No se pudo cargar ningún modelo.")

# Carga inicial
_load_model()


# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
class OHLCVRow(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Date: Optional[str] = None  # opcional

class PredictOHLCVRequest(BaseModel):
    ohlcv: List[OHLCVRow]

class PredictTickerRequest(BaseModel):
    ticker: str
    lookback_days: Optional[int] = 420
    start: Optional[str] = None   # "YYYY-MM-DD"
    end:   Optional[str] = None


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "DeepCNN Trading API",
        "endpoints": {
            "GET /health": "estado del modelo y stats",
            "GET /schema": "metadata de features/ventana",
            "POST /reload": "recarga modelo y stats",
            "POST /predict-ohlcv": "body {'ohlcv': [{Open,High,Low,Close,Volume}, ...]} (mín. 60 filas)",
            "POST /predict-ticker": "body {'ticker':'MSFT','lookback_days':420} o {'ticker','start','end'}",
        },
    }

@app.get("/health")
def health():
    return {
        "ok": MODEL is not None,
        "model_source": MODEL_SOURCE,
        "has_feature_stats": FEATURE_NAMES is not None and FEATURE_TYPES is not None and NORM_STATS is not None,
        "seq_window": SEQ_WINDOW,
        "n_features_expected": (len(FEATURE_NAMES) if FEATURE_NAMES else None),
    }

@app.get("/schema")
def schema():
    return {
        "feature_names": FEATURE_NAMES,
        "feature_types": FEATURE_TYPES,
        "seq_window": SEQ_WINDOW,
        "n_features_expected": (len(FEATURE_NAMES) if FEATURE_NAMES else None),
    }

@app.post("/reload")
def reload_model():
    _load_model()
    return {
        "ok": MODEL is not None,
        "model_source": MODEL_SOURCE,
        "has_feature_stats": FEATURE_NAMES is not None and FEATURE_TYPES is not None and NORM_STATS is not None,
    }

@app.post("/predict-ohlcv")
def predict_ohlcv(req: PredictOHLCVRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    if FEATURE_NAMES is None or FEATURE_TYPES is None or NORM_STATS is None:
        raise HTTPException(status_code=500, detail="Faltan feature_stats (feature_names/feature_types/norm_stats).")

    # 1) DataFrame OHLCV
    rows = [r.model_dump() for r in req.ohlcv]
    df = pd.DataFrame(rows, columns=["Open","High","Low","Close","Volume","Date"]).drop(columns=["Date"], errors="ignore")

    # 2) Features
    feats_df, _, _ = make_features_auto(df, windows=WINDOWS)

    # 3) Normalizar con stats del TRAIN y alinear columnas
    feats_df = feats_df.reindex(columns=FEATURE_NAMES).dropna(how="any")
    norm_df  = apply_normalizer_from_stats(feats_df, FEATURE_TYPES, NORM_STATS)

    # 4) Ventana y predicción
    X = _ensure_batch_last_window(norm_df, FEATURE_NAMES)
    try:
        if hasattr(MODEL, "signatures"):  # SavedModel
            sig = MODEL.signatures.get("serving_default") or next(iter(MODEL.signatures.values()))
            in_name = list(sig.structured_input_signature[1].keys())[0]
            out = sig(**{in_name: tf.convert_to_tensor(X, dtype=tf.float32)})
            logits = next(iter(out.values())).numpy()
        else:  # Keras
            logits = MODEL.predict(X, verbose=0)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error en inferencia: {str(e)}"})

    proba = _softmax_T(logits, T=1.0)
    yhat = proba.argmax(axis=-1).tolist()
    return {"yhat": yhat, "proba": proba.tolist(), "window_len": SEQ_WINDOW, "n_features": X.shape[-1]}

@app.post("/predict-ticker")
def predict_for_ticker(req: PredictTickerRequest):
    """
    1) yfinance → OHLCV
    2) mismas features del training
    3) normaliza con stats del TRAIN y alinea columnas
    4) última ventana [SEQ_WINDOW, F] → predicción
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    if FEATURE_NAMES is None or FEATURE_TYPES is None or NORM_STATS is None:
        raise HTTPException(status_code=500, detail="Faltan feature_stats (feature_names/feature_types/norm_stats).")

    # 1) OHLCV
    try:
        ohlcv = _download_ohlcv_yf(req.ticker, req.start, req.end, req.lookback_days)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al descargar OHLCV: {e}")

    # 2) Features
    try:
        feats_df, _, _ = make_features_auto(ohlcv, windows=WINDOWS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al construir features: {e}")

    # 3) Normalización y alineación
    try:
        feats_df = feats_df.reindex(columns=FEATURE_NAMES).dropna(how="any")
        norm_df  = apply_normalizer_from_stats(feats_df, FEATURE_TYPES, NORM_STATS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al normalizar/alinear features: {e}")

    # 4) Ventana
    try:
        X_last = _ensure_batch_last_window(norm_df, FEATURE_NAMES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo formar la ventana: {e}")

    # 5) Predicción
    try:
        if hasattr(MODEL, "signatures"):  # SavedModel
            sig = MODEL.signatures.get("serving_default") or next(iter(MODEL.signatures.values()))
            in_name = list(sig.structured_input_signature[1].keys())[0]
            out = sig(**{in_name: tf.convert_to_tensor(X_last, dtype=tf.float32)})
            logits = next(iter(out.values())).numpy()
        else:
            logits = MODEL.predict(X_last, verbose=0)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error en inferencia: {str(e)}"})

    proba = _softmax_T(logits, T=1.0).tolist()
    yhat = int(np.argmax(proba[0]))
    return {
        "ticker": req.ticker,
        "window_len": SEQ_WINDOW,
        "n_features": X_last.shape[-1],
        "yhat": yhat,
        "proba": proba[0]
    }

# ========= Cómo levantar =========
# .\venv\Scripts\Activate.ps1
# python -m uvicorn api_server:app --host 127.0.0.1 --port 9999 --reload
