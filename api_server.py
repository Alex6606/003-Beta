# ============================================================
# api_server.py — FastAPI para /predict
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI(title="DeepCNN Trading API")

# Carga tu modelo Keras entrenado (ajusta la ruta real)
MODEL_PATH = "best_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

class PredictRequest(BaseModel):
    X: list  # secuencia [T, F] o batch [B, T, F]

def _ensure_batch(x):
    x = np.array(x, dtype=np.float32)
    if x.ndim == 2:  # [T, F] -> [1, T, F]
        x = x[None, ...]
    return x

def softmax_T(logits, T: float = 1.0):
    z = logits / max(T, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)

@app.post("/predict")
def predict(req: PredictRequest):
    X = _ensure_batch(req.X)
    logits = model.predict(X, verbose=0)
    proba = softmax_T(logits, T=1.0)
    yhat = proba.argmax(axis=-1).tolist()
    return {"proba": proba.tolist(), "yhat": yhat}
