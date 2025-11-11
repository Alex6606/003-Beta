import mlflow as mf
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("DeepCNN_Trading")

# ============================================================
# mlflow_tracking.py
# ============================================================
from typing import Dict, Any
import json
import tempfile
import os
import mlflow

def start_mlflow_run(experiment_name: str, run_name: str = None, tags: Dict[str, str] = None):
    mlflow.set_tracking_uri("file:./mlruns")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, tags=tags or {})

def end_mlflow_run():
    mlflow.end_run()

def log_params(params: Dict[str, Any], prefix: str = ""):
    flat = {}
    for k, v in params.items():
        key = f"{prefix}{k}" if prefix else str(k)
        # evita estructuras gigantes anidadas
        if isinstance(v, (dict, list, tuple)):
            flat[key] = json.dumps(v, default=str)
        else:
            flat[key] = v
    mlflow.log_params(flat)

def log_metrics(metrics: Dict[str, float], prefix: str = ""):
    metr = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix else str(k)
        try:
            metr[key] = float(v)
        except Exception:
            continue
    if metr:
        mlflow.log_metrics(metr)

def log_artifacts_dict(data: Dict[str, Any], artifact_name: str = "artifacts.json"):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, artifact_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        mlflow.log_artifact(path)

def log_model_keras(model, artifact_path: str = "model"):
    try:
        import mlflow.keras as mkeras
        mkeras.log_model(keras_model=model, artifact_path=artifact_path)
    except Exception:
        # fallback genérico
        mlflow.log_text("Fallo al loggear con mlflow.keras", "model_log.txt")
