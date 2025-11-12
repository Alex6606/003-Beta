# ============================================================
# mlflow_tracking.py
# ============================================================
from typing import Dict, Any
import json
import tempfile
import os
import mlflow
from mlflow.tracking import MlflowClient

def start_mlflow_run(experiment_name: str, run_name: str = None, tags=None):
    mlflow.set_tracking_uri("file:./mlruns")  # tracking local
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name)
    elif exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

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

# mlflow_tracking.py  -------------------------------------------
# mlflow_tracking.py  -------------------------------------------
def log_model_keras(model, artifact_path: str = "model"):
    """
    Loggea un tf.keras.Model como MLflow Model.
    1) mlflow.tensorflow.log_model (preferido)
    2) Si falla, guarda SavedModel temporal y reintenta
    3) Si aun falla, deja stack en model_log.txt
    """
    import mlflow, tempfile, os, traceback
    err = []

    # Intento 1: tf.keras → mlflow.tensorflow
    try:
        import tensorflow as tf
        import mlflow.tensorflow as mltf
        if isinstance(model, tf.keras.Model):
            mltf.log_model(model, artifact_path=artifact_path)
            return
    except Exception as e:
        err.append("[mlflow.tensorflow.log_model] " + repr(e))

    # Intento 2: SavedModel temporal → log_model
    try:
        import tensorflow as tf
        import mlflow.tensorflow as mltf
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = os.path.join(td, "savedmodel")
            model.save(tmp_dir)  # guarda en formato TF SavedModel
            reloaded = tf.keras.models.load_model(tmp_dir, compile=False)
            mltf.log_model(reloaded, artifact_path=artifact_path)
            return
    except Exception as e:
        err.append("[savedmodel→reload→log_model] " + repr(e))

    # Intento 3: mensaje de error detallado en artifacts
    try:
        mlflow.log_text(
            "Fallo al loggear el modelo como MLflow Model.\n\nTracebacks:\n" +
            "\n".join(err) + "\n\n" + traceback.format_exc(),
            "model_log.txt"
        )
    except Exception:
        pass


