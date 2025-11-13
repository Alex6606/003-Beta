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
    """
    Initializes or resumes an MLflow experiment and starts a new run.
    Tracking is configured to use a local ./mlruns directory.
    """
    mlflow.set_tracking_uri("file:./mlruns")  # local tracking
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name)
    elif exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def end_mlflow_run():
    """Ends the active MLflow run."""
    mlflow.end_run()


def log_params(params: Dict[str, Any], prefix: str = ""):
    """
    Logs a dictionary of parameters to MLflow.
    Nested structures (dict/list/tuple) are serialized as JSON strings.
    """
    flat = {}
    for k, v in params.items():
        key = f"{prefix}{k}" if prefix else str(k)
        # avoid logging deeply nested large structures
        if isinstance(v, (dict, list, tuple)):
            flat[key] = json.dumps(v, default=str)
        else:
            flat[key] = v
    mlflow.log_params(flat)


def log_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Logs numerical metrics to MLflow.
    Values that cannot be converted to float are skipped.
    """
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
    """
    Saves a Python dictionary into a temporary JSON file and logs it as an MLflow artifact.
    """
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, artifact_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        mlflow.log_artifact(path)


# mlflow_tracking.py  -------------------------------------------
# mlflow_tracking.py  -------------------------------------------
def log_model_keras(model, artifact_path: str = "model"):
    """
    Logs a tf.keras.Model to MLflow as an MLflow Model.

    Workflow:
        1) Try mlflow.tensorflow.log_model (preferred)
        2) If it fails, save as a temporary SavedModel and retry logging
        3) If it still fails, store the full traceback as model_log.txt
    """
    import mlflow, tempfile, os, traceback
    err = []

    # Attempt 1: Use mlflow.tensorflow.log_model directly
    try:
        import tensorflow as tf
        import mlflow.tensorflow as mltf
        if isinstance(model, tf.keras.Model):
            mltf.log_model(model, artifact_path=artifact_path)
            return
    except Exception as e:
        err.append("[mlflow.tensorflow.log_model] " + repr(e))

    # Attempt 2: Save as SavedModel → reload → log model
    try:
        import tensorflow as tf
        import mlflow.tensorflow as mltf
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = os.path.join(td, "savedmodel")
            model.save(tmp_dir)  # save in TF SavedModel format
            reloaded = tf.keras.models.load_model(tmp_dir, compile=False)
            mltf.log_model(reloaded, artifact_path=artifact_path)
            return
    except Exception as e:
        err.append("[savedmodel→reload→log_model] " + repr(e))

    # Attempt 3: Log detailed error trace as an artifact
    try:
        mlflow.log_text(
            "Failed to log the model as an MLflow Model.\n\nTracebacks:\n" +
            "\n".join(err) + "\n\n" + traceback.format_exc(),
            "model_log.txt"
        )
    except Exception:
        pass
