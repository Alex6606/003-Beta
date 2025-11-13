# ============================================
# training_pipeline.py
# ============================================
"""
Pipeline completo de entrenamiento y evaluación para el modelo CNN-1D.
Incluye:
 - Inferencia de priors, alphas y objetivos KL
 - Entrenamiento two-phase (CE + Focal)
 - Calibración por temperatura
 - Ajuste de umbrales y biases
 - Evaluación final (TEST/VAL)
"""
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# 👇 añadido para usar mlflow.active_run() y construir model_uri
import mlflow

from backtest import y_to_signal, backtest_advanced
from calibration import find_temperature, softmax_T
from cnn_model import build_cnn_1d_logits
from mlflow_tracking import (
    log_artifacts_dict,
    log_model_keras,
    end_mlflow_run,
    log_metrics,
    log_params,
    start_mlflow_run,
)
from prior_correction import bbse_prior_shift_soft, search_logit_biases
from sequence_builder import build_cnn_sequences_for_splits
from threshold_tuning import tune_thresholds_by_class, coordinate_ascent_thresholds, apply_thresholds
from trainer import train_two_phase_v4


# === Config común ===
CLASSES = [0, 1, 2]


# === Métricas y utilidades ===
def _metrics_and_confusion(y_true, y_pred) -> Tuple[Dict[str, float], np.ndarray]:
    """Calcula macroF1, accuracy y matriz de confusión ordenada."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    m = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "acc":      float(accuracy_score(y_true, y_pred)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return m, cm


def _pred_dist(y_pred):
    n = len(y_pred)
    return {c: float((y_pred == c).sum()) / n for c in CLASSES}


def _print_cm_block(tag, cm, pred_dist=None):
    print(tag)
    if pred_dist is not None:
        items = ", ".join(f"{int(k)}: {pred_dist[k]:.3f}" for k in CLASSES)
        print(f"pred_dist={{ {items} }}")
    print(cm)


def _print_metrics_block(tag: str, m: Dict[str, float]):
    print(f"{tag} macroF1={m['macro_f1']:.4f}  acc={m['acc']:.4f}")


# === Priors / alphas ===
def _infer_priors_and_alphas(ytr_seq, cw_train_cb, beta=0.995, tau_alpha=0.90):
    """
    Estima:
      - pi_train (priors empíricos)
      - alphas (para Focal)
      - prior_target (mezcla 70% priors + 30% uniforme)
    """
    ytr_seq = np.asarray(ytr_seq).ravel()
    n_classes = int(np.max(ytr_seq)) + 1
    counts = np.bincount(ytr_seq, minlength=n_classes).astype(float)

    pi_train = counts / counts.sum()

    effective = (1 - beta) / (1 - np.power(beta, np.maximum(counts, 1.0)))
    inv_eff = 1.0 / np.maximum(effective, 1e-8)
    p = inv_eff / np.maximum(inv_eff.sum(), 1e-8)

    alphas = (p ** (1.0 / tau_alpha)).astype(np.float32)
    alphas /= np.maximum(alphas.sum(), 1e-8)

    prior_target = (0.7 * pi_train + 0.3 * (np.ones(n_classes) / n_classes)).astype(np.float32)
    return pi_train.astype(np.float32), alphas, prior_target


# === Wrapper de alto nivel ===
def train_eval_from_raw(
    X_train, y_train, X_val, y_val, X_test, y_test,
    cw_train_cb,
    pi_train=None, prior_target=None, alphas=None,
    gamma=1.2, seq_window=60, seq_step=1,
    epochs_warmup=18, epochs_finetune=12, batch_size=64,
    label_smoothing=0.02, lambda_kl=0.05, kl_temperature=1.5, tau_la=0.6,
    shrink_lambda=0.85, verbose=1,
    close_series=None, train_idx=None, val_idx=None, test_idx=None,
    feature_names=None
):
    """Construye secuencias y lanza entrenamiento/evaluación completa (con soporte para backtesting)."""
    seq_bundle = build_cnn_sequences_for_splits(
        X_train, y_train, X_test, y_test, X_val, y_val,
        window=seq_window, step=seq_step
    )

    Xtr_seq, ytr_seq = seq_bundle["train"]["X"], seq_bundle["train"]["y"]
    Xte_seq, yte_seq = seq_bundle["test"]["X"],  seq_bundle["test"]["y"]
    Xva_seq, yva_seq = seq_bundle["val"]["X"],   seq_bundle["val"]["y"]

    # Priors / alphas si faltan
    if (pi_train is None) or (prior_target is None) or (alphas is None):
        pi_train, alphas, prior_target = _infer_priors_and_alphas(ytr_seq, cw_train_cb)

    res = train_eval_one_config(
        Xtr_seq, ytr_seq, Xva_seq, yva_seq, Xte_seq, yte_seq,
        cw_train_cb=cw_train_cb,
        pi_train=pi_train,
        prior_target=prior_target,
        alphas=alphas,
        gamma=gamma,
        epochs_warmup=epochs_warmup,
        epochs_finetune=epochs_finetune,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        lambda_kl=lambda_kl,
        kl_temperature=kl_temperature,
        tau_la=tau_la,
        shrink_lambda=shrink_lambda,
        verbose=verbose,
        close_series=close_series,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_names=feature_names
    )
    return res


# === Entrenamiento, calibración y evaluación ===
def train_eval_one_config(
    Xtr_seq, ytr_seq, Xva_seq, yva_seq, Xte_seq, yte_seq,
    cw_train_cb: Dict[int, float],
    pi_train: np.ndarray,
    prior_target: np.ndarray,
    alphas: np.ndarray,
    gamma: float = 1.2,
    # arquitectura
    model_builder_kwargs: Dict[str, Any] = None,
    # entrenamiento
    epochs_warmup: int = 18, epochs_finetune: int = 12, batch_size: int = 64,
    label_smoothing: float = 0.02, lambda_kl: float = 0.05,
    kl_temperature: float = 1.5, tau_la: float = 0.6,
    # TILT / SHRINK / BIAS
    shrink_lambda: float = 0.85,
    bias_grids: Dict[str, np.ndarray] = None,
    # datos para backtest
    close_series=None,
    train_idx=None, val_idx=None, test_idx=None,
    # verbosidad
    verbose: int = 1,
    feature_names=None
) -> Dict[str, Any]:
    """
    Versión extendida con backtesting sobre train/val/test (usa backtest_advanced).
    """
    # --- Modelo base ---
    if model_builder_kwargs is None:
        model_builder_kwargs = dict(
            n_features=Xtr_seq.shape[-1], window=Xtr_seq.shape[1],
            filters=(128, 128, 64), kernels=(9, 5, 3), dilations=(1, 2, 4),
            residual=True, dropout=0.15, l2=5e-4, head_units=256,
            head_dropout=0.30, use_ln=True, output_bias=np.zeros(3, dtype=np.float32)
        )
    if bias_grids is None:
        bias_grids = dict(
            grid_d0=np.linspace(0.0, 0.20, 4),
            grid_d1=np.linspace(0.0, 0.20, 4),
            grid_d2=np.linspace(0.0, 0.25, 5),
        )

    model = build_cnn_1d_logits(**model_builder_kwargs)

    # --- Entrenamiento ---
    model, history = train_two_phase_v4(
        model,
        Xtr_seq, ytr_seq, Xva_seq, yva_seq,
        cw_train_cb=cw_train_cb,
        alphas_focal=alphas, gamma=gamma,
        pi_prior=pi_train,
        prior_target=prior_target,
        epochs_warmup=epochs_warmup,
        epochs_finetune=epochs_finetune,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        lambda_kl=lambda_kl,
        kl_temperature=kl_temperature,
        tau_la=tau_la,
        verbose=verbose,
        early_stopping=False,
        reduce_on_plateau=True
    )

    # --- Calibración (usar VAL) ---
    logits_tr   = model.predict(Xtr_seq, verbose=0)   # ✅ añadimos TRAIN para backtest
    logits_val  = model.predict(Xva_seq, verbose=0)
    logits_test = model.predict(Xte_seq, verbose=0)

    T = find_temperature(logits_val, yva_seq)
    if verbose:
        print(f"Temperature (VAL): {T:.3f}")

    proba_tr   = softmax_T(logits_tr,   T)
    proba_val  = softmax_T(logits_val,  T)
    proba_test = softmax_T(logits_test, T)

    # --- Threshold tuning (en VAL) ---
    thr0 = tune_thresholds_by_class(
        yva_seq, proba_val,
        metric_fn=lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    )
    thr_refined, best_val_macro = coordinate_ascent_thresholds(
        yva_seq, proba_val, thr0,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        rounds=2
    )
    if verbose:
        print("Umbrales base (VAL):", np.round(thr_refined, 3), "| macroF1_VAL:", round(best_val_macro, 4))

    # --- Prior shift (BBSE) para TEST ---
    pi_test_bbse, C = bbse_prior_shift_soft(
        pi_train, yva_seq, proba_val, proba_test,
        lam_ridge=1e-2, eps=1e-6
    )
    logit_shift     = np.log(np.clip(pi_test_bbse, 1e-6, 1.0)) - np.log(np.clip(pi_train, 1e-6, 1.0))
    logits_test_adj = logits_test + logit_shift.reshape(1, -1)
    proba_test_bbse = softmax_T(logits_test_adj, T)

    # --- Tilt thresholds (re-sintoniza thresholds en VAL con prior de TEST) ---
    w_ratio = (pi_test_bbse / np.clip(pi_train, 1e-6, 1.0)).reshape(1, -1)
    proba_val_tilt = proba_val * w_ratio
    proba_val_tilt /= proba_val_tilt.sum(axis=1, keepdims=True)
    thr0_tilt = tune_thresholds_by_class(
        yva_seq, proba_val_tilt,
        metric_fn=lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    )
    thr_refined_tilt, best_val_macro_tilt = coordinate_ascent_thresholds(
        yva_seq, proba_val_tilt, thr0_tilt,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        rounds=2
    )

    # --- Bias search (en VAL, sobre thresholds tilt) ---
    best_val_score, best_bias = search_logit_biases(
        logits_val, yva_seq, thr_refined_tilt, T=T,
        grid_d0=bias_grids["grid_d0"],
        grid_d1=bias_grids["grid_d1"],
        grid_d2=bias_grids["grid_d2"],
    )
    if verbose:
        print("Best bias on VAL:", tuple(float(f"{x:.3f}") for x in best_bias),
              " macroF1_VAL:", round(best_val_score, 4))

    # --- Pred FINAL en cada split ---
    # TEST: BBSE + bias + thresholds tilt
    logits_test_bbse_bias = logits_test_adj + np.array(best_bias, float).reshape(1, -1)
    proba_test_final      = softmax_T(logits_test_bbse_bias, T)
    yte_hat_final         = apply_thresholds(proba_test_final, thr_refined_tilt)

    # VAL: bias + thresholds tilt
    logits_val_bias = logits_val + np.array(best_bias, float).reshape(1, -1)
    proba_val_final = softmax_T(logits_val_bias, T)
    yva_hat_final   = apply_thresholds(proba_val_final, thr_refined_tilt)

    # TRAIN: bias + thresholds tilt (sin BBSE)
    logits_tr_bias = logits_tr + np.array(best_bias, float).reshape(1, -1)
    proba_tr_final = softmax_T(logits_tr_bias, T)
    ytr_hat_final  = apply_thresholds(proba_tr_final, thr_refined_tilt)

    # ==== Métricas y matrices TEST/VAL ====
    m_test, cm_test = _metrics_and_confusion(yte_seq, yte_hat_final)
    m_val,  cm_val  = _metrics_and_confusion(yva_seq, yva_hat_final)

    if verbose:
        print("\n[TEST] -------")
        _print_metrics_block("TEST (FINAL)", m_test)
        _print_cm_block("Matriz de confusión (TEST, FINAL):", cm_test, _pred_dist(yte_hat_final))

        print("\n[VAL] --------")
        _print_metrics_block("VAL  (FINAL)", m_val)
        _print_cm_block("Matriz de confusión (VAL,  FINAL):", cm_val, _pred_dist(yva_hat_final))

    # === Backtest AVANZADO sobre train/val/test ===
    backtest_results = {}
    if close_series is not None:
        for split_name, idx, y_pred in [
            ("train", train_idx, ytr_hat_final),
            ("val", val_idx, yva_hat_final),
            ("test", test_idx, yte_hat_final),
        ]:
            if idx is None or len(idx) == 0:
                continue

            # === Sincroniza longitudes ===
            n = min(len(idx), len(y_pred))
            aligned_idx = idx[-n:]
            aligned_signal = y_to_signal(y_pred[-n:])

            # === Corre el backtest avanzado ===
            bt = backtest_advanced(
                close=close_series.loc[aligned_idx],
                idx=aligned_idx,
                signal=aligned_signal,
                fee=0.00125,
                borrow_rate_annual=0.0025,
                freq=252,
                sl_pct=0.02,
                tp_pct=0.03
            )

            eq = bt["series"]["equity"]
            backtest_results[split_name] = {
                "final_return": float(eq.iloc[-1] - 1.0),
                "sharpe": bt["metrics"]["Sharpe"],
                "equity": eq.values.tolist(),
                "metrics": bt["metrics"],
            }

            print(f"[BACKTEST {split_name.upper()}] "
                  f"Final Return={backtest_results[split_name]['final_return']:.3f}, "
                  f"Sharpe={backtest_results[split_name]['sharpe']:.3f}")

    # --- Empaque de resultados (mantiene la forma que usa summarize_run/visualization) ---
    res = {
        "cfg": {
            "cw_train_cb": cw_train_cb,
            "alphas": alphas.tolist(),
            "gamma": gamma,
            "epochs": (epochs_warmup, epochs_finetune),
            "batch_size": batch_size,
            "label_smoothing": label_smoothing,
            "lambda_kl": lambda_kl,
            "kl_temperature": kl_temperature,
            "tau_la": tau_la,
            "shrink_lambda": shrink_lambda,
            "bias_grids": {k: list(v) for k, v in bias_grids.items()},
        },
        "history": history,
        "artifacts": {
            "T": float(T),
            "pi_train": pi_train.tolist(),
            "pi_test_bbse": pi_test_bbse.tolist(),
            "C_bbse": C.tolist(),
            "thr_refined": thr_refined.tolist(),
            "thr_refined_tilt": thr_refined_tilt.tolist(),
            "best_bias": [float(x) for x in best_bias],
        },
        "metrics": {
            "test": {"final": m_test},
            "val":  {"final": m_val},
        },
        "confusion": {
            "test": {"final": cm_test},
            "val":  {"final": cm_val},
        },
        "y_true_pred": {
            "train": (ytr_seq, ytr_hat_final),
            "val":   (yva_seq, yva_hat_final),
            "test":  (yte_seq, yte_hat_final),
        },
        "backtest": backtest_results,
    }
    run_name = f"CNN_win{Xtr_seq.shape[1]}_gamma{gamma}_sh{shrink_lambda}"
    start_mlflow_run(
        experiment_name="DeepCNN_Trading",
        run_name=run_name,
        tags={"stage": "train_eval", "model": "cnn1d"}
    )

    try:
        # 1) Hiperparámetros y config
        cfg_dict = {
            "gamma": gamma,
            "epochs_warmup": epochs_warmup,
            "epochs_finetune": epochs_finetune,
            "batch_size": batch_size,
            "label_smoothing": label_smoothing,
            "tau_la": tau_la,
            "lambda_kl": lambda_kl,
            "kl_temperature": kl_temperature,
            "shrink_lambda": shrink_lambda,
            "n_features": int(Xtr_seq.shape[-1]),
            "window": int(Xtr_seq.shape[1]),
            "architecture": model_builder_kwargs,
        }
        log_params(cfg_dict, prefix="train_")

        # 2) Métricas principales
        #   macroF1/acc de VAL y TEST finales (coinciden con res["metrics"])
        log_metrics({
            "val_macro_f1": float(m_val["macro_f1"]),
            "val_acc": float(m_val["acc"]),
            "test_macro_f1": float(m_test["macro_f1"]),
            "test_acc": float(m_test["acc"]),
        })

        # 3) Métricas de backtest si existen
        if "backtest" in locals() and isinstance(backtest_results, dict):
            for split, bt in backtest_results.items():
                mets = bt.get("metrics", {})
                # nombres estandarizados
                tolog = {}
                for mk in ["CAGR", "Sharpe", "MaxDrawdown", "AnnualVol", "WinRate"]:
                    if mk in mets:
                        tolog[f"bt_{split}_{mk}"] = float(mets[mk])
                if "final_return" in bt:
                    tolog[f"bt_{split}_FinalReturn"] = float(bt["final_return"])
                if tolog:
                    log_metrics(tolog)

        # 4) Artefactos relevantes
        art = {
        "T": float(T),
        "pi_train": pi_train.tolist(),
        "pi_test_bbse": pi_test_bbse.tolist(),
        "thr_refined": thr_refined.tolist(),
        "thr_refined_tilt": thr_refined_tilt.tolist(),
        "best_bias": [float(x) for x in best_bias],
        }
        log_artifacts_dict(art, artifact_name="artifacts.json")

        # 🔴 NUEVO: guarda el orden de columnas (‘feature_names’) si te lo mandaron
        if feature_names is not None:
            try:
                # usa directamente mlflow.log_dict por si tu helper no escribe el archivo
                import mlflow
                mlflow.log_dict({
                    "feature_names": list(map(str, feature_names)),
                    "feature_types": {f: "float" for f in feature_names},  # o tu dict real
                    "norm_stats": {f: {"mean": 0, "std": 1} for f in feature_names}  # si no guardas los reales
                }, "feature_stats.json")
                print("[MLflow] feature_stats.json loggeado.")
            except Exception as e:
                print("[WARN] No se pudo sloggear feature_stats.json:", e)
        # 5) Modelo (artefacto del run)
        log_model_keras(model, artifact_path="model")

        # 6) 🔴 NUEVO: construir y persistir el URI del modelo (sin Registry)
        import mlflow  # 👈 AÑADE ESTA LÍNEA AQUÍ
        active_run = mlflow.active_run()
        if active_run is not None:
            model_uri = f"runs:/{active_run.info.run_id}/model"
            print("[MLflow] Model URI:", model_uri)

            # lo guardamos como archivo de conveniencia para la API
            try:
                with open("latest_model_uri.txt", "w", encoding="utf-8") as f:
                    f.write(model_uri)
            except Exception as e:
                print("[WARN] No se pudo escribir latest_model_uri.txt:", e)

            # opcional: también dejarlo como artefacto JSON dentro del run
            try:
                log_artifacts_dict({"model_uri": model_uri}, artifact_name="model_uri.json")
            except Exception as e:
                print("[WARN] No se pudo loggear model_uri.json:", e)

    finally:
        end_mlflow_run()
    # === fin MLflow logging ===
    return res


def summarize_run(res: Dict[str, Any]):
    cfg  = res.get("cfg", {})
    art  = res.get("artifacts", {})
    mets = res.get("metrics", {})
    ytp  = res.get("y_true_pred", {})

    if "cw_train_cb" in cfg:
        print("class_weight:", cfg["cw_train_cb"])
    if "alphas" in cfg:
        print("alphas focal:", np.round(np.array(cfg["alphas"]), 6))
    if "pi_train" in art:
        print("pi_train:", np.round(np.array(art["pi_train"]), 3))
    if "T" in art:
        print("Temperature (VAL):", f"{art['T']:.3f}")
    if "pi_test_bbse" in art:
        print("pi_test_est (BBSE-soft):", np.round(np.array(art["pi_test_bbse"]), 3))
    if "thr_refined" in art:
        print("Umbrales base (VAL):", np.round(np.array(art["thr_refined"]), 3))
    if "thr_refined_tilt" in art:
        print("Umbrales tilt (VAL):", np.round(np.array(art["thr_refined_tilt"]), 3))
    if "best_bias" in art:
        print("Best bias on VAL:", tuple(np.round(np.array(art["best_bias"]), 3)))
    if "shrink_lambda" in cfg:
        print("shrink λ:", cfg["shrink_lambda"])

    # === TEST ===
    print("\n[TEST] -------")
    mt_final = mets.get("test", {}).get("final", {})
    _print_metrics_block("TEST (FINAL)", mt_final)
    y_true_te, y_pred_te = ytp.get("test", ([], []))

    # Si no hay confusion matrix en res, la generamos
    cm_test = confusion_matrix(y_true_te, y_pred_te, labels=[0, 1, 2])
    _print_cm_block("Matriz de confusión (TEST, FINAL):", cm_test, _pred_dist(y_pred_te))

    # === VAL ===
    print("\n[VAL] --------")
    mv_final = mets.get("val", {}).get("final", {})
    _print_metrics_block("VAL  (FINAL)", mv_final)
    y_true_va, y_pred_va = ytp.get("val", ([], []))
    cm_val = confusion_matrix(y_true_va, y_pred_va, labels=[0, 1, 2])
    _print_cm_block("Matriz de confusión (VAL, FINAL):", cm_val, _pred_dist(y_pred_va))

    print(f"\nF1-macro  TEST: {mt_final.get('macro_f1', 0):.3f} | VAL: {mv_final.get('macro_f1', 0):.3f}")
    print(f"Acc       TEST: {mt_final.get('acc', 0):.3f}   | VAL: {mv_final.get('acc', 0):.3f}")


def print_run_artifacts(art: Dict[str, Any]):
    print("\n[RUN ARGS/ARTIFACTS]")
    if 'cw_train_cb' in art:
        print(f"class_weight: {art['cw_train_cb']}")
    if 'alphas' in art:
        print("alphas focal:", np.round(np.array(art['alphas']), 6))
    if 'pi_train' in art:
        print("pi_train:", np.round(np.array(art['pi_train']), 3))
    if 'T' in art:
        print(f"Temperature (VAL): {float(art['T']):.3f}")
    if 'pi_test_bbse' in art:
        print("pi_test_est (BBSE-soft):", np.round(np.array(art['pi_test_bbse']), 3))
    if 'thr_refined' in art:
        print("Umbrales base (VAL):", np.round(np.array(art['thr_refined']), 3))
    if 'thr_refined_tilt' in art:
        print("Umbrales tilt (VAL):", np.round(np.array(art['thr_refined_tilt']), 3))
    if 'best_bias' in art:
        bb = np.array(art['best_bias'], dtype=float).tolist()
        print("Best bias on VAL:", tuple(bb))
    if 'shrink_lambda' in art:
        print(f"shrink λ: {art['shrink_lambda']}")


def _row_sums(a):
    return np.asarray(a).sum(axis=1).tolist()


def _supports(y):
    return [int((y == c).sum()) for c in CLASSES]


def sanity_check_from_res(res):
    """
    Verifica coherencia básica entre supports y matrices de confusión.
    Adaptada a la nueva estructura de 'y_true_pred' (tuplas directas).
    """
    ytp = res.get("y_true_pred", {})

    # === TEST ===
    if "test" in ytp:
        yte_true, yte_pred = ytp["test"]
        cm_test = confusion_matrix(yte_true, yte_pred, labels=[0, 1, 2])
        print("Supports TEST:", [int((yte_true == c).sum()) for c in [0, 1, 2]],
              "| Row sums TEST (FINAL):", cm_test.sum(axis=1).tolist())

    # === VAL ===
    if "val" in ytp:
        yva_true, yva_pred = ytp["val"]
        cm_val = confusion_matrix(yva_true, yva_pred, labels=[0, 1, 2])
        print("Supports VAL :", [int((yva_true == c).sum()) for c in [0, 1, 2]],
              "| Row sums VAL  (FINAL):", cm_val.sum(axis=1).tolist())


def collect_split_metrics(res):
    """
    Extrae métricas, matrices de confusión y distribuciones de señales
    para train/val/test sin imprimir nada.
    Ideal para reportes en main.py u otros módulos.
    """
    from sklearn.metrics import confusion_matrix
    out = {}

    if "y_true_pred" not in res:
        return out

    for split in ("train", "val", "test"):
        if split not in res["y_true_pred"]:
            continue

        y_true, y_pred = res["y_true_pred"][split]
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        # métricas
        m = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "acc": float(accuracy_score(y_true, y_pred)),
        }

        # distribución predicha
        pred_dist = {c: float((y_pred == c).sum()) / len(y_pred) for c in [0, 1, 2]}

        # señales (-1,0,1)
        signals = y_to_signal(y_pred)
        sig_dist = {s: float((signals == s).sum()) / len(signals) for s in [-1, 0, 1]}

        out[split] = {
            "metrics": m,
            "cm": cm,
            "pred_dist": pred_dist,
            "signal_dist": sig_dist,
            "n": len(y_pred),
        }

    return out
