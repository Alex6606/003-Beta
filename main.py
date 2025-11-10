from data_utils import get_data
from features_pipeline import build_and_normalize_features_per_split
from indicators import WINDOWS
from labeling import build_labels_for_feature_splits
from training_pipeline import train_eval_from_raw, summarize_run, sanity_check_from_res


if __name__ == "__main__":
    print("=== Descargando datos ===")
    data = get_data("MSFT")  # <-- contiene la columna 'Close'

    print("\n=== Construyendo features y normalizando ===")
    bundle = build_and_normalize_features_per_split(data, windows=WINDOWS, warmup=200)

    feat_train_n = bundle["norm"]["train"]
    feat_test_n  = bundle["norm"]["test"]
    feat_val_n   = bundle["norm"]["val"]

    print("\n=== Construyendo etiquetas ===")
    labels_bundle = build_labels_for_feature_splits(
        data_ohlcv=data,
        feat_train_n=feat_train_n,
        feat_test_n=feat_test_n,
        feat_val_n=feat_val_n,
        price_col="Close",
        horizon=3,
        threshold=0.01
    )

    X_train, y_train = labels_bundle["X"]["train"], labels_bundle["y"]["train"]
    X_test,  y_test  = labels_bundle["X"]["test"],  labels_bundle["y"]["test"]
    X_val,   y_val   = labels_bundle["X"]["val"],   labels_bundle["y"]["val"]

    # --- NEW: índices de los splits para backtesting ---
    idx_train = feat_train_n.index
    idx_test  = feat_test_n.index
    idx_val   = feat_val_n.index

    # --- Serie de precios de cierre (para backtest) ---
    close_series = data["Close"]

    print("\n=== Entrenando modelo CNN-1D ===")
    res = train_eval_from_raw(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        cw_train_cb={0: 1.83, 1: 0.968, 2: 1.012},
        gamma=1.2,
        shrink_lambda=0.877,
        verbose=1,
        # 👇 parámetros nuevos para el backtest
        close_series=close_series,
        train_idx=idx_train,
        val_idx=idx_val,
        test_idx=idx_test
    )

    print("\n=== Resultados ===")
    summarize_run(res)
    sanity_check_from_res(res)

    if "backtest" in res:
        print("\n=== Métricas de Backtest ===")
        for split, bt in res["backtest"].items():
            print(f"[{split.upper()}] Return={bt['final_return']:.3%}, Sharpe={bt['sharpe']:.2f}")
