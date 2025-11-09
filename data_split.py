# ============================================
# data_split.py
# ============================================
"""
Módulo para dividir datasets financieros en splits cronológicos 60/20/20.
Evita fugas temporales y garantiza índices ordenados tipo DatetimeIndex.
"""

import pandas as pd


def split_60_20_20(
    data: pd.DataFrame,
    cols_required=None,
    coerce_datetime=True,
    verbose=True
):
    """
    Divide un DataFrame cronológicamente en 3 subconjuntos (train, test, val).

    Parámetros
    ----------
    data : pd.DataFrame
        Dataset con índice temporal (DatetimeIndex preferido).
    cols_required : list[str], opcional
        Columnas que deben existir y no tener NaNs.
    coerce_datetime : bool
        Si True, convierte el índice a DatetimeIndex automáticamente.
    verbose : bool
        Si True, imprime tamaños y rangos de fechas.

    Retorna
    -------
    (train, test, val) : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tres subconjuntos cronológicos (sin solapamiento ni look-ahead).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Se esperaba un DataFrame en 'data'.")

    df = data.copy()

    # --- 1) Validar índice de fechas ---
    if not isinstance(df.index, pd.DatetimeIndex):
        if coerce_datetime:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError("No se pudo convertir el índice a DatetimeIndex.") from e
        else:
            raise ValueError("El índice debe ser DatetimeIndex.")

    # --- 2) Ordenar y limpiar duplicados ---
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # --- 3) Eliminar NaN en columnas requeridas ---
    if cols_required:
        faltan = [c for c in cols_required if c not in df.columns]
        if faltan:
            raise ValueError(f"Faltan columnas requeridas en 'data': {faltan}")
        df = df.dropna(subset=cols_required)

    # --- 4) Calcular divisiones ---
    n = len(df)
    if n < 50:
        raise ValueError(f"Demasiado pocas filas ({n}) para un split 60/20/20 robusto.")

    i_train_end = (60 * n) // 100
    i_test_end = (80 * n) // 100

    train = df.iloc[:i_train_end].copy()
    test = df.iloc[i_train_end:i_test_end].copy()
    val = df.iloc[i_test_end:].copy()

    # --- 5) Verificaciones de sanidad ---
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing
    assert val.index.is_monotonic_increasing
    assert len(train) + len(test) + len(val) == n

    # --- 6) Logs opcionales ---
    if verbose:
        print("Tamaños → train:", train.shape, "| test:", test.shape, "| val:", val.shape)
        print("Rangos:")
        print("  train:", train.index.min().date(), "→", train.index.max().date())
        print("  test :", test.index.min().date(),  "→", test.index.max().date())
        print("  val  :", val.index.min().date(),  "→", val.index.max().date())

    return train, test, val
