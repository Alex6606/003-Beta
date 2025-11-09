# ============================================
# cnn_model.py
# ============================================
"""
Módulo de construcción del modelo CNN-1D con bloques convolucionales
y residuales para clasificación multiclase.
"""

from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn_1d_logits(
    n_features,
    window,
    n_classes=3,
    filters=(128, 128, 64),
    kernels=(9, 5, 3),
    dilations=(1, 2, 4),
    residual=True,
    dropout=0.15,
    l2=5e-4,
    head_units=256,
    head_dropout=0.30,
    causal=True,
    use_ln=True,
    output_bias=None
):
    """
    CNN 1D con bloques residuales opcionales, normalización configurable y salida lineal (logits).

    Parámetros:
    -----------
    n_features : int
        Número de variables de entrada (features).
    window : int
        Longitud de las secuencias temporales.
    n_classes : int, opcional (default=3)
        Número de clases en la salida (para softmax).
    filters, kernels, dilations : tuplas
        Configuración por capa convolucional.
    residual : bool
        Si True, activa bloques residuales.
    dropout : float
        Dropout dentro de los bloques.
    l2 : float
        Regularización L2 en las convoluciones y densa.
    output_bias : array-like o None
        Bias inicial opcional para logits.
    """

    inp = keras.Input(shape=(window, n_features))
    x = layers.SpatialDropout1D(0.10)(inp)

    # --- Bloques internos ---
    def _conv_block(x, f, k, d):
        pad = "causal" if causal else "same"
        x = layers.Conv1D(
            f, k,
            padding=pad,
            dilation_rate=d,
            kernel_regularizer=regularizers.l2(l2),
            use_bias=True
        )(x)
        x = (layers.LayerNormalization()(x) if use_ln else layers.BatchNormalization()(x))
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
        return x

    def _residual_block(x, f, k, d):
        skip = x
        y = _conv_block(x, f, k, d)
        y = _conv_block(y, f, k, d)
        if skip.shape[-1] != y.shape[-1]:
            skip = layers.Conv1D(f, 1, padding="same")(skip)
        return layers.Add()([skip, y])

    # --- Stack convolucional ---
    for i, (f, k, d) in enumerate(zip(filters, kernels, dilations)):
        x = _residual_block(x, f, k, d) if (residual and i > 0) else _conv_block(x, f, k, d)

    # --- Head ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(head_units, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(head_dropout)(x)

    out = layers.Dense(
        n_classes,
        activation=None,  # logits
        bias_initializer=(
            keras.initializers.Constant(output_bias) if output_bias is not None else "zeros"
        ),
        name="logits"
    )(x)

    model = keras.Model(inp, out, name="cnn_1d_logits")
    return model
