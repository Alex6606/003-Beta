# ============================================
# cnn_model.py
# ============================================
"""
Module for building the 1D CNN model with convolutional and residual
blocks for multiclass classification.
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
    1D CNN with optional residual blocks, configurable normalization,
    and linear output (logits).

    Parameters
    ----------
    n_features : int
        Number of input variables (features).
    window : int
        Length of temporal sequences.
    n_classes : int, optional (default=3)
        Number of output classes (for softmax classification).
    filters, kernels, dilations : tuple
        Per-layer convolutional configuration.
    residual : bool
        If True, enables residual blocks.
    dropout : float
        Dropout inside convolutional blocks.
    l2 : float
        L2 regularization for convolutional and dense layers.
    output_bias : array-like or None
        Optional initial bias for logits.
    """

    inp = keras.Input(shape=(window, n_features))
    x = layers.SpatialDropout1D(0.10)(inp)

    # --- Internal blocks ---
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

    # --- Convolutional stack ---
    for i, (f, k, d) in enumerate(zip(filters, kernels, dilations)):
        x = _residual_block(x, f, k, d) if (residual and i > 0) else _conv_block(x, f, k, d)

    # --- Classification head ---
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
