# ============================================
# losses.py
# ============================================
"""
Módulo de pérdidas para clasificación multiclase con logits:
 - Focal Loss (sparse)
 - Cross-Entropy / Focal híbrida con logit-adjustment y regularización KL
"""

import tensorflow as tf


def sparse_categorical_focal_loss(alphas=(1.0, 1.0, 1.0), gamma=2.0):
    alphas = tf.constant(alphas, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def loss_fn(y_true, logits):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        probs = tf.nn.softmax(logits, axis=-1)
        p_t = tf.gather(probs, y_true, batch_dims=1)
        a_t = tf.gather(alphas, y_true)
        mod = tf.pow(1.0 - tf.clip_by_value(p_t, 1e-7, 1.0), gamma)
        fl = a_t * mod * ce
        return tf.reduce_mean(fl)

    return loss_fn


def make_multiclass_loss_from_logits(
    mode="ce",                 # "ce" o "focal"
    class_weight=None,         # dict {0:w0,1:w1,2:w2}
    alphas=None,               # np.array([a0,a1,a2]) si Focal
    gamma=1.0,                 # focal gamma
    pi_prior=None,             # priors (train)
    tau_la=0.6,                # intensidad del logit adjustment
    prior_target=None,         # distribución deseada para KL
    lambda_kl=0.05,            # peso KL
    kl_temperature=1.5,
    label_smoothing=0.02,
    eps=1e-8
):
    """
    Construye una función de pérdida flexible combinando:
      - CE o Focal Loss desde logits
      - Logit Adjustment (balance de clases)
      - Regularización KL suave (guardrail de distribución)
    """
    K = 3

    # ---- Preparar tensores constantes ----
    cw_vec = None
    if class_weight is not None:
        cw_vec = tf.constant(
            [class_weight.get(i, 1.0) for i in range(K)], dtype=tf.float32
        )

    if pi_prior is not None:
        pi_prior = tf.constant(pi_prior, dtype=tf.float32)
        pi_prior /= tf.reduce_sum(pi_prior)
        log_pi = tf.math.log(pi_prior + eps)
    else:
        log_pi = tf.zeros([K], dtype=tf.float32)

    if prior_target is not None:
        prior_target = tf.constant(prior_target, dtype=tf.float32)
        prior_target /= tf.reduce_sum(prior_target)

    # ---- Función principal ----
    def loss_fn(y_true, logits_raw):
        y_true = tf.cast(y_true, tf.int32)

        # Logit Adjustment (compensa desequilibrio)
        logits = logits_raw + tau_la * log_pi

        # ---- Núcleo CE / Focal ----
        if mode == "ce":
            y_one = tf.one_hot(y_true, depth=K)
            if label_smoothing > 0:
                y_one = (1.0 - label_smoothing) * y_one + label_smoothing / float(K)
            ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_one, logits=logits)
            if cw_vec is not None:
                ce *= tf.gather(cw_vec, y_true)
            core = tf.reduce_mean(ce)

        elif mode == "focal":
            p = tf.nn.softmax(logits, axis=1)
            # Evita error de rank: asegurar forma (B,)
            pt = tf.gather_nd(p, tf.stack([tf.range(tf.shape(y_true)[0]), y_true[:, 0] if len(y_true.shape)>1 else y_true], axis=1))
            alpha_vec = tf.constant(alphas if alphas is not None else [1.0/K]*K, dtype=tf.float32)
            alpha_y  = tf.gather(alpha_vec, y_true)
            focal = - alpha_y * tf.pow(1.0 - pt + eps, gamma) * tf.math.log(pt + eps)
            if cw_vec is not None:
                focal *= tf.gather(cw_vec, y_true)
            core = tf.reduce_mean(focal)
        else:
            raise ValueError("mode debe ser 'ce' o 'focal'")

        # ---- Regularización KL (batch mean) ----
        if prior_target is not None and lambda_kl > 0.0:
            p_soft = tf.nn.softmax(logits / kl_temperature, axis=1)
            batch_mean = tf.reduce_mean(p_soft, axis=0)
            kl = tf.reduce_sum(
                prior_target * (tf.math.log(prior_target + eps) - tf.math.log(batch_mean + eps))
            )
            core += lambda_kl * kl

        return core

    return loss_fn
