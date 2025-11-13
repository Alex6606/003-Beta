# ============================================
# trainer.py
# ============================================
"""
Entrenamiento en dos fases para clasificación multiclase:
 - Fase 1: Cross-Entropy + Logit Adjustment + KL
 - Fase 2: Focal Loss + Logit Adjustment + KL
Incluye manejo de early stopping o reducción de LR.
"""

from tensorflow import keras
from losses import make_multiclass_loss_from_logits



def train_two_phase_v4(
    model,
    Xtr, ytr, Xva, yva,
    cw_train_cb,              # dict de class weights
    alphas_focal, gamma,      # focal params
    pi_prior,                 # prior train
    prior_target,             # prior deseado para KL
    epochs_warmup=12, epochs_finetune=6,
    batch_size=64,
    label_smoothing=0.02,
    lambda_kl=0.05,
    kl_temperature=1.5,
    tau_la=0.6,
    verbose=1,
    early_stopping=True,          # activa MacroF1Callback
    reduce_on_plateau=True        # alternativa cuando no se usa ES
):
    """Entrenamiento en dos fases: CE → Focal."""
    callbacks_phase1, callbacks_phase2 = [], []

    # ============================================================
    # Callbacks configurables
    # ============================================================
    if early_stopping:
        try:
            from callbacks import MacroF1Callback
            cb_f1 = MacroF1Callback(
                Xva, yva,
                patience=6, reduce_lr_patience=3, factor=0.5, min_lr=1e-5, verbose=1
            )
            callbacks_phase1.append(cb_f1)
            callbacks_phase2.append(cb_f1)
        except ImportError:
            print("[WARN] MacroF1Callback no encontrado. Continuando sin early stopping.")
    elif reduce_on_plateau:
        # Reduce LR on Plateau estándar
        rlp1 = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        )
        rlp2 = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        )
        callbacks_phase1.append(rlp1)
        callbacks_phase2.append(rlp2)

    # ============================================================
    # Fase 1 → Cross-Entropy + Logit Adjustment + KL
    # ============================================================
    loss_ce = make_multiclass_loss_from_logits(
        mode="ce",
        class_weight=cw_train_cb,
        pi_prior=pi_prior, tau_la=tau_la,
        prior_target=prior_target, lambda_kl=lambda_kl,
        kl_temperature=kl_temperature,
        label_smoothing=label_smoothing
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_ce,
        metrics=["accuracy"]
    )

    print("\n===== Entrenando Fase 1: CE + LA + KL =====")
    h1 = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs_warmup,
        batch_size=batch_size,
        callbacks=callbacks_phase1,
        verbose=verbose
    )

    # ============================================================
    # Fase 2 → Focal + Logit Adjustment + KL
    # ============================================================
    loss_focal = make_multiclass_loss_from_logits(
        mode="focal",
        class_weight=cw_train_cb,
        alphas=alphas_focal, gamma=gamma,
        pi_prior=pi_prior, tau_la=tau_la,
        prior_target=prior_target,
        lambda_kl=lambda_kl, kl_temperature=kl_temperature
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=loss_focal,
        metrics=["accuracy"]
    )

    print("\n===== Entrenando Fase 2: Focal + LA + KL =====")
    h2 = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs_finetune,
        batch_size=batch_size,
        callbacks=callbacks_phase2,
        verbose=verbose
    )

    print("\n✅ Entrenamiento completado.")
    return model, {"warmup": h1.history, "finetune": h2.history}
