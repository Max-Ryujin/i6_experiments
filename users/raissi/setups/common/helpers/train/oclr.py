__all__ = ["get_oclr_config"]

import typing
import numpy as np
from textwrap import dedent
from typing import Optional


def get_oclr_function(
    num_epochs: int,
    n_steps_per_epoch: int,
    peak_lr: float = 1e-03,
    cycle_epoch: Optional[int] = None,
    initial_lr: Optional[float] = None,
    final_lr: Optional[float] = None,
) -> str:
    initial_lr = initial_lr or peak_lr / 10
    final_lr = final_lr or initial_lr / 5
    cycle_epoch = cycle_epoch or (num_epochs * 9) // 20  # 45% of the training

    return dedent(
        f"""def dynamic_learning_rate(*,
                global_train_step,
                **kwargs):
            # Increase linearly from initial_lr to peak_lr over the first cycle_epoch epochs
            # Decrease linearly from peak_lr to initial_lr over the next cycle_epoch epochs
            # Decrease linearly from initial_lr to final_lr over the last (total_epochs - 2*cycle_epoch) epochs
            initial_lr = {initial_lr}
            peak_lr = {peak_lr}
            final_lr = {final_lr}
            cycle_epoch = {cycle_epoch}
            total_epochs = {num_epochs}
            n_steps_per_epoch = {n_steps_per_epoch}

            # -- derived -- #
            steps = cycle_epoch * n_steps_per_epoch
            step_size = (peak_lr - initial_lr) / steps
            steps_final = (total_epochs - 2 * cycle_epoch) * n_steps_per_epoch
            step_size_final = (initial_lr - final_lr) / steps_final

            import tensorflow as tf
            n = tf.cast(global_train_step, tf.float32)
            return tf.where(global_train_step <= steps, initial_lr + step_size * n,
                       tf.where(global_train_step <= 2*steps, peak_lr - step_size * (n - steps), 
                           tf.maximum(initial_lr - step_size_final * (n - 2*steps), final_lr)))"""
    )


def get_onecycle_for_fintune_function(
    num_epochs: int,
    n_steps_per_epoch: int = 8250,
    const_lr: float = 8e-5,
    decay_lr: float = 1e-5,
    final_lr: float = 1e-6,
    cycle_epoch: int = 135,
) -> str:
    return dedent(
        f"""def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
               constLR    = {const_lr}
               decayLR    = {decay_lr}
               finalLR    = {final_lr}
               cycleEpoch = {cycle_epoch}
               totalEpoch = {num_epochs}
               nStep      = {n_steps_per_epoch} # steps/epoch depending on batch_size
        
               # -- derived -- #
               steps     = cycleEpoch * nStep
               stepSize  = (constLR - decayLR) / steps
               steps2    = (totalEpoch - 2 * cycleEpoch) * nStep
               stepSize2 = (decayLR - finalLR) / steps2
        
               import tensorflow as tf
               n = tf.cast(global_train_step, tf.float32)
               return tf.where(global_train_step <= steps, constLR,
                          tf.where(global_train_step <= 2*steps, constLR - stepSize * (n - steps),
                              tf.maximum(decayLR - stepSize2 * (n - 2*steps), finalLR)))"""
    )


def get_const_with_decay_function(
    num_epochs: int,
    n_steps_per_epoch: int = 2440,
    peak_lr: float = 4e-4,
    decay_lr: float = 1e-5,
    final_lr: float = 1e-6,
    const_epoch: Optional[int] = 40,
    decay_epoch: Optional[float] = 140,
    **kwargs,
) -> str:

    return dedent(
        f"""def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
              peakLR     = {peak_lr}
              decayLR    = {decay_lr}
              finalLR    = {final_lr}
              constEpoch = {const_epoch}
              decayEpoch = {decay_epoch}
              totalEpoch = {num_epochs}
              nStep      = {n_steps_per_epoch} # steps/epoch depending on batch_size
            
              # -- derived -- #
              assert constEpoch + decayEpoch <= totalEpoch
              steps1 = constEpoch * nStep
              if steps1 == 0:
                steps1 == 1
              steps2 = decayEpoch * nStep
              stepSize2 = (peakLR - decayLR) / steps2
              steps2 = steps1 + steps2
              steps3 = (totalEpoch - constEpoch - decayEpoch) * nStep
              if steps3 == 0:
                stepSize3 = 0
              else:
                stepSize3 = (decayLR - finalLR) / steps3
            
              import tensorflow as tf
              n = tf.cast(global_train_step, tf.float32)
              return tf.where(global_train_step <= steps1, peakLR,
                         tf.where(global_train_step <= steps2, peakLR - stepSize2 * (n - steps1),
                             tf.maximum(decayLR - stepSize3 * (n - steps2), finalLR)))"""
    )


# This function is designed by Wei Zhou
def get_learning_rates(
    lrate=0.001,
    pretrain=0,
    warmup=False,
    warmupRatio=0.1,
    increase=90,
    incMinRatio=0.01,
    incMaxRatio=0.3,
    constLR=0,
    decay=90,
    decMinRatio=0.01,
    decMaxRatio=0.3,
    expDecay=False,
    reset=False,
):
    # example fine tuning: get_learning_rates(lrate=5e-5, increase=0, constLR=150, decay=60, decMinRatio=0.1, decMaxRatio=1)
    # example for n epochs get_learning_rates(increase=n/2, cdecay=n/2)
    learning_rates = []
    # pretrain (optional warmup)
    if warmup:
        learning_rates += warmup_lrates(initial=lrate * warmupRatio, final=lrate, epochs=pretrain)
    else:
        learning_rates += [lrate] * pretrain
    # linear increase and/or const
    if increase > 0:
        step = lrate * (incMaxRatio - incMinRatio) / increase
        for i in range(1, increase + 1):
            learning_rates += [lrate * incMinRatio + step * i]
        if constLR > 0:
            learning_rates += [lrate * incMaxRatio] * constLR
    elif constLR > 0:
        learning_rates += [lrate] * constLR
    # linear decay
    if decay > 0:
        if expDecay:  # expotential decay (closer to newBob)
            import numpy as np

            factor = np.exp(np.log(decMinRatio / decMaxRatio) / decay)
            for i in range(1, decay + 1):
                learning_rates += [lrate * decMaxRatio * (factor**i)]
        else:
            step = lrate * (decMaxRatio - decMinRatio) / decay
            for i in range(1, decay + 1):
                learning_rates += [lrate * decMaxRatio - step * i]
    # reset and default newBob(cv) afterwards
    if reset:
        learning_rates += [lrate]
    return learning_rates


# designed by Wei Zhou
def warmup_lrates(initial=0.0001, final=0.001, epochs=20):
    lrates = []
    step_size = (final - initial) / (epochs - 1)
    for i in range(epochs):
        lrates += [initial + step_size * i]
    return lrates


def get_oclr_config(
    num_epochs: int,
    lrate: float,
) -> typing.Dict[str, typing.Any]:
    """Returns learning rate RETURNN config for OneCycle LR."""

    n = int((num_epochs // 10) * 9)
    n_rest = num_epochs - n + (num_epochs - n) % 2
    lrates = get_learning_rates(lrate=lrate / 0.3, increase=n // 2, decay=n // 2)
    lrates += list(np.linspace(lrates[-1], min([*lrates, 1e-6]), n_rest))

    assert len(lrates) == num_epochs

    return {
        "learning_rate_file": "lr.log",
        "min_learning_rate": 1e-6,
        "learning_rates": lrates,
        "learning_rate_control": "constant",
    }
