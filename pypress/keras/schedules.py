"""Epoch-based schedules for regularizer hyperparameters.

Regularizer strengths and targets often should not be enforced from epoch 0.
Predictive states that reach ~0 population weight are effectively dead: the
gradient reaching a dead state's parameters is weighted by its (vanishing) row
weights, measured at ~1e-12 against ~1e-2 for a live state in a 6-state PRESS
layer. Death is therefore close to absorbing, while over-provisioning states
early is cheap and reversible, so a penalty that pushes toward fewer states
should approach the constrained region from the state-rich side rather than
being applied at full strength from the start.

A schedule does not move a penalty's optimum -- it changes which local optimum
is reached.

All scheduled values read one shared training clock: a non-trainable variable
holding the current epoch. `pypress.keras.callbacks.RegularizerScheduler`
advances it during `fit`; each `ScheduledValue` computes its own value from it
inside the graph, so nothing has to find the scheduled values to update them.
"""

import math
import warnings

import tensorflow as tf

from ..core import schedules as core_schedules
from ..core.schedules import epoch_clock  # noqa: F401

# Clock value meaning "no scheduler is running": every value reads its `end`.
_NOT_STARTED = -1.0
_CLOCK: tf.Variable | None = None


def training_clock() -> tf.Variable:
    """The shared epoch variable every `ScheduledValue` reads.

    Created on first use, outside any enclosing `tf.function`. Holds -1 while no
    scheduler is running.
    """
    global _CLOCK
    if _CLOCK is None:
        with tf.init_scope():
            _CLOCK = tf.Variable(
                _NOT_STARTED,
                trainable=False,
                dtype=tf.float32,
                name="pypress_training_clock",
            )
    return _CLOCK


def set_epoch(epoch: int | None):
    """Sets the training clock to 'epoch', or stops it when None.

    Also sets the framework-free clock in `pypress.core.schedules`, so a
    `pypress.core` `ScheduledValue` reads the same epoch.
    """
    training_clock().assign(_NOT_STARTED if epoch is None else float(epoch))
    core_schedules.set_epoch(epoch)


def current_epoch() -> int | None:
    """The epoch on the training clock, or None when no scheduler is running."""
    value = float(training_clock().numpy())
    return None if value < 0.0 else int(value)


@tf.keras.utils.register_keras_serializable(package="pypress")
class ScheduledValue(core_schedules.ScheduledValue):
    """A scalar that varies with the training epoch, readable from a graph.

    The value is computed in-graph from the shared training clock, so updates
    made by `pypress.keras.callbacks.RegularizerScheduler` take effect inside an
    already traced `tf.function`. Assigning a plain Python float to a
    regularizer attribute mid-training does *not* work: the value is captured at
    trace time and the mutation is silently ignored.

    Everything else -- the schedules, `value_at`, the config -- is inherited from
    `pypress.core.schedules.ScheduledValue`.

    Example:
        >>> # Anneal the degrees-of-freedom target from 9 states down to 3.
        >>> target = ScheduledValue(start=9.0, end=3.0, duration_epochs=20)
        >>> reg = DegreesOfFreedom(l2=0.1, target=target)

    Note:
        While no scheduler is running the value is `end`, not `start`. Advancing
        it requires a `RegularizerScheduler` callback attached to `fit`;
        forgetting one therefore degrades to the fully enforced, un-annealed
        penalty rather than silently dropping the regularizer altogether. That
        is the safe direction: a forgotten warm-up costs you the annealing, not
        the regularization.
    """

    def value(self) -> tf.Tensor:
        """The value for the epoch on the training clock, as a float32 tensor.

        Graph-safe: reads the clock variable, so a traced function sees every
        epoch's value. Equals `value_at(current_epoch())`, or `end` while no
        scheduler is running.
        """
        epoch = tf.convert_to_tensor(training_clock())
        progress = tf.clip_by_value(
            (epoch - float(self.start_epoch)) / float(self.duration_epochs), 0.0, 1.0
        )
        start, end = tf.constant(self.start), tf.constant(self.end)
        if self.schedule == "linear":
            scheduled = start + (end - start) * progress
        elif self.schedule == "cosine":
            eased = 0.5 * (1.0 - tf.cos(math.pi * progress))
            scheduled = start + (end - start) * eased
        else:
            scheduled = start * tf.pow(end / start, progress)
        return tf.where(epoch < 0.0, end, scheduled)

    @property
    def variable(self) -> tf.Tensor:
        """Deprecated: the current value. Use `value()` instead."""
        warnings.warn(
            "ScheduledValue.variable is deprecated; use ScheduledValue.value().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.value()

    def assign_epoch(self, epoch: int) -> float:
        """Deprecated: sets the shared training clock to 'epoch'.

        This moves *every* scheduled value, not just this one. Use
        `pypress.keras.schedules.set_epoch` instead.
        """
        warnings.warn(
            "ScheduledValue.assign_epoch is deprecated; all scheduled values read "
            "one training clock. Use pypress.keras.schedules.set_epoch(epoch).",
            DeprecationWarning,
            stacklevel=2,
        )
        set_epoch(epoch)
        return self.value_at(epoch)

    @property
    def was_scheduled(self) -> bool:
        """Deprecated: whether the training clock is running."""
        warnings.warn(
            "ScheduledValue.was_scheduled is deprecated; use "
            "pypress.keras.schedules.current_epoch() is not None.",
            DeprecationWarning,
            stacklevel=2,
        )
        return current_epoch() is not None


def as_tensor(value) -> tf.Tensor:
    """Reads a scalar that may be a plain float or a `ScheduledValue`."""
    if isinstance(value, ScheduledValue):
        return value.value()
    return tf.cast(value, tf.float32)


def serialize_scalar(value):
    """Serializes a scalar that may be a plain float or a `ScheduledValue`."""
    if isinstance(value, ScheduledValue):
        return tf.keras.utils.serialize_keras_object(value)
    return float(value)


def deserialize_scalar(value):
    """Inverse of `serialize_scalar`."""
    if isinstance(value, dict):
        return tf.keras.utils.deserialize_keras_object(value)
    return value
