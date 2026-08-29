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
is reached. See `pypress.keras.callbacks.RegularizerScheduler` for the callback
that drives these during `fit`.
"""

import math

import tensorflow as tf

_SCHEDULES = ("linear", "cosine", "exponential")


@tf.keras.utils.register_keras_serializable(package="pypress")
class ScheduledValue:
    """A scalar that varies with the training epoch, readable from a graph.

    Wraps a non-trainable `tf.Variable` holding the current value. Regularizers
    read the variable, so updates made by
    `pypress.keras.callbacks.RegularizerScheduler` take effect inside an already
    traced `tf.function`. Assigning a plain Python float to a regularizer
    attribute mid-training does *not* work: the value is captured at trace time
    and the mutation is silently ignored.

    Warm-up and decay are both expressed by the endpoints -- `start < end` ramps
    a penalty up, `start > end` decays it -- so no separate direction flag is
    needed. This matters because the right direction differs per regularizer:
    `DegreesOfFreedom`, which pushes toward *fewer* effective states, should warm
    up from 0, while `MinStateSize`/`StateSizeEntropy`, which keep states alive,
    should be at full strength during the fragile early epochs and if anything
    decay later.

    Example:
        >>> # Anneal the degrees-of-freedom target from 9 states down to 3.
        >>> target = ScheduledValue(start=9.0, end=3.0, duration_epochs=20)
        >>> reg = DegreesOfFreedom(l2=0.1, target=target)

    Note:
        The backing variable is initialized to `end`, not `start`. Advancing it
        requires a `RegularizerScheduler` callback attached to `fit`; forgetting
        one therefore degrades to the fully enforced, un-annealed penalty rather
        than silently dropping the regularizer altogether. That is the safe
        direction: a forgotten warm-up costs you the annealing, not the
        regularization. `was_scheduled` reports whether a scheduler ever ran.
    """

    def __init__(
        self,
        start: float,
        end: float,
        duration_epochs: int,
        *,
        start_epoch: int = 0,
        schedule: str = "linear",
        name: str | None = None,
    ):
        """Initializes the scheduled value.

        Args:
          start: Value used at and before `start_epoch`.
          end: Value reached at `start_epoch + duration_epochs` and held after.
          duration_epochs: Number of epochs spent interpolating. Must be >= 1.
          start_epoch: Epoch at which interpolation begins. Must be >= 0.
          schedule: One of "linear", "cosine" (smooth ease in/out) or
            "exponential" (geometric interpolation, requires both endpoints > 0).
          name: Optional name for the backing variable.
        """
        if schedule not in _SCHEDULES:
            raise ValueError(
                f"'schedule' must be one of {_SCHEDULES}. Got {schedule!r}."
            )
        if duration_epochs < 1:
            raise ValueError(f"'duration_epochs' must be >= 1. Got {duration_epochs}.")
        if start_epoch < 0:
            raise ValueError(f"'start_epoch' must be >= 0. Got {start_epoch}.")
        if schedule == "exponential" and (start <= 0.0 or end <= 0.0):
            raise ValueError(
                "'exponential' schedule interpolates geometrically and needs both "
                f"endpoints > 0. Got start={start}, end={end}. Use 'linear' or "
                "'cosine' to ramp from 0."
            )

        self.start = float(start)
        self.end = float(end)
        self.duration_epochs = int(duration_epochs)
        self.start_epoch = int(start_epoch)
        self.schedule = schedule
        self._assign_count = 0
        # Initialized to `end`, so a missing scheduler degrades to the fully
        # enforced penalty instead of silently disabling it. See the class docstring.
        self.variable = tf.Variable(
            self.end,
            trainable=False,
            dtype=tf.float32,
            name=name or "scheduled_value",
        )

    def value_at(self, epoch: int) -> float:
        """Computes the scheduled value for 'epoch' (does not assign it)."""
        progress = (epoch - self.start_epoch) / self.duration_epochs
        progress = min(1.0, max(0.0, progress))

        if self.schedule == "linear":
            return self.start + (self.end - self.start) * progress
        if self.schedule == "cosine":
            eased = 0.5 * (1.0 - math.cos(math.pi * progress))
            return self.start + (self.end - self.start) * eased
        # Geometric: constant multiplicative step per unit progress.
        return self.start * (self.end / self.start) ** progress

    def assign_epoch(self, epoch: int) -> float:
        """Assigns the value for 'epoch' to the backing variable and returns it."""
        value = self.value_at(epoch)
        self.variable.assign(value)
        self._assign_count += 1
        return value

    @property
    def was_scheduled(self) -> bool:
        """Whether a scheduler has ever advanced this value."""
        return self._assign_count > 0

    def get_config(self):
        """Gets the config."""
        return {
            "start": self.start,
            "end": self.end,
            "duration_epochs": self.duration_epochs,
            "start_epoch": self.start_epoch,
            "schedule": self.schedule,
        }

    @classmethod
    def from_config(cls, config):
        """Recreates a ScheduledValue from its configuration."""
        return cls(**config)

    def __repr__(self):
        """Returns a readable representation."""
        return (
            f"ScheduledValue(start={self.start}, end={self.end}, "
            f"duration_epochs={self.duration_epochs}, "
            f"start_epoch={self.start_epoch}, schedule={self.schedule!r})"
        )


def as_tensor(value) -> tf.Tensor:
    """Reads a scalar that may be a plain float or a `ScheduledValue`."""
    if isinstance(value, ScheduledValue):
        return value.variable
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
