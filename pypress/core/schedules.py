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

All scheduled values read one shared training clock: the current epoch, held
here as a plain Python value. The training loop (a Keras callback, or a torch
loop calling `set_epoch`) advances it; each `ScheduledValue` computes its own
value from it when read, so nothing has to find the scheduled values to update
them. `pypress.keras.schedules` layers an in-graph clock on top of this one,
because a traced `tf.function` would otherwise freeze the epoch.
"""

import contextlib
import math
from typing import Callable, Iterator

_SCHEDULES = ("linear", "cosine", "exponential")

# The epoch on the clock; None means "no scheduler is running": every value
# reads its `end`.
_EPOCH: int | None = None


def set_epoch(epoch: int | None):
    """Sets the training clock to 'epoch', or stops it when None."""
    global _EPOCH
    _EPOCH = None if epoch is None else int(epoch)


def current_epoch() -> int | None:
    """The epoch on the training clock, or None when no scheduler is running."""
    return _EPOCH


@contextlib.contextmanager
def epoch_clock(epoch: int | None = None) -> Iterator[Callable[[int | None], None]]:
    """Runs a block with the training clock, restoring the previous epoch after.

    Yields `set_epoch`, so a loop advances the clock with it. The clock is
    restored when the block ends, even on an error, so a later run without a
    scheduler is not affected by this one.

    Example:
        >>> with epoch_clock() as clock:
        ...     for epoch in range(30):
        ...         clock(epoch)
        ...         train_one_epoch()

    Args:
      epoch: Optional epoch to set on entering the block.
    """
    previous = current_epoch()
    if epoch is not None:
        set_epoch(epoch)
    try:
        yield set_epoch
    finally:
        set_epoch(previous)


class ScheduledValue:
    """A scalar that varies with the training epoch.

    Reads the shared training clock when `value()` is called, so it is correct
    in eager code (PyTorch) without any bookkeeping. Under TensorFlow, use
    `pypress.keras.schedules.ScheduledValue`, which computes the same value
    in-graph.

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

    Note:
        While no scheduler is running the value is `end`, not `start`. Advancing
        the clock is up to the training loop; forgetting to do so therefore
        degrades to the fully enforced, un-annealed penalty rather than silently
        dropping the regularizer altogether. That is the safe direction: a
        forgotten warm-up costs you the annealing, not the regularization.
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
          name: Optional name, used in logs.
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
        self.name = name or "scheduled_value"

    def value_at(self, epoch: int) -> float:
        """Computes the scheduled value for 'epoch' in plain Python."""
        progress = (epoch - self.start_epoch) / self.duration_epochs
        progress = min(1.0, max(0.0, progress))

        if self.schedule == "linear":
            return self.start + (self.end - self.start) * progress
        if self.schedule == "cosine":
            eased = 0.5 * (1.0 - math.cos(math.pi * progress))
            return self.start + (self.end - self.start) * eased
        # Geometric: constant multiplicative step per unit progress.
        return self.start * (self.end / self.start) ** progress

    def value(self):
        """The value for the epoch on the training clock.

        Equals `value_at(current_epoch())`, or `end` while no scheduler is
        running.
        """
        epoch = current_epoch()
        return self.end if epoch is None else self.value_at(epoch)

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


# The serialized form of a `ScheduledValue` is the one Keras writes for its
# registered class, so a config from either backend builds the other.
_SERIALIZED_MODULE = "pypress.keras.schedules"
_SERIALIZED_NAME = "ScheduledValue"
_SERIALIZED_REGISTERED_NAME = "pypress>ScheduledValue"


def serialize_scalar(value):
    """Serializes a scalar that may be a plain float or a `ScheduledValue`."""
    if isinstance(value, ScheduledValue):
        return {
            "module": _SERIALIZED_MODULE,
            "class_name": _SERIALIZED_NAME,
            "config": value.get_config(),
            "registered_name": _SERIALIZED_REGISTERED_NAME,
        }
    return float(value)


def deserialize_scalar(value):
    """Inverse of `serialize_scalar`.

    Raises:
      ValueError: if a dict does not describe a `ScheduledValue`.
    """
    if isinstance(value, dict):
        if value.get("class_name") != _SERIALIZED_NAME:
            raise ValueError(
                f"Cannot deserialize a scalar from {value!r}; "
                f"expected a serialized {_SERIALIZED_NAME}."
            )
        return ScheduledValue.from_config(value["config"])
    return value
