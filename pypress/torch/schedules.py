"""Epoch-based schedules for regularizer hyperparameters, for PyTorch.

PyTorch is eager, so a `ScheduledValue` is a plain Python float read at call
time from one shared training clock: the current epoch, held as a Python value in
`pypress.core.schedules`. There is no callback or in-graph variable. The
training loop advances the clock once per epoch:

    >>> from pypress.torch import schedules
    >>> with schedules.epoch_clock() as clock:
    ...     for epoch in range(30):
    ...         clock(epoch)
    ...         train_one_epoch()

`epoch_clock` restores the previous epoch when the block ends, as
`pypress.keras.callbacks.RegularizerScheduler` does after `fit`. While no clock
is running every `ScheduledValue` reads its `end`.

Under `torch.compile` a changing Python value is guarded on, so the compiled
function is recompiled once per epoch for a scheduled hyperparameter.

See `pypress.core.schedules` for the schedules themselves.
"""

from ..core.schedules import (  # noqa: F401
    ScheduledValue,
    current_epoch,
    deserialize_scalar,
    epoch_clock,
    serialize_scalar,
    set_epoch,
)


def as_float(value) -> float:
    """Reads a scalar that may be a plain float or a `ScheduledValue`."""
    if isinstance(value, ScheduledValue):
        return float(value.value())
    return float(value)
