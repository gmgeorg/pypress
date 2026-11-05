"""Keras callbacks for pypress training."""

import warnings

import tensorflow as tf

from . import schedules
from .schedules import ScheduledValue

# Attributes on a Keras layer that may hold a regularizer.
_REGULARIZER_ATTRS = (
    "activity_regularizer",
    "kernel_regularizer",
    "bias_regularizer",
)


def _walk(obj, seen):
    """Yields ScheduledValues reachable from 'obj', without revisiting objects."""
    if id(obj) in seen:
        return
    seen.add(id(obj))

    if isinstance(obj, ScheduledValue):
        yield obj
        return
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _walk(item, seen)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            yield from _walk(item, seen)
        return
    # Regularizers hold their scalars (and nested regularizers) as attributes, and
    # a custom loss can hold regularizers the same way.
    if isinstance(obj, (tf.keras.regularizers.Regularizer, tf.keras.losses.Loss)):
        for item in vars(obj).values():
            yield from _walk(item, seen)


def collect_scheduled_values(model: tf.keras.Model) -> list[ScheduledValue]:
    """Deprecated: finds `ScheduledValue`s attached to a model's regularizers.

    No longer needed: every `ScheduledValue` reads the shared training clock, so
    `RegularizerScheduler` does not have to find them.

    Walks all layers (recursing into nested layers such as `PRESS`) and inspects
    their regularizer attributes, descending through container regularizers like
    `CombinedRegularizer` and `UniformAndDegreesOfFreedomRegularizer`. Also walks
    the compiled loss, for regularizers held by a custom `tf.keras.losses.Loss`.

    Args:
      model: The Keras model to inspect.

    Returns:
      The scheduled values found, in a stable order and without duplicates.
    """
    warnings.warn(
        "collect_scheduled_values is deprecated; RegularizerScheduler no longer "
        "needs to find scheduled values.",
        DeprecationWarning,
        stacklevel=2,
    )
    found, seen = [], set()

    visited_layers = set()

    def sublayers(layer):
        """Yields nested layers, including ones held as plain attributes.

        Composite layers such as `PRESS` keep their children in ordinary
        attributes rather than in Keras' tracked `_layers`, and on an unbuilt
        model the tracked list may be empty, so both are inspected.
        """
        yield from getattr(layer, "_layers", None) or getattr(layer, "layers", [])
        for value in vars(layer).values():
            if isinstance(value, tf.keras.layers.Layer):
                yield value

    def visit_layer(layer):
        if id(layer) in visited_layers:
            return
        visited_layers.add(id(layer))
        for attr in _REGULARIZER_ATTRS:
            regularizer = getattr(layer, attr, None)
            if regularizer is not None:
                found.extend(_walk(regularizer, seen))
        # A composite layer that builds its children lazily (such as `PRESS`) has
        # no sublayers yet, and holds their regularizers in a plain kwargs dict.
        # Scan container-valued attributes so schedules are found before build.
        for value in vars(layer).values():
            if isinstance(value, (dict, list, tuple, set)):
                found.extend(_walk(value, seen))
        for sublayer in sublayers(layer):
            visit_layer(sublayer)

    for layer in model.layers:
        visit_layer(layer)
    found.extend(_walk(getattr(model, "loss", None), seen))
    return found


class RegularizerScheduler(tf.keras.callbacks.Callback):
    """Drives every `ScheduledValue` by advancing the shared training clock.

    Regularizer penalties are captured inside a traced `tf.function`, so mutating
    a plain Python attribute mid-training is silently ignored -- the old value
    keeps being used with no error. A `ScheduledValue` instead computes its value
    in-graph from one shared epoch variable (`schedules.training_clock()`), and
    this callback sets that variable at the start of every epoch. Because every
    scheduled value reads the same clock, nothing has to be discovered: a value
    held by a layer, a loss, or any other object is scheduled alike.

    Attach it to `fit`; without it, every `ScheduledValue` stays at its `end`
    value for the entire run (the fully enforced penalty). When `fit` returns,
    the clock is restored to what it was before, so a later model trained
    without a scheduler is not affected by this run.

    Example:
        >>> dof = DegreesOfFreedom(
        ...     l2=0.1,
        ...     target=ScheduledValue(start=9.0, end=3.0, duration_epochs=20),
        ... )
        >>> model.fit(X, y, epochs=30, callbacks=[RegularizerScheduler()])

    Attributes:
      verbose: If True, logs the epoch set on the clock at every epoch.
    """

    def __init__(self, verbose: bool = False):
        """Initializes the callback.

        Args:
          verbose: If True, log the clock's epoch at the start of each epoch.
        """
        super().__init__()
        self.verbose = verbose
        self._previous_epoch: int | None = None

    @property
    def scheduled_values(self) -> list[ScheduledValue]:
        """Deprecated: the scheduled values reachable from the model."""
        warnings.warn(
            "RegularizerScheduler.scheduled_values is deprecated; the scheduler "
            "drives all scheduled values through the shared training clock.",
            DeprecationWarning,
            stacklevel=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return collect_scheduled_values(self.model)

    def on_train_begin(self, logs=None):
        """Remembers the clock so it can be restored after training."""
        self._previous_epoch = schedules.current_epoch()

    def on_epoch_begin(self, epoch, logs=None):
        """Sets the training clock to the epoch about to start."""
        schedules.set_epoch(epoch)
        if self.verbose:
            print(f"[RegularizerScheduler] epoch {epoch}")

    def on_train_end(self, logs=None):
        """Restores the clock to its value before training."""
        schedules.set_epoch(self._previous_epoch)
