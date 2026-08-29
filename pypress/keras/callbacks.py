"""Keras callbacks for pypress training."""

import warnings

import tensorflow as tf

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
    # Regularizers hold their scalars (and nested regularizers) as attributes.
    if isinstance(obj, tf.keras.regularizers.Regularizer):
        for item in vars(obj).values():
            yield from _walk(item, seen)


def collect_scheduled_values(model: tf.keras.Model) -> list[ScheduledValue]:
    """Finds every `ScheduledValue` attached to a model's regularizers.

    Walks all layers (recursing into nested layers such as `PRESS`) and inspects
    their regularizer attributes, descending through container regularizers like
    `CombinedRegularizer` and `UniformAndDegreesOfFreedomRegularizer`.

    Args:
      model: The Keras model to inspect.

    Returns:
      The scheduled values found, in a stable order and without duplicates.
    """
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
    return found


class RegularizerScheduler(tf.keras.callbacks.Callback):
    """Advances every `ScheduledValue` in a model's regularizers each epoch.

    Regularizer penalties are captured inside a traced `tf.function`, so mutating
    a plain Python attribute mid-training is silently ignored -- the old value
    keeps being used with no error. `ScheduledValue` stores its value in a
    `tf.Variable` instead, and this callback assigns it at the start of every
    epoch so the change actually takes effect.

    Attach it to `fit`; without it, every `ScheduledValue` stays at its `start`
    value for the entire run.

    Example:
        >>> dof = DegreesOfFreedom(
        ...     l2=0.1,
        ...     target=ScheduledValue(start=9.0, end=3.0, duration_epochs=20),
        ... )
        >>> model.fit(X, y, epochs=30, callbacks=[RegularizerScheduler()])

    Attributes:
      verbose: If True, logs each value assigned at the start of every epoch.
    """

    def __init__(self, verbose: bool = False):
        """Initializes the callback.

        Args:
          verbose: If True, log every scheduled value at the start of each epoch.
        """
        super().__init__()
        self.verbose = verbose
        self._scheduled_values: list[ScheduledValue] = []

    def set_model(self, model):
        """Binds the model and discovers the scheduled values to drive."""
        super().set_model(model)
        self._scheduled_values = collect_scheduled_values(model)
        if not self._scheduled_values:
            warnings.warn(
                "RegularizerScheduler found no ScheduledValue in the model's "
                "regularizers; it will have no effect. Pass a ScheduledValue as a "
                "regularizer's 'l2' or 'target' to schedule it.",
                stacklevel=2,
            )

    @property
    def scheduled_values(self) -> list[ScheduledValue]:
        """The scheduled values this callback drives (populated once bound)."""
        return self._scheduled_values

    def on_epoch_begin(self, epoch, logs=None):
        """Assigns each scheduled value for the epoch about to start."""
        for scheduled in self._scheduled_values:
            value = scheduled.assign_epoch(epoch)
            if self.verbose:
                print(
                    f"[RegularizerScheduler] epoch {epoch}: "
                    f"{scheduled.variable.name} = {value:.6g}"
                )
