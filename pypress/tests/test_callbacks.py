import numpy as np
import pytest
import tensorflow as tf

from pypress.keras import schedules
from pypress.keras.callbacks import RegularizerScheduler, collect_scheduled_values
from pypress.keras.layers import PRESS
from pypress.keras.regularizers import (
    CombinedRegularizer,
    DegreesOfFreedom,
    MinStateSize,
    StateSizeEntropy,
    Uniform,
)
from pypress.keras.schedules import ScheduledValue


def _press_model(regularizer, n_states=9, n_features=5):
    """A small PRESS model with 'regularizer' on the simplex activations."""
    press = PRESS(
        n_states=n_states,
        units=1,
        predictive_state_simplex_kwargs={"activity_regularizer": regularizer},
    )
    inputs = tf.keras.Input(shape=(n_features,))
    return tf.keras.Model(inputs, press(inputs))


def _data(n=256, n_features=5, seed=0):
    """Deterministic regression data."""
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n, n_features)).astype(np.float32),
        rng.standard_normal((n, 1)).astype(np.float32),
    )


class _Probe(tf.keras.callbacks.Callback):
    """Records the value of some ScheduledValues at the end of every epoch."""

    def __init__(self, *values):
        """Stores the values to record."""
        super().__init__()
        self.values = values
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        """Records every value as seen by the graph during this epoch."""
        self.history.append([float(v.value()) for v in self.values])


def _fit(model, epochs, *values, scheduler=True):
    """Fits 'model' and returns the per-epoch values of 'values'."""
    probe = _Probe(*values)
    callbacks = ([RegularizerScheduler()] if scheduler else []) + [probe]
    X, y = _data()
    model.fit(X, y, epochs=epochs, batch_size=64, verbose=0, callbacks=callbacks)
    return np.array(probe.history)


def test_scheduler_advances_values_during_fit():
    """The callback drives values epoch by epoch, ending at 'end'."""
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    model = _press_model(DegreesOfFreedom(l2=l2, target=3.0))
    model.compile(optimizer="adam", loss="mse")

    history = _fit(model, 5, l2)
    np.testing.assert_allclose(history[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0])


def test_any_holder_is_scheduled_without_discovery():
    """A value held by a closure-based loss is driven too: nothing is searched for."""
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=2)
    model = _press_model(Uniform(l2=0.1))

    def penalized_mse(y_true, y_pred):
        """MSE plus a scheduled multiple of the mean prediction."""
        return tf.reduce_mean((y_true - y_pred) ** 2) + l2.value() * 0.0

    model.compile(optimizer="adam", loss=penalized_mse)
    history = _fit(model, 3, l2)
    np.testing.assert_allclose(history[:, 0], [0.0, 0.5, 1.0])


def test_without_the_callback_the_penalty_is_fully_enforced():
    """No scheduler means no annealing -- but the regularizer still applies.

    This is the safe degradation: forgetting `RegularizerScheduler` costs the
    warm-up, not the regularization. Reading `start` instead would leave a
    warm-up pinned at l2=0, silently training with no penalty at all.
    """
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    model = _press_model(DegreesOfFreedom(l2=l2, target=3.0))
    model.compile(optimizer="adam", loss="mse")

    history = _fit(model, 3, l2, scheduler=False)
    np.testing.assert_allclose(history[:, 0], [1.0, 1.0, 1.0])
    assert schedules.current_epoch() is None


def test_clock_is_restored_after_fit():
    """A finished fit does not leave its epoch on the clock for later models."""
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=10)
    model = _press_model(DegreesOfFreedom(l2=l2, target=3.0))
    model.compile(optimizer="adam", loss="mse")

    _fit(model, 2, l2)
    assert schedules.current_epoch() is None
    assert float(l2.value()) == 1.0


def test_scheduled_penalty_actually_changes_the_loss():
    """A ramped l2 must show up in the reported loss, i.e. it survives tracing.

    Uses a zero learning rate so the weights cannot move: any change in the
    reported loss is attributable to the schedule alone, not to the optimizer
    reducing the penalty it is being charged for. The target is set far from the
    trace of an untrained layer (whose near-uniform weights give trace(H) ~ 1),
    so the penalty is substantial and scales visibly with l2.
    """
    l2 = ScheduledValue(start=0.0, end=10.0, duration_epochs=3)
    model = _press_model(DegreesOfFreedom(l2=l2, target=9.0))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0), loss="mse")

    X, y = _data()
    history = model.fit(
        X, y, epochs=4, batch_size=64, verbose=0, callbacks=[RegularizerScheduler()]
    )
    losses = history.history["loss"]
    # Epoch 0 runs at l2 = 0 (pure MSE); the penalty then ramps in monotonically.
    assert all(b > a for a, b in zip(losses, losses[1:])), losses


def test_opposing_directions_in_one_model():
    """The recommended setup: DoF warms up while MinStateSize decays to a floor."""
    dof_l2 = ScheduledValue(start=0.0, end=0.5, duration_epochs=4)
    floor_l2 = ScheduledValue(start=1.0, end=0.2, duration_epochs=4)
    reg = CombinedRegularizer(
        [
            DegreesOfFreedom(l2=dof_l2, target=3.0),
            MinStateSize(l2=floor_l2, min_share=0.05),
        ]
    )
    model = _press_model(reg)
    model.compile(optimizer="adam", loss="mse")

    history = _fit(model, 5, dof_l2, floor_l2)
    # Collapse pressure ramped up; the anti-collapse counterweight eased off.
    np.testing.assert_allclose(history[0], [0.0, 1.0])
    np.testing.assert_allclose(history[-1], [0.5, 0.2])


def test_scheduler_is_idempotent_across_repeated_fits():
    """Refitting restarts the schedule from epoch 0 rather than drifting."""
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    model = _press_model(DegreesOfFreedom(l2=l2, target=3.0))
    model.compile(optimizer="adam", loss="mse")

    first = _fit(model, 3, l2)
    second = _fit(model, 3, l2)
    np.testing.assert_allclose(second, first)


def test_values_in_an_unbuilt_sequential_model_are_scheduled():
    """A model that is still unbuilt when fit starts is scheduled like any other."""
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    press = PRESS(
        n_states=9,
        units=1,
        predictive_state_simplex_kwargs={
            "activity_regularizer": DegreesOfFreedom(l2=l2, target=3.0)
        },
    )
    model = tf.keras.Sequential([press])
    assert not model.built
    model.compile(optimizer="adam", loss="mse")

    history = _fit(model, 5, l2)
    np.testing.assert_allclose(history[-1], [1.0])


def test_collect_scheduled_values_is_deprecated_but_works():
    """The old discovery helper still finds values through containers."""
    dof_l2 = ScheduledValue(start=0.0, end=0.5, duration_epochs=4)
    dof_target = ScheduledValue(start=9.0, end=3.0, duration_epochs=4)
    reg = CombinedRegularizer(
        [
            DegreesOfFreedom(l2=dof_l2, target=dof_target),
            MinStateSize(l2=0.2, min_share=0.05),
        ]
    )
    with pytest.warns(DeprecationWarning, match="collect_scheduled_values"):
        found = collect_scheduled_values(_press_model(reg))
    assert {id(v) for v in found} == {id(dof_l2), id(dof_target)}


@pytest.mark.parametrize(
    "make_regularizer",
    [
        lambda: Uniform(l2=0.1, ema_decay=0.9),
        lambda: StateSizeEntropy(l2=0.1, ema_decay=0.9),
        lambda: MinStateSize(l2=0.1, min_share=0.05, ema_decay=0.9),
        lambda: DegreesOfFreedom(l2=0.1, target=3.0, ema_decay=0.9),
    ],
    ids=["uniform", "state_size_entropy", "min_state_size", "degrees_of_freedom"],
)
def test_ema_works_inside_a_compiled_training_step(make_regularizer):
    """EMA state is created at build time and advanced inside the traced step."""
    regularizer = make_regularizer()
    model = _press_model(regularizer)
    model.compile(optimizer="adam", loss="mse")

    X, y = _data()
    model.fit(X, y, epochs=2, batch_size=64, verbose=0)

    assert regularizer.smoothing_steps == 8
    for estimate in regularizer.smoothed_statistics().values():
        assert np.all(np.isfinite(estimate.numpy()))
