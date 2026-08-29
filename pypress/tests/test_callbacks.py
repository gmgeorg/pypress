import numpy as np
import pytest
import tensorflow as tf

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


def test_collect_finds_values_through_nested_layers_and_containers():
    """Discovery must reach into PRESS and through CombinedRegularizer."""
    dof_l2 = ScheduledValue(start=0.0, end=0.5, duration_epochs=4)
    dof_target = ScheduledValue(start=9.0, end=3.0, duration_epochs=4)
    reg = CombinedRegularizer(
        [
            (DegreesOfFreedom, {"l2": dof_l2, "target": dof_target}),
            (MinStateSize, {"l2": 0.2, "min_share": 0.05}),
        ]
    )
    found = collect_scheduled_values(_press_model(reg))
    assert len(found) == 2
    assert {id(v) for v in found} == {id(dof_l2), id(dof_target)}


def test_collect_returns_empty_when_nothing_is_scheduled():
    """Plain float regularizers yield no scheduled values."""
    assert collect_scheduled_values(_press_model(Uniform(l2=0.1))) == []


def test_scheduler_advances_values_during_fit():
    """The callback drives values epoch by epoch, ending at 'end'."""
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    model = _press_model(DegreesOfFreedom(l2=l2, target=3.0))
    model.compile(optimizer="adam", loss="mse")

    X, y = _data()
    assert float(l2.variable) == 1.0  # un-annealed fallback before any scheduling
    model.fit(
        X, y, epochs=1, batch_size=64, verbose=0, callbacks=[RegularizerScheduler()]
    )
    np.testing.assert_allclose(float(l2.variable), 0.0)  # epoch 0 -> start
    assert l2.was_scheduled

    model.fit(
        X, y, epochs=5, batch_size=64, verbose=0, callbacks=[RegularizerScheduler()]
    )
    np.testing.assert_allclose(float(l2.variable), 1.0)


def test_without_the_callback_the_penalty_is_fully_enforced():
    """No scheduler means no annealing -- but the regularizer still applies.

    This is the safe degradation: forgetting `RegularizerScheduler` costs the
    warm-up, not the regularization. Initializing at `start` instead would leave
    a warm-up pinned at l2=0, silently training with no penalty at all.
    """
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    model = _press_model(DegreesOfFreedom(l2=l2, target=3.0))
    model.compile(optimizer="adam", loss="mse")

    X, y = _data()
    model.fit(X, y, epochs=5, batch_size=64, verbose=0)
    assert float(l2.variable) == 1.0
    assert not l2.was_scheduled


def test_scheduled_penalty_actually_changes_the_loss():
    """A ramped l2 must show up in the reported loss, i.e. it survives tracing.

    Uses a zero learning rate so the weights cannot move: any change in the
    reported loss is attributable to the schedule alone, not to the optimizer
    reducing the penalty it is being charged for. The target is set far from the
    trace of an untrained layer (whose near-uniform weights give tr_kernel ~ 1),
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


def test_scheduler_warns_when_nothing_to_schedule():
    """Attaching the callback with no ScheduledValue is a no-op worth warning about."""
    model = _press_model(Uniform(l2=0.1))
    model.compile(optimizer="adam", loss="mse")
    X, y = _data()

    scheduler = RegularizerScheduler()
    with pytest.warns(UserWarning, match="found no ScheduledValue"):
        model.fit(X, y, epochs=1, batch_size=64, verbose=0, callbacks=[scheduler])
    assert scheduler.scheduled_values == []


def test_opposing_directions_in_one_model():
    """The recommended setup: DoF warms up while MinStateSize decays to a floor."""
    dof_l2 = ScheduledValue(start=0.0, end=0.5, duration_epochs=4)
    floor_l2 = ScheduledValue(start=1.0, end=0.2, duration_epochs=4)
    reg = CombinedRegularizer(
        [
            (DegreesOfFreedom, {"l2": dof_l2, "target": 3.0}),
            (MinStateSize, {"l2": floor_l2, "min_share": 0.05}),
        ]
    )
    model = _press_model(reg)
    model.compile(optimizer="adam", loss="mse")

    X, y = _data()
    model.fit(
        X, y, epochs=5, batch_size=64, verbose=0, callbacks=[RegularizerScheduler()]
    )

    # Collapse pressure ramped up; the anti-collapse counterweight eased off.
    np.testing.assert_allclose(float(dof_l2.variable), 0.5)
    np.testing.assert_allclose(float(floor_l2.variable), 0.2)


def test_scheduler_is_idempotent_across_repeated_fits():
    """Refitting restarts the schedule from epoch 0 rather than drifting."""
    l2 = ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    model = _press_model(DegreesOfFreedom(l2=l2, target=3.0))
    model.compile(optimizer="adam", loss="mse")
    X, y = _data()

    model.fit(
        X, y, epochs=3, batch_size=64, verbose=0, callbacks=[RegularizerScheduler()]
    )
    after_first = float(l2.variable)
    model.fit(
        X, y, epochs=3, batch_size=64, verbose=0, callbacks=[RegularizerScheduler()]
    )
    np.testing.assert_allclose(float(l2.variable), after_first)


def test_collect_finds_values_in_an_unbuilt_sequential_model():
    """Composite layers hold children as attributes, not in tracked `_layers`.

    On an unbuilt model the tracked list is empty, so discovery must also look at
    ordinary attributes or the schedule silently never advances.
    """
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

    found = collect_scheduled_values(model)
    assert [id(v) for v in found] == [id(l2)]

    model.compile(optimizer="adam", loss="mse")
    X, y = _data()
    model.fit(
        X, y, epochs=5, batch_size=64, verbose=0, callbacks=[RegularizerScheduler()]
    )
    np.testing.assert_allclose(float(l2.variable), 1.0)


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
    """EMA variables must be creatable from inside a traced training function.

    Calling the regularizer eagerly is not enough to exercise this: the variables
    are created lazily under `tf.init_scope()`, and an initial value derived from
    the batch tensor is out of scope there, which fails only under `fit`.
    """
    regularizer = make_regularizer()
    model = _press_model(regularizer)
    model.compile(optimizer="adam", loss="mse")

    X, y = _data()
    model.fit(X, y, epochs=2, batch_size=64, verbose=0)

    assert regularizer._smoothers
    for smoother in regularizer._smoothers.values():
        assert float(smoother.step) > 0.0
        assert np.all(np.isfinite(smoother.average.numpy()))
