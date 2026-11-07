"""`pypress.keras` delegates to `pypress.core` without changing behavior."""

import numpy as np
import pytest

from pypress.core import activations as core_act
from pypress.core import hyperparameters as hp
from pypress.core import initializers as core_init
from pypress.core import schedules as core_sched
from pypress.core import statistics as core_stats
from pypress.keras import activations as keras_act
from pypress.keras import initializers as keras_init
from pypress.keras import schedules as keras_sched
from pypress.keras import statistics as keras_stats

# Original-scale values inside each activation's invertible range.
_INVERSE_INPUTS = {
    "sigmoid": [0.01, 0.3, 0.5, 0.8, 0.99],
    "tanh": [-0.9, -0.3, 0.0, 0.2, 0.7],
    "softsign": [-0.9, -0.3, 0.0, 0.2, 0.7],
    "exponential": [0.0, 1e-6, 0.1, 0.5, 1.0],
    "softmax": [0.0, 1e-6, 0.1, 0.5, 1.0],
    "softplus": [1e-3, 0.5, 3.0, 19.0, 21.0, 60.0],
    "elu": [-0.9, -0.3, 0.0, 0.2, 3.0],
}
_DEFAULT_INVERSE_INPUT = [-0.9, -0.3, 0.0, 0.2, 0.7, 3.0]


def test_keras_and_core_agree_on_invertible_activations():
    """The tf and numpy inverse tables cover the same activations."""
    assert set(keras_act.ACTIVATION_INVERSES) == set(core_act.ACTIVATION_INVERSES)


@pytest.mark.parametrize("name", sorted(core_act.ACTIVATION_INVERSES))
def test_numpy_inverse_matches_keras_inverse(name):
    """Each numpy inverse equals its tf counterpart to float32 rounding."""
    import tensorflow as tf

    values = np.array(_INVERSE_INPUTS.get(name, _DEFAULT_INVERSE_INPUT))
    expected = keras_act.get_inverse_activation(name)(
        tf.constant(values, dtype=tf.float32)
    ).numpy()
    actual = core_act.get_inverse_activation(name)(values)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "init_values, activations",
    [
        (0.3, "sigmoid"),
        ([0.5, 2.0], ["linear", "softplus"]),
        (np.array([[0.1, 0.2, 0.7], [1.0, 2.0, 3.0]]), ["softmax", "exponential"]),
        (0.5, "tanh"),
    ],
)
def test_keras_logits_unchanged_by_numpy_init(init_values, activations):
    """Keras initializer logits equal the tf-inverse logits to float32 rounding."""
    import tensorflow as tf

    n_states, n_columns = 3, 2
    values = core_init.state_conditional_values(init_values, n_states, n_columns)
    names = keras_act.names(activations, n_columns)
    reference = np.stack(
        [
            keras_act.get_inverse_activation(name)(
                tf.constant(values[:, i], dtype=tf.float32)
            ).numpy()
            for i, name in enumerate(names)
        ],
        axis=1,
    )
    actual = keras_init.state_conditional_logits(
        init_values, n_states, n_columns, activations
    )
    np.testing.assert_allclose(actual.numpy(), reference, rtol=1e-5, atol=1e-5)


def test_statistic_shapes_are_shared():
    """The Keras statistics use the core names and shapes."""
    assert keras_stats.names() == core_stats.names()
    assert core_stats.shape("state_sizes", 4) == (4,)
    assert core_stats.shape("mean_row_entropy", 4) == ()


def test_keras_set_epoch_also_sets_the_core_clock():
    """One call moves both the in-graph clock and the Python one."""
    keras_sched.set_epoch(5)
    assert core_sched.current_epoch() == 5
    keras_sched.set_epoch(None)
    assert core_sched.current_epoch() is None


def test_scheduled_value_serialization_is_shared_between_backends():
    """Core and Keras write the same dict, and each reads the other's."""
    config = dict(start=9.0, end=3.0, duration_epochs=7, schedule="cosine")
    core_value = core_sched.ScheduledValue(**config)
    keras_value = keras_sched.ScheduledValue(**config)
    assert core_sched.serialize_scalar(core_value) == keras_sched.serialize_scalar(
        keras_value
    )
    restored = core_sched.deserialize_scalar(keras_sched.serialize_scalar(keras_value))
    assert type(restored) is core_sched.ScheduledValue
    assert restored.get_config() == keras_value.get_config()
    with pytest.raises(ValueError):
        core_sched.deserialize_scalar({"class_name": "Other", "config": {}})


def test_keras_scheduled_value_is_a_core_scheduled_value():
    """Hyperparameter validation (which is core) accepts the Keras class."""
    value = keras_sched.ScheduledValue(start=0.0, end=2.0, duration_epochs=3)
    assert isinstance(value, core_sched.ScheduledValue)
    assert hp.L2.coerce(value) is value
    with pytest.raises(ValueError, match="'l2' must be in"):
        hp.L2.coerce(core_sched.ScheduledValue(start=-1.0, end=1.0, duration_epochs=2))
    with pytest.raises(TypeError, match="cannot be scheduled"):
        hp.EMA_DECAY.coerce(value)
