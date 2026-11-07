"""The torch layers equal the Keras ones: init, forward outputs and activations."""

import numpy as np
import pytest
import tensorflow as tf
import torch

from pypress.core import activations as core_act
from pypress.keras import layers as keras_layers
from pypress.keras import regularizers as keras_regularizers
from pypress.torch import activations as torch_act
from pypress.torch import layers as torch_layers
from pypress.torch import regularizers as torch_regularizers

from .data import ATOL, RTOL

N_STATES = 4


def _features(n_rows=9, n_features=6, seed=0):
    return (
        np.random.default_rng(seed).normal(size=(n_rows, n_features)).astype("float32")
    )


def _states(n_rows=9, seed=1):
    logits = np.random.default_rng(seed).normal(size=(n_rows, N_STATES))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).astype("float32")


def _close(actual, expected, message=""):
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL, err_msg=message
    )


# -- activations --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(torch_act.ACTIVATIONS))
def test_activation_names_match_keras(name):
    """Each named activation agrees on a (K, P) table, softmax across columns."""
    table = np.random.default_rng(3).normal(size=(5, 3)).astype("float32")
    expected = tf.keras.layers.Activation(name)(tf.constant(table)).numpy()
    actual = torch_act.apply(torch.tensor(table), name).numpy()
    _close(actual, expected, name)


def test_the_invertible_activations_are_all_available():
    """Every activation with an inverse in core can be used in a torch layer."""
    assert set(core_act.ACTIVATION_INVERSES) <= set(torch_act.ACTIVATIONS)


# -- init_values --------------------------------------------------------

# Open intervals of original-scale values each invertible activation can reach.
_RANGES = {
    "sigmoid": (0.05, 0.95),
    "tanh": (-0.9, 0.9),
    "softsign": (-0.8, 0.8),
    "exponential": (0.05, 3.0),
    "softmax": (0.05, 0.95),
    "softplus": (0.1, 6.0),
    "elu": (-0.8, 2.0),
    "selu": (-1.5, 2.0),
    "leaky_relu": (-2.0, 2.0),
    "linear": (-2.0, 2.0),
}


def _init_values(name, form, n_columns=2):
    low, high = _RANGES[name]
    if form == "scalar":
        return 0.5 * (low + high)
    if form == "1d":
        return np.linspace(low, high, n_columns + 2)[1:-1]
    return np.linspace(low, high, n_columns * N_STATES + 2)[1:-1].reshape(
        n_columns, N_STATES
    )


@pytest.mark.parametrize("form", ["scalar", "1d", "2d"])
@pytest.mark.parametrize("name", sorted(core_act.ACTIVATION_INVERSES))
def test_means_init_values_give_equal_logits(name, form):
    """Logits from `init_values` match for every invertible activation and form."""
    values = _init_values(name, form)
    keras_layer = keras_layers.PredictiveStateMeans(
        units=2, activation=name, init_values=values
    )
    keras_layer(tf.constant(_states()))
    torch_layer = torch_layers.PredictiveStateMeans(
        N_STATES, 2, activation=name, init_values=values
    )
    _close(torch_layer.logits.detach().numpy(), keras_layer._logits.numpy(), name)
    _close(
        torch_layer(torch.tensor(_states())).detach().numpy(),
        keras_layer(tf.constant(_states())).numpy(),
    )


@pytest.mark.parametrize("form", ["scalar", "1d", "2d"])
def test_params_per_column_init_values_give_equal_logits(form):
    """Per-column activations invert column by column in both backends."""
    activations = ["linear", "softplus", "sigmoid"]
    values = {
        "scalar": 0.4,
        "1d": [0.3, 1.5, 0.6],
        "2d": np.array(
            [[0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], [0.2, 0.4, 0.6, 0.8]]
        ),
    }[form]
    keras_layer = keras_layers.PredictiveStateParams(
        n_params_per_state=3, activations=activations, init_values=values
    )
    keras_layer(tf.constant(_states()))
    torch_layer = torch_layers.PredictiveStateParams(
        N_STATES, 3, activations, init_values=values
    )
    _close(torch_layer.logits.detach().numpy(), keras_layer._logits.numpy())


def test_default_initialization_has_the_glorot_range():
    """The default logits are Glorot-uniform: the same bound as Keras."""
    limit = np.sqrt(6.0 / (N_STATES + 3))
    torch.manual_seed(0)
    logits = torch_layers.PredictiveStateMeans(N_STATES, 3).logits.detach().numpy()
    assert np.abs(logits).max() <= limit
    assert np.abs(logits).max() > 0.5 * limit


def test_simplex_default_initialization_is_uniform_like_keras():
    """Zero kernel and bias: uniform weights, so trace(H) = 1."""
    x = _features()
    keras_out = keras_layers.PredictiveStateSimplex(N_STATES)(tf.constant(x)).numpy()
    torch_out = torch_layers.PredictiveStateSimplex(N_STATES, 6)(torch.tensor(x))
    _close(torch_out.detach().numpy(), keras_out)
    _close(torch_out.detach().numpy(), np.full((9, N_STATES), 1 / N_STATES))


# -- forward outputs through from_keras ---------------------------------


@pytest.mark.parametrize("use_regularizer", [False, True])
def test_simplex_from_keras(use_regularizer):
    """Copied weights (kernel transposed) give equal weights."""
    regularizer = (
        keras_regularizers.Uniform(l2=0.5, ema_decay=0.9) if use_regularizer else None
    )
    keras_layer = keras_layers.PredictiveStateSimplex(
        N_STATES,
        kernel_initializer="glorot_uniform",
        bias_initializer="random_normal",
        activity_regularizer=regularizer,
    )
    x = _features()
    expected = keras_layer(tf.constant(x)).numpy()
    torch_layer = torch_layers.PredictiveStateSimplex.from_keras(keras_layer)
    _close(torch_layer(torch.tensor(x)).detach().numpy(), expected)
    if use_regularizer:
        assert isinstance(torch_layer.activity_regularizer, torch_regularizers.Uniform)
        assert torch_layer.activity_regularizer.get_config() == regularizer.get_config()
    else:
        assert torch_layer.activity_regularizer is None


@pytest.mark.parametrize(
    "activation",
    ["linear", "sigmoid", "softmax", "softplus", ["linear", "softplus", "tanh"]],
    ids=str,
)
def test_means_from_keras(activation):
    """Mixture outputs and the state-conditional means agree."""
    units = 3
    keras_layer = keras_layers.PredictiveStateMeans(units=units, activation=activation)
    states = _states()
    expected = keras_layer(tf.constant(states)).numpy()
    torch_layer = torch_layers.PredictiveStateMeans.from_keras(keras_layer)
    _close(torch_layer(torch.tensor(states)).detach().numpy(), expected)
    _close(
        torch_layer.state_conditional_means.detach().numpy(),
        keras_layer.state_conditional_means.numpy(),
    )


@pytest.mark.parametrize("flatten_output", [True, False])
@pytest.mark.parametrize("activations", ["softplus", ["linear", "softplus"]], ids=str)
def test_params_from_keras(flatten_output, activations):
    """Broadcast parameters agree, in both output layouts."""
    keras_layer = keras_layers.PredictiveStateParams(
        n_params_per_state=2,
        activations=activations,
        flatten_output=flatten_output,
        init_values=[0.5, 1.5],
    )
    states = _states()
    expected = keras_layer(tf.constant(states)).numpy()
    torch_layer = torch_layers.PredictiveStateParams.from_keras(keras_layer)
    actual = torch_layer(torch.tensor(states)).detach().numpy()
    assert actual.shape == expected.shape
    _close(actual, expected)


def test_params_constructed_directly_match_keras_output_layout():
    """A torch layer built with the same arguments emits the same param-major layout."""
    kwargs = dict(activations=["linear", "softplus"], init_values=[0.5, 1.5])
    keras_out = keras_layers.PredictiveStateParams(
        n_params_per_state=2, flatten_output=True, **kwargs
    )(tf.constant(_states())).numpy()
    torch_out = torch_layers.PredictiveStateParams(N_STATES, 2, **kwargs)(
        torch.tensor(_states())
    )
    _close(torch_out.detach().numpy(), keras_out)


@pytest.mark.parametrize("activation", ["linear", ["linear", "sigmoid"]], ids=str)
def test_press_from_keras(activation):
    """The whole layer, and its predictive states, agree."""
    keras_layer = keras_layers.PRESS(
        units=2,
        n_states=N_STATES,
        activation=activation,
        predictive_state_simplex_kwargs=dict(
            kernel_initializer="glorot_uniform", bias_initializer="random_normal"
        ),
    )
    x = _features()
    expected = keras_layer(tf.constant(x)).numpy()
    torch_layer = torch_layers.PRESS.from_keras(keras_layer)
    _close(torch_layer(torch.tensor(x)).detach().numpy(), expected)
    _close(
        torch_layer.predictive_states(torch.tensor(x)).detach().numpy(),
        keras_layer.predictive_states(tf.constant(x)).numpy(),
    )
    _close(
        torch_layer.state_conditional_means.detach().numpy(),
        keras_layer.state_conditional_means.numpy(),
    )


def test_gradients_of_a_press_loss_match():
    """A mean-squared-error gradient w.r.t. the table agrees after from_keras."""
    keras_layer = keras_layers.PRESS(
        units=1,
        n_states=N_STATES,
        predictive_state_simplex_kwargs=dict(kernel_initializer="glorot_uniform"),
    )
    x, y = _features(), np.random.default_rng(4).normal(size=(9, 1)).astype("float32")
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(keras_layer(tf.constant(x)) - y))
    kernel = keras_layer.predictive_state_simplex.kernel
    keras_grad = tape.gradient(loss, kernel).numpy()

    torch_layer = torch_layers.PRESS.from_keras(keras_layer)
    torch_loss = ((torch_layer(torch.tensor(x)) - torch.tensor(y)) ** 2).mean()
    torch_loss.backward()
    grad = torch_layer.predictive_state_simplex.linear.weight.grad.numpy()
    _close(grad.T, keras_grad)


def test_from_keras_requires_a_built_layer():
    """An unbuilt Keras layer cannot be copied."""
    with pytest.raises(ValueError, match="not built"):
        torch_layers.PredictiveStateMeans.from_keras(
            keras_layers.PredictiveStateMeans(units=1)
        )
