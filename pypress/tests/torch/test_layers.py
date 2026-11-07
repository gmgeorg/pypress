"""Tests of the torch PRESS layers."""

import numpy as np
import pytest
import torch
from torch import nn

from pypress.torch import layers as L
from pypress.torch import regularization_loss
from pypress.torch import regularizers as R


def _x(n=12, d=6, seed=0):
    return torch.tensor(
        np.random.default_rng(seed).normal(size=(n, d)), dtype=torch.float32
    )


# -- PredictiveStateSimplex ---------------------------------------------


@pytest.mark.parametrize("n_states", [1, 2, 5])
def test_simplex_rows_sum_to_one(n_states):
    """Outputs live on the simplex for any K."""
    out = L.PredictiveStateSimplex(n_states, 6)(_x())
    assert out.shape == (12, n_states)
    torch.testing.assert_close(out.sum(1), torch.ones(12))


def test_simplex_starts_uniform_with_zero_init():
    """Zero kernel and bias give 1 / K everywhere."""
    out = L.PredictiveStateSimplex(4, 6)(_x())
    torch.testing.assert_close(out, torch.full((12, 4), 0.25))


def test_simplex_lazy_in_features_materializes_on_the_first_call():
    """in_features=None infers the size, and still starts from zeros."""
    layer = L.PredictiveStateSimplex(3)
    assert layer.in_features is None
    out = layer(_x(d=7))
    assert layer.in_features == 7
    assert layer.linear.weight.shape == (3, 7)
    torch.testing.assert_close(out, torch.full((12, 3), 1 / 3))
    assert layer.linear.weight.abs().sum() == 0 and layer.linear.bias.abs().sum() == 0


def test_simplex_lazy_layer_honours_a_non_zero_initializer():
    """The initializer is applied when the lazy layer materializes."""
    torch.manual_seed(0)
    layer = L.PredictiveStateSimplex(3, kernel_initializer="glorot_uniform")
    layer(_x())
    assert layer.linear.weight.abs().sum() > 0


def test_simplex_lazy_layer_loads_a_state_dict_before_its_first_call():
    """Saved weights load into a layer that has not seen an input yet."""
    trained = L.PredictiveStateSimplex(3, 6, kernel_initializer="glorot_uniform")
    fresh = L.PredictiveStateSimplex(3)
    fresh.load_state_dict(trained.state_dict())
    torch.testing.assert_close(fresh(_x()), trained(_x()))


def test_simplex_initializers_and_validation():
    """Named and callable initializers work; bad ones fail at construction."""
    layer = L.PredictiveStateSimplex(
        3,
        4,
        kernel_initializer=lambda t: nn.init.constant_(t, 0.5),
        bias_initializer="ones",
    )
    assert (layer.linear.weight == 0.5).all() and (layer.linear.bias == 1).all()
    with pytest.raises(ValueError, match="Unknown initializer"):
        L.PredictiveStateSimplex(3, 4, kernel_initializer="nope")
    with pytest.raises(ValueError, match="must be >= 1"):
        L.PredictiveStateSimplex(0, 4)


def test_simplex_trains_and_the_loss_decreases():
    """A few Adam steps reduce a regression loss through simplex + means."""
    torch.manual_seed(0)
    x = _x(200)
    y = (x[:, :1] > 0).float() * 3.0
    model = nn.Sequential(L.PredictiveStateSimplex(4, 6), L.PredictiveStateMeans(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


# -- activity regularizers ----------------------------------------------


def _regularized(**kwargs):
    reg = R.DegreesOfFreedom(l2=1.0, target=2.0, **kwargs)
    return L.PredictiveStateSimplex(
        4, 6, kernel_initializer="glorot_uniform", activity_regularizer=reg
    )


def test_penalty_is_stored_in_training_mode_only():
    """regularization_loss is the stored penalty, and zero in eval mode."""
    torch.manual_seed(0)
    layer = _regularized()
    assert regularization_loss(layer).item() == 0.0  # nothing stored before a forward
    weights = layer(_x())
    expected = layer.activity_regularizer.penalty(weights)
    torch.testing.assert_close(regularization_loss(layer), expected)
    layer.eval()
    layer(_x())
    assert layer.regularization_penalty is None
    assert regularization_loss(layer).item() == 0.0


def test_regularization_loss_sums_layers_and_backpropagates():
    """Penalties from several layers add up and carry a gradient."""
    torch.manual_seed(0)
    model = nn.Sequential(_regularized(), nn.Linear(4, 4), _regularized())
    model[2] = L.PredictiveStateSimplex(
        4, 4, kernel_initializer="glorot_uniform", activity_regularizer=R.Uniform(1.0)
    )
    model(_x())
    stored = [m.regularization_penalty for m in (model[0], model[2])]
    torch.testing.assert_close(regularization_loss(model), stored[0] + stored[1])
    regularization_loss(model).backward()
    assert model[0].linear.weight.grad.abs().sum() > 0


def test_regularization_loss_of_a_model_without_regularizers_is_zero():
    """No stored penalties: a scalar zero on the model's device."""
    value = regularization_loss(nn.Sequential(L.PredictiveStateSimplex(3, 6)))
    assert value.ndim == 0 and value.item() == 0.0
    assert regularization_loss(nn.Identity()).item() == 0.0


def test_the_regularizer_state_is_part_of_the_layer_and_follows_train_eval():
    """EMA buffers exist from construction and freeze under eval()."""
    layer = _regularized(ema_decay=0.9)
    assert "activity_regularizer._estimator.ema_step" in layer.state_dict()
    layer(_x())
    layer(_x())
    assert layer.activity_regularizer.smoothing_steps == 2
    layer.eval()
    layer(_x())
    assert layer.activity_regularizer.smoothing_steps == 2
    layer.train()
    layer(_x())
    assert layer.activity_regularizer.smoothing_steps == 3


def test_a_stateful_regularizer_cannot_serve_two_layers():
    """The averages are per-layer state."""
    reg = R.Uniform(l2=1.0, ema_decay=0.9)
    first = L.PredictiveStateSimplex(4, 6, activity_regularizer=reg)
    with pytest.raises(ValueError, match="needs its own instance"):
        L.PredictiveStateSimplex(4, 6, activity_regularizer=reg)
    assert first.activity_regularizer is reg
    shared = R.Uniform(l2=1.0)  # no state: sharing is fine
    L.PredictiveStateSimplex(4, 6, activity_regularizer=shared)
    L.PredictiveStateSimplex(4, 6, activity_regularizer=shared)


def test_state_dict_round_trip_restores_weights_and_averages():
    """A fresh layer loaded from a trained one continues identically."""
    torch.manual_seed(0)
    trained = _regularized(ema_decay=0.8)
    for seed in range(3):
        trained(_x(seed=seed))
    fresh = _regularized(ema_decay=0.8)
    fresh.load_state_dict(trained.state_dict())
    assert fresh.activity_regularizer.smoothing_steps == 3
    x = _x(seed=9)
    torch.testing.assert_close(fresh(x), trained(x))
    torch.testing.assert_close(
        fresh.regularization_penalty, trained.regularization_penalty
    )


def test_combined_regularizer_on_a_layer():
    """A composite attached to a layer binds every stateful child."""
    combined = R.CombinedRegularizer(
        [R.Uniform(l2=0.1, ema_decay=0.9), R.MinStateSize(l2=1.0, min_share=0.1)]
    )
    layer = L.PredictiveStateSimplex(4, 6, activity_regularizer=combined)
    assert any("ema_step" in key for key in layer.state_dict())
    layer(_x())
    assert regularization_loss(layer).ndim == 0


# -- PredictiveStateMeans -----------------------------------------------


@pytest.mark.parametrize("n_states,units", [(1, 1), (2, 4), (5, 10)])
def test_means_shapes(n_states, units):
    """Output is (batch, units); the table is (K, units)."""
    states = L.PredictiveStateSimplex(n_states, 6)(_x())
    means = L.PredictiveStateMeans(n_states, units)
    assert means(states).shape == (12, units)
    assert means.state_conditional_means.shape == (n_states, units)


def test_means_is_the_weighted_sum_of_the_table():
    """output = weights @ table, with init_values on the original scale."""
    means = L.PredictiveStateMeans(3, 1, activation="sigmoid", init_values=0.8)
    torch.testing.assert_close(means.state_conditional_means, torch.full((3, 1), 0.8))
    one_hot = torch.eye(3)
    torch.testing.assert_close(means(one_hot), torch.full((3, 1), 0.8))


def test_means_init_values_forms_and_errors():
    """Scalar, 1D and 2D init_values expand; bad shapes and activations fail."""
    two_d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    means = L.PredictiveStateMeans(3, 2, init_values=two_d)
    torch.testing.assert_close(
        means.state_conditional_means, torch.tensor(two_d.T, dtype=torch.float32)
    )
    per_unit = L.PredictiveStateMeans(3, 2, init_values=[1.0, 2.0])
    torch.testing.assert_close(
        per_unit.state_conditional_means[:, 1], torch.full((3,), 2.0)
    )
    with pytest.raises(ValueError, match="2D 'init_values'"):
        L.PredictiveStateMeans(3, 2, init_values=np.ones((3, 2)))
    with pytest.raises(ValueError, match="Unknown activation"):
        L.PredictiveStateMeans(3, 2, activation="relu", init_values=1.0)


def test_means_rejects_the_wrong_number_of_states():
    """The input must have one column per state."""
    with pytest.raises(ValueError, match="3 columns"):
        L.PredictiveStateMeans(3, 1)(torch.ones(4, 5))


def test_activation_forms_name_function_module_and_per_column():
    """Names, functions and modules are interchangeable; sequences go per column."""
    states = torch.softmax(_x(d=3), dim=-1)
    by_name = L.PredictiveStateMeans(3, 2, activation="softplus")
    by_function = L.PredictiveStateMeans(3, 2, activation=torch.nn.functional.softplus)
    by_module = L.PredictiveStateMeans(3, 2, activation=nn.Softplus())
    for other in (by_function, by_module):
        other.load_state_dict(by_name.state_dict())
        torch.testing.assert_close(other(states), by_name(states))

    per_column = L.PredictiveStateMeans(3, 2, activation=["linear", "softplus"])
    per_column.load_state_dict(by_name.state_dict())
    table = per_column.state_conditional_means
    torch.testing.assert_close(table[:, 0], by_name.logits[:, 0])
    torch.testing.assert_close(
        table[:, 1], torch.nn.functional.softplus(by_name.logits[:, 1])
    )


def test_activation_validation():
    """Unknown names, wrong counts and unnamed callables with init_values fail."""
    with pytest.raises(ValueError, match="Unknown activation"):
        L.PredictiveStateMeans(3, 1, activation="nope")
    with pytest.raises(ValueError, match="one per column"):
        L.PredictiveStateMeans(3, 2, activation=["linear"])
    with pytest.raises(ValueError, match="known, invertible"):
        L.PredictiveStateMeans(3, 1, activation=lambda t: t * 2, init_values=1.0)
    # An unnamed callable is fine without init_values.
    L.PredictiveStateMeans(3, 1, activation=lambda t: t * 2)


def test_activation_module_parameters_are_registered():
    """A learnable activation module follows the layer (state_dict, .to)."""
    means = L.PredictiveStateMeans(3, 2, activation=[nn.PReLU(), "linear"])
    assert any(
        "weight" in key and "_activation_modules" in key for key in means.state_dict()
    )
    means.to(torch.float64)
    assert means.logits.dtype == torch.float64


def test_softmax_activation_normalizes_across_units_per_state():
    """A single softmax applies across the columns of each state."""
    means = L.PredictiveStateMeans(3, 4, activation="softmax")
    torch.testing.assert_close(means.state_conditional_means.sum(1), torch.ones(3))


def test_means_state_dict_round_trip_and_device_move():
    """Weights save/load; .to() moves them."""
    means = L.PredictiveStateMeans(3, 2)
    other = L.PredictiveStateMeans(3, 2)
    other.load_state_dict(means.state_dict())
    torch.testing.assert_close(other.logits, means.logits)
    assert list(means.state_dict()) == ["logits"]
    assert means.to("meta").logits.device.type == "meta"


# -- PredictiveStateParams ----------------------------------------------


def test_params_shapes_and_values():
    """Broadcast to every observation, in either layout."""
    states = torch.softmax(_x(d=3), dim=-1)
    kwargs = dict(activations=["linear", "softplus"], init_values=[0.0, 1.0])
    nested = L.PredictiveStateParams(3, 2, flatten_output=False, **kwargs)(states)
    assert nested.shape == (12, 3, 2)
    torch.testing.assert_close(nested[0, :, 0], torch.zeros(3))
    torch.testing.assert_close(nested[0, :, 1], torch.ones(3), rtol=1e-5, atol=1e-6)
    flat = L.PredictiveStateParams(3, 2, **kwargs)(states)
    assert flat.shape == (12, 6)


def test_params_flatten_order_is_parameter_major():
    """[s0_p0, s1_p0, ..., sK_p0, s0_p1, ...], the layout ColumnLayout relies on."""
    layer = L.PredictiveStateParams(3, 2)
    with torch.no_grad():
        layer.logits.copy_(torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    flat = layer(torch.ones(2, 3) / 3)
    expected = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    torch.testing.assert_close(flat[0], expected)
    torch.testing.assert_close(flat[1], expected)


def test_params_gradient_reaches_the_table():
    """The broadcast table is a differentiable function of the logits."""
    layer = L.PredictiveStateParams(3, 2)
    layer(torch.ones(5, 3)).sum().backward()
    assert layer.logits.grad.abs().sum() > 0


def test_params_validation_and_extra_leading_dims():
    """n_params_per_state >= 1; inputs may have extra leading dimensions."""
    with pytest.raises(ValueError, match="n_params_per_state"):
        L.PredictiveStateParams(3, 0)
    layer = L.PredictiveStateParams(3, 2, flatten_output=False)
    assert layer(torch.ones(4, 7, 3)).shape == (4, 7, 3, 2)
    assert L.PredictiveStateParams(3, 2)(torch.ones(4, 7, 3)).shape == (4, 7, 6)


# -- PRESS --------------------------------------------------------------


def test_press_composes_simplex_and_means():
    """PRESS equals its parts, and exposes states and means."""
    torch.manual_seed(0)
    press = L.PRESS(
        n_states=4,
        units=2,
        in_features=6,
        predictive_state_simplex_kwargs=dict(kernel_initializer="glorot_uniform"),
        predictive_state_means_kwargs=dict(init_values=[1.0, 2.0]),
    )
    x = _x()
    states = press.predictive_states(x)
    torch.testing.assert_close(press(x), press.predictive_state_means(states))
    assert press.state_conditional_means.shape == (4, 2)
    torch.testing.assert_close(
        press.state_conditional_means[0], torch.tensor([1.0, 2.0])
    )


def test_press_regularization_flows_through_the_simplex_kwargs():
    """An activity regularizer given to PRESS is found by regularization_loss."""
    press = L.PRESS(
        4,
        1,
        6,
        predictive_state_simplex_kwargs=dict(activity_regularizer=R.Uniform(1.0)),
    )
    press(_x())
    assert regularization_loss(press).ndim == 0
    press.eval()
    press(_x())
    assert regularization_loss(press).item() == 0.0


def test_press_state_dict_and_lazy_input():
    """PRESS with a lazy simplex round-trips through state_dict."""
    press = L.PRESS(3, 1)
    press(_x())
    other = L.PRESS(3, 1)
    other.load_state_dict(press.state_dict())
    torch.testing.assert_close(other(_x()), press(_x()))


def test_a_layer_with_a_stateful_regularizer_can_be_pickled_and_copied():
    """The per-layer ownership record does not stop pickling or deep copies."""
    import copy
    import pickle

    layer = _regularized(ema_decay=0.9)
    layer(_x())
    for clone in (pickle.loads(pickle.dumps(layer)), copy.deepcopy(layer)):
        assert clone.activity_regularizer.smoothing_steps == 1
        torch.testing.assert_close(clone(_x()), layer(_x()))
