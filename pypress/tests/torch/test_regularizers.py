"""Tests of the torch regularizers: values, validation, state and modes."""

import copy
import math

import pytest
import torch
from torch import nn

from pypress.torch import regularizers as R
from pypress.torch import schedules

from .conftest import simplex_weights


def _hard(n_states, n_alive, n_rows=60):
    """Hard assignments spread over the first 'n_alive' states."""
    x = torch.zeros(n_rows, n_states)
    x[torch.arange(n_rows), torch.arange(n_rows) % n_alive] = 1.0
    return x


# -- values -------------------------------------------------------------


def test_uniform_is_zero_for_uniform_weights_and_positive_otherwise(weights):
    """Uniform rows cost nothing; a random batch costs something."""
    assert R.Uniform(l2=1.0).penalty(torch.full((8, 4), 0.25)).item() == pytest.approx(
        0.0, abs=1e-7
    )
    assert R.Uniform(l2=1.0).penalty(weights).item() > 0.0


def test_uniform_is_target_entropy_with_full_fraction(weights):
    """`Uniform` equals `TargetEntropy(entropy_fraction=1)`."""
    a = R.Uniform(l2=0.7).penalty(weights)
    b = R.TargetEntropy(l2=0.7, entropy_fraction=1.0).penalty(weights)
    torch.testing.assert_close(a, b)


def test_target_entropy_matches_the_closed_form(weights):
    """penalty = l2 * ((target - H) / log K) ** 2."""
    entropy = -(weights * weights.log()).sum(1).mean()
    expected = 2.0 * ((0.9 - entropy) / math.log(5)) ** 2
    torch.testing.assert_close(
        R.TargetEntropy(l2=2.0, target=0.9).penalty(weights), expected
    )


def test_target_entropy_default_target_is_log_half_k(weights):
    """Without a target the fraction defaults to 0.5."""
    default = R.TargetEntropy(l2=1.0).penalty(weights)
    explicit = R.TargetEntropy(l2=1.0, target=math.log(2.5)).penalty(weights)
    torch.testing.assert_close(default, explicit)


def test_dof_is_zero_at_the_target_for_hard_assignments():
    """Three live states give trace(H) = 3, so target 3 costs nothing."""
    x = _hard(8, 3)
    assert R.DegreesOfFreedom(l2=1.0, target=3.0).penalty(x).item() == pytest.approx(
        0.0, abs=1e-6
    )
    assert R.DegreesOfFreedom(l2=1.0, target=5.0).penalty(x).item() == pytest.approx(
        4.0
    )


def test_dof_normalize_divides_the_deviation_by_k():
    """normalize=True scales the squared deviation by 1 / K ** 2."""
    x = _hard(8, 3)
    plain = R.DegreesOfFreedom(l2=1.0, target=5.0).penalty(x)
    normalized = R.DegreesOfFreedom(l2=1.0, target=5.0, normalize=True).penalty(x)
    assert normalized.item() == pytest.approx(plain.item() / 64)


def test_state_size_entropy_penalizes_dead_states_and_is_bounded():
    """A dead state costs; the penalty never exceeds l2."""
    reg = R.StateSizeEntropy(l2=1.5)
    assert reg.penalty(torch.full((8, 4), 0.25)).item() == pytest.approx(0.0, abs=1e-7)
    value = reg.penalty(_hard(6, 1)).item()
    assert 0.0 < value <= 1.5 + 1e-6


def test_min_state_size_zero_when_every_state_clears_the_floor():
    """Exactly balanced hard assignments clear any floor <= 1 / K."""
    x = _hard(4, 4)
    assert R.MinStateSize(l2=1.0, min_share=0.2).penalty(x).item() == pytest.approx(
        0.0, abs=1e-7
    )
    assert R.MinStateSize(l2=1.0, min_share=0.2).penalty(_hard(4, 2)).item() > 0.0


def test_min_state_size_rejects_an_infeasible_floor():
    """A floor K states cannot meet fails at call time."""
    with pytest.raises(ValueError, match="not satisfiable"):
        R.MinStateSize(l2=1.0, min_share=0.25).penalty(torch.full((10, 5), 0.2))
    R.MinStateSize(l2=1.0, min_share=0.25).penalty(torch.full((10, 4), 0.25))


def test_a_schedule_is_checked_over_its_whole_range():
    """The strictest value a schedule reaches decides satisfiability."""
    reg = R.MinStateSize(
        l2=1.0, min_share=schedules.ScheduledValue(0.05, 0.3, duration_epochs=3)
    )
    with pytest.raises(ValueError, match="not satisfiable"):
        reg.penalty(torch.full((10, 5), 0.2))


def test_combined_regularizer_sums_children(weights):
    """The composite adds up its children."""
    children = [R.Uniform(l2=0.3), R.DegreesOfFreedom(l2=0.1, target=2.0)]
    expected = sum(child.penalty(weights) for child in children)
    combined = R.CombinedRegularizer(children)
    torch.testing.assert_close(combined.penalty(weights), expected)
    torch.testing.assert_close(combined(weights), expected)


def test_combined_regularizer_accepts_a_plain_callable_child(weights):
    """A child without `penalty` is called directly."""

    class Constant(nn.Module):
        def forward(self, x):
            return torch.tensor(0.25)

    uniform = R.Uniform(l2=1.0)
    combined = R.CombinedRegularizer([uniform, Constant()])
    torch.testing.assert_close(
        combined.penalty(weights), uniform.penalty(weights) + 0.25
    )


# -- gradients ----------------------------------------------------------


def test_min_state_size_gradient_revives_a_starved_state():
    """The gradient pushes mass into the starved states and is finite."""
    x = torch.tensor([[0.97, 0.02, 0.01]] * 2, requires_grad=True)
    R.MinStateSize(l2=1.0, min_share=0.2).penalty(x).backward()
    assert (x.grad[:, 1:] < 0).all() and (x.grad[:, 0] > 0).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "reg",
    [
        R.Uniform(1.0),
        R.DegreesOfFreedom(1.0, 2.0),
        R.StateSizeEntropy(1.0),
        R.MinStateSize(1.0, min_share=0.1),
    ],
)
def test_gradients_are_finite_with_an_empty_state_and_k_equal_one(reg):
    """Zero columns and K = 1 stay finite in value and gradient."""
    x = torch.tensor([[0.5, 0.5, 0.0]] * 4, requires_grad=True)
    reg.penalty(x).backward()
    assert torch.isfinite(x.grad).all()
    single = torch.ones(3, 1, requires_grad=True)
    value = reg.penalty(single)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(single.grad).all()


# -- validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: R.Uniform(l2=-1.0), "'l2' must be in"),
        (lambda: R.Uniform(ema_decay=1.0), "'ema_decay' must be in"),
        (lambda: R.TargetEntropy(target=-1.0), "'target' must be in"),
        (lambda: R.TargetEntropy(target=1.0, entropy_fraction=0.5), "not both"),
        (lambda: R.TargetEntropy(entropy_fraction=0.0), "'entropy_fraction'"),
        (lambda: R.MinStateSize(min_share=1.5), "'min_share'"),
        (lambda: R.StateSizeEntropy(active_fraction=0.0), "'active_fraction'"),
        (lambda: R.DegreesOfFreedom(target=0.5), "'target' must be in"),
        (lambda: R.DegreesOfFreedom(l2=True), "must be a number"),
    ],
)
def test_invalid_hyperparameters_are_rejected(build, message):
    """Bad values fail at construction."""
    with pytest.raises((ValueError, TypeError), match=message):
        build()


def test_a_schedule_is_validated_at_both_endpoints():
    """A scheduled value that can leave the bounds is rejected."""
    bad = schedules.ScheduledValue(start=-1.0, end=1.0, duration_epochs=2)
    with pytest.raises(ValueError, match="over the whole schedule"):
        R.Uniform(l2=bad)
    with pytest.raises(TypeError, match="cannot be scheduled"):
        R.Uniform(ema_decay=schedules.ScheduledValue(0.1, 0.5, 2))


def test_unknown_arguments_are_rejected():
    """Only declared hyperparameters are accepted."""
    with pytest.raises(TypeError):
        R.Uniform(l2=1.0, bogus=1)


def test_scheduled_l2_follows_the_clock(weights):
    """A scheduled l2 scales the penalty with the epoch on the clock."""
    reg = R.Uniform(l2=schedules.ScheduledValue(0.0, 2.0, duration_epochs=2))
    base = R.Uniform(l2=1.0).penalty(weights).item()
    schedules.set_epoch(0)
    assert reg.penalty(weights).item() == 0.0
    schedules.set_epoch(1)
    assert reg.penalty(weights).item() == pytest.approx(base)
    schedules.set_epoch(None)
    assert reg.penalty(weights).item() == pytest.approx(2 * base)


# -- config -------------------------------------------------------------


def test_config_round_trip_and_registry(weights):
    """get_config / from_config / get rebuild an equivalent regularizer."""
    reg = R.DegreesOfFreedom(l2=0.4, target=3.0, normalize=True, ema_decay=0.9)
    assert reg.get_config() == {
        "l2": 0.4,
        "target": 3.0,
        "ema_decay": 0.9,
        "normalize": True,
    }
    rebuilt = R.DegreesOfFreedom.from_config(reg.get_config())
    assert rebuilt.get_config() == reg.get_config()
    assert isinstance(R.get("Uniform"), R.Uniform)
    assert isinstance(R.get({"class_name": "MinStateSize", "l2": 2.0}), R.MinStateSize)
    assert R.get(reg) is reg and R.get(None) is None
    with pytest.raises(ValueError, match="Unknown regularizer"):
        R.get("Nope")
    with pytest.raises(ValueError, match="does not name"):
        R.from_config({"l2": 1.0})


def test_combined_config_round_trip(weights):
    """The composite serializes and rebuilds its children."""
    combined = R.CombinedRegularizer(
        [R.Uniform(l2=0.5), R.MinStateSize(l2=1.0, min_share=0.1)]
    )
    rebuilt = R.CombinedRegularizer.from_config(combined.get_config())
    torch.testing.assert_close(
        rebuilt.penalty(weights[:, :5]), combined.penalty(weights[:, :5])
    )
    assert [type(r) for r in rebuilt.regularizers] == [R.Uniform, R.MinStateSize]


def test_repr_shows_the_hyperparameters():
    """print(module) is informative."""
    assert "l2=0.5" in repr(R.Uniform(l2=0.5))


# -- EMA state ----------------------------------------------------------


def _ema_reg():
    return R.DegreesOfFreedom(l2=1.0, target=2.0, ema_decay=0.8)


def test_ema_state_is_buffers_in_the_state_dict():
    """Averages and step count are buffers, not parameters."""
    reg = _ema_reg()
    reg.bind(5)
    assert sorted(reg.state_dict()) == [
        "_estimator.ema_step",
        "_estimator.trace_denominator_ema",
        "_estimator.trace_numerator_ema",
    ]
    assert list(reg.parameters()) == []
    assert R.Uniform(l2=1.0).state_dict() == {}


def test_ema_advances_in_training_mode_only(weights):
    """eval() freezes the averages, train() resumes them."""
    reg = _ema_reg()
    for _ in range(3):
        reg.penalty(weights)
    assert reg.smoothing_steps == 3
    reg.eval()
    reg.penalty(weights)
    assert reg.smoothing_steps == 3
    reg.train()
    reg.penalty(weights)
    assert reg.smoothing_steps == 4


def test_ema_first_call_equals_the_batch_estimate(weights):
    """Bias correction makes step 1 exactly the un-smoothed value."""
    smoothed = _ema_reg().penalty(weights)
    plain = R.DegreesOfFreedom(l2=1.0, target=2.0).penalty(weights)
    torch.testing.assert_close(smoothed, plain)


def test_ema_tracks_the_bias_corrected_average():
    """Smoothed state sizes equal an explicit bias-corrected EMA."""
    reg = R.MinStateSize(l2=1.0, min_share=0.1, ema_decay=0.8)
    ema = torch.zeros(5)
    for step in range(1, 6):
        x = simplex_weights(40, 5, seed=step)
        reg.penalty(x)
        ema = 0.8 * ema + 0.2 * x.sum(0)
        torch.testing.assert_close(
            reg.smoothed_statistics()["state_sizes"],
            ema / (1 - 0.8**step),
            rtol=1e-4,
            atol=1e-5,
        )


def test_ema_gradient_flows_to_the_batch_not_the_history(weights):
    """The penalty is differentiable w.r.t. the current batch only."""
    reg = _ema_reg()
    reg.penalty(weights)
    x = weights.clone().requires_grad_()
    reg.penalty(x).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(not b.requires_grad for b in reg.buffers())


def test_state_dict_round_trip_includes_the_averages(weights):
    """Saved averages and steps are restored into a fresh regularizer."""
    reg = _ema_reg()
    for seed in range(3):
        reg.penalty(simplex_weights(32, 5, seed=seed))
    fresh = _ema_reg()
    fresh.bind(5)
    fresh.load_state_dict(reg.state_dict())
    assert fresh.smoothing_steps == 3
    nxt = simplex_weights(32, 5, seed=9)
    torch.testing.assert_close(fresh.penalty(nxt), reg.penalty(nxt))


def test_state_dict_loads_into_a_regularizer_that_never_saw_k(weights):
    """A loss-held regularizer has no state until used; loading creates it."""
    reg = _ema_reg()
    reg.penalty(weights)
    fresh = _ema_reg()
    assert fresh.state_dict() == {}
    fresh.load_state_dict(reg.state_dict())
    assert fresh.smoothing_steps == 1
    torch.testing.assert_close(fresh.penalty(weights), reg.penalty(weights))
    with pytest.raises(ValueError, match="K = 5"):
        fresh.penalty(torch.full((4, 3), 1 / 3))


def test_state_dict_loads_into_a_mean_row_entropy_only_regularizer(weights):
    """K cannot be read from a scalar statistic; loading still works."""
    reg = R.Uniform(l2=1.0, ema_decay=0.9)
    reg.penalty(weights)
    fresh = R.Uniform(l2=1.0, ema_decay=0.9)
    fresh.load_state_dict(reg.state_dict())
    torch.testing.assert_close(fresh.penalty(weights), reg.penalty(weights))


def test_state_is_created_by_the_first_call_and_checks_k(weights):
    """Using a stateful regularizer on a different K is an error."""
    reg = _ema_reg()
    assert reg.smoothed_statistics() == {}
    reg.penalty(weights)
    assert set(reg.smoothed_statistics()) == {"trace_numerator", "trace_denominator"}
    with pytest.raises(ValueError, match="separate instance"):
        reg.penalty(simplex_weights(8, 3))


def test_bind_rejects_a_second_layer():
    """The averages are per-layer state: one regularizer, one layer."""
    reg = _ema_reg()
    first, second = nn.Linear(2, 2), nn.Linear(2, 2)
    reg.bind(5, layer=first)
    reg.bind(5, layer=first)
    with pytest.raises(ValueError, match="needs its own instance"):
        reg.bind(5, layer=second)
    R.Uniform(l2=1.0).bind(5, layer=second)  # nothing to bind: fine


def test_buffers_move_with_the_module(weights):
    """`.to(...)` moves and casts the averages."""
    reg = _ema_reg()
    reg.penalty(weights)
    reg.to(torch.float64)
    assert all(b.dtype == torch.float64 for b in reg.buffers())
    reg.to("meta")
    assert all(b.device.type == "meta" for b in reg.buffers())


def test_the_estimator_is_created_on_the_input_device(weights):
    """A lazily created state lives where the batch does."""
    reg = _ema_reg()
    reg.penalty(weights.to("meta"))
    assert all(b.device.type == "meta" for b in reg.buffers())


def test_float64_inputs_work_against_float32_buffers():
    """Smoothing casts between the buffer dtype and the batch dtype."""
    reg = _ema_reg()
    x = simplex_weights(16, 4).double()
    assert reg.penalty(x).dtype == torch.float64
    assert reg.penalty(x).dtype == torch.float64


def test_deepcopy_copies_the_averages(weights):
    """A copy has its own state."""
    reg = _ema_reg()
    reg.penalty(weights)
    clone = copy.deepcopy(reg)
    clone.penalty(weights)
    assert (reg.smoothing_steps, clone.smoothing_steps) == (1, 2)


def test_combined_state_dict_holds_each_childs_averages(weights):
    """Children are submodules: their buffers are in the composite's state."""
    combined = R.CombinedRegularizer([_ema_reg(), R.Uniform(l2=1.0)])
    combined.bind(5)
    keys = combined.state_dict()
    assert all(key.startswith("regularizers.0._estimator.") for key in keys)
    combined.eval()
    assert not combined.regularizers[0].training
