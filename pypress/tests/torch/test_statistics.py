"""Tests of the torch statistics, against closed forms."""

import math

import pytest
import torch

from pypress.torch import statistics

from .conftest import simplex_weights


def test_state_sizes_and_trace_terms(weights):
    """Sizes are column sums; the trace numerator sums squares."""
    torch.testing.assert_close(statistics.state_sizes(weights), weights.sum(0))
    torch.testing.assert_close(statistics.trace_numerator(weights), (weights**2).sum(0))
    assert statistics.compute("trace_denominator", weights).shape == (5,)


def test_mean_row_entropy_of_uniform_weights_is_log_k():
    """Uniform rows have the maximum entropy, log K."""
    uniform = torch.full((10, 4), 0.25)
    assert statistics.mean_row_entropy(uniform).item() == pytest.approx(math.log(4))


def test_degrees_of_freedom_of_hard_assignments_counts_non_empty_states():
    """trace(H) equals the number of non-empty states for hard assignments."""
    hard = torch.zeros(12, 5)
    hard[torch.arange(12), torch.arange(12) % 3] = 1.0  # states 0, 1, 2 only
    value = statistics.degrees_of_freedom(
        statistics.trace_numerator(hard), statistics.state_sizes(hard)
    )
    assert value.item() == pytest.approx(3.0)


def test_safe_log_replaces_only_exact_zeros():
    """Zeros become the floor; small positive values keep their true log."""
    p = torch.tensor([0.0, 1e-30, 0.5])
    result = statistics.safe_log(p)
    assert result[0].item() == -1e9
    assert result[1].item() == pytest.approx(math.log(1e-30), rel=1e-5)


def test_xlogx_has_a_finite_gradient_at_zero():
    """The gradient at exactly 0 is finite (and so is its value, 0)."""
    x = torch.tensor([0.0, 0.5], requires_grad=True)
    statistics.safe_xlogx(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert statistics.safe_xlogx(x)[0].item() == 0.0


def test_entropy_and_shares_of_a_dead_state():
    """A dead state has share 0 and contributes nothing to the entropy."""
    shares = statistics.state_shares(torch.tensor([2.0, 2.0, 0.0]))
    assert shares.tolist() == [0.5, 0.5, 0.0]
    assert statistics.entropy(shares).item() == pytest.approx(math.log(2))


def test_state_shares_of_all_zero_sizes_is_finite():
    """The 1e-8 guard keeps an all-zero batch from dividing by zero."""
    assert torch.isfinite(statistics.state_shares(torch.zeros(3))).all()


def test_names_and_shapes_come_from_core():
    """Names and shapes are the shared registry."""
    assert set(statistics.names()) == set(statistics._STATISTICS)
    assert statistics.shape("trace_numerator", 6) == (6,)
    assert statistics.shape("mean_row_entropy", 6) == ()
    assert simplex_weights(3, 2).shape == (3, 2)
