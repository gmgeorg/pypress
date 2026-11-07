"""The torch statistics equal the Keras ones to float32 rounding."""

import numpy as np
import pytest
import tensorflow as tf
import torch

from pypress.keras import statistics as keras_stats
from pypress.torch import statistics as torch_stats

from .data import ATOL, RTOL

_WEIGHT_STATISTICS = sorted(torch_stats._STATISTICS)


def test_the_same_statistics_exist():
    """Both backends compute the same named statistics, with the same shapes."""
    assert keras_stats.names() == torch_stats.names()
    assert sorted(keras_stats._STATISTICS) == _WEIGHT_STATISTICS
    assert keras_stats.shape("state_sizes", 7) == torch_stats.shape("state_sizes", 7)


@pytest.mark.parametrize("name", _WEIGHT_STATISTICS)
def test_statistic_values_match(name, batch):
    """Every statistic has equal values on random, one-hot and degenerate batches."""
    expected = keras_stats.compute(name, tf.constant(batch)).numpy()
    actual = torch_stats.compute(name, torch.tensor(batch)).numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name", _WEIGHT_STATISTICS)
def test_statistic_gradients_match(name, batch):
    """Gradients of each statistic's sum are equal, including at exact zeros."""
    x = tf.Variable(batch)
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(keras_stats.compute(name, x))
    expected = tape.gradient(total, x).numpy()

    y = torch.tensor(batch, requires_grad=True)
    torch_stats.compute(name, y).sum().backward()
    np.testing.assert_allclose(y.grad.numpy(), expected, rtol=RTOL, atol=ATOL)


def test_safe_log_and_xlogx_match_at_zero():
    """The zero handling (value and finite gradient) agrees."""
    values = np.array([0.0, 1e-30, 1e-6, 0.3, 1.0], dtype=np.float32)
    np.testing.assert_allclose(
        torch_stats.safe_log(torch.tensor(values)).numpy(),
        keras_stats.safe_log(tf.constant(values)).numpy(),
        rtol=RTOL,
    )
    x = tf.Variable(values)
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(keras_stats.safe_xlogx(x))
    expected = tape.gradient(total, x).numpy()
    y = torch.tensor(values, requires_grad=True)
    torch_stats.safe_xlogx(y).sum().backward()
    np.testing.assert_allclose(y.grad.numpy(), expected, rtol=RTOL, atol=ATOL)
    assert np.all(np.isfinite(y.grad.numpy()))


def test_state_shares_entropy_and_degrees_of_freedom_match(batch):
    """The derived helpers (shares, entropy, trace ratio) agree."""
    sizes = keras_stats.state_sizes(tf.constant(batch))
    shares_k = keras_stats.state_shares(sizes)
    shares_t = torch_stats.state_shares(torch.tensor(batch).sum(dim=0))
    np.testing.assert_allclose(shares_t.numpy(), shares_k.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        torch_stats.entropy(shares_t).numpy(),
        keras_stats.entropy(shares_k).numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    numerator = keras_stats.trace_numerator(tf.constant(batch))
    np.testing.assert_allclose(
        torch_stats.degrees_of_freedom(
            torch_stats.trace_numerator(torch.tensor(batch)),
            torch_stats.state_sizes(torch.tensor(batch)),
        ).numpy(),
        keras_stats.degrees_of_freedom(numerator, sizes).numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
