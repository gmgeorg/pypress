"""Module for testing utils module."""

from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from .. import utils


def _test_weights() -> Tuple[pd.DataFrame, pd.Series]:
    # Create a sample weights DataFrame.
    weights = pd.DataFrame(
        {"0": [0.1, 0.2, 0.8], "1": [0.3, 0.0, 0.1], "2": [0.6, 0.8, 0.1]},
        index=["foo", "bar", "hello"],
    )
    # Expected state sizes are computed as column sums.
    expected_size = pd.Series([1.1, 0.4, 1.5], index=["0", "1", "2"])
    expected_size.index.name = "state"
    return weights, expected_size


def test_state_size():
    """Test that state_size correctly computes the column sums."""
    w, expected_size = _test_weights()
    pd.testing.assert_series_equal(utils.state_size(w), expected_size)


def test_col_normalizer():
    """Test that col_normalize normalizes each column to sum to 1."""
    w, _ = _test_weights()
    norm_w = utils.col_normalize(w)
    col_sums_expected = pd.Series([1.0, 1.0, 1.0], index=["0", "1", "2"])
    col_sums_expected.index.name = "state"
    pd.testing.assert_series_equal(norm_w.sum(axis=0), col_sums_expected)


def test_tf_state_size():
    """Test that tf_state_size computes the same column sums as the pandas version."""
    w, _ = _test_weights()
    tf_s = utils.tf_state_size(w)
    s = utils.state_size(w)
    np.testing.assert_allclose(s.values, tf_s.numpy())


def test_tf_col_normalize():
    """Test that tf_col_normalize produces the same result as the pandas version."""
    w, _ = _test_weights()
    tf_nw = utils.tf_col_normalize(w)
    nw = utils.col_normalize(w)
    np.testing.assert_allclose(nw.values, tf_nw.numpy())


# --- Tests for tr_kernel ---
def test_tr_kernel_identity():
    # For an identity matrix of shape (2,2), each column is already normalized (L2 norm=1),
    # so tf_col_normalize(identity) should be identity, and then
    # tr_kernel should equal the sum of diag(identity^T * identity) = 2.
    weights = tf.eye(2, dtype=tf.float32)
    result = utils.tr_kernel(weights)
    expected = 2.0
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


def test_tr_kernel_custom():
    # Test with a custom matrix.
    # For example, use a 3x2 matrix.
    weights = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    # Normalize columns manually:
    col_sum = tf.reduce_sum(weights, axis=0)
    normalized = weights / col_sum
    # Compute kernel: K = normalized^T * weights, trace = sum(diag(K))
    kernel = tf.matmul(tf.transpose(normalized), weights)
    expected = tf.reduce_sum(tf.linalg.diag_part(kernel))
    result = utils.tr_kernel(weights)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


def test_agg_data_by_state_mean():
    """Create a small numeric DataFrame as data."""
    data = pd.DataFrame(
        {
            "feature1": [10, 20, 30],
            "feature2": [1, 2, 3],
            "non_numeric": ["a", "b", "c"],
        }
    )
    # Create weights DataFrame with same number of rows and two states.
    weights = pd.DataFrame({"state1": [1, 2, 3], "state2": [4, 5, 6]})
    # agg_state = (norm_weights^T dot df_numeric)^T, so shape should be (2,2) (features x states)
    agg_state = utils.agg_data_by_state(data, weights, agg_fn="mean")

    # Expected result can be computed manually:
    # For feature1:
    #   For state1: (1/6)*10 + (2/6)*20 + (3/6)*30 = (10 + 40 + 90)/6 = 140/6 = 23.3333
    #   For state2: (4/15)*10 + (5/15)*20 + (6/15)*30 = (40 + 100 + 180)/15 = 320/15 = 21.3333
    # For feature2:
    #   For state1: (1/6)*1 + (2/6)*2 + (3/6)*3 = (1 + 4 + 9)/6 = 14/6 ≈ 2.3333
    #   For state2: (4/15)*1 + (5/15)*2 + (6/15)*3 = (4 + 10 + 18)/15 = 32/15 ≈ 2.1333
    expected = pd.DataFrame(
        {"state1": [23.3333, 2.3333], "state2": [21.3333, 2.1333]},
        index=["feature1", "feature2"],
    )
    # Because of floating point differences, use assert_frame_equal with a tolerance.
    pd.testing.assert_frame_equal(agg_state.round(4), expected.round(4))
