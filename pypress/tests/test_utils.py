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


def test_kernel_matrix_pandas():
    """Test kernel_matrix with pandas DataFrame."""
    # Create sample weights
    weights = pd.DataFrame(
        {"s1": [0.1, 0.2, 0.7], "s2": [0.8, 0.1, 0.1], "s3": [0.5, 0.3, 0.2]},
        index=["a", "b", "c"],
    )

    # Compute kernel matrix
    K = utils.kernel_matrix(weights)

    # Check type and shape
    assert isinstance(K, pd.DataFrame)
    assert K.shape == (3, 3)

    # Check index is preserved
    pd.testing.assert_index_equal(K.index, weights.index)
    pd.testing.assert_index_equal(K.columns, weights.index)

    # Manually compute expected kernel
    norm_weights = utils.col_normalize(weights)
    expected_K = weights.dot(norm_weights.transpose())
    pd.testing.assert_frame_equal(K, expected_K)

    # Check symmetry property for this specific computation
    # Note: kernel is NOT necessarily symmetric in general, but we can verify
    # that our computation matches the formula


def test_kernel_matrix_numpy():
    """Test kernel_matrix with numpy array."""
    # Create sample weights
    weights = np.array([[0.1, 0.8, 0.5], [0.2, 0.1, 0.3], [0.7, 0.1, 0.2]])

    # Compute kernel matrix
    K = utils.kernel_matrix(weights)

    # Check type and shape
    assert isinstance(K, np.ndarray)
    assert K.shape == (3, 3)

    # Manually compute expected kernel
    col_sums = weights.sum(axis=0, keepdims=True)
    norm_weights = weights / col_sums
    expected_K = weights @ norm_weights.T

    np.testing.assert_allclose(K, expected_K, rtol=1e-10)


def test_kernel_matrix_trace_consistency():
    """Test that trace of kernel_matrix matches tr_kernel for TensorFlow."""
    # Create sample weights
    weights_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Compute kernel matrix using numpy/pandas
    K_np = utils.kernel_matrix(weights_np)
    trace_np = np.trace(K_np)

    # Compute trace using TensorFlow tr_kernel
    weights_tf = tf.constant(weights_np, dtype=tf.float32)
    trace_tf = utils.tr_kernel(weights_tf)

    # They should match
    np.testing.assert_allclose(trace_np, trace_tf.numpy(), rtol=1e-6)


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


def test_initialize_from_y_1d_means():
    """Test initialize_from_y with 1D data for PredictiveStateMeans."""
    np.random.seed(42)
    y = np.random.randn(100) + 5  # Mean around 5

    n_states = 3
    init_values = utils.initialize_from_y(y, n_states=n_states, return_params=False)

    # Should return array of shape (units, n_states) = (1, 3)
    assert init_values.shape == (1, n_states)

    # GMM should identify different cluster means (not all the same!)
    cluster_means = init_values[0, :]

    # Cluster means should span the data range
    assert cluster_means.min() < 5.0 < cluster_means.max(), (
        "GMM should find clusters spanning the data distribution"
    )

    # Average of cluster means should be close to overall mean
    np.testing.assert_allclose(cluster_means.mean(), 5.0, atol=1.0)


def test_initialize_from_y_2d_means():
    """Test initialize_from_y with 2D data for PredictiveStateMeans."""
    np.random.seed(42)
    y = np.random.randn(100, 3) + 2  # Mean around 2 for all 3 outputs

    units = 3
    n_states = 5
    init_values = utils.initialize_from_y(y, n_states=n_states, return_params=False)

    # Should return array of shape (units, n_states) = (3, 5)
    assert init_values.shape == (units, n_states)

    # All rows should have the same cluster means (same GMM fitted to flattened data)
    np.testing.assert_allclose(init_values[0, :], init_values[1, :])
    np.testing.assert_allclose(init_values[0, :], init_values[2, :])

    # Average of cluster means should be close to overall mean of 2
    np.testing.assert_allclose(init_values.mean(), 2.0, atol=1.0)


def test_initialize_from_y_params():
    """Test initialize_from_y with return_params=True for PredictiveStateParams."""
    np.random.seed(42)
    y = np.random.randn(200) + 10  # Mean 10, std 1

    n_states = 3
    init_values = utils.initialize_from_y(y, n_states=n_states, return_params=True)

    # Should return tuple of (means_array, stds_array)
    assert isinstance(init_values, tuple)
    assert len(init_values) == 2

    means_array, stds_array = init_values

    # Each should be shape (n_states,)
    assert means_array.shape == (n_states,)
    assert stds_array.shape == (n_states,)

    # Means should be close to 10
    np.testing.assert_allclose(means_array, 10.0, atol=2.0)

    # Stds should be close to 1
    np.testing.assert_allclose(stds_array, 1.0, atol=1.0)


def test_initialize_from_y_pandas_series():
    """Test initialize_from_y with pandas Series input."""
    np.random.seed(42)
    y = pd.Series(np.random.randn(100) + 3)

    n_states = 2
    init_values = utils.initialize_from_y(y, n_states=n_states, return_params=False)

    assert init_values.shape == (1, n_states)

    # Average of cluster means should be close to overall mean
    np.testing.assert_allclose(init_values.mean(), 3.0, atol=1.0)


def test_initialize_from_y_with_layer():
    """Integration test: verify initialize_from_y works with actual layer."""
    from pypress.keras.layers import PredictiveStateMeans

    np.random.seed(42)
    y = np.random.randn(100) + 7

    n_states = 4
    # Get initialization from data
    init_values = utils.initialize_from_y(y, n_states=n_states, return_params=False)

    # init_values should be shape (1, 4)
    assert init_values.shape == (1, n_states)

    # Use it to initialize layer
    layer = PredictiveStateMeans(units=1, activation="linear", init_values=init_values)

    # Build the layer
    dummy_input = tf.ones((10, n_states))
    _ = layer(dummy_input)

    # Check that state_conditional_means are initialized to the cluster means
    means = layer.state_conditional_means  # Shape: (n_states, units)
    expected_values = init_values.T  # Transpose to (n_states, units)

    # State means should match the initialized values
    np.testing.assert_allclose(means.numpy(), expected_values, atol=1.0)
