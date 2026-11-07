"""Tests for pypress.analysis and the in-graph statistics it mirrors."""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from pypress import analysis
from pypress.keras import initializers, statistics


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
    pd.testing.assert_series_equal(analysis.state_size(w), expected_size)


def test_col_normalizer():
    """Test that col_normalize normalizes each column to sum to 1."""
    w, _ = _test_weights()
    norm_w = analysis.col_normalize(w)
    col_sums_expected = pd.Series([1.0, 1.0, 1.0], index=["0", "1", "2"])
    col_sums_expected.index.name = "state"
    pd.testing.assert_series_equal(norm_w.sum(axis=0), col_sums_expected)


def test_graph_state_sizes_match_analysis():
    """Test that tf_state_size computes the same column sums as the pandas version."""
    w, _ = _test_weights()
    tf_s = statistics.state_sizes(tf.constant(w.values))
    s = analysis.state_size(w)
    np.testing.assert_allclose(s.values, tf_s.numpy())


def test_graph_col_normalize_matches_analysis():
    """Test that tf_col_normalize produces the same result as the pandas version."""
    w, _ = _test_weights()
    values = tf.constant(w.values)
    tf_nw = values / statistics.state_sizes(values)
    nw = analysis.col_normalize(w)
    np.testing.assert_allclose(nw.values, tf_nw.numpy())


def test_degrees_of_freedom_identity():
    # For an identity matrix of shape (2,2), each column is already normalized (L2 norm=1),
    # so tf_col_normalize(identity) should be identity, and then
    # tr_kernel should equal the sum of diag(identity^T * identity) = 2.
    weights = tf.eye(2, dtype=tf.float32)
    result = statistics.degrees_of_freedom(
        statistics.trace_numerator(weights), statistics.state_sizes(weights)
    )
    expected = 2.0
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


def test_degrees_of_freedom_custom():
    # Test with a custom matrix.
    # For example, use a 3x2 matrix.
    weights = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    # Normalize columns manually:
    col_sum = tf.reduce_sum(weights, axis=0)
    normalized = weights / col_sum
    # Compute smoother matrix: H = normalized^T * weights, trace = sum(diag(H))
    kernel = tf.matmul(tf.transpose(normalized), weights)
    expected = tf.reduce_sum(tf.linalg.diag_part(kernel))
    result = statistics.degrees_of_freedom(
        statistics.trace_numerator(weights), statistics.state_sizes(weights)
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


def test_smoother_matrix_pandas():
    """Test kernel_matrix with pandas DataFrame."""
    # Create sample weights
    weights = pd.DataFrame(
        {"s1": [0.1, 0.2, 0.7], "s2": [0.8, 0.1, 0.1], "s3": [0.5, 0.3, 0.2]},
        index=["a", "b", "c"],
    )

    # Compute kernel matrix
    H = analysis.smoother_matrix(weights)

    # Check type and shape
    assert isinstance(H, pd.DataFrame)
    assert H.shape == (3, 3)

    # Check index is preserved
    pd.testing.assert_index_equal(H.index, weights.index)
    pd.testing.assert_index_equal(H.columns, weights.index)

    # Manually compute expected kernel
    norm_weights = analysis.col_normalize(weights)
    expected_H = weights.dot(norm_weights.transpose())
    pd.testing.assert_frame_equal(H, expected_H)

    # Check symmetry property for this specific computation
    # Note: kernel is NOT necessarily symmetric in general, but we can verify
    # that our computation matches the formula


def test_smoother_matrix_numpy():
    """Test kernel_matrix with numpy array."""
    # Create sample weights
    weights = np.array([[0.1, 0.8, 0.5], [0.2, 0.1, 0.3], [0.7, 0.1, 0.2]])

    # Compute kernel matrix
    H = analysis.smoother_matrix(weights)

    # Check type and shape
    assert isinstance(H, np.ndarray)
    assert H.shape == (3, 3)

    # Manually compute expected kernel
    col_sums = weights.sum(axis=0, keepdims=True)
    norm_weights = weights / col_sums
    expected_H = weights @ norm_weights.T

    np.testing.assert_allclose(H, expected_H, rtol=1e-10)


def test_smoother_matrix_trace_consistency():
    """Test that trace of kernel_matrix matches tr_kernel for TensorFlow."""
    # Create sample weights
    weights_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Compute kernel matrix using numpy/pandas
    K_np = analysis.smoother_matrix(weights_np)
    trace_np = np.trace(K_np)

    # Compute trace using TensorFlow tr_kernel
    weights_tf = tf.constant(weights_np, dtype=tf.float32)
    trace_tf = statistics.degrees_of_freedom(
        statistics.trace_numerator(weights_tf), statistics.state_sizes(weights_tf)
    )

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
    agg_state = analysis.agg_data_by_state(data, weights, agg_fn="mean")

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
    init_values = initializers.initialize_from_y(
        y, n_states=n_states, return_params=False
    )

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
    init_values = initializers.initialize_from_y(
        y, n_states=n_states, return_params=False
    )

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
    n_params_per_state = 2
    init_values = initializers.initialize_from_y(
        y, n_states=n_states, return_params=True
    )

    # Should return array of shape (n_params_per_state, n_states)
    assert isinstance(init_values, np.ndarray)
    assert init_values.shape == (n_params_per_state, n_states)

    # Row 0: means, Row 1: stds
    means_array = init_values[0, :]
    stds_array = init_values[1, :]

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
    init_values = initializers.initialize_from_y(
        y, n_states=n_states, return_params=False
    )

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
    init_values = initializers.initialize_from_y(
        y, n_states=n_states, return_params=False
    )

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


# -------- Tests for safe_log --------


def test_safe_log_matches_log_for_positive_values():
    """For p > 0, safe_log should exactly match tf.math.log."""
    p = tf.constant([0.001, 0.1, 0.5, 1.0], dtype=tf.float32)
    np.testing.assert_allclose(
        statistics.safe_log(p).numpy(), tf.math.log(p).numpy(), atol=1e-6
    )


def test_safe_log_zero_uses_neg_inf_floor():
    """Exact zeros are floored at neg_inf instead of producing -inf/NaN."""
    p = tf.constant([0.0, 0.5], dtype=tf.float32)
    result = statistics.safe_log(p).numpy()
    assert result[0] == -1e9
    np.testing.assert_allclose(result[1], np.log(0.5), atol=1e-6)


def test_safe_log_custom_neg_inf():
    """A custom neg_inf value is used for exact zeros."""
    p = tf.constant([0.0], dtype=tf.float32)
    result = statistics.safe_log(p, neg_inf=-42.0).numpy()
    np.testing.assert_allclose(result, [-42.0], atol=1e-6)


def test_safe_log_gradient_finite_at_zero():
    """Gradient at p == 0 must be finite (0), not NaN, unlike log(p) directly."""
    p = tf.Variable([0.0, 0.5], dtype=tf.float32)
    with tf.GradientTape() as tape:
        y = statistics.safe_log(p)
    grad = tape.gradient(y, p).numpy()
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(grad[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(grad[1], 1.0 / 0.5, atol=1e-4)


def test_safe_log_preserves_shape_and_dtype():
    """safe_log preserves the input tensor's shape and dtype."""
    p = tf.constant([[0.2, 0.0], [0.5, 0.3]], dtype=tf.float64)
    result = statistics.safe_log(p)
    assert result.shape == p.shape
    assert result.dtype == p.dtype


# -------- Tests for safe_xlogx --------


def test_safe_xlogx_matches_xlogx_for_positive_values():
    """For x > 0, safe_xlogx should exactly match x * log(x)."""
    x = tf.constant([0.001, 0.1, 0.5, 1.0], dtype=tf.float32)
    expected = (x * tf.math.log(x)).numpy()
    np.testing.assert_allclose(statistics.safe_xlogx(x).numpy(), expected, atol=1e-6)


def test_safe_xlogx_zero_is_zero():
    """x*log(x) -> 0 as x -> 0; safe_xlogx should return exactly 0 at x == 0."""
    x = tf.constant([0.0, 0.5], dtype=tf.float32)
    result = statistics.safe_xlogx(x).numpy()
    assert result[0] == 0.0
    np.testing.assert_allclose(result[1], 0.5 * np.log(0.5), atol=1e-6)


def test_safe_xlogx_gradient_bounded_at_zero():
    """Gradient at x == 0 must be finite and small, unlike x*safe_log(x) which
    reintroduces safe_log's large neg_inf floor via the product rule."""
    x = tf.Variable([0.0, 0.5], dtype=tf.float32)
    with tf.GradientTape() as tape:
        y = statistics.safe_xlogx(x)
    grad = tape.gradient(y, x).numpy()
    assert np.all(np.isfinite(grad))
    # Should be nowhere near safe_log's -1e9 floor leaking through the product rule.
    assert abs(grad[0]) < 100.0
    np.testing.assert_allclose(grad[1], np.log(0.5) + 1.0, atol=1e-4)


def test_safe_xlogx_preserves_shape_and_dtype():
    """safe_xlogx preserves the input tensor's shape and dtype."""
    x = tf.constant([[0.2, 0.0], [0.5, 0.3]], dtype=tf.float64)
    result = statistics.safe_xlogx(x)
    assert result.shape == x.shape
    assert result.dtype == x.dtype


def test_numpy_degrees_of_freedom_equals_trace_of_smoother_matrix():
    """analysis.degrees_of_freedom is trace(H) without forming H."""
    weights = np.random.default_rng(0).dirichlet(np.ones(4), size=50)
    np.testing.assert_allclose(
        analysis.degrees_of_freedom(weights),
        np.trace(analysis.smoother_matrix(weights)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        analysis.degrees_of_freedom(pd.DataFrame(weights)),
        analysis.degrees_of_freedom(weights),
    )


def test_col_normalize_numpy():
    """col_normalize also takes arrays."""
    weights = np.random.default_rng(1).dirichlet(np.ones(3), size=10)
    np.testing.assert_allclose(analysis.col_normalize(weights).sum(axis=0), 1.0)


def test_agg_data_by_state_accepts_arrays():
    """Arrays in, array out; mixed inputs give a DataFrame labelled by the DataFrame."""
    data = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
    weights = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    expected = np.array([[140 / 6, 320 / 15], [14 / 6, 32 / 15]])

    result = analysis.agg_data_by_state(data, weights)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)

    weights_df = pd.DataFrame(weights, columns=["s0", "s1"], index=[7, 8, 9])
    mixed = analysis.agg_data_by_state(data, weights_df)
    assert list(mixed.columns) == ["s0", "s1"]
    np.testing.assert_allclose(mixed.values, expected)

    data_df = pd.DataFrame(data, columns=["a", "b"], index=[7, 8, 9])
    mixed = analysis.agg_data_by_state(data_df, weights)
    assert list(mixed.index) == ["a", "b"]
    np.testing.assert_allclose(mixed.values, expected)


def test_agg_data_by_state_rejects_mismatched_rows():
    """Data and weights must describe the same observations."""
    with pytest.raises(ValueError, match="same observations"):
        analysis.agg_data_by_state(np.ones((3, 2)), np.ones((4, 2)) / 2)
