import numpy as np
import pytest
import tensorflow as tf

# Import your module and functions.
# Adjust the import paths as necessary.
from pypress.keras.initializers import (
    PredictiveStateMeansInitializer,
    _get_predictive_state_means_init,
)


def test_get_predictive_state_means_init_with_float():
    """Test that a float init_value returns the expected tensor."""
    init_value = 0.5
    n_states = 3
    expected = np.ones((1, n_states)) * init_value
    result = _get_predictive_state_means_init(init_value, n_states, 1)
    np.testing.assert_allclose(result.numpy(), expected)
    assert isinstance(result, tf.Variable)


def test_get_predictive_state_means_init_with_1d_array():
    """Test that a 1D numpy array works correctly."""
    units = 4
    n_states = 3
    init_value = np.array([0.1, 0.2, 0.3, 0.4])
    expected = init_value[:, np.newaxis] * np.ones((units, n_states))
    result = _get_predictive_state_means_init(init_value, n_states, units)
    np.testing.assert_allclose(result.numpy(), expected)


def test_get_predictive_state_means_init_invalid_array():
    """Test that providing an invalid array shape raises a ValueError."""
    units = 4
    n_states = 3
    # Here, init_value is a 2D array (column vector) instead of a 1D row vector.
    init_value = np.array([[0.1], [0.2], [0.3], [0.4]])
    with pytest.raises(ValueError):
        _get_predictive_state_means_init(init_value, n_states, units)


def test_get_predictive_state_means_init_invalid_type():
    """Test that an invalid type for init_value raises a ValueError."""
    units = 4
    n_states = 3
    init_value = "not a valid type"
    with pytest.raises(ValueError):
        _get_predictive_state_means_init(init_value, n_states, units)


def test_predictive_state_means_initializer_class_with_float():
    """Test the initializer class when init_value is a float."""
    init_value = 0.5
    n_states = 3
    units = 4
    initializer = PredictiveStateMeansInitializer(init_value, n_states, units=units)
    # The initializer uses shape[-1] as units.
    shape = (2, units)  # e.g., two outputs each with `units` features.
    result = initializer(shape)
    expected = np.ones((units, n_states)) * init_value
    np.testing.assert_allclose(result.numpy(), expected)


def test_predictive_state_means_initializer_class_with_array():
    """Test the initializer class when init_value is a 1D numpy array."""
    units = 4
    n_states = 3
    init_value = np.array([0.1, 0.2, 0.3, 0.4])
    initializer = PredictiveStateMeansInitializer(init_value, n_states)
    shape = (2, units)
    result = initializer(shape)
    expected = init_value[:, np.newaxis] * np.ones((units, n_states))
    np.testing.assert_allclose(result.numpy(), expected)
