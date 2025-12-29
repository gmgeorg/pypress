import numpy as np
import pytest
import tensorflow as tf

# Import your module and functions.
# Adjust the import paths as necessary.
from pypress.keras.initializers import (
    PredictiveStateMeansInitializer,
    PredictiveStateParamsInitializer,
    _get_predictive_state_means_init,
)
from pypress.keras.activations import get_inverse_activation


def test_get_inverse_activation_linear():
    """Test inverse of linear activation is identity."""
    inverse_fn = get_inverse_activation("linear")
    x = tf.constant([1.0, 2.0, 3.0])
    result = inverse_fn(x)
    np.testing.assert_allclose(result.numpy(), x.numpy())


def test_get_inverse_activation_sigmoid():
    """Test inverse of sigmoid is logit."""
    inverse_fn = get_inverse_activation("sigmoid")
    # Test with valid probability values
    probs = tf.constant([0.5, 0.8, 0.2])
    logits = inverse_fn(probs)
    # Apply sigmoid to get back probabilities
    probs_recovered = tf.nn.sigmoid(logits)
    np.testing.assert_allclose(probs.numpy(), probs_recovered.numpy(), rtol=1e-5)


def test_get_inverse_activation_softplus():
    """Test inverse of softplus."""
    inverse_fn = get_inverse_activation("softplus")
    # Test with positive values (valid for softplus output)
    values = tf.constant([1.0, 2.0, 3.0])
    logits = inverse_fn(values)
    # Apply softplus to get back values
    values_recovered = tf.nn.softplus(logits)
    np.testing.assert_allclose(values.numpy(), values_recovered.numpy(), rtol=1e-5)


def test_get_inverse_activation_unknown():
    """Test that unknown activation raises ValueError."""
    with pytest.raises(ValueError, match="Unknown activation"):
        get_inverse_activation("unknown_activation")


def test_get_inverse_activation_no_inverse():
    """Test that activation without inverse raises ValueError."""
    with pytest.raises(ValueError, match="Unknown activation"):
        get_inverse_activation("relu")


def test_get_inverse_activation_tanh():
    """Test inverse of tanh is arctanh."""
    inverse_fn = get_inverse_activation("tanh")
    # Test with valid tanh output values (between -1 and 1)
    values = tf.constant([0.0, 0.5, -0.5, 0.9])
    logits = inverse_fn(values)
    # Apply tanh to get back values
    values_recovered = tf.nn.tanh(logits)
    np.testing.assert_allclose(values.numpy(), values_recovered.numpy(), rtol=1e-5)


def test_get_inverse_activation_exponential():
    """Test inverse of exponential is log."""
    inverse_fn = get_inverse_activation("exponential")
    # Test with positive values (valid for exp output)
    values = tf.constant([1.0, 2.0, 5.0, 10.0])
    logits = inverse_fn(values)
    # Apply exp to get back values
    values_recovered = tf.exp(logits)
    np.testing.assert_allclose(values.numpy(), values_recovered.numpy(), rtol=1e-5)


def test_get_inverse_activation_leaky_relu():
    """Test inverse of leaky_relu."""
    inverse_fn = get_inverse_activation("leaky_relu")
    # Test with both positive and negative values
    values = tf.constant([2.0, -0.4, 1.0, -0.1])
    logits = inverse_fn(values)
    # Apply leaky_relu to get back values (alpha=0.2 default)
    values_recovered = tf.nn.leaky_relu(logits, alpha=0.2)
    np.testing.assert_allclose(values.numpy(), values_recovered.numpy(), rtol=1e-5)


def test_get_inverse_activation_elu():
    """Test inverse of elu."""
    inverse_fn = get_inverse_activation("elu")
    # Test with positive and negative values
    # For ELU, negative values are bounded below by -alpha (default -1.0)
    values = tf.constant([2.0, -0.5, 1.0, -0.9])
    logits = inverse_fn(values)
    # Apply elu to get back values (alpha=1.0 default)
    values_recovered = tf.nn.elu(logits)
    np.testing.assert_allclose(values.numpy(), values_recovered.numpy(), rtol=1e-5)


def test_get_inverse_activation_softsign():
    """Test inverse of softsign."""
    inverse_fn = get_inverse_activation("softsign")
    # Test with values in valid range (-1, 1) for softsign output
    values = tf.constant([0.0, 0.5, -0.5, 0.8, -0.8])
    logits = inverse_fn(values)
    # Apply softsign to get back values: x / (1 + |x|)
    values_recovered = tf.nn.softsign(logits)
    np.testing.assert_allclose(values.numpy(), values_recovered.numpy(), rtol=1e-5)


def test_get_inverse_activation_selu():
    """Test inverse of selu."""
    inverse_fn = get_inverse_activation("selu")
    # Test with positive and negative values
    # For SELU, negative values are bounded below by -scale*alpha
    values = tf.constant([2.0, -1.0, 1.0, -0.5, 3.0])
    logits = inverse_fn(values)
    # Apply selu to get back values
    values_recovered = tf.nn.selu(logits)
    np.testing.assert_allclose(values.numpy(), values_recovered.numpy(), rtol=1e-5)


def test_get_predictive_state_means_init_with_float():
    """Test that a float init_values returns the expected tensor with linear activation."""
    init_values = 0.5
    n_states = 3
    units = 1
    # With linear activation, logits should equal init_values
    # Shape should be (n_states, units)
    expected = np.ones((n_states, units)) * init_values
    result = _get_predictive_state_means_init(
        init_values, n_states, units, activation="linear"
    )
    np.testing.assert_allclose(result.numpy(), expected)
    assert isinstance(result, tf.Variable)


def test_get_predictive_state_means_init_with_1d_array():
    """Test that a 1D numpy array works correctly with linear activation."""
    units = 4
    n_states = 3
    init_values = np.array([0.1, 0.2, 0.3, 0.4])
    # With linear activation, logits should equal init_values
    # Shape should be (n_states, units)
    expected = np.ones((n_states, units)) * init_values[np.newaxis, :]
    result = _get_predictive_state_means_init(
        init_values, n_states, units, activation="linear"
    )
    np.testing.assert_allclose(result.numpy(), expected)


def test_get_predictive_state_means_init_invalid_array():
    """Test that providing an invalid array shape raises a ValueError."""
    units = 4
    n_states = 3
    # Here, init_values is a 2D array (column vector) instead of a 1D row vector.
    init_values = np.array([[0.1], [0.2], [0.3], [0.4]])
    with pytest.raises(ValueError, match="must be a 1D array"):
        _get_predictive_state_means_init(
            init_values, n_states, units, activation="linear"
        )


def test_get_predictive_state_means_init_invalid_type():
    """Test that an invalid type for init_values raises a ValueError."""
    units = 4
    n_states = 3
    init_values = "not a valid type"
    with pytest.raises(ValueError, match="must be float or np.ndarray"):
        _get_predictive_state_means_init(
            init_values, n_states, units, activation="linear"
        )


def test_predictive_state_means_initializer_class_with_float():
    """Test the initializer class when init_values is a float with linear activation."""
    init_values = 0.5
    n_states = 3
    units = 4
    initializer = PredictiveStateMeansInitializer(
        init_values, n_states, activation="linear", units=units
    )
    # The initializer uses shape[-1] as units.
    shape = (n_states, units)
    result = initializer(shape)
    # With linear activation, logits should equal init_values
    # Shape should be (n_states, units)
    expected = np.ones((n_states, units)) * init_values
    np.testing.assert_allclose(result.numpy(), expected)


def test_predictive_state_means_initializer_class_with_array():
    """Test the initializer class when init_values is a 1D numpy array with linear activation."""
    units = 4
    n_states = 3
    init_values = np.array([0.1, 0.2, 0.3, 0.4])
    initializer = PredictiveStateMeansInitializer(
        init_values, n_states, activation="linear"
    )
    shape = (n_states, units)
    result = initializer(shape)
    # With linear activation, logits should equal init_values
    # Shape should be (n_states, units)
    expected = np.ones((n_states, units)) * init_values[np.newaxis, :]
    np.testing.assert_allclose(result.numpy(), expected)


def test_predictive_state_means_initializer_with_sigmoid():
    """Test that sigmoid activation properly converts probabilities to logits."""
    init_prob = 0.8  # Probability on original scale
    n_states = 3
    units = 1

    initializer = PredictiveStateMeansInitializer(
        init_values=init_prob, n_states=n_states, activation="sigmoid", units=units
    )
    shape = (n_states, units)
    logits = initializer(shape)

    # Apply sigmoid to logits - should recover init_prob
    # Shape should be (n_states, units)
    probs = tf.nn.sigmoid(logits)
    expected_probs = np.ones((n_states, units)) * init_prob
    np.testing.assert_allclose(probs.numpy(), expected_probs, rtol=1e-5)


def test_predictive_state_means_initializer_with_softplus():
    """Test that softplus activation properly converts values to logits."""
    init_values = 2.0  # Value on original scale
    n_states = 3
    units = 1

    initializer = PredictiveStateMeansInitializer(
        init_values=init_values, n_states=n_states, activation="softplus", units=units
    )
    shape = (n_states, units)
    logits = initializer(shape)

    # Apply softplus to logits - should recover init_values
    # Shape should be (n_states, units)
    values = tf.nn.softplus(logits)
    expected_values = np.ones((n_states, units)) * init_values
    np.testing.assert_allclose(values.numpy(), expected_values, rtol=1e-5)


def test_predictive_state_means_layer_with_init_values():
    """Test PredictiveStateMeans layer with init_values parameter."""
    from pypress.keras.layers import PredictiveStateMeans

    # Test with linear activation
    init_values = 0.5
    layer = PredictiveStateMeans(units=1, activation="linear", init_values=init_values)

    # Build the layer with dummy input
    n_states = 5
    dummy_input = tf.random.uniform((10, n_states))
    output = layer(dummy_input)

    # Check that state_conditional_means are initialized to init_values
    means = layer.state_conditional_means
    expected_means = np.ones((n_states, 1)) * init_values
    np.testing.assert_allclose(means.numpy(), expected_means, rtol=1e-5)

    # Output shape should be (batch_size, units)
    assert output.shape == (10, 1)


def test_predictive_state_means_layer_with_sigmoid_init():
    """Test PredictiveStateMeans layer with sigmoid activation and init_value.

    This is the ultimate test: verify that the layer outputs the correct probability
    on the original scale (not logits).
    """
    from pypress.keras.layers import PredictiveStateMeans

    # Initialize to 0.8 probability
    init_prob = 0.8
    layer = PredictiveStateMeans(units=1, activation="sigmoid", init_values=init_prob)

    # Build the layer
    n_states = 3
    batch_size = 5
    dummy_input = tf.random.uniform((batch_size, n_states))
    output = layer(dummy_input)

    # Check that state_conditional_means are approximately init_prob
    means = layer.state_conditional_means
    expected_means = np.ones((n_states, 1)) * init_prob
    np.testing.assert_allclose(means.numpy(), expected_means, rtol=1e-5)

    # THE ULTIMATE TEST: Verify output shape and values with one-hot encoding
    # Output shape should be (batch_size, units)
    assert output.shape == (batch_size, 1)

    # Test forward pass with one-hot encoding for each state
    # Each state should output a value close to init_prob after mixing
    for state_idx in range(n_states):
        # Create one-hot input: probability 1.0 for this state, 0.0 for others
        one_hot_input = tf.one_hot([state_idx], depth=n_states, dtype=tf.float32)
        state_output = layer(one_hot_input)

        # Output should be (1, 1) with value close to init_prob
        assert state_output.shape == (1, 1)
        np.testing.assert_allclose(
            state_output.numpy(),
            [[init_prob]],
            rtol=1e-5,
            err_msg=f"State {state_idx} output doesn't match init_prob={init_prob}",
        )


def test_predictive_state_params_initializer_gaussian():
    """Test PredictiveStateParamsInitializer with Gaussian (mean, std) parameters."""
    # Gaussian distribution with mean=0.0, std=1.0
    init_mean = 0.0
    init_std = 1.0
    n_states = 4
    activations = ["linear", "softplus"]  # mean: linear, std: softplus

    initializer = PredictiveStateParamsInitializer(
        init_values=[init_mean, init_std], n_states=n_states, activations=activations
    )

    shape = (n_states, 2)  # 2 params per state
    logits = initializer(shape)

    # Check shape
    assert logits.shape == shape

    # Apply activations to get back original values
    # Split by parameter
    mean_logits = logits[:, 0]
    std_logits = logits[:, 1]

    # Apply activations
    recovered_means = mean_logits  # linear activation
    recovered_stds = tf.nn.softplus(std_logits)  # softplus activation

    # Check recovered values match init_values
    expected_means = np.ones(n_states) * init_mean
    expected_stds = np.ones(n_states) * init_std

    np.testing.assert_allclose(recovered_means.numpy(), expected_means, rtol=1e-5)
    np.testing.assert_allclose(recovered_stds.numpy(), expected_stds, rtol=1e-5)


def test_predictive_state_params_layer_with_gaussian_init():
    """Test PredictiveStateParams layer with Gaussian (mean, std) initialization."""
    from pypress.keras.layers import PredictiveStateParams

    # Initialize Gaussian parameters: mean=0.5, std=2.0
    init_mean = 0.5
    init_std = 2.0
    n_states = 3

    layer = PredictiveStateParams(
        n_params_per_state=2,
        activations=["linear", "softplus"],
        init_values=[init_mean, init_std],
        flatten_output=False,
    )

    # Build the layer
    dummy_input = tf.random.uniform((10, n_states))
    output = layer(dummy_input)

    # Output shape should be (batch_size, n_states, n_params_per_state)
    assert output.shape == (10, n_states, 2)

    # Check that state_conditional_params are initialized correctly
    params = layer.state_conditional_params
    assert params.shape == (n_states, 2)

    # Extract mean and std parameters
    means = params[:, 0]
    stds = params[:, 1]

    # Check initialization values
    expected_means = np.ones(n_states) * init_mean
    expected_stds = np.ones(n_states) * init_std

    np.testing.assert_allclose(means.numpy(), expected_means, rtol=1e-5)
    np.testing.assert_allclose(stds.numpy(), expected_stds, rtol=1e-5)


def test_predictive_state_params_initializer_state_specific_gaussian():
    """Test PredictiveStateParams layer with state-specific Gaussian parameters.

    Tests initialization with different mean and std for each state:
    - State 0: mean=-10, std=1
    - State 1: mean=0, std=20
    - State 2: mean=100, std=5

    This is the ultimate test: create a layer, manually set its weights to
    state-specific values, and verify the layer outputs the correct parameters
    on the original scale (not logits).
    """
    from pypress.keras.layers import PredictiveStateParams

    n_states = 3
    n_params_per_state = 2
    activations = ["linear", "softplus"]  # mean: linear, std: softplus

    # Target parameters for each state (mean, std) on ORIGINAL scale
    target_params = np.array(
        [
            [-10.0, 1.0],  # State 0
            [0.0, 20.0],  # State 1
            [100.0, 5.0],  # State 2
        ]
    )

    # Create layer (init_values doesn't matter since we'll overwrite weights)
    layer = PredictiveStateParams(
        n_params_per_state=n_params_per_state,
        activations=activations,
        init_values=[0.0, 1.0],  # Will be overwritten
        flatten_output=False,
    )

    # Build the layer
    batch_size = 10
    dummy_input = tf.ones((batch_size, n_states))
    _ = layer(dummy_input)  # Build layer

    # Now manually set the weights to target values (convert to logits first)
    target_means = target_params[:, 0]  # Already on linear scale (no conversion needed)
    target_stds = target_params[:, 1]  # Need to convert to softplus logits

    # Get inverse activation function for softplus
    from pypress.keras.activations import get_inverse_activation

    inverse_softplus = get_inverse_activation("softplus")
    target_std_logits = inverse_softplus(tf.constant(target_stds, dtype=tf.float32))

    # Combine into weight matrix (n_states, n_params_per_state)
    target_logits = tf.stack([target_means, target_std_logits], axis=1)

    # Set the layer's weights
    layer._theta_logits.assign(target_logits)

    # Now get the layer's output parameters (should be on original scale)
    output_params = layer.state_conditional_params

    # Verify shape
    assert output_params.shape == (n_states, n_params_per_state)

    # Extract means and stds from layer output
    output_means = output_params[:, 0]
    output_stds = output_params[:, 1]

    # THE ULTIMATE TEST: Check that layer outputs match target params on ORIGINAL scale
    np.testing.assert_allclose(
        output_means.numpy(),
        target_params[:, 0],
        rtol=1e-5,
        err_msg="Means don't match expected values",
    )
    np.testing.assert_allclose(
        output_stds.numpy(),
        target_params[:, 1],
        rtol=1e-5,
        err_msg="Standard deviations don't match expected values",
    )

    # Also verify that the layer's forward pass outputs these same parameters
    # When we pass in a one-hot encoding, we should get back the corresponding state's params
    for state_idx in range(n_states):
        # Create one-hot input for this state
        one_hot_input = tf.one_hot([state_idx], depth=n_states, dtype=tf.float32)
        output = layer(one_hot_input)

        # Output should be (1, n_states, n_params_per_state) with flatten_output=False
        assert output.shape == (1, n_states, n_params_per_state)

        # Extract the parameters for the selected state
        state_params = output[0, state_idx, :]

        # Verify they match the target parameters for this state
        np.testing.assert_allclose(
            state_params.numpy(),
            target_params[state_idx, :],
            rtol=1e-5,
            err_msg=f"State {state_idx} parameters don't match",
        )
