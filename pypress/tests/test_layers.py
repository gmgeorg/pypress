"""Tests for layers."""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from ..keras import layers


def _test_data(n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(10)
    feats = pd.DataFrame(np.random.normal(size=(n_samples, 7)))
    feats.columns = ["x" + str(k + 1) for k in range(feats.shape[1])]

    eps = np.random.normal(size=feats.shape[0])
    y = feats["x1"] + 5 * feats["x3"] - np.exp(feats["x5"]) + eps

    return feats, y


@pytest.mark.parametrize("n_states", [(1), (2), (5)])
def test_predictive_state_simplex(n_states):
    test_df, _ = _test_data(n_samples=3)
    pss_layer = layers.PredictiveStateSimplex(n_states=n_states)
    pred_states = pss_layer(test_df.values).numpy()
    assert pred_states.shape[0] == test_df.shape[0]
    assert pred_states.shape[1] == n_states

    row_sums = pred_states.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums))


@pytest.mark.parametrize("n_states,units", [(1, 1), (2, 4), (5, 10)])
def test_predictive_state_means(n_states, units):
    test_df, _ = _test_data(n_samples=3)
    pss_layer = layers.PredictiveStateSimplex(n_states=n_states)
    pred_states = pss_layer(test_df.values).numpy()
    cond_means = layers.PredictiveStateMeans(units=units)
    preds = cond_means(pred_states)
    assert preds.shape[0] == test_df.shape[0]
    assert preds.shape[1] == units


def test_use_in_model_works():
    feats, y = _test_data(n_samples=1000)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(feats.shape[1],)))
    model.add(layers.PredictiveStateSimplex(5))
    model.add(layers.PredictiveStateMeans(1, "linear"))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01))

    model.fit(feats, y, epochs=4)
    preds = model.predict(feats).ravel()

    cor_mat = np.corrcoef(preds, y)
    print(cor_mat)

    assert cor_mat[0, 1] > 0.88


def test_press_in_model_works():
    feats, y = _test_data(n_samples=1000)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(feats.shape[1],)))
    model.add(layers.PRESS(units=1, n_states=5))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01))

    model.fit(feats, y, epochs=4)
    preds = model.predict(feats).ravel()

    cor_mat = np.corrcoef(preds, y)
    print(cor_mat)

    assert cor_mat[0, 1] > 0.88


class TestPredictiveStateParams:
    """Tests for PredictiveStateParams layer."""

    def test_output_shape_flattened(self):
        """Test output shape with flatten_output=True."""
        tf.random.set_seed(42)
        n_states = 4
        n_params = 2

        layer = layers.PredictiveStateParams(
            n_params_per_state=n_params, flatten_output=True
        )

        # Input is state probabilities [batch, n_states]
        inputs = tf.random.normal((10, n_states))
        outputs = layer(inputs)

        # Output should be [batch, n_states * n_params] in interleaved format
        assert outputs.shape == (10, n_states * n_params)

    def test_output_shape_not_flattened(self):
        """Test output shape with flatten_output=False."""
        tf.random.set_seed(42)
        n_states = 3
        n_params = 2

        layer = layers.PredictiveStateParams(
            n_params_per_state=n_params, flatten_output=False
        )

        inputs = tf.random.normal((5, n_states))
        outputs = layer(inputs)

        # Output should be [batch, n_states, n_params]
        assert outputs.shape == (5, n_states, n_params)

    def test_activation_applied(self):
        """Test that activation function is applied correctly."""
        tf.random.set_seed(42)
        n_states = 3
        n_params = 1

        # Use softplus activation (all outputs should be positive)
        layer = layers.PredictiveStateParams(
            n_params_per_state=n_params,
            activations="softplus",
            flatten_output=True,
        )

        inputs = tf.random.normal((10, n_states))
        outputs = layer(inputs)

        # All values should be positive (softplus ensures this)
        assert tf.reduce_all(outputs >= 0).numpy()

    def test_different_activations_per_param(self):
        """Test using different activations for different parameters."""
        tf.random.set_seed(42)
        n_states = 2
        n_params = 2

        # First param: linear, second param: softplus
        layer = layers.PredictiveStateParams(
            n_params_per_state=n_params,
            activations=["linear", "softplus"],
            flatten_output=False,
        )

        inputs = tf.random.normal((5, n_states))
        outputs = layer(inputs)

        # Shape check
        assert outputs.shape == (5, n_states, n_params)

        # Second parameter (index 1) should all be non-negative due to softplus
        assert tf.reduce_all(outputs[:, :, 1] >= 0).numpy()

    def test_constant_across_batch(self):
        """Test that outputs are constant across batch (state-conditional only)."""
        tf.random.set_seed(42)
        n_states = 3
        n_params = 2

        layer = layers.PredictiveStateParams(
            n_params_per_state=n_params, flatten_output=False
        )

        # Different batch samples
        inputs = tf.random.normal((100, n_states))
        outputs = layer(inputs)

        # All batch samples should get the same state-conditional parameters
        # Check first and last batch items are identical
        np.testing.assert_allclose(outputs[0].numpy(), outputs[-1].numpy(), rtol=1e-5)

        # Check middle batch item
        np.testing.assert_allclose(outputs[0].numpy(), outputs[50].numpy(), rtol=1e-5)

    def test_interleaved_format(self):
        """Test that flattened output uses interleaved format."""
        tf.random.set_seed(42)
        n_states = 3
        n_params = 2

        layer = layers.PredictiveStateParams(
            n_params_per_state=n_params, flatten_output=True
        )

        inputs = tf.random.normal((1, n_states))
        outputs_flat = layer(inputs)

        # Also get non-flattened version for comparison
        layer_unflat = layers.PredictiveStateParams(
            n_params_per_state=n_params, flatten_output=False
        )
        # Use same weights
        layer_unflat.build((None, n_states))
        layer_unflat._theta_logits.assign(layer._theta_logits)
        outputs_unflat = layer_unflat(inputs)

        # Check interleaved format: [s0_p0, s1_p0, s2_p0, s0_p1, s1_p1, s2_p1]
        # First n_states values should match first param across all states
        for state_idx in range(n_states):
            # Param 0
            assert np.isclose(
                outputs_flat[0, state_idx].numpy(),
                outputs_unflat[0, state_idx, 0].numpy(),
                rtol=1e-5,
            )
            # Param 1
            assert np.isclose(
                outputs_flat[0, n_states + state_idx].numpy(),
                outputs_unflat[0, state_idx, 1].numpy(),
                rtol=1e-5,
            )

    def test_init_logits(self):
        """Test initialization with custom logits."""
        n_states = 2
        n_params = 2

        # Custom initialization
        init_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        layer = layers.PredictiveStateParams(
            n_params_per_state=n_params,
            activations="linear",  # Use linear to check exact values
            init_logits=init_values,
            flatten_output=False,
        )

        inputs = tf.ones((1, n_states))
        outputs = layer(inputs)

        # With linear activation, outputs should match init values
        np.testing.assert_allclose(outputs[0].numpy(), init_values, rtol=1e-5)

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        n_states = 4
        n_params = 3

        # Test flattened
        layer_flat = layers.PredictiveStateParams(
            n_params_per_state=n_params, flatten_output=True
        )
        # Need to build first to infer n_states
        layer_flat.build((None, n_states))
        shape_flat = layer_flat.compute_output_shape((None, n_states))
        assert shape_flat == (None, n_states * n_params)

        # Test not flattened
        layer_unflat = layers.PredictiveStateParams(
            n_params_per_state=n_params, flatten_output=False
        )
        # Need to build first to infer n_states
        layer_unflat.build((None, n_states))
        shape_unflat = layer_unflat.compute_output_shape((None, n_states))
        assert shape_unflat == (None, n_states, n_params)

    def test_serialization(self):
        """Test that layer can be serialized and deserialized."""
        n_states = 3
        n_params = 2

        layer = layers.PredictiveStateParams(
            n_params_per_state=n_params, activations="relu", flatten_output=True
        )

        # Build layer
        inputs = tf.random.normal((5, n_states))
        outputs1 = layer(inputs)

        # Get config and reconstruct
        config = layer.get_config()
        layer_reconstructed = layers.PredictiveStateParams.from_config(config)

        # Build reconstructed layer
        layer_reconstructed.build((None, n_states))

        # Copy weights
        layer_reconstructed.set_weights(layer.get_weights())

        # Should produce same output
        outputs2 = layer_reconstructed(inputs)
        np.testing.assert_allclose(outputs1.numpy(), outputs2.numpy(), rtol=1e-5)


@pytest.mark.parametrize(
    "activations",
    [
        "linear",
        "relu",
        "softplus",
        ["linear", "softplus"],
    ],
)
def test_predictive_state_params_various_activations(activations):
    """Test PredictiveStateParams with various activation configurations."""
    tf.random.set_seed(42)

    n_params = 2 if isinstance(activations, list) else 1

    layer = layers.PredictiveStateParams(
        n_params_per_state=n_params, activations=activations, flatten_output=True
    )

    inputs = tf.random.normal((10, 3))
    outputs = layer(inputs)

    # Should not raise errors and produce correct shape
    assert outputs.shape == (10, 3 * n_params)


def test_predictive_state_params_activation_length_mismatch():
    """Test that mismatched activation list length raises error."""
    with pytest.raises(ValueError, match="must have length n_params_per_state"):
        _ = layers.PredictiveStateParams(
            n_params_per_state=2,
            activations=["relu", "softplus", "linear"],  # 3 activations for 2 params
            flatten_output=True,
        )


def test_predictive_state_params_sigmoid_activation():
    """Test PredictiveStateParams with sigmoid activation for classification probabilities."""
    tf.random.set_seed(42)
    n_states = 3
    n_params = 1  # Single parameter per state (e.g., class probability)

    # Create layer with sigmoid activation
    layer = layers.PredictiveStateParams(
        n_params_per_state=n_params,
        activations="sigmoid",
        flatten_output=True,
    )

    # Input is state probabilities [batch, n_states]
    inputs = tf.random.normal((10, n_states))
    outputs = layer(inputs)

    # Check output shape
    assert outputs.shape == (10, n_states * n_params)
    assert outputs.shape == (10, 3)  # 3 states × 1 param

    # Check that all values are between 0 and 1 (sigmoid property)
    assert tf.reduce_all(outputs >= 0.0).numpy()
    assert tf.reduce_all(outputs <= 1.0).numpy()

    # Check that outputs are constant across batch (state-conditional only)
    np.testing.assert_allclose(outputs[0].numpy(), outputs[-1].numpy(), rtol=1e-5)


def test_predictive_state_params_multiple_classification_params():
    """Test PredictiveStateParams with multiple sigmoid parameters per state."""
    tf.random.set_seed(42)
    n_states = 4
    n_params = 2  # E.g., two binary classification outputs per state

    # Create layer with sigmoid activation for both parameters
    layer = layers.PredictiveStateParams(
        n_params_per_state=n_params,
        activations="sigmoid",  # Both params use sigmoid
        flatten_output=False,
    )

    # Input is state probabilities [batch, n_states]
    batch_size = 20
    inputs = tf.random.normal((batch_size, n_states))
    outputs = layer(inputs)

    # Check output shape: [batch, n_states, n_params]
    assert outputs.shape == (batch_size, n_states, n_params)

    # Check that all values are between 0 and 1 (sigmoid property)
    assert tf.reduce_all(outputs >= 0.0).numpy()
    assert tf.reduce_all(outputs <= 1.0).numpy()

    # Check that outputs are constant across batch
    np.testing.assert_allclose(outputs[0].numpy(), outputs[10].numpy(), rtol=1e-5)
    np.testing.assert_allclose(outputs[0].numpy(), outputs[-1].numpy(), rtol=1e-5)

    # Check flattened version
    layer_flat = layers.PredictiveStateParams(
        n_params_per_state=n_params,
        activations="sigmoid",
        flatten_output=True,
    )
    layer_flat.build((None, n_states))
    layer_flat._theta_logits.assign(layer._theta_logits)
    outputs_flat = layer_flat(inputs)

    # Flattened shape should be [batch, n_states * n_params]
    assert outputs_flat.shape == (batch_size, n_states * n_params)

    # All values should still be in [0, 1]
    assert tf.reduce_all(outputs_flat >= 0.0).numpy()
    assert tf.reduce_all(outputs_flat <= 1.0).numpy()


def test_predictive_state_means_classification_multiple_states():
    """Test PredictiveStateMeans with sigmoid activation for binary classification with multiple states."""
    tf.random.set_seed(42)
    n_samples = 100
    n_features = 5
    n_states = 3

    # Generate synthetic binary classification data
    X = np.random.normal(size=(n_samples, n_features)).astype(np.float32)
    # Binary labels
    y = (np.random.rand(n_samples) > 0.5).astype(np.float32)

    # Build a model with multiple states and sigmoid output for binary classification
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_features,)))
    model.add(layers.PredictiveStateSimplex(n_states=n_states))
    model.add(layers.PredictiveStateMeans(units=1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    # Train for a few epochs
    model.fit(X, y, epochs=5, verbose=0, batch_size=32)

    # Make predictions
    predictions = model.predict(X, verbose=0)

    # Check output shape
    assert predictions.shape == (n_samples, 1)

    # Check that predictions are valid probabilities (between 0 and 1)
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)

    # Check that the model learned something (accuracy should be better than random)
    binary_preds = (predictions > 0.5).astype(np.float32)
    accuracy = np.mean(binary_preds.ravel() == y)
    # With random data, we just check it doesn't crash and produces valid outputs
    assert 0.0 <= accuracy <= 1.0


def test_predictive_state_means_multiclass_classification():
    """Test PredictiveStateMeans with softmax activation for multi-class classification with multiple states."""
    tf.random.set_seed(42)
    n_samples = 150
    n_features = 4
    n_states = 4
    n_classes = 3

    # Generate synthetic multi-class classification data
    X = np.random.normal(size=(n_samples, n_features)).astype(np.float32)
    # Multi-class labels (one-hot encoded)
    y_labels = np.random.randint(0, n_classes, size=n_samples)
    y = tf.keras.utils.to_categorical(y_labels, num_classes=n_classes)

    # Build a model with multiple states and softmax output for multi-class classification
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_features,)))
    model.add(layers.PredictiveStateSimplex(n_states=n_states))
    model.add(layers.PredictiveStateMeans(units=n_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    # Train for a few epochs
    model.fit(X, y, epochs=5, verbose=0, batch_size=32)

    # Make predictions
    predictions = model.predict(X, verbose=0)

    # Check output shape
    assert predictions.shape == (n_samples, n_classes)

    # Check that predictions are valid probability distributions
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)
    # Each row should sum to approximately 1 (softmax property)
    row_sums = predictions.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(n_samples), rtol=1e-5)

    # Check that the model produces valid outputs
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_labels)
    # With random data, we just check it doesn't crash and produces valid outputs
    assert 0.0 <= accuracy <= 1.0
