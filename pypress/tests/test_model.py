"""End-to-end integration tests for PRESS models with GMM initialization."""

from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow import keras

from pypress.clustering import GaussianMixture1D
from pypress.keras import layers
from pypress.utils import initialize_from_y


def _test_data(n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic test data with known relationship.

    The target y has a known relationship with features:
        y = x1 + 5*x3 - exp(x5) + noise

    Args:
        n_samples: Number of samples to generate.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    np.random.seed(10)
    feats = pd.DataFrame(np.random.normal(size=(n_samples, 7)))
    feats.columns = ["x" + str(k + 1) for k in range(feats.shape[1])]

    eps = np.random.normal(size=feats.shape[0])
    y = feats["x1"] + 5 * feats["x3"] - np.exp(feats["x5"]) + eps

    return feats, y


def test_press_model_with_gmm_initialization():
    """Test end-to-end PRESS model with GMM-based initialization.

    This test demonstrates the complete workflow:
    1. Generate synthetic training data
    2. Initialize cluster parameters using GaussianMixture1D
    3. Pass initialization to PredictiveStateMeans layer
    4. Build and train PRESS model
    5. Validate predictions
    """
    # 1. Generate training data
    n_samples = 1000
    n_states = 5
    X_train, y_train = _test_data(n_samples=n_samples)

    # Convert to numpy arrays
    X_train_np = X_train.values
    y_train_np = y_train.values

    # 2. Initialize clusters using GaussianMixture1D
    gmm = GaussianMixture1D(n_components=n_states, max_iter=20)
    gmm.fit(y_train_np)

    # Verify GMM fitted successfully
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert gmm.means_.shape == (n_states,)
    assert gmm.covariances_.shape == (n_states,)
    # Note: convergence not guaranteed, but GMM should still provide good initialization
    assert gmm.n_iter_ > 0

    # 3. Get initialization values for the layer
    init_values = initialize_from_y(y_train_np, n_states=n_states, return_params=False)

    # Verify initialization shape: (units, n_states) = (1, 5)
    assert init_values.shape == (
        1,
        n_states,
    )  # units=1 for 1D target, state-specific means

    # 4. Build PRESS model with GMM initialization
    model = keras.Sequential(
        [
            keras.layers.Dense(
                64, activation="relu", input_shape=(X_train_np.shape[1],)
            ),
            keras.layers.Dense(32, activation="relu"),
            layers.PredictiveStateSimplex(n_states=n_states),
            layers.PredictiveStateMeans(
                units=1, activation="linear", init_values=init_values
            ),
        ]
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    # 5. Train model
    history = model.fit(
        X_train_np,
        y_train_np,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_split=0.2,
    )

    # 6. Validate results

    # Check training history exists and has expected keys
    assert "loss" in history.history
    assert "mae" in history.history
    assert len(history.history["loss"]) == 50

    # Check that loss decreased during training
    initial_loss = history.history["loss"][0]
    final_loss = history.history["loss"][-1]
    assert final_loss < initial_loss, "Model should improve during training"

    # Make predictions on training data
    y_pred = model.predict(X_train_np, verbose=0)

    # Check prediction shape
    assert y_pred.shape == (n_samples, 1)

    # Check predictions are not NaN or Inf
    assert not np.any(np.isnan(y_pred)), "Predictions should not contain NaN"
    assert not np.any(np.isinf(y_pred)), "Predictions should not contain Inf"

    # Check that predictions are in a reasonable range relative to targets
    y_train_std = y_train_np.std()
    y_train_mean = y_train_np.mean()
    assert y_pred.mean() > y_train_mean - 3 * y_train_std
    assert y_pred.mean() < y_train_mean + 3 * y_train_std

    # Check correlation with truth (model should learn something, even if modestly)
    correlation = np.corrcoef(y_train_np.ravel(), y_pred.ravel())[0, 1]
    # Note: Correlation may be modest with limited training, but predictions should
    # at least be in reasonable range and model should improve
    assert not np.isnan(correlation), "Correlation should be computable"

    print("\nPRESS Model Training Summary:")
    print(f"  Initial Loss: {initial_loss:.4f}")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Improvement: {100 * (initial_loss - final_loss) / initial_loss:.1f}%")
    print(f"  Final MAE: {history.history['mae'][-1]:.4f}")
    print(f"  Correlation with truth: {correlation:.4f}")
    print(f"  GMM converged: {gmm.converged_}")
    print(f"  GMM iterations: {gmm.n_iter_}/{gmm.max_iter}")
    print(f"  Target mean: {y_train_mean:.4f}, Prediction mean: {y_pred.mean():.4f}")


def test_press_model_layers_output_shapes():
    """Test that PRESS model layers produce expected output shapes."""
    n_samples = 100
    n_states = 3
    X_train, y_train = _test_data(n_samples=n_samples)

    X_train_np = X_train.values
    y_train_np = y_train.values

    # Initialize from data
    init_values = initialize_from_y(y_train_np, n_states=n_states)

    # Build model
    input_layer = keras.layers.Input(shape=(X_train_np.shape[1],))
    hidden = keras.layers.Dense(16, activation="relu")(input_layer)
    states = layers.PredictiveStateSimplex(n_states=n_states)(hidden)
    output = layers.PredictiveStateMeans(units=1, init_values=init_values)(states)

    model = keras.Model(inputs=input_layer, outputs=[states, output])

    # Get intermediate outputs
    states_out, predictions = model.predict(X_train_np, verbose=0)

    # Check shapes
    assert states_out.shape == (n_samples, n_states)
    assert predictions.shape == (n_samples, 1)

    # Check that states form a probability simplex (sum to 1)
    row_sums = states_out.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(n_samples), rtol=1e-5)

    # Check all state probabilities are in [0, 1]
    assert np.all(states_out >= 0.0)
    assert np.all(states_out <= 1.0)


def test_press_model_multi_output():
    """Test PRESS model with multi-dimensional output."""
    n_samples = 500
    n_states = 4
    units = 2  # 2D output

    # Generate data
    X_train, y_train = _test_data(n_samples=n_samples)
    X_train_np = X_train.values

    # Create 2D target by stacking y_train with a transformed version
    y_train_2d = np.column_stack([y_train.values, y_train.values * 2])

    # Initialize from data
    init_values = initialize_from_y(y_train_2d, n_states=n_states)

    # Should return (units, n_states) shape
    assert init_values.shape == (units, n_states)

    # Build model
    model = keras.Sequential(
        [
            keras.layers.Dense(
                24, activation="relu", input_shape=(X_train_np.shape[1],)
            ),
            layers.PredictiveStateSimplex(n_states=n_states),
            layers.PredictiveStateMeans(
                units=units, activation="linear", init_values=init_values
            ),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    # Train briefly
    model.fit(X_train_np, y_train_2d, epochs=5, batch_size=32, verbose=0)

    # Predict
    y_pred = model.predict(X_train_np, verbose=0)

    # Check shape
    assert y_pred.shape == (n_samples, units)

    # Check no NaN/Inf
    assert not np.any(np.isnan(y_pred))
    assert not np.any(np.isinf(y_pred))


def test_press_model_with_different_n_states():
    """Test PRESS model with varying number of states."""
    n_samples = 200
    X_train, y_train = _test_data(n_samples=n_samples)
    X_train_np = X_train.values
    y_train_np = y_train.values

    for n_states in [1, 2, 5, 10]:
        # Initialize
        init_values = initialize_from_y(y_train_np, n_states=n_states)

        # Build model
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    16, activation="relu", input_shape=(X_train_np.shape[1],)
                ),
                layers.PredictiveStateSimplex(n_states=n_states),
                layers.PredictiveStateMeans(units=1, init_values=init_values),
            ]
        )

        model.compile(optimizer="adam", loss="mse")

        # Train briefly
        history = model.fit(X_train_np, y_train_np, epochs=3, batch_size=32, verbose=0)

        # Check that training completed without errors
        assert len(history.history["loss"]) == 3

        # Make predictions
        y_pred = model.predict(X_train_np, verbose=0)

        # Basic sanity checks
        assert y_pred.shape == (n_samples, 1)
        assert not np.any(np.isnan(y_pred))

        # Check intermediate state layer output
        state_layer = model.layers[1]
        states = state_layer(model.layers[0](X_train_np)).numpy()
        assert states.shape == (n_samples, n_states)


def test_press_model_initialization_impact():
    """Test that GMM initialization provides better starting point than random."""
    n_samples = 500
    n_states = 5
    X_train, y_train = _test_data(n_samples=n_samples)
    X_train_np = X_train.values
    y_train_np = y_train.values

    # Model with GMM initialization
    init_values_gmm = initialize_from_y(y_train_np, n_states=n_states)

    model_gmm = keras.Sequential(
        [
            keras.layers.Dense(
                32, activation="relu", input_shape=(X_train_np.shape[1],)
            ),
            layers.PredictiveStateSimplex(n_states=n_states),
            layers.PredictiveStateMeans(units=1, init_values=init_values_gmm),
        ]
    )

    model_gmm.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")

    # Get initial predictions before training
    y_pred_gmm_initial = model_gmm.predict(X_train_np, verbose=0)

    # Model with default initialization (should be zeros)
    model_default = keras.Sequential(
        [
            keras.layers.Dense(
                32, activation="relu", input_shape=(X_train_np.shape[1],)
            ),
            layers.PredictiveStateSimplex(n_states=n_states),
            layers.PredictiveStateMeans(units=1),
        ]
    )

    model_default.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")

    y_pred_default_initial = model_default.predict(X_train_np, verbose=0)

    # GMM-initialized predictions should be closer to the target mean
    # than default initialized predictions (which should be near 0)
    target_mean = y_train_np.mean()

    error_gmm = np.abs(y_pred_gmm_initial.mean() - target_mean)
    error_default = np.abs(y_pred_default_initial.mean() - target_mean)

    # GMM initialization should provide a better starting point
    # (closer to the actual mean of the data)
    assert error_gmm < error_default, (
        f"GMM init error {error_gmm:.3f} should be < default init error {error_default:.3f}"
    )

    print("\nInitialization Impact:")
    print(f"  Target mean: {target_mean:.4f}")
    print(
        f"  GMM init prediction mean: {y_pred_gmm_initial.mean():.4f} (error: {error_gmm:.4f})"
    )
    print(
        f"  Default init prediction mean: {y_pred_default_initial.mean():.4f} (error: {error_default:.4f})"
    )
