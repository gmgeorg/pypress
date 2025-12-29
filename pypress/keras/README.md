# pypress.keras

Keras/TensorFlow implementation of Predictive State Smoothing (PRESS) layers.

## Modules

### `layers.py`

Core PRESS layers for building models:

- **`PredictiveStateSimplex`**: Maps input features X to predictive state probabilities P(S|X) via softmax
- **`PredictiveStateMeans`**: Computes weighted mixture of state-conditional means: Σ P(S_j|X) · μ_j
- **`PredictiveStateParams`**: Learns state-conditional parameters (e.g., distribution params) independent of features
- **`PRESS`**: Convenience wrapper combining simplex and means layers

### `initializers.py`

Custom initializers for PRESS layers:

- **`PredictiveStateMeansInitializer`**: Initialize state-conditional means from observed data on original scale (automatically converts to logits using inverse activations)
- **`PredictiveStateParamsInitializer`**: Initialize state-conditional parameters with different activations per parameter (e.g., Gaussian [mean, std] with ['linear', 'softplus'])

### `activations.py`

Activation inverse functions for initialization:

- **`get_inverse_activation()`**: Returns inverse of activation functions for converting original scale values to logits
- **`ACTIVATION_INVERSES`**: Registry of supported activation inverses:
  - `linear` (identity)
  - `sigmoid` (logit)
  - `softplus` (inverse softplus)
  - `tanh` (arctanh)
  - `exponential` (log)
  - `leaky_relu` (conditional inverse)
  - `elu` (conditional inverse)
  - `softsign` (inverse softsign)
  - `softmax` (log approximation)
  - `selu` (conditional inverse)

### `regularizers.py`

Regularizers for controlling predictive state distributions:

- **`Uniform`**: Encourages uniform state distribution across samples
- **`DegreesOfFreedom`**: Controls effective number of states (penalizes non-uniform state usage)
- **`Combined`**: Combines multiple regularizers with different strengths

## Usage Examples

### Basic Regression with Mean Initialization

```python
import numpy as np
from pypress.keras.layers import PredictiveStateSimplex, PredictiveStateMeans

# Initialize means to empirical data mean
y_mean = np.mean(y_train, axis=0)

# Build PRESS model
simplex = PredictiveStateSimplex(n_states=5)
means = PredictiveStateMeans(
    units=1,
    activation="linear",
    init_values=y_mean  # Original scale initialization
)

# Use in Keras model
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    simplex,
    means
])
```

### Gaussian Distribution with PredictiveStateParams

```python
from pypress.keras.layers import PredictiveStateSimplex, PredictiveStateParams

# Initialize Gaussian parameters: mean=0, std=1
simplex = PredictiveStateSimplex(n_states=5)
params = PredictiveStateParams(
    n_params_per_state=2,
    activations=["linear", "softplus"],  # mean: linear, std: softplus
    init_values=[0.0, 1.0],  # [mean, std] on original scale
    flatten_output=False
)

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu'),
    simplex,
    params
])
```

## Architecture

PRESS decomposes p(y|X) via predictive states:

```text
p(y|X) = Σ_j p(y|s_j) · p(s_j|X)
```

where:

- **p(s_j|X)**: Predictive state probabilities (from `PredictiveStateSimplex`)
- **p(y|s_j)**: State-conditional distributions (parameterized by `PredictiveStateMeans` or `PredictiveStateParams`)
- **Conditional independence**: y ⊥ X | s (outputs independent of features given state)
