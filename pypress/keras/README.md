# pypress.keras

Keras/TensorFlow implementation of Predictive State Smoothing (PRESS) layers.

## Modules

### `layers.py`

Core PRESS layers for building models:

- **`PredictiveStateSimplex`**: Maps input features X to predictive state probabilities P(S|X) via softmax
- **`PredictiveStateMeans`**: Computes weighted mixture of state-conditional means: Σ P(S_j|X) · μ_j
- **`PredictiveStateParams`**: Learns state-conditional parameters (e.g., distribution params)
  independent of features
- **`PRESS`**: Convenience wrapper combining simplex and means layers

### `initializers.py`

Custom initializers for PRESS layers:

- **`PredictiveStateMeansInitializer`**: Initialize state-conditional means from observed data on
  original scale (automatically converts to logits using inverse activations)
- **`PredictiveStateParamsInitializer`**: Initialize state-conditional parameters with different
  activations per parameter (e.g., Gaussian [mean, std] with ['linear', 'softplus'])

### `activations.py`

Activation inverse functions for initialization:

- **`get_inverse_activation()`**: Returns inverse of activation functions for converting original
  scale values to logits
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

- **`TargetEntropy`**: Penalizes deviation of the *mean row entropy* from a target, i.e. how
  sharp each sample's state assignment is
- **`Uniform`**: Special case of `TargetEntropy` targeting uniform weights across states within
  each row
- **`StateSizeEntropy`**: Penalizes states carrying ~0 weight across the whole population, via
  the normalized KL divergence of the state-size marginal from uniform usage. Catches
  over-provisioned `K` (many dead states), which `Uniform` and `DegreesOfFreedom` are both blind to
- **`MinStateSize`**: Enforces a per-state floor on population share ("at least 5% of
  observations per state"). Exact where `StateSizeEntropy` is only a proxy; use it when state
  sizes are legitimately unequal but none may vanish
- **`DegreesOfFreedom`**: Penalizes deviation of the implied kernel trace (the *effective* number
  of states) from a target
- **`Combined`**: Combines multiple regularizers with different strengths

### `schedules.py`

- **`ScheduledValue`**: A regularizer scalar (`l2`, or `DegreesOfFreedom`'s `target`) that
  varies with the training epoch. Backed by a `tf.Variable`, so updates survive `tf.function`
  tracing -- mutating a plain Python attribute mid-training is silently ignored

### `callbacks.py`

- **`RegularizerScheduler`**: Advances every `ScheduledValue` in a model's regularizers at the
  start of each epoch. Without it attached to `fit`, scheduled values stay at `start`

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
