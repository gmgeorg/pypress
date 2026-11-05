# Layers

`pypress.keras.layers` has four layers. [concepts.md](concepts.md) explains the model they
implement.

| Layer | Input | Output | Learns |
| --- | --- | --- | --- |
| `PredictiveStateSimplex(n_states)` | features, `(n, d)` | predictive state weights $W$, `(n, K)` | a softmax map from features to states |
| `PredictiveStateMeans(units)` | $W$, `(n, K)` | the predictive mixture $W \mu$, `(n, units)` | one state-conditional mean per state |
| `PredictiveStateParams(n_params_per_state)` | $W$, `(n, K)` | every state's parameters, `(n, K * P)` or `(n, K, P)` | P state-conditional parameters per state |
| `PRESS(units, n_states)` | features | the predictive mixture | `Simplex` followed by `Means`, as one layer |

Layers that take $W$ infer the number of states $K$ from their input, so only the simplex (or
`PRESS`) needs `n_states`.

## `PRESS`, or the two layers

`PRESS` is the usual entry point. Arguments for the sub-layers go through
`predictive_state_simplex_kwargs` and `predictive_state_means_kwargs`:

```python
from pypress.keras import layers, regularizers

press = layers.PRESS(
    units=1,
    n_states=6,
    activation="sigmoid",                       # binary classification
    predictive_state_simplex_kwargs={"activity_regularizer": regularizers.Uniform(0.1)},
    predictive_state_means_kwargs={"init_values": 0.3},
)
model = tf.keras.Sequential([tf.keras.Input(shape=(n_features,)), press])  # builds `press`

press.predictive_states(X)          # W, the predictive state weights
press.state_conditional_means       # mu, shape (K, units)
press.predictive_state_simplex      # the sub-layers, e.g. for their weights
press.predictive_state_means
```

The same model as two layers, which is useful when something else needs $W$:

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(n_features,)),
    layers.PredictiveStateSimplex(n_states=6, activity_regularizer=regularizers.Uniform(0.1)),
    layers.PredictiveStateMeans(units=1, activation="sigmoid", init_values=0.3),
])
```

## Activations

`activation` on `PredictiveStateMeans` and `PRESS`, and `activations` on
`PredictiveStateParams`, take a name (`"linear"`, `"sigmoid"`, `"softplus"`, ...), a Keras
activation function, a Keras `Activation` layer, or a sequence with one per column (one per
unit, or one per parameter).

A single activation applies to the whole table of shape `(K, columns)`. For most activations that
is elementwise. For `"softmax"` it normalizes **across the columns of each state**:

* On `PredictiveStateMeans` with `units` equal to the number of classes, each state gets its own
  class distribution, and the predictive mixture is a class distribution too. Use it with
  `categorical_crossentropy`.
* On `PredictiveStateParams`, each state's P parameters sum to 1.

The softmax over *states* is the one in `PredictiveStateSimplex`, which makes each row of $W$ sum
to 1.

## Initialization

By default the state-conditional values start from Glorot-uniform logits. Pass `init_values` on
the **original scale**, after the activation, to start from known values. The layer inverts the
activation for you (for example a logit for `"sigmoid"`). Both `PredictiveStateMeans` and
`PredictiveStateParams` accept the same forms, with P the number of columns (`units` or
`n_params_per_state`):

| `init_values` | Meaning |
| --- | --- |
| a number | every entry, e.g. `0.3` |
| a 1D array of length P | one value per column, the same in every state, e.g. `np.mean(y_train, axis=0)` or `[0.0, 1.0]` |
| a 2D array of shape `(P, K)` | state-specific values, one row per column |

Activations with a known inverse: `linear`, `sigmoid`, `softplus`, `tanh`, `exponential`,
`leaky_relu`, `elu`, `softsign`, `selu`, and `softmax` (inverted up to a constant). Other
activations, such as a custom layer, can't be combined with `init_values`; the layer raises
`ValueError` at construction.

### Initialization from outputs

`initialize_from_y` fits a one-dimensional Gaussian mixture with $K$ components to the training
outputs, and returns state-specific `init_values`. That way each state starts at a different part
of the output distribution, instead of all states starting equal.

```python
from pypress.keras.initializers import initialize_from_y

means = layers.PredictiveStateMeans(units=1, init_values=initialize_from_y(y_train, n_states=6))

params = layers.PredictiveStateParams(
    n_params_per_state=2,
    activations=["linear", "softplus"],      # [mean, std]
    init_values=initialize_from_y(y_train, n_states=6, return_params=True),
)
```

For multi-output `y`, all outputs are pooled into one fit, so every output unit starts from the
same per-state means. That suits outputs on a common scale.

## State-conditional distribution parameters

`PredictiveStateMeans` predicts a mean. To model a whole predictive distribution, give each state
its own distribution parameters with `PredictiveStateParams`, and write the mixture likelihood in
the loss. For example, a Gaussian mixture:

```python
import numpy as np
import tensorflow as tf
from pypress.keras import layers

n_states = 5
inputs = tf.keras.Input(shape=(n_features,))
W = layers.PredictiveStateSimplex(n_states=n_states)(inputs)              # (n, K)
theta = layers.PredictiveStateParams(
    n_params_per_state=2, activations=["linear", "softplus"], flatten_output=False
)(W)                                                                     # (n, K, 2)
model = tf.keras.Model(inputs, [W, theta])


def mixture_nll(y, W, theta):
    """Negative log-likelihood of y under sum_k W_k * Normal(mu_k, sigma_k)."""
    mu, sigma = theta[..., 0], theta[..., 1]
    log_normal = -0.5 * ((y - mu) / sigma) ** 2 - tf.math.log(sigma) - 0.5 * np.log(2 * np.pi)
    return -tf.reduce_mean(tf.reduce_logsumexp(tf.math.log(W + 1e-12) + log_normal, axis=1))
```

Train it with a custom training step, or by concatenating `W` and `theta` into one output and
splitting them in the loss. With `flatten_output=True` (the default) the output is
`(n, K * P)`, ordered parameter-first: all states' first parameter, then all states' second.

## Saving and loading

All layers, regularizers, schedules and initializers are registered Keras-serializable objects,
so `model.save("model.keras")` and `tf.keras.models.load_model("model.keras")` work without
`custom_objects`. Saved models include:

* the activations and `init_values`, in JSON-safe form;
* every regularizer setting, including `ScheduledValue`s;
* the `ema_decay` running averages, as variables of the layer they regularize.

Models saved with `pypress` 0.3 still load. Configs in older formats are read, and
models saved before the running averages were stored load with fresh averages. Deprecations are
listed in the [CHANGELOG](../CHANGELOG.md).
