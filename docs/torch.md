# PyTorch backend

`pypress.torch` is a native PyTorch port of the PRESS layers and regularizers: `nn.Module`
layers, regularizers with a plain `penalty(x)`, and no TensorFlow. It computes the same
quantities as `pypress.keras`; the parity tests in `pypress/tests/parity/` check values,
gradients and state against Keras.

```bash
pip install "pypress[torch] @ git+https://github.com/gmgeorg/pypress.git"
```

## Training a PRESS model

pypress ships no `fit` loop for torch. A standalone model is a short loop:

```python
import numpy as np
import torch

from pypress import analysis
from pypress.torch import layers, regularization_loss, regularizers, schedules

rng = np.random.default_rng(0)
X = torch.tensor(rng.standard_normal((2000, 10)), dtype=torch.float32)
y = X[:, :1] + 5 * X[:, 2:3] - torch.exp(X[:, 4:5]) + torch.randn(2000, 1)
y = (y - y.mean()) / y.std()  # standardize, so the penalty weights are on the loss's scale

epochs = 40
reg = regularizers.CombinedRegularizer([
    # Guard: every state keeps at least 5% of the observations.
    regularizers.MinStateSize(l2=5.0, min_share=0.05, ema_decay=0.95),
    # Complexity: let states differentiate first, then prune toward 4 effective states.
    regularizers.DegreesOfFreedom(
        l2=schedules.ScheduledValue(start=0.0, end=2.0, duration_epochs=epochs // 2),
        target=4.0,
        ema_decay=0.95,
    ),
])
press = layers.PRESS(
    n_states=9, units=1, in_features=10,
    predictive_state_simplex_kwargs={"activity_regularizer": reg},
)
optimizer = torch.optim.NAdam(press.parameters(), lr=0.01)

press.train()
with schedules.epoch_clock() as clock:  # restores the previous epoch when the block ends
    for epoch in range(epochs):
        clock(epoch)  # advances every ScheduledValue
        for idx in torch.randperm(len(X)).split(256):
            optimizer.zero_grad()
            mse = torch.nn.functional.mse_loss(press(X[idx]), y[idx])
            loss = mse + regularization_loss(press)  # add the stored penalties
            loss.backward()
            optimizer.step()

press.eval()
with torch.no_grad():
    W = press.predictive_states(X).numpy()  # (n, K) predictive state weights
print(analysis.state_size(W, normalize=True))
print(analysis.degrees_of_freedom(W))
```

## Where the penalty is added

PyTorch has no activity regularizers, so there are two patterns. Both give the same value:
`penalty(x)` is the amount added to the per-example mean loss, at any batch size.

* **On a layer.** `PredictiveStateSimplex(..., activity_regularizer=reg)` (or the
  `predictive_state_simplex_kwargs` of `PRESS`). In training mode the forward stores the
  penalty; `pypress.torch.regularization_loss(model)` sums the stored penalties of every layer,
  and you add the result to your loss. In eval mode nothing is stored, so it is zero. The
  regularizer is a submodule, so its running averages are in the model's `state_dict()`.
* **In a loss.** A loss module that owns the regularizer calls `reg.penalty(weights)` itself
  (this is how a likelihood loss on `PredictiveStateParams` outputs uses them). The running
  averages then live in the loss module's `state_dict()`. Create the regularizer with the loss
  and keep the loss in the checkpoint.

A regularizer with `ema_decay` keeps per-layer state, so one instance serves one layer (or one
loss). Its running averages advance only in training mode, which is the default on
construction. `model.train()` and `model.eval()` reach a regularizer held as a submodule; call
`reg.eval()` yourself for one a loss holds.

## Scheduling

`ScheduledValue` is a plain Python float read from one shared epoch clock when the penalty is
computed; there is no callback. Call `schedules.set_epoch(epoch)` at the start of each epoch, or
use `with schedules.epoch_clock() as clock: clock(epoch)`, which restores the previous epoch
afterwards, as `RegularizerScheduler` does after `fit`. While no clock is running every value
reads its `end` (the fully enforced penalty). Under `torch.compile`, a changing Python value is
guarded on, so a scheduled hyperparameter triggers one recompile per epoch.

`pypress.keras.schedules.set_epoch` also moves this clock, so one call drives both backends.

## Differences from Keras

| | Keras | PyTorch |
| --- | --- | --- |
| Constructor sizes | `PredictiveStateMeans(units)`; `n_states` inferred at build | `PredictiveStateMeans(n_states, units)`: `n_states` first, always explicit |
| Input size | inferred at build | `PredictiveStateSimplex(n_states, in_features=None)`: `None` is lazy (`nn.LazyLinear`) |
| Activity regularizer | added to the loss by Keras | `regularization_loss(model)`, or call `reg.penalty(x)` |
| Regularizer state | Keras layer variables | buffers: in `state_dict()`, moved by `.to(device)` |
| `ScheduledValue.value()` | float32 tensor, in-graph | Python float |
| Training loop | `fit` + `RegularizerScheduler` | your loop + `set_epoch` / `epoch_clock` |
| Activations | Keras names, functions, layers | the same names (`linear`, `sigmoid`, `softplus`, `tanh`, `exponential`, `leaky_relu` with slope 0.2, `elu`, `softsign`, `softmax`, `selu`, `relu`, `swish`/`silu`, `gelu`), torch functions or `nn.Module`s |

Not ported (deprecated in Keras): `DegreesOfFreedom(df=...)`, the tuple form of
`CombinedRegularizer`, `UniformAndDegreesOfFreedomRegularizer`, `collect_scheduled_values`, the
`ScheduledValue.variable` / `assign_epoch` / `was_scheduled` properties and the `pypress.utils`
shims.

Defaults match Keras: a table without `init_values` starts Glorot-uniform, the simplex starts
with a zero kernel and bias (uniform weights, `trace(H) = 1`). A callable or module activation
without a known name cannot be combined with `init_values`, as in Keras.

## Moving between backends

* **Configs.** A regularizer's `get_config()` is identical in both backends, so a config from
  one builds the other: `pypress.torch.regularizers.from_config(keras_reg.get_config() |
  {"class_name": "Uniform"})`. `regularizers.serialize(reg)` writes the form
  `tf.keras.regularizers.serialize` writes, and `regularizers.get` accepts either.
* **Weights.** `PredictiveStateSimplex.from_keras(layer)`, `PredictiveStateMeans.from_keras`,
  `PredictiveStateParams.from_keras` and `PRESS.from_keras` copy a built Keras layer, weights
  included (the Dense kernel is transposed). An activity regularizer is converted through its
  config; its running averages are not copied.

## Saving

Use `state_dict()` and `load_state_dict()`. The simplex weights, the table of
state-conditional values and the regularizer averages and step counts are all in it. A lazy
`PredictiveStateSimplex(n_states)` and a regularizer that has not seen `K` yet both load a
saved state dict directly.
