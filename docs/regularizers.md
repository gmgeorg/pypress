# Regularizers

PRESS regularizers attach as the `activity_regularizer` of `PredictiveStateSimplex` (or through
`PRESS(predictive_state_simplex_kwargs={"activity_regularizer": ...})`). Each one sees a batch of
predictive state weights $W$ (rows sum to 1) and adds a **penalty** on a statistic of it. Each
controls a different property, and knowing what each one sees at initialization is what makes
them tunable.

| Regularizer | Controls | Value at init (uniform weights) |
| --- | --- | --- |
| `TargetEntropy` / `Uniform` | How sharp each *row* is: one observation's state assignment | mean row entropy $= \log K$, the maximum |
| `StateSizeEntropy` | Whether state shares are spread across all states | penalty $= 0$ |
| `MinStateSize` | A per-state floor: "at least x% of observations per state" | penalty $= 0$ |
| `DegreesOfFreedom` | The effective number of states, $\operatorname{trace}(H)$ | $\operatorname{trace}(H) = 1$, **not** $K$ |

Three consequences are worth internalizing:

* **$\operatorname{trace}(H)$ starts at 1 and grows.** At initialization every row of $W$ is
  identical, so the smoother matrix has rank one however many states you asked for. A
  `DegreesOfFreedom` target below `n_states` therefore *caps differentiation from the first
  step* rather than pruning states later. If you want a small model, reduce `n_states`; use
  `DegreesOfFreedom` to prune once the states mean something.
* **`StateSizeEntropy` and `MinStateSize` are free at initialization.** They start satisfied and
  only bite once a state loses its share, so they act as guards and need no warm-up.
* **`Uniform` and `DegreesOfFreedom` cannot see a dead state.** A $K = 50$ model with 45 states
  at a near-zero share has the same degrees of freedom and mean row entropy as an honest $K = 5$
  model. That is what `StateSizeEntropy` and `MinStateSize` are for.

## Recommended setup

```python
import tensorflow as tf
from pypress.keras import callbacks, layers, regularizers, schedules

n_states, epochs = 9, 40

reg = regularizers.CombinedRegularizer([
    # Guard: no state may hold less than 5% of the observations.
    # Zero at init, so full strength from epoch 0: it only bites if a state shrinks.
    regularizers.MinStateSize(l2=5.0, min_share=0.05, ema_decay=0.95),

    # Complexity: let states differentiate first, then prune toward 4 effective states.
    # Warmed up from 0, because trace(H) starts at 1 and must be allowed to grow.
    regularizers.DegreesOfFreedom(
        l2=schedules.ScheduledValue(start=0.0, end=2.0, duration_epochs=epochs // 2),
        target=4.0,
        ema_decay=0.95,
    ),
])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(n_features,)),
    layers.PRESS(units=1, n_states=n_states,
                 predictive_state_simplex_kwargs={"activity_regularizer": reg}),
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01))
model.fit(X, y, epochs=epochs, batch_size=256,
          callbacks=[callbacks.RegularizerScheduler()])   # required for the schedule
```

## Settings

Every regularizer takes `l2` (the penalty weight, $\geq 0$) and `ema_decay` (see
[below](#batch-size-and-ema_decay)). The others:

| Regularizer | Setting | Range | Default |
| --- | --- | --- | --- |
| `TargetEntropy` | `target` (nats) or `entropy_fraction` (target $= \log(\text{fraction} \cdot K)$), not both | $\geq 0$ / $(0, 1]$ | fraction 0.5 |
| `Uniform` | none: `TargetEntropy` with fraction 1, i.e. target $\log K$ | | |
| `StateSizeEntropy` | `active_fraction` | $(0, 1]$ | 1.0 |
| `MinStateSize` | `min_share` | $(0, 1]$, with `min_share * K <= 1` | $1 / (2K)$ |
| `DegreesOfFreedom` | `target`, `normalize` | $\geq 1$ / bool | 1.0 / `False` |

Invalid values raise `ValueError` at construction, including for every value a schedule can
reach.

`CombinedRegularizer([...])` sums several regularizers on one layer, and serializes each of them
with the model.

**One smoothed regularizer per layer.** A regularizer with `ema_decay` keeps running averages for
the layer it is attached to. Give each layer its own instance; attaching one to two layers raises
`ValueError`. Regularizers without `ema_decay` can be shared.

## Choosing `l2`

The entropy and state-share penalties are bounded in $[0, \text{l2}]$ and do not depend on $K$,
so their `l2` is the worst-case cost of total violation. `DegreesOfFreedom(target=...)` keeps
its target as an absolute number of states. Set `normalize=True` to divide its error by
`n_states`, so an `l2` tuned for a fractional error transfers across state counts.

Pick each weight as a meaningful fraction of the loss you actually see. With a standardized
target (MSE around 1 at init), an `l2` of roughly 1 to 5 makes a penalty bind, while `l2 = 0.05`
leaves it decorative.

## Scheduling

Any setting above except `ema_decay` and `normalize` accepts a `ScheduledValue` in place of a
number, with `linear`, `cosine` or `exponential` interpolation:

```python
schedules.ScheduledValue(start=0.0, end=2.0, duration_epochs=20, start_epoch=0,
                         schedule="linear")
```

The direction is set by the endpoints (`start < end` warms up, `start > end` decays), and the
right direction differs per regularizer, so do **not** ramp them all together.

| Regularizer | Schedule | Why |
| --- | --- | --- |
| `DegreesOfFreedom` | `l2`: `0 -> strength` | $\operatorname{trace}(H)$ starts at 1; let it grow on the fit signal, then prune |
| `DegreesOfFreedom` | `target`: `n_states -> desired` (alternative) | More aggressive: forces differentiation first, then prunes gradually |
| `TargetEntropy` | `entropy_fraction`: `1.0 -> desired` | Row entropy starts at $\log K$, i.e. fraction 1.0, so the constraint starts where the model already is |
| `MinStateSize` | none (constant `l2`) | Already satisfied at init; it is the counterweight during the fragile early epochs |
| `StateSizeEntropy` | none (constant `l2`) | Same |

Why schedule at all: a dead state receives almost no gradient (measured at around $10^{-12}$,
against $10^{-2}$ for a live state in a 6-state model), so losing a state is close to
irreversible, while having too many states early is cheap. A penalty that pushes toward fewer
states should therefore approach from the state-rich side. A schedule does not move a penalty's
optimum; it changes which local optimum training reaches.

All scheduled values read one training clock, and `RegularizerScheduler` advances it each epoch.
It must be passed to `fit` for a schedule to move. Without it, every `ScheduledValue` reads its
`end` value: you get the fully enforced penalty without the annealing, never a silently disabled
regularizer. The clock is reset when `fit` ends.

## Batch size and `ema_decay`

The state shares and $\operatorname{trace}(H)$ are estimated from one minibatch, and every
penalty is convex in them, so each batch estimate is biased *upward*. Running more batches per
epoch does not remove that bias; only a larger batch, or `ema_decay`, does.

For example, with a population that exactly meets a 5% floor (true penalty 0), the mean
`MinStateSize` penalty at `batch_size=32` is 0.052 whether averaged over 200 or 5000 batches, and
a compliant state is flagged in about 82% of batches. The effective floor is then well above the
one you asked for.

Either keep `batch_size * min_share` at roughly 10 or more (so `batch_size >= 200` for a 5%
floor), or set `ema_decay` (0.9 to 0.99). That evaluates the penalty on bias-corrected moving
averages of the statistics across batches, while gradients still flow through the current batch.
The averages only advance on training batches, and are saved with the model.
