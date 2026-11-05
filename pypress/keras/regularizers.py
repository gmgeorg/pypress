"""Regularizers (penalties) on predictive state weights.

Every regularizer here is the same pipeline:

1. **statistic**: reduce a batch of predictive state weights to named
   population statistics (`pypress.keras.statistics`);
2. **cross-batch estimator** (optional, `ema_decay`): replace each statistic's
   value by a bias-corrected moving average across training batches;
3. **penalty function**: a pure function of the statistics and K, scaled by
   `l2`.

`penalty(x)` returns that value: the amount added to the training loss. Keras
divides an activity regularizer's loss by the batch size, so `__call__` returns
`penalty(x)` times the batch size, and the strength of `l2` does not depend on
the batch size.

Scalar settings are declared once as `Hyperparameter`s, which own their
validation, in-graph reading and serialization. See CONTEXT.md for the domain
terms.
"""

import math
import warnings
from typing import Dict, Sequence

import tensorflow as tf

from . import hyperparameters as hp
from . import statistics

_Hyperparameter = hp.Hyperparameter


def _in_training() -> bool:
    """Whether the current Keras layer call is a training call.

    Activity regularizers are invoked by Keras inside the layer's call context,
    which records the `training` flag. `fit` sets it to True. A call that leaves
    it unset, such as `layer(x)` or `PRESS.predictive_states(x)`, counts as
    inference, as it does for Keras's own layers (e.g. `Dropout`). Outside any
    layer call (a regularizer called directly) there is no context, and this
    counts as training.
    """
    try:
        from keras.src.backend.common import global_state
    except ImportError:  # pragma: no cover - Keras internals moved
        return True
    context = global_state.get_global_attribute("current_call_ctx")
    if context is None:
        return True
    return bool(getattr(context, "training", None))


class CrossBatchEstimator(tf.keras.layers.Layer):
    """Bias-corrected moving average of named statistics across training batches.

    The value returned is the smoothed estimate; the gradient is passed straight
    through to the current batch, so smoothing changes *what* the penalty is
    evaluated on without detaching it from the weights being trained. Same
    arrangement as EMA codebooks in VQ-VAE.

    The average only advances on training calls: `evaluate` and `predict` read
    the current estimate without updating it. It is a Keras layer so that, once
    bound to the layer it regularizes, its state is part of that layer's
    variables and is saved and checkpointed with the model.
    """

    def __init__(self, decay: float, statistic_names: Sequence[str], **kwargs):
        """Initializes the estimator.

        Args:
          decay: EMA decay in [0, 1). Effective sample size is
            ~ batch_size / (1 - decay).
          statistic_names: Names of the statistics to smooth.
          **kwargs: Keras layer arguments.
        """
        super().__init__(trainable=False, **kwargs)
        self.decay = float(decay)
        self.statistic_names = tuple(statistic_names)
        self.n_states: int | None = None
        self.averages: Dict[str, tf.Variable] = {}
        self.step = None

    def create_state(self, n_states: int):
        """Creates the averages for 'n_states' predictive states (once)."""
        if self.n_states is not None:
            if n_states != self.n_states:
                raise ValueError(
                    f"This regularizer's smoothed statistics were created for "
                    f"K = {self.n_states} predictive states, but it is now applied "
                    f"to K = {n_states}. A regularizer with 'ema_decay' keeps "
                    "per-layer state: use a separate instance for each layer."
                )
            return
        with tf.init_scope():
            for name in self.statistic_names:
                self.averages[name] = self.add_weight(
                    name=f"{name}_ema",
                    shape=statistics.shape(name, n_states),
                    initializer="zeros",
                    trainable=False,
                )
            self.step = self.add_weight(
                name="ema_step", shape=(), initializer="zeros", trainable=False
            )
        self.n_states = n_states
        self.built = True

    def estimates(self) -> Dict[str, tf.Tensor]:
        """Current bias-corrected estimates, keyed by statistic name."""
        decay = tf.constant(self.decay)
        step = tf.convert_to_tensor(self.step)
        correction = tf.maximum(1.0 - tf.pow(decay, step), 1e-8)
        return {
            name: tf.convert_to_tensor(average) / correction
            for name, average in self.averages.items()
        }

    @property
    def steps(self) -> int:
        """Number of training batches averaged so far."""
        return 0 if self.step is None else int(self.step.numpy())

    def update(
        self, values: Dict[str, tf.Tensor], n_states: int
    ) -> Dict[str, tf.Tensor]:
        """Advances the averages with 'values' (on training calls only).

        Returns the smoothed statistics, with gradients flowing to 'values'.
        """
        self.create_state(n_states)
        if _in_training():
            decay = tf.constant(self.decay)
            for name, value in values.items():
                average = self.averages[name]
                average.assign(
                    decay * tf.convert_to_tensor(average)
                    + (1.0 - decay) * tf.stop_gradient(tf.cast(value, average.dtype))
                )
            self.step.assign_add(1.0)

        has_history = tf.convert_to_tensor(self.step) > 0.0
        smoothed = {}
        for name, estimate in self.estimates().items():
            value = values[name]
            # Before any training batch there is no history: use the batch value.
            estimate = tf.where(has_history, tf.cast(estimate, value.dtype), value)
            smoothed[name] = tf.stop_gradient(estimate - value) + value
        return smoothed

    def load_own_variables(self, store):
        """Loads the averages; models saved before they existed load with none."""
        if not store or len(store.keys()) == 0:
            return
        super().load_own_variables(store)


def _batch_size(x, dtype) -> tf.Tensor:
    """The number of rows of 'x', as a scalar of 'dtype'."""
    return tf.cast(tf.shape(x)[0], dtype)


def _undo_keras_batch_mean(penalty: tf.Tensor, x) -> tf.Tensor:
    """Scales 'penalty' by the batch size, cancelling Keras's division by it.

    `Layer.__call__` divides an activity regularizer's loss by the batch size of
    the output (`keras/src/layers/layer.py`), which suits penalties that sum over
    the batch. PRESS penalties are already batch statistics.
    """
    return penalty * _batch_size(x, penalty.dtype)


class _PenaltyRegularizer(tf.keras.regularizers.Regularizer):
    """Base for PRESS regularizers: statistic -> estimator -> penalty.

    Subclasses declare `HYPERPARAMETERS` (the scalar settings) and `STATISTICS`
    (the names of the population statistics their penalty consumes), and
    implement `_penalty(stats, n_states)`.

    All statistics are estimated from a single minibatch, because these attach
    as `activity_regularizer` on `PredictiveStateSimplex`, and every penalty here
    is a convex function of them. Jensen's inequality then makes each batch
    estimate biased *upward*: `E[f(stat_batch)] >= f(E[stat_batch])`. Running
    more batches per epoch shrinks the variance of the epoch average but not
    this bias -- only a larger batch, or averaging the statistic across batches
    (`ema_decay`), does.

    Measured with a population that is exactly compliant (K = 9, one state at
    exactly 5%, `MinStateSize(min_share=0.05)`, so the true penalty is 0), the
    mean penalty is 0.0515 at `batch_size=32` and identical whether averaged over
    200 or 5000 batches; it falls to 0.0009 at `batch_size=1024`.
    """

    HYPERPARAMETERS: tuple = (hp.L2, hp.EMA_DECAY)
    STATISTICS: tuple = ()

    def __init__(self, **kwargs):
        """Coerces the declared hyperparameters found in 'kwargs'.

        Args:
          **kwargs: Values for the declared hyperparameters; anything else is
            passed to the Keras regularizer base class.
        """
        declared = {param.name: param for param in self.HYPERPARAMETERS}
        values = {
            name: kwargs.pop(name, param.default) for name, param in declared.items()
        }
        super().__init__(**kwargs)
        self._hyperparameters = {
            name: declared[name].coerce(value) for name, value in values.items()
        }
        decay = self._hyperparameters.get("ema_decay")
        self._estimator = (
            None
            if decay is None or not self.STATISTICS
            else CrossBatchEstimator(decay, self.STATISTICS)
        )
        self._bound_layer = None

    # -- hyperparameters ---------------------------------------------------

    def _value(self, name: str):
        """A hyperparameter as configured (float, None or ScheduledValue)."""
        return self._hyperparameters.get(name)

    def _tensor(self, name: str) -> tf.Tensor:
        """A hyperparameter read in-graph as a float32 tensor."""
        return _Hyperparameter.tensor(self._hyperparameters[name])

    @property
    def l2(self):
        """The penalty multiplier, as configured (float or ScheduledValue)."""
        return self._value("l2")

    @property
    def ema_decay(self) -> float | None:
        """The EMA decay, or None when statistics are used un-smoothed."""
        return self._value("ema_decay")

    # -- cross-batch estimator ----------------------------------------------

    def bind(self, layer: tf.keras.layers.Layer, n_states: int) -> list:
        """Creates per-layer state for 'layer' and returns it for tracking.

        Called by `PredictiveStateSimplex.build`, which keeps the returned
        estimators as tracked sub-layers so their state is saved with the model.

        Raises:
          ValueError: if this regularizer already holds state for another layer.
        """
        if self._estimator is None:
            return []
        if self._bound_layer is not None and self._bound_layer is not layer:
            raise ValueError(
                f"{type(self).__name__} with 'ema_decay' is already applied to "
                f"layer '{self._bound_layer.name}'; its smoothed statistics are "
                f"per-layer state, so layer '{layer.name}' needs its own instance."
            )
        self._bound_layer = layer
        self._estimator.create_state(n_states)
        return [self._estimator]

    def smoothed_statistics(self) -> Dict[str, tf.Tensor]:
        """Current bias-corrected estimates of the smoothed statistics.

        Empty when `ema_decay` is unset or before the first call.
        """
        if self._estimator is None or self._estimator.step is None:
            return {}
        return self._estimator.estimates()

    @property
    def smoothing_steps(self) -> int:
        """Number of training batches the smoothed statistics have averaged."""
        return 0 if self._estimator is None else self._estimator.steps

    # -- pipeline -----------------------------------------------------------

    def _penalty(self, stats: Dict[str, tf.Tensor], n_states: int) -> tf.Tensor:
        """The penalty, before scaling by `l2`, as a function of the statistics."""
        raise NotImplementedError

    def penalty(self, x) -> tf.Tensor:
        """The penalty this regularizer adds to the training loss for batch 'x'.

        This is `l2` times the penalty function of the statistics of 'x': the
        amount added to the (per-example mean) loss, independent of the batch
        size. With `ema_decay`, calling it advances the running averages, as a
        training step does.
        """
        n_states = int(x.shape[-1])
        batch = {name: statistics.compute(name, x) for name in self.STATISTICS}
        if self._estimator is None:
            return self._tensor("l2") * self._penalty(batch, n_states)
        smoothed = self._estimator.update(batch, n_states)
        return self._tensor("l2") * self._smoothed_penalty(smoothed, batch, n_states)

    def _smoothed_penalty(
        self,
        smoothed: Dict[str, tf.Tensor],
        batch: Dict[str, tf.Tensor],
        n_states: int,
    ) -> tf.Tensor:
        """The penalty, before scaling by `l2`, with cross-batch smoothing.

        'smoothed' holds the smoothed statistics, with gradients flowing to the
        batch values; 'batch' holds the batch values. By default the penalty is
        `_penalty` of the smoothed statistics. A subclass overrides this when
        differentiating its penalty at the smoothed statistics would point the
        wrong way for the current batch.
        """
        return self._penalty(smoothed, n_states)

    def __call__(self, x):
        """The activity-regularizer loss for 'x': `penalty(x)` times the batch size.

        Keras divides an activity regularizer's loss by the batch size, because
        its built-in ones sum over the batch. PRESS penalties are already batch
        statistics, so this cancels that division: the loss Keras adds is exactly
        `penalty(x)`, at any batch size. Use `penalty` to inspect the value.
        """
        return _undo_keras_batch_mean(self.penalty(x), x)

    # -- serialization ------------------------------------------------------

    def _extra_config(self) -> dict:
        """Config entries that are not declared hyperparameters."""
        return {}

    def get_config(self):
        """Returns every declared hyperparameter plus any extra options."""
        config = {
            param.name: param.serialize(self._hyperparameters[param.name])
            for param in self.HYPERPARAMETERS
        }
        config.update(self._extra_config())
        return config


def _max_entropy(n_states: int) -> float:
    """log(K), the maximum entropy over K states (guarded to be > 0)."""
    return max(math.log(max(n_states, 1)), 1e-8)


def _log_active_states(fraction: tf.Tensor, n_states: int) -> tf.Tensor:
    """log(fraction * K), clamped at one active state so it is >= 0."""
    return tf.math.log(tf.maximum(1.0, fraction * float(n_states)))


@tf.keras.utils.register_keras_serializable(package="pypress")
class TargetEntropy(_PenaltyRegularizer):
    """Penalizes mean row entropy deviating from a target value.

    By default, targets log(0.5 * K), representing a balanced 50% regime overlap
    anchor (where K = x.shape[1] at call time). Uses a squared (L2) penalty to
    provide smooth gradient behavior near equilibrium.

    Target = log(K) is maximum overlap (uniform distribution).
    Target = 0 is minimum overlap (deterministic / single-state assignment).

    The deviation (target - mean entropy) is normalized by log(K), the maximum
    possible entropy for K columns, before squaring. Without this, the penalty's
    dynamic range grows as log(K)^2 while the (unrelated) mixture negative
    log-likelihood's does not, so the effective regularization strength would
    silently depend on the number of states K. Normalizing keeps the penalty
    bounded in [0, l2] regardless of K, so `l2` means the same thing across
    different values of K.
    """

    HYPERPARAMETERS = (
        hp.L2,
        _Hyperparameter("target", default=None, low=0.0, optional=True),
        _Hyperparameter(
            "entropy_fraction",
            default=None,
            low=0.0,
            high=1.0,
            low_inclusive=False,
            optional=True,
        ),
        hp.EMA_DECAY,
    )
    STATISTICS = ("mean_row_entropy",)

    # Fraction of K used as the target when neither target is given.
    _default_entropy_fraction = 0.5

    def __init__(
        self,
        l2: float = 0.0,
        *,
        target: float | None = None,
        entropy_fraction: float | None = None,
        ema_decay: float | None = None,
        **kwargs,
    ):
        """Initializes the regularizer.

        Args:
          l2: Penalty multiplier weight for the squared deviation from target.
          target: Target mean entropy value, in nats (>= 0). If None, derived
            from `entropy_fraction`, with K = x.shape[1] computed dynamically at
            call time. May be a `ScheduledValue`.
          entropy_fraction: Target expressed as a fraction of K, i.e. the target
            is `log(entropy_fraction * K)`. Unlike `target` this does not depend
            on K, so it is the schedulable knob to prefer. Must be in (0, 1] and
            is mutually exclusive with `target`. Defaults to 0.5 (1.0 for
            `Uniform`).

            At initialization the weights are uniform, so the mean row entropy is
            exactly `log(K)`, i.e. `entropy_fraction = 1.0`. Scheduling it from
            1.0 down to the intended value therefore starts the constraint
            exactly where the model already is and tightens from there, rather
            than pulling against the initialization from epoch 0.
          ema_decay: If set, evaluate the penalty on a bias-corrected moving
            average of the mean row entropy across training batches.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        if target is not None and entropy_fraction is not None:
            raise ValueError(
                "Pass either 'target' (absolute, in nats) or 'entropy_fraction' "
                "(relative to K), not both."
            )
        super().__init__(
            l2=l2,
            target=target,
            entropy_fraction=entropy_fraction,
            ema_decay=ema_decay,
            **kwargs,
        )

    @property
    def target(self):
        """The entropy target, as configured (float, None or ScheduledValue)."""
        return self._value("target")

    @property
    def entropy_fraction(self):
        """The target as a fraction of K (float or ScheduledValue)."""
        fraction = self._value("entropy_fraction")
        return self._default_entropy_fraction if fraction is None else fraction

    def _penalty(self, stats, n_states):
        """Squared, log(K)-normalized deviation of mean row entropy from target."""
        if self.target is not None:
            target = self._tensor("target")
        else:
            target = _log_active_states(
                _Hyperparameter.tensor(self.entropy_fraction), n_states
            )
        # max_entropy is guarded for K == 1, where deviation is also 0.
        deviation = (target - stats["mean_row_entropy"]) / _max_entropy(n_states)
        return tf.math.square(deviation)


@tf.keras.utils.register_keras_serializable(package="pypress")
class Uniform(TargetEntropy):
    """Penalizes weights if they are not uniform across columns (1 / K).

    Special case of TargetEntropy where the target is fixed to the maximum
    possible entropy for uniform weights, log(# of columns), always computed
    dynamically from the input. For a custom target, use TargetEntropy directly.
    """

    HYPERPARAMETERS = (hp.L2, hp.EMA_DECAY)
    _default_entropy_fraction = 1.0

    def __init__(self, l2: float = 0.0, *, ema_decay: float | None = None, **kwargs):
        """Initializes the uniform regularizer.

        Args:
          l2: penalty multiplier weight for the squared deviation from target.
          ema_decay: If set, smooth the mean row entropy across batches.
          **kwargs: addl keyword arguments to regularizers.
        """
        _PenaltyRegularizer.__init__(self, l2=l2, ema_decay=ema_decay, **kwargs)


@tf.keras.utils.register_keras_serializable(package="pypress")
class StateSizeEntropy(_PenaltyRegularizer):
    """Penalizes population-level state usage concentrated on a few states.

    Complements `TargetEntropy`/`Uniform`, which look at the *rows* of the weight
    matrix (how sharp each observation's state assignment is), and
    `DegreesOfFreedom`, which measures the *effective* number of states. Neither
    sees a dead state: it has ~0 state share, so it contributes ~0 to trace(H)
    and leaves the mean row entropy essentially untouched. A model with K = 50
    states of which 45 are dead is therefore indistinguishable from an honest
    K = 5 model under both.

    This regularizer looks at the state shares instead,

        p_k = state_size(W)_k / sum_l state_size(W)_l,

    and penalizes their deviation from uniform usage via the (normalized)
    Kullback-Leibler divergence

        penalty = l2 * max(0, log(active_fraction * K) - H(p)) / log(K),

    where H(p) is the Shannon entropy of the state shares. For
    `active_fraction=1.0` the hinge is inactive (H(p) <= log(K) always) and the
    penalty reduces to l2 * KL(p || Uniform_K) / log(K).

    `exp(H(p))` is the effective number of live states, so the penalty is a smooth
    surrogate for the fraction of states left unused: given an equal fit and equal
    degrees of freedom, it prefers the model that actually uses the states it has.

    Dividing by log(K), the maximum possible entropy for K columns, keeps the
    penalty bounded in [0, l2] regardless of the number of states, so `l2` means
    the same thing across different values of K -- matching the K-normalization
    used by `TargetEntropy`. Note that H(p) alone does *not* discriminate a
    collapsed model from a small honest one (both have the same marginal entropy);
    it is the size of that entropy *relative to* log(K) that does.

    Together with `TargetEntropy` this bounds both halves of the mutual information
    between inputs and states, I(x; state) = H(p) - E_i[H(w_i)]: `TargetEntropy`
    controls the mean row entropy, this controls the marginal entropy. The two
    penalties are complementary, not redundant.

    Note on smoothness: for `active_fraction=1.0` the penalty behaves like half the
    chi-squared divergence near the uniform optimum, so its gradient vanishes
    smoothly there. For `active_fraction < 1.0` the hinge introduces a kink at the
    target, the same mild subgradient behaviour as a ReLU or an L1 penalty.
    """

    HYPERPARAMETERS = (
        hp.L2,
        _Hyperparameter(
            "active_fraction", default=1.0, low=0.0, high=1.0, low_inclusive=False
        ),
        hp.EMA_DECAY,
    )
    STATISTICS = ("state_sizes",)

    def __init__(
        self,
        l2: float = 0.0,
        *,
        active_fraction: float = 1.0,
        ema_decay: float | None = None,
        **kwargs,
    ):
        """Initializes the regularizer.

        Args:
          l2: Penalty multiplier for the normalized KL divergence of the state
            shares from uniform usage. The penalty is bounded in [0, l2].
          active_fraction: Fraction of the K states required to be effectively
            live, i.e. the target is `exp(H(p)) >= active_fraction * K`. Must be
            in (0, 1]. The default of 1.0 targets fully uniform state usage;
            smaller values only penalize usage below that floor and leave models
            that spread across more states unpenalized.
          ema_decay: If set, evaluate the penalty on a bias-corrected moving
            average of state sizes across training batches.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        super().__init__(
            l2=l2, active_fraction=active_fraction, ema_decay=ema_decay, **kwargs
        )

    @property
    def active_fraction(self):
        """The required fraction of live states (float or ScheduledValue)."""
        return self._value("active_fraction")

    def _penalty(self, stats, n_states):
        """One-sided, log(K)-normalized shortfall of the state-share entropy."""
        marginal_entropy = statistics.entropy(
            statistics.state_shares(stats["state_sizes"])
        )
        target = _log_active_states(self._tensor("active_fraction"), n_states)
        # One-sided hinge (`max(0, .)`), not an activation: usage *above* the
        # target must cost exactly zero, never a negative penalty.
        return tf.maximum(0.0, target - marginal_entropy) / _max_entropy(n_states)


@tf.keras.utils.register_keras_serializable(package="pypress")
class MinStateSize(_PenaltyRegularizer):
    """Penalizes predictive states holding less than `min_share` of the population.

    Expresses the practical requirement "a state is only worth having if at least
    x% of observations fall in it" directly, in the units the modeler thinks in.
    Given the state shares

        p_k = state_size(W)_k / sum_l state_size(W)_l,

    the penalty is the mean squared relative shortfall below the floor:

        penalty = l2 * mean_k( max(0, 1 - p_k / min_share) ** 2 ).

    It is exactly zero if and only if every state clears `min_share`, and it says
    nothing about how the mass is distributed among states that already clear it --
    unlike `StateSizeEntropy`, which pulls the whole marginal toward uniform. Use
    this when the true state sizes are legitimately unequal but none may vanish.

    Prefer this over `StateSizeEntropy` for a per-state floor. Entropy is a single
    scalar and cannot encode K separate floors: at K = 9, a marginal with one state
    at 0.1% and the other eight sharing the rest has higher entropy than the worst
    configuration that respects a 5% floor, so an entropy-based penalty scores it
    zero while this one does not.

    The relative form `p_k / min_share` and the mean over states keep the penalty
    bounded in [0, l2) and scale-invariant in K, so `l2` means the same thing across
    different numbers of states -- matching the K-normalization used by
    `TargetEntropy` and `StateSizeEntropy`. The worst case (all mass on one state)
    approaches `l2 * (K - 1) / K`.

    The hinge is squared, so the penalty is continuously differentiable at the floor:
    the gradient fades to zero as a state reaches `min_share` rather than dropping
    discontinuously.
    """

    HYPERPARAMETERS = (
        hp.L2,
        _Hyperparameter(
            "min_share",
            default=None,
            low=0.0,
            high=1.0,
            low_inclusive=False,
            optional=True,
        ),
        hp.EMA_DECAY,
    )
    STATISTICS = ("state_sizes",)

    # Fraction of the fair share (1 / K) used as the dynamic default floor.
    _default_fair_share_fraction = 0.5

    def __init__(
        self,
        l2: float = 0.0,
        *,
        min_share: float | None = None,
        ema_decay: float | None = None,
        **kwargs,
    ):
        """Initializes the regularizer.

        Args:
          l2: Penalty multiplier for the mean squared relative shortfall. The
            penalty is bounded in [0, l2).
          min_share: Minimum state share required of every state, e.g. 0.05 for
            "at least 5% of observations per state". Must be in (0, 1]. Note
            `min_share * K <= 1` is required for the floor to be satisfiable at
            all; this is checked at call time, once K is known. If None, defaults
            to half the fair share, `0.5 / K`, computed dynamically from K.
          ema_decay: If set, evaluate the penalty on a bias-corrected moving
            average of state sizes across training batches. Strongly recommended
            when `batch_size * min_share` is below ~10.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        super().__init__(l2=l2, min_share=min_share, ema_decay=ema_decay, **kwargs)

    @property
    def min_share(self):
        """The per-state floor, as configured (float, None or ScheduledValue)."""
        return self._value("min_share")

    def _floor(self, n_states: int) -> tf.Tensor:
        """Resolves the floor for K states, defaulting to half the fair share."""
        if self.min_share is None:
            return tf.constant(self._default_fair_share_fraction / n_states)
        # Validate the strictest value the schedule can reach, not just today's.
        largest = max(_Hyperparameter.bounds(self.min_share))
        if largest * n_states > 1.0:
            raise ValueError(
                f"'min_share' of {largest} is not satisfiable with K = {n_states} "
                f"states: it would require {largest * n_states:.2f} of the "
                f"population. Use min_share <= {1.0 / n_states:.4f}, or fewer states."
            )
        return self._tensor("min_share")

    def _penalty(self, stats, n_states):
        """Mean squared relative shortfall of state shares below the floor."""
        shares = statistics.state_shares(stats["state_sizes"])
        # One-sided hinge, squared for a gradient that fades to zero at the floor.
        shortfall = tf.maximum(0.0, 1.0 - shares / self._floor(n_states))
        return tf.math.reduce_mean(tf.math.square(shortfall))


def _trace(stats: Dict[str, tf.Tensor]) -> tf.Tensor:
    """trace(H) from the trace statistics in 'stats'."""
    return statistics.degrees_of_freedom(
        stats["trace_numerator"], stats["trace_denominator"]
    )


@tf.keras.utils.register_keras_serializable(package="pypress")
class DegreesOfFreedom(_PenaltyRegularizer):
    """Penalizes degrees of freedom, trace(H), deviating from a target.

    The smoother matrix implied by the predictive state weights is

        H = W * D^(-1) * W' in R^{N x N},

    where D is a diagonal matrix with D_kk = size of state k = sum_i w_ik.

    Degrees of freedom of a kernel smoother is the trace of its smoother matrix.
    In general the trace must be computed from the full N x N diagonal, which can
    be prohibitive if N is large. Due to the special structure of H and
    properties of the trace operator, this simplifies to

        trace(H) = trace(W * D^-1 * W') = trace(W' * W * D^-1),

    which is the trace of a K x K matrix, where K << N is the number of states.

    Note this measures the *effective* number of states: for hard assignments
    `trace(H)` equals the number of non-empty states exactly. A `target` well
    below the layer's `n_states` therefore does not merely tolerate dead states,
    it optimizes for them. If the intent is a smaller model, reduce `n_states`.

    Both `l2` and `target` accept a `pypress.keras.schedules.ScheduledValue`.
    Which to schedule depends on where training starts: at initialization the
    weights are uniform, so `trace(H)` is exactly 1 regardless of `n_states` --
    the states are all alive but indistinguishable, and the trace *grows* as they
    differentiate. A fixed `target` well below `n_states` therefore caps
    differentiation from the first step rather than pruning states later.

    Warming up `l2` from 0 is usually the right choice: it lets the trace grow on
    the fit signal alone and starts pruning once the states mean something.
    Annealing `target` downward from roughly `n_states` is the more aggressive
    alternative, which actively forces differentiation before pruning.

    `target` is always an absolute effective-state count. Set `normalize=True`
    to divide its deviation by K = x.shape[1] before squaring, so an `l2` tuned
    for a given fractional trace error transfers across different `n_states`.
    This changes only penalty scale, never the target being optimized.
    """

    HYPERPARAMETERS = (
        hp.L2,
        _Hyperparameter("target", default=1.0, low=1.0),
        hp.EMA_DECAY,
    )
    # Smoothed separately, then divided: averaging the ratio itself would keep
    # the small-batch bias of a ratio estimator.
    STATISTICS = ("trace_numerator", "trace_denominator")

    def __init__(
        self,
        l2: float = 0.0,
        target: float = 1.0,
        df: float = None,
        *,
        normalize: bool = False,
        ema_decay: float | None = None,
        **kwargs,
    ):
        """Initializes the regularizer.

        Args:
          l2: l2 penalty parameter for l2 * (target - trace(H)) ** 2. May be a
            `ScheduledValue`.
          target: Absolute degrees-of-freedom target. Must be >= 1. May be a
            `ScheduledValue`.
          df: Deprecated alias for `target`.
          normalize: Divide the target deviation by K = x.shape[1] before
            squaring. This makes the penalty scale-invariant in K while retaining
            `target` in its natural effective-state-count units.
          ema_decay: If set, average trace(H)'s per-state numerator and
            denominator across training batches before taking their ratio.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        if df is not None:
            warnings.warn(
                "'df' is deprecated. Use 'target' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            target = df
        super().__init__(l2=l2, target=target, ema_decay=ema_decay, **kwargs)
        self._normalize = bool(normalize)

    @property
    def target(self):
        """The degrees-of-freedom target, as configured (float or ScheduledValue)."""
        return self._value("target")

    @property
    def normalize(self):
        """Whether the target deviation is normalized by the number of states."""
        return self._normalize

    def _penalty(self, stats, n_states):
        """Squared deviation of trace(H) from the target."""
        return self._deviation_penalty(_trace(stats), n_states)

    def _smoothed_penalty(self, smoothed, batch, n_states):
        """The smoothed trace(H)'s squared deviation, with the batch trace's gradient.

        The value, and the size and sign of the gradient, come from the smoothed
        trace(H); its direction comes from the batch trace(H). Differentiating
        the smoothed trace itself is wrong for a state that is shrinking: its
        running averages still hold its earlier size, which makes the gradient
        shrink it further, until the model collapses onto one state.
        """
        smoothed_trace = tf.stop_gradient(_trace(smoothed))
        batch_trace = _trace(batch)
        trace = smoothed_trace + (batch_trace - tf.stop_gradient(batch_trace))
        return self._deviation_penalty(trace, n_states)

    def _deviation_penalty(self, trace, n_states):
        """Squared deviation of 'trace' from the target, optionally normalized."""
        deviation = trace - self._tensor("target")
        if self._normalize:
            # K, rather than batch size, is the stable architectural scale.
            deviation = deviation / float(max(n_states, 1))
        return tf.square(deviation)

    def _extra_config(self):
        """Adds the `normalize` option."""
        return {"normalize": self._normalize}


def _resolve_regularizer_class(name: str):
    """Looks up a regularizer class by its (old-config) name."""
    registered = tf.keras.utils.get_registered_object(f"pypress>{name}")
    if registered is not None:
        return registered
    return globals()[name]


@tf.keras.utils.register_keras_serializable(package="pypress")
class CombinedRegularizer(tf.keras.regularizers.Regularizer):
    """Sums the penalties of several regularizers.

    Example:
        >>> reg = CombinedRegularizer([
        ...     MinStateSize(l2=5.0, min_share=0.05),
        ...     DegreesOfFreedom(l2=1.0, target=4.0),
        ... ])

    The older form, a list of `(regularizer_class, kwargs)` tuples, still works
    but is deprecated.
    """

    def __init__(self, regularizers=None, *, regularizer_tuples=None, **kwargs):
        """Initializes the composite.

        Args:
          regularizers: Regularizer instances to sum. A list of
            `(regularizer_class, kwargs)` tuples is accepted but deprecated.
          regularizer_tuples: Deprecated keyword form of the tuple list.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        super().__init__(**kwargs)
        if regularizer_tuples is not None:
            if regularizers is not None:
                raise ValueError(
                    "Pass 'regularizers' or 'regularizer_tuples', not both."
                )
            regularizers = regularizer_tuples
        regularizers = list(regularizers or [])
        if any(isinstance(item, tuple) for item in regularizers):
            warnings.warn(
                "CombinedRegularizer with (class, kwargs) tuples is deprecated; "
                "pass regularizer instances instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            regularizers = [
                item[0](**item[1]) if isinstance(item, tuple) else item
                for item in regularizers
            ]
        self.regularizers = regularizers

    def penalty(self, x) -> tf.Tensor:
        """The sum of the children's penalties on 'x'; see `_PenaltyRegularizer.penalty`.

        A child without `penalty` (e.g. a built-in Keras regularizer) contributes
        what Keras would add for it: its loss divided by the batch size.
        """
        total_penalty = 0.0
        for reg in self.regularizers:
            if callable(getattr(reg, "penalty", None)):
                total_penalty += reg.penalty(x)
            else:
                total_penalty += reg(x) / _batch_size(x, tf.float32)
        return total_penalty

    def __call__(self, x):
        """The sum of the children's activity-regularizer losses on 'x'.

        Each PRESS child already cancels Keras's division by the batch size, so
        the loss Keras adds for this regularizer is exactly `penalty(x)`.
        """
        total_penalty = 0.0
        for reg in self.regularizers:
            total_penalty += reg(x)
        return total_penalty

    def bind(self, layer: tf.keras.layers.Layer, n_states: int) -> list:
        """Binds each child that keeps per-layer state; see `_PenaltyRegularizer.bind`."""
        state = []
        for reg in self.regularizers:
            bind = getattr(reg, "bind", None)
            if callable(bind):
                state.extend(bind(layer, n_states))
        return state

    def get_config(self):
        """Serializes each child regularizer."""
        return {
            "regularizers": [
                tf.keras.regularizers.serialize(reg) for reg in self.regularizers
            ]
        }

    @classmethod
    def from_config(cls, config):
        """Recreates the composite; also reads configs from pypress <= 0.3."""
        config = dict(config)
        if "regularizer_tuples" in config:
            # Old format: [(class name, kwargs), ...].
            regularizers = [
                _resolve_regularizer_class(name)(**kwargs)
                for name, kwargs in config.pop("regularizer_tuples")
            ]
        else:
            regularizers = [
                tf.keras.regularizers.deserialize(item)
                for item in config.pop("regularizers", [])
            ]
        return cls(regularizers=regularizers, **config)


@tf.keras.utils.register_keras_serializable(package="pypress")
class UniformAndDegreesOfFreedomRegularizer(CombinedRegularizer):
    """Deprecated: `Uniform` (or `TargetEntropy`) plus `DegreesOfFreedom`.

    Use `CombinedRegularizer([Uniform(...), DegreesOfFreedom(...)])` instead.

    Keyword arguments:
      uniform_l2: float, penalty weight for the Uniform regularizer (squared deviation).
      dof_l2: float, penalty weight for the DegreesOfFreedom regularizer.
      dof_target: Absolute target value for the degrees of freedom.
      dof_normalize: Normalize the DoF target deviation by K before squaring.
      target_entropy: If set, use `TargetEntropy` with this target instead of
        `Uniform`.
    """

    def __init__(
        self,
        uniform_l2: float = 0.0,
        dof_l2: float = 0.0,
        dof_target: float = 1.0,
        dof_normalize: bool = False,
        target_entropy: float = None,
        **kwargs,
    ):
        """Initializes the pair."""
        warnings.warn(
            "UniformAndDegreesOfFreedomRegularizer is deprecated; use "
            "CombinedRegularizer([Uniform(...), DegreesOfFreedom(...)]).",
            DeprecationWarning,
            stacklevel=2,
        )
        entropy = (
            TargetEntropy(l2=uniform_l2, target=target_entropy)
            if target_entropy is not None
            else Uniform(l2=uniform_l2)
        )
        dof = DegreesOfFreedom(l2=dof_l2, target=dof_target, normalize=dof_normalize)
        super().__init__(regularizers=[entropy, dof], **kwargs)
        self._dof_normalize = bool(dof_normalize)

    @property
    def uniform_l2(self):
        """Penalty weight of the entropy regularizer."""
        return self.regularizers[0].l2

    @property
    def dof_l2(self):
        """Penalty weight of the degrees-of-freedom regularizer."""
        return self.regularizers[1].l2

    @property
    def dof_target(self):
        """Degrees-of-freedom target."""
        return self.regularizers[1].target

    @property
    def dof_normalize(self):
        """Whether the degrees-of-freedom deviation is normalized by K."""
        return self._dof_normalize

    @property
    def target_entropy(self):
        """Explicit entropy target, or None for `Uniform`."""
        entropy = self.regularizers[0]
        return None if isinstance(entropy, Uniform) else entropy.target

    def get_config(self):
        """Gets the config, in the original flat format."""
        return {
            "uniform_l2": _Hyperparameter.serialize(self.uniform_l2),
            "dof_l2": _Hyperparameter.serialize(self.dof_l2),
            "dof_target": _Hyperparameter.serialize(self.dof_target),
            "dof_normalize": self.dof_normalize,
            "target_entropy": _Hyperparameter.serialize(self.target_entropy),
        }

    @classmethod
    def from_config(cls, config):
        """Recreates the pair from its flat config."""
        return cls(**config)
