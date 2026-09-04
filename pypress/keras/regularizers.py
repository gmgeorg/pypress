"""Regularizers for PRESS weights."""

import tensorflow as tf
import warnings

from . import schedules
from .. import utils


def _schedule_bounds(value) -> tuple:
    """Returns every value a scalar can take: both endpoints if it is scheduled."""
    if isinstance(value, schedules.ScheduledValue):
        return (value.start, value.end)
    return () if value is None else (value,)


class _EMASmoother:
    """Bias-corrected exponential moving average of a tensor, across batches.

    Value is the smoothed estimate; the gradient is passed straight through to
    the current batch, so smoothing changes *what* the penalty is evaluated on
    without detaching it from the weights being trained. Same arrangement as EMA
    codebooks in VQ-VAE.
    """

    def __init__(self, decay: float, name: str):
        """Initializes the smoother.

        Args:
          decay: EMA decay in [0, 1). Effective sample size is
            ~ batch_size / (1 - decay).
          name: Name prefix for the backing variables.
        """
        self._decay = decay
        self._name = name
        self.average = None
        self.step = None

    def __call__(self, value: tf.Tensor) -> tf.Tensor:
        """Updates the average with 'value' and returns the smoothed estimate."""
        if self.average is None:
            # Lifted out of any enclosing tf.function so the variables are created
            # once, eagerly, on the first (tracing) call. The initial value is
            # built from the static shape and dtype rather than from `value`
            # itself: inside `init_scope` a graph tensor from the enclosing
            # function is out of scope and cannot be read.
            with tf.init_scope():
                self.average = tf.Variable(
                    tf.zeros(value.shape, dtype=value.dtype),
                    trainable=False,
                    name=f"{self._name}_ema",
                )
                self.step = tf.Variable(
                    0.0, trainable=False, name=f"{self._name}_ema_step"
                )

        decay = tf.cast(self._decay, value.dtype)
        self.average.assign(
            decay * self.average + (1.0 - decay) * tf.stop_gradient(value)
        )
        self.step.assign_add(1.0)
        # Adam-style bias correction, so early batches are not dragged toward the
        # zero initialization.
        corrected = self.average / tf.maximum(1.0 - tf.pow(decay, self.step), 1e-8)
        return tf.stop_gradient(corrected - value) + value


class _SmoothedRegularizer(tf.keras.regularizers.Regularizer):
    """Base for PRESS regularizers, with a schedulable `l2` and optional smoothing.

    Every regularizer here reduces a batch of weights to one or more summary
    statistics and then penalizes those. All of those statistics are estimated
    from a single minibatch, because these attach as `activity_regularizer` on
    `PredictiveStateSimplex`, and every penalty here is a convex function of them.
    Jensen's inequality then makes each batch estimate biased *upward*:
    `E[f(stat_batch)] >= f(E[stat_batch]) = f(stat_population)`. Running more
    batches per epoch shrinks the variance of the epoch average but not this bias
    -- only a larger batch, or averaging the statistic across batches, does.

    Measured with a population that is exactly compliant (K = 9, one state at
    exactly 5%, `MinStateSize(min_share=0.05)`, so the true penalty is 0), the
    mean penalty is 0.0515 at `batch_size=32` and identical whether averaged over
    200 or 5000 batches; it falls to 0.0009 at `batch_size=1024`.

    `ema_decay` smooths the statistics across batches instead. What gets smoothed
    is per-regularizer -- the row-entropy mean, the state-size marginal, or the
    kernel trace's numerator and denominator separately -- since each penalty
    consumes a different statistic.
    """

    def __init__(self, l2: float = 0.0, *, ema_decay: float | None = None, **kwargs):
        """Initializes the shared machinery.

        Args:
          l2: Penalty multiplier. May be a float or a
            `pypress.keras.schedules.ScheduledValue`.
          ema_decay: If set, evaluate the penalty on bias-corrected moving
            averages of its summary statistics rather than on single-batch
            estimates. Must be in [0, 1). Typical values are 0.9 to 0.99.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        if ema_decay is not None and not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"'ema_decay' must be in [0, 1). Got {ema_decay}.")
        super().__init__(**kwargs)
        self._l2 = schedules.deserialize_scalar(l2)
        self._ema_decay = ema_decay
        self._smoothers: dict[str, _EMASmoother] = {}

    @property
    def l2(self):
        """The penalty multiplier, as configured (float or ScheduledValue)."""
        return self._l2

    @property
    def ema_decay(self) -> float | None:
        """The EMA decay, or None when statistics are used un-smoothed."""
        return self._ema_decay

    def _smooth(self, name: str, value: tf.Tensor) -> tf.Tensor:
        """Smooths a named summary statistic across batches, if enabled."""
        if self._ema_decay is None:
            return value
        if name not in self._smoothers:
            self._smoothers[name] = _EMASmoother(self._ema_decay, name)
        return self._smoothers[name](value)

    def _base_config(self):
        """Config entries shared by all regularizers here."""
        config = {"l2": schedules.serialize_scalar(self._l2)}
        if self._ema_decay is not None:
            config["ema_decay"] = float(self._ema_decay)
        return config


class _StateSizeRegularizer(_SmoothedRegularizer):
    """Base for regularizers acting on the population marginal over states.

    Turns a batch of PRESS weights into the state-size marginal
    `p_k = state_size(W)_k / sum_j state_size(W)_j`, smoothed across batches when
    `ema_decay` is set. See `_SmoothedRegularizer` for why smoothing matters.
    """

    def _state_shares(self, x):
        """Computes the state-size marginal, EMA-smoothed if `ema_decay` is set."""
        state_sizes = self._smooth("state_sizes", utils.tf_state_size(x))
        # Rows of a PRESS weight matrix sum to 1, but normalize explicitly so the
        # marginal is a proper distribution for any non-negative input.
        total = tf.maximum(tf.math.reduce_sum(state_sizes), 1e-8)
        return state_sizes / total


@tf.keras.utils.register_keras_serializable(package="pypress")
class TargetEntropy(_SmoothedRegularizer):
    """Penalizes weights if mean entropy deviates from a specific target value.

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

    # Fraction of K used as the dynamic default target active states.
    _entropy_fraction = 0.5

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
          target: Target mean entropy value, in nats. If None, derived from
            `entropy_fraction`, with K = x.shape[1] computed dynamically at call
            time. May be a `ScheduledValue`.
          entropy_fraction: Target expressed as a fraction of K, i.e. the target
            is `log(entropy_fraction * K)`. Unlike `target` this does not depend
            on K, so it is the schedulable knob to prefer. Must be in (0, 1] and
            is mutually exclusive with `target`. Defaults to the class's
            `_entropy_fraction` (0.5 here, 1.0 for `Uniform`).

            At initialization the weights are uniform, so the mean row entropy is
            exactly `log(K)`, i.e. `entropy_fraction = 1.0`. Scheduling it from
            1.0 down to the intended value therefore starts the constraint
            exactly where the model already is and tightens from there, rather
            than pulling against the initialization from epoch 0.
          ema_decay: If set, evaluate the penalty on a bias-corrected moving
            average of the mean row entropy across batches. See
            `_SmoothedRegularizer`.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        if target is not None and entropy_fraction is not None:
            raise ValueError(
                "Pass either 'target' (absolute, in nats) or 'entropy_fraction' "
                "(relative to K), not both."
            )
        entropy_fraction = schedules.deserialize_scalar(entropy_fraction)
        if isinstance(entropy_fraction, schedules.ScheduledValue):
            bounds = (entropy_fraction.start, entropy_fraction.end)
        else:
            bounds = () if entropy_fraction is None else (entropy_fraction,)
        for bound in bounds:
            if not 0.0 < bound <= 1.0:
                raise ValueError(
                    f"'entropy_fraction' must be in (0, 1]. Got {entropy_fraction}."
                )

        super().__init__(l2=l2, ema_decay=ema_decay, **kwargs)
        self._target = schedules.deserialize_scalar(target)
        self._entropy_fraction = (
            self._entropy_fraction if entropy_fraction is None else entropy_fraction
        )

    def _target_value(self, x):
        """Computes or retrieves the target entropy value for input tensor 'x'."""
        if self._target is not None:
            return schedules.as_tensor(self._target)

        K = tf.cast(x.shape[1], dtype=tf.float32)
        # Clamped at 1.0 active state so log(...) is non-negative for K >= 1
        fraction = schedules.as_tensor(self._entropy_fraction)
        active_states = tf.maximum(1.0, fraction * K)
        return tf.math.log(active_states)

    def _max_entropy(self, x):
        """Computes log(K), the maximum possible entropy for K = x.shape[1] columns."""
        K = tf.cast(x.shape[1], dtype=tf.float32)
        return tf.math.log(tf.maximum(K, 1.0))

    def __call__(self, x):
        """Computes squared penalty for deviation of mean entropy from target.

        The deviation is normalized by log(K) so the penalty is scale-invariant
        with respect to K = x.shape[1], keeping it bounded in [0, l2].
        """
        entropy_per_row = -1.0 * tf.math.reduce_sum(utils.safe_xlogx(x), axis=1)
        mean_entropy = self._smooth(
            "mean_row_entropy", tf.math.reduce_mean(entropy_per_row)
        )

        # Squared, K-normalized deviation (L2 penalty) for smooth optimizer updates
        deviation = self._target_value(x) - mean_entropy
        # max_entropy is 0 only when K == 1, in which case deviation is also 0
        # (both entropy and its clamped target are 0); guard against 0/0.
        normalized_deviation = deviation / tf.maximum(self._max_entropy(x), 1e-8)
        return schedules.as_tensor(self._l2) * tf.math.square(normalized_deviation)

    @property
    def target(self):
        """The entropy target, as configured (float, None or ScheduledValue)."""
        return self._target

    @property
    def entropy_fraction(self):
        """The target as a fraction of K (float or ScheduledValue)."""
        return self._entropy_fraction

    def get_config(self):
        """Returns the serializable config dictionary for Keras."""
        config = self._base_config()
        if self._target is not None:
            config["target"] = schedules.serialize_scalar(self._target)
        elif self._entropy_fraction != type(self)._entropy_fraction:
            config["entropy_fraction"] = schedules.serialize_scalar(
                self._entropy_fraction
            )
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class Uniform(TargetEntropy):
    """Penalizes weights if they are not uniform across columns (1 / J).

    Special case of TargetEntropy where the target is fixed to the maximum
    possible entropy for uniform weights, log(# of columns), always computed
    dynamically from the input. For a custom target, use TargetEntropy directly.
    """

    _entropy_fraction = 1.0

    def __init__(self, l2: float = 0.0, *, ema_decay: float | None = None, **kwargs):
        """Initializes the uniform regularizer.

        Args:
          l2: penalty multiplier weight for the squared deviation from target.
          ema_decay: If set, smooth the mean row entropy across batches.
          **kwargs: addl keyword arguments to regularizers.
        """
        super().__init__(l2=l2, target=None, ema_decay=ema_decay, **kwargs)

    def get_config(self):
        """Gets the config."""
        return self._base_config()


@tf.keras.utils.register_keras_serializable(package="pypress")
class StateSizeEntropy(_StateSizeRegularizer):
    """Penalizes population-level state usage concentrated on a few states.

    Complements `TargetEntropy`/`Uniform`, which look at the *rows* of the weight
    matrix (how sharp each observation's state assignment is), and
    `DegreesOfFreedom`, which measures the *effective* number of states. Neither
    sees a state that is unused across the entire population: a dead state has
    ~0 column mass, so it contributes ~0 to the kernel trace and leaves the mean
    row entropy essentially untouched. A model with K = 50 states of which 45 are
    dead is therefore indistinguishable from an honest K = 5 model under both.

    This regularizer looks at the column marginal instead,

        p_k = state_size(W)_k / sum_j state_size(W)_j,

    the share of the population assigned to state k, and penalizes its deviation
    from uniform usage via the (normalized) Kullback-Leibler divergence

        penalty = l2 * max(0, log(active_fraction * K) - H(p)) / log(K),

    where H(p) is the Shannon entropy of the marginal. For `active_fraction=1.0`
    the hinge is inactive (H(p) <= log(K) always) and the penalty reduces to
    l2 * KL(p || Uniform_K) / log(K).

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
            sizes from uniform usage. The penalty is bounded in [0, l2].
          active_fraction: Fraction of the K states required to be effectively
            live, i.e. the target is `exp(H(p)) >= active_fraction * K`. Must be
            in (0, 1]. The default of 1.0 targets fully uniform state usage;
            smaller values only penalize usage below that floor and leave models
            that spread across more states unpenalized.
          ema_decay: If set, evaluate the penalty on a bias-corrected moving
            average of state sizes across batches. See `_StateSizeRegularizer`.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        active_fraction = schedules.deserialize_scalar(active_fraction)
        for bound in _schedule_bounds(active_fraction):
            if not 0.0 < bound <= 1.0:
                raise ValueError(
                    f"'active_fraction' must be in (0, 1]. Got {active_fraction}."
                )
        super().__init__(l2=l2, ema_decay=ema_decay, **kwargs)
        self._active_fraction = active_fraction

    def _target_value(self, x):
        """Computes the target marginal entropy, log(active_fraction * K)."""
        K = tf.cast(x.shape[1], dtype=tf.float32)
        # Clamped at 1.0 active state so log(...) is non-negative for K >= 1
        fraction = schedules.as_tensor(self._active_fraction)
        active_states = tf.maximum(1.0, fraction * K)
        return tf.math.log(active_states)

    def _max_entropy(self, x):
        """Computes log(K), the maximum possible entropy for K = x.shape[1] columns."""
        K = tf.cast(x.shape[1], dtype=tf.float32)
        return tf.math.log(tf.maximum(K, 1.0))

    def __call__(self, x):
        """Computes the one-sided, K-normalized penalty on marginal state usage."""
        state_shares = self._state_shares(x)
        marginal_entropy = -1.0 * tf.math.reduce_sum(utils.safe_xlogx(state_shares))
        # One-sided hinge (`max(0, .)`), not an activation: usage *above* the
        # target must cost exactly zero, never a negative penalty.
        shortfall = tf.maximum(0.0, self._target_value(x) - marginal_entropy)
        # max_entropy is 0 only when K == 1, in which case shortfall is also 0
        # (both the marginal entropy and its clamped target are 0); guard 0/0.
        return (
            schedules.as_tensor(self._l2)
            * shortfall
            / tf.maximum(self._max_entropy(x), 1e-8)
        )

    def get_config(self):
        """Gets the config."""
        config = self._base_config()
        config["active_fraction"] = schedules.serialize_scalar(self._active_fraction)
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class MinStateSize(_StateSizeRegularizer):
    """Penalizes predictive states holding less than `min_share` of the population.

    Expresses the practical requirement "a state is only worth having if at least
    x% of observations fall in it" directly, in the units the modeler thinks in.
    Given the state-size marginal

        p_k = state_size(W)_k / sum_j state_size(W)_j,

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
          min_share: Minimum fraction of the population required in every state,
            e.g. 0.05 for "at least 5% of observations per state". Must be in
            (0, 1]. Note `min_share * K <= 1` is required for the floor to be
            satisfiable at all; this is checked at call time, once K is known.
            If None, defaults to half the fair share, `0.5 / K`, computed
            dynamically from K = x.shape[1].
          ema_decay: If set, evaluate the penalty on a bias-corrected moving
            average of state sizes across batches. Strongly recommended when
            `batch_size * min_share` is below ~10. See `_StateSizeRegularizer`.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        min_share = schedules.deserialize_scalar(min_share)
        for bound in _schedule_bounds(min_share):
            if not 0.0 < bound <= 1.0:
                raise ValueError(f"'min_share' must be in (0, 1]. Got {min_share}.")
        super().__init__(l2=l2, ema_decay=ema_decay, **kwargs)
        self._min_share = min_share

    def _min_share_value(self, x):
        """Resolves the floor for input 'x', defaulting to half the fair share."""
        K = float(x.shape[1])
        if self._min_share is None:
            return self._default_fair_share_fraction / K

        # Validate the strictest value the schedule can reach, not just today's.
        largest = max(_schedule_bounds(self._min_share))
        if largest * K > 1.0:
            raise ValueError(
                f"'min_share' of {largest} is not satisfiable with K = {int(K)} "
                f"states: it would require {largest * K:.2f} of the population. "
                f"Use min_share <= {1.0 / K:.4f}, or fewer states."
            )
        return schedules.as_tensor(self._min_share)

    def __call__(self, x):
        """Computes the mean squared relative shortfall below the state-size floor."""
        state_shares = self._state_shares(x)
        # One-sided hinge (`max(0, .)`), not an activation: a state above the floor
        # must cost exactly zero, never a negative penalty. Squared for a gradient
        # that fades to zero at the floor instead of dropping discontinuously.
        shortfall = tf.maximum(0.0, 1.0 - state_shares / self._min_share_value(x))
        return schedules.as_tensor(self._l2) * tf.math.reduce_mean(
            tf.math.square(shortfall)
        )

    def get_config(self):
        """Gets the config."""
        config = self._base_config()
        config["min_share"] = (
            None
            if self._min_share is None
            else schedules.serialize_scalar(self._min_share)
        )
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class DegreesOfFreedom(_SmoothedRegularizer):
    """Penalizes weights if the resulting kernel matrix deviates from target degrees of freedom.

    PRESS kernel smoother implied by predictive states equals

        K = W * D^(-1) * W' in R^{N x N},

    where D is a diagonal matrix with D_ii = size of state i = sum_j w_i,j.

    Degrees of freedom of a kernel smoother is equal to the trace of the kernel matrix.
    In general the trace must be computed from the full N x N kernel matrix diagonal,
    which can be prohibitive if N is large.  However, due to special structure
    of the PRESS kernel and properties of trace operator, this can be simplified as

        trace(K) = trace(W * D^-1 * W') = trace(W' * W * D^-1),

    which is the trace of a J x J matrix, where J << N is the number of states.

    Penalizer here is penalizing if the empirical degrees of freedom is different
    to target value.

    Note this measures the *effective* number of states: for hard assignments
    `trace(K)` equals the number of non-empty states exactly. A `target` well
    below the layer's `n_states` therefore does not merely tolerate dead states,
    it optimizes for them. If the intent is a smaller model, reduce `n_states`.

    Both `l2` and `target` accept a `pypress.keras.schedules.ScheduledValue`.
    Which to schedule depends on where training starts: at initialization the
    weights are uniform, so `trace(K)` is exactly 1 regardless of `n_states` --
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
          l2: l2 penalty parameter for l2 * (df - df(kernel)) ** 2. May be a
            `ScheduledValue`.
          target: Absolute degrees-of-freedom target. Must be >= 1. May be a
            `ScheduledValue`.
          df: Deprecated alias for `target`.
          normalize: Divide the target deviation by K = x.shape[1] before
            squaring. This makes the penalty scale-invariant in K while retaining
            `target` in its natural effective-state-count units.
          ema_decay: If set, average the kernel trace's per-state numerator and
            denominator across batches before taking their ratio. The trace is a
            ratio estimator, so it is biased on small batches even before the
            squared penalty is applied. See `_SmoothedRegularizer`.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        if df is not None:
            warnings.warn("'df' is deprecated. Use 'target' instead.")
            target = df

        target = schedules.deserialize_scalar(target)
        if isinstance(target, schedules.ScheduledValue):
            assert min(target.start, target.end) >= 1.0, (
                "Target for degrees of freedom must be >= 1 over the whole "
                f"schedule. Got {target!r}."
            )
        else:
            assert target >= 1.0, (
                f"Target for degrees of freedom must be >= 1. Got {target}."
            )

        super().__init__(l2=l2, ema_decay=ema_decay, **kwargs)
        self._target = target
        self._normalize = normalize

    def __call__(self, x):
        """Computes squared penalty for deviation from target degrees of freedom."""
        # Smooth the numerator and denominator separately, then take the ratio:
        # averaging the ratio itself would keep the small-batch ratio bias.
        numerator, denominator = utils.tr_kernel_terms(x)
        trace = tf.reduce_sum(
            self._smooth("trace_numerator", numerator)
            / tf.maximum(self._smooth("trace_denominator", denominator), 1e-8)
        )
        deviation = trace - schedules.as_tensor(self._target)
        if self._normalize:
            # K, rather than batch size, is the stable architectural scale.
            # The trace's lower bound is 1, so exact endpoint invariance is not
            # possible for finite K; fractional deviations are comparable.
            K = tf.cast(x.shape[1], dtype=trace.dtype)
            deviation = deviation / tf.maximum(K, 1.0)
        return schedules.as_tensor(self._l2) * tf.square(deviation)

    @property
    def target(self):
        """The degrees-of-freedom target, as configured (float or ScheduledValue)."""
        return self._target

    @property
    def normalize(self):
        """Whether the target deviation is normalized by the number of states."""
        return self._normalize

    def get_config(self):
        """Gets the config."""
        config = self._base_config()
        config["target"] = schedules.serialize_scalar(self._target)
        if self._normalize:
            config["normalize"] = self._normalize
        config["df"] = None
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class UniformAndDegreesOfFreedomRegularizer(tf.keras.regularizers.Regularizer):
    """
    A combined regularizer that sums two penalties:
      1. Uniform penalty (to penalize deviations from uniformity across columns)
      2. DegreesOfFreedom penalty (to penalize deviations of the implied kernel trace
         from a target degrees of freedom)

    Keyword arguments:
      uniform_l2: float, penalty weight for the Uniform regularizer (squared deviation).
      dof_l2: float, penalty weight for the DegreesOfFreedom regularizer.
      dof_target: Absolute target value for the degrees of freedom.
      dof_normalize: Normalize the DoF target deviation by K before squaring.
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
        """Initializes the class."""
        super().__init__(**kwargs)
        self.uniform_l2 = uniform_l2
        self.dof_l2 = dof_l2
        self.dof_target = dof_target
        self.dof_normalize = dof_normalize
        self.target_entropy = target_entropy
        # Explicitly instantiate the two internal regularizers. Uniform's target is
        # fixed to log(# of columns); a custom target_entropy requires TargetEntropy.
        self._uniform = (
            TargetEntropy(l2=self.uniform_l2, target=target_entropy)
            if target_entropy is not None
            else Uniform(l2=self.uniform_l2)
        )
        self._dof = DegreesOfFreedom(
            l2=self.dof_l2,
            target=self.dof_target,
            normalize=self.dof_normalize,
        )

    def __call__(self, x):
        # Apply both regularizers and return their sum.
        return self._uniform(x) + self._dof(x)

    def get_config(self):
        """Gets the config."""
        config = {
            "uniform_l2": schedules.serialize_scalar(self.uniform_l2),
            "dof_l2": schedules.serialize_scalar(self.dof_l2),
            "dof_target": schedules.serialize_scalar(self.dof_target),
        }
        if self.dof_normalize:
            config["dof_normalize"] = self.dof_normalize
        if self.target_entropy is not None:
            config["target_entropy"] = schedules.serialize_scalar(self.target_entropy)
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class CombinedRegularizer(tf.keras.regularizers.Regularizer):
    """
    A generic combined regularizer that sums the penalties from a list of regularizers.
    This version accepts a list of tuples of the form:

        [(regularizer_constructor, kwargs_dict), ...]

    and instantiates each regularizer accordingly.
    """

    def __init__(self, regularizer_tuples, **kwargs):
        """Initializes class."""
        super().__init__(**kwargs)
        self.regularizer_tuples = regularizer_tuples
        # Instantiate each regularizer from its constructor and kwargs.
        self.regularizers = [ctor(**kw) for (ctor, kw) in regularizer_tuples]

    def __call__(self, x):
        """Evaluates the regularizer on input."""
        total_penalty = 0.0
        for reg in self.regularizers:
            total_penalty += reg(x)
        return total_penalty

    def get_config(self):
        """Gets the config."""
        # For simplicity, we store the list of tuples as (ctor.__name__, kwargs) pairs.
        config = {
            "regularizer_tuples": [
                (ctor.__name__, kw) for (ctor, kw) in self.regularizer_tuples
            ]
        }
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreates the CombinedRegularizer from its configuration.

        The config is expected to have a key "regularizer_tuples" containing a list of
        tuples of (constructor name, kwargs). We assume that the corresponding constructors
        are registered (or available via direct import) and we look them up.
        """
        # Get the list of tuples from the config.
        regularizer_tuples = config.pop("regularizer_tuples")
        # Import the module and look up the constructors by name.
        import pypress.keras.regularizers as regs

        new_tuples = []
        for ctor_name, kwargs in regularizer_tuples:
            # Get the constructor from the module by name.
            ctor = getattr(regs, ctor_name)
            new_tuples.append((ctor, kwargs))
        return cls(regularizer_tuples=new_tuples, **config)
