"""Regularizers (penalties) on predictive state weights, in PyTorch.

A port of `pypress.keras.regularizers`: the same classes, the same
hyperparameters and the same `get_config()` keys, so a config from one backend
builds the other (see `from_config`).

Every regularizer is the same pipeline:

1. **statistic**: reduce a batch of predictive state weights to named
   population statistics (`pypress.torch.statistics`);
2. **cross-batch estimator** (optional, `ema_decay`): replace each statistic's
   value by a bias-corrected moving average across training batches;
3. **penalty function**: a pure function of the statistics and K, scaled by
   `l2`.

`penalty(x)` returns that value: the amount to add to the training loss.
Calling the module is the same thing. PyTorch has no activity regularizers, so
there is nothing to undo: either call `reg.penalty(weights)` from your own loss,
or attach the regularizer to a layer
(`PredictiveStateSimplex(..., activity_regularizer=reg)`) and add
`pypress.torch.regularization_loss(model)` to the loss.

The regularizers are `nn.Module`s. The running averages of a regularizer with
`ema_decay` and its step count are buffers: they are in `state_dict()` and move
with `.to(device)`. The averages advance iff the regularizer is in training mode,
which is the default on construction; `model.train()` / `model.eval()` reach it
when it is a submodule.

Scalar settings are declared once as `Hyperparameter`s
(`pypress.core.hyperparameters`), which own their validation and serialization.
See CONTEXT.md for the domain terms.
"""

import math
import weakref
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from ..core import hyperparameters as hp
from ..core import statistics as core_statistics
from . import schedules, statistics

_Hyperparameter = hp.Hyperparameter

# Prefix of the registered names Keras uses for these classes.
_REGISTERED_PREFIX = "pypress>"
_SERIALIZED_MODULE = "pypress.keras.regularizers"


class CrossBatchEstimator(nn.Module):
    """Bias-corrected moving average of named statistics across training batches.

    The value returned is the smoothed estimate; the gradient is passed straight
    through to the current batch, so smoothing changes *what* the penalty is
    evaluated on without detaching it from the weights being trained. Same
    arrangement as EMA codebooks in VQ-VAE.

    The average only advances in training mode: in eval mode the current
    estimate is read without updating it. The averages (`<statistic>_ema`) and
    the step count (`ema_step`) are buffers, created by `create_state` once K is
    known. Loading a `state_dict` into an estimator that has not seen K yet
    creates them from the saved shapes.
    """

    def __init__(self, decay: float, statistic_names: Sequence[str]):
        """Initializes the estimator.

        Args:
          decay: EMA decay in [0, 1). Effective sample size is
            ~ batch_size / (1 - decay).
          statistic_names: Names of the statistics to smooth.
        """
        super().__init__()
        self.decay = float(decay)
        self.statistic_names = tuple(statistic_names)
        self.n_states: int | None = None
        self._created = False

    @property
    def is_created(self) -> bool:
        """Whether the averages exist yet."""
        return self._created

    def create_state(
        self,
        n_states: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Creates the averages for 'n_states' predictive states (once).

        Raises:
          ValueError: if the averages already exist for another K.
        """
        if self._created:
            if self.n_states is not None and n_states != self.n_states:
                raise ValueError(
                    f"This regularizer's smoothed statistics were created for "
                    f"K = {self.n_states} predictive states, but it is now applied "
                    f"to K = {n_states}. A regularizer with 'ema_decay' keeps "
                    "per-layer state: use a separate instance for each layer."
                )
            return
        for name in self.statistic_names:
            shape = core_statistics.shape(name, n_states)
            self.register_buffer(
                self._buffer_name(name), torch.zeros(shape, device=device, dtype=dtype)
            )
        self.register_buffer("ema_step", torch.zeros((), device=device, dtype=dtype))
        self.n_states = n_states
        self._created = True

    @staticmethod
    def _buffer_name(name: str) -> str:
        """Name of the buffer holding the average of statistic 'name'."""
        return f"{name}_ema"

    def _average(self, name: str) -> torch.Tensor:
        """The raw (not bias-corrected) running average of statistic 'name'."""
        return getattr(self, self._buffer_name(name))

    def estimates(self) -> Dict[str, torch.Tensor]:
        """Current bias-corrected estimates, keyed by statistic name.

        Empty before the state exists.
        """
        if not self._created:
            return {}
        step = self.ema_step
        decay = torch.full_like(step, self.decay)
        correction = (1.0 - decay.pow(step)).clamp_min(1e-8)
        return {name: self._average(name) / correction for name in self.statistic_names}

    @property
    def steps(self) -> int:
        """Number of training batches averaged so far."""
        return int(self.ema_step.item()) if self._created else 0

    def update(
        self, values: Dict[str, torch.Tensor], n_states: int
    ) -> Dict[str, torch.Tensor]:
        """Advances the averages with 'values' (in training mode only).

        Returns the smoothed statistics, with gradients flowing to 'values'.
        """
        first_value = next(iter(values.values()))
        self.create_state(n_states, device=first_value.device)
        if self.training:
            with torch.no_grad():
                for name, value in values.items():
                    average = self._average(name)
                    average.copy_(
                        self.decay * average
                        + (1.0 - self.decay) * value.detach().to(average.dtype)
                    )
                self.ema_step.add_(1.0)

        has_history = self.ema_step > 0.0
        smoothed = {}
        for name, estimate in self.estimates().items():
            value = values[name]
            # Before any training batch there is no history: use the batch value.
            estimate = torch.where(has_history, estimate.to(value.dtype), value)
            smoothed[name] = (estimate - value).detach() + value
        return smoothed

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Creates the buffers from the saved shapes when K was not known yet."""
        step_key = prefix + "ema_step"
        if not self._created and step_key in state_dict:
            for name in self.statistic_names:
                saved = state_dict.get(prefix + self._buffer_name(name))
                if saved is not None:
                    self.register_buffer(
                        self._buffer_name(name), torch.zeros_like(saved)
                    )
            self.register_buffer("ema_step", torch.zeros_like(state_dict[step_key]))
            self._created = True
            sized = [
                state_dict[prefix + self._buffer_name(name)].shape[0]
                for name in self.statistic_names
                if core_statistics.shape(name, 1) == (1,)
                and prefix + self._buffer_name(name) in state_dict
            ]
            self.n_states = sized[0] if sized else None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _PenaltyRegularizer(nn.Module):
    """Base for PRESS regularizers: statistic -> estimator -> penalty.

    Subclasses declare `HYPERPARAMETERS` (the scalar settings) and `STATISTICS`
    (the names of the population statistics their penalty consumes), and
    implement `_penalty(stats, n_states)`.

    All statistics are estimated from a single minibatch, and every penalty here
    is a convex function of them. Jensen's inequality then makes each batch
    estimate biased *upward*: `E[f(stat_batch)] >= f(E[stat_batch])`. Running
    more batches per epoch shrinks the variance of the epoch average but not
    this bias -- only a larger batch, or averaging the statistic across batches
    (`ema_decay`), does. See `pypress.keras.regularizers` for measurements.
    """

    HYPERPARAMETERS: tuple = (hp.L2, hp.EMA_DECAY)
    STATISTICS: tuple = ()

    def __init__(self, **kwargs):
        """Coerces the declared hyperparameters found in 'kwargs'.

        Args:
          **kwargs: Values for the declared hyperparameters.

        Raises:
          TypeError: on a keyword that is not a declared hyperparameter.
        """
        super().__init__()
        declared = {param.name: param for param in self.HYPERPARAMETERS}
        unknown = set(kwargs) - set(declared)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected arguments {sorted(unknown)}."
            )
        self._hyperparameters = {
            name: param.coerce(kwargs.get(name, param.default))
            for name, param in declared.items()
        }
        decay = self._hyperparameters.get("ema_decay")
        self._estimator = (
            None
            if decay is None or not self.STATISTICS
            else CrossBatchEstimator(decay, self.STATISTICS)
        )
        self._bound_layer: Optional[weakref.ReferenceType] = None

    def __getstate__(self):
        """Pickles without the weak reference to the bound layer (it cannot be)."""
        state = self.__dict__.copy()
        state["_bound_layer"] = None
        return state

    # -- hyperparameters ---------------------------------------------------

    def _value(self, name: str):
        """A hyperparameter as configured (float, None or ScheduledValue)."""
        return self._hyperparameters.get(name)

    def _scalar(self, name: str) -> float:
        """A hyperparameter read now as a float (a schedule reads the clock)."""
        return schedules.as_float(self._hyperparameters[name])

    @property
    def l2(self):
        """The penalty multiplier, as configured (float or ScheduledValue)."""
        return self._value("l2")

    @property
    def ema_decay(self) -> float | None:
        """The EMA decay, or None when statistics are used un-smoothed."""
        return self._value("ema_decay")

    # -- cross-batch estimator ----------------------------------------------

    def bind(self, n_states: int, layer: Optional[nn.Module] = None):
        """Creates the running averages for 'n_states' predictive states.

        Called by `PredictiveStateSimplex` when the regularizer is attached to
        it, so the buffers exist from construction and `state_dict()` loading is
        plain. A regularizer used from a loss need not call it: the averages are
        then created by the first `penalty` call, or by `load_state_dict`.

        Args:
          n_states: Number of predictive states, K.
          layer: The layer this regularizer is attached to, if any.

        Raises:
          ValueError: if this regularizer already holds state for another layer.
        """
        if self._estimator is None:
            return
        owner = None if self._bound_layer is None else self._bound_layer()
        if layer is not None:
            if owner is not None and owner is not layer:
                raise ValueError(
                    f"{type(self).__name__} with 'ema_decay' is already applied to "
                    f"a {type(owner).__name__}; its smoothed statistics are "
                    f"per-layer state, so this {type(layer).__name__} needs its "
                    "own instance."
                )
            self._bound_layer = weakref.ref(layer)
        self._estimator.create_state(n_states)

    def smoothed_statistics(self) -> Dict[str, torch.Tensor]:
        """Current bias-corrected estimates of the smoothed statistics.

        Empty when `ema_decay` is unset or before the state exists.
        """
        return {} if self._estimator is None else self._estimator.estimates()

    @property
    def smoothing_steps(self) -> int:
        """Number of training batches the smoothed statistics have averaged."""
        return 0 if self._estimator is None else self._estimator.steps

    # -- pipeline -----------------------------------------------------------

    def _penalty(self, stats: Dict[str, torch.Tensor], n_states: int) -> torch.Tensor:
        """The penalty, before scaling by `l2`, as a function of the statistics."""
        raise NotImplementedError

    def penalty(self, x: torch.Tensor) -> torch.Tensor:
        """The penalty this regularizer adds to the training loss for batch 'x'.

        This is `l2` times the penalty function of the statistics of 'x': the
        amount added to the (per-example mean) loss, independent of the batch
        size. With `ema_decay`, calling it advances the running averages in
        training mode, as a training step does.
        """
        n_states = int(x.shape[-1])
        batch = {name: statistics.compute(name, x) for name in self.STATISTICS}
        if self._estimator is None:
            return self._scalar("l2") * self._penalty(batch, n_states)
        smoothed = self._estimator.update(batch, n_states)
        return self._scalar("l2") * self._smoothed_penalty(smoothed, batch, n_states)

    def _smoothed_penalty(
        self,
        smoothed: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        n_states: int,
    ) -> torch.Tensor:
        """The penalty, before scaling by `l2`, with cross-batch smoothing.

        'smoothed' holds the smoothed statistics, with gradients flowing to the
        batch values; 'batch' holds the batch values. By default the penalty is
        `_penalty` of the smoothed statistics. A subclass overrides this when
        differentiating its penalty at the smoothed statistics would point the
        wrong way for the current batch.
        """
        return self._penalty(smoothed, n_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The penalty for batch 'x'; see `penalty`."""
        return self.penalty(x)

    # -- serialization ------------------------------------------------------

    def _extra_config(self) -> dict:
        """Config entries that are not declared hyperparameters."""
        return {}

    def get_config(self) -> Dict[str, Any]:
        """Returns every declared hyperparameter plus any extra options.

        The keys and values equal those of the Keras class of the same name.
        """
        config = {
            param.name: param.serialize(self._hyperparameters[param.name])
            for param in self.HYPERPARAMETERS
        }
        config.update(self._extra_config())
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Recreates the regularizer from `get_config()`."""
        return cls(**config)

    def extra_repr(self) -> str:
        """The configured hyperparameters, for `print(module)`."""
        return ", ".join(f"{key}={value!r}" for key, value in self._config_items())

    def _config_items(self):
        """Hyperparameters (and extras) as configured, for display."""
        items = list(self._hyperparameters.items())
        items.extend(self._extra_config().items())
        return items


def _max_entropy(n_states: int) -> float:
    """log(K), the maximum entropy over K states (guarded to be > 0)."""
    return max(math.log(max(n_states, 1)), 1e-8)


def _log_active_states(fraction: float, n_states: int) -> float:
    """log(fraction * K), clamped at one active state so it is >= 0."""
    return math.log(max(1.0, fraction * float(n_states)))


class TargetEntropy(_PenaltyRegularizer):
    """Penalizes mean row entropy deviating from a target value.

    By default, targets log(0.5 * K), representing a balanced 50% regime overlap
    anchor (where K = x.shape[1] at call time). Uses a squared (L2) penalty to
    provide smooth gradient behavior near equilibrium.

    Target = log(K) is maximum overlap (uniform distribution).
    Target = 0 is minimum overlap (deterministic / single-state assignment).

    The deviation (target - mean entropy) is normalized by log(K), the maximum
    possible entropy for K columns, before squaring, so the penalty is bounded
    in [0, l2] regardless of K. See `pypress.keras.regularizers.TargetEntropy`.
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
    ):
        """Initializes the regularizer.

        Args:
          l2: Penalty multiplier weight for the squared deviation from target.
          target: Target mean entropy value, in nats (>= 0). If None, derived
            from `entropy_fraction`, with K = x.shape[1] computed dynamically at
            call time. May be a `ScheduledValue`.
          entropy_fraction: Target expressed as a fraction of K, i.e. the target
            is `log(entropy_fraction * K)`. Must be in (0, 1] and is mutually
            exclusive with `target`. Defaults to 0.5 (1.0 for `Uniform`).
          ema_decay: If set, evaluate the penalty on a bias-corrected moving
            average of the mean row entropy across training batches.
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
            target = self._scalar("target")
        else:
            target = _log_active_states(
                schedules.as_float(self.entropy_fraction), n_states
            )
        # max_entropy is guarded for K == 1, where deviation is also 0.
        deviation = (target - stats["mean_row_entropy"]) / _max_entropy(n_states)
        return deviation.square()


class Uniform(TargetEntropy):
    """Penalizes weights if they are not uniform across columns (1 / K).

    Special case of TargetEntropy where the target is fixed to the maximum
    possible entropy for uniform weights, log(# of columns), always computed
    dynamically from the input. For a custom target, use TargetEntropy directly.
    """

    HYPERPARAMETERS = (hp.L2, hp.EMA_DECAY)
    _default_entropy_fraction = 1.0

    def __init__(self, l2: float = 0.0, *, ema_decay: float | None = None):
        """Initializes the uniform regularizer.

        Args:
          l2: penalty multiplier weight for the squared deviation from target.
          ema_decay: If set, smooth the mean row entropy across batches.
        """
        _PenaltyRegularizer.__init__(self, l2=l2, ema_decay=ema_decay)


class StateSizeEntropy(_PenaltyRegularizer):
    """Penalizes population-level state usage concentrated on a few states.

    Looks at the state shares

        p_k = state_size(W)_k / sum_l state_size(W)_l,

    and penalizes their deviation from uniform usage:

        penalty = l2 * max(0, log(active_fraction * K) - H(p)) / log(K),

    where H(p) is the Shannon entropy of the state shares. The penalty is bounded
    in [0, l2] regardless of K. See `pypress.keras.regularizers.StateSizeEntropy`
    for the full discussion.
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
    ):
        """Initializes the regularizer.

        Args:
          l2: Penalty multiplier for the normalized KL divergence of the state
            shares from uniform usage. The penalty is bounded in [0, l2].
          active_fraction: Fraction of the K states required to be effectively
            live, i.e. the target is `exp(H(p)) >= active_fraction * K`. Must be
            in (0, 1].
          ema_decay: If set, evaluate the penalty on a bias-corrected moving
            average of state sizes across training batches.
        """
        super().__init__(l2=l2, active_fraction=active_fraction, ema_decay=ema_decay)

    @property
    def active_fraction(self):
        """The required fraction of live states (float or ScheduledValue)."""
        return self._value("active_fraction")

    def _penalty(self, stats, n_states):
        """One-sided, log(K)-normalized shortfall of the state-share entropy."""
        marginal_entropy = statistics.entropy(
            statistics.state_shares(stats["state_sizes"])
        )
        target = _log_active_states(self._scalar("active_fraction"), n_states)
        # One-sided hinge (`max(0, .)`), not an activation: usage *above* the
        # target must cost exactly zero, never a negative penalty.
        return torch.relu(target - marginal_entropy) / _max_entropy(n_states)


class MinStateSize(_PenaltyRegularizer):
    """Penalizes predictive states holding less than `min_share` of the population.

    Given the state shares p_k, the penalty is the mean squared relative
    shortfall below the floor:

        penalty = l2 * mean_k( max(0, 1 - p_k / min_share) ** 2 ).

    It is exactly zero if and only if every state clears `min_share`, and is
    bounded in [0, l2) regardless of K. See
    `pypress.keras.regularizers.MinStateSize` for the full discussion.
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
        """
        super().__init__(l2=l2, min_share=min_share, ema_decay=ema_decay)

    @property
    def min_share(self):
        """The per-state floor, as configured (float, None or ScheduledValue)."""
        return self._value("min_share")

    def _floor(self, n_states: int) -> float:
        """Resolves the floor for K states, defaulting to half the fair share."""
        if self.min_share is None:
            return self._default_fair_share_fraction / n_states
        # Validate the strictest value the schedule can reach, not just today's.
        largest = max(_Hyperparameter.bounds(self.min_share))
        if largest * n_states > 1.0:
            raise ValueError(
                f"'min_share' of {largest} is not satisfiable with K = {n_states} "
                f"states: it would require {largest * n_states:.2f} of the "
                f"population. Use min_share <= {1.0 / n_states:.4f}, or fewer states."
            )
        return self._scalar("min_share")

    def _penalty(self, stats, n_states):
        """Mean squared relative shortfall of state shares below the floor."""
        shares = statistics.state_shares(stats["state_sizes"])
        # One-sided hinge, squared for a gradient that fades to zero at the floor.
        shortfall = torch.relu(1.0 - shares / self._floor(n_states))
        return shortfall.square().mean()


def _trace(stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """trace(H) from the trace statistics in 'stats'."""
    return statistics.degrees_of_freedom(
        stats["trace_numerator"], stats["trace_denominator"]
    )


class DegreesOfFreedom(_PenaltyRegularizer):
    """Penalizes degrees of freedom, trace(H), deviating from a target.

    The smoother matrix implied by the predictive state weights is

        H = W * D^(-1) * W' in R^{N x N},

    where D is a diagonal matrix with D_kk = size of state k = sum_i w_ik, and

        trace(H) = trace(W' * W * D^-1),

    the trace of a K x K matrix. It measures the *effective* number of states:
    for hard assignments it equals the number of non-empty states exactly.

    Both `l2` and `target` accept a `pypress.torch.schedules.ScheduledValue`.
    `target` is always an absolute effective-state count. Set `normalize=True`
    to divide its deviation by K = x.shape[1] before squaring, so an `l2` tuned
    for a given fractional trace error transfers across different `n_states`.
    See `pypress.keras.regularizers.DegreesOfFreedom` for the full discussion.
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
        *,
        normalize: bool = False,
        ema_decay: float | None = None,
    ):
        """Initializes the regularizer.

        Args:
          l2: l2 penalty parameter for l2 * (target - trace(H)) ** 2. May be a
            `ScheduledValue`.
          target: Absolute degrees-of-freedom target. Must be >= 1. May be a
            `ScheduledValue`.
          normalize: Divide the target deviation by K = x.shape[1] before
            squaring. This makes the penalty scale-invariant in K while retaining
            `target` in its natural effective-state-count units.
          ema_decay: If set, average trace(H)'s per-state numerator and
            denominator across training batches before taking their ratio.
        """
        super().__init__(l2=l2, target=target, ema_decay=ema_decay)
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
        smoothed_trace = _trace(smoothed).detach()
        batch_trace = _trace(batch)
        trace = smoothed_trace + (batch_trace - batch_trace.detach())
        return self._deviation_penalty(trace, n_states)

    def _deviation_penalty(self, trace, n_states):
        """Squared deviation of 'trace' from the target, optionally normalized."""
        deviation = trace - self._scalar("target")
        if self._normalize:
            # K, rather than batch size, is the stable architectural scale.
            deviation = deviation / float(max(n_states, 1))
        return deviation.square()

    def _extra_config(self):
        """Adds the `normalize` option."""
        return {"normalize": self._normalize}


class CombinedRegularizer(nn.Module):
    """Sums the penalties of several regularizers.

    Example:
        >>> reg = CombinedRegularizer([
        ...     MinStateSize(l2=5.0, min_share=0.05),
        ...     DegreesOfFreedom(l2=1.0, target=4.0),
        ... ])

    The children are submodules, so their running averages are in this module's
    `state_dict()`. A child without a `penalty` method is called as `child(x)`.
    """

    def __init__(self, regularizers: Sequence[nn.Module] = ()):
        """Initializes the composite.

        Args:
          regularizers: Regularizer instances to sum.
        """
        super().__init__()
        self.regularizers = nn.ModuleList(regularizers)

    def penalty(self, x: torch.Tensor) -> torch.Tensor:
        """The sum of the children's penalties on 'x'."""
        total_penalty = 0.0
        for reg in self.regularizers:
            penalty = getattr(reg, "penalty", None)
            total_penalty = total_penalty + (
                penalty(x) if callable(penalty) else reg(x)
            )
        return total_penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The sum of the children's penalties on 'x'; see `penalty`."""
        return self.penalty(x)

    def bind(self, n_states: int, layer: Optional[nn.Module] = None):
        """Binds each child that keeps per-layer state; see `_PenaltyRegularizer.bind`."""
        for reg in self.regularizers:
            bind = getattr(reg, "bind", None)
            if callable(bind):
                bind(n_states, layer)

    def get_config(self) -> Dict[str, Any]:
        """Serializes each child regularizer, as Keras does."""
        return {"regularizers": [serialize(reg) for reg in self.regularizers]}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Recreates the composite from `get_config()`."""
        config = dict(config)
        return cls(regularizers=[get(item) for item in config.pop("regularizers", [])])


_REGULARIZERS = {
    cls.__name__: cls
    for cls in (
        TargetEntropy,
        Uniform,
        StateSizeEntropy,
        MinStateSize,
        DegreesOfFreedom,
        CombinedRegularizer,
    )
}


def serialize(regularizer: nn.Module) -> Dict[str, Any]:
    """Config of 'regularizer' in the form `tf.keras.regularizers.serialize` writes.

    The result equals the Keras serialization of the Keras class of the same
    name, so it can be read by either backend's `from_config` / `get`.
    """
    name = type(regularizer).__name__
    return {
        "module": _SERIALIZED_MODULE,
        "class_name": name,
        "config": regularizer.get_config(),
        "registered_name": _REGISTERED_PREFIX + name,
    }


def from_config(config: Dict[str, Any]) -> nn.Module:
    """Builds a regularizer from a config naming its class.

    Accepts the form `serialize` writes (and Keras writes), and the flat form: a
    regularizer's `get_config()` plus a `"class_name"` entry, e.g.
    `from_config(keras_reg.get_config() | {"class_name": "Uniform"})`.

    Raises:
      ValueError: if the class name is missing or unknown.
    """
    config = dict(config)
    name = config.pop("registered_name", None) or config.get("class_name")
    config.pop("class_name", None)
    config.pop("module", None)
    if name is None:
        raise ValueError(f"The config does not name a regularizer class: {config!r}.")
    cls = _lookup(name)
    inner = config.pop("config", None)
    return cls.from_config(config if inner is None else inner)


def _lookup(name: str) -> type:
    """The regularizer class called 'name' (with or without the Keras prefix)."""
    bare = name.removeprefix(_REGISTERED_PREFIX)
    if bare not in _REGULARIZERS:
        raise ValueError(
            f"Unknown regularizer '{name}'. Known regularizers: {sorted(_REGULARIZERS)}."
        )
    return _REGULARIZERS[bare]


def get(identifier) -> nn.Module | None:
    """Resolves a regularizer from a name, a config dict, or an instance.

    Args:
      identifier: None, a class name (built with default arguments), a config
        accepted by `from_config`, or an existing regularizer (returned as is).
    """
    if identifier is None or isinstance(identifier, nn.Module):
        return identifier
    if isinstance(identifier, str):
        return _lookup(identifier)()
    if isinstance(identifier, dict):
        return from_config(identifier)
    if callable(identifier):
        return identifier
    raise ValueError(f"Cannot interpret {identifier!r} as a regularizer.")
