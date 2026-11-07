"""PRESS layers, in PyTorch.

A port of `pypress.keras.layers`:

    X --> PredictiveStateSimplex --> (S_1, ..., S_K) --> PredictiveStateMeans --> y

Differences from the Keras layers:

* Sizes are given at construction, so every parameter exists (and can be
  initialized and loaded) before the first call. `n_states` comes first in each
  constructor. `PredictiveStateSimplex` infers its input size lazily when
  `in_features` is None.
* An activity regularizer is a submodule. In training mode the simplex stores its
  penalty after each forward, and `regularization_loss(model)` sums the stored
  penalties for you to add to your loss. See `pypress.torch.regularizers`.
* `from_keras(layer)` copies a built Keras layer, weights included.
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import nn

from ..core import initializers as core_init
from . import activations as pr_act
from . import regularizers as pr_reg
from .activations import ActivationSpec

InitValues = core_init.InitValues

_ZERO_INITIALIZERS = ("zeros", "zero")


def _initialize(tensor: torch.Tensor, initializer: Union[str, Callable]):
    """Initializes 'tensor' in place with a named or callable initializer."""
    with torch.no_grad():
        if callable(initializer):
            initializer(tensor)
        elif initializer in _ZERO_INITIALIZERS:
            nn.init.zeros_(tensor)
        elif initializer in ("ones",):
            nn.init.ones_(tensor)
        elif initializer in ("glorot_uniform", "xavier_uniform"):
            nn.init.xavier_uniform_(tensor)
        elif initializer in ("glorot_normal", "xavier_normal"):
            nn.init.xavier_normal_(tensor)
        else:
            raise ValueError(
                f"Unknown initializer '{initializer}'. Use 'zeros', 'ones', "
                "'glorot_uniform', 'glorot_normal' or a function of the tensor."
            )


class _LazyLinear(nn.LazyLinear):
    """A `LazyLinear` that applies the layer's initializers when it materializes."""

    def __init__(self, out_features, kernel_initializer, bias_initializer, bias=True):
        """Stores the initializers; the weights are created on the first call."""
        super().__init__(out_features, bias=bias)
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    def reset_parameters(self) -> None:
        """Initializes with the layer's initializers instead of torch's default."""
        if not self.has_uninitialized_params() and self.in_features != 0:
            _initialize(self.weight, self._kernel_initializer)
            if self.bias is not None:
                _initialize(self.bias, self._bias_initializer)


class PredictiveStateSimplex(nn.Module):
    """Layer that implements the predictive state simplex for PRESS model.

    Maps input features to predictive state weights, a K-dimensional simplex:

        X --> (S_1, ..., S_K) in Delta^{N x K}

    In Goerg (2017, 2018) this is also known as the epsilon-mapping from
    features to states. A linear map followed by a softmax; with the default zero
    initialization the weights start uniform, so `trace(H) = 1`.

    Example:
        >>> simplex = PredictiveStateSimplex(n_states=5, in_features=10)
        >>> weights = simplex(torch.randn(100, 10))  # (100, 5), rows sum to 1
    """

    def __init__(
        self,
        n_states: int,
        in_features: Optional[int] = None,
        *,
        activity_regularizer: Optional[nn.Module] = None,
        kernel_initializer: Union[str, Callable] = "zeros",
        bias_initializer: Union[str, Callable] = "zeros",
        bias: bool = True,
    ):
        """Initializes the layer.

        Args:
          n_states: Number of predictive states, K. Must be >= 1.
          in_features: Number of input features. If None, it is inferred from the
            first input (`nn.LazyLinear`); parameters then exist after that call.
          activity_regularizer: Optional regularizer from
            `pypress.torch.regularizers`, applied to the predictive state weights.
            It becomes a submodule, so its running averages are in `state_dict()`.
          kernel_initializer: "zeros", "ones", "glorot_uniform", "glorot_normal"
            or a function that initializes a (K, in_features) tensor in place.
          bias_initializer: Same, for the bias vector.
          bias: Whether the linear map has a bias.
        """
        super().__init__()
        if n_states < 1:
            raise ValueError(
                f"Number of predictive states must be >= 1. Got {n_states}."
            )
        self.n_states = int(n_states)
        if in_features is None:
            self.linear = _LazyLinear(
                self.n_states, kernel_initializer, bias_initializer, bias
            )
        else:
            self.linear = nn.Linear(int(in_features), self.n_states, bias=bias)
            _initialize(self.linear.weight, kernel_initializer)
            if bias:
                _initialize(self.linear.bias, bias_initializer)

        self.activity_regularizer = activity_regularizer
        bind = getattr(activity_regularizer, "bind", None)
        if callable(bind):
            bind(self.n_states, layer=self)
        self.regularization_penalty: Optional[torch.Tensor] = None

    def __getstate__(self):
        """Copies and pickles without the stored penalty, which is part of a graph."""
        state = self.__dict__.copy()
        state["regularization_penalty"] = None
        return state

    @property
    def in_features(self) -> Optional[int]:
        """Number of input features; None until a lazy layer has seen an input."""
        return self.linear.in_features or None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predictive state weights of 'x', shape (..., n_states).

        In training mode with an activity regularizer, also stores its penalty in
        `regularization_penalty` (see `regularization_loss`); in eval mode nothing
        is stored.
        """
        weights = torch.softmax(self.linear(x), dim=-1)
        if self.activity_regularizer is not None and self.training:
            self.regularization_penalty = self.activity_regularizer(weights)
        else:
            self.regularization_penalty = None
        return weights

    @classmethod
    def from_keras(cls, layer) -> "PredictiveStateSimplex":
        """Copies a built `pypress.keras.layers.PredictiveStateSimplex`.

        The kernel is transposed from Keras's (in, K) to torch's (K, in). An
        activity regularizer is converted through its config; its running
        averages are not copied.
        """
        kernel, *bias = _keras_weights(layer, 2 if layer.use_bias else 1)
        regularizer = _regularizer_from_keras(layer.activity_regularizer)
        out = cls(
            layer._n_states,
            kernel.shape[0],
            activity_regularizer=regularizer,
            bias=layer.use_bias,
        )
        with torch.no_grad():
            out.linear.weight.copy_(torch.from_numpy(kernel.T.copy()))
            if layer.use_bias:
                out.linear.bias.copy_(torch.from_numpy(bias[0]))
        return out

    def extra_repr(self) -> str:
        """Sizes, for `print(module)`."""
        return f"n_states={self.n_states}, in_features={self.in_features}"


class _StateConditionalTable(nn.Module):
    """Base layer holding a K x P table of state-conditional parameters.

    The table is learned as logits; an activation maps them to the original
    scale. A single activation applies to the whole table (so "softmax" normalizes
    across the columns of each state); a sequence gives one activation per
    column. `init_values` are on the original scale and are inverted through the
    activations; they take the forms described in `pypress.core.initializers`.
    """

    def __init__(
        self,
        n_states: int,
        n_columns: int,
        activations: ActivationSpec,
        init_values: Optional[InitValues],
    ):
        """Initializes the table.

        Raises:
          ValueError: if a sequence of activations does not have one per column,
            an activation name is unknown, or 'init_values' is given with an
            activation that has no known inverse.
        """
        super().__init__()
        if n_states < 1:
            raise ValueError(
                f"Number of predictive states must be >= 1. Got {n_states}."
            )
        if n_columns < 1:
            raise ValueError(f"Number of columns must be >= 1. Got {n_columns}.")
        self.n_states = int(n_states)
        self.n_columns = int(n_columns)
        self._activations = pr_act.validate(activations, self.n_columns)
        # Activation modules (which may hold parameters) follow the layer.
        columns = (
            [self._activations]
            if pr_act.is_single(self._activations)
            else list(self._activations)
        )
        self._activation_modules = nn.ModuleList(
            a for a in columns if isinstance(a, nn.Module)
        )

        if init_values is None:
            logits = torch.empty(self.n_states, self.n_columns)
            nn.init.xavier_uniform_(logits)
        else:
            # Fail at construction rather than at the first call.
            names = pr_act.names(self._activations, self.n_columns)
            logits = torch.tensor(
                core_init.state_conditional_logits(
                    init_values, self.n_states, self.n_columns, names
                ),
                dtype=torch.float32,
            )
        self.logits = nn.Parameter(logits)

    def _table(self) -> torch.Tensor:
        """The activated table, shape (n_states, n_columns)."""
        return pr_act.apply(self.logits, self._activations)

    def _check_states(self, x: torch.Tensor):
        """Raises ValueError if 'x' does not have one column per state."""
        if x.shape[-1] != self.n_states:
            raise ValueError(
                f"Expected predictive state weights with {self.n_states} columns. "
                f"Got shape {tuple(x.shape)}."
            )

    def extra_repr(self) -> str:
        """Sizes, for `print(module)`."""
        return f"n_states={self.n_states}, n_columns={self.n_columns}"


class PredictiveStateMeans(_StateConditionalTable):
    """Layer that computes a predictive mixture of state-conditional means.

    The prediction is a weighted sum of state-conditional means, where the
    weights are the predictive state weights P(S | X) from the
    `PredictiveStateSimplex` layer:

        output = sum_{k=1}^{K} w_k(X) * mu_k

    The layer learns a table of state-conditional means of shape (n_states,
    units); row k is the mean mu_k of state k.

    Input shape:
        (batch_size, n_states): predictive state weights (rows sum to 1).

    Output shape:
        (batch_size, units)

    Example:
        >>> simplex = PredictiveStateSimplex(n_states=5)
        >>> means = PredictiveStateMeans(n_states=5, units=1, activation="linear")
        >>> predictions = means(simplex(torch.randn(100, 10)))  # (100, 1)
    """

    def __init__(
        self,
        n_states: int,
        units: int,
        activation: ActivationSpec = "linear",
        init_values: Optional[InitValues] = None,
    ):
        """Initializes the PredictiveStateMeans layer.

        Args:
            n_states: Number of predictive states, K.
            units: Number of output units. Use units=1 for scalar regression.
            activation: Activation applied to the means: a name (e.g. "linear",
                "sigmoid", "softmax"), a torch function or module, or a sequence
                with one per unit. A single activation applies across units, so
                "softmax" gives a distribution over units per state.
            init_values: Optional initial means on the original scale (after
                activation). If None, the logits start Glorot-uniform. Can be:
                - a scalar: every mean;
                - a 1D array of shape (units,): broadcast across states;
                - a 2D array of shape (units, n_states): state-specific.
        """
        super().__init__(n_states, int(units), activation, init_values)
        self.units = int(units)

    @property
    def state_conditional_means(self) -> torch.Tensor:
        """State-conditional means mu_k, shape (n_states, units), after activation."""
        return self._table()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """The predictive mixture: inputs @ state_conditional_means.

        Args:
            inputs: Predictive state weights, shape (batch_size, n_states).

        Returns:
            Predictions, shape (batch_size, units).
        """
        self._check_states(inputs)
        return inputs @ self.state_conditional_means

    @classmethod
    def from_keras(cls, layer) -> "PredictiveStateMeans":
        """Copies a built `pypress.keras.layers.PredictiveStateMeans`."""
        (logits,) = _keras_weights(layer, 1)
        out = cls(
            layer._n_states,
            layer._units,
            _activations_from_keras(layer._activations),
        )
        with torch.no_grad():
            out.logits.copy_(torch.from_numpy(logits))
        return out


class PredictiveStateParams(_StateConditionalTable):
    """Layer that learns state-conditional parameters independent of input features.

    Each predictive state k has its own parameters theta_k (P of them), learned
    as a table of shape (n_states, n_params_per_state). Unlike
    `PredictiveStateMeans`, which mixes its table with the predictive state
    weights, this layer broadcasts the whole table to every observation, so a
    downstream loss can use each state's parameters (e.g. a mixture likelihood).

    Use cases for distribution parameters:
        - Gaussian: [mu, sigma] with activations=["linear", "softplus"]
        - Gamma: [alpha, beta] with activations="softplus"

    Input shape:
        (batch_size, n_states): predictive state weights. Only the batch shape is
        used.

    Output shape:
        - If flatten_output=True: (batch_size, n_states * n_params_per_state)
          Format: [s0_p0, s1_p0, ..., sK_p0, s0_p1, s1_p1, ..., sK_p1, ...]
          (parameter-first ordering)
        - If flatten_output=False: (batch_size, n_states, n_params_per_state)

    Example:
        >>> layer = PredictiveStateParams(n_states=3, n_params_per_state=2,
        ...                               activations=["linear", "softplus"],
        ...                               init_values=[0.0, 1.0],
        ...                               flatten_output=False)
        >>> layer(torch.randn(10, 3)).shape
        torch.Size([10, 3, 2])
    """

    def __init__(
        self,
        n_states: int,
        n_params_per_state: int,
        activations: ActivationSpec = "linear",
        flatten_output: bool = True,
        init_values: Optional[InitValues] = None,
    ):
        """Initializes the PredictiveStateParams layer.

        Args:
            n_states: Number of predictive states, K.
            n_params_per_state: Number of parameters per state (P). Must be >= 1.
            activations: Activation(s) applied to the parameters: a name, a torch
                function or module for all of them, or a sequence with one per
                parameter.
            flatten_output: If True, output shape is (batch_size, n_states *
                n_params_per_state) with parameter-first ordering. If False,
                (batch_size, n_states, n_params_per_state).
            init_values: Optional initial values on the original scale (after
                activation). If None, the logits start Glorot-uniform. Can be:
                - a scalar: every parameter of every state;
                - a 1D array of shape (n_params_per_state,): broadcast across
                  states, e.g. [0.0, 1.0] for a Gaussian [mean, std];
                - a 2D array of shape (n_params_per_state, n_states):
                  state-specific, e.g. from `initialize_from_y(...,
                  return_params=True)`.
        """
        if int(n_params_per_state) < 1:
            raise ValueError(
                f"'n_params_per_state' must be >= 1. Got {n_params_per_state}."
            )
        super().__init__(n_states, int(n_params_per_state), activations, init_values)
        self.n_params_per_state = int(n_params_per_state)
        self.flatten_output = bool(flatten_output)

    @property
    def state_conditional_params(self) -> torch.Tensor:
        """State-conditional parameters, shape (n_states, n_params_per_state)."""
        return self._table()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Broadcasts the state-conditional parameters across the batch.

        Args:
            inputs: Predictive state weights, shape (batch_size, n_states). Only
                the leading (batch) shape is used.

        Returns:
            - If flatten_output=True: shape (batch_size, n_states *
              n_params_per_state), parameter-first: [s0_p0, s1_p0, ..., s0_p1, ...]
            - If flatten_output=False: shape (batch_size, n_states,
              n_params_per_state)
        """
        self._check_states(inputs)
        lead = tuple(inputs.shape[:-1])
        theta = self.state_conditional_params.expand(*lead, self.n_states, -1)
        if self.flatten_output:
            # (batch, states, params) -> (batch, params, states) -> flat
            theta = theta.transpose(-1, -2).reshape(
                *lead, self.n_states * self.n_params_per_state
            )
        return theta.contiguous()

    @classmethod
    def from_keras(cls, layer) -> "PredictiveStateParams":
        """Copies a built `pypress.keras.layers.PredictiveStateParams`."""
        (logits,) = _keras_weights(layer, 1)
        out = cls(
            layer._n_states,
            layer._n_params_per_state,
            _activations_from_keras(layer._activations),
            flatten_output=layer._flatten_output,
        )
        with torch.no_grad():
            out.logits.copy_(torch.from_numpy(logits))
        return out


class PRESS(nn.Module):
    """Implements mixture distribution of features -> predictive state simplex -> outputs.

    The PRESS layer is essentially a mixture distribution over state-conditional
    predictive distributions as

        P(y | X) = \\sum_{k=1}^{K} P(y | X, s_k) * P(s_k | X)
                 = \\sum_{k=1}^{K} P(y | s_k) * P(s_k | X)

    For tensors this means that

        N x d = (N x K) * (K x d)

    where d is the number of output units.
    """

    def __init__(
        self,
        n_states: int,
        units: int,
        in_features: Optional[int] = None,
        activation: ActivationSpec = "linear",
        *,
        predictive_state_simplex_kwargs: Optional[Dict[str, Any]] = None,
        predictive_state_means_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the PRESS layer.

        Args:
          n_states: Number of predictive states.
          units: Output units.
          in_features: Number of input features; None infers it on the first call.
          activation: Activation function (or name) of the means.
          predictive_state_simplex_kwargs: Keyword arguments passed to
            `PredictiveStateSimplex` (e.g. `activity_regularizer`).
          predictive_state_means_kwargs: Keyword arguments passed to
            `PredictiveStateMeans` (e.g. `init_values`).
        """
        super().__init__()
        self.n_states = int(n_states)
        self.units = int(units)
        self.predictive_state_simplex = PredictiveStateSimplex(
            self.n_states, in_features, **(predictive_state_simplex_kwargs or {})
        )
        self.predictive_state_means = PredictiveStateMeans(
            self.n_states,
            self.units,
            activation,
            **(predictive_state_means_kwargs or {}),
        )

    def predictive_states(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predictive states of input."""
        return self.predictive_state_simplex(inputs)

    @property
    def state_conditional_means(self) -> torch.Tensor:
        """Gets the state-conditional means."""
        return self.predictive_state_means.state_conditional_means

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calls the layer to produce outputs."""
        return self.predictive_state_means(self.predictive_states(inputs))

    @classmethod
    def from_keras(cls, layer) -> "PRESS":
        """Copies a built `pypress.keras.layers.PRESS`, sub-layers included."""
        simplex = PredictiveStateSimplex.from_keras(layer.predictive_state_simplex)
        means = PredictiveStateMeans.from_keras(layer.predictive_state_means)
        out = cls(means.n_states, means.units, simplex.in_features)
        out.predictive_state_simplex = simplex
        out.predictive_state_means = means
        return out


def regularization_loss(module: nn.Module) -> torch.Tensor:
    """Sum of the activity-regularizer penalties stored by 'module''s layers.

    PyTorch has no activity regularizers, so a `PredictiveStateSimplex` with an
    `activity_regularizer` stores its penalty on each training-mode forward. Add
    this to the loss after the forward pass:

        >>> loss = mse(model(x), y) + regularization_loss(model)

    Penalties are only stored in training mode, so after `model.eval()` this is
    zero. A layer's stored penalty is replaced by the next forward.

    Returns:
      A scalar tensor; zero when no layer stored a penalty.
    """
    total = None
    for layer in module.modules():
        penalty = getattr(layer, "regularization_penalty", None)
        if penalty is not None:
            total = penalty if total is None else total + penalty
    if total is None:
        parameter = next(module.parameters(), None)
        device = None if parameter is None else parameter.device
        return torch.zeros((), device=device)
    return total


# -- from_keras helpers ---------------------------------------------------


def _keras_weights(layer, n_weights: int):
    """The numpy weights of a built Keras layer, checking there are 'n_weights'."""
    weights = [w.numpy() for w in layer.weights]
    if len(weights) < n_weights:
        raise ValueError(
            f"The Keras layer '{layer.name}' is not built; call it on an input "
            "(or call build) before converting."
        )
    return [np.asarray(w, dtype=np.float32) for w in weights[:n_weights]]


def _activations_from_keras(spec) -> ActivationSpec:
    """A Keras activation spec as torch activation names.

    Raises:
      ValueError: if an activation has no registered name.
    """
    from ..keras import activations as keras_act

    if keras_act.is_single(spec):
        columns = [spec]
    else:
        columns = list(spec)
    resolved = [keras_act.name(a) for a in columns]
    if any(n is None for n in resolved):
        raise ValueError(
            f"Cannot convert the Keras activation(s) {spec!r}: they have no "
            "registered name. Build the torch layer with a torch activation."
        )
    return resolved[0] if keras_act.is_single(spec) else resolved


def _regularizer_from_keras(regularizer) -> Optional[nn.Module]:
    """A Keras activity regularizer as its torch counterpart (state not copied)."""
    if regularizer is None:
        return None
    import tensorflow as tf

    return pr_reg.from_config(tf.keras.regularizers.serialize(regularizer))
