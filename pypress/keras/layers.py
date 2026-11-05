"""Module for PRESS layers."""

from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf

from . import activations as pr_act
from . import initializers as pr_init
from .activations import ActivationSpec


def _init_values_from_config(
    init_values: Optional[Union[float, list]],
) -> Optional[Union[float, np.ndarray]]:
    """Restores array-valued `init_values` from their JSON list form."""
    return (
        np.array(init_values) if isinstance(init_values, (list, tuple)) else init_values
    )


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateSimplex(tf.keras.layers.Dense):
    """Layer that implements the predictive state simplex for PRESS model.

    Maps input features to predictive state simplex.

        X --> (S_1, ..., S_K) in Delta^{N x K}

    to a K-dimensional simplex.  In Goerg (2017, 2018) this is also known as the
    epsilon-mapping from features to states.
    """

    def __init__(
        self,
        n_states: int,
        kernel_initializer: str = "zeros",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        """Initializes the class.

        Args:
          n_states: number of predictive states.
          **kwargs: keyword arguments passed to Dense().
        """
        assert n_states >= 1, (
            f"Number of predictive states must be >= 1. Got {n_states}."
        )
        self._n_states = n_states
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        super().__init__(
            units=self._n_states,
            activation="softmax",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            **kwargs,
        )

    def build(self, input_shape):
        """Builds the layer and any per-layer state of its activity regularizer.

        A regularizer with cross-batch smoothing (`ema_decay`) keeps running
        averages for this layer. They are created here and tracked as sub-layers,
        so they appear in the layer's variables and are saved with the model.
        """
        bind = getattr(self.activity_regularizer, "bind", None)
        self.regularizer_state = bind(self, self._n_states) if callable(bind) else []
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer for serialization.

        Overrides `Dense.get_config()`, which emits `units`/`activation` --
        `__init__` doesn't accept those; it hardcodes `n_states`/"softmax" instead.
        """
        config = super().get_config()
        config.pop("units", None)
        config.pop("activation", None)
        config["n_states"] = self._n_states
        return config


class _StateConditionalTable(tf.keras.layers.Layer):
    """Base layer holding a K x P table of state-conditional parameters.

    The table is learned as logits; an activation maps them to the original
    scale. A single activation applies to the whole table (so "softmax" normalizes
    across the columns of each state); a sequence gives one activation per
    column. `init_values` are on the original scale and are inverted through the
    activations; they take the forms described in `pypress.keras.initializers`.

    Subclasses decide what to do with the table: `PredictiveStateMeans` mixes it
    with the predictive state weights, `PredictiveStateParams` broadcasts it.
    The number of states K is inferred from the input at build time.
    """

    # Weight name and the config key of the activations, set by subclasses.
    _WEIGHT_NAME = "state_conditional_logits"
    _ACTIVATION_KEY = "activations"

    def __init__(
        self,
        n_columns: int,
        activations: ActivationSpec,
        init_values: Optional[pr_init.InitValues],
        **kwargs,
    ):
        """Initializes the table.

        Raises:
          ValueError: if a sequence of activations does not have one per column,
            or 'init_values' is given with an activation that has no known
            inverse.
        """
        super().__init__(**kwargs)
        self._n_columns = int(n_columns)
        self._activations = pr_act.deserialize(activations)
        self._init_values = _init_values_from_config(init_values)
        self._n_states = None
        self._logits = None

        self._per_column = not pr_act.is_single(self._activations)
        if self._per_column:
            columns = list(self._activations)
            if len(columns) != self._n_columns:
                raise ValueError(
                    f"A sequence of activations needs one per column "
                    f"({self._n_columns}). Got {len(columns)}."
                )
            self._activation_layers = [pr_act.to_layer(a) for a in columns]
        else:
            self._activation_layers = [pr_act.to_layer(self._activations)]

        if self._init_values is not None:
            # Fail at construction rather than at the first call.
            pr_act.names(self._activations, self._n_columns)

    def build(self, input_shape):
        """Creates the (n_states, n_columns) logits; n_states is the last input axis."""
        self._n_states = int(input_shape[-1])
        if self._init_values is None:
            initializer = "glorot_uniform"
        else:
            initializer = pr_init.StateConditionalInitializer(
                self._init_values, pr_act.names(self._activations, self._n_columns)
            )
        self._logits = self.add_weight(
            name=self._WEIGHT_NAME,
            shape=(self._n_states, self._n_columns),
            initializer=initializer,
            trainable=True,
        )
        super().build(input_shape)

    def _table(self) -> tf.Tensor:
        """The activated table, shape (n_states, n_columns)."""
        if not self._per_column:
            return self._activation_layers[0](self._logits)
        columns = tf.split(self._logits, num_or_size_splits=self._n_columns, axis=1)
        return tf.concat(
            [act(c) for act, c in zip(self._activation_layers, columns)], axis=1
        )

    def get_config(self) -> Dict[str, Any]:
        """Returns the config, with activations and init_values in JSON-safe form."""
        config = super().get_config()
        config.update(
            {
                self._ACTIVATION_KEY: pr_act.serialize(self._activations),
                "init_values": (
                    self._init_values.tolist()
                    if isinstance(self._init_values, np.ndarray)
                    else self._init_values
                ),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateMeans(_StateConditionalTable):
    """Layer that computes a predictive mixture of state-conditional means.

    The prediction is a weighted sum of state-conditional means, where the
    weights are the predictive state weights P(S | X) from the
    PredictiveStateSimplex layer:

        output = sum_{k=1}^{K} w_k(X) * mu_k

    The layer learns a table of state-conditional means of shape (n_states,
    units); row k is the mean mu_k of state k. The number of states is inferred
    from the input.

    Input shape:
        (batch_size, n_states): predictive state weights (rows sum to 1).

    Output shape:
        (batch_size, units)

    Example:
        >>> simplex = PredictiveStateSimplex(n_states=5)
        >>> means = PredictiveStateMeans(units=1, activation="linear")
        >>> predictions = means(simplex(tf.random.normal((100, 10))))  # (100, 1)
    """

    _WEIGHT_NAME = "state_conditional_mean_logits"
    _ACTIVATION_KEY = "activation"

    def __init__(
        self,
        units: int,
        activation: ActivationSpec = "linear",
        init_values: Optional[pr_init.InitValues] = None,
        **kwargs,
    ):
        """Initializes the PredictiveStateMeans layer.

        Args:
            units: Number of output units. Use units=1 for scalar regression.
            activation: Activation applied to the means: a name (e.g. "linear",
                "sigmoid", "softmax"), a Keras activation function or layer, or a
                sequence with one per unit. A single activation applies across
                units, so "softmax" gives a distribution over units per state.
            init_values: Optional initial means on the original scale (after
                activation). If None, uses Keras's default initialization. Can be:
                - a scalar: every mean;
                - a 1D array of shape (units,): broadcast across states;
                - a 2D array of shape (units, n_states): state-specific.
                Examples:
                - For regression: init_values=np.mean(y_train, axis=0)
                - For classification with sigmoid: init_values=0.8
            **kwargs: Additional keyword arguments passed to Layer().
        """
        self._units = int(units)
        super().__init__(
            n_columns=self._units,
            activations=activation,
            init_values=init_values,
            **kwargs,
        )

    @property
    def state_conditional_means(self) -> tf.Tensor:
        """State-conditional means mu_k, shape (n_states, units), after activation."""
        return self._table()

    def compute_output_shape(self, input_shape):
        """Computes the output shape, (batch_size, units)."""
        return tuple(input_shape[:-1]) + (self._units,)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """The predictive mixture: inputs @ state_conditional_means.

        Args:
            inputs: Predictive state weights, shape (batch_size, n_states).

        Returns:
            Predictions, shape (batch_size, units).
        """
        return tf.matmul(inputs, self.state_conditional_means)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer for serialization."""
        config = super().get_config()
        config["units"] = self._units
        return config


def _serialize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Serializes sub-layer kwargs that may hold Keras objects.

    Values such as a regularizer or initializer are live objects and are not
    JSON-serializable, so a config carrying them raw cannot be saved.
    """
    return {
        key: tf.keras.utils.serialize_keras_object(value)
        for key, value in kwargs.items()
    }


def _deserialize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of `_serialize_kwargs`."""
    return {
        key: tf.keras.utils.deserialize_keras_object(value)
        for key, value in kwargs.items()
    }


@tf.keras.utils.register_keras_serializable(package="pypress")
class PRESS(tf.keras.layers.Layer):
    """Implements mixture distribution of features -> predictive state simplex -> outputs.

    The PRESS layer is essentially a mixture distribution over state-conditional predictive
    distributions as

        P(y | X) = \\sum_{k=1}^{K} P(y | X, s_k) * P(s_k | X)
                 = \\sum_{k=1}^{K} P(y | s_k) * P(s_k | X)


    For tensors this means that

        N x d = (N x K) * (K x d)

    where d is the number of output units.
    """

    def __init__(
        self,
        units: int,
        n_states: int,
        activation: ActivationSpec = "linear",
        predictive_state_simplex_kwargs: Optional[Dict[str, Any]] = None,
        predictive_state_means_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initializes the PRESS layer.

        Args:
          units: output units.
          n_states: number of predictive states.
          activation: activation function (or string).
          predictive_state_simplex_kwargs: keyword arguments passed to PredictiveStateSimplex()
            layer.
          predictive_state_means_kwargs: keyword arguments passed to PredictiveStates
          **kwargs: keyword arguments passed to Layer().
        """
        super().__init__(**kwargs)
        self._units = units
        self._n_states = n_states
        self._activation = pr_act.deserialize(activation)
        self._predictive_state_simplex_kwargs = predictive_state_simplex_kwargs or {}
        self._predictive_state_means_kwargs = predictive_state_means_kwargs or {}

        # Sub-layers are constructed here rather than in build(). Keras then tracks
        # them from construction, so they appear in `.layers`, and their weights,
        # losses and regularizers are reachable before the first call -- which
        # anything inspecting the model (a scheduler callback, `summary()`) needs.
        # It also makes build() idempotent: creating them there would silently
        # replace trained sub-layers if the layer were ever rebuilt.
        self._predictive_state_simplex = PredictiveStateSimplex(
            n_states=self._n_states, **self._predictive_state_simplex_kwargs
        )
        self._predictive_state_means = PredictiveStateMeans(
            units=self._units,
            activation=self._activation,
            **self._predictive_state_means_kwargs,
        )

    @property
    def predictive_state_simplex(self) -> "PredictiveStateSimplex":
        """The sub-layer mapping features to predictive state probabilities."""
        return self._predictive_state_simplex

    @property
    def predictive_state_means(self) -> "PredictiveStateMeans":
        """The sub-layer holding the state-conditional means."""
        return self._predictive_state_means

    def build(self, input_shape):
        """Builds both sub-layers so their weights exist before the first call.

        Building an already-built sub-layer is skipped rather than repeated: Keras
        locks a built layer against new state, and re-running the build would in
        any case discard weights that have already been trained.
        """
        if not self._predictive_state_simplex.built:
            self._predictive_state_simplex.build(input_shape)
        if not self._predictive_state_means.built:
            states_shape = tuple(input_shape[:-1]) + (self._n_states,)
            self._predictive_state_means.build(states_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Computes the output shape, (..., units)."""
        return tuple(input_shape[:-1]) + (self._units,)

    def predictive_states(self, inputs: tf.Tensor) -> tf.Tensor:
        """Predictive states of input."""
        return self._predictive_state_simplex(inputs)

    @property
    def state_conditional_means(self) -> tf.Tensor:
        """Gets the state-conditional means."""
        return self._predictive_state_means.state_conditional_means

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the layer to produce outputs."""
        pred_states = self.predictive_states(inputs)
        return self._predictive_state_means(pred_states)

    def get_config(self) -> Dict[str, Any]:
        """Returns configuration."""
        config = super().get_config()
        config.update(
            {
                "units": self._units,
                "n_states": self._n_states,
                "activation": pr_act.serialize(self._activation),
                "predictive_state_means_kwargs": _serialize_kwargs(
                    self._predictive_state_means_kwargs
                ),
                "predictive_state_simplex_kwargs": _serialize_kwargs(
                    self._predictive_state_simplex_kwargs
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PRESS":
        """Recreates the layer, rebuilding nested Keras objects in the kwargs."""
        config = dict(config)
        for key in ("predictive_state_means_kwargs", "predictive_state_simplex_kwargs"):
            if config.get(key):
                config[key] = _deserialize_kwargs(config[key])
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="pypress")
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
        (batch_size, n_states): predictive state weights. Only the shape is used.

    Output shape:
        - If flatten_output=True: (batch_size, n_states * n_params_per_state)
          Format: [s0_p0, s1_p0, ..., sK_p0, s0_p1, s1_p1, ..., sK_p1, ...]
          (parameter-first ordering)
        - If flatten_output=False: (batch_size, n_states, n_params_per_state)

    Example:
        >>> layer = PredictiveStateParams(n_params_per_state=2,
        ...                                activations=["linear", "softplus"],
        ...                                init_values=[0.0, 1.0],
        ...                                flatten_output=False)
        >>> layer(tf.random.normal((10, 3))).shape
        TensorShape([10, 3, 2])
    """

    _WEIGHT_NAME = "state_conditional_params_logits"
    _ACTIVATION_KEY = "activations"

    def __init__(
        self,
        n_params_per_state: int,
        activations: ActivationSpec = "linear",
        flatten_output: bool = True,
        init_values: Optional[pr_init.InitValues] = None,
        **kwargs,
    ):
        """Initializes the PredictiveStateParams layer.

        Args:
            n_params_per_state: Number of parameters per state (P). Must be >= 1.
            activations: Activation(s) applied to the parameters: a name, a Keras
                activation function or layer for all of them, or a sequence with
                one per parameter.
            flatten_output: If True, output shape is (batch_size, n_states *
                n_params_per_state) with parameter-first ordering. If False,
                (batch_size, n_states, n_params_per_state).
            init_values: Optional initial values on the original scale (after
                activation). If None, uses "glorot_uniform". Can be:
                - a scalar: every parameter of every state;
                - a 1D array of shape (n_params_per_state,): broadcast across
                  states, e.g. [0.0, 1.0] for a Gaussian [mean, std];
                - a 2D array of shape (n_params_per_state, n_states):
                  state-specific, e.g. from `initialize_from_y(...,
                  return_params=True)`.
            **kwargs: Additional keyword arguments passed to Layer().
        """
        if int(n_params_per_state) < 1:
            raise ValueError(
                f"'n_params_per_state' must be >= 1. Got {n_params_per_state}."
            )
        self._n_params_per_state = int(n_params_per_state)
        self._flatten_output = bool(flatten_output)
        super().__init__(
            n_columns=self._n_params_per_state,
            activations=activations,
            init_values=init_values,
            **kwargs,
        )

    @property
    def state_conditional_params(self) -> tf.Tensor:
        """State-conditional parameters, shape (n_states, n_params_per_state)."""
        return self._table()

    def compute_output_shape(self, input_shape):
        """Computes the output shape; n_states is read from the input shape."""
        n_states = input_shape[-1]
        if self._flatten_output:
            flat = None if n_states is None else n_states * self._n_params_per_state
            return (input_shape[0], flat)
        return (input_shape[0], n_states, self._n_params_per_state)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Broadcasts the state-conditional parameters across the batch.

        Args:
            inputs: Predictive state weights, shape (batch_size, n_states). Only
                the batch size is used.

        Returns:
            - If flatten_output=True: shape (batch_size, n_states *
              n_params_per_state), parameter-first: [s0_p0, s1_p0, ..., s0_p1, ...]
            - If flatten_output=False: shape (batch_size, n_states,
              n_params_per_state)
        """
        theta = tf.expand_dims(self.state_conditional_params, axis=0)  # [1, K, P]
        # Adding zeros shaped like the input broadcasts theta to [N, K, P].
        theta = tf.zeros_like(tf.expand_dims(inputs, axis=-1)) + theta

        if self._flatten_output:
            # (batch, states, params) -> (batch, params, states) -> flat
            theta = tf.transpose(theta, [0, 2, 1])
            theta = tf.reshape(theta, [-1, self._n_states * self._n_params_per_state])

        return theta

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer for serialization.

        Note:
            n_states is not included in config as it's inferred from input shape.
        """
        config = super().get_config()
        config.update(
            n_params_per_state=self._n_params_per_state,
            flatten_output=self._flatten_output,
        )
        return config
