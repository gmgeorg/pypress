"""Module for PRESS layers."""

from typing import Any, Dict, List, Optional, Union, Sequence

import numpy as np
import tensorflow as tf

from . import initializers as pr_init


ActivationSpec = Union[
    str,
    tf.keras.layers.Layer,
    Sequence[Union[str, tf.keras.layers.Layer]],
]


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateSimplex(tf.keras.layers.Dense):
    """Layer that implements the predictive state simplex for PRESS model.

    Maps input features to predictive state simplex.

        X --> (S_1, ..., S_J) in Delta^{N x J}

    to a J-dimensional simplex.  In Goerg (2017, 2018) this is also known as the
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


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateMeans(tf.keras.layers.Layer):
    """Layer that computes a weighted mixture of state-conditional means.

    This layer implements the core PRESS prediction mechanism by computing a weighted
    sum of state-conditional means, where the weights are the predictive state
    probabilities P(S | X) from the PredictiveStateSimplex layer.

    Mathematical formulation:
        output = sum_{j=1}^{J} w_j(X) * mu_j

    where:
        - w_j(X) = P(S_j | X) are the predictive state probabilities (input)
        - mu_j are the learnable state-conditional means (weights of this layer)
        - J is the number of predictive states

    The layer learns a matrix of state-conditional means of shape (n_states, units),
    where each row j represents the mean output mu_j for state j. The input state
    probabilities are used to compute a weighted mixture via matrix multiplication.

    Input shape:
        (batch_size, n_states): Predictive state probabilities P(S | X) from
        PredictiveStateSimplex layer (softmax outputs summing to 1)

    Output shape:
        (batch_size, units): Weighted mixture of state-conditional means

    Example:
        >>> # Create predictive state simplex and means layers
        >>> simplex = PredictiveStateSimplex(n_states=5)
        >>> means = PredictiveStateMeans(units=1, activation="linear")
        >>>
        >>> # Forward pass
        >>> X = tf.random.normal((100, 10))  # batch of features
        >>> state_probs = simplex(X)  # (100, 5)
        >>> predictions = means(state_probs)  # (100, 1)
    """

    def __init__(
        self,
        units: int,
        activation: Union[str, tf.keras.layers.Activation] = "linear",
        predictive_state_means_init_logits: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Initializes the PredictiveStateMeans layer.

        Args:
            units: Number of output units (dimensionality of predictions). For scalar
                regression, use units=1. For multi-output regression, use units > 1.
            activation: Activation function to apply to state-conditional mean logits.
                Can be a string (e.g., "linear", "relu", "sigmoid") or a
                tf.keras.layers.Activation layer. Common choices:
                - "linear": for regression (default)
                - "sigmoid": for binary classification or bounded outputs
                - "softmax": for multi-class classification (when units > 1)
            predictive_state_means_init_logits: Optional numpy array for custom
                initialization of state-conditional mean logits. If None, uses default
                Keras initialization. Can be:
                - A scalar: all means initialized to this value
                - A 1D array of shape (units,): broadcast across all states
            **kwargs: Additional keyword arguments passed to Layer().
        """
        super().__init__(**kwargs)
        self._units = units
        self._activation = activation
        self._predictive_state_means_init_logits = predictive_state_means_init_logits

        self._activation_layer = (
            tf.keras.layers.Activation(self._activation)
            if isinstance(self._activation, str)
            else activation
        )
        self._state_conditional_means = None

    def build(self, input_shape: List[int]):
        """Build the layer by creating trainable state-conditional mean weights.

        The weight matrix has shape (n_states, units) where each row j contains
        the mean vector mu_j for state j.

        Args:
            input_shape: Shape of input tensor, expected to be (batch_size, n_states).
                The n_states dimension (input_shape[1]) determines the number of
                state-conditional means to learn.
        """
        input_shape = list(input_shape)
        scm_shape = tuple(input_shape[1:] + [self._units])
        if self._predictive_state_means_init_logits is None:
            initializer = None
        else:
            initializer = pr_init.PredictiveStateMeansInitializer(
                self._predictive_state_means_init_logits, n_states=self._units
            )

        self._state_conditional_mean_logits = self.add_weight(
            name="state_conditional_mean_logits",
            shape=scm_shape,
            initializer=initializer,
            trainable=True,
        )

    @property
    def state_conditional_means(self) -> tf.Tensor:
        """Returns state-conditional means after applying activation function.

        The activation function is applied element-wise to the logits to produce
        the final state-conditional means mu_j for j=1..J.

        Returns:
            Tensor of shape (n_states, units) containing the activated state-conditional
            means. Each row j represents the mean output mu_j for predictive state j.
        """
        return self._activation_layer(self._state_conditional_mean_logits)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass: compute weighted mixture of state-conditional means.

        Computes the PRESS prediction as a weighted sum:
            output[i] = sum_{j=1}^{J} inputs[i,j] * mu_j

        This is implemented efficiently via matrix multiplication:
            output = inputs @ state_conditional_means

        where:
            - inputs[i,j] = P(S_j | X_i) are the state probabilities for sample i
            - mu_j are the state-conditional means (rows of state_conditional_means)

        Args:
            inputs: Tensor of shape (batch_size, n_states) representing predictive
                state probabilities P(S | X). Each row should be a probability
                distribution (non-negative, summing to 1).

        Returns:
            Tensor of shape (batch_size, units) containing the predicted outputs
            as a weighted mixture of state-conditional means.
        """
        return tf.matmul(inputs, self.state_conditional_means)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration with keys:
            - units: Number of output units
            - activation: Activation function specification
            - predictive_state_means_init_logits: Initial logits (if provided)
        """
        config = super().get_config()
        config.update(
            {
                "units": self._units,
                "activation": self._activation,
                "predictive_state_means_init_logits": self._predictive_state_means_init_logits,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class PRESS(tf.keras.layers.Layer):
    """Implements mixture distribution of features -> predictive state simplex -> outputs.

    The PRESS layer is essentially a mixture distribution over state-conditional predictive
    distributions as

        P(y | X) = \\sum_{j=1}^{J} P(y | X, s_j) * P(s_j | X)
                 = \\sum_{j=1}^{J} P(y | s_j) * P(s_j | X)


    For tensors this means that

        N x d = (N x J) * (J x d)

    where d is the number of output units.
    """

    def __init__(
        self,
        units: int,
        n_states: int,
        activation: Union[str, tf.keras.layers.Activation] = "linear",
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
        self._activation = activation
        self._predictive_state_simplex_kwargs = predictive_state_simplex_kwargs or {}
        self._predictive_state_means_kwargs = predictive_state_means_kwargs or {}

    def build(self, input_shape):
        """Builds the layer based on input_shape."""
        self._predictive_state_simplex = PredictiveStateSimplex(
            n_states=self._n_states, **self._predictive_state_simplex_kwargs
        )
        self._predictive_state_means = PredictiveStateMeans(
            units=self._units,
            activation=self._activation,
            **self._predictive_state_means_kwargs,
        )

    def predictive_states(self, inputs: tf.Tensor) -> tf.Tensor:
        """Predictive states of input."""
        return self._predictive_state_simplex(inputs)

    @property
    def state_conditional_means(self) -> tf.Tensor:
        """Gets the state-conditional means."""
        return self._prediction_state_means.state_conditional_means

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
                "activation": self._activation,
                "predictive_state_means_kwargs": self._predictive_state_means_kwargs,
                "predictive_state_simplex_kwargs": self._predictive_state_simplex_kwargs,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateParams(tf.keras.layers.Layer):
    """Layer that learns state-conditional parameters independent of input features.

    This layer implements state-conditional parameters theta_j for j=1..J that are
    independent of input features X. Each state has its own set of parameters that
    can be activated with different activation functions.

    The layer learns a weight matrix of shape [J, P] where J is the number of states
    and P is the number of parameters per state. These parameters are broadcasted
    across the batch dimension.

    Relationship to PredictiveStateMeans:
        PredictiveStateMeans is a special case of this layer that:
        1. Only models the mean (P=1 parameter per state for scalar outputs)
        2. Does NOT broadcast parameters across batch - instead it computes a weighted
           mixture: output = sum_j w_j * mu_j (i.e., it multiplies by state probabilities)

        In contrast, PredictiveStateParams broadcasts the same parameter values to all
        batch samples and can model multiple parameters per state. This is useful for:
        - Distribution parameters (e.g., [mean, variance] or [location, scale, shape])
        - Multiple output dimensions that need state-conditional parameters
        - Cases where you want to access raw state parameters without mixing

    Use cases for distribution parameters:
        For probabilistic models, you can use this layer to learn state-conditional
        distribution parameters. For example:
        - Gaussian: [mu, sigma] with activations=["linear", "softplus"]
        - Gamma: [alpha, beta] with activations=["softplus", "softplus"]
        - Normal with fixed variance: [mu] with activations="linear"

    Input shape:
        (batch_size, n_states): Predictive state probabilities P(S | X)

    Output shape:
        - If flatten_output=True: (batch_size, n_states * n_params_per_state)
          Format: [s0_p0, s1_p0, ..., sJ_p0, s0_p1, s1_p1, ..., sJ_p1, ...]
          (parameter-first ordering)
        - If flatten_output=False: (batch_size, n_states, n_params_per_state)

    Example:
        >>> # Single activation for all parameters
        >>> layer = PredictiveStateParams(n_params_per_state=2,
        ...                                activations="softplus")
        >>> inputs = tf.random.normal((10, 3))  # batch of state probabilities (3 states)
        >>> outputs = layer(inputs)
        >>> outputs.shape
        TensorShape([10, 6])  # flattened: 3 states * 2 params

        >>> # Different activations per parameter
        >>> layer = PredictiveStateParams(n_params_per_state=2,
        ...                                activations=["linear", "softplus"],
        ...                                flatten_output=False)
        >>> outputs = layer(inputs)
        >>> outputs.shape
        TensorShape([10, 3, 2])  # not flattened (3 states, 2 params)
    """

    def __init__(
        self,
        n_params_per_state: int,
        activations: ActivationSpec = "linear",
        flatten_output: bool = True,
        init_logits: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Initializes the PredictiveStateParams layer.

        Args:
            n_params_per_state: Number of parameters per state (P). Must be >= 1.
            activations: Activation function(s) to apply to parameters. Can be:
                - A string (e.g., "linear", "relu", "softplus"): same activation for all parameters
                - A tf.keras.layers.Layer: custom activation layer for all parameters
                - A sequence of strings/layers: one activation per parameter
                  (length must equal n_params_per_state)
            flatten_output: If True, output shape is (batch_size, n_states * n_params_per_state)
                with parameter-first ordering. If False, output shape is
                (batch_size, n_states, n_params_per_state).
            init_logits: Optional numpy array of shape (n_states, n_params_per_state)
                for custom initialization of parameter logits. If None, uses
                "glorot_uniform" initialization. If provided, the first dimension
                (n_states) must match the input shape.
            **kwargs: Additional keyword arguments passed to Layer().

        Note:
            The number of states (n_states) is automatically inferred from the input
            shape during the build() call, similar to PredictiveStateMeans layer.
        """
        super().__init__(**kwargs)
        self._n_states = None  # Will be inferred from input_shape in build()
        self._n_params_per_state = int(n_params_per_state)
        self._activations = activations
        self._flatten_output = bool(flatten_output)
        self._init_logits = init_logits

        # Build activation layers now (no weights needed)
        self._activation_layers = self._make_activation_layers(
            activations=self._activations, n_params=self._n_params_per_state
        )

        self._theta_logits = None  # [K, P]

    @staticmethod
    def _make_activation_layers(
        activations: ActivationSpec, n_params: int
    ) -> list[tf.keras.layers.Layer]:
        """Create activation layers for each parameter.

        Args:
            activations: Activation specification (string, layer, or sequence of these).
            n_params: Number of parameters (required length if activations is a sequence).

        Returns:
            List of tf.keras.layers.Layer objects, one per parameter.

        Raises:
            ValueError: If activations is a sequence with length != n_params.
        """

        def to_layer(a: Union[str, tf.keras.layers.Layer]) -> tf.keras.layers.Layer:
            return tf.keras.layers.Activation(a) if isinstance(a, str) else a

        if isinstance(activations, (str, tf.keras.layers.Layer)):
            layer = to_layer(activations)
            return [layer for _ in range(n_params)]

        # Sequence case
        acts = list(activations)
        if len(acts) != n_params:
            raise ValueError(
                f"If activation is a list/sequence, it must have length n_params_per_state={n_params}, "
                f"got len(activation)={len(acts)}."
            )
        return [to_layer(a) for a in acts]

    def build(self, input_shape):
        """Build the layer by creating trainable weights.

        Infers n_states from input_shape[1] and creates the state-conditional
        parameter logits with shape (n_states, n_params_per_state).

        Args:
            input_shape: Shape of input tensor, expected to be (batch_size, n_states).
                The n_states dimension (input_shape[1]) determines the number of
                state-conditional parameters to learn.
        """
        input_shape = list(input_shape)
        self._n_states = input_shape[1]  # Infer n_states from input

        if self._init_logits is not None:
            # Validate init_logits shape if provided
            if self._init_logits.shape[0] != self._n_states:
                raise ValueError(
                    f"init_logits first dimension {self._init_logits.shape[0]} "
                    f"does not match inferred n_states {self._n_states} from input_shape."
                )
            init = tf.constant_initializer(self._init_logits)
        else:
            init = "glorot_uniform"

        self._theta_logits = self.add_weight(
            name="state_conditional_params_logits",
            shape=(self._n_states, self._n_params_per_state),
            initializer=init,
            trainable=True,
        )
        super().build(input_shape)

    @property
    def state_conditional_params(self) -> tf.Tensor:
        """Returns state-conditional parameters after applying activations.

        Each parameter can have its own activation function. The logits are split
        by parameter, activated individually, then concatenated back together.

        Returns:
            Tensor of shape (n_states, n_params_per_state) containing the activated
            state-conditional parameters theta_j for each state j and parameter p.
        """
        logits = self._theta_logits  # [K,P]
        cols = tf.split(
            logits, num_or_size_splits=self._n_params_per_state, axis=1
        )  # list of [K,1]
        activated = [
            act(c) for act, c in zip(self._activation_layers, cols)
        ]  # list of [K,1]
        return tf.concat(activated, axis=1)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of input tensor (batch_size, n_states).

        Returns:
            Output shape tuple:
            - If flatten_output=True: (batch_size, n_states * n_params_per_state)
            - If flatten_output=False: (batch_size, n_states, n_params_per_state)
        """
        # input_shape is (batch_size, n_states)
        if self._flatten_output:
            return (input_shape[0], self._n_states * self._n_params_per_state)
        return (input_shape[0], self._n_states, self._n_params_per_state)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass: broadcast state-conditional parameters across batch.

        The state-conditional parameters are independent of the input features X,
        so the same parameter values are returned for each batch sample. The inputs
        are only used to determine the batch size for broadcasting.

        Args:
            inputs: Tensor of shape (batch_size, n_states) representing state
                probabilities P(S | X). Only used for batch size; values are not used.

        Returns:
            Tensor of state-conditional parameters:
            - If flatten_output=True: shape (batch_size, n_states * n_params_per_state)
              with parameter-first ordering: [s0_p0, s1_p0, ..., sJ_p0, s0_p1, ...]
            - If flatten_output=False: shape (batch_size, n_states, n_params_per_state)
        """
        # Get theta: [K, P]
        theta = self.state_conditional_params

        # Broadcast to [N, K, P] without explicitly fetching batch_size
        # inputs * 0 is a trick to get a tensor of [N, K] with zeros
        # then we expand dims and add theta
        theta = tf.expand_dims(theta, axis=0)  # [1, K, P]

        # This effectively broadcasts theta across the batch dimension of inputs
        theta = tf.zeros_like(tf.expand_dims(inputs, axis=-1)) + theta

        if self._flatten_output:
            # Flatten with parameter-first ordering: transpose (batch, states, params)
            # to (batch, params, states), then reshape to (batch, states * params)
            # This gives [s0_p0, s1_p0, ..., sJ_p0, s0_p1, s1_p1, ..., sJ_p1, ...]
            theta = tf.transpose(theta, [0, 2, 1])
            theta = tf.reshape(theta, [-1, self._n_states * self._n_params_per_state])

        return theta

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration with keys:
            - n_params_per_state: Number of parameters per state
            - activations: Activation specification
            - flatten_output: Whether output is flattened
            - init_logits: Initial logits (if provided)

        Note:
            n_states is not included in config as it's inferred from input shape.
        """
        config = super().get_config()
        config.update(
            dict(
                n_params_per_state=self._n_params_per_state,
                activations=self._activations,
                flatten_output=self._flatten_output,
                init_logits=self._init_logits,
            )
        )
        return config
