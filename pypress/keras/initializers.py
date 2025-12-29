"""Initializers for PRESS layers."""

from typing import Union, Optional, Sequence

import tensorflow as tf
import numpy as np

from .activations import get_inverse_activation


def _get_predictive_state_means_init(
    init_values: Union[float, np.ndarray],
    n_states: int,
    units: int,
    activation: str = "linear",
) -> tf.Tensor:
    """Gets initial weights for predictive state means.

    Converts init_values from original scale to logit scale using the inverse
    activation function.

    Args:
        init_values: Initial values on original scale (after activation).
        n_states: Number of predictive states.
        units: Number of output units.
        activation: Activation function name (e.g., "linear", "sigmoid").

    Returns:
        TensorFlow variable with initial logits (before activation) of shape
        (n_states, units).

    Raises:
        ValueError: If init_values has wrong shape or type, or if activation
            is not supported.
    """
    # Get inverse activation function
    inverse_fn = get_inverse_activation(activation)

    # Build the initialized value on original scale with shape (n_states, units)
    ones = np.ones((n_states, units))
    val = None
    if isinstance(init_values, (float, int)):
        val = ones * init_values
    elif isinstance(init_values, np.ndarray):
        assert init_values.shape[0] == units, (
            f"init_values shape[0]={init_values.shape[0]} must match units={units}"
        )
        if len(init_values.shape) == 1:
            # Broadcast (units,) to (n_states, units)
            val = ones * init_values[np.newaxis, :]
        else:
            raise ValueError("Initial values must be a 1D array (row vector).")
    else:
        raise ValueError(
            f"'init_values' must be float or np.ndarray. Got {type(init_values)}."
        )

    if val is None:
        raise RuntimeError(f"Failed to create initialization. Got {init_values}")

    # Convert from original scale to logit scale using inverse activation
    val_logits = inverse_fn(tf.constant(val, dtype=tf.float32))

    return tf.Variable(val_logits)


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateMeansInitializer(tf.keras.initializers.Initializer):
    """Initializer for weights of the PredictiveStateMeans() layer.

    This initializer takes mean values on the original scale (after activation)
    and converts them to logits (before activation) using the inverse activation
    function. This allows users to provide initialization values in the natural
    scale of their data.

    Example:
        >>> # For regression with linear activation, initialize to empirical mean
        >>> y_mean = np.mean(y_train, axis=0)  # e.g., [0.5]
        >>> initializer = PredictiveStateMeansInitializer(
        ...     init_values=y_mean,
        ...     n_states=5,
        ...     activation="linear"
        ... )
        >>>
        >>> # For classification with sigmoid, initialize to 0.8 probability
        >>> initializer = PredictiveStateMeansInitializer(
        ...     init_values=0.8,
        ...     n_states=3,
        ...     activation="sigmoid"
        ... )
    """

    def __init__(
        self,
        init_values: Union[float, np.ndarray],
        n_states: int,
        activation: str = "linear",
        units: Optional[int] = None,
        **kwargs,
    ):
        """Initializes class.

        Args:
            init_values: Initial values on original scale (after activation).
                Can be a float (broadcasted to all units and states) or a 1D
                numpy array of shape (units,) broadcasted across states.
            n_states: Number of predictive states.
            activation: Activation function name (e.g., "linear", "sigmoid",
                "softplus"). Must be a supported activation with a known inverse.
            units: Output units; required if init_values is float; optional if
                init_values is an array (inferred from array shape).
        """
        super().__init__(**kwargs)
        self._init_values = init_values
        self._n_states = n_states
        self._activation = activation
        self._units = units
        if self._units is None:
            if isinstance(init_values, (float, int)):
                self._units = 1
            elif isinstance(init_values, np.ndarray):
                self._units = init_values.shape[0]

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
                supported. If not specified, `tf.keras.backend.floatx()` is used,
                which default to `float32` unless you configured it otherwise
                (via `tf.keras.backend.set_floatx(float_dtype)`).
            **kwargs: Additional keyword arguments.

        Returns:
            Initialized tensor with logits (before activation) computed from
            init_values using the inverse activation function.
        """
        return _get_predictive_state_means_init(
            self._init_values,
            n_states=self._n_states,
            units=shape[-1],
            activation=self._activation,
        )


def _get_predictive_state_params_init(
    init_values: Sequence[Union[float, np.ndarray]],
    n_states: int,
    n_params_per_state: int,
    activations: Sequence[str],
) -> tf.Tensor:
    """Gets initial weights for predictive state parameters.

    Converts init_values from original scale to logit scale using inverse
    activation functions for each parameter.

    Args:
        init_values: Sequence of initial values (one per parameter) on original scale.
            Each element can be a scalar or 1D array. Must have length n_params_per_state.
        n_states: Number of predictive states.
        n_params_per_state: Number of parameters per state.
        activations: Sequence of activation function names (one per parameter).

    Returns:
        TensorFlow variable with initial logits of shape (n_states, n_params_per_state).

    Raises:
        ValueError: If init_values length doesn't match n_params_per_state or if
            activations are not supported.
    """
    if len(init_values) != n_params_per_state:
        raise ValueError(
            f"init_values must have length {n_params_per_state}, got {len(init_values)}"
        )

    if len(activations) != n_params_per_state:
        raise ValueError(
            f"activations must have length {n_params_per_state}, got {len(activations)}"
        )

    # Build initialized values for each parameter
    param_logits = []
    for i, (init_val, activation) in enumerate(zip(init_values, activations)):
        # Get inverse activation for this parameter
        inverse_fn = get_inverse_activation(activation)

        # Build value on original scale (n_states,)
        if isinstance(init_val, (float, int)):
            val = np.ones(n_states) * init_val
        elif isinstance(init_val, np.ndarray):
            if len(init_val.shape) == 0:  # scalar array
                val = np.ones(n_states) * float(init_val)
            elif len(init_val.shape) == 1 and init_val.shape[0] == 1:
                # Single element array
                val = np.ones(n_states) * init_val[0]
            else:
                raise ValueError(
                    f"init_values[{i}] must be scalar or single-element array, "
                    f"got shape {init_val.shape}"
                )
        else:
            raise ValueError(
                f"init_values[{i}] must be float or np.ndarray, got {type(init_val)}"
            )

        # Convert to logits
        val_logits = inverse_fn(tf.constant(val, dtype=tf.float32))
        param_logits.append(val_logits[:, tf.newaxis])  # Make (n_states, 1)

    # Concatenate along parameter axis: (n_states, n_params_per_state)
    result = tf.concat(param_logits, axis=1)
    return tf.Variable(result)


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateParamsInitializer(tf.keras.initializers.Initializer):
    """Initializer for weights of the PredictiveStateParams() layer.

    This initializer takes parameter values on the original scale (after activation)
    and converts them to logits (before activation) using inverse activation functions.
    Each parameter can have its own activation and initialization value.

    Example:
        >>> # For Gaussian distribution: [mean, std] with ['linear', 'softplus']
        >>> initializer = PredictiveStateParamsInitializer(
        ...     init_values=[0.0, 1.0],  # mean=0, std=1
        ...     n_states=5,
        ...     activations=["linear", "softplus"]
        ... )
    """

    def __init__(
        self,
        init_values: Sequence[Union[float, np.ndarray]],
        n_states: int,
        activations: Sequence[str],
        **kwargs,
    ):
        """Initializes class.

        Args:
            init_values: Sequence of initial values on original scale (one per parameter).
                Each element can be a scalar. Must have length n_params_per_state.
            n_states: Number of predictive states.
            activations: Sequence of activation function names (one per parameter).
                Must have same length as init_values.
        """
        super().__init__(**kwargs)
        self._init_values = list(init_values)
        self._n_states = n_states
        self._activations = list(activations)
        self._n_params_per_state = len(self._init_values)

        # Validate lengths match
        if len(self._activations) != self._n_params_per_state:
            raise ValueError(
                f"activations length {len(self._activations)} must match "
                f"init_values length {self._n_params_per_state}"
            )

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor, expected to be (n_states, n_params_per_state).
            dtype: Optional dtype of the tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Initialized tensor with logits (before activation) computed from
            init_values using inverse activation functions.
        """
        return _get_predictive_state_params_init(
            self._init_values,
            n_states=self._n_states,
            n_params_per_state=self._n_params_per_state,
            activations=self._activations,
        )
