"""Module for PRESS layers."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from . import initializers as pr_init


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
        kernel_initializer="zeros",
        bias_initializer="zero",
        **kwargs,
    ):
        """Initializes the class.

        Args:
          n_states: number of predictive states.
          **kwargs: keyword arguments passed to Dense().
        """
        assert (
            n_states >= 1
        ), f"Number of predictive states must be >= 1. Got {n_states}."
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


class PredictiveStateMeans(tf.keras.layers.Layer):
    """Layer that implements predicting from the predictive state simplex (input) to outputs.

    Input to this layer must be the output of PredictiveStateSimplex, which are a 'softmax' weight
    representation (w_1, ..., w_J).

    PredictiveStateMeans(X) --> \\sum_{j=1}^{J} w_j(X) * mu_j
    """

    def __init__(
        self,
        units: int,
        activation: Union[str, tf.keras.layers.Activation] = "linear",
        predictive_state_means_init_logits: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Initializes the class."""
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
        scm_shape = input_shape[1:] + [self._units]

        if self._predictive_state_means_init_logits is None:
            initializer = None
        else:
            initializer = pr_init.PredictiveStateMeansInitializer(
                self._predictive_state_means_init_logits, n_states=self._n_states
            )

        self._state_conditional_mean_logits = self.add_weight(
            "state_conditional_mean_logits",
            shape=scm_shape,
            initializer=initializer,
            trainable=True,
        )

    @property
    def state_conditional_means(self) -> tf.Tensor:
        """Returns state conditional means (after activation)."""
        return self._activation_layer(self._state_conditional_mean_logits)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Implements call of layer to compute mixture of means."""
        return tf.matmul(inputs, self.state_conditional_means)

    def get_config(self) -> Dict[str, Any]:
        """Returns configuration."""
        config = super().get_config()
        config.update(
            {
                "units": self._units,
                "activation": self._activation,
                "predictive_state_means_init_logits": self._predictive_state_means_init_logits,
            }
        )
        return config


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
