"""Initializers for state-conditional parameters, from original-scale values.

A table of state-conditional parameters has one row per predictive state and
one column per parameter (for `PredictiveStateMeans`, one column per output
unit). Its weights are logits: the layer applies an activation to them. Users
give initial values on the original scale -- after the activation -- and the
initializer inverts the activation column by column.

`init_values` takes the same three forms everywhere:

- a scalar: every entry of the table;
- a 1D array of length P (one value per column): broadcast across states;
- a 2D array of shape (P, K): state-specific values, one row per column.
"""

from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from ..core import initializers as core_init
from ..core.initializers import (  # noqa: F401
    InitValues,
    initialize_from_y,
    state_conditional_values,
)
from . import activations as pr_act


def state_conditional_logits(
    init_values: InitValues,
    n_states: int,
    n_columns: int,
    activations: Union[str, Sequence[str]] = "linear",
) -> tf.Tensor:
    """Logits of shape (n_states, n_columns) whose activations equal 'init_values'.

    The logits are computed in numpy (`pypress.core.initializers`) and returned as
    a float32 tensor.

    Args:
      init_values: Values on the original scale, in one of the forms described
        in the module docstring.
      n_states: Number of predictive states, K.
      n_columns: Number of parameters per state, P.
      activations: One activation (a name, Keras function or `Activation`
        layer) for all columns, or one per column.

    Raises:
      ValueError: on a shape mismatch, or an activation without a known inverse.
    """
    names = pr_act.names(activations, n_columns)
    logits = core_init.state_conditional_logits(init_values, n_states, n_columns, names)
    return tf.constant(logits, dtype=tf.float32)


def _to_config(init_values: InitValues):
    """JSON-safe form of 'init_values'."""
    return init_values.tolist() if isinstance(init_values, np.ndarray) else init_values


@tf.keras.utils.register_keras_serializable(package="pypress")
class StateConditionalInitializer(tf.keras.initializers.Initializer):
    """Initializes a (n_states, n_columns) table of logits from original-scale values.

    Example:
        >>> # Gaussian [mean, std] per state, the same in every state.
        >>> init = StateConditionalInitializer([0.0, 1.0], ["linear", "softplus"])
        >>> init((5, 2)).shape
        TensorShape([5, 2])
    """

    def __init__(
        self,
        init_values: InitValues,
        activations: Union[str, Sequence[str]] = "linear",
    ):
        """Initializes the initializer.

        Args:
          init_values: Values on the original scale (after activation).
          activations: One activation name for all columns, or one per column.
        """
        self._init_values = init_values
        self._activations = activations

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns the logits for a table of 'shape' (n_states, n_columns)."""
        n_states, n_columns = shape
        logits = state_conditional_logits(
            self._init_values, n_states, n_columns, self._activations
        )
        return tf.cast(logits, dtype or tf.keras.backend.floatx())

    def get_config(self):
        """Gets the config."""
        return {
            "init_values": _to_config(self._init_values),
            "activations": self._activations,
        }


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateMeansInitializer(StateConditionalInitializer):
    """Initializer for the state-conditional means of `PredictiveStateMeans`.

    Example:
        >>> # Sigmoid outputs initialized at 0.8 probability in every state.
        >>> init = PredictiveStateMeansInitializer(0.8, n_states=3, activation="sigmoid")
    """

    def __init__(
        self,
        init_values: InitValues,
        n_states: Optional[int] = None,
        activation: str = "linear",
        units: Optional[int] = None,
        **kwargs,
    ):
        """Initializes the initializer.

        Args:
          init_values: Values on the original scale (after activation).
          n_states: Unused; the number of states comes from the weight shape.
            Kept for backwards compatibility.
          activation: Activation name applied to the means.
          units: Unused; the number of units comes from the weight shape. Kept
            for backwards compatibility.
        """
        super().__init__(init_values=init_values, activations=activation)

    def get_config(self):
        """Gets the config."""
        return {
            "init_values": _to_config(self._init_values),
            "activation": self._activations,
        }


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateParamsInitializer(StateConditionalInitializer):
    """Initializer for the state-conditional parameters of `PredictiveStateParams`.

    Example:
        >>> # Gaussian [mean, std], the same in every state.
        >>> init = PredictiveStateParamsInitializer(
        ...     init_values=[0.0, 1.0], n_states=5, activations=["linear", "softplus"]
        ... )
    """

    def __init__(
        self,
        init_values: InitValues,
        n_states: Optional[int] = None,
        activations: Union[str, Sequence[str]] = "linear",
        **kwargs,
    ):
        """Initializes the initializer.

        Args:
          init_values: Values on the original scale (after activation).
          n_states: Unused; the number of states comes from the weight shape.
            Kept for backwards compatibility.
          activations: One activation name for all parameters, or one per
            parameter.
        """
        super().__init__(init_values=init_values, activations=activations)
