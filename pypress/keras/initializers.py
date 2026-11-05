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
import pandas as pd
import tensorflow as tf

from . import activations as pr_act
from .activations import get_inverse_activation

InitValues = Union[float, Sequence[float], Sequence[Sequence[float]], np.ndarray]


def state_conditional_values(
    init_values: InitValues, n_states: int, n_columns: int
) -> np.ndarray:
    """Expands 'init_values' into a table of shape (n_states, n_columns).

    Raises:
      ValueError: if 'init_values' is not a number or array, or its shape does
        not match the table.
    """
    if isinstance(init_values, bool) or isinstance(init_values, str):
        raise ValueError(
            f"'init_values' must be a number or an array. Got {type(init_values)}."
        )
    try:
        values = np.asarray(init_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"'init_values' must be a number or an array. Got {type(init_values)}."
        ) from error

    if values.ndim == 0:
        return np.full((n_states, n_columns), float(values))
    if values.ndim == 1:
        if values.shape[0] != n_columns:
            raise ValueError(
                f"1D 'init_values' must have one value per column ({n_columns}). "
                f"Got shape {values.shape}."
            )
        return np.tile(values, (n_states, 1))
    if values.ndim == 2:
        if values.shape != (n_columns, n_states):
            raise ValueError(
                f"2D 'init_values' must have shape ({n_columns}, {n_states}): one "
                f"row per column, one entry per state. Got shape {values.shape}."
            )
        return values.T
    raise ValueError(
        f"'init_values' must be a scalar, 1D or 2D array. Got shape {values.shape}."
    )


def state_conditional_logits(
    init_values: InitValues,
    n_states: int,
    n_columns: int,
    activations: Union[str, Sequence[str]] = "linear",
) -> tf.Tensor:
    """Logits of shape (n_states, n_columns) whose activations equal 'init_values'.

    Args:
      init_values: Values on the original scale, in one of the forms described
        in the module docstring.
      n_states: Number of predictive states, K.
      n_columns: Number of parameters per state, P.
      activations: One activation name for all columns, or one per column.

    Raises:
      ValueError: on a shape mismatch, or an activation without a known inverse.
    """
    values = state_conditional_values(init_values, n_states, n_columns)
    names = pr_act.names(activations, n_columns)
    if len(names) != n_columns:
        raise ValueError(
            f"Need one activation per column ({n_columns}). Got {len(names)}."
        )
    columns = [
        get_inverse_activation(name)(tf.constant(values[:, i], dtype=tf.float32))
        for i, name in enumerate(names)
    ]
    return tf.stack(columns, axis=1)


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


def initialize_from_y(
    y: Union[np.ndarray, pd.Series, pd.DataFrame],
    n_states: int,
    return_params: bool = False,
    max_iter: int = 10,
) -> np.ndarray:
    """Initial state-conditional values from a 1D Gaussian mixture fitted to y.

    Fits `pypress.clustering.GaussianMixture1D` with `n_states` components to
    the training outputs, and returns `init_values` for `PredictiveStateMeans`
    or `PredictiveStateParams`.

    For multi-output y, all outputs are pooled into one 1D sample before the
    fit, so every output unit gets the same per-state means. This is a simple
    heuristic: it suits outputs on a common scale.

    Args:
        y: Training outputs of shape (n_samples,) or (n_samples, n_outputs).
        n_states: Number of predictive states (mixture components).
        return_params: If False, returns state-conditional means for
            `PredictiveStateMeans`. If True, returns [mean, std] per state for
            `PredictiveStateParams` with a Gaussian parameterization.
        max_iter: Maximum number of EM iterations. Default is 10.

    Returns:
        If return_params=False: array of shape (n_outputs, n_states) of the
            mixture means, the same row for every output.
        If return_params=True: array of shape (2, n_states); row 0 holds the
            means and row 1 the standard deviations, for
            `activations=["linear", "softplus"]`.

    Example:
        >>> y_train = np.random.randn(1000, 2)
        >>> layer = PredictiveStateMeans(
        ...     units=2, init_values=initialize_from_y(y_train, n_states=5)
        ... )
        >>> layer = PredictiveStateParams(
        ...     n_params_per_state=2,
        ...     activations=["linear", "softplus"],
        ...     init_values=initialize_from_y(y_train, n_states=5, return_params=True),
        ... )
    """
    from ..clustering import GaussianMixture1D

    y = np.asarray(y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y)
    units = 1 if y.ndim == 1 else y.shape[1]

    gmm = GaussianMixture1D(n_components=n_states, max_iter=max_iter)
    gmm.fit(y.ravel())

    if return_params:
        return np.vstack([gmm.means_, gmm.covariances_])
    return np.tile(gmm.means_, (units, 1))
