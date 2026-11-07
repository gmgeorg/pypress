"""Initial state-conditional values, in numpy.

A table of state-conditional parameters has one row per predictive state and
one column per parameter (for `PredictiveStateMeans`, one column per output
unit). Its weights are logits: the layer applies an activation to them. Users
give initial values on the original scale -- after the activation -- and the
logits are the activations' inverses, applied column by column.

`init_values` takes the same three forms everywhere:

- a scalar: every entry of the table;
- a 1D array of length P (one value per column): broadcast across states;
- a 2D array of shape (P, K): state-specific values, one row per column.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd

from . import activations

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
    activation_names: Union[str, Sequence[str]] = "linear",
) -> np.ndarray:
    """Logits of shape (n_states, n_columns) whose activations equal 'init_values'.

    Args:
      init_values: Values on the original scale, in one of the forms described
        in the module docstring.
      n_states: Number of predictive states, K.
      n_columns: Number of parameters per state, P.
      activation_names: One activation name for all columns, or one per column.

    Returns:
      A float64 array; the caller casts it to its framework's dtype.

    Raises:
      ValueError: on a shape mismatch, or an activation without a known inverse.
    """
    values = state_conditional_values(init_values, n_states, n_columns)
    return activations.invert(values, activation_names)


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
