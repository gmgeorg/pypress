"""Post-hoc analysis of fitted predictive state weights, in numpy and pandas.

Every function takes the predictive state weights `W` of a set of observations
(one row per observation, one column per predictive state, rows summing to 1),
for example the output of `PRESS.predictive_states(X)`. DataFrames keep their
index and column names; arrays come back as arrays.

The in-graph counterparts used by the regularizers live in
`pypress.keras.statistics`. See CONTEXT.md for the domain terms used here.
"""

from typing import Union

import numpy as np
import pandas as pd

_STATE_COL = "state"

Weights = Union[pd.DataFrame, np.ndarray]


def state_size(
    weights: Weights, normalize: bool = False
) -> Union[pd.Series, np.ndarray]:
    """State size per predictive state: the column sums of the weights.

    Args:
      weights: Predictive state weights, shape (n_samples, n_states).
      normalize: If True, returns state shares (sizes divided by their total).
    """
    col_sums = weights.sum(axis=0)
    if isinstance(weights, pd.DataFrame):
        col_sums.index.name = _STATE_COL

    if normalize:
        col_sums = col_sums / col_sums.sum()
    return col_sums


def col_normalize(weights: Weights) -> Weights:
    """Divides each column of the weights by its state size."""
    if isinstance(weights, pd.DataFrame):
        return weights.divide(state_size(weights), axis=1)
    weights = np.asarray(weights)
    return weights / weights.sum(axis=0, keepdims=True)


def smoother_matrix(weights: Weights) -> Weights:
    """The smoother matrix H = W @ col_normalize(W).T, shape (n_samples, n_samples).

    H maps observed outputs to fitted values; it is O(n^2) in memory. For its
    trace use `degrees_of_freedom`, which is O(n * K).

    Example:
        >>> weights = pd.DataFrame({'s1': [0.1, 0.2, 0.7], 's2': [0.8, 0.1, 0.1]})
        >>> smoother_matrix(weights).shape
        (3, 3)
    """
    if isinstance(weights, pd.DataFrame):
        return weights.dot(col_normalize(weights).transpose())
    weights = np.asarray(weights)
    return weights @ col_normalize(weights).T


def degrees_of_freedom(weights: Weights) -> float:
    """trace(H), the effective number of states, without forming H.

    Uses trace(H) = sum_k (sum_i w_ik^2) / (sum_i w_ik).
    """
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(np.square(weights).sum(axis=0) / weights.sum(axis=0)))


def agg_data_by_state(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    weights: Weights,
    agg_fn: str = "mean",
) -> Union[pd.DataFrame, np.ndarray]:
    """Weighted aggregate of each numeric column of 'data' per predictive state.

    Args:
      data: The observations, shape (n_samples, n_features): a DataFrame (its
        numeric columns are used), a Series, or an array.
      weights: Their predictive state weights, shape (n_samples, n_states).
      agg_fn: The aggregate; only "mean" is supported.

    Returns:
      One row per feature and one column per predictive state. An array when
      'data' and 'weights' are both arrays, otherwise a DataFrame labelled by the
      DataFrame inputs (an array input takes the other input's row index).

    Raises:
      ValueError: if 'data' and 'weights' have different numbers of rows.
    """
    if agg_fn != "mean":
        raise NotImplementedError(f"agg_fn={agg_fn!r} not supported")
    if len(data) != len(weights):
        raise ValueError(
            f"'data' has {len(data)} rows but 'weights' has {len(weights)}; "
            "they must describe the same observations."
        )
    if not isinstance(data, (pd.DataFrame, pd.Series)) and not isinstance(
        weights, pd.DataFrame
    ):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return (col_normalize(np.asarray(weights, dtype=float)).T @ data).T

    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        data = np.asarray(data)
        data = pd.DataFrame(data.reshape(len(data), -1), index=weights.index)
    if not isinstance(weights, pd.DataFrame):
        weights = pd.DataFrame(np.asarray(weights), index=data.index)
    df_numeric = data.select_dtypes("number")
    agg_state = col_normalize(weights).transpose().dot(df_numeric).transpose()
    agg_state.columns.name = None
    return agg_state
