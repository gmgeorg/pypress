"""Utility functions."""

from typing import Union

import pandas as pd
import tensorflow as tf
import numpy as np

_STATE_COL = "state"


def state_size(
    weights: Union[pd.DataFrame, np.ndarray]
) -> Union[pd.Series, np.ndarray]:
    """Computes size of states as total sum of probabilities per state."""
    col_sums = weights.sum(axis=0)
    if isinstance(weights, pd.DataFrame):
        col_sums.index.name = _STATE_COL
    return col_sums


def tf_state_size(weights: tf.Tensor) -> tf.Tensor:
    """Computes size of states for input Tensor."""
    return tf.reduce_sum(weights, axis=0)


def col_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Colum normalizes DataFrame (usually 'weights')."""
    return df.divide(state_size(df), axis=1)


def tf_col_normalize(x: tf.Tensor) -> tf.Tensor:
    """Computes column normalized tensor."""
    return tf.divide(x, tf_state_size(x))


def agg_data_by_state(
    data: pd.DataFrame, weights: pd.DataFrame, agg_fn: str = "mean"
) -> pd.DataFrame:
    """Aggregates data by state. Will only aggregate numeric features."""
    if isinstance(data, pd.Series):
        data = data.to_frame()
    df_numeric = data.select_dtypes("number")
    if agg_fn == "mean":
        agg_state = col_normalize(weights).transpose().dot(df_numeric).transpose()
        agg_state.columns.name = None
        return agg_state

    raise NotImplementedError("agg_fn='%s' not supported", agg_fn)
