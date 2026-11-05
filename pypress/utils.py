"""Utility functions, kept for backwards compatibility.

New code should import from the module that owns each function:

- `pypress.analysis`: post-hoc analysis of fitted predictive state weights
  (`state_size`, `col_normalize`, `smoother_matrix`, `degrees_of_freedom`,
  `agg_data_by_state`).
- `pypress.keras.statistics`: in-graph statistics and numerics used by the
  regularizers (`state_sizes`, `trace_numerator`, `safe_log`, `safe_xlogx`).
- `pypress.keras.initializers`: `initialize_from_y`.
"""

import warnings
from typing import Tuple

import tensorflow as tf

from .analysis import (  # noqa: F401
    agg_data_by_state,
    col_normalize,
    smoother_matrix,
    state_size,
)
from .keras import statistics as _statistics
from .keras.initializers import initialize_from_y  # noqa: F401
from .keras.statistics import safe_log, safe_xlogx  # noqa: F401

# The smoother matrix H under its earlier name.
kernel_matrix = smoother_matrix


def _deprecated(name: str, replacement: str):
    """Warns that 'name' is deprecated in favour of 'replacement'."""
    warnings.warn(
        f"pypress.utils.{name} is deprecated; use {replacement}.",
        DeprecationWarning,
        stacklevel=3,
    )


def tf_state_size(weights: tf.Tensor) -> tf.Tensor:
    """Deprecated: use `pypress.keras.statistics.state_sizes`."""
    _deprecated("tf_state_size", "pypress.keras.statistics.state_sizes")
    return _statistics.state_sizes(weights)


def tf_col_normalize(x: tf.Tensor) -> tf.Tensor:
    """Deprecated: divides each column by its state size."""
    _deprecated("tf_col_normalize", "x / pypress.keras.statistics.state_sizes(x)")
    return tf.divide(x, _statistics.state_sizes(x))


def tr_kernel_terms(weights: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Deprecated: per-state numerator and denominator of trace(H)."""
    _deprecated(
        "tr_kernel_terms",
        "pypress.keras.statistics.trace_numerator and .state_sizes",
    )
    return _statistics.trace_numerator(weights), _statistics.state_sizes(weights)


def tr_kernel(weights: tf.Tensor) -> tf.Tensor:
    """Deprecated: trace(H) of a batch of predictive state weights."""
    _deprecated(
        "tr_kernel",
        "pypress.analysis.degrees_of_freedom (numpy) or "
        "pypress.keras.statistics.degrees_of_freedom (in-graph)",
    )
    return tf.reduce_sum(
        _statistics.trace_numerator(weights) / _statistics.state_sizes(weights)
    )
