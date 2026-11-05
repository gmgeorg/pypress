"""Population statistics of predictive state weights, computed in-graph.

Every PRESS penalty is a function of one or more of these statistics of a batch
of predictive state weights `W` (rows sum to 1, one column per predictive
state). Each statistic has a name and a shape that depends only on `K`, the
number of predictive states, so a cross-batch estimator can hold its running
average in a variable created before the first batch is seen.

See CONTEXT.md for the domain terms used here.
"""

from typing import Callable, Dict, Tuple

import tensorflow as tf

from .. import utils


def mean_row_entropy(weights: tf.Tensor) -> tf.Tensor:
    """Mean over observations of the row entropy of the predictive state weights."""
    return -tf.reduce_mean(tf.reduce_sum(utils.safe_xlogx(weights), axis=1))


def state_sizes(weights: tf.Tensor) -> tf.Tensor:
    """State size per predictive state: the column sums of the weights."""
    return tf.reduce_sum(weights, axis=0)


def trace_numerator(weights: tf.Tensor) -> tf.Tensor:
    """Per-state numerator of trace(H): the column sums of the squared weights."""
    return tf.reduce_sum(tf.square(weights), axis=0)


def state_shares(sizes: tf.Tensor) -> tf.Tensor:
    """Normalizes state sizes into state shares (a distribution over states).

    Rows of predictive state weights sum to 1, but this normalizes explicitly so
    the result is a proper distribution for any non-negative input.
    """
    return sizes / tf.maximum(tf.reduce_sum(sizes), 1e-8)


def entropy(p: tf.Tensor) -> tf.Tensor:
    """Shannon entropy (nats) of a distribution 'p' over states."""
    return -tf.reduce_sum(utils.safe_xlogx(p))


def degrees_of_freedom(numerator: tf.Tensor, denominator: tf.Tensor) -> tf.Tensor:
    """trace(H) from its per-state numerator and denominator (state sizes).

    The terms are kept separate so a caller can average each across batches
    before taking the ratio: averaging the ratio itself would keep the
    small-batch bias of a ratio estimator.
    """
    return tf.reduce_sum(numerator / tf.maximum(denominator, 1e-8))


# name -> (function of the weights, shape as a function of K)
_STATISTICS: Dict[str, Tuple[Callable[[tf.Tensor], tf.Tensor], Callable]] = {
    "mean_row_entropy": (mean_row_entropy, lambda n_states: ()),
    "state_sizes": (state_sizes, lambda n_states: (n_states,)),
    "trace_numerator": (trace_numerator, lambda n_states: (n_states,)),
    "trace_denominator": (state_sizes, lambda n_states: (n_states,)),
}


def compute(name: str, weights: tf.Tensor) -> tf.Tensor:
    """Computes the statistic called 'name' on a batch of predictive state weights."""
    return _STATISTICS[name][0](weights)


def shape(name: str, n_states: int) -> Tuple[int, ...]:
    """Shape of the statistic called 'name' for 'n_states' predictive states."""
    return tuple(_STATISTICS[name][1](n_states))


def names() -> Tuple[str, ...]:
    """Names of all known statistics."""
    return tuple(_STATISTICS)
