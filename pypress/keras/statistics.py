"""Population statistics of predictive state weights, computed in-graph.

Every PRESS penalty is a function of one or more of these statistics of a batch
of predictive state weights `W` (rows sum to 1, one column per predictive
state). Each statistic has a name and a shape that depends only on `K`, the
number of predictive states, so a cross-batch estimator can hold its running
average in a variable created before the first batch is seen.

See CONTEXT.md for the domain terms used here.
"""

from typing import Callable, Dict

import tensorflow as tf

from ..core.statistics import names, shape  # noqa: F401


def safe_log(p: tf.Tensor, neg_inf: float = -1e9, name: str | None = None) -> tf.Tensor:
    """Numerically stable log(p) for p in [0, 1], e.g. softmax/simplex outputs.

    Equivalent in spirit to computing log-weights via `log_softmax(logits)` directly
    (no artificial floor on small-but-nonzero values), for cases where only the
    already-softmaxed probabilities are available and not the underlying logits.
    `log(p + eps)` imposes a hard floor at `log(eps)` (e.g. ~ -13.8 for eps=1e-6),
    which distorts sharp/near-degenerate states whose true log-probability is far
    below that floor. This instead only substitutes `neg_inf` for exact zeros
    (float32 softmax underflow), and does so via `tf.where` so no NaN/inf gradient
    flows back through `tf.math.log` at p == 0.
    """
    p = tf.convert_to_tensor(p)
    is_positive = p > 0
    safe_p = tf.where(is_positive, p, tf.ones_like(p))
    return tf.where(
        is_positive,
        tf.math.log(safe_p),
        tf.fill(tf.shape(p), tf.constant(neg_inf, p.dtype)),
        name=name,
    )


def safe_xlogx(x: tf.Tensor, name: str | None = None) -> tf.Tensor:
    """Numerically stable x * log(x) for x in [0, 1], e.g. entropy terms.

    Unlike `x * safe_log(x)`, this substitutes the *entire product* with 0 at x == 0,
    matching the mathematical limit `lim_{x->0} x*log(x) = 0`. Multiplying `x` by
    `safe_log(x)` directly would reintroduce `safe_log`'s large `neg_inf` floor value
    into the gradient at x == 0 via the product rule (`d/dx[x*f(x)] = f(x) + x*f'(x)`),
    producing a large gradient spike instead of the mathematically correct, bounded one.
    """
    x = tf.convert_to_tensor(x)
    is_positive = x > 0
    safe_x = tf.where(is_positive, x, tf.ones_like(x))
    return tf.where(is_positive, x * tf.math.log(safe_x), tf.zeros_like(x), name=name)


def mean_row_entropy(weights: tf.Tensor) -> tf.Tensor:
    """Mean over observations of the row entropy of the predictive state weights."""
    return -tf.reduce_mean(tf.reduce_sum(safe_xlogx(weights), axis=1))


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
    return -tf.reduce_sum(safe_xlogx(p))


def degrees_of_freedom(numerator: tf.Tensor, denominator: tf.Tensor) -> tf.Tensor:
    """trace(H) from its per-state numerator and denominator (state sizes).

    The terms are kept separate so a caller can average each across batches
    before taking the ratio: averaging the ratio itself would keep the
    small-batch bias of a ratio estimator.
    """
    return tf.reduce_sum(numerator / tf.maximum(denominator, 1e-8))


# name -> function of the weights. The shape of each statistic is in
# `pypress.core.statistics`, shared with the PyTorch implementation.
_STATISTICS: Dict[str, Callable[[tf.Tensor], tf.Tensor]] = {
    "mean_row_entropy": mean_row_entropy,
    "state_sizes": state_sizes,
    "trace_numerator": trace_numerator,
    "trace_denominator": state_sizes,
}


def compute(name: str, weights: tf.Tensor) -> tf.Tensor:
    """Computes the statistic called 'name' on a batch of predictive state weights."""
    return _STATISTICS[name](weights)
