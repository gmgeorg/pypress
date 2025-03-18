"""Regularizers for PRESS weights."""

import tensorflow as tf

from .. import utils

_EPS = 1e-6


@tf.keras.utils.register_keras_serializable(package="pypress")
class Uniform(tf.keras.regularizers.Regularizer):
    """Penalizes weights if they are not uniform across columns (1 / J).

    Does this by computing Shannon Entropy across each row and taking difference
    to maximum possible entropy for uniform weights (log(# of columns))
    """

    def __init__(self, l1: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._l1 = l1

    def __call__(self, x):
        """Computes l1 penalty given weight weights 'x'."""
        entropy_per_row = -1 * tf.math.reduce_sum(x * tf.math.log(x + _EPS), axis=1)
        return self._l1 * (
            tf.math.log(tf.cast(x.shape[1], dtype=tf.float32))
            - tf.math.reduce_mean(entropy_per_row)
        )

    def get_config(self):
        return {"l1": float(self._l1)}


def tr_kernel(weights: tf.Tensor) -> tf.Tensor:
    """Computes trace of kernel matrix implied by PRESS tensor."""
    return tf.reduce_sum(
        tf.linalg.diag_part(
            tf.matmul(tf.transpose(utils.tf_col_normalize(weights)), weights)
        )
    )


@tf.keras.utils.register_keras_serializable(package="pypress")
class DegreesOfFreedom(tf.keras.regularizers.Regularizer):
    """Penalizes weights if the resulting kernel matrix deviates from target degrees of freedom.

    PRESS kernel smoother implied by predictive states equals

        K = W * D^(-1) * W' in R^{N x N},

    where D is a diagonal matrix with D_ii = size of state i = sum_j w_i,j.

    Degrees of freedom of a kernel smoother is equal to the trace of the kernel matrix.
    In general the trace must be computed from the full N x N kernel matrix diagonal,
    which can be prohibitive if N is large.  However, due to special structure
    of the PRESS kernel and properties of trace operator, this can be simplified as

        trace(K) = trace(W * D^-1 * W') = trace(W' * W * D^-1),

    which is the trace of a J x J matrix, where J << N is the number of states.

    Penalizer here is penalizing if the empirical degrees of freedom is different
    to target value.
    """

    def __init__(self, l1: float = 0.0, df: float = 1.0):
        """Initializes the regularizer.

        Args:
          l1: l1 penalty parameter for l1 * |df - df(kernel)|
          df: degrees of freedom parameter target value. Must be >= 1.
        """
        assert df >= 1.0, f"Target for degrees of freedom must be >= 1. Got {df}."
        self._df = df
        self._l1 = l1

    def __call__(self, x):
        """Computes penalty based on L1 deviation from target degrees of freedom."""
        return self._l1 * tf.abs(tr_kernel(x) - self._df)

    def get_config(self):
        return {"l1": float(self._l1), "df": float(self._df)}
