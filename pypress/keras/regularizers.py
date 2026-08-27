"""Regularizers for PRESS weights."""

import tensorflow as tf
import warnings

from .. import utils


@tf.keras.utils.register_keras_serializable(package="pypress")
class TargetEntropy(tf.keras.regularizers.Regularizer):
    """Penalizes weights if mean entropy deviates from a specific target value.

    By default, targets log(0.5 * K), representing a balanced 50% regime overlap
    anchor (where K = x.shape[1] at call time). Uses a squared (L2) penalty to
    provide smooth gradient behavior near equilibrium.

    Target = log(K) is maximum overlap (uniform distribution).
    Target = 0 is minimum overlap (deterministic / single-state assignment).

    The deviation (target - mean entropy) is normalized by log(K), the maximum
    possible entropy for K columns, before squaring. Without this, the penalty's
    dynamic range grows as log(K)^2 while the (unrelated) mixture negative
    log-likelihood's does not, so the effective regularization strength would
    silently depend on the number of states K. Normalizing keeps the penalty
    bounded in [0, l2] regardless of K, so `l2` means the same thing across
    different values of K.
    """

    # Fraction of K used as the dynamic default target active states.
    _entropy_fraction = 0.5

    def __init__(self, l2: float = 0.0, *, target: float | None = None, **kwargs):
        """Initializes the regularizer.

        Args:
          l2: Penalty multiplier weight for the squared deviation from target.
          target: Target mean entropy value. If None, defaults to
            `log(0.5 * K)`, with K = x.shape[1] computed dynamically at call time.
          **kwargs: Additional keyword arguments for Keras regularizers.
        """
        super().__init__(**kwargs)
        self._l2 = l2
        self._target = target

    def _target_value(self, x):
        """Computes or retrieves the target entropy value for input tensor 'x'."""
        if self._target is not None:
            return self._target

        K = tf.cast(x.shape[1], dtype=tf.float32)
        # Clamped at 1.0 active state so log(...) is non-negative for K >= 1
        active_states = tf.maximum(1.0, self._entropy_fraction * K)
        return tf.math.log(active_states)

    def _max_entropy(self, x):
        """Computes log(K), the maximum possible entropy for K = x.shape[1] columns."""
        K = tf.cast(x.shape[1], dtype=tf.float32)
        return tf.math.log(tf.maximum(K, 1.0))

    def __call__(self, x):
        """Computes squared penalty for deviation of mean entropy from target.

        The deviation is normalized by log(K) so the penalty is scale-invariant
        with respect to K = x.shape[1], keeping it bounded in [0, l2].
        """
        entropy_per_row = -1.0 * tf.math.reduce_sum(utils.safe_xlogx(x), axis=1)
        mean_entropy = tf.math.reduce_mean(entropy_per_row)

        # Squared, K-normalized deviation (L2 penalty) for smooth optimizer updates
        deviation = self._target_value(x) - mean_entropy
        # max_entropy is 0 only when K == 1, in which case deviation is also 0
        # (both entropy and its clamped target are 0); guard against 0/0.
        normalized_deviation = deviation / tf.maximum(self._max_entropy(x), 1e-8)
        return self._l2 * tf.math.square(normalized_deviation)

    def get_config(self):
        """Returns the serializable config dictionary for Keras."""
        config = {"l2": float(self._l2)}
        if self._target is not None:
            config["target"] = float(self._target)
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class Uniform(TargetEntropy):
    """Penalizes weights if they are not uniform across columns (1 / J).

    Special case of TargetEntropy where the target is fixed to the maximum
    possible entropy for uniform weights, log(# of columns), always computed
    dynamically from the input. For a custom target, use TargetEntropy directly.
    """

    _entropy_fraction = 1.0

    def __init__(self, l2: float = 0.0, **kwargs):
        """Initializes the uniform regularizer.

        Args:
          l2: penalty multiplier weight for the squared deviation from target.
          **kwargs: addl keyword arguments to regularizers.
        """
        super().__init__(l2=l2, target=None, **kwargs)

    def get_config(self):
        """Gets the config."""
        return {"l2": float(self._l2)}


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

    def __init__(
        self, l2: float = 0.0, target: float = 1.0, df: float = None, **kwargs
    ):
        """Initializes the regularizer.

        Args:
          l2: l2 penalty parameter for l2 * (df - df(kernel)) ** 2
          target: degrees of freedom parameter target value. Must be >= 1.
        """
        assert target >= 1.0, (
            f"Target for degrees of freedom must be >= 1. Got {target}."
        )
        if df is not None:
            warnings.warn("'df' is deprecated. Use 'target' instead.")
            target = df

        super().__init__(**kwargs)
        self._target = target
        self._l2 = l2

    def __call__(self, x):
        """Computes squared penalty for deviation from target degrees of freedom."""
        return self._l2 * tf.square(utils.tr_kernel(x) - self._target)

    def get_config(self):
        """Gets the config."""
        return {"l2": float(self._l2), "target": float(self._target), "df": None}


@tf.keras.utils.register_keras_serializable(package="pypress")
class UniformAndDegreesOfFreedomRegularizer(tf.keras.regularizers.Regularizer):
    """
    A combined regularizer that sums two penalties:
      1. Uniform penalty (to penalize deviations from uniformity across columns)
      2. DegreesOfFreedom penalty (to penalize deviations of the implied kernel trace
         from a target degrees of freedom)

    Keyword arguments:
      uniform_l2: float, penalty weight for the Uniform regularizer (squared deviation).
      dof_l2: float, penalty weight for the DegreesOfFreedom regularizer.
      dof_target: float, target value for the degrees of freedom.
    """

    def __init__(
        self,
        uniform_l2: float = 0.0,
        dof_l2: float = 0.0,
        dof_target: float = 1.0,
        target_entropy: float = None,
        **kwargs,
    ):
        """Initializes the class."""
        super().__init__(**kwargs)
        self.uniform_l2 = uniform_l2
        self.dof_l2 = dof_l2
        self.dof_target = dof_target
        self.target_entropy = target_entropy
        # Explicitly instantiate the two internal regularizers. Uniform's target is
        # fixed to log(# of columns); a custom target_entropy requires TargetEntropy.
        self._uniform = (
            TargetEntropy(l2=self.uniform_l2, target=target_entropy)
            if target_entropy is not None
            else Uniform(l2=self.uniform_l2)
        )
        self._dof = DegreesOfFreedom(l2=self.dof_l2, target=self.dof_target)

    def __call__(self, x):
        # Apply both regularizers and return their sum.
        return self._uniform(x) + self._dof(x)

    def get_config(self):
        """Gets the config."""
        config = {
            "uniform_l2": float(self.uniform_l2),
            "dof_l2": float(self.dof_l2),
            "dof_target": float(self.dof_target),
        }
        if self.target_entropy is not None:
            config["target_entropy"] = float(self.target_entropy)
        return config


@tf.keras.utils.register_keras_serializable(package="pypress")
class CombinedRegularizer(tf.keras.regularizers.Regularizer):
    """
    A generic combined regularizer that sums the penalties from a list of regularizers.
    This version accepts a list of tuples of the form:

        [(regularizer_constructor, kwargs_dict), ...]

    and instantiates each regularizer accordingly.
    """

    def __init__(self, regularizer_tuples, **kwargs):
        """Initializes class."""
        super().__init__(**kwargs)
        self.regularizer_tuples = regularizer_tuples
        # Instantiate each regularizer from its constructor and kwargs.
        self.regularizers = [ctor(**kw) for (ctor, kw) in regularizer_tuples]

    def __call__(self, x):
        """Evaluates the regularizer on input."""
        total_penalty = 0.0
        for reg in self.regularizers:
            total_penalty += reg(x)
        return total_penalty

    def get_config(self):
        """Gets the config."""
        # For simplicity, we store the list of tuples as (ctor.__name__, kwargs) pairs.
        config = {
            "regularizer_tuples": [
                (ctor.__name__, kw) for (ctor, kw) in self.regularizer_tuples
            ]
        }
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreates the CombinedRegularizer from its configuration.

        The config is expected to have a key "regularizer_tuples" containing a list of
        tuples of (constructor name, kwargs). We assume that the corresponding constructors
        are registered (or available via direct import) and we look them up.
        """
        # Get the list of tuples from the config.
        regularizer_tuples = config.pop("regularizer_tuples")
        # Import the module and look up the constructors by name.
        import pypress.keras.regularizers as regs

        new_tuples = []
        for ctor_name, kwargs in regularizer_tuples:
            # Get the constructor from the module by name.
            ctor = getattr(regs, ctor_name)
            new_tuples.append((ctor, kwargs))
        return cls(regularizer_tuples=new_tuples, **config)
