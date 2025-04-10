"""Regularizers for PRESS weights."""

import tensorflow as tf
import warnings

from .. import utils

_EPS = 1e-6


@tf.keras.utils.register_keras_serializable(package="pypress")
class Uniform(tf.keras.regularizers.Regularizer):
    """Penalizes weights if they are not uniform across columns (1 / J).

    Does this by computing Shannon Entropy across each row and taking difference
    to maximum possible entropy for uniform weights (log(# of columns))
    """

    def __init__(self, l1: float = 0.0, **kwargs):
        """Initializes the uniform regularizer.

        Args:
          l1: penalty term.
          **kwargs: addl keyword arguments to regularizers.
        """
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
        """Gets the config."""
        return {"l1": float(self._l1)}


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
        self, l1: float = 0.0, target: float = 1.0, df: float = None, **kwargs
    ):
        """Initializes the regularizer.

        Args:
          l1: l1 penalty parameter for l1 * |df - df(kernel)|
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
        self._l1 = l1

    def __call__(self, x):
        """Computes penalty based on L1 deviation from target degrees of freedom."""
        return self._l1 * tf.abs(utils.tr_kernel(x) - self._target)

    def get_config(self):
        """Gets the config."""
        return {"l1": float(self._l1), "target": float(self._target)}


@tf.keras.utils.register_keras_serializable(package="pypress")
class UniformAndDegreesOfFreedomRegularizer(tf.keras.regularizers.Regularizer):
    """
    A combined regularizer that sums two penalties:
      1. Uniform penalty (to penalize deviations from uniformity across columns)
      2. DegreesOfFreedom penalty (to penalize deviations of the implied kernel trace
         from a target degrees of freedom)

    Keyword arguments:
      uniform_l1: float, penalty weight for the Uniform regularizer.
      dof_l1: float, penalty weight for the DegreesOfFreedom regularizer.
      dof_target: float, target value for the degrees of freedom.
    """

    def __init__(
        self, uniform_l1: float = 0.0, dof_l1: float = 0.0, dof_target: float = 1.0
    ):
        """Initializes the class."""
        self.uniform_l1 = uniform_l1
        self.dof_l1 = dof_l1
        self.dof_target = dof_target
        # Explicitly instantiate the two internal regularizers:
        self._uniform = Uniform(l1=self.uniform_l1)
        self._dof = DegreesOfFreedom(l1=self.dof_l1, target=self.dof_target)

    def __call__(self, x):
        # Apply both regularizers and return their sum.
        return self._uniform(x) + self._dof(x)

    def get_config(self):
        """Gets the config."""
        return {
            "uniform_l1": self.uniform_l1,
            "dof_l1": self.dof_l1,
            "dof_target": self.dof_target,
        }


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
        Recreates the CompositeRegularizer from its configuration.

        The config is expected to have a key "regularizer_tuples" containing a list of
        tuples of (constructor name, kwargs). We assume that the corresponding constructors
        are registered (or available via direct import) and we look them up.
        """
        # Get the list of tuples from the config.
        regularizer_tuples = config.pop("regularizer_tuples")
        # In this simple example, we assume that the constructor names in the tuples match
        # the actual classes accessible from the pypress.keras.regularizers module.
        # For a more robust implementation, you might use a registry.
        # Here we import the module and look up the constructors by name.
        import pypress.keras.regularizers as regs

        new_tuples = []
        for ctor_name, kwargs in regularizer_tuples:
            # Get the constructor from the module by name.
            ctor = getattr(regs, ctor_name)
            new_tuples.append((ctor, kwargs))
        return cls(regularizer_tuples=new_tuples, **config)
