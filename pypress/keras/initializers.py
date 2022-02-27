"""Initializers for PRESS layers."""

from typing import Union, Optional, List

import tensorflow as tf
import numpy as np


def _get_predictive_state_means_init(
    init_value: Union[float, np.ndarray], n_states: int, units: int
) -> tf.Tensor:
    """Gets initial weights for predictive state means."""

    ones = np.ones((units, n_states))
    val = None
    if isinstance(init_value, float):
        val = ones * init_value
    elif isinstance(init_value, np.ndarray):
        assert init_value.shape[0] == units
        if len(init_value.shape) == 1:
            val = init_value[:, np.newaxis] * ones
        else:
            raise ValueError("Initial value must be a row vector.")
    else:
        raise ValueError(
            f"'init_value' must be float or np.ndarray. Got {type(init_value)}."
        )
    if val is not None:
        return tf.Variable(val)
    raise RuntimeError(f"Wrong inputs. Got {init_value}")


def _array_initializer(init_value, units, n_states):
    return lambda: _get_predictive_state_means_init(init_value, n_states, units)


@tf.keras.utils.register_keras_serializable(package="pypress")
class PredictiveStateMeansInitializer(tf.keras.initializers.Initializer):
    """Initializer for weights of the PredictiveStateMeans() layer."""

    def __init__(
        self,
        init_value: Union[float, np.ndarray],
        n_states: int,
        units: Optional[int] = None,
        **kwargs,
    ):
        """Initializes class.

        Args:
          init_value: float or numpy array with initial estimate of predictive states.
          n_states: number of states.
          units: output units; required if init_value is float; optional if init_value is an array.
        """
        super().__init__(**kwargs)
        self._init_value = init_value
        self._n_states = n_states
        self._units = units
        if self._units is None:
            if isinstance(init_value, float):
                self._units = 1
            elif isinstance(init_value, np.ndarray):
                self._units = init_value.shape[0]

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
           supported. If not specified, `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`).
          **kwargs: Additional keyword arguments.
        """
        return _get_predictive_state_means_init(
            self._init_value, n_states=self._n_states, units=shape[-1]
        )
