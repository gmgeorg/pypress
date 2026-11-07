"""Declared regularizer hyperparameters, for Keras.

The declaration, validation and serialization of a regularizer's scalar
settings is framework-free and lives in `pypress.core.hyperparameters`. This
module adds what needs TensorFlow: deserializing a schedule into the Keras
`ScheduledValue` and reading a setting as a tensor inside the training graph.
"""

import dataclasses

import tensorflow as tf

from ..core import hyperparameters as core_hp
from . import schedules


@dataclasses.dataclass(frozen=True)
class Hyperparameter(core_hp.Hyperparameter):
    """A declared scalar setting of a Keras regularizer; see the core class."""

    @staticmethod
    def _deserialize(value: dict):
        """Builds the Keras `ScheduledValue` from its serialized form."""
        return schedules.deserialize_scalar(value)

    @staticmethod
    def tensor(value) -> tf.Tensor:
        """Reads a stored setting as a float32 tensor, in-graph."""
        return schedules.as_tensor(value)

    @staticmethod
    def serialize(value):
        """Config representation of a stored setting."""
        return None if value is None else schedules.serialize_scalar(value)


L2 = Hyperparameter(**dataclasses.asdict(core_hp.L2))
EMA_DECAY = Hyperparameter(**dataclasses.asdict(core_hp.EMA_DECAY))
