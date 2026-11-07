"""Shared pytest fixtures."""

import sys

import pytest

from pypress.core import schedules as core_schedules


@pytest.fixture(autouse=True)
def _reset_training_clock():
    """Stops the shared training clock after every test, so tests stay independent."""
    yield
    core_schedules.set_epoch(None)
    # The Keras clock is a tf.Variable; only reset it if Keras was imported.
    keras_schedules = sys.modules.get("pypress.keras.schedules")
    if keras_schedules is not None:
        keras_schedules.set_epoch(None)
