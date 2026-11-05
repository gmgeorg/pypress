"""Shared pytest fixtures."""

import pytest

from pypress.keras import schedules


@pytest.fixture(autouse=True)
def _reset_training_clock():
    """Stops the shared training clock after every test, so tests stay independent."""
    yield
    schedules.set_epoch(None)
