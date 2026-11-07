"""Tests of the Python epoch clock used by the torch regularizers."""

import pytest

from pypress.torch import schedules


def test_value_follows_the_clock_and_defaults_to_end():
    """A stopped clock reads `end`; set_epoch moves the value."""
    value = schedules.ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    assert value.value() == 1.0
    schedules.set_epoch(0)
    assert value.value() == 0.0
    schedules.set_epoch(2)
    assert value.value() == 0.5
    schedules.set_epoch(None)
    assert value.value() == 1.0


def test_epoch_clock_restores_the_previous_epoch():
    """`with epoch_clock()` leaves the clock as it found it."""
    schedules.set_epoch(1)
    with schedules.epoch_clock() as clock:
        clock(9)
        assert schedules.current_epoch() == 9
    assert schedules.current_epoch() == 1


def test_as_float_reads_floats_and_schedules():
    """`as_float` is the single read path for settings."""
    assert schedules.as_float(0.25) == 0.25
    value = schedules.ScheduledValue(start=1.0, end=3.0, duration_epochs=2)
    schedules.set_epoch(1)
    assert schedules.as_float(value) == 2.0


def test_scheduled_value_validates_its_arguments():
    """The core validation applies."""
    with pytest.raises(ValueError, match="duration_epochs"):
        schedules.ScheduledValue(start=0.0, end=1.0, duration_epochs=0)
    with pytest.raises(ValueError, match="exponential"):
        schedules.ScheduledValue(0.0, 1.0, 2, schedule="exponential")
