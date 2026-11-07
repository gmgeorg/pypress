"""`ScheduledValue` reads the same values on both backends."""

import numpy as np
import pytest

from pypress.keras import schedules as keras_schedules
from pypress.torch import schedules as torch_schedules

_CONFIGS = [
    dict(start=9.0, end=3.0, duration_epochs=6, start_epoch=2, schedule="linear"),
    dict(start=0.0, end=5.0, duration_epochs=4, start_epoch=0, schedule="cosine"),
    dict(start=0.1, end=10.0, duration_epochs=5, start_epoch=3, schedule="exponential"),
]
# Before the start, during the ramp, at its end and well after it.
_EPOCHS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 50]


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda c: c["schedule"])
def test_values_match_at_every_epoch(config):
    """Values before start, during and after the ramp are equal."""
    keras_value = keras_schedules.ScheduledValue(**config)
    torch_value = torch_schedules.ScheduledValue(**config)
    for epoch in _EPOCHS:
        keras_schedules.set_epoch(epoch)
        torch_schedules.set_epoch(epoch)
        np.testing.assert_allclose(
            torch_value.value(),
            float(keras_value.value()),
            rtol=1e-5,
            err_msg=str(epoch),
        )


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda c: c["schedule"])
def test_stopped_clock_reads_end_in_both(config):
    """`set_epoch(None)` makes every value read its `end`."""
    keras_value = keras_schedules.ScheduledValue(**config)
    torch_value = torch_schedules.ScheduledValue(**config)
    keras_schedules.set_epoch(3)
    keras_schedules.set_epoch(None)
    torch_schedules.set_epoch(3)
    torch_schedules.set_epoch(None)
    assert float(keras_value.value()) == pytest.approx(config["end"], rel=1e-6)
    assert torch_value.value() == config["end"]
    assert torch_schedules.current_epoch() is keras_schedules.current_epoch() is None


def test_one_set_epoch_call_moves_the_keras_and_torch_clocks():
    """The Keras `set_epoch` also drives torch-side values."""
    config = _CONFIGS[0]
    torch_value = torch_schedules.ScheduledValue(**config)
    keras_schedules.set_epoch(5)
    assert torch_value.value() == pytest.approx(torch_value.value_at(5))
