import numpy as np
import pytest
import tensorflow as tf

from pypress.keras.schedules import (
    ScheduledValue,
    as_tensor,
    current_epoch,
    deserialize_scalar,
    serialize_scalar,
    set_epoch,
)


def test_linear_schedule_endpoints_and_midpoint():
    """Linear interpolates evenly and clamps outside the window."""
    sched = ScheduledValue(start=0.0, end=1.0, duration_epochs=10)
    assert sched.value_at(0) == 0.0
    np.testing.assert_allclose(sched.value_at(5), 0.5)
    assert sched.value_at(10) == 1.0
    # Held after the window, and clamped before it.
    assert sched.value_at(50) == 1.0
    assert sched.value_at(-3) == 0.0


def test_start_epoch_delays_the_ramp():
    """Nothing moves before start_epoch."""
    sched = ScheduledValue(start=0.0, end=1.0, duration_epochs=4, start_epoch=3)
    assert sched.value_at(0) == 0.0
    assert sched.value_at(3) == 0.0
    np.testing.assert_allclose(sched.value_at(5), 0.5)
    assert sched.value_at(7) == 1.0


def test_decay_direction_is_just_reversed_endpoints():
    """start > end decays; no separate direction flag is needed."""
    sched = ScheduledValue(start=1.0, end=0.1, duration_epochs=10)
    assert sched.value_at(0) == 1.0
    np.testing.assert_allclose(sched.value_at(5), 0.55)
    np.testing.assert_allclose(sched.value_at(10), 0.1)


def test_cosine_schedule_is_monotone_and_eases():
    """Cosine hits the same endpoints and midpoint but eases in and out."""
    sched = ScheduledValue(start=0.0, end=1.0, duration_epochs=10, schedule="cosine")
    values = [sched.value_at(e) for e in range(11)]
    assert values[0] == 0.0
    np.testing.assert_allclose(values[-1], 1.0, atol=1e-9)
    np.testing.assert_allclose(sched.value_at(5), 0.5, atol=1e-9)
    assert all(b >= a for a, b in zip(values, values[1:]))
    # Eases in: slower than linear over the first quarter.
    assert sched.value_at(2) < 0.2


def test_exponential_schedule_is_geometric():
    """Exponential interpolates geometrically -- useful for annealing a target."""
    sched = ScheduledValue(
        start=9.0, end=1.0, duration_epochs=4, schedule="exponential"
    )
    np.testing.assert_allclose(sched.value_at(0), 9.0)
    np.testing.assert_allclose(sched.value_at(2), 3.0, atol=1e-6)  # sqrt(9 * 1)
    np.testing.assert_allclose(sched.value_at(4), 1.0, atol=1e-6)


def test_exponential_rejects_non_positive_endpoints():
    """Geometric interpolation cannot start or end at 0."""
    with pytest.raises(ValueError, match="endpoints > 0"):
        ScheduledValue(start=0.0, end=1.0, duration_epochs=4, schedule="exponential")


def test_invalid_arguments_rejected():
    """Bad schedule name, duration and start_epoch all fail loudly."""
    with pytest.raises(ValueError, match="schedule"):
        ScheduledValue(start=0.0, end=1.0, duration_epochs=4, schedule="quadratic")
    with pytest.raises(ValueError, match="duration_epochs"):
        ScheduledValue(start=0.0, end=1.0, duration_epochs=0)
    with pytest.raises(ValueError, match="start_epoch"):
        ScheduledValue(start=0.0, end=1.0, duration_epochs=4, start_epoch=-1)


def test_unscheduled_value_reads_end_not_start():
    """A forgotten scheduler must degrade to the enforced penalty, not to none.

    Reading `start` would mean that omitting `RegularizerScheduler` silently
    disables the regularizer for the whole run (a warm-up starts at l2=0).
    Reading `end` degrades to the un-annealed penalty instead.
    """
    sched = ScheduledValue(start=0.0, end=1.0, duration_epochs=10)
    assert current_epoch() is None
    assert float(sched.value()) == 1.0

    set_epoch(0)
    assert float(sched.value()) == 0.0
    set_epoch(5)
    np.testing.assert_allclose(float(sched.value()), 0.5)
    set_epoch(None)
    assert float(sched.value()) == 1.0


@pytest.mark.parametrize("schedule", ["linear", "cosine", "exponential"])
def test_graph_value_matches_python_value_at(schedule):
    """value() computed in-graph agrees with value_at() for every epoch."""
    sched = ScheduledValue(
        start=9.0, end=1.0, duration_epochs=6, start_epoch=2, schedule=schedule
    )
    for epoch in range(12):
        set_epoch(epoch)
        np.testing.assert_allclose(
            float(sched.value()), sched.value_at(epoch), rtol=1e-5, err_msg=str(epoch)
        )


def test_clock_read_propagates_into_a_traced_function():
    """The whole point: clock updates must survive tf.function tracing."""
    sched = ScheduledValue(start=1.0, end=5.0, duration_epochs=4)

    @tf.function
    def penalty():
        return as_tensor(sched) * 2.0

    set_epoch(0)
    np.testing.assert_allclose(float(penalty()), 2.0)
    set_epoch(4)
    np.testing.assert_allclose(float(penalty()), 10.0)


def test_as_tensor_accepts_plain_floats():
    """as_tensor is the single read path for both floats and ScheduledValues."""
    np.testing.assert_allclose(float(as_tensor(0.25)), 0.25)
    sched = ScheduledValue(start=0.25, end=1.0, duration_epochs=2)
    set_epoch(0)
    np.testing.assert_allclose(float(as_tensor(sched)), 0.25)


def test_deprecated_variable_and_assign_epoch_still_work():
    """The pre-clock API warns but keeps its meaning."""
    sched = ScheduledValue(start=0.0, end=1.0, duration_epochs=10)
    with pytest.warns(DeprecationWarning, match="value"):
        assert float(sched.variable) == 1.0
    with pytest.warns(DeprecationWarning, match="set_epoch"):
        assert sched.assign_epoch(5) == 0.5
    np.testing.assert_allclose(float(sched.value()), 0.5)
    with pytest.warns(DeprecationWarning):
        assert sched.was_scheduled


def test_scalar_serialization_roundtrip():
    """Floats stay floats; ScheduledValues survive a config round-trip."""
    assert serialize_scalar(0.5) == 0.5
    assert deserialize_scalar(0.5) == 0.5

    sched = ScheduledValue(
        start=9.0, end=3.0, duration_epochs=7, start_epoch=2, schedule="cosine"
    )
    restored = deserialize_scalar(serialize_scalar(sched))
    assert isinstance(restored, ScheduledValue)
    assert restored.get_config() == sched.get_config()
    assert restored.value_at(5) == sched.value_at(5)
