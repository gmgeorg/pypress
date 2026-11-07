"""Tests of the framework-free `pypress.core`; they need neither TensorFlow nor torch."""

import subprocess
import sys

import numpy as np
import pytest

from pypress.core import activations as core_act
from pypress.core import hyperparameters as hp
from pypress.core import schedules as core_sched
from pypress.core import statistics as core_stats


def test_core_imports_without_tensorflow_or_torch():
    """Core, analysis and clustering import with both frameworks unavailable."""
    code = (
        "import sys\n"
        "for blocked in ('tensorflow', 'keras', 'torch'):\n"
        "    sys.modules[blocked] = None\n"
        "import pypress.core.activations, pypress.core.hyperparameters\n"
        "import pypress.core.initializers, pypress.core.schedules\n"
        "import pypress.core.statistics, pypress.analysis, pypress.clustering\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_invert_validates_the_number_of_activations():
    """One activation per column is required, and each needs an inverse."""
    with pytest.raises(ValueError, match="one activation per column"):
        core_act.invert(np.ones((2, 3)), ["linear", "linear"])
    with pytest.raises(ValueError, match="Unknown activation"):
        core_act.invert(np.ones((2, 1)), "relu")


def test_core_clock_is_python_and_defaults_to_end():
    """With the clock stopped, a core ScheduledValue reads its end."""
    value = core_sched.ScheduledValue(start=0.0, end=1.0, duration_epochs=4)
    core_sched.set_epoch(None)
    assert value.value() == 1.0
    core_sched.set_epoch(2)
    try:
        assert value.value() == 0.5
        assert core_sched.current_epoch() == 2
    finally:
        core_sched.set_epoch(None)


def test_epoch_clock_restores_the_previous_epoch():
    """`epoch_clock` hands back a setter and restores the clock, even on error."""
    core_sched.set_epoch(3)
    try:
        with pytest.raises(RuntimeError):
            with core_sched.epoch_clock() as clock:
                clock(7)
                assert core_sched.current_epoch() == 7
                raise RuntimeError
        assert core_sched.current_epoch() == 3
        with core_sched.epoch_clock(epoch=1):
            assert core_sched.current_epoch() == 1
        assert core_sched.current_epoch() == 3
    finally:
        core_sched.set_epoch(None)


def test_core_hyperparameter_deserializes_core_schedules():
    """A serialized schedule becomes a core ScheduledValue in core."""
    serialized = core_sched.serialize_scalar(
        core_sched.ScheduledValue(start=1.0, end=2.0, duration_epochs=3)
    )
    coerced = hp.L2.coerce(serialized)
    assert type(coerced) is core_sched.ScheduledValue
    assert hp.L2.serialize(coerced) == serialized


@pytest.mark.parametrize(
    "package, blocked, extra",
    [("pypress.keras", "tensorflow", "tf"), ("pypress.torch", "torch", "torch")],
)
def test_backend_package_explains_how_to_install_a_missing_framework(
    package, blocked, extra
):
    """Importing a backend without its framework names the extra to install."""
    code = (
        "import sys\n"
        f"sys.modules[{blocked!r}] = None\n"
        "try:\n"
        f"    import {package}\n"
        "except ImportError as error:\n"
        "    print(error)\n"
        "else:\n"
        "    raise SystemExit('import unexpectedly succeeded')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert f"pypress[{extra}]" in result.stdout


def test_core_statistic_shapes_and_names():
    """Shapes depend only on K."""
    assert core_stats.shape("state_sizes", 4) == (4,)
    assert core_stats.shape("mean_row_entropy", 4) == ()
    assert set(core_stats.names()) == {
        "mean_row_entropy",
        "state_sizes",
        "trace_numerator",
        "trace_denominator",
    }


def test_core_scalar_serialization_round_trips_and_rejects_other_classes():
    """Floats stay floats; ScheduledValues survive; other dicts are rejected."""
    assert core_sched.serialize_scalar(0.5) == 0.5
    value = core_sched.ScheduledValue(
        start=9.0, end=3.0, duration_epochs=7, schedule="cosine"
    )
    restored = core_sched.deserialize_scalar(core_sched.serialize_scalar(value))
    assert restored.get_config() == value.get_config()
    with pytest.raises(ValueError):
        core_sched.deserialize_scalar({"class_name": "Other", "config": {}})


def test_core_hyperparameters_validate_bounds_and_schedules():
    """Bounds are checked at both ends of a schedule."""
    assert hp.L2.coerce(1) == 1.0
    with pytest.raises(ValueError, match="'l2' must be in"):
        hp.L2.coerce(core_sched.ScheduledValue(start=-1.0, end=1.0, duration_epochs=2))
    with pytest.raises(TypeError, match="cannot be scheduled"):
        hp.EMA_DECAY.coerce(core_sched.ScheduledValue(0.1, 0.5, 2))
    with pytest.raises(ValueError, match="required"):
        hp.Hyperparameter("x").coerce(None)
