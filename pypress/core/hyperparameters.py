"""Declared regularizer hyperparameters.

A regularizer declares each of its scalar settings once, as a `Hyperparameter`,
and this module owns that scalar's whole lifecycle: deserializing it from a
config, validating it (over the whole schedule when it is a `ScheduledValue`)
and emitting it back into a config. Regularizers never call into `schedules`
directly. Reading a setting as a tensor is framework-specific and lives in
`pypress.keras.hyperparameters`; the torch regularizers read the float directly.
"""

import dataclasses
import math
from typing import Any

from . import schedules


@dataclasses.dataclass(frozen=True)
class Hyperparameter:
    """A declared scalar setting of a regularizer.

    Attributes:
      name: Constructor argument and config key.
      default: Value used when the argument is not given.
      low: Lower bound, or None for unbounded.
      high: Upper bound, or None for unbounded.
      low_inclusive: Whether `low` itself is allowed.
      high_inclusive: Whether `high` itself is allowed.
      schedulable: Whether a `ScheduledValue` is accepted.
      optional: Whether None is accepted (meaning "derive it at call time").
    """

    name: str
    default: Any = None
    low: float | None = None
    high: float | None = None
    low_inclusive: bool = True
    high_inclusive: bool = True
    schedulable: bool = True
    optional: bool = False

    def coerce(self, value):
        """Deserializes and validates 'value'; returns the value to store.

        Raises:
          ValueError: if the value (or any value its schedule can reach) is out
            of bounds, or is None for a required hyperparameter.
          TypeError: if a `ScheduledValue` is given for a non-schedulable one.
        """
        if isinstance(value, dict):
            value = self._deserialize(value)
        if value is None:
            if not self.optional:
                raise ValueError(f"'{self.name}' is required. Got None.")
            return None
        if isinstance(value, schedules.ScheduledValue):
            if not self.schedulable:
                raise TypeError(f"'{self.name}' cannot be scheduled. Got {value!r}.")
            for bound in (value.start, value.end):
                self._check(bound, f"over the whole schedule. Got {value!r}.")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{self.name}' must be a number. Got {value!r}.")
        value = float(value)
        self._check(value, f"Got {value}.")
        return value

    def _check(self, value: float, detail: str):
        """Raises ValueError if 'value' lies outside the declared interval."""
        too_low = self.low is not None and (
            value < self.low or (value == self.low and not self.low_inclusive)
        )
        too_high = self.high is not None and (
            value > self.high or (value == self.high and not self.high_inclusive)
        )
        if too_low or too_high or math.isnan(value):
            raise ValueError(f"'{self.name}' must be in {self.interval()} {detail}")

    def interval(self) -> str:
        """Human-readable interval, e.g. '(0, 1]'."""
        left = "[" if self.low_inclusive and self.low is not None else "("
        right = "]" if self.high_inclusive and self.high is not None else ")"
        low = "-inf" if self.low is None else f"{self.low:g}"
        high = "inf" if self.high is None else f"{self.high:g}"
        return f"{left}{low}, {high}{right}"

    @staticmethod
    def bounds(value) -> tuple:
        """Every value a stored setting can take: both endpoints if scheduled."""
        if isinstance(value, schedules.ScheduledValue):
            return (value.start, value.end)
        return () if value is None else (value,)

    @staticmethod
    def _deserialize(value: dict):
        """Builds a `ScheduledValue` from its config; frameworks override this."""
        return schedules.deserialize_scalar(value)

    @staticmethod
    def serialize(value):
        """Config representation of a stored setting."""
        return None if value is None else schedules.serialize_scalar(value)


L2 = Hyperparameter("l2", default=0.0, low=0.0)
EMA_DECAY = Hyperparameter(
    "ema_decay",
    default=None,
    low=0.0,
    high=1.0,
    high_inclusive=False,
    schedulable=False,
    optional=True,
)
