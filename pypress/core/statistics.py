"""Names and shapes of the population statistics behind the PRESS penalties.

Every penalty is a function of one or more statistics of a batch of predictive
state weights `W`. Each statistic has a name and a shape that depends only on
`K`, the number of predictive states, so a cross-batch estimator can allocate
its running average before the first batch is seen. The functions computing the
statistics live in the framework packages (`pypress.keras.statistics`,
`pypress.torch.statistics`); this module is what they agree on.
"""

from typing import Callable, Dict, Tuple

# name -> shape as a function of K
STATISTIC_SHAPES: Dict[str, Callable[[int], Tuple[int, ...]]] = {
    "mean_row_entropy": lambda n_states: (),
    "state_sizes": lambda n_states: (n_states,),
    "trace_numerator": lambda n_states: (n_states,),
    "trace_denominator": lambda n_states: (n_states,),
}


def shape(name: str, n_states: int) -> Tuple[int, ...]:
    """Shape of the statistic called 'name' for 'n_states' predictive states."""
    return tuple(STATISTIC_SHAPES[name](n_states))


def names() -> Tuple[str, ...]:
    """Names of all known statistics."""
    return tuple(STATISTIC_SHAPES)
