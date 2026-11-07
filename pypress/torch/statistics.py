"""Population statistics of predictive state weights, in PyTorch.

Every PRESS penalty is a function of one or more of these statistics of a batch
of predictive state weights `W` (rows sum to 1, one column per predictive
state). This is a one-for-one port of `pypress.keras.statistics`; the names and
shapes are shared through `pypress.core.statistics`.

See CONTEXT.md for the domain terms used here.
"""

from typing import Callable, Dict

import torch

from ..core.statistics import names, shape  # noqa: F401


def safe_log(p: torch.Tensor, neg_inf: float = -1e9) -> torch.Tensor:
    """Numerically stable log(p) for p in [0, 1], e.g. softmax/simplex outputs.

    Only exact zeros (float32 softmax underflow) are replaced by `neg_inf`, with
    `torch.where` so no NaN/inf gradient flows back through `log` at p == 0. See
    `pypress.keras.statistics.safe_log`.
    """
    is_positive = p > 0
    safe_p = torch.where(is_positive, p, torch.ones_like(p))
    return torch.where(is_positive, torch.log(safe_p), torch.full_like(p, neg_inf))


def safe_xlogx(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable x * log(x) for x in [0, 1], e.g. entropy terms.

    Substitutes the *entire product* with 0 at x == 0, matching the limit
    `lim_{x->0} x*log(x) = 0`, so the gradient there is bounded. See
    `pypress.keras.statistics.safe_xlogx`.
    """
    is_positive = x > 0
    safe_x = torch.where(is_positive, x, torch.ones_like(x))
    return torch.where(is_positive, x * torch.log(safe_x), torch.zeros_like(x))


def mean_row_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Mean over observations of the row entropy of the predictive state weights."""
    return -safe_xlogx(weights).sum(dim=1).mean()


def state_sizes(weights: torch.Tensor) -> torch.Tensor:
    """State size per predictive state: the column sums of the weights."""
    return weights.sum(dim=0)


def trace_numerator(weights: torch.Tensor) -> torch.Tensor:
    """Per-state numerator of trace(H): the column sums of the squared weights."""
    return weights.square().sum(dim=0)


def state_shares(sizes: torch.Tensor) -> torch.Tensor:
    """Normalizes state sizes into state shares (a distribution over states)."""
    return sizes / sizes.sum().clamp_min(1e-8)


def entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of a distribution 'p' over states."""
    return -safe_xlogx(p).sum()


def degrees_of_freedom(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """trace(H) from its per-state numerator and denominator (state sizes).

    The terms are kept separate so a caller can average each across batches
    before taking the ratio: averaging the ratio itself would keep the
    small-batch bias of a ratio estimator.
    """
    return (numerator / denominator.clamp_min(1e-8)).sum()


# name -> function of the weights. The shape of each statistic is in
# `pypress.core.statistics`, shared with the Keras implementation.
_STATISTICS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean_row_entropy": mean_row_entropy,
    "state_sizes": state_sizes,
    "trace_numerator": trace_numerator,
    "trace_denominator": state_sizes,
}


def compute(name: str, weights: torch.Tensor) -> torch.Tensor:
    """Computes the statistic called 'name' on a batch of predictive state weights."""
    return _STATISTICS[name](weights)
