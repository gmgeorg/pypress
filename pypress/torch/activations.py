"""Activation functions for PRESS layers, in PyTorch.

An activation is given by name, as a torch function, or as an `nn.Module`;
a table of state-conditional parameters takes one for the whole table or one
per column, as `pypress.keras.activations` does. The names that have an inverse
(`pypress.core.activations`) can be used with `init_values`; `relu`, `swish`/
`silu` and `gelu` are valid activations without one.

A single activation applies to the whole (K, P) table, so `"softmax"`
normalizes across the P columns of each state; per-column activations apply to
one column each.
"""

from typing import Callable, List, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F

# An activation given by name, as a function, or as a module.
Activation = Union[str, Callable[[torch.Tensor], torch.Tensor], nn.Module]
# One activation for a whole table, or one per column.
ActivationSpec = Union[Activation, Sequence[Activation]]

# Slope of the Keras "leaky_relu", which the inverse in `pypress.core` assumes.
_LEAKY_SLOPE = 0.2

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "linear": lambda x: x,
    "sigmoid": torch.sigmoid,
    "softplus": F.softplus,
    "tanh": torch.tanh,
    "exponential": torch.exp,
    "leaky_relu": lambda x: F.leaky_relu(x, _LEAKY_SLOPE),
    "elu": F.elu,
    "softsign": F.softsign,
    "softmax": lambda x: F.softmax(x, dim=-1),
    "selu": F.selu,
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "gelu": F.gelu,
}

# Functions that have a registered name.
_FUNCTION_NAMES = {
    torch.sigmoid: "sigmoid",
    F.sigmoid: "sigmoid",
    F.softplus: "softplus",
    torch.tanh: "tanh",
    F.tanh: "tanh",
    torch.exp: "exponential",
    F.elu: "elu",
    F.softsign: "softsign",
    F.selu: "selu",
    F.relu: "relu",
    F.silu: "silu",
    F.gelu: "gelu",
}


def _module_name(module: nn.Module) -> Optional[str]:
    """The registered name of an activation module, or None if it has none."""
    if isinstance(module, nn.Identity):
        return "linear"
    if isinstance(module, nn.Sigmoid):
        return "sigmoid"
    if isinstance(module, nn.Tanh):
        return "tanh"
    if isinstance(module, nn.Softsign):
        return "softsign"
    if isinstance(module, nn.SELU):
        return "selu"
    if isinstance(module, nn.ReLU):
        return "relu"
    if isinstance(module, nn.SiLU):
        return "silu"
    if isinstance(module, nn.GELU):
        return "gelu" if module.approximate == "none" else None
    if isinstance(module, nn.Softplus):
        return "softplus" if module.beta == 1.0 and module.threshold == 20.0 else None
    if isinstance(module, nn.ELU):
        return "elu" if module.alpha == 1.0 else None
    if isinstance(module, nn.LeakyReLU):
        return "leaky_relu" if module.negative_slope == _LEAKY_SLOPE else None
    if isinstance(module, nn.Softmax):
        return "softmax" if module.dim in (-1, 1) else None
    return None


def is_single(spec: ActivationSpec) -> bool:
    """Whether 'spec' is one activation (rather than one per column)."""
    return isinstance(spec, (str, nn.Module)) or callable(spec)


def to_callable(activation: Activation) -> Callable[[torch.Tensor], torch.Tensor]:
    """The activation as a function of a tensor: names are looked up.

    Raises:
      ValueError: on an unknown name.
      TypeError: if 'activation' is neither a name, a function nor a module.
    """
    if isinstance(activation, str):
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Supported activations: "
                f"{sorted(ACTIVATIONS)}."
            )
        return ACTIVATIONS[activation]
    if callable(activation):
        return activation
    raise TypeError(
        f"An activation must be a name, a function or a module. Got {activation!r}."
    )


def name(activation: Activation) -> Optional[str]:
    """The registered name of 'activation', or None if it has none.

    Resolves names, torch functions and activation modules, so an initializer
    can look up the inverse of each.
    """
    if isinstance(activation, str):
        return activation
    if isinstance(activation, nn.Module):
        return _module_name(activation)
    return _FUNCTION_NAMES.get(activation)


def names(spec: ActivationSpec, n_columns: int) -> List[str]:
    """One activation name per column, for inverting original-scale values.

    Raises:
      ValueError: if a column's activation has no name, e.g. a custom module.
    """
    columns = [spec] * n_columns if is_single(spec) else list(spec)
    resolved = [name(a) for a in columns]
    if any(n is None for n in resolved):
        raise ValueError(
            "'init_values' needs every activation to be a known, invertible "
            f"activation (a name, torch function or activation module). Got {spec!r}."
        )
    return resolved


def validate(spec: ActivationSpec, n_columns: int) -> ActivationSpec:
    """Checks 'spec' against a table with 'n_columns' columns; returns it.

    Raises:
      ValueError: on an unknown name, or when a sequence does not have one
        activation per column.
    """
    if is_single(spec):
        to_callable(spec)
        return spec
    columns = list(spec)
    if len(columns) != n_columns:
        raise ValueError(
            f"A sequence of activations needs one per column ({n_columns}). "
            f"Got {len(columns)}."
        )
    for activation in columns:
        to_callable(activation)
    return columns


def apply(table: torch.Tensor, spec: ActivationSpec) -> torch.Tensor:
    """Applies 'spec' to a (K, P) table: to the whole table, or column by column."""
    if is_single(spec):
        return to_callable(spec)(table)
    columns = table.split(1, dim=1)
    return torch.cat([to_callable(a)(c) for a, c in zip(spec, columns)], dim=1)
