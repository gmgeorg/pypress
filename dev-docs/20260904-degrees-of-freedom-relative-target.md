# DegreesOfFreedom normalization and K-scale invariance

## Finding

`DegreesOfFreedom` originally used an absolute effective-state target:

```text
l2 * (trace(Kernel) - target) ** 2
```

For row-stochastic PRESS weights, the trace ranges approximately from one to
`K` (more exactly, it cannot exceed `min(batch_size, K)`). Therefore a fixed
`l2` has a larger possible effect as `K` grows when the intended target itself
scales with `K`.

This is not the same situation as row entropy. Entropy has the natural maximum
`log(K)`, while DoF is a count: `target=4` means four effective states and must
remain a valid, useful absolute objective when the architecture changes.

## Decision

Preserve the existing absolute `target` API. Add an opt-in `normalize` boolean
for the scale-invariant penalty form:

```text
penalty = l2 * ((trace(Kernel) - target) / K) ** 2
```

`target` continues to accept a `ScheduledValue` and stays in effective-state
units. The combined convenience regularizer exposes the same option as
`dof_normalize`.

The normalizer is `K`, not batch size. Batch size would make the penalty’s scale
depend on the data loader rather than the architecture. The finite-K lower trace
bound of one means endpoint behavior cannot be exactly invariant (for example,
one effective state is `1/K` rather than zero), but equal fractional deviations
from their absolute targets receive exactly equal penalties.

## Scope and complementarity

This change does not make DoF detect dead configured states. A hard-assignment
model using five states has trace five whether it has `K=5` or `K=50`; DoF
continues to correctly treat those as equally smooth. `StateSizeEntropy` or
`MinStateSize` remains responsible for penalizing the 45 unused states in the
over-provisioned model.

## Verification

The test suite asserts that a 10-state model with trace three and target five,
and a 20-state model with trace six and target ten, produce the same penalty
with `normalize=True`: both have normalized deviation `-0.2`, and therefore
penalty `0.04` at `l2=1`. It also covers a scheduled absolute-target
serialization round trip with normalization enabled.
