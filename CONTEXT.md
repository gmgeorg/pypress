# PRESS (Predictive State Smoothing)

A model that predicts an output by first mapping features to a probability distribution over a
small number of latent predictive states, then mixing per-state output parameters that do not
depend on the features. This glossary fixes the words used for those pieces.

## Predictive states

**Predictive state**:
A latent discrete variable `S` such that, given `S`, the output is conditionally independent of
the features. The model has a fixed number of them, `n_states` (written `K` in prose).
_Avoid_: cluster, component, mixture component, hidden unit

**State probability**:
`P(S = k | X)`, the probability that one observation is in predictive state `k`.
_Avoid_: activation, assignment, soft label

**Predictive state weights**:
An observation's state probabilities taken together as a vector over the `K` predictive states,
summing to 1; stacked over observations they form the matrix `W`.
_Avoid_: weights (alone; collides with trainable layer weights), embedding

**Predictive state simplex**:
The probability simplex over the `K` predictive states, where each observation's predictive state
weights live; also the mapping from features onto it.
_Avoid_: embedding, encoder

**Row entropy**:
The entropy of one observation's predictive state weights; low means the observation is sharply
assigned to few states. Maximal (`log K`) when the weights are uniform.
_Avoid_: sharpness, confidence

**State size**:
The total probability mass a predictive state receives across a population of observations (a
column sum of `W`).
_Avoid_: column mass, population weight, state count

**State share**:
A state size divided by the total, i.e. the fraction of the population assigned to that state.
_Avoid_: state usage, marginal, normalized state size

**Dead state**:
A predictive state whose state share is near zero across the population.
_Avoid_: unused state, empty state, collapsed state

## Outputs

**State-conditional mean**:
The per-state expected output `mu_k`, a parameter of the model that does not depend on the
features.
_Avoid_: state mean, cluster center, centroid

**State-conditional parameters**:
The general form of the state-conditional mean: a per-state vector of output-distribution
parameters (for example location and scale), independent of the features.
_Avoid_: state params, theta (in prose)

**Predictive mixture**:
The model's prediction: the state-conditional means (or parameters) averaged with each
observation's predictive state weights.
_Avoid_: weighted output, smoothed prediction

## Complexity

**Smoother matrix**:
The `n x n` PRESS kernel smoothing matrix `H = W · colnorm(W)^T` implied by the predictive state
weights, mapping observed outputs to fitted outputs. "Kernel" here is in the kernel-smoothing
(kernel density estimation) sense, unrelated to a neural network layer's `kernel` weights that
merely share the name.
_Avoid_: kernel (alone), similarity matrix, Gram matrix

**Degrees of freedom**:
The trace of the smoother matrix, `trace(H)`. It is 1 when all observations have identical
predictive state weights and at most `K`. Synonym: **effective number of states**.
_Avoid_: model complexity, trace(K) (`K` is the number of states)

**Penalty**:
A regularizer's scalar cost on a summary statistic of the predictive state weights (row entropy,
state shares, or degrees of freedom), added to the training loss.
_Avoid_: regularization loss, constraint

## Lifecycle

**Initialization from outputs**:
Seeding the state-conditional means from a one-dimensional mixture fitted to the training
outputs, so states start spread across the output range.
_Avoid_: y-init, GMM init, warm start

**Predictive clustering**:
Grouping observations by their predictive state weights, so members of a group share a predictive
distribution for the output, not similar features.
_Avoid_: clustering (alone), segmentation
