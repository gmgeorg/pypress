# Changelog

All notable changes `pypress` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## pypress v0.3.0 - Aug 29, 2026

### Added

* `StateSizeEntropy` regularizer: penalizes predictive states carrying ~0 weight
  across the population, via the normalized KL divergence of the state-size
  marginal from uniform usage. Catches over-provisioned `K`, which `Uniform` and
  `DegreesOfFreedom` are both blind to — a `K = 50` model with 45 dead states has
  the same kernel trace and mean row entropy as an honest `K = 5` model
* `MinStateSize` regularizer: a per-state floor in population units ("at least 5%
  of observations per state"). Exact where `StateSizeEntropy` is only a proxy —
  one scalar cannot encode `K` separate floors — and independent of `K`.
  Infeasible floors (`min_share * K > 1`) raise instead of training against an
  impossible target
* `ema_decay` on every regularizer: evaluates the penalty on bias-corrected
  moving averages of its summary statistics instead of single-minibatch
  estimates, which are biased upward because the penalties are convex. More
  batches per epoch does not remove that bias; a larger batch or this does.
  Gradients still flow through the current batch
* `pypress.keras.schedules.ScheduledValue` and
  `pypress.keras.callbacks.RegularizerScheduler`: anneal any regularizer scalar
  over epochs rather than enforcing it from epoch 0. Values are `tf.Variable`
  backed, since mutating a plain Python attribute mid-training is captured at
  trace time and silently ignored. A `ScheduledValue` initializes to `end`, so
  omitting the callback costs the annealing, not the regularization

### Changed

* `TargetEntropy` gained `entropy_fraction`, expressing the target as a fraction
  of `K` (`log(entropy_fraction * K)`) rather than in absolute nats, so it need
  not be recomputed when `n_states` changes. Mutually exclusive with `target`
* `PRESS` constructs its `PredictiveStateSimplex` / `PredictiveStateMeans`
  sub-layers in `__init__` rather than `build()`, so Keras tracks them from
  construction and they are reachable before the first call. Rebuilding is now a
  no-op rather than silently replacing trained sub-layers. Added
  `predictive_state_simplex` / `predictive_state_means` properties and
  `compute_output_shape`
* `utils.tr_kernel_terms` exposes the kernel trace's per-state numerator and
  denominator, so each can be averaged across batches before taking the ratio.
  `utils.tr_kernel` is unchanged and now built on it

### Fixed

* `PRESS.get_config` emitted live Keras objects for the nested sub-layer kwargs,
  so a model carrying a regularizer or initializer could not be saved at all.
  Now serialized properly, with a matching `from_config`

### Documentation

* README gained a `## Regularizers` section: what each regularizer controls,
  what each sees at initialization, how to choose `l2`, which knobs to schedule
  and in which direction, and batch-size / `ema_decay` guidance

### Note on tuning

At initialization the simplex weights are uniform, so `trace(K)` is exactly 1
regardless of `n_states` and grows only as states differentiate. A
`DegreesOfFreedom` target below `n_states` therefore caps differentiation from
the first step rather than pruning states later. Warming up its `l2` from 0 is
usually preferable. `StateSizeEntropy` and `MinStateSize` are exactly zero at
initialization and need no warm-up.

## pypress v0.2.4 - Aug 27, 2026

### Fixed

* `TargetEntropy`/`Uniform` entropy penalty was not scale-invariant in `K` (the
  number of states/columns): the squared deviation from target entropy had a
  dynamic range that grew as `log(K)^2`, so the effective regularization
  strength silently depended on `K`. The `(target - mean_entropy)` deviation
  is now normalized by `log(K)` (the maximum possible entropy) before
  squaring, keeping the penalty bounded in `[0, l2]` regardless of `K`. This
  changes penalty magnitudes for existing `l2` values tuned under the old
  formula — retune if you rely on absolute penalty scale

### Changed

* `DegreesOfFreedom` penalty changed from L1 (`l1 * |target - df(kernel)|`) to
  squared L2 (`l2 * (target - df(kernel)) ** 2`) for smoother gradients near
  the target and consistency with `TargetEntropy`/`Uniform`. The
  constructor/config parameter was renamed from `l1` to `l2` to reflect this.
  `UniformAndDegreesOfFreedomRegularizer`'s `dof_l1` parameter was renamed to
  `dof_l2` to match

## pypress v0.2.2 - Aug 21, 2026

### Changed

* Bumped transitive/dev dependencies (`werkzeug`, `urllib3`, `setuptools`, `requests`,
  `pytest`, `pillow`, `keras`, `idna`, `markdown`, `pygments`, `wheel`, `protobuf`) to
  their minimum patched versions, resolving all outstanding Dependabot alerts
* Relaxed dev-dependency pins (`pytest`, `seaborn`, `tqdm`, `scikit-learn`, `scipy`,
  `fastcluster`, `shap`, `pydot`) from caret (`^`) to open-ended (`>=`) version floors

## pypress v0.2.1 - Aug 21, 2026

### Fixed

* `PRESS.state_conditional_means` raised `AttributeError` on every access due to a
  typo'd internal attribute reference
* `PredictiveStateSimplex.get_config()` inherited `Dense`'s config, which emitted
  `units`/`activation` keys incompatible with `PredictiveStateSimplex.__init__()`;
  saving/loading this layer (or a `Sequential` containing it) now round-trips correctly
* `PredictiveStateMeans`/`PredictiveStateParams` with an array-valued `init_values`
  failed to reload after a full `.keras` model save/load, since the raw `np.ndarray`
  didn't survive Keras's JSON-based config serialization
* `activations.get_inverse_activation` docstring incorrectly claimed `"softmax"` had
  no inverse

## pypress v0.2.0 - Aug 21, 2026

### Changed

* Restructured entropy regularizers: `TargetEntropy` is now the base class (dynamic
  default target `log(0.5 * K)`, clamped to a minimum of 1 active state); `Uniform` is
  a special case (target `log(K)`) and no longer accepts a `target_entropy` override —
  use `TargetEntropy` directly for a custom target
* Entropy penalty changed from L1 to squared L2 for smoother gradients near the
  target; the constructor/config parameter on `TargetEntropy` and `Uniform` was
  renamed from `l1` to `l2` to reflect this (`DegreesOfFreedom` is unaffected — it
  remains L1). `UniformAndDegreesOfFreedomRegularizer`'s `uniform_l1` parameter was
  renamed to `uniform_l2` to match
* Added `utils.safe_log` and `utils.safe_xlogx` for numerically stable
  `log(p)`/`x * log(x)` on softmax/simplex outputs, avoiding both the value-floor
  bias and gradient spikes of the previous `log(p + eps)` pattern

## pypress v0.1.2 - Dec 29, 2025

### Fixed

* `PredictiveStateParams` initialization: `initialize_from_y(return_params=True)` now
  returns a single 2D `np.ndarray` of shape `(n_params_per_state, n_states)` instead of
  a `(means, stds)` tuple, and `init_values` only accepts that 2D array format —
  removing the previous, error-prone support for mismatched list/tuple lengths

## pypress v0.1.0 - Dec 29, 2025

### Added

* `pypress.clustering.GaussianMixture1D`: a lightweight 1D Gaussian Mixture Model
  estimator (scikit-learn-style API) for state-specific, data-driven layer
  initialization
* `utils.initialize_from_y()`: initializes `PredictiveStateMeans`/`PredictiveStateParams`
  layers directly from training data via GMM clustering
* `utils.kernel_matrix()`: computes the full kernel matrix from PRESS layer weights
* `PredictiveStateMeansInitializer`/`PredictiveStateParamsInitializer` now accept
  `init_values` on the original (post-activation) scale and convert to logits via the
  new `activations.get_inverse_activation()` registry, and support state-specific
  initialization via `(units, n_states)`-shaped arrays

## pypress v0.0.7 - Dec 23, 2025

### Added

* New `PredictiveStateParams` layer for learning state-conditional distribution parameters
  (e.g., [mean, variance]) with flexible per-parameter activation functions
* Comprehensive documentation for all `PredictiveStateMeans` and `PredictiveStateParams` methods
* Test coverage for classification tasks with sigmoid/softmax activations (binary and multi-class)
* Test coverage for `PredictiveStateParams` with sigmoid activation for probabilistic outputs

### Changed

* Enhanced documentation explaining relationship between `PredictiveStateMeans` (mixing) vs
  `PredictiveStateParams` (broadcasting)

## pypress v0.0.1 - Nov 28, 2021

Initial release of `pypress`.

## TEMPLATE: pypress vX.Y.Z

### Added

* ...

### Changed

* ...

### Deprecated

* ...

### Removed

* ...

### Fixed

* ...
