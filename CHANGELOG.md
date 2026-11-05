# Changelog

All notable changes `pypress` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## pypress v0.5.3 - Nov 5, 2026

### Fixed

* **`DegreesOfFreedom` with `ema_decay` could collapse the model onto one
  state.** The penalty was differentiated at the smoothed trace(H). For a
  shrinking state, the running averages still held its earlier size, so the
  gradient shrank it further, and states died one after another. In a 12-state
  digits model this happened at every `l2` tried, down to 0.05. The smoothed
  trace(H) now sets the penalty's value, size and sign; the batch trace(H) sets
  the gradient's direction. The same model now reaches its target of 10
  effective states.

* `ema_decay` running averages advanced on calls that leave `training` unset,
  such as `PRESS.predictive_states(x)` or `model(x)`. Only training calls (as in
  `fit`) advance them now.

* `analysis.agg_data_by_state` accepts numpy arrays, as the module documents.
  Two arrays give an array; a DataFrame input labels the result. It raises
  `ValueError` when data and weights have different numbers of rows.

## pypress v0.5.2 - Nov 5, 2026

### Changed

* **Regularizer strength no longer depends on the batch size (breaking).**
  Keras divides an `activity_regularizer`'s loss by the batch size, which suits
  its built-in regularizers (sums over the batch) but not PRESS penalties
  (already batch statistics). Every PRESS penalty was therefore applied at
  `l2 / batch_size`: 1/256 of its documented strength at `batch_size=256`. The
  regularizers now cancel that division, so the loss Keras adds is exactly the
  penalty, bounded by `l2` as documented, at any batch size.

  Models trained with existing settings are regularized `batch_size` times more
  strongly after upgrading. To keep the old behaviour, divide each `l2` by the
  batch size you train with.

### Added

* `penalty(x)` on every regularizer and on `CombinedRegularizer`: the amount
  the regularizer adds to the loss for a batch of predictive state weights.
  Calling a regularizer directly now returns that value times the batch size,
  which is what Keras expects of an activity regularizer.

## pypress v0.5.0 - Nov 5, 2026

### Added

* `pypress.analysis.degrees_of_freedom(weights)`: trace(H) in numpy, without
  forming the O(n^2) smoother matrix.

### Changed

* `PredictiveStateMeans` and `PredictiveStateParams` share one table of
  state-conditional parameters, so both accept the same `init_values` forms (a
  scalar, one value per column, or a `(P, K)` array) and the same activations
  (a name, a Keras function or `Activation` layer, or one per column).
  Initializing through an `Activation` layer or function now works when it has
  a known inverse.

* A single activation on `PredictiveStateParams` now applies to the whole
  table, as it does for `PredictiveStateMeans`. This only changes non-elementwise
  activations: a single `"softmax"` now normalizes across the parameters of each
  state, where it used to return 1 for every parameter.

* `pypress.utils` is split by what calls it. Post-hoc analysis is in
  `pypress.analysis` (`state_size`, `col_normalize`, `smoother_matrix`,
  `agg_data_by_state`). `safe_log`/`safe_xlogx` are in
  `pypress.keras.statistics`, and `initialize_from_y` is in
  `pypress.keras.initializers`. The old `pypress.utils` names still work.

* The `PredictiveStateMeansInitializer` and `PredictiveStateParamsInitializer`
  classes are now thin subclasses of `StateConditionalInitializer`, with
  `get_config()`. They take the number of states from the weight shape.

### Deprecated

* `pypress.utils.tf_state_size`, `tf_col_normalize`, `tr_kernel_terms` and
  `tr_kernel`: use `pypress.keras.statistics` (in-graph) or
  `pypress.analysis.degrees_of_freedom` (numpy).

### Fixed

* `PredictiveStateParams(init_values=[0.0, 1.0])` (one value per parameter)
  raised `ValueError`, although it was documented.

* `PredictiveStateParams.compute_output_shape` raised `TypeError` before the
  layer was built.

* An `Activation` layer object passed to `PRESS`, `PredictiveStateMeans` or
  `PredictiveStateParams` made the config non-JSON-serializable, so the model
  could not be saved.

* Wrong-shaped `init_values` for `PredictiveStateMeans` raised `AssertionError`;
  it is now a `ValueError`, as for `PredictiveStateParams`.

* `initialize_from_y` docstring described multi-output y wrongly: all outputs
  are pooled into one fit, and every output gets the same per-state means.

## pypress v0.4.0 - Nov 5, 2026

### Changed

* Regularizers are now one pipeline: a named population statistic
  (`pypress.keras.statistics`), an optional cross-batch estimator, and a pure
  penalty function. Scalar settings are declared once as `Hyperparameter`s
  (`pypress.keras.hyperparameters`), which own validation (over the whole
  schedule), in-graph reading and serialization.

* `get_config()` of every regularizer now emits all of its settings, using
  `None` for unset optional ones.

* `ScheduledValue` computes its value in-graph from one shared training clock
  (`schedules.training_clock()`). `RegularizerScheduler` advances only that
  clock, so a scheduled value held anywhere — a layer, a custom loss, a
  closure — is driven without being discovered. The clock is restored when
  `fit` ends.

* `CombinedRegularizer` takes regularizer instances and serializes each child
  through Keras.

* Invalid hyperparameters raise `ValueError` (previously `AssertionError` for
  `DegreesOfFreedom.target`). `TargetEntropy.target` must be >= 0 and `l2` >= 0.

### Deprecated

* `CombinedRegularizer` with `(class, kwargs)` tuples (and the
  `regularizer_tuples=` keyword). Configs in the old format still load.

* `UniformAndDegreesOfFreedomRegularizer`: use
  `CombinedRegularizer([Uniform(...), DegreesOfFreedom(...)])`.

* `ScheduledValue.variable`, `.assign_epoch()`, `.was_scheduled`: use
  `.value()`, `schedules.set_epoch()`, `schedules.current_epoch()`.

* `callbacks.collect_scheduled_values` and
  `RegularizerScheduler.scheduled_values`.

* `DegreesOfFreedom(df=...)` now warns with `DeprecationWarning` (was
  `UserWarning`).

### Removed

* `RegularizerScheduler` no longer warns when a model has no `ScheduledValue`,
  since it no longer searches the model.

### Fixed

* `ema_decay` averages only advance on training batches; `evaluate` and
  `predict` no longer contaminate them.

* `ema_decay` state is tracked by the layer it regularizes, so it appears in
  `model.variables` and is saved and checkpointed. Models saved before this
  load with fresh averages.

* Applying one smoothed regularizer to two layers raises a clear `ValueError`
  instead of crashing on a shape mismatch, or silently mixing statistics when
  the shapes agree.

* `CombinedRegularizer.get_config()` with a `ScheduledValue` in a child is now
  JSON-serializable.

* `UniformAndDegreesOfFreedomRegularizer.from_config` no longer leaves raw
  config dicts in its attributes.

* `RegularizerScheduler` docstring said unscheduled values stay at `start`;
  they read `end`.

## pypress v0.3.0 - Aug 29, 2026

### Added

* `DegreesOfFreedom(normalize=True)`: an opt-in K-normalized penalty for the
  kernel trace. It preserves the absolute effective-state `target` while
  dividing its deviation by `n_states`, so an `l2` tuned for a fractional
  effective-state deviation transfers across state counts

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
