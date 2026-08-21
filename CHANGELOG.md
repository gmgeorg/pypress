# Changelog

All notable changes `pypress` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
