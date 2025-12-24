# Changelog

All notable changes `pypress` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## pypress v0.0.7 - Dec 23, 2025

### Added

* New `PredictiveStateParams` layer for learning state-conditional distribution parameters (e.g., [mean, variance]) with flexible per-parameter activation functions
* Comprehensive documentation for all `PredictiveStateMeans` and `PredictiveStateParams` methods
* Test coverage for classification tasks with sigmoid/softmax activations (binary and multi-class)
* Test coverage for `PredictiveStateParams` with sigmoid activation for probabilistic outputs

### Changed

* Enhanced documentation explaining relationship between `PredictiveStateMeans` (mixing) vs `PredictiveStateParams` (broadcasting)

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
