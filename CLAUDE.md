# Claude Context: pypress

## Project Overview
Predictive State Smoothing (PRESS) - A semi-nonparametric ML algorithm implemented in TensorFlow/Keras for high-dimensional regression and classification with variable selection.

## Key Architecture

- **Core Layers**: `PredictiveStateSimplex` and `PredictiveStateMeans` (or combined `PRESS()` wrapper)

- **Regularizers**: `Uniform` and `DegreesOfFreedom` for controlling state distributions

- **Custom Initializers**: `PredictiveStateMeansInitializer` for proper weight initialization

## Project Structure
```
pypress/
├── pypress/
│   ├── __init__.py          # Exports __version__
│   ├── _version.py          # Dynamic version from pyproject.toml
│   ├── utils.py             # State operations, kernel functions
│   ├── keras/
│   │   ├── layers.py        # Core PRESS layers
│   │   ├── regularizers.py  # Uniform, DegreesOfFreedom, Combined
│   │   └── initializers.py  # Custom weight initialization
│   └── tests/               # 29 tests (all passing)
├── pyproject.toml           # Single source of truth for version
└── poetry.lock
```

## Dependencies
- **TensorFlow**: `>=2.11.0,<3.0.0`
- **NumPy**: `>=2.0.0`
- **Pandas**: `>=1.5.0`
- **Dev**: pytest `^9.0.2`

## Development Commands
```bash
# Install dependencies
poetry install

# Run tests (29 tests, all passing)
poetry run pytest pypress/tests/ -v
```

## Important Notes
- PRESS decomposes p(y|X) into predictive states: p(y|X) = Σ p(y|s_j) · p(s_j|X)
- Conditional independence: y ⊥ X | s_j (outputs independent of features given state)
- Predictive states are minimal sufficient statistics for y
- Similar to Mixture Density Networks but with conditional independence structure

## Papers
- Goerg (2018): Classification using PRESS
- Goerg (2017): Scalable non-parametric regression with PRESS
