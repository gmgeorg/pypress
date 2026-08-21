# Claude Context: pypress

Predictive State Smoothing (PRESS): a semi-nonparametric ML algorithm implemented in
`tf.keras` for high-dimensional regression/classification with variable selection.
See [README.md](README.md) for the algorithm background, papers, and usage examples.

## Structure

```text
pypress/
├── clustering.py           # Clustering utilities for predictive states
├── utils.py                 # State operations, kernel functions
├── keras/
│   ├── layers.py            # PredictiveStateSimplex, PredictiveStateMeans, PRESS
│   ├── regularizers.py      # Uniform, DegreesOfFreedom
│   ├── initializers.py      # PredictiveStateMeansInitializer
│   └── activations.py
└── tests/
```

## Development

```bash
poetry install
poetry run pytest pypress/tests/ -v
poetry run ruff check --fix . && poetry run ruff format .
```

Pre-commit hooks (ruff, markdownlint-cli2, codespell, interrogate) run on commit/push — see `.pre-commit-config.yaml`.
