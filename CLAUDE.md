# Claude Context: pypress

Predictive State Smoothing (PRESS): a semi-nonparametric ML algorithm implemented in
`tf.keras` for high-dimensional regression/classification with variable selection.
See [README.md](README.md) for the algorithm background, papers, and usage examples.

## Structure

```text
pypress/
├── analysis.py, clustering.py  # numpy/pandas only
├── utils.py                 # Deprecated shims (TF)
├── core/                    # Framework-free (no TF, no torch): hyperparameters, schedules
│                            #   + epoch clock, activation inverses, init values, statistic shapes
├── keras/                   # layers, regularizers, initializers, activations, callbacks
├── torch/                   # nn.Module port: layers, regularizers, statistics, schedules,
│                            #   activations; see docs/torch.md
└── tests/{core,keras,torch,parity}/   # each dir skips itself if its framework is missing
```

`keras` and `torch` mirror each other (same class names and `get_config()` keys); the parity
tests in `tests/parity/` hold them equal, so change both together. Anything framework-free
belongs in `core/`. TensorFlow and PyTorch are the optional extras `pypress[tf]` /
`pypress[torch]`; the dev group installs both.

## Development

```bash
poetry install
poetry run pytest pypress/tests/ -v      # whole suite (~30 s, mostly TF + torch import)
poetry run pytest pypress/tests/torch pypress/tests/parity   # one backend + parity
poetry run ruff check --fix . && poetry run ruff format .
```

Pre-commit hooks (ruff, markdownlint-cli2, codespell, interrogate) run on commit/push — see `.pre-commit-config.yaml`.
