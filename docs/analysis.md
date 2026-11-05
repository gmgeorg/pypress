# Analyzing a fitted model

A fitted PRESS model is a set of interpretable predictive states. `pypress.analysis` works on the
predictive state weights $W$ of any set of observations. It takes numpy arrays or pandas
DataFrames, and DataFrames keep their index and column names.

## Getting the predictive state weights

```python
import pandas as pd
from pypress import analysis

W = press.predictive_states(X_test).numpy()                    # (n, K), rows sum to 1
W = pd.DataFrame(W, index=X_test.index,
                 columns=[f"s{k}" for k in range(W.shape[1])])
```

For a model built from the two layers, call the `PredictiveStateSimplex` layer on the features.

Each observation's most likely state, `W.idxmax(axis=1)`, gives a hard **predictive
clustering**. The weights themselves are the soft version.

## How much of the population each state holds

```python
analysis.state_size(W)                  # column sums: the state size
analysis.state_size(W, normalize=True)  # state shares, summing to 1
```

A **dead state** has a share near zero. It contributes nothing to predictions and receives
almost no gradient. Some dead states are expected when `n_states` is generous. Many dead states
mean `n_states` is too large, or that a regularizer is missing; see
[regularizers.md](regularizers.md).

## Effective number of states

```python
analysis.degrees_of_freedom(W)   # trace(H), between 1 and K
```

This is the degrees of freedom of the smoother matrix $H$ (see [concepts.md](concepts.md)).
Compare it with `n_states`: a model with $K = 9$ and a value of 3.2 effectively uses about three
states. It is computed in $O(nK)$ without forming $H$.

## The smoother matrix

```python
H = analysis.smoother_matrix(W)   # (n, n)
```

Row $i$ of $H$ holds how much each observation contributes to the prediction for observation $i$;
observations that share predictive states contribute more to each other. $H$ is $n \times n$, so
use it on a sample, for example as a heatmap of a few hundred observations sorted by their most
likely state.

## Describing the states

```python
analysis.agg_data_by_state(X_test, W)   # one row per numeric feature, one column per state
```

This gives each numeric feature's mean within each state, weighted by $W$. Together with
`press.state_conditional_means` it describes each state as "observations that look like this,
and have this outcome".

`analysis.col_normalize(W)` gives the weights used for those averages: each column of $W$ divided
by its state size.

[`notebooks/demo.ipynb`](../notebooks/demo.ipynb) walks through these steps on a real dataset.
