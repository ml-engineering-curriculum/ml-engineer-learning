# 05 — scikit-learn: estimators and transformers

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 01, 02, 03.

## Motivation

scikit-learn is the dominant tabular ML library in Python. Even
when the *final* model is XGBoost, LightGBM, or a PyTorch network,
the surrounding preprocessing, splitting, and evaluation code in
production ML codebases is overwhelmingly scikit-learn. The reason
is the *API*: every object — model, scaler, encoder, imputer,
feature selector, dimension reducer — exposes the same tiny
interface. Learn it once, use it forever.

This chapter introduces the core interface (estimators and
transformers), the conventions that make objects composable, and how
to write a custom transformer that plays nicely with the rest of the
ecosystem. Chapter 06 then plugs the pieces into `Pipeline` and
`ColumnTransformer`, which is where the leakage-resistance comes
from.

## The five-method API

Almost every scikit-learn object implements some subset of:

| Method | Lives on | What it does |
|---|---|---|
| `fit(X, y=None)` | every estimator | Learn parameters from training data; return `self`. |
| `transform(X)` | every transformer | Apply the learned transformation; return new `X'`. |
| `fit_transform(X, y=None)` | every transformer | `fit` then `transform`, often faster than the two separately. |
| `predict(X)` | every supervised model | Return predicted labels / values. |
| `predict_proba(X)` | classifiers | Return predicted class probabilities. |
| `score(X, y)` | most supervised models | Return a default scoring metric. |

This is the entire surface. A `StandardScaler` and a
`RandomForestClassifier` look almost identical from the outside.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler()
scaler.fit(X_train)              # learn column means/stds
X_train_std = scaler.transform(X_train)

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train_std, y_train)    # learn trees
y_pred = clf.predict(X_test_std)
```

## Estimator vs. transformer vs. predictor

The vocabulary the docs use:

- **Estimator** — anything with a `fit` method. The umbrella term.
- **Transformer** — an estimator that also has `transform`. Used for
  feature preprocessing.
- **Predictor** — an estimator that also has `predict`. Used for
  modelling.

Almost every scikit-learn object is one or both of the last two. The
distinction matters because the role determines where the object
slots into a `Pipeline` (chapter 06): transformers as intermediate
steps, predictors only at the end.

## Conventions: input shape, hyperparameters, and learned state

scikit-learn enforces strong conventions, and following them is what
lets pipelines work:

1. **`X` is `(n_samples, n_features)`**, 2-D. A 1-D vector raises;
   reshape with `X.reshape(-1, 1)` if you have a single feature.
2. **`y` is `(n_samples,)`** for single-output tasks, `(n_samples,
   n_outputs)` for multi-output.
3. **Hyperparameters are passed to `__init__`** and stored as
   attributes with the *same name*. So `LogisticRegression(C=0.1)`
   stores `clf.C == 0.1`. Do not transform the value in `__init__`
   — it must be retrievable unchanged via `get_params`.
4. **Learned state lives on attributes ending in `_`.** A scaler's
   learned column means are `scaler.mean_`; a fitted random forest's
   trees are `clf.estimators_`. This trailing-underscore convention
   is how you (and the library) distinguish "what the user
   configured" from "what `fit` produced".
5. **`get_params` / `set_params`** introspect and override
   hyperparameters by name. They are what makes `GridSearchCV` and
   `RandomizedSearchCV` work.

If you write a custom estimator that breaks any of these
conventions, the rest of the ecosystem (cross-validation, grid
search, pipelines) will not work with it. Conform.

## A grand tour: what's in `sklearn.preprocessing`

The transformers you will reach for most often:

- **`StandardScaler`** — subtract column mean, divide by column std.
- **`MinMaxScaler`** — rescale each column to `[0, 1]`.
- **`RobustScaler`** — like `StandardScaler` but uses median /
  interquartile range; robust to outliers.
- **`OneHotEncoder`** — turn categorical columns into one-hot
  vectors. Pass `handle_unknown="ignore"` if you may see unseen
  categories at predict time.
- **`OrdinalEncoder`** — map each unique category to an integer.
  Use only when the categories have a meaningful order, or for
  tree-based models that handle integer-coded categoricals natively.
- **`SimpleImputer`** — fill missing values with mean / median / most
  frequent / a constant. The right place for what you would
  otherwise hand-roll with `df.fillna(...)`.

Every one of these is a transformer with `fit` and `transform`. Used
through a `Pipeline` (chapter 06), they `fit` only on the training
fold — which is the whole point.

## Custom transformers: when and how

You will eventually need a transformation that scikit-learn does not
ship. Two ways to add one:

### 1. `FunctionTransformer` for stateless transforms

If the transformation is a pure function — no learned state, same
behaviour on train and test — wrap it in
[`FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html):

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

log1p_tx = FunctionTransformer(np.log1p, validate=True)
log1p_tx.fit(X_train)        # no-op
X_train_log = log1p_tx.transform(X_train)
```

This is the right tool for log transforms, clipping, simple
arithmetic on columns, etc.

### 2. Subclass `BaseEstimator` and `TransformerMixin` for stateful transforms

When the transform must *learn* something from the training data
(a column mean, a vocabulary, a fitted KDE), write a real class:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class WinsoriseTransformer(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] training-set quantiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        # store hyperparameters as-is; do NOT transform them here.
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_bounds_ = np.quantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)
```

Things to notice:

- `__init__` only stores hyperparameters. Anything that needs the
  data lives in `fit`.
- Learned state is stored on `self.<name>_` (trailing underscore).
- `fit` returns `self`. Pipelines depend on this.
- The convention `n_features_in_` is part of the scikit-learn
  estimator contract (since 0.24); it enables shape-mismatch
  detection at `transform` time.

scikit-learn ships a
[`check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html)
utility that runs the full conformance test suite against your
class. Run it once when you write a custom transformer; it catches
many of the subtle convention violations described above. You will
also re-encounter `WinsoriseTransformer` (or its cousin) in the
exercises for chapter 08, where you'll unit-test it.

## `fit` vs. `fit_transform`: when does the difference matter?

For a single object, none — `fit_transform(X)` is equivalent to
`fit(X).transform(X)` (with maybe a small speed-up). The difference
matters at the level of the *pipeline*:

- **Training**: call `pipeline.fit(X_train, y_train)`. Every
  intermediate transformer is fit *and then transforms* `X_train`
  for the next step.
- **Predicting**: call `pipeline.predict(X_test)`. Every
  intermediate transformer *only transforms* `X_test`; its `fit`
  was already done on training data.

This is what makes pipelines leakage-resistant: at predict time, no
fitting happens, so test-set statistics cannot bleed into the
preprocessing parameters.

## Random state: reproducibility and `random_state`

Every scikit-learn estimator that uses randomness (random forests,
gradient boosting, train/test splits, KMeans, ...) takes a
`random_state` argument. Pass an integer to get a reproducible run:

```python
clf = RandomForestClassifier(n_estimators=100, random_state=0)
```

The same integer always produces the same fit on the same data.
This is the building block on which `mod-106` (experiment tracking)
rests; never leave `random_state=None` in a model that you intend to
report numbers from.

## Pitfalls collected

1. **Fitting on the test set.** `scaler.fit_transform(X_test)`
   instead of `scaler.transform(X_test)` is the canonical leakage
   bug. Pipelines (chapter 06) prevent it.
2. **Reshaping a 1-D vector.** Pandas Series → `ValueError: Expected
   2D array, got 1D array`. The fix is `series.to_numpy().reshape(-1,
   1)` or — better — work in a `DataFrame` and use
   `ColumnTransformer`.
3. **`OneHotEncoder` with unseen categories.** Default is to raise
   at transform time. Pass `handle_unknown="ignore"` (and accept
   that unseen categories become all-zero rows) or
   `handle_unknown="infrequent_if_exist"` if you have an
   "infrequent" bucket.
4. **Not setting `random_state`.** Random forests, gradient boosting,
   and cross-validation splits are all stochastic. Without
   `random_state`, two runs will disagree on the third decimal place
   and you will burn a day debugging it.
5. **Forgetting `fit` returns `self`.** Custom transformers must
   return `self`, otherwise `pipeline.fit(X).transform(X)` cannot
   chain.

## Summary

- The whole scikit-learn API is `fit`, `transform`, `predict`, and a
  handful of conventions around hyperparameter storage and learned
  state.
- Built-in transformers cover the routine preprocessing surface
  (scaling, encoding, imputing).
- Custom transformations stay ecosystem-compatible if you subclass
  `BaseEstimator` and `TransformerMixin` and follow the
  trailing-underscore / `__init__` conventions.
- `random_state` is the reproducibility lever — never leave it
  unset in code you intend to publish numbers from.

## Where this goes next

Chapter 06 picks up `Pipeline` and `ColumnTransformer`, which is
where the leakage-resistance and the train/test discipline are
actually enforced. Every transformer you learned about here becomes
a step inside a pipeline there.
