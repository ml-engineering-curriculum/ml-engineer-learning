# 06 — Pipelines and `ColumnTransformer`

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 03, 05.

## Motivation

Chapter 05 introduced the scikit-learn estimator API: a handful of
fit/transform/predict methods plus a few naming conventions. This
chapter assembles those pieces into the two constructions that make
real-world preprocessing tractable:

- **`Pipeline`** — chain a sequence of transformers followed by an
  optional final estimator. Treat the whole chain as a single
  estimator.
- **`ColumnTransformer`** — apply *different* transformers to
  *different* columns of a heterogeneous `DataFrame`, then
  concatenate the result.

Together they solve three problems that imperative preprocessing code
keeps running into:

1. **Leakage** — fitting preprocessing parameters on data that
   includes the test set.
2. **Drift between train and serve** — applying a different
   preprocessing function at predict time than at fit time, because
   the two live in different functions.
3. **Hyperparameter search complexity** — searching over choices of
   imputer, scaler, and model in one sweep.

If you adopt one workflow change from this whole module, make it
"always wrap preprocessing + model in a `Pipeline`".

## Leakage, concretely

The textbook example. You have a feature with missing values, and
you want to impute the median:

```python
# WRONG — leakage across the train/test split.
median = X["age"].median()
X["age"] = X["age"].fillna(median)
X_train, X_test = train_test_split(X, random_state=0)

# Still wrong — leakage across folds in cross-validation.
median = X_train["age"].median()
X_train["age"] = X_train["age"].fillna(median)
X_test["age"]  = X_test["age"].fillna(median)
scores = cross_val_score(estimator, X_train, y_train, cv=5)
# inside cross_val_score, every fold uses the SAME median computed
# from the full training set.
```

The right way uses a `Pipeline` so the imputer is *re-fit on the
inner training fold every time* cross-validation rolls:

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LogisticRegression(max_iter=1000, random_state=0)),
])

scores = cross_val_score(pipe, X_train, y_train, cv=5)
# every fold re-fits the imputer, scaler, AND model on its own
# training portion. No leakage.
```

This pattern is the entire point. The rest of the chapter is
"how to express the construction cleanly when the data is
heterogeneous."

## `Pipeline`: a chain of named steps

A `Pipeline` is a list of `(name, estimator)` tuples. Every step
except the last must be a transformer (have `transform`); the last
may be a transformer *or* a predictor.

```python
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LogisticRegression()),
])
```

Behaviour:

- `pipe.fit(X, y)` calls `fit_transform` on every transformer in
  order, feeding the result forward, then calls `fit` on the final
  estimator.
- `pipe.predict(X)` calls `transform` on every transformer in order,
  then `predict` on the final estimator.
- `pipe.fit_transform(X)` only works if the final step is a
  transformer.

Two convenience features worth knowing:

- **Access a step by name**: `pipe.named_steps["model"]` or
  `pipe["model"]`.
- **Set or search hyperparameters by nested name**: the syntax
  `"<step>__<param>"` reaches into a step. Used by `GridSearchCV`:

  ```python
  from sklearn.model_selection import GridSearchCV
  param_grid = {
      "imputer__strategy": ["median", "mean"],
      "model__C":          [0.1, 1.0, 10.0],
  }
  search = GridSearchCV(pipe, param_grid, cv=5)
  search.fit(X_train, y_train)
  ```

The double-underscore is not a typo; it's how scikit-learn flattens
nested parameter names.

## `ColumnTransformer`: heterogeneous columns

Most real data has columns of different *kinds*: numeric, categorical,
text, dates. They need different preprocessing. The naive
implementation is a forest of `if`/`else` branches in custom Python.
The clean implementation is `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_cols     = ["age", "income", "tenure_months"]
categorical_cols = ["country", "device", "plan"]

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe,     numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])
```

`preprocess` is itself a transformer. You can drop it into another
`Pipeline`:

```python
model = Pipeline([
    ("preprocess", preprocess),
    ("clf",        LogisticRegression(max_iter=1000, random_state=0)),
])
model.fit(X_train, y_train)
```

Behaviour:

- For each `(name, transformer, columns)` triple, the transformer is
  fit on the named columns of the training data and then transforms
  those columns at predict time.
- The outputs are **horizontally concatenated** (after dense-or-sparse
  promotion) into a single matrix.
- The `remainder=` argument controls what happens to columns you did
  not list. Default `"drop"` drops them silently; `"passthrough"`
  keeps them; a transformer instance applies *that* to the rest.
  Be deliberate: `remainder="drop"` is responsible for many
  "I added a feature and it had no effect" bugs.

### Selecting columns by dtype

For wider feature sets, listing every column name is brittle.
`make_column_selector` lets you select by dtype:

```python
from sklearn.compose import make_column_selector

preprocess = ColumnTransformer([
    ("num", numeric_pipe,     make_column_selector(dtype_include="number")),
    ("cat", categorical_pipe, make_column_selector(dtype_include=["object", "category"])),
])
```

This is the right tool when you trust your DataFrame dtypes (chapter
03's `category` and nullable types pay off here).

## Inspecting the pipeline

When something goes wrong — and it will — you want to look inside.

- `pipe.named_steps` — dict of `{name: estimator}` for inspection.
- `pipe[:-1]` — slice off the final predictor to get just the
  preprocessing chain. Call `pipe[:-1].fit_transform(X)` to see the
  matrix the model actually sees.
- `pipe.set_output(transform="pandas")` (since scikit-learn 1.2)
  makes transformers return `DataFrame`s with sensible column names.
  Wonderful for debugging; toggleable per pipeline.

```python
preprocess.set_output(transform="pandas")
preprocess.fit_transform(X_train.head()).head()
# Now the encoded columns have readable names like cat__country_US.
```

Once you have seen a `ColumnTransformer` output displayed as a
DataFrame with named columns, you do not go back.

## Caching to skip recomputation

Pipeline steps can be expensive — a heavy text vectoriser, a
fitted PCA. Pass `memory=` to cache fitted intermediate steps across
calls (useful inside a grid search where only the *final* step
changes):

```python
pipe = Pipeline(
    [("vectorise", TfidfVectorizer()),
     ("model",     LogisticRegression())],
    memory="./.sklearn_cache",
)
```

scikit-learn hashes the inputs + hyperparameters and reuses fitted
steps when possible. Not magic; not always a win. Profile before
reaching for it.

## Patterns to copy

A few small templates you will reuse:

### Tabular classification: numeric + categorical + tree model

```python
preprocess = ColumnTransformer([
    ("num", "passthrough",                                          numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"),                 categorical_cols),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("clf",        GradientBoostingClassifier(random_state=0)),
])
```

Trees do not need scaling — that's why `"passthrough"` is fine for
the numeric columns.

### Tabular classification: linear model

```python
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale",  StandardScaler())]),               numeric_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("ohe",    OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
])
model = Pipeline([("preprocess", preprocess),
                  ("clf",        LogisticRegression(max_iter=1000, random_state=0))])
```

Linear models *do* need scaling, and they explode when the input has
NaNs.

### Cross-validated grid search

```python
search = GridSearchCV(
    model,
    {"clf__C": [0.1, 1.0, 10.0]},
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

## Pitfalls collected

1. **`fit_transform` on the test set.** A `Pipeline` makes this
   syntactically wrong — the test path goes through `transform`
   only — which is the whole point.
2. **`remainder=` left as default.** Silent column-dropping is a
   common source of "the feature I added isn't moving the metric".
   Set it explicitly.
3. **Column names disappearing.** Without `set_output("pandas")`,
   the post-`ColumnTransformer` matrix has positional columns only.
   For debugging and SHAP-style interpretability work, set the
   output to pandas.
4. **Step naming collisions in `GridSearchCV`.** The double-
   underscore parameter syntax depends on unique step names. If you
   have nested `Pipeline`s, the names compound:
   `preprocess__num__impute__strategy`.
5. **Forgetting `handle_unknown` on `OneHotEncoder`.** Unseen
   categories at predict time will raise. Set it deliberately.
6. **Imbalanced classes.** A `Pipeline` does not magically balance
   classes; pair it with `StratifiedKFold`, `class_weight=`, or
   `imbalanced-learn`'s `Pipeline` subclass if you need
   resampling steps.

## Summary

- A `Pipeline` chains transformers into a single estimator.
  `fit` fits each step in turn; `predict` only transforms — which
  is what eliminates train/test leakage in cross-validation and at
  serving time.
- A `ColumnTransformer` routes different columns through different
  transformers and concatenates the result. The combination of
  `ColumnTransformer` + `Pipeline` is the workhorse of tabular ML
  preprocessing in scikit-learn.
- The `step__param` syntax flattens nested hyperparameters for grid
  search.
- `set_output("pandas")` is a debugging quality-of-life win;
  `memory=` is a niche caching win.
- The cardinal rule: **never preprocess outside the pipeline**.

## Where this goes next

Chapter 07 covers reproducible environments — the layer below all of
this code, which is what makes "model trained on Monday" reproducible
the following Friday. Chapter 08 closes the module with testing,
including how to unit-test the custom transformers and pipelines
you build here.
