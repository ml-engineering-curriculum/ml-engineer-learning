# exercise-02 — scikit-learn pipeline design

> **Estimated effort:** 3 hours.
> **Companion chapters:** [05 — scikit-learn: estimators and transformers](../05-sklearn-estimators-and-transformers.md), [06 — Pipelines and `ColumnTransformer`](../06-pipelines-and-columntransformer.md).

## Objective

Build a real preprocessing + modelling `Pipeline` on a heterogeneous
tabular dataset — numeric and categorical columns, missing values,
unseen categories at predict time — and prove the pipeline eliminates
train/test leakage. Along the way you will write one small custom
transformer, wire a `ColumnTransformer`, run a
`GridSearchCV` over nested hyperparameters, and inspect the fitted
model's intermediate output as a DataFrame.

By the end you should be able to design a leakage-free preprocessing
pipeline for any tabular dataset in under an hour.

## Prerequisites

- Python 3.11+, scikit-learn ≥ 1.4, NumPy ≥ 1.26, Pandas ≥ 2.1,
  `pytest`.
- Chapters 05 and 06 of this module.
- The output of exercise-01 (`build_user_features`) is a natural
  input, but any tabular dataset with mixed numeric + categorical
  columns and some missing values works. `sklearn.datasets.fetch_openml`
  ships a few appropriate ones (see "Dataset choice" below).

## Dataset choice

Pick one dataset with mixed numeric + categorical columns and at
least one column with missing values. Two safe defaults:

- The **Adult** (a.k.a. Census Income) dataset from
  `fetch_openml("adult", version=2, as_frame=True)`. Binary
  classification target (`>50K`). ~48k rows, ~14 columns.
- The **Ames Housing** dataset from
  `fetch_openml("house_prices", as_frame=True)`. Regression target
  (`SalePrice`). ~1.5k rows, ~80 columns — noisier and messier,
  which is useful practice.

You may also use the feature matrix from exercise-01 with a
synthetic `y` (e.g. future-90d-spend) — call it out in your
`REPORT.md`.

## Problem statement

### Part A — split first, then look (≈ 15 min)

In `data.py`, write a `load()` function that returns
`(X, y, numeric_cols, categorical_cols)`. Do the train/test split
**first** with a fixed `random_state`. From this point on, treat
`X_test` / `y_test` as a locked box: do not compute *any* statistic
on it, do not sneak a peek at its dtypes, do not use it to decide
which columns are categorical.

Write it down in `REPORT.md`: "I split before I looked." This is
the single-most-common leakage mistake, and the discipline is
worth internalising.

### Part B — a first pipeline (≈ 45 min)

In `pipeline.py`, build a preprocessing + logistic regression
pipeline. Requirements:

1. A `ColumnTransformer` with two branches:
   - **Numeric branch**: `SimpleImputer(strategy="median")` →
     `StandardScaler()`.
   - **Categorical branch**: `SimpleImputer(strategy="most_frequent")` →
     `OneHotEncoder(handle_unknown="ignore")`.
2. The `ColumnTransformer` is wrapped in a `Pipeline` with a final
   `LogisticRegression(max_iter=1000, random_state=0)`.
3. Set `remainder=` **explicitly** on the `ColumnTransformer` — do
   not accept the default. Justify your choice in `REPORT.md`
   (usually `"drop"`, occasionally `"passthrough"`; never silence).
4. Call `preprocess.set_output(transform="pandas")` and print the
   first five rows of the transformed training matrix. Sanity-check
   that the one-hot column names look right.

Fit on the train split; report the accuracy (Adult) or RMSE (Ames)
on the held-out test split. This is your baseline.

### Part C — a custom transformer (≈ 30 min)

In `transformers.py`, implement a `RareCategoryGrouper` transformer
that maps any categorical value seen fewer than `min_count` times in
the training data to a single `"__rare__"` bucket. Signature:

```python
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_count: int = 10): ...
    def fit(self, X, y=None): ...   # learn the per-column keep-sets
    def transform(self, X): ...     # replace out-of-set values with "__rare__"
```

Requirements:

- Store the learned per-column keep-sets on `self.keep_sets_` (note
  the trailing underscore — chapter 05 convention).
- Store `self.n_features_in_ = X.shape[1]` in `fit`.
- Support both a `numpy.ndarray` and a `pandas.DataFrame` input.
- At `transform` time, values seen at fit time pass through
  untouched; unseen values (including `pd.NA` / `NaN`) become
  `"__rare__"`.
- `fit` must return `self`.

Slot it into your `ColumnTransformer`'s categorical branch *before*
the `OneHotEncoder`. Refit the pipeline. What happens to the number
of one-hot columns? Note it in `REPORT.md`.

Run `sklearn.utils.estimator_checks.check_estimator(RareCategoryGrouper())`
and either (a) get a clean run, or (b) note in `REPORT.md` which
checks you deliberately skip and why. (For a categorical-only
transformer, some of the numeric-input checks fail legitimately.)

### Part D — hyperparameter search over the whole pipeline (≈ 45 min)

Use `GridSearchCV` to tune both a preprocessing choice and a
model hyperparameter in one sweep. At minimum:

- `preprocess__num__impute__strategy`: `["median", "mean"]`
- `preprocess__cat__rare__min_count`: `[5, 10, 50]`
- `model__C`: `[0.1, 1.0, 10.0]`

Use `cv=5`, a scoring metric appropriate to your task, and
`n_jobs=-1`. Report the best parameters and the mean CV score.

Then — **and this is the important part** — retrain the best
pipeline on the full training split and report the score on the
locked test box. Compare to the baseline from Part B.

In `REPORT.md`, answer:

- Which hyperparameter mattered most?
- Did the best CV score match the held-out test score, or did you
  see optimism? By how much?

### Part E — prove leakage would have happened (≈ 30 min)

Write two versions of the imputation step in `leakage_demo.py`:

1. **Leaky**: fit `SimpleImputer` on the full `X` (train + test
   combined) before splitting.
2. **Correct**: fit `SimpleImputer` inside the pipeline, so it
   only sees the training fold in each CV split.

Run a 5-fold cross-validation of the *classifier only* under each
regime. Report both mean scores. On most datasets the leaky
version scores slightly higher — because it has silently used
information from the fold it is being evaluated on.

Add a one-paragraph explanation to `REPORT.md` of *why* the leaky
version scores higher, in terms of what the imputer "sees" in each
regime. Do not just report the number; explain the mechanism.

## Starter guidance

- Do Part A *first*. The whole exercise is much easier if you know
  you never touched the test set.
- Use `make_column_selector(dtype_include="number")` and
  `dtype_include=["object", "category", "string"]` rather than
  hard-coding column lists. Your `ColumnTransformer` becomes robust
  to schema changes.
- If the fitted `ColumnTransformer` produces a huge one-hot matrix
  (thousands of columns), your dataset has a high-cardinality
  categorical. That's exactly the case `RareCategoryGrouper`
  solves.
- When you set `n_jobs=-1` in `GridSearchCV`, do it *outside* your
  notebook — parallel joblib workers do not play well with Jupyter's
  kernel on some platforms.
- For debugging: `pipe[:-1].fit_transform(X_train.head())` shows
  you the exact matrix the classifier sees. It is the single most
  useful pipeline-debugging expression.

## Acceptance criteria

- The train/test split happens *before* any inspection or
  imputation. `X_test`, `y_test` do not appear in any `fit` call
  outside `pipe.predict`/`pipe.score`.
- The baseline pipeline (Part B) fits, transforms, and scores on
  the held-out test set. `set_output(transform="pandas")` is used
  and the printed transformed matrix has readable column names.
- `RareCategoryGrouper` (Part C) passes at least the following
  four hand-written tests in `tests/test_transformers.py`:
  1. Rare values are mapped to `"__rare__"` at transform time.
  2. Common values pass through unchanged.
  3. Unseen values (never present at fit time) become `"__rare__"`.
  4. `check_is_fitted(tx)` raises `NotFittedError` before `fit`.
- `GridSearchCV` (Part D) runs to completion; the best-params and
  best-CV-score are reported; the retrained best pipeline is scored
  on the held-out test box.
- The leakage demo (Part E) reports both scores and includes the
  mechanistic explanation in `REPORT.md`.
- No `fit_transform` call touches `X_test`. Grep your source; the
  string should not appear next to `X_test`.

## Stretch goals (optional)

- Replace `LogisticRegression` with
  `HistGradientBoostingClassifier` — a tree model that natively
  handles missing values. Rewire the numeric branch to *not*
  impute or scale, and compare test scores.
- Add a **feature-selection** step (`SelectKBest` or `SelectFromModel`)
  as an additional pipeline stage. Include `k` or the selector's
  threshold in the grid search.
- Use `sklearn.pipeline.FeatureUnion` (or a second
  `ColumnTransformer`) to add a manually-engineered feature
  alongside the raw columns — e.g. a numeric-column interaction —
  and confirm the model uses it.
- Save the fitted pipeline with `joblib.dump(pipe, "model.joblib")`
  and reload it in a fresh Python process. Predict on one row and
  confirm the result is identical. This is the shape of what
  `mod-108` (model deployment) picks up.

## What to hand in

```
exercise-02/
├── data.py                  # load() with pre-split discipline
├── pipeline.py              # baseline + tuned pipeline builders
├── transformers.py          # RareCategoryGrouper
├── leakage_demo.py          # Part E — leaky vs. correct
├── tests/
│   ├── test_transformers.py
│   └── test_pipeline.py     # fit/predict/score sanity checks
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
