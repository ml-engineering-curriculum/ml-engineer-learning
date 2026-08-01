# exercise-03 — Tabular feature engineering

> **Estimated effort:** 3 hours.
> **Companion chapters:** [07 — Tabular feature engineering](../07-tabular-feature-engineering.md), and (for pipeline mechanics) [`mod-102` chapter 06 — Pipelines and `ColumnTransformer`](../../mod-102-python-ml-toolchain/06-pipelines-and-columntransformer.md).

## Objective

Build a full feature-engineering pipeline that transforms the
cleaned dataset from exercise-02 into a feature matrix ready for a
classifier or regressor. Along the way you will:

- Use every major transformer family from chapter 07 (numeric
  scaling, categorical encoding, datetime extraction, entity-level
  aggregations).
- Build at least one **entity-level windowed feature** that
  respects point-in-time correctness.
- Use `TargetEncoder` correctly for a high-cardinality
  categorical.
- Confirm — by construction — that the whole thing is leakage-free
  when run through `cross_val_score`.
- Compare a `HistGradientBoostingClassifier` (no scaling / no
  imputation needed) against a `LogisticRegression` (both
  required) using the same pipeline shape.

The goal is not the last percentage point of accuracy. It is a
`Pipeline` object that another engineer can `joblib.load` and
`.predict` on, byte-for-byte identically to your training run.

## Prerequisites

- Chapters 05, 06, and 07 of this module.
- `mod-102` chapter 06.
- The output of exercise-02 (`clean()` function producing a tidy
  DataFrame). If you skipped exercise-02, use the UCI Adult
  dataset from `fetch_openml("adult", version=2, as_frame=True)`
  as a fallback.
- Python 3.11+, scikit-learn ≥ 1.4 (for `TargetEncoder` and
  `OneHotEncoder(min_frequency=)`), Pandas ≥ 2.1, `pytest`.

## Problem statement

### Part A — split first (≈ 10 min)

In `data.py`, write `load()` that returns
`(X, y, feature_metadata)` and immediately splits into
`(X_train, X_test, y_train, y_test)`. Requirements:

- If the dataset has a natural time axis, split by time (not
  randomly). The test set is strictly after the training set.
- If the dataset has a natural group axis (user, patient, session)
  but no time, use `GroupShuffleSplit` (or `GroupKFold` for CV).
- Otherwise, `train_test_split(random_state=0, stratify=y)`.

Write down which split you chose and why in `REPORT.md`. This is
the point at which the chapter-06 leakage discipline starts. Do
not touch `X_test` from here until the very end.

### Part B — the baseline pipeline (≈ 40 min)

In `pipeline.py`, build `build_baseline_pipeline() -> Pipeline`
with:

- **Numeric branch.** `SimpleImputer(strategy="median",
  add_indicator=True)` → `StandardScaler()`.
- **Low-cardinality categorical branch.** `SimpleImputer(strategy="most_frequent")`
  → `OneHotEncoder(handle_unknown="ignore", min_frequency=25)`.
- **Datetime branch.** A custom transformer or
  `FunctionTransformer` that extracts year, month, day-of-week,
  hour, cyclical `hour_sin` / `hour_cos`, and days-since-event
  (relative to a reference timestamp you compute *inside* the
  transformer's `fit` from the training data only).
- **`remainder="drop"`** explicitly on the `ColumnTransformer`.
- **`set_output(transform="pandas")`** on the preprocessor so the
  first `.head()` of transformed data is inspectable.
- **Final estimator.** `LogisticRegression(max_iter=1000,
  random_state=0)`.

Fit on `X_train` only. Report score on the test set. This is your
baseline.

Print `pipe[:-1].fit_transform(X_train.head())` — the transformed
matrix as a DataFrame with readable column names — and paste the
first 5 rows into `REPORT.md`. Sanity-check that the one-hot
column names look right.

### Part C — a high-cardinality categorical with `TargetEncoder` (≈ 30 min)

Add a third branch to the `ColumnTransformer`: any categorical
column with `nunique() > 50` in the training data gets its own
branch using `TargetEncoder`. Requirements:

- Use `sklearn.preprocessing.TargetEncoder(smooth="auto", cv=5)`.
- **Inside the pipeline.** Do not compute per-category target means
  by hand.
- Confirm from the source that `TargetEncoder`'s cross-fitting is
  what protects against leakage when it runs inside a
  `cross_val_score`. Note this in `REPORT.md`.

If your dataset has no high-cardinality categorical, add one
synthetically or use the Ames Housing "Neighborhood" column
(~25 unique values on the small side; still worth encoding
target-style).

### Part D — a windowed entity feature (≈ 45 min)

This is the trickiest part of the exercise. Build a feature that
summarises an entity's history and respects point-in-time
correctness.

Concretely: for each row `(entity_id, event_ts)` in your dataset,
compute a feature that is the mean (or count, or sum) of some
other column over the same entity's rows with `ts < event_ts`
within a rolling window.

Suggested features by dataset:

- **Adult:** N/A — the dataset has no time axis; skip Part D and
  document that in `REPORT.md`, or synthesise a `row_index` and
  build a "previous same-country mean age" feature.
- **Ames Housing:** N/A time-wise; use `YrSold` as a pseudo-time
  and compute "mean SalePrice in the same Neighborhood in prior
  years".
- **NYC Taxi:** "mean fare for this pickup location in the
  previous 24 hours".
- **Exercise-02 output:** "user's total spend in the 30 days
  before this order".

Requirements:

1. Compute the feature using **only rows strictly before the label
   row's timestamp** (`ts < event_ts`, not `≤`). Verify this in a
   test: construct two rows with the same key and consecutive
   timestamps; the later row's feature must not include the
   earlier row's target value.
2. Package as a custom transformer that fits on training-only data
   and can `transform` unseen rows (i.e., it stores enough state
   to compute the feature at serving time). Extensions: use
   `pd.merge_asof(direction="backward",
   allow_exact_matches=False)`.
3. Slot into the `ColumnTransformer` as a new branch.

**Test the leakage-free property.** In
`tests/test_temporal_feature.py`, construct a small DataFrame
where a rolling mean *would* differ if it included the label row's
target vs. if it didn't. Assert your transformer produces the
non-leaky answer.

### Part E — a grid search over the pipeline (≈ 30 min)

Use `GridSearchCV` (or `RandomizedSearchCV`) with an appropriate
CV strategy — `TimeSeriesSplit` for time-ordered data,
`GroupKFold` for grouped data, `StratifiedKFold` otherwise. Search
at least three hyperparameters spanning both preprocessing and the
model:

Example:

```python
{
    "features__num__impute__strategy": ["median", "mean"],
    "features__lowc__ohe__min_frequency": [10, 25, 50],
    "clf__C":                            [0.1, 1.0, 10.0],
}
```

Report best params, best CV score, and the score on the locked
`X_test`.

Answer in `REPORT.md`:

- Did the best CV score match the held-out test score, or did you
  see optimism?
- Which hyperparameter mattered most?
- If your test score is lower than your CV score by more than 2
  percentage points, why do you think that is? Distribution
  shift? A leaky feature you didn't catch? Small test set?

### Part F — swap in a gradient booster (≈ 20 min)

Rebuild the pipeline with `HistGradientBoostingClassifier`
(regression: `HistGradientBoostingRegressor`) as the final
estimator. Simplify the numeric and categorical branches
accordingly:

- Numeric: **no** imputation (`HGB` handles missing values
  natively). **No** scaling. Just pass through.
- Categorical: pass through (with `dtype="category"`) and set
  `categorical_features="from_dtype"` on the classifier. Or use
  the ordinal encoding path.

Report the test score. Compare:

- Test score vs. Part E's logistic regression.
- Number of pipeline steps.
- Wall-clock training time.

## Starter guidance

- **Use `make_column_selector`** to pick numeric / categorical /
  datetime columns by dtype. Your pipeline stays robust when a
  new column appears upstream.
- **Save the fitted pipeline** with `joblib.dump(pipe,
  "model.joblib")` and load it in a fresh process. Predict on one
  row from `X_test`. Confirm the result matches training-time
  prediction. This is a preview of `mod-108`.
- **Log the shape of every intermediate matrix** during
  development. `pipe[:-1].fit_transform(X_train.head()).shape`
  answers most of "why did my grid search take an hour" questions.
- For the temporal feature (Part D), `merge_asof` is your friend.
  Its `allow_exact_matches=False` is the point-in-time-correctness
  switch.
- If `TargetEncoder`'s cross-fitting is confusing, read the
  docstring and its "How does target encoding avoid data leakage?"
  section carefully. This is the single most-often-messed-up
  transformation in tabular ML.

## Acceptance criteria

- The split (Part A) is done first and is appropriate for the
  dataset's structure. `X_test` does not appear in any `fit()`
  call after the split.
- The baseline pipeline (Part B) fits, transforms, and scores. A
  screenshot / paste of the transformed matrix (Part B) is in
  `REPORT.md`.
- The `TargetEncoder` branch (Part C) is inside a `Pipeline` and
  runs under `cross_val_score` without warnings about leakage.
- The temporal feature (Part D) has a test that would fail on a
  naive `groupby.rolling` that includes the current row, and
  passes on the correct implementation. If your dataset has no
  time axis, document the skip.
- The grid search (Part E) uses the correct CV strategy for the
  data type; results are reported.
- The gradient booster variant (Part F) is reported alongside the
  linear model results.
- The final pipeline can be saved and loaded via `joblib` and
  produces identical predictions in a fresh process.
- No `fit_transform` call touches `X_test`. `git grep` for
  `X_test` shows only `pipe.predict`, `pipe.score`, and printing
  statements.

## Stretch goals (optional)

- **Feature interaction discovery.** Use
  `HistGradientBoostingRegressor`'s built-in partial dependence to
  visualise a two-way interaction. Add a manual interaction
  feature and see if it helps a linear model.
- **Calibration check.** Use
  `sklearn.calibration.CalibrationDisplay` on the classifier's
  predicted probabilities. A well-calibrated model matters for
  downstream decisions. See `mod-107`.
- **Serving-time round-trip test.** Write a tiny Flask / FastAPI
  endpoint that loads the pipeline and serves predictions.
  Confirm one prediction from the endpoint matches a batched
  prediction on the same row. Preview of `mod-108`.
- **Try `catboost` or `lightgbm`** as the final estimator with
  native categorical support. Compare against `HistGradientBoosting`.
- **Feature selection.** Add `SelectFromModel(HistGradientBoosting...)`
  as a preprocessing step. Search over the number of kept
  features in the grid search.

## What to hand in

```
exercise-03/
├── data.py                       # load() + split
├── pipeline.py                   # build_baseline_pipeline(), build_hgb_pipeline()
├── transformers.py               # datetime features + windowed entity feature
├── tests/
│   ├── test_pipeline.py          # fit/predict/joblib round-trip
│   └── test_temporal_feature.py  # point-in-time correctness
├── model.joblib                  # a saved fitted pipeline
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
