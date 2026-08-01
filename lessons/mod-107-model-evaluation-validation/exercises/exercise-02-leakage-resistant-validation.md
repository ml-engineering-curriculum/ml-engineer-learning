# exercise-02 — Leakage-resistant validation

> **Estimated effort:** 3 hours.
> **Companion chapters:** [06 — Splits, leakage, and CV](../06-splits-leakage-and-cross-validation.md), [07 — Time-based validation](../07-time-based-validation.md), [08 — Group-based and nested CV](../08-group-based-and-nested-cv.md).

## Objective

Design and defend a validation strategy for a dataset
that has both a time index *and* a group structure.
Prove — by measurement, not by argument — that a naive
split leaks information, and that your strategy does
not. By the end you should have:

- A **leakage demonstration** — the same model trained
  and evaluated three ways (random shuffle, time-only,
  time+group), with metrics showing the leak-inflation
  gap.
- A **nested-CV report** that produces an honest
  hyperparameter-selection score and a distinct test-
  set score, using time+group splits throughout.
- A **`validation_config.py`** — a small module that
  encodes the chosen splitting policy so future runs
  cannot silently fall back to a random shuffle.
- **`NOTES.md`** with the leakage taxonomy check
  (chapter 06) applied to your data and the pipeline.

The point of the exercise is to internalise the
"reported metric depends on the split" lesson at the
level where you can spot leakage in someone else's PR.

## Prerequisites

- Chapters 06–08 of this module.
- Python 3.11+; `scikit-learn ≥ 1.4`; `pandas`, `numpy`.
- Optional: `sktime` if you want a first-class
  `GroupTimeSeriesSplit`; otherwise implement it
  yourself.

## The dataset

Use one of the following (all publicly available; all
have both time and group structure):

- **Online Retail II** (UCI) — invoice-level e-commerce
  transactions. Group = customer; time = invoice date.
  Task = predict whether a customer will make a purchase
  in the next 30 days.
- **NYC Taxi trip data** — subset one month. Group =
  taxi medallion / pickup zone; time = pickup time.
  Task = predict trip fare or trip-duration bucket.
- **A time-series feature you build from Kaggle's
  "Store Sales - Time Series Forecasting"** — groups =
  (store, item) pairs; time = date; task = 7-day sales
  forecast.
- **Your own dataset**, provided it has both a plausible
  group column (an entity that produces multiple rows)
  and a time index. Document why you picked it.

State your dataset choice up front in `NOTES.md`.

## Problem statement

### Part A — Build the pipeline (≈ 30 min)

Implement a training pipeline that:

- Loads the dataset and produces a `pandas.DataFrame`
  with a `date` column, a group column, features, and
  a target.
- Wraps preprocessing (scaling, imputation, encoding,
  any target-derived transformations) inside a
  scikit-learn `Pipeline` or `ColumnTransformer`.
- Uses a **single model class** for the whole exercise
  (a `HistGradientBoostingClassifier` /
  `HistGradientBoostingRegressor` is a good default; you
  are studying the split, not the model).

Commit this pipeline; every subsequent measurement uses
the *same* model, so the only variable across
comparisons is the split.

### Part B — Demonstrate the leak (≈ 45 min)

Score the pipeline three ways, on the same held-out test
set:

1. **Random 5-fold `KFold(shuffle=True)`** on all data
   except the held-out test.
2. **`TimeSeriesSplit(n_splits=5)`** on all data except
   the held-out test, ignoring groups.
3. **Group-aware time split** — for each of 5 folds,
   hold out a contiguous late-in-time window across
   *all* groups (implement or use `sktime`'s
   `GroupTimeSeriesSplit`).

For each strategy, report:

- **The CV mean and standard deviation** of your
  chosen metric.
- **The held-out test-set score** (the test set is
  the *latest* time window, and is disjoint from all
  three CV setups' training folds).
- **The gap** between the CV mean and the test-set
  score.

You should see the gap widen as you move from strategy
3 → 2 → 1. Write a paragraph in `NOTES.md` naming which
kinds of leakage each strategy allows, referring to the
leakage taxonomy from chapter 06.

If the gaps are similar (they should not be, on a
well-chosen dataset), your dataset either has weak
group / time correlation or you have already prevented
leakage another way. Pick a different dataset or engineer
a leaky feature (see Part D stretch) to make the effect
visible.

### Part C — Nested CV with the right split (≈ 45 min)

Using the group+time split from Part B strategy 3:

- Build a `GridSearchCV` over a small hyperparameter
  grid (~5-10 candidates) using the group+time split as
  the *inner* CV.
- Wrap that in an *outer* group+time CV via
  `cross_val_score` to get an honest expected-
  performance number.
- Refit the tuned pipeline on the full dev set and
  score once on the held-out test.

Report:

- **The nested-CV mean and std.** State: "this is the
  expected performance of the *procedure*, not a
  specific model."
- **The final-refit hyperparameters.** From
  `GridSearchCV.best_params_` on the full dev set.
- **The test-set score.**
- **A sentence stating whether the nested-CV mean and
  test score agree** (within roughly one standard
  deviation). A large gap here is a diagnostic —
  investigate before publishing.

### Part D — Encode the policy (≈ 30 min)

Author `validation_config.py`:

```python
def get_test_holdout(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (dev, test) DataFrames.

    Test is the latest time window and is disjoint from
    dev by both time and group where applicable.
    """


def get_cv_splitter(policy: str):
    """
    Return the splitter object for the named policy.

    Raises if `policy` is not one of the three approved
    strategies. This module is the single place where
    the split policy lives; downstream training scripts
    import from here.
    """
```

The design goal is that no future contributor can quietly
switch to a random shuffle. Add a `pytest` test that
imports the module and asserts:

- `get_cv_splitter("random")` raises with a message
  explaining why random splits are forbidden.
- `get_cv_splitter("time")` and `("group_time")` return
  the expected splitter types.
- `get_test_holdout(df)` produces a test set whose
  minimum timestamp is strictly greater than the
  maximum dev timestamp.

### Part E — Reflection (≈ 15 min)

In `NOTES.md`:

- **The leakage taxonomy check.** For each of the five
  leakage kinds in chapter 06, either name a place it
  could have entered the pipeline (and how you guarded
  against it) or explain why the dataset structure
  rules it out.
- **The gap between naive CV and honest CV, in metric
  points.** Was it larger or smaller than you expected?
- **The most surprising fold.** Which CV fold produced
  the most anomalous metric and why? (Regime change,
  seasonal position, small validation set?)
- **One place you would go looking for leakage first**
  if you inherited a colleague's model tomorrow.

## Starter guidance

- **Cut the test set first.** Before any CV work,
  reserve the latest time window as the untouched
  holdout. The rest is dev.
- **Fit preprocessing inside the pipeline.** Any
  `StandardScaler` / imputer / encoder outside a
  `Pipeline` is a leakage bug; chapter 06 covers this.
- **Sort by time before splitting.** `TimeSeriesSplit`
  uses row order, not dates. Unsorted data produces
  meaningless folds silently.
- **Insert a `gap` if labels depend on the future.** If
  your target is "purchase in the next 30 days", the
  split needs `gap = 30 days` worth of rows between
  train and validation.
- **Score with a companion metric.** Chapter 03's rule:
  report at least two metrics from different families
  for every split strategy.

## Acceptance criteria

- The three splits in Part B produce measurably
  different metrics; the gap between naive-CV and
  test-set score shrinks monotonically across the
  three strategies (or you have documented a plausible
  exception).
- The nested CV in Part C uses group+time splits in
  both the inner and outer loops (`groups` correctly
  threaded through both).
- `validation_config.py` exists, forbids random
  splits, and passes the tests you wrote.
- `NOTES.md` contains the leakage-taxonomy audit, the
  gap analysis, the surprising-fold discussion, and
  the "where to look first" answer.
- All code committed under `exercise-02/`.

## Stretch goals (optional)

- **Engineer a leaky feature deliberately.** Add a
  feature computed on the full dataset (a target-
  encoded categorical or a rolling mean without a
  `gap`). Show its effect on the random-CV score vs
  the group+time CV score. Then correct the feature
  (fit inside the pipeline) and show the corrected
  gap.
- **Group-time split with heterogeneous horizons.**
  Some rows have a 7-day label horizon, others a
  30-day. Adjust the `gap` per-row and report on the
  additional complexity.
- **Compare against `sktime`'s `SlidingWindowSplitter`
  and `ExpandingWindowSplitter`.** Which one better
  reflects your deployment? Document the choice.
- **Bootstrap confidence intervals** for the test
  score using `sklearn.utils.resample`. Report a 95%
  CI alongside the point estimate.
- **Contribute a `GroupTimeSeriesSplit` implementation**
  as a Gist and link it in `NOTES.md`. This primitive
  is genuinely missing from scikit-learn (as of
  writing).

## What to hand in

```
exercise-02/
├── src/
│   ├── pipeline.py
│   └── validation_config.py
├── tests/
│   └── test_validation_config.py
├── notebooks/
│   ├── 01_leakage_demonstration.ipynb   # Part B
│   └── 02_nested_cv.ipynb                # Part C
├── reports/
│   └── leakage_taxonomy_audit.md         # Part E
├── data/
│   └── README.md                          # dataset source + license
├── pyproject.toml
└── NOTES.md
```

Solutions live in the paired `ml-engineer-solutions` repo
— do not peek until you have submitted yours.
