# 07 — Cross-validation strategies

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** `mod-103` chapter 06 (leakage), and
> chapters 02–06 of this module.

## Motivation

A held-out test set gives you one number. Cross-validation gives you
several numbers plus a variance estimate.

That extra information is not a nice-to-have. It is what lets you
say "model A is better than model B" with a straight face when the
two are 0.5 percentage points apart on a single hold-out. It is
what makes hyperparameter search (chapter 08) trustable rather than
p-hacked. And on small datasets — which is most datasets — it is
what makes the difference between "a plausible estimate of
generalisation error" and "an accidentally-optimistic number the
model won't reproduce in production."

But cross-validation is only as honest as the split it uses. A
k-fold CV that randomly shuffles rows will happily leak future data
into the past on a time-series problem, or spread a user's rows
across every fold on a grouped one. This chapter is the split-picking
discipline: pick the split that matches the shape of your data, and
the CV score will match your deployed model's behaviour.

## The base cross-validator: `KFold`

`sklearn.model_selection.KFold` splits the dataset into `n_splits`
folds. In each of `n_splits` iterations, one fold is the validation
set and the rest are training. The final CV score is typically the
mean (with the std as an uncertainty estimate).

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

Two settings matter:

- **`shuffle=True`.** With `shuffle=False`, folds are taken as
  contiguous slices of `X` — which is disastrous if your data is
  sorted by target or by time. Default is `False` for backward-
  compatibility reasons; always set `shuffle=True` unless the row
  order is meaningful (see `TimeSeriesSplit` below).
- **`random_state`.** Reproducibility. Set an integer.

Choosing `n_splits`:

- **5** is the standard default. Cheap, low-variance, easy to
  interpret.
- **10** if the dataset is small (< ~5000 rows) — each fold sees
  more of the data.
- **3** if fits are very expensive and you are exploring.
- **`n` (leave-one-out, LOOCV)** — mostly a theoretical construct;
  high variance and expensive. Skip it.

Repeated CV (`RepeatedKFold`) — running k-fold with multiple
`random_state`s — is worth it when the score variance across folds
is high (small dataset, imbalanced classes).

## `StratifiedKFold`: for imbalanced classification

Random k-fold on a 5%-positive-class problem occasionally puts a
fold with 2% positives and another with 8%. That is not variance
you want in your CV.

`StratifiedKFold` preserves the class distribution in each fold:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
```

`cross_val_score` (and friends) use `StratifiedKFold` by default when
`y` is discrete and the estimator is a classifier. Making the choice
explicit is still worth it — for documentation, and because the
default has shifted across sklearn versions.

For regression, `sklearn.model_selection.KFold` is the base; there is
no built-in equivalent to stratification on a continuous target
(though you can construct one by binning `y` and stratifying on the
bin).

## `GroupKFold`: for grouped rows

If the dataset has multiple rows per entity (multiple sessions per
user, multiple images per patient, multiple invoices per company)
and at prediction time the model will see new entities, random
splitting **cheats**. Rows from the same user appear in both train
and validation folds; the model memorises per-user information; CV
overstates its generalisation to new users.

`GroupKFold` guarantees no group appears in more than one fold:

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
scores = cross_val_score(
    pipe, X, y, cv=cv,
    groups=X["user_id"],   # required
    scoring="roc_auc",
    n_jobs=-1,
)
```

Notable properties:

- **Deterministic.** `GroupKFold` does not have a `random_state` or
  `shuffle=` — folds are assigned to approximately balance the
  group counts. Since scikit-learn 1.4, `StratifiedGroupKFold`
  adds stratification on top; `GroupShuffleSplit` gives you a
  randomised group-based holdout.
- **Folds are not equal in row count.** If group sizes vary
  wildly, some folds have more rows than others. The mean CV
  score is still a reasonable summary.

The choice between `KFold` and `GroupKFold` is not stylistic; it is
substantive. On grouped data, a random split can inflate an accuracy
metric by tens of percentage points relative to the group-aware
split. Do the group-aware one.

## `TimeSeriesSplit`: for time-ordered data

Time-series data has a natural forward direction. Rows from the
future must not be in the training fold when validating on the past.

`TimeSeriesSplit` produces expanding-window folds:

```
Fold 1: [    train    ] [val] ... ... ... ...
Fold 2: [        train        ] [val] ... ...
Fold 3: [            train            ] [val]
```

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5, test_size=None, gap=0)
for train_idx, val_idx in cv.split(X):
    ...
```

Options:

- **`test_size`** — size of each validation fold (in rows). Default
  is `n_samples // (n_splits + 1)`.
- **`gap`** — number of rows to leave between train and validation.
  Useful when there is a lag between "feature time" and "label
  time" that would otherwise leak through.
- **`max_train_size`** — cap on training-window size. Use for
  rolling-window CV instead of expanding-window.

Rules:

- **The data must already be sorted by time.** `TimeSeriesSplit`
  splits on row order, not on any timestamp column.
- **The train window is always before the validation window.**
  Every fold respects the temporal boundary.
- **The validation windows do not cover all the data.** The first
  chunk of rows never appears in any validation set (there is no
  earlier "train" for it).

For heavier machinery — grouped time-series (both time-ordered *and*
per-entity), embargo periods (a gap between train and validation to
avoid autocorrelation leakage), purge-and-embargo CV — reach for
`sktime` or a hand-rolled splitter. `mod-107` covers evaluation
pathologies specific to time-series in more depth.

## `LeaveOneGroupOut`, `LeavePGroupsOut`, `LeaveOneOut`

Specialty splitters:

- **`LeaveOneGroupOut`** — one fold per group. The extreme case of
  `GroupKFold`. Use when the groups are the natural evaluation
  unit (patients in a medical study, hospitals in a multi-site
  benchmark).
- **`LeavePGroupsOut`** — leave out combinations of `p` groups.
  Expensive; rarely needed.
- **`LeaveOneOut`** — one fold per row. Theoretically the least
  biased, empirically the highest variance, always expensive.
  Skip.

## `cross_val_score` vs. `cross_validate`

`cross_val_score` returns a 1-D array of per-fold scores.
`cross_validate` returns a dict with per-fold train and validation
scores, fit times, and score times — much more useful for diagnosing
what happened.

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    pipe, X, y, cv=cv,
    scoring=["roc_auc", "average_precision", "neg_log_loss"],
    return_train_score=True,
    return_estimator=True,
    n_jobs=-1,
)

import pandas as pd
print(pd.DataFrame({
    "val_auc":    results["test_roc_auc"],
    "train_auc":  results["train_roc_auc"],
    "fit_secs":   results["fit_time"],
}).round(3))
```

Two habits worth building:

- **`return_train_score=True`.** A large gap between train and
  validation scores is the overfitting signal. Small gap + low
  score = high bias / underfitting. Large gap + low val = high
  variance / overfitting.
- **Multiple scoring metrics.** `average_precision` (PR-AUC) and
  `roc_auc` disagree on imbalanced problems; `neg_log_loss`
  disagrees with both on miscalibrated ones. Log all three at
  fit time.

## Where the pipeline goes

**Every preprocessing step must be inside the `Pipeline` that
`cross_val_score` fits.** Otherwise, information from the
validation fold leaks into the training fold via the preprocessor.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000)),
])

scores = cross_val_score(pipe, X, y, cv=cv)  # scaler is refit inside each fold
```

If you `SimpleImputer().fit_transform(X)` before `cross_val_score`,
the imputer has seen every row's mean/median — including the
validation rows — and CV overstates generalisation. This mistake is
almost too obvious to mention until you have made it three times.

## Nested CV: when you tune and evaluate at once

**One CV loop for hyperparameter tuning, another CV loop for
generalisation estimation.** Otherwise the tuned model's CV score is
optimistic — you have optimised on the same data you are reporting.

`GridSearchCV` inside `cross_val_score`:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

grid = GridSearchCV(pipe, param_grid={"clf__C": [0.1, 1, 10]}, cv=3)
outer_scores = cross_val_score(grid, X, y, cv=5, n_jobs=-1)
print(f"Nested CV score: {outer_scores.mean():.4f} ± {outer_scores.std():.4f}")
```

Cost: `n_outer × n_inner` fits per hyperparameter combination.
Expensive; typically reserved for small datasets and papers, not
production tuning. In practice, most teams use a single CV for
hyperparameter selection and a single locked hold-out set for the
final estimate. That is not statistically pristine — the CV score is
slightly optimistic due to selection — but it is what shipping teams
do. Chapter 08 covers when to spend the budget on nested CV.

## How many folds, and does it matter?

The number of folds affects the CV score's bias–variance trade-off:

- **Few folds (3):** each training set is only 67% of the data.
  The estimator is under-trained; CV score is *pessimistic*
  relative to a full-data fit. Low variance across folds.
- **Many folds (10):** each training set is 90% of the data. CV
  score approximates the full-data model's performance well. Higher
  variance because validation folds are small.
- **`n = LOOCV`:** each training set is 99.9% of the data. Nearly
  unbiased; extremely high variance and very expensive.

5 is a compromise that works on most datasets. 10 is worth using on
datasets smaller than ~10k rows. Do not chase LOOCV.

## Scoring: the third argument to CV

Every CV function takes a `scoring=` argument that decides what to
maximise. The full catalogue is at `sklearn.metrics.get_scorer_names()`.

The ones you will actually use, roughly:

**Classification:**

- `accuracy`
- `roc_auc`, `roc_auc_ovr`, `roc_auc_ovo`
- `average_precision`
- `f1`, `f1_micro`, `f1_macro`, `f1_weighted`
- `neg_log_loss`
- `neg_brier_score`

**Regression:**

- `r2` (default for regressors)
- `neg_mean_squared_error`, `neg_root_mean_squared_error`
- `neg_mean_absolute_error`
- `neg_mean_absolute_percentage_error`

The `neg_` prefix is because `sklearn` maximises scores; a negated
loss has "higher is better" semantics.

Custom scoring:

```python
from sklearn.metrics import make_scorer, f1_score
score = make_scorer(f1_score, pos_label=1, average="binary")
cross_val_score(pipe, X, y, cv=cv, scoring=score)
```

Pick the metric that matches how you will evaluate the model in
production — the CV metric drives every model-selection decision
downstream. `mod-107` has the extended argument for which metric to
pick when.

## Decision tree for the split

```
Is there a time axis in the data?
├── Yes: TimeSeriesSplit (add `gap` if features/labels have a lag)
└── No:
    Are there multiple rows per entity, and will inference see new entities?
    ├── Yes: GroupKFold / StratifiedGroupKFold / GroupShuffleSplit
    └── No:
        Is the target imbalanced (classification)?
        ├── Yes: StratifiedKFold
        └── No:  KFold
```

If two conditions hold — time-ordered AND grouped — you need a
group-time-series splitter (`sktime`'s `SlidingWindowSplitter` with
`GroupShuffleSplit`-style grouping, or a hand-rolled one).

## A worked example, comparing splits

```python
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
y = (y == ">50K").astype(int)
X = X.select_dtypes(include="number").fillna(0)     # keep it simple

pipe = Pipeline([("scale", StandardScaler()),
                 ("clf",   LogisticRegression(max_iter=1000))])

# Adult has no natural groups or time — use StratifiedKFold.
for name, cv in [
    ("KFold",              KFold(5, shuffle=True, random_state=0)),
    ("StratifiedKFold",    StratifiedKFold(5, shuffle=True, random_state=0)),
]:
    s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"{name:20s}  AUC = {s.mean():.4f} ± {s.std():.4f}")
```

On the Adult dataset the difference between `KFold` and
`StratifiedKFold` is small (the classes are ~24% / 76%). On a 1%-
positive dataset the difference in variance across folds becomes
much larger.

## Pitfalls collected

1. **`shuffle=False` by accident.** `KFold`'s default; on
   target-sorted or time-sorted data it produces catastrophic CV
   estimates. Always `shuffle=True` unless you meant `TimeSeriesSplit`.
2. **Fitting a preprocessor outside `cross_val_score`.** Leaks
   validation-fold information into training. Wrap everything in
   a `Pipeline`.
3. **Using `KFold` on grouped data.** Optimistic by 5–30 percentage
   points, depending on how much per-group memorisation is
   possible.
4. **Using `KFold` (random) on time-series.** Future rows in the
   training fold; test rows from the past that the model in
   production would never see. Use `TimeSeriesSplit`.
5. **Reporting a mean CV score without a std.** A CV score of 0.85
   ± 0.01 is a very different claim from 0.85 ± 0.10; the second
   might not even be meaningfully different from 0.75.
6. **Nested CV with the same `random_state` on inner and outer
   splits.** Some validation rows appear in an inner training set
   in the same fold. Set independent `random_state`s.
7. **`scoring="accuracy"` on an imbalanced binary problem.**
   Predicting "always negative" scores 95% on a 5% positive class;
   the metric hides the problem. Use `roc_auc` or PR-AUC.
8. **Early stopping (chapter 06) that peeks at the CV validation
   fold.** If the estimator uses `eval_set=(X_val, y_val)` where
   `X_val, y_val` is the current CV validation fold, you have
   made early stopping part of hyperparameter tuning on the score
   you are reporting. Use a nested inner split for early stopping
   or set aside a separate `eval_set` at the top level.

## Summary

- `KFold` (with `shuffle=True`) is the default. `StratifiedKFold`
  for imbalanced classification. `GroupKFold` for grouped rows.
  `TimeSeriesSplit` for time-ordered data.
- The split must match the shape of the data at inference time. A
  mismatched split can inflate the CV score by tens of percentage
  points.
- All preprocessing goes inside a `Pipeline`; `cross_val_score`
  refits the pipeline per fold.
- Report the mean *and* the standard deviation across folds.
- Nested CV isolates hyperparameter tuning from generalisation
  estimation. In practice, most teams use a single CV for tuning
  and a single held-out test set for the final number.

## Where this goes next

Chapter 08 covers the hyperparameter search machinery — grid,
random, halving, Bayesian (Optuna) — that all sit on top of the CV
loop this chapter defined.
