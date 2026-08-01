# 08 — Group-based and nested cross-validation

> **Reading time:** ~25 minutes.
> **Prerequisite chapter:** 06.

## Motivation

Chapter 06 named two failure modes that vanilla k-fold
does not solve. This chapter is the fix for both:

- **Group leakage** — multiple rows for the same entity
  end up on both sides of the split. The model memorises
  the entity rather than learning the task. Fixed with
  `GroupKFold` and its variants.
- **Selection bias in reported scores** — the same CV
  loop is used to tune hyperparameters and to report the
  final metric, and the reported number is biased upward.
  Fixed with **nested cross-validation**.

Both are common enough to deserve their own chapter, and
both are well-supported in scikit-learn once you know the
right primitive.

## Group-based splitting: the problem

Suppose you have a medical dataset with 500 patients and
5,000 rows (10 rows per patient on average — repeated
visits, imaging scans, etc.). A random 5-fold split on
rows produces folds where the same patient's rows are on
both sides. The model can memorise "patient 42 has this
condition" and score well on the validation fold that
also contains patient 42.

This is **group leakage**. The metric on the split reports
"how well can we memorise a patient?"; the deployment
reports something worse when the model sees a new patient
for the first time. The gap between the two is often the
entire useful signal.

Other examples:

- **Recommendations**: many sessions per user. A user's
  taste is memorisable; splitting on rows lets the model
  overfit to individuals rather than learn a general
  recommendation policy.
- **Text classification with long documents chunked into
  passages**: the same document's passages share
  vocabulary and topic; splitting on passages rather than
  on documents lets the model memorise per-document
  quirks.
- **Sensor / device data**: the same device produces
  many rows with correlated noise characteristics; split
  by device.
- **Product reviews**: multiple reviews per user or per
  product; splits should hold out users or products,
  depending on which the model is being generalised
  across.

## `GroupKFold` — the fix

`sklearn.model_selection.GroupKFold` splits so that no
group appears in more than one fold. Pass the group
identifier explicitly:

```python
from sklearn.model_selection import GroupKFold

# groups is an array-like of length n_samples, one group ID per row.
gkf = GroupKFold(n_splits=5)
for tr_idx, val_idx in gkf.split(X, y, groups=patient_id):
    train_groups = set(patient_id[tr_idx])
    val_groups   = set(patient_id[val_idx])
    assert train_groups.isdisjoint(val_groups)   # no overlap
```

Key properties:

- **No group in two folds.** The split guarantees this
  strictly.
- **Fold size is approximate.** Groups are variable-
  sized (some patients have 3 visits, others have 30),
  so fold sizes will not be exactly equal.
- **Not stratified by default.** `GroupKFold` does not
  balance classes across folds; if class balance is
  important, use `StratifiedGroupKFold` (added in
  scikit-learn 1.0).

`StratifiedGroupKFold` is the combination you almost
always want for classification with grouped data —
respects both class balance and group membership. Do
double-check the fold class balances, because with
tightly-clustered groups the strata cannot always be
made exact.

## Choosing the right group

The group is not always obvious. Ask:

- **What entity will the model be generalised across in
  production?** If deployment sees new patients, group by
  patient. If deployment sees new sessions from known
  users, group by user is *too strict* — held-out users
  make training see a mismatched distribution. If
  deployment sees repeated documents but new sentences,
  do not group at all.
- **What is the smallest atom of correlated structure?**
  If sensor readings correlate within a device but not
  across devices, device is the group. If sensor readings
  correlate within a manufacturing batch across many
  devices, batch is the group.

The test: **would the deployment ever see two rows from
the same group in split configurations?** If not, hold
groups out.

For truly leave-one-entity-out evaluation (drug discovery
on new molecules, forecasting on a new store), use
`LeaveOneGroupOut` or `LeavePGroupsOut`. These give an
even sharper generalisation bound at higher variance —
useful when there are few groups and you want to
characterise per-group performance.

## Combining group with time

Some datasets are both temporal *and* grouped: multiple
customers, each with their own time series; multiple
sensors, each with historical logs. Two questions
determine the split:

- **Do you need to generalise across time?** If the
  deployment predicts *future* events, the split must
  respect time (chapter 07).
- **Do you need to generalise across groups?** If the
  deployment predicts for *new* entities, the split must
  respect groups (this chapter).

Both together requires a per-group time split — for each
group, hold out a contiguous late-in-time window, and
combine across groups. Scikit-learn does not currently
ship a first-class `GroupTimeSeriesSplit`; the pattern is
straightforward to implement, or use `sktime` for a
production implementation.

<!-- needs-research: check if a group-aware TimeSeriesSplit
exists in scikit-learn 1.5+, and if not, cite sktime's
GroupTimeSeriesSplit as the standard implementation. -->

## Nested cross-validation: the problem

The single most common mistake in reporting CV numbers:

```python
# WRONG — inner CV picks hyperparameters, outer CV reports
# the score of the tuned model. But the outer folds' validation
# rows were used by the inner search. Selection bias.

from sklearn.model_selection import GridSearchCV, cross_val_score

search = GridSearchCV(estimator, param_grid, cv=5)
scores = cross_val_score(search, X, y, cv=5)
print(f"score = {scores.mean():.3f}")   # optimistically biased
```

The bias is subtle. Every outer-fold validation set is
also used, indirectly, by the inner search — because the
inner search runs on the outer-fold's *training* set,
picks the hyperparameter with the best inner-CV score,
and then the outer fold's validation set is scored by
the tuned model. The tuning has selected against the
noise in the outer-fold's data.

The size of the bias depends on the dataset and the size
of the hyperparameter grid, but it is typically a few
metric points. Enough to matter when comparing two
models.

## The nested pattern

Two separate cross-validation loops, one nested inside
the other:

- **Inner CV** — hyperparameter selection. For each
  outer-fold's training set, run an inner CV to pick the
  best hyperparameters.
- **Outer CV** — score reporting. Score the tuned model
  on the outer-fold's held-out set.

In scikit-learn the pattern falls out of composing the
existing primitives:

```python
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000)),
])
param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

search = GridSearchCV(
    pipe, param_grid, cv=inner, scoring="roc_auc",
)

nested_scores = cross_val_score(
    search, X, y, cv=outer, scoring="roc_auc",
)
print(f"nested CV roc_auc = "
      f"{nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

Two things to notice:

- **The inner CV** (`inner`) is what `GridSearchCV` uses
  to pick hyperparameters. **The outer CV** (`outer`) is
  what `cross_val_score` uses to report the metric of
  the tuned pipeline.
- **The two CVs are independent.** Different random
  states, different fold assignments. If they shared
  fold assignments, the inner search would still be
  optimising against the outer's held-out data.

The reported number is the honest expected performance
of a model tuned on data of this size and evaluated on
data of this size. It is not the score of a *specific*
hyperparameter setting — it is the score of the *tuning
procedure*.

## What nested CV does not do

- **It does not pick a single hyperparameter setting.** The
  outer loop picks a different set of "best"
  hyperparameters on each fold. That is expected — the
  outer loop scores the *procedure*, not a fixed model.
  To ship a single model, refit on the full dataset with
  a `GridSearchCV`-selected hyperparameter set and note
  the nested CV number as the expected performance.
- **It does not remove all bias.** The final refit on the
  full dataset uses the same procedure; the deployment
  score is expected (in probability) to be near the
  nested CV mean, but variance is not eliminated.
- **It is expensive.** Runtime is
  $\text{outer\_folds} \times \text{grid\_size} \times
  \text{inner\_folds}$ model fits. With 5 outer × 5 inner
  × 20 candidates, that is 500 fits. Budget for it. Use
  `RandomizedSearchCV` in the inner loop if grid search is
  infeasible.

## When nested CV is worth the cost

Not always. Two situations where the ~5-10× cost is
justified:

- **The result is going to be published or compared to a
  strong baseline.** The bias in single-CV tuning is
  often larger than the improvement claim; nested CV
  makes the comparison honest.
- **The dataset is small.** Small datasets have high CV
  variance, so the tuning selects noise more aggressively,
  and the bias is worse.

Two situations where it usually is not:

- **The dataset is very large.** Tuning bias shrinks as
  data grows; a single held-out validation set (rather
  than CV) is often enough for both tuning and reporting.
- **Coarse hyperparameter grids.** Two or three
  candidates for one hyperparameter → very little
  selection to bias against; single CV is defensible.

## Reporting nested CV honestly

Two numbers to report:

- **The nested CV mean and standard deviation.** This is
  the honest expected performance.
- **The final refit's hyperparameters** — chosen by
  `GridSearchCV.fit(X, y).best_params_` after nested
  evaluation. State: "we selected these hyperparameters by
  running the same procedure on the full dev set; the
  nested CV score is the expected out-of-sample
  performance of a model chosen this way."

If different outer folds picked different "best"
hyperparameters, that variability *is* a diagnostic: the
loss landscape is flat, or the model is not sensitive
enough for these hyperparameters to matter, or the data
size is too small for stable selection.

## Combining nested CV with group / time splits

The two ideas compose. Pass the group-aware or time-aware
splitter as the outer and/or inner CV:

```python
from sklearn.model_selection import GroupKFold

outer = GroupKFold(n_splits=5)
inner = GroupKFold(n_splits=5)   # different subset each fold

# The groups vector must be passed to both loops.
search = GridSearchCV(pipe, param_grid, cv=inner)
nested_scores = cross_val_score(
    search, X, y, cv=outer, groups=groups,
    fit_params={"groups": groups},   # <-- passed to inner CV
)
```

Passing `groups` to the inner search requires either the
`fit_params` bridge shown or a wrapper that intercepts
`groups` — the API is not as clean as one would hope.
[`sklearn.model_selection.HalvingGridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html)
and third-party libraries like `scikit-optimize` sometimes
handle this more cleanly; verify before shipping.

<!-- needs-research: verify current scikit-learn API for
passing `groups` from an outer `cross_val_score` to an
inner `GridSearchCV.cv=GroupKFold(...)` cleanly. As of
sklearn 1.4 this required `fit_params`; check 1.5+. -->

## A minimum-viable pattern

The template for a project where the data has group
structure and hyperparameters matter:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_val_score,
    train_test_split,
)


# 1. Split off a test set that also respects groups.
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
tr_idx, te_idx = next(gss.split(X, y, groups=groups))
X_dev, X_test = X.iloc[tr_idx], X.iloc[te_idx]
y_dev, y_test = y.iloc[tr_idx], y.iloc[te_idx]
groups_dev, groups_test = groups.iloc[tr_idx], groups.iloc[te_idx]

# 2. Nested group CV for tuning + honest reporting.
inner = StratifiedGroupKFold(n_splits=5)
outer = StratifiedGroupKFold(n_splits=5)

pipe = HistGradientBoostingClassifier(random_state=0)
param_grid = {"max_leaf_nodes": [15, 31, 63], "learning_rate": [0.05, 0.1]}

search = GridSearchCV(pipe, param_grid, cv=inner, scoring="average_precision")
nested = cross_val_score(
    search, X_dev, y_dev, groups=groups_dev,
    cv=outer, scoring="average_precision",
    fit_params={"groups": groups_dev},
)
print(f"nested CV ap = {nested.mean():.3f} ± {nested.std():.3f}")

# 3. Fit the final tuned model on all dev data.
search.fit(X_dev, y_dev, groups=groups_dev)
best = search.best_estimator_
print(f"chosen params = {search.best_params_}")

# 4. Report on the untouched, group-disjoint test set.
from sklearn.metrics import average_precision_score
proba = best.predict_proba(X_test)[:, 1]
print(f"test ap = {average_precision_score(y_test, proba):.3f}")
```

Five features:

- **Test set is group-disjoint** from dev (via
  `GroupShuffleSplit`) — same rule as CV.
- **Inner and outer CVs are both group-aware.**
- **Two numbers are reported.** The nested CV mean (the
  procedure's expected score) and the test score (the
  frozen model's realised score). They should agree
  within a few standard deviations; a large gap is a
  diagnostic that either dev or test is unrepresentative.
- **Hyperparameters are reported explicitly.**
- **The tuned model is refit on the full dev set** for
  eventual deployment.

## Common pitfalls

1. **Passing `groups` to the outer loop but not the
   inner.** The inner search leaks groups; the outer
   metric is optimistic even though the outer split is
   group-disjoint.
2. **Choosing groups that are too coarse.** Grouping by
   country when the dataset spans years and thousands of
   country-specific rows leaves you with 5 groups, no
   power, and a 5-fold split that is a leave-one-country-
   out.
3. **Choosing groups that are too fine.** Grouping by
   session when many sessions per user is fine and you
   care about generalising across users. Over-strict
   grouping wastes signal.
4. **Comparing nested CV means as if they were paired.**
   Two models' nested CV means differ by 0.005; you
   conclude one is better. But the fold assignments were
   different; the difference is within CV noise. Use the
   Nadeau–Bengio corrected test or a paired setup where
   both models are scored on the same folds.
5. **Assuming `StratifiedGroupKFold` produces exactly
   stratified folds.** With few groups or highly
   imbalanced group sizes, the strata cannot always be
   made exact. Inspect the class balance per fold.
6. **Reporting only the outer CV without stating that
   hyperparameters differed by fold.** The outer number
   is the procedure's expected score, not the score of a
   specific hyperparameter setting. State this.

## Summary

- Standard k-fold fails when the data has group
  structure (multiple rows per entity) — use
  `GroupKFold` / `StratifiedGroupKFold` /
  `LeaveOneGroupOut`; pick the group to match the
  deployment's generalisation target.
- Combine group and time splits when the data has both;
  scikit-learn does not ship this primitive, but the
  pattern is straightforward.
- Standard "tune with CV, report with CV" produces
  optimistically biased scores. **Nested CV** fixes it:
  inner CV for hyperparameter selection, outer CV for
  score reporting.
- The nested CV score is the expected score of the
  *procedure*, not a specific fitted model. Refit the
  procedure on the full dev set to get the deployable
  model; the nested score is its expected performance.
- Cost of nested CV is $\text{outer} \times \text{inner}
  \times \text{grid}$ fits. Justified for publication-
  quality comparisons and small-dataset regimes.

## Where this goes next

Chapter 09 turns to baselines — the reference models
against which any nested CV mean should be compared, and
without which "our model scores 0.83" is a number without
context.
