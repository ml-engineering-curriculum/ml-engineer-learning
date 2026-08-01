# 06 — Splits, leakage, and cross-validation

> **Reading time:** ~35 minutes.
> **Prerequisite chapter:** 01.

## Motivation

The metric is a function of the split. Change the split
and the number moves — sometimes cosmetically, often
catastrophically. Getting the split wrong is the single
most common source of "the model looked great in
development and disappointed in production" stories.

This chapter is the validation-strategy foundation for the
rest of the module. Three ideas do most of the work:

1. **Held-out data must genuinely be unseen.** The
   validation and test sets must not be reachable from the
   training set through *any* pipeline path — direct
   overlap, preprocessing that fits on all rows, target
   leakage through features, correlated rows in different
   splits.
2. **Cross-validation is a variance-reduction tool, not a
   leakage cure.** It repeats the same random split
   several times and averages the metric so the reported
   number is more stable. It does nothing about the ways
   your split is systematically wrong.
3. **The right split mimics the deployment.** If the model
   will predict tomorrow's events from today's data, the
   split has to hold out future data. If the model will
   predict for a user it has never seen, users must be
   split, not rows. Chapters 07 and 08 make these
   concrete; this chapter names them.

## The leakage taxonomy

Leakage is any pipeline path that leaks information from
the target (or from the held-out data) into the training
signal, producing an optimistic score that does not
generalise. Five families cover almost every case seen in
practice:

### 1. Target leakage — the future is in the features

The training data contains a feature that would not be
available at prediction time. Two representative examples:

- **Fraud detection**: a `days_since_flagged` feature that
  is only populated *after* a fraud analyst reviews the
  transaction. At training time it is on every fraud row;
  at prediction time it is on none.
- **Churn prediction**: a `days_until_last_login` feature
  computed against a snapshot date that is after the
  observation window. It correlates perfectly with the
  target because the target is derived from the same
  future observation.

The give-away: **the feature is causally downstream of the
target.** If including it makes the model implausibly
good (F1 = 0.99 on a hard problem), that is the alarm.

**How to catch it:**

- Every feature has to answer the question "would this
  value have existed at time-of-prediction?" A **feature
  dictionary** that records the source table and the
  timestamp semantics for every column pays for itself
  the first time it surfaces one of these.
- Look at the per-feature importance from the trained
  model. A single feature carrying 90% of the importance
  on a problem that should be hard is a leakage flag.

### 2. Train/test contamination — the same rows are on both sides

The simplest and most embarrassing form. Duplicates in the
raw data end up on both sides of a random split. Two rows
that are trivially derivable from each other (the same
event logged twice, the same user's session split into two
rows) do too.

**How to catch it:** deduplicate before splitting. On any
dataset where a user, session, patient, or entity can
produce multiple rows, dedupe by the entity key and either
keep one row per entity or use group-based splits (chapter
08).

### 3. Group leakage — correlated rows on both sides

The subtler cousin. Rows are not literal duplicates but
are strongly correlated because they share an underlying
entity. Common examples:

- **Medical imaging**: two scans of the same patient in
  train and test. The patient's anatomy is shared; the
  model can memorise the patient rather than the disease.
- **Recommendations**: multiple sessions for the same
  user in train and test. The user's taste vector is
  shared; the model learns the user, not the task.
- **Text**: paragraphs from the same document in train
  and test. The topic vocabulary and stylistic
  fingerprint carry over.

The metric on such a split reports "how well can we
memorise the group?" The deployment reports something
much worse when the model sees a new group. Group-based
splits (chapter 08) are the fix.

### 4. Temporal leakage — the past sees the future

The training set contains rows from *after* the validation
or test rows. On i.i.d. data this is fine — the samples
are exchangeable. On temporal data it is a catastrophe.

Two flavours:

- **Direct temporal leakage** — a random split shuffles
  rows across time; some training rows are from later
  dates than some test rows. The model learns "how the
  world was in period X+1" and reports on period X.
- **Indirect temporal leakage** — a feature that
  encodes future information indirectly (a rolling mean
  computed on the whole dataset before the split; a
  target-encoded categorical feature computed on the full
  training set including future rows).

**How to catch it:** if the data has a `date` column and
the model's deployment scenario is "predict future from
past", the validation split must respect time. Chapter 07
covers walk-forward and expanding-window splits.

### 5. Preprocessing leakage — the pipeline saw the test set

The classic scikit-learn footgun. Fit any preprocessing
step (`StandardScaler`, `PCA`, `SelectKBest`,
`SimpleImputer`, target encoder, feature selector) on the
full dataset *before* splitting. The scaler now knows the
mean of the test set; the feature selector chose features
based on their signal on the test set; the target encoder
computed category means using future targets.

The remedy is universal:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression()),
])
```

A scikit-learn `Pipeline` fits every step on the training
fold only during cross-validation and applies the fitted
transformer to the validation fold. Wrap **all**
preprocessing that has learned parameters inside a
`Pipeline` (or `ColumnTransformer`). "Just scale before
you split" is the paved road to leakage.

`sklearn.model_selection.cross_val_score` on a `Pipeline`
does the right thing automatically; on raw arrays after
manual preprocessing it does not.

## The three-way split

For any project where the model will be **published, shipped,
or compared to a baseline you care about**, you want three
partitions:

- **Train** — the data the model fits on.
- **Validation** — the data used to *choose* the model
  (hyperparameters, features, thresholds, early stopping).
  Touched many times.
- **Test** — a single holdout, cut before any of the above,
  never seen until the final report. Touched *once*.

Cross-validation replaces the train / validation split with
$k$ folds that let you use all non-test data for both
purposes without a single "wasted" holdout. The **test set
sits outside the CV loop**.

The confusion this fixes: after fifty CV runs you have
implicitly tuned to the CV splits. The CV-mean number is
biased upward — you cannot avoid this, only bound it. The
untouched test set is the honesty check.

```python
from sklearn.model_selection import train_test_split, KFold

# Cut the test set first, and never touch it until the end.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)

# Use dev data for the whole model-development loop, via CV.
cv = KFold(n_splits=5, shuffle=True, random_state=0)
for tr_idx, val_idx in cv.split(X_dev):
    ...  # fit on tr_idx, score on val_idx
```

## Standard k-fold cross-validation

The default. Split the dev set into $k$ equal folds; for
each fold, train on the other $k-1$ folds and score on the
held-out fold. Report the mean and standard deviation of
the metric.

**Choosing $k$:**

- **$k = 5$** — a good default. Bias / variance trade-off
  is reasonable; runtime is $5 \times$ single-model fit.
- **$k = 10$** — slightly lower bias, higher variance,
  $10 \times$ runtime. Standard in the ISLR / classical
  ML curriculum.
- **$k = n$ (leave-one-out)** — near-zero bias, high
  variance, and $n \times$ runtime. Rarely worth it
  outside very small datasets.
- **$k = 3$** — when compute is scarce and you have lots
  of data. Higher bias but the mean is still informative
  as long as each fold is large enough for a stable
  metric.

`sklearn.model_selection.KFold(n_splits=5, shuffle=True,
random_state=0)` is the workhorse. `shuffle=True` is
important — the default `shuffle=False` produces
contiguous folds, which combined with any hidden ordering
in the data (temporal, alphabetical, imbalance-swept) can
produce catastrophic folds.

## Stratified k-fold

Class balance in each fold matches the class balance in
the full dataset. Use it for classification, always.
Without stratification, a $k=5$ split on a 5% positive
class can produce folds with 3% and 7% positive rate — a
big variance source in the reported metric.

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
```

`cross_val_score` on a classifier defaults to
`StratifiedKFold` automatically — but only if you pass the
classifier directly, not if you wrap it in something that
looks like a regressor to scikit-learn. When in doubt, pass
the CV object explicitly.

## Repeated k-fold — reducing variance further

For small datasets, the fold assignment itself is a
significant variance source. Repeat the CV several times
with different fold assignments and average:

```python
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=3, random_state=0
)
```

Cost: $n\_repeats \times$ the base cost. Payoff: the
reported metric's confidence interval is tighter, which
matters when comparing two models whose scores differ by
less than the fold-to-fold noise.

## Reporting the metric — mean, spread, and confidence

A single number from CV is undercommunication. Report:

- **Mean.** The point estimate.
- **Standard deviation across folds.** A rough proxy for
  the metric's sampling noise.
- **Individual fold scores.** Look at them. A mean of
  0.85 with fold scores `[0.94, 0.93, 0.92, 0.91, 0.55]`
  is a red flag — one fold is doing something
  systematically different. Investigate that fold before
  reporting the mean.

For paired comparison ("model A vs model B, which is
better?"), the naive comparison of fold means often
overstates significance. The correct test is a **paired**
one (each fold's score for A minus B), and even that is
biased on standard CV because folds share training data.
Nadeau & Bengio (2003) proposed a **corrected resampled
$t$-test** for this case; scikit-learn's documentation on
model evaluation cites it explicitly.

<!-- needs-research: cite Nadeau & Bengio (2003)
"Inference for the Generalization Error" and the sklearn
docs page on statistical model comparison for the
corrected resampled t-test formulation. -->

## When simple CV is wrong

Standard k-fold assumes rows are i.i.d. — exchangeable
under random reordering. That assumption fails in at
least three common situations, each of which has its own
CV variant:

- **Data has a time index.** Chapter 07: walk-forward /
  expanding-window / rolling-origin splits, never
  shuffled.
- **Data has groups.** Chapter 08: `GroupKFold` /
  `StratifiedGroupKFold` / `LeaveOneGroupOut`.
- **You are tuning hyperparameters.** Chapter 08's second
  half: **nested CV** so the inner loop's tuning doesn't
  bias the outer loop's score.

## The nested CV preview

The single most common failure of CV in practice:

```python
# BROKEN — the reported score is optimistically biased
from sklearn.model_selection import GridSearchCV, cross_val_score

grid = GridSearchCV(estimator, param_grid, cv=5)
score = cross_val_score(grid, X, y, cv=5).mean()   # ???
```

`GridSearchCV` runs an inner 5-fold CV to pick
hyperparameters. `cross_val_score` runs an outer 5-fold
CV. The outer folds' validation sets *are used by the
inner search* to pick hyperparameters — the score you
report includes selection bias.

The fix — proper **nested CV** — is chapter 08. For now,
the takeaway: if you tune with CV *and* report with CV,
they need to be separate splits.

## A minimum-viable CV pattern

The concrete template most of your work should look like:

```python
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Cut a test set and lock it away.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)

# 2. Assemble the pipeline. All preprocessing inside the pipe.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000)),
])

# 3. Cross-validate on the dev data with stratified folds.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_validate(
    pipe, X_dev, y_dev,
    cv=cv,
    scoring=["roc_auc", "average_precision", "f1"],
    return_train_score=True,
)

for metric in ["test_roc_auc", "test_average_precision", "test_f1"]:
    values = scores[metric]
    print(f"{metric}: {values.mean():.3f} ± {values.std():.3f}")

# 4. Only after model selection is frozen: report on the test set.
pipe.fit(X_dev, y_dev)
final_score = pipe.score(X_test, y_test)  # or whatever metric
print(f"test score = {final_score:.3f}")
```

Five features to notice:

- **The test set is cut once.** Nothing in step 3 touches
  it. `test_size=0.2` is a starting point — bigger if the
  dataset is large enough to spare it and the metric on
  test needs low variance.
- **The pipeline holds every learned preprocessing step.**
  No `fit_transform` outside the CV loop.
- **`stratify=y`** in the initial split; `StratifiedKFold`
  in the CV loop. Class balance preserved end to end.
- **Multiple scoring metrics** via `cross_validate`
  (which returns a dict). One-metric CV hides the
  companion metric that a report needs.
- **Train scores returned** (`return_train_score=True`)
  — the gap between train and test score across folds is
  a rough overfitting diagnostic.

## Common pitfalls

1. **Scaling / imputing / feature-selecting before the
   split.** Universal beginner mistake. Wrap in a
   `Pipeline`, always.
2. **Shuffled k-fold on time-series data.** The metric is
   inflated by future information. Use time-series CV
   (chapter 07).
3. **`shuffle=False` when data is not i.i.d. ordered.**
   The folds become the first fifth, second fifth, etc.,
   of the data — which are often systematically different
   subpopulations. Combine with a hidden ordering and the
   metric is destroyed.
4. **Ignoring group structure.** If each user / patient /
   customer produces multiple rows, standard CV leaks
   groups across folds. Chapter 08.
5. **Tuning hyperparameters with the same CV as the
   reported CV.** Selection bias inflates the reported
   score. Use nested CV.
6. **Reporting only the CV mean.** Publish the standard
   deviation and — if the comparison is close — the
   individual fold scores.
7. **Using leave-one-out on anything larger than a few
   hundred samples.** Very high variance; very expensive;
   rarely more informative than 10-fold.
8. **Refitting on the full dataset for deployment without
   noting it.** After all CV is done, you typically refit
   on `X_dev` (or `X` including test) for deployment.
   That model has no measured score — its "expected"
   score is the CV mean, but it is not directly measured.
   State this explicitly in the report.

## Summary

- Leakage comes in five forms — target, contamination,
  group, temporal, preprocessing — and standard CV catches
  none of them.
- A three-way split (train / validation / test) is the
  minimum; CV replaces train / validation with folds. The
  test set sits outside the CV loop and is touched once.
- Use `StratifiedKFold` for classification, wrap all
  preprocessing in a `Pipeline`, and report mean +
  standard deviation across folds.
- CV assumes i.i.d. rows. When that fails (time, groups,
  hyperparameter tuning), reach for the specialised
  variants in chapters 07 and 08.

## Where this goes next

Chapter 07 covers time-based validation — walk-forward,
expanding-window, and the "you cannot shuffle a
time-series" rule that makes it possible to evaluate a
forecasting model honestly.
