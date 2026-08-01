# 07 — Time-based validation

> **Reading time:** ~25 minutes.
> **Prerequisite chapter:** 06.

## Motivation

Time-series data breaks the assumption every random split
depends on: that the rows are exchangeable. When today's
row is correlated with yesterday's, when the model will
be used to predict *tomorrow* from data available *today*,
a shuffled cross-validation reports on a world that will
never exist.

The rule is simple: **the training set may not contain
any observation whose timestamp is after the validation
or test observations.** Every time-series validation
scheme is a variant of that rule. This chapter walks the
three schemes you will use (walk-forward, expanding
window, rolling origin), covers the subtleties (gap
periods, target-lag leakage, seasonality), and shows how
to implement each with scikit-learn primitives.

## Why random splits fail on time-series

Two mechanisms make random-split CV optimistic on temporal
data:

**Autocorrelation.** In most time series, `y[t]` is
correlated with `y[t-1]`. A random split puts many
adjacent rows on opposite sides of the fold; the training
set contains rows that are essentially "the answers" for
the nearby validation rows. The model learns to interpolate
between known adjacent points — a task that never happens
in production, where every prediction is strictly
extrapolative.

**Regime changes.** Consumer behaviour shifts after
lockdowns; fraud patterns evolve as attackers adapt;
demand curves change with new products. A random split
lets the model see rows from every regime in training and
test on rows from every regime. Production sees only the
newest regime after the last training update.

Both effects inflate the reported metric. The gap between
random-split CV and time-honest CV is often ten or more
metric points on real forecasting problems.

## The core primitive: forward-only splits

Every time-series CV scheme cuts a **prediction window**
of length $h$ (the horizon) at some time $t$, using all
observations before $t$ for training. Vary $t$ and you
have a fold assignment.

Two axes of variation:

- **Training window** — expanding (use all history up to
  $t$) or rolling (use a fixed-width history preceding
  $t$).
- **Prediction window** — one point ahead, a small
  horizon (e.g. next 7 days), or a longer batch (next
  quarter).

Combine them:

| Training | Prediction | Name |
|---|---|---|
| Expanding (all history so far) | 1 step | Walk-forward validation |
| Expanding | $h$ steps | Expanding-window CV |
| Rolling (fixed width) | 1 or $h$ steps | Rolling-origin CV |
| Expanding | 1 step, refit each step | Recursive one-step-ahead |

The choice depends on how the model will be used in
production. If the deployment refits weekly on all
history and predicts one day ahead, walk-forward with
$h = 1$ and expanding training matches. If the deployment
uses a fixed one-year window and predicts a week ahead,
rolling-origin with a one-year training window and
$h = 7$ matches.

## Scikit-learn's `TimeSeriesSplit`

Scikit-learn ships one time-series splitter. Read the
docstring carefully; the defaults are subtle.

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(
    n_splits=5,
    max_train_size=None,   # None = expanding window
    test_size=None,        # None = auto-compute equal-size folds
    gap=0,                 # gap between train end and test start
)

for train_idx, test_idx in cv.split(X):
    print(f"train {train_idx[0]}..{train_idx[-1]}   "
          f"test {test_idx[0]}..{test_idx[-1]}")
```

Two knobs worth memorising:

- **`max_train_size`.** `None` → expanding window (each
  fold's training set grows). An integer → rolling window
  (each fold's training set is capped at that many rows).
- **`gap`.** Positive integer → skip that many rows
  between the end of the training set and the start of
  the test set. This is where you handle target-lag
  leakage (below).

**Assumption:** the DataFrame is sorted by time. If it is
not, `TimeSeriesSplit` will produce nonsense splits
silently. Sort explicitly before splitting.

## Gap periods and target-lag leakage

Some target definitions require future information to
label. A "next-30-day churn" label for a customer at time
$t$ requires observing that customer through $t + 30$. If
you use that row in training with an observation
timestamp of $t$, you are implicitly using data through
$t + 30$.

The fix: **insert a gap between training and testing
equal to the maximum label horizon.** For a 30-day churn
label with daily observations, `gap = 30` in
`TimeSeriesSplit`. Without the gap, the training set at
time $t$ has labels that depend on the future values now
being predicted.

The same trap appears in feature engineering:

- **Rolling-window features** with window $w$ leak $w$
  future rows if computed on the full dataset before
  splitting.
- **Target-encoded categorical features** where the
  category mean is computed with the target — the
  future's target contaminates the past's feature.
- **Lagged features** that reach into the "future"
  relative to the observation timestamp.

The countermeasure is universal: compute features using
**only data available at each row's observation
timestamp**. Doing this efficiently for large datasets is
what feature stores (Feast, Tecton) exist for; on a small
scale, a strict "features must be time-safe" review of the
pipeline is enough. Any per-column time-safety spec
belongs in the feature dictionary from chapter 06.

## Multiple horizons — evaluating a forecast at each step

A forecasting model that predicts the next 7 days has 7
different error distributions — the first-day error is
smaller than the seventh-day error. Report per-horizon
metrics separately, not just aggregated:

```python
import pandas as pd
import numpy as np


def per_horizon_metric(y_true, y_pred, horizons):
    df = pd.DataFrame({"h": horizons, "err": y_true - y_pred})
    return df.groupby("h").agg(
        mae=("err", lambda e: np.abs(e).mean()),
        rmse=("err", lambda e: np.sqrt((e ** 2).mean())),
        n=("err", "size"),
    )
```

A single "MAE across all horizons" number can hide a
model that is fine one day out and useless seven days
out. Report the table; publish the aggregate as a summary
only after the reader has seen the per-horizon breakdown.

## Seasonality and the "does my split cross a season?" test

A one-year training set with a two-month validation set
that spans a seasonal transition (holidays, tax season,
weather change) will underperform models validated on
mid-season windows. Two guards:

- **Cover at least one full seasonal cycle in every
  training fold.** For weekly seasonality that is weeks;
  for annual seasonality that is a year. Anything shorter
  guarantees a seasonally biased fold.
- **Validate on multiple seasonal positions.** With
  monthly data and annual seasonality, five folds ending
  at five different months of the year gives a better
  picture than five folds all ending in December.

`TimeSeriesSplit` alone does not enforce this; check by
inspecting the timestamp ranges of your folds.

## Blocked / group-blocked time splits

Two variants of the basic time split are common enough to
name:

**Blocked time-series CV.** Skip a **buffer** period at
both ends of the test window (not just the beginning).
This is stronger than `TimeSeriesSplit`'s single-sided
`gap` when the label horizon can extend on either side.
`sklearn.model_selection.TimeSeriesSplit(gap=...)` covers
the leading gap; a trailing buffer requires custom
splitting or a helper like `mlxtend.evaluate` or
`sktime.split`.

**Group time-series CV.** The data has a time index *and*
a group structure (multiple entities each with their own
time series — customers, stores, sensors). You want
splits that hold out future observations across *all*
groups, not just some. `sklearn.model_selection.
TimeSeriesSplit` on a global sort works if entities are
independent; otherwise chapter 08's `GroupKFold` combined
with a per-group time cutoff is the pattern.

<!-- needs-research: check if scikit-learn has added a
GroupTimeSeriesSplit primitive. Otherwise cite mlxtend /
sktime implementations. -->

## Recursive vs direct forecasting evaluation

Two ways a multi-step forecaster produces predictions:

- **Direct.** Train one model per horizon (`y[t+1]`,
  `y[t+2]`, ..., `y[t+h]`). Evaluate each independently.
  The metric per horizon is the metric of that horizon's
  model.
- **Recursive.** Train one model for the one-step-ahead
  prediction. To forecast $h$ steps ahead, feed each
  prediction back as input for the next. Errors
  compound.

The two produce different metrics for the same underlying
capability. The evaluation should match the deployment
scheme: if production uses recursive prediction, evaluate
recursively — otherwise the reported multi-step error
does not correspond to reality. Never evaluate a
recursive forecaster by scoring each horizon
independently using ground-truth inputs; that mimics a
direct forecaster and reports a much better number than
you will see live.

## An end-to-end pattern

The template for a supervised time-series problem where a
row's label depends on data up to `label_horizon` days
in the future.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# Assume df has a "date" column, features, and target y.
df = df.sort_values("date").reset_index(drop=True)
X = df.drop(columns=["date", "target"])
y = df["target"]

LABEL_HORIZON_DAYS = 30
# TimeSeriesSplit works on row positions, not calendar days.
# When rows are one-per-day, gap = LABEL_HORIZON_DAYS is right.
# For irregular sampling, translate carefully.

cv = TimeSeriesSplit(
    n_splits=5,
    max_train_size=None,      # expanding window
    gap=LABEL_HORIZON_DAYS,   # avoid target-lag leakage
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=1.0)),
])

fold_metrics = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X)):
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = pipe.predict(X.iloc[val_idx])
    mae = mean_absolute_error(y.iloc[val_idx], pred)
    fold_metrics.append({
        "fold":       fold,
        "train_from": df["date"].iloc[tr_idx[0]],
        "train_to":   df["date"].iloc[tr_idx[-1]],
        "val_from":   df["date"].iloc[val_idx[0]],
        "val_to":     df["date"].iloc[val_idx[-1]],
        "n_train":    len(tr_idx),
        "n_val":      len(val_idx),
        "mae":        mae,
    })

print(pd.DataFrame(fold_metrics))
```

Read the printed table before believing the mean. Check:

- **Train / validation dates are non-overlapping** with
  the correct gap between them.
- **No fold's validation set is in a weird part of the
  year** relative to the others (the seasonality guard).
- **Fold metrics are consistent.** A single fold with
  dramatically worse metric points to a regime change or
  a data-quality issue in that period, not a model issue.

## Common pitfalls

1. **Shuffled k-fold on time-series.** The universal
   mistake. `KFold(shuffle=True)` on temporal data
   inflates every metric.
2. **Forgetting to sort by time before
   `TimeSeriesSplit`.** The splitter uses row order, not
   timestamp; unsorted data produces meaningless folds.
3. **Zero `gap` on labels that depend on the future.**
   The training set uses data from *after* the validation
   set through the label definition; the metric is
   optimistic.
4. **Global preprocessing on the full time-series before
   splitting.** A `StandardScaler` fit before splitting
   uses the future's mean; a rolling-mean feature computed
   on the full series leaks future values.
5. **Not matching the evaluation to the deployment
   horizon.** A model deployed to forecast 7 days ahead
   evaluated one day ahead reports a much better number
   than the deployment will see.
6. **Recursive forecaster evaluated with ground-truth
   inputs.** The compounded error is the deployment
   behaviour; evaluating each step in isolation misses it.
7. **Validation on a single time window.** One `test`
   window at the end of the data is one draw from a
   distribution. Use multi-fold time-series CV so the
   metric mean is stable.
8. **Reporting aggregate metrics without per-horizon
   breakdown.** The user cares whether the model is
   worse further out; a single mean averages that away.

## Summary

- Time-series data breaks the i.i.d. assumption. Random
  splits leak future information; the reported metric is
  optimistic.
- The core rule: training data may never come from after
  the validation / test window. Use walk-forward,
  expanding-window, or rolling-origin CV; scikit-learn's
  `TimeSeriesSplit` implements these.
- Insert a `gap` equal to the maximum label horizon to
  avoid target-lag leakage. Compute features using only
  data available at each row's observation time.
- Cover full seasonal cycles in each training fold;
  validate across multiple seasonal positions; match the
  evaluation horizon and recursion scheme to the
  deployment.
- Report per-horizon metrics and per-fold results before
  aggregating — the aggregate hides the important
  variation.

## Where this goes next

Chapter 08 covers the two other cases where standard CV
misleads: **group-based** splits (multiple correlated rows
per entity) and **nested** CV (honest hyperparameter
selection without inflating the outer-loop score).
