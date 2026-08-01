# 02 — Metrics for regression

> **Reading time:** ~25 minutes.
> **Prerequisite chapter:** 01.

## Motivation

Regression looks metric-simple compared to classification.
There is no confusion matrix, no threshold, no imbalance
story. The metric is a scalar summary of "how far off were
we?"

That simplicity hides the important choice. **How far off**
can be measured with the mean absolute error, the mean
squared error, the root mean squared error, R², MAPE,
sMAPE, quantile loss, Huber loss, and half a dozen others.
They rank models differently. They react differently to
outliers, to skewed targets, to zeros in the ground truth,
and to the units the target is measured in. Pick the wrong
one and the model you promote is the model that happened to
be optimised for a number that does not match the decision.

This chapter grounds the four regression metrics you will
reach for in almost every real project — MAE, MSE, RMSE, R²
— and explains when to reach past them.

## The four you must know cold

Let $y$ be the ground truth, $\hat{y}$ the prediction, $n$
the number of samples, and $\bar{y}$ the mean of $y$.

| Metric | Formula | Units | Robust to outliers? |
|---|---|---|---|
| **MAE** — Mean Absolute Error | $\frac{1}{n} \sum |y_i - \hat{y}_i|$ | same as $y$ | yes |
| **MSE** — Mean Squared Error | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | $y^2$ | no |
| **RMSE** — Root Mean Squared Error | $\sqrt{\text{MSE}}$ | same as $y$ | no |
| **R²** — coefficient of determination | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | unitless | no |

Scikit-learn implements all four in
[`sklearn.metrics`](https://scikit-learn.org/stable/api/sklearn.metrics.html):
`mean_absolute_error`, `mean_squared_error` (with
`squared=False` for RMSE in older versions, or the dedicated
`root_mean_squared_error` in scikit-learn ≥ 1.4), and
`r2_score`. Every serious modelling framework provides them
under the same names.

The rest of the chapter is what each one means in practice —
and when it is the wrong one to report.

## MAE — the honest default

$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

MAE is the average magnitude of the error, in the units of
the target. If the target is "days until churn", MAE is a
number of days. If it is "USD revenue", MAE is dollars.

**When to reach for it first:**

- You want a stakeholder-readable number ("we are off by
  about £340 per property, on average").
- Your target has occasional huge outliers you do *not*
  want dominating the metric (single fraudulent transactions
  in a spend-forecasting dataset).
- The cost of an error is roughly linear in its magnitude.

**When to be suspicious of it:**

- Large errors matter *disproportionately* — the cost of a
  10× error is more than 10× the cost of a 1× error. In
  that world MAE under-weights the tail; use MSE / RMSE.
- The target is heavy-tailed and you want the model to work
  hard on the tail. MAE is happy with a model that
  predicts the median.

**Optimal predictor.** The constant that minimises MAE on
a dataset is the **median** of $y$. That is worth
remembering — it explains why an MAE-optimised model often
produces conservative predictions on skewed targets. It is
also what your dumbest baseline (chapter 09) should be:
"predict the median" is the MAE-optimal constant model.

## MSE and RMSE — the outlier-sensitive pair

$\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$
$\text{RMSE} = \sqrt{\text{MSE}}$

The square in MSE has two consequences:

1. **The units are squared.** "Off by 42 dollars squared"
   is not a sentence a human uses. This is why RMSE — the
   square root of MSE — is what you report to humans. RMSE
   is in the same units as the target.
2. **Large errors dominate.** A single error of 10 units
   contributes $100$ to the sum; a hundred errors of 1
   contribute $100$ total. MSE and RMSE trade the entire
   metric on the tail.

**When to reach for RMSE:**

- Large errors are much worse than small ones — cost is
  roughly quadratic in error (energy, physical safety
  margins, financial exposure with convex loss).
- The training loss is already MSE (default for most linear
  regressors, `HistGradientBoostingRegressor` with
  `loss="squared_error"`). Then RMSE is the metric that
  aligns with the objective.
- You want a metric that is more sensitive to bad edge
  cases than MAE.

**Optimal predictor.** The constant that minimises MSE is
the **mean** of $y$. Again, this is your MSE baseline.

**Gotcha — RMSE is not a "better MAE".** RMSE ≥ MAE
always, and the gap between them widens as the error
distribution gets more skewed. The two metrics measure
different things; you can build models that improve MAE
while worsening RMSE (a model that trades a few very large
misses for many small improvements). Report both if the
choice matters.

## R² — the unitless "variance explained"

$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y}_i)^2}$

R² compares the model's squared error against a baseline
that always predicts the mean. Two useful reference points:

- **R² = 1** — the model predicts every point exactly.
- **R² = 0** — the model is no better than the mean-
  predictor baseline.
- **R² < 0** — the model is *worse* than always predicting
  the mean. This happens in practice and is a red flag.

**When it helps:**

- Comparing models across datasets with different scales.
  RMSE = 5 is impressive on a $y$ that ranges 0–10 and
  irrelevant on a $y$ that ranges 0–10,000. R² normalises.
- Communicating "how much of the variance did we capture?"
  to statistically literate stakeholders.

**When it misleads:**

- On a **held-out test set**, R² is computed against the
  test-set variance, not the training-set variance. A test
  set with a smaller natural spread than the training set
  will produce a lower R² even if RMSE is fine.
- **R² does not always increase with more features** on
  test data (it does on training data). If you see a
  polished-looking positive R² on training and a negative
  R² on test, you have overfit — chapter 06's job to
  detect.
- Time series R² is often misleadingly high when a naive
  "predict yesterday" model would score well. Compare
  against the persistence baseline (chapter 09) before
  celebrating.

## Percentage errors — MAPE, sMAPE, and the zero problem

Percentage errors sound like the right choice when the
target's magnitude varies wildly ("we forecast demand from
zero to millions"). They rarely are:

$\text{MAPE} = \frac{100}{n} \sum \left|\frac{y_i - \hat{y}_i}{y_i}\right|$

The zero-division problem is not academic: any $y_i = 0$
makes MAPE undefined; any $y_i$ near zero blows the metric
up. That is why the M4 forecasting competition and most of
the modern forecasting literature use **sMAPE** (symmetric
MAPE) or scale-free alternatives like **MASE** (mean
absolute scaled error), which normalises against a naive
seasonal forecast rather than against $y$ directly. If your
target has zeros or near-zeros, MAPE is the wrong metric —
report MAE alongside it or switch to MASE.

<!-- needs-research: cite Hyndman & Koehler (2006) "Another
look at measures of forecast accuracy" for the MASE
definition and the critique of MAPE. -->

## When the target is skewed — log-space metrics and quantile loss

Two situations force you off the vanilla four:

**Heavy-tailed positive targets** (revenue, house prices,
click-through counts). Fitting on the raw target lets the
tail dominate the loss. Two standard moves:

- **Log-transform the target.** Fit $\log(1 + y)$; report
  RMSE in log-space (**RMSLE** — Root Mean Squared
  Logarithmic Error). RMSLE penalises under-prediction and
  over-prediction asymmetrically and is scale-free (a 10%
  error is the same whether $y$ is 10 or 10 million).
- **Fit a quantile.** Instead of predicting the mean or
  median, predict the 90th percentile — useful when
  under-prediction is far worse than over-prediction
  (staffing, capacity planning). Scikit-learn's
  `GradientBoostingRegressor(loss="quantile", alpha=0.9)`
  and LightGBM / XGBoost's quantile objectives train
  directly against **pinball loss**, the appropriate metric
  for the same target.

**Asymmetric costs** (energy over-supply vs blackouts,
booking cancellations vs no-shows). Report a quantile
metric that matches the cost structure. A model tuned on
symmetric RMSE will happily over- and under-predict in
equal measure, which is often the wrong trade.

## Huber loss and the "robust regression" spectrum

Between MAE (fully outlier-robust) and MSE (fully outlier-
sensitive) sits **Huber loss**: quadratic near zero,
linear once the error exceeds a threshold $\delta$. It is
the "give me most of MSE's smoothness but stop the tail
from dominating" compromise, and it is what
`HuberRegressor` in scikit-learn implements.

Huber is worth reaching for when:

- You know the training data has some fraction of
  contaminated / outlier rows you cannot easily filter.
- You want a differentiable loss (gradient-boosted trees'
  `loss="huber"`) that still cares about the tail.

## A worked example

The pattern to internalise: pick a task, pick the metric,
justify the choice, and report a companion.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Two very different models
lr = LinearRegression().fit(X_tr, y_tr)
gb = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)

for name, model in [("linear", lr), ("gbdt", gb)]:
    pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2   = r2_score(y_te, pred)
    print(f"{name:>6}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

# Baselines to compare against — these come from chapter 09
median_pred = np.full_like(y_te, np.median(y_tr))
mean_pred   = np.full_like(y_te, np.mean(y_tr))
print(f"median  MAE={mean_absolute_error(y_te, median_pred):.3f}")
print(f"  mean  RMSE={np.sqrt(mean_squared_error(y_te, mean_pred)):.3f}")
```

Two lessons in ten lines: the median-predictor is the
MAE-optimal constant baseline; the mean-predictor is the
RMSE-optimal constant baseline. Any model you report should
comfortably beat both.

## The "which one do I report?" decision tree

Ask, in order:

1. **What decision does the model support?** If the answer
   is "we act differently when the number is off by a large
   amount", RMSE. If it is "we act linearly in the error",
   MAE. If it is "we care about the top / bottom quantile
   only", quantile loss.
2. **Is the target skewed / heavy-tailed?** If yes, log-
   transform the target and report RMSLE, or switch to
   MASE if there is a natural seasonal baseline.
3. **Do you need to compare across scales?** R² normalises
   across datasets with very different $y$ ranges — but
   never report R² *alone*; report RMSE or MAE with it.
4. **Report two.** Always. The primary metric anchors the
   decision; a companion of a different family catches the
   case where the primary is being gamed.

## Common pitfalls

1. **Reporting only R² on the training set.** Training R²
   is a monotone function of the number of parameters; it
   always goes up. Test R² is the number that matters.
2. **Reporting MAPE when the target has zeros.** The metric
   is undefined at zero and unstable near it. Use MAE or
   MASE.
3. **Comparing RMSE across datasets with different scales
   without normalising.** RMSE = 500 on `y ∈ [0, 1000]` is
   terrible; on `y ∈ [0, 10^6]` it is negligible.
4. **Optimising the wrong loss for the reported metric.** A
   model trained with squared-error loss and reported on
   MAE is under-optimised for the metric it is judged on.
   If MAE is the metric, train with an MAE-consistent loss
   (`loss="absolute_error"` in gradient boosters,
   `HuberRegressor`, quantile models with $\alpha = 0.5$).
5. **Aggregating the same metric two ways.** "Mean per-user
   RMSE" and "RMSE across all users" are different numbers;
   the first weights all users equally, the second weights
   by row count. State which one you are computing.
6. **Reporting one number without a baseline.** RMSE = 3.2
   means nothing without "the mean-predictor baseline was
   4.1". Chapter 09 is the discipline; do it now.

## Summary

- MAE (median-optimal, outlier-robust, same units), RMSE
  (mean-optimal, outlier-sensitive, same units), R²
  (unitless, comparable across scales) are the four you
  must know without looking up.
- Percentage errors (MAPE) fail near zero; switch to MASE
  or log-space RMSLE for heavy-tailed / scale-varying
  targets.
- The metric you report should reflect the cost structure
  of the decision. Report a companion metric from a
  different family so the primary can be sanity-checked.
- Optimise the loss that matches the metric you will
  report; otherwise the model is under-tuned for its own
  benchmark.

## Where this goes next

Chapter 03 crosses over to classification, where the metric
choice interacts with the decision **threshold** and with
**class imbalance** — and where accuracy alone is almost
always the wrong number to publish.
