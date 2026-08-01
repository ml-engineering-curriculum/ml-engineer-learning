# 04 — Calibration

> **Reading time:** ~25 minutes.
> **Prerequisite chapter:** 03.

## Motivation

Consider two classifiers that both hit ROC-AUC 0.90 on the
same test set. Model A outputs scores in the range
[0.2, 0.6]; Model B outputs scores that, when you bin
predictions at 0.7 and look at what actually happened,
seventy percent of them are positive. Both rank equally
well. Only one produces numbers you can call *probabilities*
without lying.

Any use of the output that treats it as a probability — a
threshold set by a cost calculation ("act above the
threshold where expected value is positive"), a downstream
aggregation ("sum the predicted probability across a
million users to get expected fraud losses"), a UI ("we
are 82% sure this claim is legitimate") — assumes
calibration. Yet most training regimes optimise ranking
and not calibration, so the assumption is unwarranted
without a check.

This chapter covers the vocabulary (reliability, ECE,
Brier score), the tools (Platt scaling, isotonic
regression, temperature scaling), and the point in the
pipeline where you insert them.

## What "calibrated" means

A classifier is **calibrated** if, among the predictions it
assigns probability $p$, the true positive rate is also $p$.

Formally, for a score $\hat{p}(x)$:

$$P(y = 1 \mid \hat{p}(x) = p) = p \quad \text{for all } p \in [0, 1]$$

In practice you cannot check the equality pointwise — there
are usually not enough samples at any exact score value.
You bin the predictions and check bin by bin.

**Two dimensions worth separating in your head:**

- **Ranking / discrimination.** Does a higher score
  correspond to a more-likely-positive example? Measured
  by ROC-AUC, PR-AUC.
- **Calibration.** Is the score on the *right scale*? Do
  the numbers behave like probabilities?

A well-calibrated model must also rank well (badly ranked
predictions cannot be well calibrated). But a well-ranked
model can be miscalibrated in either direction — over-
confident (scores are too extreme) or under-confident
(scores clump around 0.5).

## The reliability diagram

The primary diagnostic tool. Sort predictions into $B$
bins by predicted probability, and for each bin plot:

- **x-axis:** mean predicted probability in the bin.
- **y-axis:** empirical fraction of positives in the bin.

A perfectly calibrated model lies on the diagonal $y = x$.
Bins above the diagonal mean the model is under-confident
(predicts 0.6, is right 0.8 of the time). Bins below mean
over-confident.

Scikit-learn provides `sklearn.calibration.calibration_curve`
to compute the two arrays, and `CalibrationDisplay` to plot
them:

```python
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

fig, ax = plt.subplots()
CalibrationDisplay.from_predictions(
    y_true, y_proba, n_bins=15, strategy="quantile", ax=ax
)
ax.plot([0, 1], [0, 1], "k--", label="perfect")
ax.legend()
plt.show()
```

Two knobs:

- **`n_bins`** — more bins gives finer resolution but
  emptier bins (higher variance in the empirical rate).
  10–20 is usually right; drop when you have small
  samples.
- **`strategy`** — `"uniform"` uses equal-width bins;
  `"quantile"` uses equal-population bins. Quantile is the
  safer default because it prevents empty bins from
  dominating the plot.

## Expected Calibration Error (ECE)

A single-number summary of the reliability diagram: the
weighted average, across bins, of the gap between
predicted probability and empirical positive rate.

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

where $B_b$ is bin $b$, $|B_b|$ is its population,
$\text{acc}(B_b)$ is the empirical fraction of positives,
and $\text{conf}(B_b)$ is the mean predicted probability.

Reference points:

- **ECE = 0** — perfect calibration.
- **ECE small (< 0.02)** — the model can be treated as a
  probability for most downstream uses.
- **ECE large (> 0.1)** — the model is a good ranker but a
  poor probability; either recalibrate or use it only for
  ranking-based decisions.

Two caveats worth carrying:

- **ECE depends on the binning.** Different `n_bins` and
  binning strategies give different values; report the
  binning you used.
- **ECE hides direction.** An ECE of 0.05 could be over-
  confidence in one region and under-confidence in
  another; the reliability plot tells you which. Report
  both.

Scikit-learn does not currently ship an `expected_calibration_error`
function; compute it from `calibration_curve` outputs, or
use `torchmetrics.classification.CalibrationError`.

<!-- needs-research: verify whether sklearn ≥ 1.6 has added
a first-class ECE implementation. -->

## Brier score

The Brier score is the mean squared error between the
predicted probability and the binary outcome:

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2$$

Range: $[0, 1]$. Zero is perfect; higher is worse. The
"predict the base rate for everything" baseline scores
$p(1 - p)$ where $p$ is the class prevalence.

**Two properties that make it worth reporting:**

1. **Proper scoring rule.** Like log loss, Brier is
   minimised by predicting the true conditional
   probability. Reporting Brier alongside ROC-AUC gives
   you one ranking metric and one calibration-sensitive
   metric.
2. **Decomposition.** Brier = reliability + resolution +
   uncertainty. The reliability component isolates the
   calibration gap; the resolution component measures how
   much the predictions vary from the mean; uncertainty is
   the base rate. The full decomposition (Murphy, 1973)
   is beyond level 20; the takeaway is that Brier
   integrates calibration and sharpness into one number.

**Brier vs log loss.** Log loss is more sensitive to
extreme mistakes (predicting 0.99 for the wrong class);
Brier is bounded and more forgiving. In practice both are
worth reporting for a probabilistic classifier — log loss
if the downstream use is sensitive to blowups, Brier if
you want a bounded, decomposable summary.

`sklearn.metrics.brier_score_loss` is the reference
implementation.

## Why models are miscalibrated by default

Different training regimes fail calibration in different
directions. Knowing which failure mode your model is prone
to helps pick the recalibrator:

- **Logistic regression** trained on log-loss is roughly
  calibrated out of the box. When it fails to be, the
  cause is usually class weighting, resampling, or
  regularisation strong enough to shrink the estimated
  probabilities toward the base rate.
- **Naive Bayes** is systematically overconfident because
  its feature-independence assumption double-counts
  correlated features. It is a classical example of a
  strong ranker with catastrophic calibration.
- **SVM** (hinge loss) does not produce probabilities at
  all. `predict_proba` uses Platt scaling under the hood,
  which is a calibrator, not a native probability.
- **Random forests** are typically well-calibrated near
  the extremes and under-confident in the middle — an
  averaging effect from voting across trees.
- **Gradient-boosted decision trees** (XGBoost, LightGBM)
  trained on `binary:logistic` / `binary_logloss` are
  usually reasonably calibrated, but heavy regularisation
  or early stopping can leave residual miscalibration.
- **Deep neural networks trained with cross-entropy** are
  systematically **overconfident** on modern architectures.
  Guo et al. (2017) documented this for ResNet-style
  models and popularised **temperature scaling** as the
  fix.
- **Resampling / class-weighting** — SMOTE, downsampling,
  and `class_weight="balanced"` all break calibration.
  The model no longer sees the true prior; its scores are
  on the resampled prior. Recalibrate on the original-
  distribution validation set, always, after resampling.

## Recalibration methods

Post-hoc recalibration takes a trained model's scores and
maps them to calibrated probabilities using a small,
held-out calibration set (that the base model did *not*
see during training).

### Platt scaling (sigmoid)

Fit a one-parameter logistic regression to map raw scores
to probabilities:

$$\hat{p} = \sigma(A \cdot s + B)$$

where $s$ is the raw score. Two parameters; requires very
little data; monotone; parametric. Works well when the
miscalibration is a smooth sigmoid warp — the classic case
being SVM scores.

### Isotonic regression

Fit a non-parametric, monotone step function that maps raw
scores to probabilities. More flexible than Platt;
requires more data (~ 1000+ samples on the calibration
set); can capture arbitrary monotone miscalibration
shapes.

### Temperature scaling (deep learning)

For a neural network with logits $z$, divide by a scalar
$T$ before softmax:

$$\hat{p} = \text{softmax}(z / T)$$

One parameter, fit by minimising log loss on a held-out
set. Preserves the argmax (so accuracy is unchanged) and
addresses the "over-confident logits" failure mode of
modern deep classifiers. This is the recommended
recalibrator for deep classifiers per Guo et al. (2017);
it is simple, cheap, and effective.

### Which to use

| Situation | Recalibrator |
|---|---|
| SVM, small calibration set | Platt |
| Random forest, moderate calibration set | Isotonic |
| GBDT, moderate calibration set | Isotonic (or leave alone) |
| Deep classifier | Temperature scaling |
| Multi-class, needs full recalibration | Vector / matrix scaling |

Scikit-learn wraps Platt and Isotonic behind
[`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html):

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

base = HistGradientBoostingClassifier(random_state=0)
cal = CalibratedClassifierCV(
    base, method="isotonic", cv=5
).fit(X_tr, y_tr)

proba = cal.predict_proba(X_te)[:, 1]
```

The `cv=5` means "fit five base models on 4/5 of the
training data, calibrate each on the held-out fold,
average the resulting predictions". This avoids the
"who calibrates the calibrator?" problem — do not use the
training set for calibration; the base model has already
seen it.

Temperature scaling is not in scikit-learn; a common
implementation is ~30 lines with `scipy.optimize`, or use
`torchmetrics` / `netcal`.

<!-- needs-research: `netcal` calibration library — check
current status and recommended methods list. -->

## When calibration is (and isn't) worth doing

**Do calibrate when:**

- The score is used in a downstream expected-value
  calculation (fraud loss, revenue lift, expected
  utility).
- The score is displayed to a user or auditor as a
  probability.
- Predictions are aggregated (sum, mean, threshold set by
  base rate) across large populations.
- You resampled or reweighted classes during training.
- The classifier is a deep network trained with cross-
  entropy — assume overconfidence until proven otherwise.

**Don't bother when:**

- The score is used only for ranking (recommendation
  ordering, top-k retrieval). Calibration does not change
  the ranking.
- The threshold is picked on a held-out validation set for
  a target precision / recall — the threshold absorbs
  any smooth monotone miscalibration.
- You have very little data ( < 1000 for isotonic, < 100
  for Platt); recalibration will overfit and make things
  worse.

## The evaluation loop, with calibration folded in

The end-to-end pattern for a classifier where calibration
matters:

```
     ┌───────────────┐
     │   train set   │──┐
     └───────────────┘  │  fit base model
                        ▼
                  ┌───────────┐
                  │   base    │
                  │  model    │
                  └─────┬─────┘
                        │
    ┌───────────────┐   │  score on
    │ calib set     │──▶┴─▶  scores
    └───────────────┘        │
                             ▼
                        ┌────────────┐
                        │ recalibr.  │  fit sigmoid /
                        │  fit       │  isotonic / temp.
                        └─────┬──────┘
                              │
    ┌───────────────┐         │
    │   test set    │─────────┴─▶  reliability plot
    └───────────────┘             ECE, Brier, log loss
                                  on final calibrated scores
```

Three sets: **train** (fits the base model), **calib**
(fits the recalibrator; base model does not see it),
**test** (untouched until the end). This is a three-way
split, distinct from the two-way split of chapter 03.
`CalibratedClassifierCV` handles the train / calib split
internally by cross-validation; the test split is still
your responsibility.

## Common pitfalls

1. **Calibrating on the training set.** The base model
   already fits the training set closely; calibrating on
   the same data produces near-perfect apparent
   calibration and nothing generalises. Always use a
   held-out split.
2. **Reporting calibration only in aggregate.** ECE
   averages away signed error. A reliability plot with an
   over-confident right tail and an under-confident middle
   averages to near-zero ECE. Report both.
3. **Recalibrating after class-weighting without
   correcting the prior.** If you trained with balanced
   class weights, the model's scores are on the balanced
   prior; recalibrate on original-distribution data and
   the miscalibration goes away.
4. **Using isotonic with tiny calibration sets.** Isotonic
   is non-parametric; with < ~1000 calibration samples it
   overfits, and the recalibrated scores can be *worse*.
   Use Platt or temperature scaling instead.
5. **Comparing Brier scores across datasets with different
   base rates.** Brier is scale-dependent; use it within
   a dataset, not across.
6. **Assuming a well-ranking model is well-calibrated.**
   ROC-AUC = 0.95 says nothing about calibration.
7. **Silently miscalibrating by threshold-tuning on a
   different set than you'll deploy on.** If you pick the
   0.5 → 0.3 threshold on the calibration set to hit a
   precision target, the deployment threshold is 0.3 on
   the deployed base rate, which may differ.

## Summary

- Calibration = "when the model says 0.7, it is right 70%
  of the time". Distinct from ranking / discrimination.
- Diagnose with a reliability diagram; summarise with ECE
  and Brier. Report both — ECE hides direction, Brier is
  scale-dependent.
- Deep classifiers trained on cross-entropy are
  systematically overconfident; temperature scaling is
  the standard fix. GBDT is usually close; RF and SVM
  usually need Platt / isotonic; naive Bayes is broken
  and needs recalibration always.
- Recalibrate on a held-out calibration split — not the
  training set. `CalibratedClassifierCV` handles this via
  internal CV.
- Skip calibration when the score is only used for
  ranking; do it always when the score is treated as a
  probability, aggregated across a population, or shown
  to a user.

## Where this goes next

Chapter 05 covers ranking metrics — nDCG, MAP, MRR,
hit@k — for the tasks where the *order* of the predictions
matters more than any single classification or probability.
