# 03 — Metrics for classification

> **Reading time:** ~35 minutes.
> **Prerequisite chapter:** 02.

## Motivation

Classification looks metric-friendly: predict a class,
compare against the label, count how often you were right.
Accuracy. Done.

Two facts make it much richer than that:

1. **Most classifiers output a score, not a class.** The
   class comes from a threshold — 0.5 by default, but
   often wrong. Changing the threshold changes every
   count-based metric.
2. **Class balance is rarely 50/50.** On a 99/1 dataset a
   model that always predicts the majority class hits 99%
   accuracy. Accuracy is a poor summary when the classes
   are imbalanced or the costs are asymmetric.

This chapter covers the confusion-matrix vocabulary, the
threshold-based metrics (accuracy, precision, recall, F1,
specificity), the threshold-independent metrics (ROC-AUC,
PR-AUC), and the small set of decisions that pin down
which one you should report.

Calibration — "does the score behave like a probability?"
— is its own topic and lives in chapter 04. Ranking metrics
(nDCG, MAP, MRR) are chapter 05.

## The confusion matrix

For binary classification with positive class `1` and
negative class `0`:

| | Predicted 1 | Predicted 0 |
|---|---|---|
| **Actual 1** | **TP** (true positive) | **FN** (false negative) |
| **Actual 0** | **FP** (false positive) | **TN** (true negative) |

Everything in this chapter is a ratio of these four cells.
Memorise the layout; the metrics fall out of it.

```python
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
```

Scikit-learn's `confusion_matrix` returns rows-actual /
columns-predicted, and `.ravel()` flattens in the order
above. That is the convention used through the rest of this
module.

## Accuracy — and why it fails

$\text{accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$

Accuracy answers "what fraction of predictions were
correct?". Two failure modes make it unfit for a headline
metric on most real problems:

**Class imbalance.** On a fraud dataset where 0.1% of
transactions are fraud, a model that predicts "not fraud"
for everything hits 99.9% accuracy and catches zero
fraud. Any metric that reports "99.9%" as good is
uninformative here.

**Asymmetric costs.** In cancer screening a false negative
(missed cancer) is not the same as a false positive
(unnecessary follow-up). Accuracy weights them equally.

The rule of thumb: **report accuracy only when the classes
are roughly balanced and the costs of each error type are
roughly equal.** Otherwise accuracy hides the number that
matters.

Multi-class accuracy has the same failure modes at
each-class scale, which is why per-class recall / precision
and confusion-matrix inspection are the multi-class
equivalents.

## Precision, recall, and their trade-off

$\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$  — of what we flagged as positive, how much really was?

$\text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$  — of what really was positive, how much did we catch?

Reading them into everyday sentences:

- **High precision, low recall.** "When the alarm goes
  off, it's usually right — but it misses a lot."
- **Low precision, high recall.** "It catches almost
  everything — but the alerts are mostly noise."

Almost every operational trade-off in a classifier is one
of these two dials against the other. Moving the threshold
trades one for the other; changing the model can shift the
whole frontier.

**When to prioritise precision.** The cost of a false
positive is high compared to the cost of a false negative.
Spam filters (flagging a legit email hurts more than
missing spam), content-moderation auto-actions, medical
tests that trigger invasive follow-up.

**When to prioritise recall.** The cost of a false
negative is high. Cancer screening, industrial defect
detection, security intrusion detection. In these cases
"we can afford to look at more suspects" is true; "we
cannot afford to miss one" is the harder constraint.

**Specificity** — $\frac{\text{TN}}{\text{TN} + \text{FP}}$
— is the negative-class analogue of recall. It appears
prominently in medical statistics; recall of the positive
class is called **sensitivity** in the same field.

## F1 and its cousins

$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$

F1 is the harmonic mean of precision and recall. Two
useful properties:

- **The harmonic mean punishes small values.** If either
  precision or recall is near zero, F1 is near zero. This
  is why F1 is preferred to the arithmetic mean on
  imbalanced problems — arithmetic mean lets a strong
  precision paper over a terrible recall.
- **F1 assumes precision and recall are equally
  important.** They rarely are. When the trade-off is
  asymmetric, use $F_\beta$:

$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}$

$\beta > 1$ weights recall more (use $\beta = 2$ when
false negatives are twice as bad as false positives).
$\beta < 1$ weights precision more.

**Multi-class F1**. `sklearn.metrics.f1_score(...,
average=...)` accepts:

- `"binary"` — the default; the F1 of the positive class.
- `"macro"` — arithmetic mean of the per-class F1s. Each
  class weighted equally. Rewards uniform performance.
- `"micro"` — F1 computed on total TP / FP / FN across all
  classes. On multi-class classification (no multi-label)
  this equals accuracy — beware.
- `"weighted"` — per-class F1 weighted by support (number
  of true instances). Behaves like accuracy on imbalanced
  problems.

Pick the average based on whether you want per-class parity
(**macro**), aggregate throughput (**micro** / weighted),
or a specific class (**binary**). Reporting only the default
average without stating which one is a common documentation
failure — always name the averaging strategy.

## The threshold is a hyperparameter

Almost every scikit-learn classifier exposes both
`predict` (returns the class) and `predict_proba` (returns
the score). `predict` picks the class with the highest
score; for binary problems that is the 0.5 threshold on the
positive-class probability. That threshold is a modelling
choice, not a physical constant:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]

for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    y_hat = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, y_hat).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    print(f"t={threshold:.1f}  P={precision:.3f}  R={recall:.3f}")
```

Every count-based metric — accuracy, precision, recall, F1,
specificity — depends on the threshold. Reporting them
without stating the threshold is meaningless. Two useful
conventions:

- **Pick the threshold on the validation set** for a
  specified operational criterion ("maximise F1", "hit
  recall ≥ 0.9 at maximum precision", "hold precision ≥
  0.8"). Report the *test-set* metrics at *that* threshold.
- **Do not pick the threshold on the test set.** That is
  a form of leakage — the number no longer reflects
  unseen behaviour.

## ROC-AUC — threshold-independent, imbalance-sensitive

The Receiver Operating Characteristic curve plots the
**true positive rate** (recall) against the **false
positive rate** ($\frac{\text{FP}}{\text{FP} + \text{TN}}$)
as the threshold sweeps from 0 to 1. ROC-AUC — the area
under that curve — summarises "how well does the score
separate the two classes, without picking a threshold?"

Two ways to read the ROC-AUC value:

- **Range and reference points.** 1.0 is perfect
  separation; 0.5 is random; below 0.5 is a model with
  inverted predictions.
- **Interpretable probability.** ROC-AUC equals the
  probability that a random positive example scores higher
  than a random negative example. That is the definition
  worth remembering — it is what the number means, not
  what it is.

**When ROC-AUC is a good fit:**

- You want a threshold-independent comparison across
  models.
- The classes are roughly balanced, or you genuinely care
  about performance across the whole score range.
- You want to communicate model quality without picking a
  business threshold yet.

**When ROC-AUC misleads:** on heavily imbalanced data.
The FPR denominator is the entire negative class, which is
huge, so a large increase in false positives barely moves
the curve. A ROC-AUC of 0.99 can hide a precision of 0.05
if the positive class is rare. On imbalanced problems,
reach for PR-AUC.

## PR-AUC — the imbalanced-classes companion

The precision-recall curve plots precision against recall
as the threshold varies. PR-AUC is its area. Unlike ROC-
AUC, both axes are ratios *within the positive class* —
precision has "predicted positives" in the denominator,
recall has "actual positives". The negative class does not
appear.

Consequences:

- **PR-AUC is sensitive to positive-class prevalence.** A
  perfect classifier has PR-AUC = 1; the "predict
  positive-class prevalence" baseline has PR-AUC ≈ base
  rate. On a 1% positive class, the baseline PR-AUC is
  about 0.01 — a big number to beat, and a meaningful
  reference.
- **PR-AUC is the right summary metric for rare-event
  detection** (fraud, disease, defect, intrusion). Every
  benchmark on class-imbalanced data should report PR-AUC;
  reporting only ROC-AUC on 0.1% positive data is a
  well-known mistake.

Two implementations to know:

- `sklearn.metrics.roc_auc_score(y_true, y_score)` —
  ROC-AUC.
- `sklearn.metrics.average_precision_score(y_true,
  y_score)` — a stepwise integration of the PR curve
  that is the accepted "PR-AUC" in scikit-learn.
  `sklearn.metrics.auc(precision, recall)` uses the
  trapezoidal rule and is not recommended for PR curves
  (see the scikit-learn documentation for the reason).

<!-- needs-research: cite Davis & Goadrich (2006) "The
Relationship Between Precision-Recall and ROC Curves" for
the canonical discussion of when PR is more informative
than ROC. -->

## Multi-class metrics — a compressed tour

Every metric above extends to multi-class, with the same
"how do you average across classes?" wrinkle F1 already
exposed.

- **Multi-class accuracy** — fraction of correct
  predictions across all classes. Same failure modes as
  binary accuracy under imbalance.
- **Per-class precision / recall / F1** — the confusion
  matrix generalises to a $K \times K$ table; each class
  has its own P / R / F1 by treating it as the positive
  class in a one-vs-rest split.
- **Macro-averaged / weighted-averaged / micro-averaged**
  — as discussed under F1. Macro treats all classes as
  equally important; weighted weights by support; micro
  aggregates counts before averaging.
- **Multi-class ROC-AUC.** `roc_auc_score(y_true, y_proba,
  multi_class="ovr", average="macro")` computes a one-vs-
  rest AUC per class and averages. `multi_class="ovo"`
  uses one-vs-one pairwise AUCs (Hand & Till formulation).
- **Multi-class log loss.** $-\frac{1}{n} \sum_i \sum_k
  y_{ik} \log \hat{p}_{ik}$, where $y_{ik}$ is one-hot.
  This is the training loss of most neural networks and a
  serviceable summary metric — but it is *scale-
  sensitive to calibration*, which chapter 04 makes
  explicit.

## Log loss — the probabilistic scoring rule

Log loss (a.k.a. cross-entropy) is the metric that punishes
confidently-wrong predictions the most:

$\text{log loss} = -\frac{1}{n} \sum_i \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$

Reading it:

- **Range.** $[0, \infty)$. Zero is perfect; higher is
  worse.
- **What it rewards.** Predicting a probability close to
  the truth. Predicting 0.51 for a positive is fine;
  predicting 0.99 confidently for a wrong class is
  catastrophic.
- **Why it matters here.** Log loss is a **proper scoring
  rule**: it is minimised by predicting the true
  conditional probability. Models trained on log loss
  produce well-calibrated scores *in expectation*. That is
  the segue into chapter 04.

The scikit-learn implementation is `log_loss`. Two gotchas:

- **Undefined at 0 or 1.** The metric involves $\log
  \hat{p}$ and $\log(1 - \hat{p})$; a probability of
  exactly zero on a positive example is $-\infty$
  contribution. `log_loss` clips to `[eps, 1 - eps]` for
  numerical stability; you should be aware the clip is
  hiding what would otherwise be an infinite loss.
- **Reported units.** Log loss is unitless but scale-
  dependent — you cannot compare log loss across datasets
  with different class balances without normalising.

## A worked comparison

The point of comparing all these on the same task is to
watch the metrics disagree on which model is "best".

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# 5% positive class — mildly imbalanced
X, y = make_classification(
    n_samples=20_000, n_features=20, n_informative=5,
    weights=[0.95, 0.05], random_state=0,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y,
)

models = {
    "logreg": LogisticRegression(max_iter=2000).fit(X_tr, y_tr),
    "rf":     RandomForestClassifier(random_state=0).fit(X_tr, y_tr),
    "dumb":   None,   # always-predict-majority baseline
}

for name, model in models.items():
    if name == "dumb":
        proba = np.full(len(y_te), y_tr.mean())
        y_hat = np.zeros_like(y_te)
    else:
        proba = model.predict_proba(X_te)[:, 1]
        y_hat = (proba >= 0.5).astype(int)

    print(f"\n=== {name} ===")
    print(f"accuracy  = {accuracy_score(y_te, y_hat):.3f}")
    print(f"precision = {precision_score(y_te, y_hat, zero_division=0):.3f}")
    print(f"recall    = {recall_score(y_te, y_hat, zero_division=0):.3f}")
    print(f"f1        = {f1_score(y_te, y_hat, zero_division=0):.3f}")
    print(f"roc_auc   = {roc_auc_score(y_te, proba):.3f}")
    print(f"pr_auc    = {average_precision_score(y_te, proba):.3f}")
    print(f"log_loss  = {log_loss(y_te, np.clip(proba, 1e-6, 1 - 1e-6)):.3f}")
```

Two things to observe when you run this:

1. **The dumb baseline hits ~95% accuracy.** Everything
   else about it is uninformative. Accuracy as a headline
   would say all three models are within 5% of each other.
2. **PR-AUC separates the models much more clearly than
   ROC-AUC does.** That is the imbalance story from above,
   made concrete.

## The "which metric do I report?" decision tree

Ask, in order:

1. **Is the class balance roughly even and are the costs
   roughly symmetric?** If yes, accuracy is defensible.
   If no, keep going.
2. **Do I need a single number that does not depend on the
   threshold?** For balanced data, ROC-AUC. For imbalanced
   data (positive class < ~20%), PR-AUC. Report both if
   you can — they answer slightly different questions.
3. **Am I picking an operating threshold today?** Report
   precision, recall, and F1 (or F$_\beta$) at that
   threshold. State the threshold explicitly. Do not tune
   it on the test set.
4. **Do I care about the model as a probability, not just
   as a ranker?** Report log loss and go to chapter 04 for
   calibration.
5. **Always report a companion.** A ROC-AUC alongside
   accuracy. A PR-AUC alongside precision / recall at the
   chosen threshold. A log-loss alongside F1.

## Common pitfalls

1. **Reporting accuracy on imbalanced data.** The single
   most common mistake in classification write-ups.
2. **Reporting only ROC-AUC on rare-event detection.** The
   number looks fine while the model is useless.
3. **Tuning the decision threshold on the test set.** The
   test set is for reporting, not for choosing the
   operating point.
4. **`predict_proba` on models that don't produce
   probabilities.** SVMs and models trained on hinge loss
   produce scores that are *monotone* in probability but
   not probabilities themselves. Chapter 04's calibration
   step is required before treating them as probabilities.
5. **Micro-averaged F1 that equals accuracy.** On multi-
   class classification with a single label per row,
   micro-F1 equals accuracy. Reporting it as if it were an
   imbalance-robust metric is a mistake.
6. **`f1_score(y_true, y_pred)` on a multi-class
   problem.** Default `average="binary"` raises; readers
   who do not spot the raise pick `average="weighted"` by
   accident, which behaves like accuracy. Choose the
   average deliberately.
7. **Comparing log loss across datasets with different
   class balances.** The metric is scale-sensitive; use it
   to compare models on the same dataset, not to compare
   the same model across datasets.

## Summary

- The confusion matrix (TP, FP, FN, TN) is the primitive.
  Every count-based classification metric is a ratio of
  its cells.
- Accuracy is fine when classes are balanced and costs are
  symmetric — and misleading everywhere else. Reach for
  precision / recall / F$_\beta$ when the decision has an
  asymmetric cost.
- Every count-based metric depends on the decision
  threshold; state it. Pick it on validation, not on test.
- ROC-AUC is threshold-independent and imbalance-
  sensitive. PR-AUC is the right threshold-independent
  metric for rare-positive-class problems.
- Log loss is a proper scoring rule — the segue into
  calibration.

## Where this goes next

Chapter 04 is calibration: distinguishing "the model ranks
correctly" from "the model's score behaves like a
probability", and the reliability diagram / Brier score /
isotonic-regression toolbox for getting from the first to
the second.
