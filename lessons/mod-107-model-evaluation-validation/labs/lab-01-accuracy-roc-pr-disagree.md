# lab-01 — Accuracy, ROC-AUC, and PR-AUC disagree under class imbalance

> **Duration:** ~30–45 minutes.
> **Companion chapters:** [03 — Metrics for classification](../03-metrics-for-classification.md), [09 — Baselines that tell you something](../09-baselines.md).

## Objective

Reproduce chapter 03's central imbalance claim — "a model with ROC-AUC 0.99
can hide a precision of 0.05 if the positive class is rare" — on a synthetic
dataset small enough to eyeball. By the end you will have (a) built a
dataset with a 1% positive-class base rate, (b) scored four models on it —
`DummyClassifier(most_frequent)`, `DummyClassifier(stratified)`, a tuned
logistic regression, and an intentionally weak random classifier — (c)
watched **accuracy** declare the dummy model a winner, **ROC-AUC** declare
all three "real" models similar, and **PR-AUC** actually separate them, and
(d) confirmed that `average_precision_score`'s baseline is approximately the
positive-class base rate (chapter 03's "PR-AUC ≈ base rate" reference
point), which is what makes the metric meaningful on rare events.

This lab is deliberately narrower than **exercise-01** (which authors a full
decision-tree rubric, implements a `metric_report` module, and defends four
case studies from a menu). The lab is the isolation chamber where you *see*
the imbalance disagreement one row of a baselines table at a time; the
exercise then wires the rule into a reusable report module.

## Prerequisites

- Python 3.11+, `scikit-learn ≥ 1.4`, `numpy`, `pandas`.
- Chapters 03 and 09 read through once.
- A CPU is enough; nothing in this lab needs a GPU.
- Any virtual environment (`uv`, `poetry`, or a plain `venv`).

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-01-imbalance-metrics && cd lab-01-imbalance-metrics
touch imbalance_metrics.py
```

Everything below goes in `imbalance_metrics.py`. Run with
`python imbalance_metrics.py` after each step and read the output before
moving on.

Start the file with imports and a seed:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(seed=0)
```

## Steps

### Step 1 — Build a 1%-positive-class dataset (≈ 5 min)

Chapter 03's canonical imbalance example is a fraud-like dataset. Build
20,000 rows with roughly 1% positives, a modest amount of signal, and a
stratified test split.

Add to `imbalance_metrics.py`:

```python
X, y = make_classification(
    n_samples=20_000,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    weights=[0.99, 0.01],   # 1% positives
    flip_y=0.01,
    random_state=0,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y,
)

print("train rows:", X_tr.shape[0], "  positives:", int(y_tr.sum()),
      " base rate:", f"{y_tr.mean():.4f}")
print("test  rows:", X_te.shape[0], "  positives:", int(y_te.sum()),
      " base rate:", f"{y_te.mean():.4f}")
```

**Expected output.** 15,000 training rows and 5,000 test rows; each with a
positive base rate near `0.0100`. `stratify=y` is what keeps the base rate
identical across the split — chapter 06's rule for classification splits.

**Checkpoint.** On a 1% positive dataset the "always predict negative"
baseline hits 99% accuracy without catching anything. That is chapter 03's
opening complaint about accuracy; the next steps make it concrete.

### Step 2 — Score the trivial baselines with `DummyClassifier` (≈ 5 min)

Chapter 09's **sanity floor**: before any model, record what a zero-thought
baseline gets on the chosen metric suite. Chapter 09 names two
`DummyClassifier` strategies worth reporting — `"most_frequent"` and
`"stratified"` — and warns that a run whose `model_f1 < baseline_f1` is
visibly broken.

Add to `imbalance_metrics.py`:

```python
def score_all(name, y_true, y_hat, proba):
    """Return a dict of every metric chapter 03 names, plus a label."""
    return {
        "model":       name,
        "accuracy":    accuracy_score(y_true, y_hat),
        "precision":   precision_score(y_true, y_hat, zero_division=0),
        "recall":      recall_score(y_true, y_hat, zero_division=0),
        "f1":          f1_score(y_true, y_hat, zero_division=0),
        "roc_auc":     roc_auc_score(y_true, proba),
        "pr_auc":      average_precision_score(y_true, proba),
    }


dummy_mf = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
dummy_st = DummyClassifier(strategy="stratified", random_state=0).fit(X_tr, y_tr)

rows = [
    score_all(
        "DummyClassifier(most_frequent)",
        y_te,
        dummy_mf.predict(X_te),
        dummy_mf.predict_proba(X_te)[:, 1],
    ),
    score_all(
        "DummyClassifier(stratified)",
        y_te,
        dummy_st.predict(X_te),
        dummy_st.predict_proba(X_te)[:, 1],
    ),
]
```

**Expected output.** (No print yet — we'll print after step 3.) The two
dummies you have logged produce:

- `most_frequent` — always predicts 0, so `accuracy ≈ 0.99` (the base rate
  of the majority class), `precision = recall = f1 = 0.0`,
  `roc_auc = 0.5`, `pr_auc ≈ 0.01` (the base rate — chapter 03's reference
  point).
- `stratified` — predicts positives at their base rate, at random, so
  `accuracy ≈ 0.98` (because ~1% of its positives land on the 1% actual
  positives), `roc_auc ≈ 0.5`, `pr_auc ≈ 0.01`.

**Checkpoint.** `pr_auc ≈ 0.01` on both dummies is exactly chapter 03's
line: "the 'predict positive-class prevalence' baseline has PR-AUC ≈ base
rate". That is the number any "real" model has to clear to be meaningful.
`roc_auc = 0.5` on both dummies is the matching ROC-AUC reference point.

### Step 3 — Score a tuned logistic regression and a weak random classifier (≈ 10 min)

Chapter 09's **fair comparison**: a strong-classical baseline the model
should beat. Fit a logistic regression with `class_weight="balanced"` so
the loss is not dominated by negatives, and — for contrast — a deliberately
weak classifier that outputs almost-noise scores so you can watch the
metrics react.

Add to `imbalance_metrics.py`:

```python
# A reasonable strong-classical baseline.
logreg = LogisticRegression(
    max_iter=2000, class_weight="balanced", random_state=0,
).fit(X_tr, y_tr)

# A deliberately weak "model": a tiny bit of signal plus heavy noise.
# This simulates an under-fit model so the metrics can disagree visibly.
class WeakClassifier:
    def __init__(self, signal=0.05):
        self.signal = signal

    def predict_proba(self, X):
        base = RNG.uniform(0.0, 0.3, size=(X.shape[0], 1))
        # A tiny amount of true-feature signal on top of the noise.
        base += self.signal * (X[:, :1] > 0).astype(float)
        p1 = base.clip(0.0, 1.0)
        return np.hstack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


weak = WeakClassifier(signal=0.05)

for name, model in [
    ("LogisticRegression (balanced)", logreg),
    ("WeakClassifier (noise + tiny signal)", weak),
]:
    proba = model.predict_proba(X_te)[:, 1]
    y_hat = model.predict(X_te)
    rows.append(score_all(name, y_te, y_hat, proba))

df = pd.DataFrame(rows).round(4)
print(df.to_string(index=False))
```

Run `python imbalance_metrics.py`.

**Expected output.** A four-row table whose shape resembles:

```
                               model  accuracy  precision  recall      f1  roc_auc  pr_auc
      DummyClassifier(most_frequent)    0.9900     0.0000  0.0000  0.0000   0.5000  0.0100
         DummyClassifier(stratified)    0.98xx     0.0xxx  0.0xxx  0.0xxx   0.5xxx  0.0xxx
       LogisticRegression (balanced)    0.9xxx     0.1xxx  0.8xxx  0.2xxx   0.98xx  0.3xxx
WeakClassifier (noise + tiny signal)    0.98xx     0.0xxx  0.0xxx  0.0xxx   0.5xxx  0.0xxx
```

The exact digits depend on the RNG, but the ordinal pattern is what
matters: `most_frequent` and `WeakClassifier` have accuracy comparable to
the logistic regression (all three are within ~1 point of 99%), yet their
PR-AUC is pinned near the 0.01 base rate while the logistic regression's
PR-AUC is roughly an order of magnitude higher.

**Checkpoint.** This is chapter 03's "Accuracy — and why it fails" and "PR-
AUC — the imbalanced-classes companion" made numerical. Accuracy on the
right-hand side is nearly identical across three of the four rows, so
accuracy-as-headline would treat the dummy and the weak classifier as
near-ties with the real model. ROC-AUC separates the real model from the
random ones — but on this dataset ROC-AUC is still misleading at the
decision threshold (the next step makes this visible). PR-AUC is the
metric that puts the strong-classical row in a visibly different league
from everything else.

### Step 4 — Open the confusion matrix at the default 0.5 threshold (≈ 10 min)

Chapter 03: "every count-based metric depends on the threshold; state it."
The ROC-AUC and PR-AUC numbers above are threshold-free summaries; the
precision / recall / F1 columns were computed at `0.5`. Pull the confusion
matrix for the logistic regression at that threshold and read it against
the model's PR-AUC claim.

Add to `imbalance_metrics.py`:

```python
proba = logreg.predict_proba(X_te)[:, 1]
y_hat = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, y_hat).ravel()
print("\nLogisticRegression @ threshold 0.5:")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  precision = TP/(TP+FP) = {tp/(tp+fp):.3f}"
      if (tp + fp) else "  precision undefined (no positives predicted)")
print(f"  recall    = TP/(TP+FN) = {tp/(tp+fn):.3f}"
      if (tp + fn) else "  recall undefined (no positives in test)")
```

**Expected output.** A confusion matrix roughly on the order of
`TN ≈ 4,700`, `FP ≈ 250`, `FN ≈ 10`, `TP ≈ 40` (exact counts vary with
the RNG and class_weight), plus a precision-and-recall pair. Precision is
low (15–20%) because the model — weighted to rescue recall — generates
many false positives relative to the tiny positive population.

**Checkpoint.** The ROC-AUC near 0.98 and the precision near 0.15 live
inside the same model's predictions. Chapter 03's warning — "a ROC-AUC of
0.99 can hide a precision of 0.05 if the positive class is rare" — is
exactly this configuration: the ROC-AUC's false-positive-rate denominator
is `FP + TN ≈ 4,950`, so the 250 false positives barely move the ROC
curve; precision's denominator is `TP + FP ≈ 290`, so the same 250 false
positives dominate. Two sides of the same prediction, two very different
summaries.

### Step 5 — Sweep the threshold on validation (≈ 5 min)

Chapter 03's rule: pick the threshold on the validation set for a specified
operational criterion. Reproduce the chapter's threshold-sweep snippet on
the logistic regression — the threshold is a hyperparameter, not a
physical constant.

Add to `imbalance_metrics.py`:

```python
print("\nThreshold sweep on logistic regression (test proba — demo only):")
print(f"  base rate = {y_te.mean():.4f}")
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    y_hat_t = (proba >= t).astype(int)
    p = precision_score(y_te, y_hat_t, zero_division=0)
    r = recall_score(y_te, y_hat_t, zero_division=0)
    f = f1_score(y_te, y_hat_t, zero_division=0)
    print(f"  t={t:.1f}   precision={p:.3f}   recall={r:.3f}   f1={f:.3f}")
```

**Expected output.** As the threshold moves from `0.1` to `0.9`, recall
falls monotonically and precision rises. There is a threshold near the
middle where F1 peaks — on this dataset typically around `t ≈ 0.4–0.6`
depending on the RNG. At `t = 0.9` recall collapses almost to zero.

**Checkpoint.** This is "The threshold is a hyperparameter" from chapter
03: every count-based metric depends on it, reporting them without stating
the threshold is meaningless, and the choice is a *validation* decision.
The output above is **not** a real validation sweep — it is demo code on
the test set so you can see the shape of the trade-off. In real work, you
pick `t` on a held-out validation fold for a target precision or recall,
then report precision / recall / F1 on test at **that** threshold.
Chapter 03 pitfall #3 ("Tuning the decision threshold on the test set")
spells this out.

### Step 6 — Write down the one-line "which metric do I report?" (≈ 5 min)

Everything in the final table is the input to chapter 03's "which metric
do I report?" decision tree. For this dataset — 1% positives, costs
asymmetric enough that `class_weight="balanced"` was a sensible training
choice, no probability aggregation downstream — the decision tree lands on:

- **PR-AUC** as the threshold-independent primary.
- **Precision and recall** (and F1 or F_β) at a validation-chosen
  threshold as the operational numbers.
- **ROC-AUC** reported as a companion — chapter 03's "report both if you
  can; they answer slightly different questions".
- Accuracy omitted or reported with the explicit caveat that the majority
  baseline hits 99%.

Add a final block to `imbalance_metrics.py` that prints the one-line
recommendation so the output of your lab run ends with the decision:

```python
print("\nDecision (chapter 03 decision tree):")
print("  primary:    PR-AUC (average_precision_score)")
print("  companion:  ROC-AUC + (precision, recall, F1) at validation-picked threshold")
print("  trivial baseline target: PR-AUC >",
      f"{y_te.mean():.4f}  (positive base rate)")
```

**Expected output.** A three-line decision summary with the base-rate PR-
AUC floor spelled out.

**Checkpoint.** Every number in the lab's main table maps to a row of
chapter 03's decision tree. The `most_frequent` and `WeakClassifier` rows
are the "fail the sanity floor" case. The logistic regression row is the
"strong classical baseline" row from chapter 09's four-row report. The
PR-AUC gap between them is what makes the comparison informative — which
is the whole point.

## Expected final artefact

A single file `lab-01-imbalance-metrics/imbalance_metrics.py` containing
the six step blocks above. Running `python imbalance_metrics.py` end to
end should take well under a minute on a laptop and print (in order) the
split summary, the four-row metrics table, the confusion matrix at
`t = 0.5`, the five-threshold sweep, and the one-line decision summary.

## Final checkpoint

Without re-reading chapter 03, answer:

1. In the step-3 table, `DummyClassifier(most_frequent)` and the logistic
   regression are within one percentage point on accuracy. A stakeholder
   says "both are ~99% accurate, so they're both fine." Which **one
   number** from the same table do you point at to show that the
   `DummyClassifier` is not fine, and what reference value do you compare
   it against?
2. In step 4, the logistic regression's ROC-AUC is near 0.98 while its
   precision at `t = 0.5` is below 0.20. State in your own words *why*
   the two numbers can both be true for the same set of predictions —
   specifically, what denominator makes each metric react differently to
   the 250 false positives.
3. The step-5 sweep shows F1 peaking between the `0.1` and `0.9`
   thresholds. Is reporting that peak F1 as "the model's F1" a defensible
   claim, or does chapter 03 call it out as a pitfall? Which pitfall, and
   what is the alternative?
4. For the deployment where the positive class is 1% and missing a
   positive is 10× more expensive than a false positive, chapter 03's F_β
   formula gives a specific β. Which β, which metric family does the
   choice sit in, and what sentence would you add to the PR description
   so a reviewer knows which trade-off you made?

If you can answer all four, chapter 03's imbalance story is internalised
and you are ready to apply it on real datasets in exercise-01
(metric selection rubric) and exercise-03 (baselines + error analysis).

## Where to go next

- **`exercise-01` ("Metric selection rubric")** turns this lab's one-line
  decision into a full decision tree covering regression, classification,
  calibration, and ranking, plus a `metric_report.py` module that reports
  a primary metric, a companion, and baselines automatically.
- **`exercise-03` ("Baselines and error analysis")** reuses the
  `DummyClassifier` rows from step 2 as part of the four-row baselines
  table, and extends with per-slice metrics and error clustering on top of
  the same model.
- **`lab-02` ("Preprocessing leakage: scaler before vs inside the
  `Pipeline`")** moves from "did we pick the right metric?" to "is the
  number itself honest?" by reproducing chapter 06's preprocessing-leakage
  footgun on a two-line change.
- **`lab-03` ("Reliability diagram and recalibrating a biased classifier")**
  takes the logistic regression you trained here — which is a strong
  ranker but miscalibrated because of `class_weight="balanced"` — and
  walks through chapter 04's `CalibratedClassifierCV` fix, with ECE and
  Brier scores before and after.
