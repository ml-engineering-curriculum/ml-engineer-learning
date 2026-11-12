# lab-03 — Reliability diagram and recalibrating a biased classifier

> **Duration:** ~30–45 minutes.
> **Companion chapter:** [04 — Calibration](../04-calibration.md).

## Objective

Reproduce chapter 04's "a strong ranker can be a bad probability" story on
a classifier whose miscalibration is **predictable** — one trained with
`class_weight="balanced"` on an imbalanced dataset, where chapter 04
explicitly warns that "the model no longer sees the true prior; its scores
are on the resampled prior". By the end you will have (a) fit a well-
ranking logistic regression that is *not* well-calibrated, (b) drawn its
reliability diagram and computed ECE and Brier, (c) recalibrated it with
`CalibratedClassifierCV(method="isotonic")` using an internal held-out
split, and (d) compared ECE / Brier / log loss / ROC-AUC before and after
to confirm recalibration moves the probability score closer to the diagonal
without disturbing the ranking.

This lab is deliberately narrower than **exercise-01** (which authors a
full metric-selection rubric including a calibration branch) and
**exercise-04** (which produces the three-dimensional slice report for
fairness and robustness). The lab makes chapter 04's two-axis framing —
ranking versus calibration — concrete on a single classifier small enough
to eyeball.

## Prerequisites

- Python 3.11+, `scikit-learn ≥ 1.4`, `numpy`, `matplotlib`.
- Chapter 04 read through once.
- A CPU is enough.
- Any virtual environment (`uv`, `poetry`, or a plain `venv`).

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-03-calibration && cd lab-03-calibration
touch calibration.py
```

Start the file with imports and a seed:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(seed=0)
```

## Steps

### Step 1 — Build a 5%-positive dataset and a three-way split (≈ 5 min)

Chapter 04's three-way split: **train** (fits the base model), **calib**
(fits the recalibrator; base model does not see it), **test** (untouched
until the end). `CalibratedClassifierCV` handles the train/calib split
internally via CV, so for this lab we just need a train / test cut on an
imbalanced dataset.

Add to `calibration.py`:

```python
X, y = make_classification(
    n_samples=20_000,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    weights=[0.95, 0.05],   # 5% positives
    flip_y=0.01,
    random_state=0,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y,
)
print("train rows:", X_tr.shape[0], " base rate:", f"{y_tr.mean():.4f}")
print("test  rows:", X_te.shape[0], " base rate:", f"{y_te.mean():.4f}")
```

**Expected output.** 15,000 training rows / 5,000 test rows, both with a
positive base rate near `0.0500`. `stratify=y` preserves the base rate.

**Checkpoint.** Chapter 04's rule — "recalibrate on original-distribution
data" — means the *test* base rate matters. If you resample the training
set (next step) and never evaluate calibration on an original-prior
holdout, you have no idea whether the scores are on the deployed prior or
the resampled prior.

### Step 2 — Fit a well-ranking, mis-calibrated base model (≈ 5 min)

Chapter 04's named failure mode: `class_weight="balanced"` makes the
model see a 50/50 prior during training, so its scores are calibrated to
the balanced prior, not the deployed 5% prior. The ROC-AUC will be fine;
the predicted probabilities will not.

Add to `calibration.py`:

```python
base = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",   # chapter 04's "breaks calibration" lever
    random_state=0,
).fit(X_tr, y_tr)

proba_base = base.predict_proba(X_te)[:, 1]

print("\nBase model (class_weight='balanced'):")
print(f"  ROC-AUC   = {roc_auc_score(y_te, proba_base):.4f}")
print(f"  Brier     = {brier_score_loss(y_te, proba_base):.4f}")
print(f"  log loss  = {log_loss(y_te, np.clip(proba_base, 1e-6, 1 - 1e-6)):.4f}")
print(f"  mean proba = {proba_base.mean():.4f}   (vs test base rate = "
      f"{y_te.mean():.4f})")
```

**Expected output.** ROC-AUC near `0.88–0.92` (strong ranking), but the
**mean predicted probability** is much higher than the `0.05` test base
rate — typically around `0.25–0.45`. That gap is the signature of chapter
04's "training resampling / reweighting breaks calibration": the model
believes the base rate is 50%, so it over-predicts positives relative to
reality.

**Checkpoint.** The two-dimensions framing from chapter 04 is already
visible in the numbers. ROC-AUC says "the ranking is good"; the mean
probability vs the base rate says "the scores are not on the right
scale". A downstream expected-value calculation that treats
`proba_base` as a probability of fraud / churn / default would
over-count by roughly the ratio between the mean proba and the base
rate. Which is chapter 04's "when to calibrate: the score is used in a
downstream expected-value calculation" bullet.

### Step 3 — Compute ECE and draw the reliability diagram (≈ 10 min)

Chapter 04's primary calibration diagnostic: sort predictions into bins,
plot empirical positive rate vs mean predicted probability, read how far
the curve sits from the diagonal. A single-number summary is **ECE** —
the weighted-by-bin-size mean of the gap.

Scikit-learn's `calibration_curve` returns the two arrays
(`prob_true`, `prob_pred`); you compute ECE yourself from them (chapter
04: "scikit-learn does not currently ship an `expected_calibration_error`
function; compute it from `calibration_curve` outputs"). Use
`strategy="quantile"` and `n_bins=15` per chapter 04's advice.

Add to `calibration.py`:

```python
def expected_calibration_error(y_true, proba, n_bins=15, strategy="quantile"):
    """ECE from calibration_curve outputs.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    We weight each bin by the number of samples that fell into it —
    calibration_curve gives us (prob_true, prob_pred); we also need
    the per-bin counts, which we recompute from the binning.
    """
    if strategy == "quantile":
        edges = np.quantile(proba, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    # Guard against degenerate repeated edges (quantile on heavy-tail proba).
    edges = np.unique(edges)
    bin_ids = np.clip(np.digitize(proba, edges[1:-1]), 0, len(edges) - 2)
    ece = 0.0
    n = len(proba)
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = proba[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


prob_true, prob_pred = calibration_curve(
    y_te, proba_base, n_bins=15, strategy="quantile"
)
ece_base = expected_calibration_error(
    y_te, proba_base, n_bins=15, strategy="quantile"
)
print(f"\nBase model ECE (quantile, n_bins=15): {ece_base:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax.plot(prob_pred, prob_true, "o-", label="base model")
ax.set_xlabel("mean predicted probability (bin)")
ax.set_ylabel("empirical positive rate (bin)")
ax.set_title(f"Reliability diagram — base model\nECE = {ece_base:.3f}")
ax.legend()
fig.savefig("reliability_base.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved reliability_base.png")
```

Run `python calibration.py`.

**Expected output.** An ECE value typically in the `0.15–0.35` range —
comfortably above chapter 04's "ECE > 0.1 is a poor probability" line —
and a saved `reliability_base.png` whose curve sits consistently *below*
the diagonal (over-confident: when the model predicts 0.5 the empirical
positive rate is much lower). The curve is well-separated from the
diagonal across almost every bin.

**Checkpoint.** You can now see chapter 04's "ROC-AUC says nothing about
calibration" claim in one picture. The same classifier that reports
ROC-AUC near 0.9 produces a reliability diagram that is nowhere near the
diagonal. ECE > 0.1 says "either recalibrate or use the score only for
ranking-based decisions".

### Step 4 — Recalibrate with `CalibratedClassifierCV(method="isotonic")` (≈ 10 min)

Chapter 04's `CalibratedClassifierCV` recipe. Isotonic is the right
choice here because we have > 1,000 calibration samples per fold (the
chapter's threshold for using isotonic rather than Platt) and the
miscalibration is monotone but not necessarily sigmoid-shaped. `cv=5`
means "fit five base models on 4/5 of the training data, calibrate each
on the held-out fold, average the resulting predictions" — exactly
chapter 04's "who calibrates the calibrator?" solution.

Add to `calibration.py`:

```python
cal = CalibratedClassifierCV(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
    method="isotonic",
    cv=5,
).fit(X_tr, y_tr)

proba_cal = cal.predict_proba(X_te)[:, 1]
ece_cal = expected_calibration_error(
    y_te, proba_cal, n_bins=15, strategy="quantile"
)
prob_true_c, prob_pred_c = calibration_curve(
    y_te, proba_cal, n_bins=15, strategy="quantile"
)

print("\nCalibrated model (CalibratedClassifierCV isotonic, cv=5):")
print(f"  ROC-AUC   = {roc_auc_score(y_te, proba_cal):.4f}")
print(f"  Brier     = {brier_score_loss(y_te, proba_cal):.4f}")
print(f"  log loss  = {log_loss(y_te, np.clip(proba_cal, 1e-6, 1 - 1e-6)):.4f}")
print(f"  ECE       = {ece_cal:.4f}")
print(f"  mean proba = {proba_cal.mean():.4f}   (vs test base rate = "
      f"{y_te.mean():.4f})")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax.plot(prob_pred,   prob_true,   "o-",  label=f"base (ECE={ece_base:.3f})")
ax.plot(prob_pred_c, prob_true_c, "s--", label=f"isotonic (ECE={ece_cal:.3f})")
ax.set_xlabel("mean predicted probability (bin)")
ax.set_ylabel("empirical positive rate (bin)")
ax.set_title("Reliability diagram — before vs after isotonic recalibration")
ax.legend()
fig.savefig("reliability_before_after.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved reliability_before_after.png")
```

Run `python calibration.py`.

**Expected output.** Four things, each in its own column:

- **ROC-AUC** should be essentially unchanged (within a thousandth or two)
  — chapter 04's point that recalibration is a monotone transform of the
  scores and so preserves ranking.
- **ECE** should drop markedly, typically into the `0.01–0.03` range —
  below chapter 04's "ECE < 0.02 is treatable as probability" line.
- **Brier** should drop and **log loss** should drop — both are proper
  scoring rules, so a better-calibrated model with the same ranking
  scores better on both.
- **Mean predicted probability** should now be close to the `0.05` test
  base rate.

The saved figure shows the "before" curve below the diagonal and the
"after" curve hugging it. That visual difference is the whole point of
chapter 04.

**Checkpoint.** Chapter 04's rule: "ROC-AUC = 0.95 says nothing about
calibration" is now a two-column numerical fact — the ROC-AUC is
unchanged between the two rows, and every calibration-sensitive metric
(ECE, Brier, log loss, mean proba vs base rate) has moved. The
threshold-independent summary (ROC-AUC) is a ranker's metric; the
scoring-rule metrics are the calibration-aware ones.

### Step 5 — Verify the decision at `t = 0.5` has actually shifted (≈ 5 min)

Chapter 04's note on recalibration and threshold tuning: isotonic
recalibration is a monotone warp of the scores, so **it does not change
the ranking** and therefore does not change AUCs. But at a fixed
threshold like `0.5`, the number of positives flagged changes, because
`0.5` on the base-model score means something different from `0.5` on
the calibrated score.

Add to `calibration.py`:

```python
y_hat_base = (proba_base >= 0.5).astype(int)
y_hat_cal  = (proba_cal  >= 0.5).astype(int)
print(f"\nAt threshold 0.5:")
print(f"  base      positives predicted = {y_hat_base.sum():5d}  "
      f"(fraction {y_hat_base.mean():.4f})")
print(f"  calibrated positives predicted = {y_hat_cal.sum():5d}  "
      f"(fraction {y_hat_cal.mean():.4f})")
print(f"  test positive count          = {y_te.sum():5d}  "
      f"(fraction {y_te.mean():.4f})")
```

**Expected output.** The base model at `t = 0.5` typically flags many
more rows than the calibrated model does — often 10-20% of the test set
vs just a few percent after calibration. Neither matches the true `0.05`
rate exactly, because the operating threshold is a *separate* decision,
but the calibrated flag-rate lives in the same neighbourhood as the base
rate.

**Checkpoint.** Chapter 04's "Don't bother calibrating when ... the
threshold is picked on a held-out validation set" bullet is subtler than
it sounds: the calibrated and uncalibrated models produce the *same*
ROC-AUC, PR-AUC, and best-F1 across a threshold sweep — but they
disagree about *where* those operating points live on the `[0, 1]`
score axis. If downstream code assumes `score >= 0.5` means
"probability of positive is at least 50%", that assumption only holds
after calibration. If downstream code uses a validation-picked
threshold `t*` and nothing else, calibration changes `t*` without
changing the resulting metrics.

### Step 6 — Encode the invariants as a test (≈ 5 min)

Chapter 04's two invariants a recalibration should satisfy:

- **Ranking is preserved** — ROC-AUC does not change meaningfully
  (recalibration is a monotone transform).
- **Calibration improves** — ECE / Brier / log loss all go down.

Append to `calibration.py`:

```python
def test_isotonic_preserves_ranking() -> None:
    # Chapter 04: isotonic is monotone, so ROC-AUC is preserved up to the
    # tie-handling of the recalibrator. A millisecond of floating-point
    # slack is fine; a > 0.01 drop would be a bug.
    auc_base = roc_auc_score(y_te, proba_base)
    auc_cal  = roc_auc_score(y_te, proba_cal)
    assert abs(auc_base - auc_cal) < 0.01, (
        f"ROC-AUC moved by {auc_base - auc_cal:+.4f}; recalibration "
        "should preserve ranking within tie-handling slack"
    )
    print(f"OK: ROC-AUC preserved ({auc_base:.4f} -> {auc_cal:.4f}).")


def test_calibration_improves() -> None:
    # Chapter 04: ECE, Brier, and log loss should all decrease on a model
    # that was mis-calibrated by resampling / reweighting.
    brier_base = brier_score_loss(y_te, proba_base)
    brier_cal  = brier_score_loss(y_te, proba_cal)
    ll_base = log_loss(y_te, np.clip(proba_base, 1e-6, 1 - 1e-6))
    ll_cal  = log_loss(y_te, np.clip(proba_cal,  1e-6, 1 - 1e-6))
    assert ece_cal < ece_base, (
        f"ECE did not decrease: {ece_base:.4f} -> {ece_cal:.4f}"
    )
    assert brier_cal < brier_base, (
        f"Brier did not decrease: {brier_base:.4f} -> {brier_cal:.4f}"
    )
    assert ll_cal < ll_base, (
        f"log loss did not decrease: {ll_base:.4f} -> {ll_cal:.4f}"
    )
    print("OK: ECE, Brier, and log loss all decreased after recalibration.")


test_isotonic_preserves_ranking()
test_calibration_improves()
```

Run `python calibration.py`.

**Expected output.** Two `OK` lines at the end of the run. The first
asserts the "monotone warp preserves ranking" property; the second
asserts the three scoring-rule metrics all moved in the right direction.

**Checkpoint.** On a real project, these two assertions would live under
`tests/test_calibration.py` and run in CI whenever the training pipeline
changes. They are a defence against two realistic regressions: a
recalibrator that silently reverses the ranking (badly configured
isotonic on tiny data — chapter 04's "Using isotonic with tiny
calibration sets" pitfall) and a pipeline change that quietly disables
the recalibration step so the deployed scores are back on the resampled
prior.

## Expected final artefact

A directory `lab-03-calibration/` containing:

```
lab-03-calibration/
├── calibration.py                  # the six step blocks
├── reliability_base.png            # step 3 — base-model reliability curve
└── reliability_before_after.png    # step 4 — overlay of base vs calibrated
```

Running `python calibration.py` end to end should take under a minute on
a laptop and print (in order) the split summary, the base-model
scoring-rule metrics, the base ECE, the calibrated-model scoring-rule
metrics, the before/after `t = 0.5` positives counts, and the two
assertion `OK` lines. The two PNGs land in the working directory.

## Final checkpoint

Without re-reading chapter 04, answer:

1. In step 2 the base model's mean predicted probability was far above
   the 5% base rate. Chapter 04 names the specific training-time choice
   that caused this. Which choice, and why does it break calibration
   while leaving ROC-AUC intact?
2. In step 4, ROC-AUC did not move but Brier and log loss both dropped
   after isotonic recalibration. Explain in your own words why *both*
   facts are necessarily true of a correct monotone recalibrator, from
   the definition of each metric.
3. Chapter 04's recommendation table lists Platt for SVM, isotonic for
   random forest / GBDT, and temperature scaling for deep classifiers.
   For this lab you chose isotonic. Suppose the training set had 300
   samples instead of 15,000 — which recalibrator does chapter 04 say
   to pick, and what failure mode is isotonic prone to on small data?
4. The step-5 output showed that `t = 0.5` means something different on
   the base and calibrated scores. If the production code uses a
   validation-tuned threshold rather than a hardcoded 0.5, does chapter
   04 say recalibration is still worth doing? Which of chapter 04's
   "do / don't bother" bullets is relevant?

If you can answer all four, chapter 04's calibration story is
internalised and you are ready to apply it on a real model in the
calibration branch of exercise-01 and on the probability-aggregation
part of a real deployment.

## Where to go next

- **`exercise-01` ("Metric selection rubric")** includes a calibration
  branch; this lab's base model, recalibrated model, and the ECE / Brier
  / log loss numbers can plug directly into the `metric_report.py`
  module that exercise builds.
- **`exercise-04` ("Fairness and robustness slices")** reports a
  subgroup-level metric table; combining with calibration is a natural
  next step — the chapter calls out that recalibration on the full
  population can leave subgroup calibration *worse*, which is the open
  door to a stretch goal.
- **`lab-01` ("Accuracy, ROC-AUC, and PR-AUC disagree under class
  imbalance")** is the "which metric do I report?" companion; **`lab-02`
  ("Preprocessing leakage")** is the "is the number honest?" companion.
  The three labs together produce the vocabulary chapters 02–06 teach.
