# lab-02 — Prediction-distribution shift from a new-segment ramp: why "retrain on shift" is a trap

> **Duration:** ~35–45 minutes.
> **Companion chapter:** [04 — Concept drift and prediction-distribution shift](../04-concept-and-prediction-drift.md).

## Objective

Reproduce chapter 04's Scenario A verbatim — *"a new geography launches;
users there have different feature values; the model predicts more 'high
risk' because the region legitimately has more high-risk users. Data drift
is real; concept drift is not; the model is right. Retraining on the
drifted window without careful sampling **can hurt**"* — on a synthetic
dataset you can run on a laptop. By the end you will have (a) trained a
tiny classifier on an **old-segment-only** reference, (b) ramped in a
**second, legitimately different segment** over 14 days and watched the
`ml_prediction_score` histogram drift (`P(ŷ)` shifts, chapter 02's primary
label-free signal), (c) measured per-feature PSI on the input and watched
**data drift fire** (chapter 03), (d) joined synthetic ground truth from
the *true* relationship and watched **accuracy stay flat** (chapter 05's
"the model is still right" check), and (e) retrained **only on the
drifted last-7-days** and watched the new model **regress on the old
segment** — the exact failure mode chapter 06 pitfall #1 ("retraining on
drift alone") names, and the shape chapter 04's "Prediction shift is a
symptom, not a cause" section warns about.

This lab is deliberately narrower than **exercise-02** (which builds a
full drift-detection pipeline with calibrated thresholds and three
synthetic-incident test suites) and **exercise-03** (which builds the
auditable retraining trigger that would stop this kind of retrain). The
lab is the isolation chamber where you *see* why the three label-free
signals from chapters 03–04 can all fire while the model is still right
— and why a decision rule that retrains on any of them in isolation can
make things worse.

## Prerequisites

- Python 3.11+.
- `numpy`, `pandas`, `scikit-learn`, `matplotlib` installed (`uv venv &&
  uv pip install numpy pandas scikit-learn matplotlib` is fine).
- Chapter 04 read through once; chapter 03 for PSI context (lab-01 is a
  direct feeder — this lab reuses its `psi` function).
- A CPU is enough.

## Setup

Create the lab directory and a single file:

```bash
mkdir -p lab-02-prediction-shift && cd lab-02-prediction-shift
touch shift_lab.py
```

Start `shift_lab.py` with imports and the PSI function from lab-01 (the
chapter-03 epsilon version — copy it verbatim; no module boundary needed
for a lab):

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(seed=0)


def psi(reference: np.ndarray, current: np.ndarray,
        bins: np.ndarray, eps: float = 1e-6) -> float:
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    p = ref_hist / max(ref_hist.sum(), 1)
    q = cur_hist / max(cur_hist.sum(), 1)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Chapter 03 shortlist: JSD is bounded in [0, ln 2] and natural for
    probability distributions like the prediction histogram."""
    p = np.clip(p / max(p.sum(), 1e-12), eps, None)
    q = np.clip(q / max(q.sum(), 1e-12), eps, None)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) +
                 0.5 * np.sum(q * np.log(q / m)))
```

## Steps

### Step 1 — Build two legitimately different user segments (≈ 5 min)

Chapter 04's Scenario A is a two-segment world: the model was trained on
one segment (old users; the "launched" region); a second segment (new
users; the newly-launched region) ramps up in production. Both segments
have internally-consistent `P(y|X)` — the model is **right** on each
when evaluated inside the segment — but their marginal distributions of
`X` are different.

Append:

```python
def sample_segment(n: int, segment: str) -> pd.DataFrame:
    """Two features — `tenure_months` and `activity_score` — and a
    binary label whose *true* relationship is the same for both
    segments (chapter 04's Scenario A: concept is stable, data moves).

    - Segment A (old):  tenure ~ N(24, 8),  activity ~ N(0.6, 0.15)
    - Segment B (new):  tenure ~ N(6,  3),  activity ~ N(0.3, 0.15)

    The true `P(y=1 | X)` is identical in both segments:
    logistic in `(-0.05 * tenure + 2.0 * (1 - activity))`,
    i.e. 'low tenure and/or low activity → more likely to churn.'
    Segment B just has more of both — it is legitimately higher-risk.
    """
    if segment == "A":
        tenure = np.clip(RNG.normal(24, 8, n), 1, 120)
        activity = np.clip(RNG.normal(0.6, 0.15, n), 0, 1)
    else:
        tenure = np.clip(RNG.normal(6, 3, n), 1, 120)
        activity = np.clip(RNG.normal(0.3, 0.15, n), 0, 1)
    logit = -0.05 * tenure + 2.0 * (1.0 - activity) - 0.8
    p1 = 1.0 / (1.0 + np.exp(-logit))
    y = (RNG.uniform(size=n) < p1).astype(int)
    return pd.DataFrame({"tenure": tenure, "activity": activity,
                         "segment": segment, "y": y})


# Training data: 10,000 rows of segment A only. The new segment does not
# exist yet at training time.
train = sample_segment(10_000, segment="A")

print("train segment counts:", train["segment"].value_counts().to_dict())
print("train base rate:     ", round(train["y"].mean(), 3))
```

**Expected output.**

```
train segment counts: {'A': 10000}
train base rate:      0.?? (~0.20)
```

**Checkpoint.** The training set is a pure-segment-A world; the model will
never have seen segment B at training time. Chapter 04's Scenario A opens
here.

### Step 2 — Train a tiny model and freeze the reference distributions (≈ 5 min)

Chapter 02's reference profile stores the training-time distributions of
the inputs and the prediction histogram; chapter 03's drift job reads
them; chapter 04's prediction-shift signal is a drift test on the
prediction histogram specifically.

Append:

```python
# Fit a simple logistic regression — chapter 04 explicitly names
# "well-calibrated tree ensembles and logistic-regression variants" as the
# models CBPE assumes work. Logistic regression on the true logit is also
# the oracle for this synthetic world.
model_v1 = LogisticRegression().fit(train[["tenure", "activity"]], train["y"])

# Freeze the chapter-02 reference distributions on the training set.
reference_tenure = train["tenure"].to_numpy()
reference_activity = train["activity"].to_numpy()
reference_proba = model_v1.predict_proba(train[["tenure", "activity"]])[:, 1]

# Bin edges from the reference — same edges applied to current windows
# later (chapter 03 pitfall #4: different bin edges return silent
# nonsense).
tenure_bins   = np.quantile(reference_tenure,   np.linspace(0, 1, 11))
activity_bins = np.quantile(reference_activity, np.linspace(0, 1, 11))
proba_bins = np.array(
    [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

print("reference P(ŷ=1) mean:",
      round(float(reference_proba.mean()), 3))
print("reference P(ŷ=1) p95: ",
      round(float(np.quantile(reference_proba, 0.95)), 3))
```

**Expected output.** Something like:

```
reference P(ŷ=1) mean: 0.20?
reference P(ŷ=1) p95:  0.60?
```

**Checkpoint.** You now have the three artifacts chapter 02 asks for in
the smallest useful form: reference feature distributions (as `np.array`
samples), the reference **prediction** distribution (for chapter 04's
`P(ŷ)` drift test), and bin edges derived once from the reference. A
real `reference_profile.json` (exercise-01) would persist these to disk;
the lab holds them in memory.

### Step 3 — Ramp the second segment in over 14 days and record the three label-free signals (≈ 10 min)

Chapter 04's Scenario A is a *ramp*: segment B's share of traffic rises
from 0% to ~40% over roughly two weeks. Simulate it, and record the
three signals you have *before any labels arrive* — the per-feature
drift (chapter 03), the multivariate signal via a cheap PCA-style
reconstruction check (chapter 03 shortlist's "cheap complementary
signal"), and the prediction-distribution shift (chapter 04's free
proxy).

Append:

```python
N_DAYS = 14
DAILY_N = 2_000

rows = []
for day in range(N_DAYS):
    # Share of segment B rises linearly from 0.00 on day 0 to 0.40 on day
    # 13. Realistic for a new region ramping up in the product.
    share_B = 0.40 * (day / (N_DAYS - 1))
    n_B = int(DAILY_N * share_B)
    n_A = DAILY_N - n_B

    df = pd.concat([
        sample_segment(n_A, segment="A"),
        sample_segment(n_B, segment="B"),
    ], ignore_index=True)

    proba = model_v1.predict_proba(df[["tenure", "activity"]])[:, 1]

    # Chapter 03 — per-feature PSI.
    psi_tenure   = psi(reference_tenure,   df["tenure"].to_numpy(),
                       tenure_bins)
    psi_activity = psi(reference_activity, df["activity"].to_numpy(),
                       activity_bins)

    # Chapter 04 — prediction-distribution shift. JSD on the P(ŷ)
    # histogram vs. the reference prediction histogram; chapter 03
    # shortlist: "Probability distribution → JSD."
    ref_hist = np.histogram(reference_proba, bins=proba_bins)[0]
    cur_hist = np.histogram(proba,           bins=proba_bins)[0]
    jsd_proba = jsd(ref_hist, cur_hist)

    rows.append({
        "day": day, "share_B": share_B,
        "psi_tenure": psi_tenure,
        "psi_activity": psi_activity,
        "jsd_proba":    jsd_proba,
        "pred_mean":    float(proba.mean()),
        "pred_p95":     float(np.quantile(proba, 0.95)),
    })

daily = pd.DataFrame(rows)
print(daily.round(3).to_string(index=False))
```

**Expected output.** A 14-row table with rising numbers in all three
drift columns. Something like:

```
 day  share_B  psi_tenure  psi_activity  jsd_proba  pred_mean  pred_p95
   0    0.000       0.00?         0.00?      0.00?      0.20?      0.60?
   1    0.031       0.01?         0.01?      0.00?      0.21?      0.61?
   ...
  13    0.400       0.6??         0.6??      0.0??      0.3??      0.8??
```

Three chapters' signals all fire together: per-feature PSI on both
`tenure` and `activity` (chapter 03) rises monotonically from ~0 to a
clearly-breached value; JSD on the prediction histogram (chapter 04)
rises monotonically; the live prediction mean rises from ~0.20 to
~0.30+ (a ~50% jump in the predicted-positive rate — chapter 04's
"predicted churn 2× as often this week" business-readable framing).
Draw them:

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].plot(daily["day"], daily["psi_tenure"], label="tenure")
axes[0].plot(daily["day"], daily["psi_activity"], label="activity")
axes[0].set_title("ch 03 — per-feature PSI")
axes[0].set_xlabel("day"); axes[0].set_ylabel("PSI"); axes[0].legend()
axes[1].plot(daily["day"], daily["jsd_proba"])
axes[1].set_title(r"ch 04 — JSD on $P(\hat{y})$")
axes[1].set_xlabel("day"); axes[1].set_ylabel("JSD")
axes[2].plot(daily["day"], daily["pred_mean"], label="mean")
axes[2].plot(daily["day"], daily["pred_p95"],  label="p95")
axes[2].set_title("ch 04 — live prediction stats")
axes[2].set_xlabel("day"); axes[2].legend()
fig.tight_layout()
fig.savefig("shift_signals.png", dpi=120)
print("wrote shift_signals.png")
```

**Checkpoint.** You have reproduced chapter 04's claim that prediction
shift **responds to both data drift and concept drift**: the prediction
histogram moves because `P(X)` moved, not because `P(y|X)` did. At this
point an oncall engineer looking only at label-free signals has every
reason to believe the model is broken. Chapter 04's next move — and
the point of this lab — is to look at the ground truth before deciding
what to do about it.

### Step 4 — Join synthetic ground truth and watch accuracy stay flat (≈ 7 min)

Chapter 05 is the plumbing this step assumes was already there: every
prediction has a `prediction_id`, labels arrive in the warehouse with
the same key, and a `predictions_labeled` table joins them. For the
lab, the ground truth **is** the `y` column the `sample_segment`
function emitted — pretend those were the chapter-05 labels, now
mature.

Append:

```python
# For day 13 — peak-drift day — join labels and measure accuracy. The
# labels are generated from the *true* relationship, which is the same
# in both segments; the model trained on segment A applies that same
# relationship to the rows from segment B. Prediction: accuracy is
# flat; the model is not broken.

day_13 = pd.concat([
    sample_segment(DAILY_N - int(DAILY_N * 0.40), segment="A"),
    sample_segment(int(DAILY_N * 0.40),           segment="B"),
], ignore_index=True)

pred = (model_v1.predict_proba(day_13[["tenure", "activity"]])[:, 1]
        >= 0.5).astype(int)
day_13["pred"] = pred

print("overall accuracy (day 13):",
      round((day_13["pred"] == day_13["y"]).mean(), 3))
# The chapter-04 "slice the dashboard" discipline: break accuracy out by
# the slice that matters (segment).
print("slice accuracy:")
for seg in ["A", "B"]:
    sub = day_13[day_13["segment"] == seg]
    acc = (sub["pred"] == sub["y"]).mean()
    print(f"  segment {seg}: n={len(sub)}  accuracy={acc:.3f}")

# And the reference-window accuracy for comparison (chapter 04 pitfall
# #5: "No baseline for the metric. 'Accuracy is 0.83' is meaningless
# without 'the baseline is 0.87.'").
train_pred = (model_v1.predict_proba(train[["tenure", "activity"]])[:, 1]
              >= 0.5).astype(int)
print("reference (train-set) accuracy:",
      round((train_pred == train["y"]).mean(), 3))
```

**Expected output.** Something like:

```
overall accuracy (day 13):          0.8??
slice accuracy:
  segment A: n=1200  accuracy=0.8??
  segment B: n=800   accuracy=0.8??
reference (train-set) accuracy:     0.8??
```

Every one of those numbers is within noise of the training-set
reference. Chapter 04's Scenario A diagnosis: **the model is right. Data
drift is real. Prediction drift is real. Concept drift is not.** The
three label-free signals from step 3 all fired; the one accuracy signal
that matters is quiet. This is the specific shape chapter 04 warns about
twice — "Prediction shift is a symptom, not a cause" and "Retraining on
prediction shift alone is chapter 06's first pitfall."

**Checkpoint.** Without the step-4 ground-truth join you cannot
distinguish Scenario A from Scenario B (chapter 04) — the same
label-free signals fire in both. Chapter 05's whole infrastructure
exists to make this step possible, delayed-by-weeks or not.

### Step 5 — Spring the trap: retrain on the drifted last-7-days and measure the old segment (≈ 10 min)

Chapter 06 pitfall #1: "Retraining on drift alone … can make things
worse." Chapter 06's "Sampling for the retrain — the trap door" section
is specific: *"A retrain that trains only on the last 7 days of a
drifting distribution forgets everything the previous 6 months knew."*
Reproduce it.

Append:

```python
# A naïve trigger fires on day 13: multivariate drift up, prediction
# shift up, feature drift sustained. It retrains the model on the most
# recent 7 days of production traffic (days 7 through 13), stratified
# only by whatever showed up (so ~25% of the retrain data is segment B,
# 75% segment A) — chapter 06's "trap door" sampling.
recent_days = []
for day in range(7, N_DAYS):
    share_B = 0.40 * (day / (N_DAYS - 1))
    n_B = int(DAILY_N * share_B)
    n_A = DAILY_N - n_B
    recent_days.append(pd.concat([
        sample_segment(n_A, segment="A"),
        sample_segment(n_B, segment="B"),
    ], ignore_index=True))
retrain_df = pd.concat(recent_days, ignore_index=True)

print("retrain segment counts:", retrain_df["segment"].value_counts().to_dict())

model_v2 = LogisticRegression().fit(
    retrain_df[["tenure", "activity"]], retrain_df["y"],
)

# Now measure *both* models on a clean evaluation sample from each
# segment — chapter 04's "Slice the monitoring dashboard" + chapter 06's
# "Backtest on a stable holdout" + "Slice checks" from the evaluation
# gate.
eval_A = sample_segment(5_000, segment="A")
eval_B = sample_segment(5_000, segment="B")


def acc(model, df):
    pred = (model.predict_proba(df[["tenure", "activity"]])[:, 1]
            >= 0.5).astype(int)
    return float((pred == df["y"]).mean())


table = pd.DataFrame([
    {"model": "v1 (segment-A training)",
     "acc_segment_A": acc(model_v1, eval_A),
     "acc_segment_B": acc(model_v1, eval_B)},
    {"model": "v2 (drifted-7-day retrain)",
     "acc_segment_A": acc(model_v2, eval_A),
     "acc_segment_B": acc(model_v2, eval_B)},
])
print(table.round(3).to_string(index=False))
```

**Expected output.** Something like:

```
                     model  acc_segment_A  acc_segment_B
    v1 (segment-A training)          0.8??          0.8??
v2 (drifted-7-day retrain)           0.7??          0.8??
```

The retrained `v2` improves on segment B by a marginal amount (segment B
rows are now in the training set) and **regresses on segment A** by a
larger amount — segment A rows are now a minority in a short, drifting
training window. The aggregate accuracy may stay roughly flat because
segment A is still the majority of production traffic — chapter 06's
"silent retraining regression" named failure mode: *"Every retrain
slightly regresses on a small slice. Over many retrains the slice's
accuracy has degraded 20%. The aggregate metric is fine. Nobody
noticed."*

Try chapter 06's actual discipline instead — sampling a long window
with stratification and recency weighting — and confirm it fixes the
regression:

```python
# Chapter 06's fix: retrain window spans long enough to include
# pre-drift *and* post-drift; stratify by cohort; use sample_weight for
# recency bias if you want to favour the new segment without unlearning
# the old one. Here we use the training set (segment A, from pre-drift)
# combined with the drifted recent window — chapter 06's
# "long enough to include pre-drift and post-drift" rule.
long_window = pd.concat([train, retrain_df], ignore_index=True)
model_v3 = LogisticRegression().fit(
    long_window[["tenure", "activity"]], long_window["y"],
)
print()
print("with chapter-06 long-window + pre-drift fix:")
print({
    "acc_segment_A": round(acc(model_v3, eval_A), 3),
    "acc_segment_B": round(acc(model_v3, eval_B), 3),
})
```

**Expected output.** Segment-A accuracy returns to the `v1` level (the
pre-drift training data is back in the fit) while segment-B accuracy
stays roughly where `v2` had it. Chapter 06's "Sampling for the retrain"
rule, in exactly two lines of code.

**Checkpoint.** You have reproduced both the trap (short window, drifted
data, aggregate-only check → silent slice regression) and the fix (long
window spanning pre- and post-drift → both slices protected). Chapter
06's evaluation-gate rule — "the baseline must predate the drifted
window" — is what catches the `v2` regression before promotion; this
lab is the slice-accuracy row that would produce the block.

### Step 6 — Write down what the signals meant and what the right action was (≈ 3 min)

Chapter 04's and chapter 06's main lesson from this scenario fits in a
short `NOTES.md`. Write it in the lab directory so the discipline is
one grep away for the next reader.

```markdown
# lab-02 — reading the three signals

Scenario A (chapter 04): new segment ramps up; features move;
predictions shift; model is still right.

Signals seen in the lab:
- ch 03 — per-feature PSI on `tenure` and `activity`: both rise
  monotonically with segment-B share. Fires.
- ch 04 — JSD on `P(ŷ)` histogram: rises monotonically. Fires.
- ch 04 — predicted-positive mean: rises from ~0.20 to ~0.30+. Fires.
- ch 05 — slice accuracy on segments A and B: flat vs. the
  training-set reference. Does **not** fire.

Diagnosis: data drift is real (the new region's users genuinely have
different features); concept drift is not (`P(y|X)` is unchanged);
the model is right.

Correct action (chapter 06):
1. Trigger emits `INVESTIGATE`, not `RETRAIN` (chapter 06 pitfall
   #1 — drift alone is not a retrain signal).
2. The investigation finds the new segment is legitimate traffic,
   not a bug. Nothing to mitigate at serving time.
3. The retrain, when it happens, uses a long window spanning
   pre- and post-drift with cohort stratification — not the last
   7 days (chapter 06 "trap door" sampling).
4. The evaluation gate compares the candidate against a stable
   pre-drift holdout (segment A at training time) and runs slice
   checks on segments A and B — a candidate that regresses segment
   A below threshold is blocked (chapter 06 pitfall #4).

What the wrong action would have cost:
- The step-5 `v2` retrain improved segment B marginally and
  regressed segment A noticeably. Aggregate accuracy barely moved
  because segment A still dominates traffic — the "silent
  retraining regression" (chapter 06). Over several retraining
  cycles, segment A's accuracy would decay below acceptable while
  the dashboard stayed green.
```

**Checkpoint.** The one-page summary names the chapter-specific pitfalls
(ch 04 "Prediction shift is a symptom, not a cause"; ch 06 pitfall #1
"Retraining on drift alone"; ch 06 pitfall #4 "No slice metrics in the
gate"; ch 06 "Sampling for the retrain — the trap door") and the
specific correct action. Chapter 07 would use this `NOTES.md` as the
runbook entry for "drift fires but accuracy is flat."

## Expected final artefact

A lab directory containing:

- `shift_lab.py` — the one-file simulation: two segments, a reference
  model, a 14-day ramp, PSI + JSD time-series, slice-accuracy join, the
  `v1 / v2 / v3` retrain comparison.
- `shift_signals.png` — the three-panel time-series of per-feature
  PSI, prediction-histogram JSD, and live prediction mean / p95 over
  14 days.
- `NOTES.md` — the "reading the three signals" summary mapping each
  signal to its chapter and the correct action.

Running `python shift_lab.py` prints the day-by-day drift table, the
slice-accuracy rows for day 13, the `v1 / v2` regression table, the `v3`
corrected row, and writes `shift_signals.png`.

## Final checkpoint

Without re-reading chapter 04, answer:

1. In step 3 both per-feature PSI and JSD on `P(ŷ)` rose
   monotonically. In step 4 slice accuracy did not move at all. State
   in one sentence why chapter 04 calls the step-3 signals a
   "**symptom, not a cause**", and name one more chapter-04 signal
   (not used in the lab) that would also be elevated in Scenario A.
2. In step 5 the `v2` retrain **improved segment B** and **regressed
   segment A**, while aggregate accuracy barely moved. Which
   chapter-06 named failure mode is this ("silent …", "self-
   reinforcing …", or "we retrained but nothing …"), and which
   chapter-06 evaluation-gate check (there are five) would have
   blocked `v2` from promotion?
3. In step 5 the `v3` fit used `train + retrain_df` rather than
   `retrain_df` alone. Which two chapter-06 rules about retrain
   sampling does that concatenation satisfy, and what would you add
   on top of it if segment B's share of traffic kept growing?
4. For a model whose prediction-distribution shift is firing, no
   feature drift has fired, and **no accuracy alert has yet fired
   because labels are still maturing**, write the chapter-06
   decision-rule action in one word (`DO_NOTHING` / `INVESTIGATE` /
   `RETRAIN` / `ROLLBACK`) and explain in one sentence why chapter
   04 argues for that action rather than the next one up.
5. What chapter-05 piece of infrastructure (named in one word) was
   the thing that made step 4 possible, and why is this lab's
   Scenario A diagnosis fundamentally impossible without it?

If you can answer all five, chapter 04's "prediction shift is a
symptom" and chapter 06's "retraining on drift alone is a pitfall"
are internalised and you are ready to apply them in exercise-03's
`decide_action` rule and exercise-04's post-mortem walkthrough.

## Where to go next

- **`lab-03` ("Debounce, hysteresis, cooldown on a noisy drift
  signal")** takes the single-window JSD-on-`P(ŷ)` score you just
  computed and feeds it through a 60-day noisy time series, showing
  how chapter 06's three anti-noise patterns change which of the
  day-3-14 single spikes fire a trigger.
- **`exercise-02` ("Data drift detection")** turns the ad-hoc PSI /
  JSD calls here into a drift job with CLI, calibrated thresholds,
  and three synthetic-incident test cases — Scenario A (this lab's
  shape) is one of them.
- **`exercise-03` ("Retraining trigger design")** writes the actual
  `decide_action` pure function that would have emitted
  `INVESTIGATE` on day 13 of this lab rather than `RETRAIN`. The
  simulation in Part D plants three incidents, one of which is this
  scenario.
- **`exercise-04` ("Production regression post-mortem")** uses the
  "retrain that made things worse" or "new segment of users"
  patterns from chapter 07's gallery as the scenario — the step-5
  `v2` retrain would be the post-mortem subject if it had shipped.
