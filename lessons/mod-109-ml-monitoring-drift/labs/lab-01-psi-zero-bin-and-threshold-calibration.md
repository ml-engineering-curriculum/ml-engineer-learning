# lab-01 — PSI on synthetic drift: the zero-bin bomb and a backfill-calibrated threshold

> **Duration:** ~35–45 minutes.
> **Companion chapter:** [03 — Data drift detection](../03-data-drift-detection.md).

## Objective

Reproduce chapter 03's two load-bearing claims about PSI — (a) "add a small
epsilon to zero bins or `ln(0)` blows up" and (b) "the `0.1 / 0.2` thresholds
are a starting point, not a law; calibrate per feature from a historical
backfill" — on a 60-line synthetic dataset you can run on a laptop. By the
end you will have (a) implemented PSI naïvely and watched it return `inf`
/ `nan` on a new category, (b) added the `1e-6` clip the reference
implementation uses and watched the same call return a finite score, (c)
watched PSI respond to a mean shift, a variance shift, and a new-category
injection, (d) backfilled a sliding current window against a frozen
reference on **two features with different volatility profiles** (a stable
one and a naturally volatile one) and seen that the same `0.1` cutoff
over-alerts on one and under-alerts on the other, and (e) picked a
per-feature threshold from the backfill distribution using the chapter's
`max(1.5 × baseline_p99, absolute_floor)` heuristic.

This lab is deliberately narrower than **exercise-02** (which asks you to
implement PSI, KS, chi-squared, JSD, Wasserstein, a domain classifier, a
scheduled drift job with CLI, a calibration script, and a three-shape
synthetic-incident test suite). The lab is the isolation chamber where you
*see* PSI's one-line failure mode and the per-feature threshold argument
once, on a dataset you can eyeball; the exercise then wires the primitive
into a reusable drift pipeline.

## Prerequisites

- Python 3.11+.
- `numpy`, `pandas`, `matplotlib` installed in a virtual environment
  (`uv venv && uv pip install numpy pandas matplotlib` is fine).
- Chapter 03 read through once; the frame — reference, current, distance,
  threshold — is the four pieces this lab walks.
- A CPU is enough; no GPU, no Docker, no warehouse, no real model.

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-01-psi && cd lab-01-psi
touch psi_lab.py
```

Everything below goes in `psi_lab.py`. Run with `python psi_lab.py` after
each step and read the output before moving on.

Start the file with the imports and a fixed seed:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=0)
```

## Steps

### Step 1 — Write PSI the naïve way and watch it blow up (≈ 5 min)

Chapter 03's PSI formula is `PSI = Σ (q_i − p_i) · ln(q_i / p_i)` with `p_i`
the reference proportions and `q_i` the current proportions across a fixed
set of bins. Write it exactly as the formula reads first, with no guards,
and see what happens.

Append to `psi_lab.py`:

```python
def psi_naive(reference: np.ndarray, current: np.ndarray,
              bins: np.ndarray) -> float:
    """PSI, straight from the formula. No epsilon."""
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    p = ref_hist / ref_hist.sum()
    q = cur_hist / cur_hist.sum()
    return float(np.sum((q - p) * np.log(q / p)))


# Reference: a tight Gaussian around 0.5 on [0, 1], say a feature that is
# always roughly in the middle of its allowed range.
reference = np.clip(RNG.normal(loc=0.5, scale=0.1, size=10_000), 0.0, 1.0)

# Current window: same distribution, but a handful of samples pushed out
# into a bin that was empty in the reference.
current = np.concatenate([
    np.clip(RNG.normal(loc=0.5, scale=0.1, size=9_800), 0.0, 1.0),
    np.full(200, 0.98),   # a brand-new mass in the top bin
])

bins = np.linspace(0.0, 1.0, 11)   # 10 equal-width bins

print("naïve PSI (no epsilon):", psi_naive(reference, current, bins))
```

Run `python psi_lab.py`.

**Expected output.** The last line prints something like `naïve PSI (no
epsilon): inf` or — depending on which bin is zero in which direction —
`nan`, plus a `RuntimeWarning: divide by zero encountered in log` from
NumPy. Chapter 03's "add a small epsilon to zero bins" is not optional;
the formula is literally undefined when a bin has zero mass in one
distribution and non-zero in the other.

**Checkpoint.** You have just reproduced chapter 03 pitfall #5 ("Zero-bin
`ln(0)`. Add epsilon. Every serious PSI implementation does this.") in one
function call. The top bin is empty in the reference (`ref_hist[-1] = 0`)
and non-empty in the current (`cur_hist[-1] = 200`) — `np.log(0.02 / 0)`
is `+inf`, and `(0.02 − 0) × +inf` is `+inf`. On a real monitoring job
this is what pages every oncall the moment a new category or a new tail
value appears — before you get to look at *which* feature moved.

### Step 2 — Add the epsilon clip from the chapter and watch the same call return a finite score (≈ 3 min)

Chapter 03's reference implementation uses `np.clip(p, 1e-6, None)` on both
proportion arrays before the log. Add that exact guard:

```python
def psi(reference: np.ndarray, current: np.ndarray,
        bins: np.ndarray, eps: float = 1e-6) -> float:
    """PSI with the chapter 03 epsilon clip on zero bins."""
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    p = ref_hist / max(ref_hist.sum(), 1)
    q = cur_hist / max(cur_hist.sum(), 1)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


print("epsilon PSI:         ", round(psi(reference, current, bins), 4))
```

**Expected output.** A finite number in the neighbourhood of `0.10–0.20`
(the exact value depends on your platform's `make_classification`-free
sampler, but it is bounded, positive, and interpretable). The 200
rows in the new top bin shift the proportion in that bin from ~0 to
0.02; the clip replaces the reference's `0` with `1e-6`; `(0.02 − 1e-6)
× ln(0.02 / 1e-6)` is ~0.2 by itself, which is most of the final score.

**Checkpoint.** Same inputs, same formula, one extra line → finite,
alertable, calibratable number. That one line (plus the companion rule
from chapter 03 — "Treat 'new category' as a separate signal" via
chapter 02's `unknown_category` counter rather than folding it into PSI)
is the boundary between a drift pipeline that pages on every new tail
value and one that doesn't.

### Step 3 — Watch PSI respond to three canonical drift shapes (≈ 7 min)

Chapter 03's shortlist: PSI handles continuous and categorical features
with the same formula; the score grows with dissimilarity on the shapes
the field cares about. Verify that on three canonical shifts: a mean
shift, a variance shift, and a new-category injection.

Append:

```python
def gauss(loc, scale, n=10_000):
    return np.clip(RNG.normal(loc=loc, scale=scale, size=n), 0.0, 1.0)

bins_numeric = np.linspace(0.0, 1.0, 11)

ref_num = gauss(0.5, 0.1)

shifts = {
    "identical":            gauss(0.5, 0.1),
    "mean shift +0.05":     gauss(0.55, 0.1),
    "mean shift +0.10":     gauss(0.60, 0.1),
    "mean shift +0.15":     gauss(0.65, 0.1),
    "variance x 2":         gauss(0.5, 0.2),
    "variance x 4":         gauss(0.5, 0.4),
}

rows = [
    {"shape": name, "psi": round(psi(ref_num, cur, bins_numeric), 4)}
    for name, cur in shifts.items()
]
print(pd.DataFrame(rows).to_string(index=False))
```

**Expected output.** Something like:

```
                 shape    psi
             identical  0.0000
      mean shift +0.05  0.0?
      mean shift +0.10  0.1?
      mean shift +0.15  0.3?
         variance x 2   0.2?
         variance x 4   0.8?
```

The exact digits depend on your seed and your platform's RNG; the **shape**
is what to watch. Two properties fall out that chapter 03 relies on:

- `identical` ≈ 0 (reflexivity — `d(P, P) ≈ 0`; exercise-02 Part A has a
  test for exactly this).
- The score grows **monotonically** with the size of the shift (mean
  `+0.05` → `+0.10` → `+0.15` produces a rising PSI column). Chapter 03
  cites this as the property that makes PSI alertable.

Now the categorical case. Chapter 03's "Chi-squared … and PSI" section
notes that for categoricals the bins are the categories themselves; new
categories are handled by chapter 02's `unknown_category` counter, **not**
by the PSI calculation. Simulate it:

```python
def psi_categorical(ref_counts: dict, cur_counts: dict,
                    eps: float = 1e-6) -> float:
    """PSI over a fixed category set. Does not score new categories."""
    categories = sorted(ref_counts)
    ref_total = sum(ref_counts.values())
    cur_total = sum(cur_counts.get(c, 0) for c in categories)
    p = np.array([ref_counts[c] / ref_total for c in categories])
    q = np.array([cur_counts.get(c, 0) / max(cur_total, 1)
                  for c in categories])
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


# Reference categorical split: 62% basic, 31% pro, 7% enterprise.
# (Same shape as chapter 02's reference_profile.json example.)
ref_cat = {"basic": 6_200, "pro": 3_100, "enterprise": 700}

# Three current-window shapes:
same   = {"basic": 6_200, "pro": 3_100, "enterprise": 700}
grown  = {"basic": 5_200, "pro": 3_500, "enterprise": 1_300}   # enterprise ramp
new    = {"basic": 5_800, "pro": 2_900, "enterprise": 500,
          "enterprise-plus": 800}                              # schema change

for name, cur in [("same", same), ("grown enterprise", grown),
                  ("new category appears", new)]:
    print(f"psi_categorical({name}): "
          f"{psi_categorical(ref_cat, cur):.4f}")
```

**Expected output.**

```
psi_categorical(same):                0.0000
psi_categorical(grown enterprise):    0.0?
psi_categorical(new category appears): 0.0?...
```

The **"new category appears"** case is the one chapter 03 flags twice:
the `enterprise-plus` mass (800 rows) is **not scored** by this function
— the categories iterator is drawn from `ref_counts` only. A naïve
implementation that iterated `cur_counts` instead would either emit
`ln(q / 0)` → `inf` or silently absorb the new mass into the existing
categories. Chapter 03's explicit discipline: "Treat 'new category' as a
separate signal (chapter 02's `unknown_category` counter) rather than
folding it into the same test." Chapter 02's feature-anomaly counter is
the alert for the schema-change case; PSI is the alert for shifts within
the known category set.

**Checkpoint.** You have confirmed three claims from chapter 03's
univariate shortlist on a dataset tiny enough to eyeball: PSI returns
`0` on identical distributions, grows monotonically with mean shift,
grows with variance shift, and is structurally unable to score a
previously-unseen category. The last one is what makes the chapter-02
feature-anomaly counter a required-not-nice-to-have piece of the
surface.

### Step 4 — Backfill PSI on two features with different volatility (≈ 10 min)

Chapter 03's "Thresholds — calibrate, don't guess" states: *"A feature
that is naturally volatile in the training set … will bounce over `0.1`
every day without anything being wrong."* Reproduce that on two features
side-by-side.

Append:

```python
# Pretend we have 90 days of production traffic. Each day, draw a current
# window of 2,000 rows for two synthetic features, score PSI against the
# frozen reference, and record the result. No injected drift — every
# number you see is the "normal wobble" of a stable system.

N_DAYS = 90
WINDOW_N = 2_000

# Feature A: stable — tight Gaussian, low day-to-day variance.
#   Reference: 10,000 rows ~ N(0.50, 0.10)
#   Each daily window: 2,000 rows ~ N(0.50, 0.10) + tiny per-day jitter
ref_A = gauss(0.50, 0.10)

# Feature B: naturally volatile — bimodal with a day-to-day mode weight
# that wobbles by ±10%. Reference is 50/50; daily windows wobble around it.
def ref_B_sample(n):
    """Bimodal: half near 0.20, half near 0.80."""
    a = np.clip(RNG.normal(0.20, 0.05, n // 2), 0.0, 1.0)
    b = np.clip(RNG.normal(0.80, 0.05, n - n // 2), 0.0, 1.0)
    return np.concatenate([a, b])

ref_B = ref_B_sample(10_000)

daily_A, daily_B = [], []
for day in range(N_DAYS):
    # Feature A: same N(0.50, 0.10), with a per-day jitter in the mean of
    # ±0.005 (very stable in practice).
    jitter = float(RNG.uniform(-0.005, 0.005))
    cur_A = gauss(0.50 + jitter, 0.10, WINDOW_N)
    daily_A.append(psi(ref_A, cur_A, bins_numeric))

    # Feature B: 50/50 ± a per-day mode-weight wobble.
    mode_weight = float(np.clip(RNG.normal(0.50, 0.05), 0.1, 0.9))
    n_left = int(WINDOW_N * mode_weight)
    cur_B = np.concatenate([
        np.clip(RNG.normal(0.20, 0.05, n_left), 0.0, 1.0),
        np.clip(RNG.normal(0.80, 0.05, WINDOW_N - n_left), 0.0, 1.0),
    ])
    daily_B.append(psi(ref_B, cur_B, bins_numeric))

daily_A = np.array(daily_A)
daily_B = np.array(daily_B)

print(f"feature A (stable):   p50={np.percentile(daily_A, 50):.4f}  "
      f"p99={np.percentile(daily_A, 99):.4f}  max={daily_A.max():.4f}")
print(f"feature B (volatile): p50={np.percentile(daily_B, 50):.4f}  "
      f"p99={np.percentile(daily_B, 99):.4f}  max={daily_B.max():.4f}")

print("\nalerts under the folk `PSI < 0.1` cutoff:")
print(f"  feature A: {(daily_A >= 0.10).sum()} / {N_DAYS} days")
print(f"  feature B: {(daily_B >= 0.10).sum()} / {N_DAYS} days")
```

**Expected output.** Something like:

```
feature A (stable):   p50=0.001?  p99=0.00?   max=0.00?
feature B (volatile): p50=0.01?   p99=0.0?    max=0.1?

alerts under the folk `PSI < 0.1` cutoff:
  feature A: 0 / 90 days
  feature B: 5–20 / 90 days
```

(Exact counts move with the seed; the **direction** is the point.) Under
the folk `0.1 / 0.2` rule, **feature A under-alerts** (its natural wobble
is so tiny that `0.1` would miss any real drift that didn't completely
reshape the distribution) and **feature B over-alerts** (its mode-weight
wobble routinely drives PSI toward the cutoff even though *nothing is
wrong*). This is chapter 03's "A feature that is naturally volatile …
will bounce over `0.1` every day without anything being wrong" claim,
played out on the same two lines of code that an exercise-02 calibration
script would see on real production traffic.

Draw the backfill distributions:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for ax, daily, name in [(ax1, daily_A, "feature A (stable)"),
                        (ax2, daily_B, "feature B (volatile)")]:
    ax.hist(daily, bins=20, alpha=0.7)
    ax.axvline(0.10, color="red", linestyle="--",
               label="folk 0.1 threshold")
    p99 = float(np.percentile(daily, 99))
    ax.axvline(max(1.5 * p99, 0.05), color="green", linestyle="--",
               label="calibrated: max(1.5*p99, 0.05)")
    ax.set_title(name)
    ax.set_xlabel("daily PSI")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("psi_backfill.png", dpi=120)
print("wrote psi_backfill.png")
```

Open `psi_backfill.png`. For feature A the folk `0.1` line is way out in
the right tail — alerts would almost never fire; for feature B the folk
`0.1` line cuts through the body of the distribution — alerts would fire
constantly on normal traffic. The **calibrated** line
(`max(1.5 × p99, 0.05)`) sits just above the right tail of each feature's
normal wobble.

**Checkpoint.** The one-rule-fits-all `PSI < 0.1` cutoff is doing a
different job for each feature, and it is the wrong job for both. The
calibrated rule puts each feature's threshold where chapter 03 says it
belongs: above the 99th percentile of normal.

### Step 5 — Pick the per-feature threshold and sanity-check it against an injected incident (≈ 7 min)

Chapter 03's calibration output is a per-feature threshold value that a
drift job reads at runtime. Compute both from the backfill, pretend they
are now in your `thresholds.yaml`, and then inject an obvious incident to
verify each threshold still fires.

Append:

```python
ABSOLUTE_FLOOR = 0.05

def calibrate(backfill_scores: np.ndarray) -> float:
    p99 = float(np.percentile(backfill_scores, 99))
    return float(max(1.5 * p99, ABSOLUTE_FLOOR))


thresholds = {
    "feature_A": calibrate(daily_A),
    "feature_B": calibrate(daily_B),
}
print("calibrated thresholds:", thresholds)

# Inject an incident: feature A's mean shifts from 0.50 to 0.65 (large and
# clearly out of distribution for a stable feature); feature B's mode
# weight shifts from 50/50 to 80/20 (large even for a volatile feature).
incident_A = gauss(0.65, 0.10, WINDOW_N)
incident_B = np.concatenate([
    np.clip(RNG.normal(0.20, 0.05, int(WINDOW_N * 0.80)), 0.0, 1.0),
    np.clip(RNG.normal(0.80, 0.05, WINDOW_N - int(WINDOW_N * 0.80)), 0.0, 1.0),
])

score_A = psi(ref_A, incident_A, bins_numeric)
score_B = psi(ref_B, incident_B, bins_numeric)

print(f"incident feature A: PSI={score_A:.4f}  "
      f"threshold={thresholds['feature_A']:.4f}  "
      f"breached={score_A > thresholds['feature_A']}")
print(f"incident feature B: PSI={score_B:.4f}  "
      f"threshold={thresholds['feature_B']:.4f}  "
      f"breached={score_B > thresholds['feature_B']}")
```

**Expected output.** Both incidents breach their respective thresholds
(`breached=True`), but the thresholds themselves are different values —
feature A's sits near `0.05` (the absolute floor — a stable feature with
`p99 ≈ 0.003` would otherwise have a threshold of `0.0045` that would
trip on a rounding error), feature B's sits well above `0.05` to cover
its natural mode-weight wobble.

**Checkpoint.** You now have the two artifacts chapter 03 argues a drift
job needs: a **per-feature threshold** derived from the feature's own
back-tested stable-period distribution, and an **absolute floor** below
which the threshold is not allowed to fall (otherwise a very stable
feature's threshold is `0.0045` and it pages on a rounding error).
Chapter 06's debounce / hysteresis / cooldown machinery (`decide_action`
rule) sits downstream of both.

### Step 6 — Write down the one-line "what the chapter-03 backfill procedure produced" (≈ 3 min)

Chapter 03's four-step calibration procedure reduces to a short
operational sentence. Write it in a `NOTES.md` in the lab directory so a
reviewer of your future exercise-02 calibration script can see where the
discipline came from.

```markdown
# lab-01 — calibration notes

Backfill procedure (chapter 03):
1. Freeze the reference profile at training time; store the bin edges
   with it.
2. For every feature, run `psi(reference, current)` on each daily
   window over the last 90 days of production traffic. The reference
   does not move; only the current window slides.
3. Look at the distribution of each feature's daily PSI. Pick the
   threshold as `max(1.5 * p99, absolute_floor)`.
4. Re-calibrate when the model is re-versioned (new reference) and
   re-visit periodically as the world moves.

Why the folk `PSI < 0.1` fails:
- Stable features: `p99` can be ~0.003; `0.1` is 30× the real signal.
- Volatile features: `p99` can exceed `0.1`; alerts fire every day on
  normal traffic (the "every feature is red every day" dashboard
  nobody looks at).

Add-ons (same chapter):
- Add `1e-6` to zero proportions before `np.log` (otherwise `inf` /
  `nan`).
- Bin edges come from the reference; the same edges apply to both
  reference and current (different edges return silent nonsense).
- Treat "new category" as a separate signal (ch 02's
  `unknown_category` counter), not as a PSI event — PSI is structurally
  unable to score a bin that was empty in the reference.
- The reference is training-time distribution; **do not** slide it.
```

**Checkpoint.** You have derived the entirety of the chapter-03
calibration argument from one `.py` file and `psi_backfill.png`. Every
line in `NOTES.md` traces back to a specific number you measured in the
lab, not a cargo-cult default from a blog post.

## Expected final artefact

A lab directory containing:

- `psi_lab.py` — the single file that implements naïve PSI, epsilon
  PSI, categorical PSI (with the "do not score unseen categories"
  discipline), a 90-day backfill on two feature shapes, a calibration
  function, and a two-incident sanity check.
- `psi_backfill.png` — the two-panel histogram of daily PSI for the
  stable and volatile features, with the folk `0.1` cutoff and the
  calibrated threshold overlaid.
- `NOTES.md` — the short operational summary of the calibration
  procedure and the pitfalls it covers.

Running `python psi_lab.py` prints seven labelled lines (naïve PSI
blowup, finite PSI after epsilon, shift-shape table, categorical PSI
table, two backfill p50/p99/max summaries, folk-cutoff alert counts,
calibrated thresholds, two incident verifications) and writes
`psi_backfill.png`.

## Final checkpoint

Without re-reading chapter 03, answer:

1. In step 1 the naïve PSI returned `inf`. Which specific term in the
   `Σ (q_i − p_i) · ln(q_i / p_i)` sum produced that value, and which
   two-line fix in step 2 made it finite? Why is the fix chapter 03
   pitfall #5?
2. In step 3 you watched `psi_categorical` *silently ignore* the
   `enterprise-plus` mass when a new category appeared in the current
   window. Which chapter-02 metric is the correct place for that
   signal to fire instead, and why does chapter 03 argue new
   categories should **not** be folded into the same PSI event?
3. In step 4 the folk `0.1` cutoff under-alerted on feature A and
   over-alerted on feature B. State in your own words why one cutoff
   cannot be right for both features, and name the two numbers the
   chapter-03 calibration procedure needs to compute to replace it.
4. In step 4 you ran the backfill against a **frozen** reference, not
   the "last 30 days of production." Which specific chapter-03
   pitfall would a sliding reference walk into, and what does its
   drift score look like as a model becomes unusable?
5. For a feature whose backfill p99 is `0.003`, what threshold does
   the chapter's `max(1.5 × baseline_p99, absolute_floor)` heuristic
   recommend, and what would go wrong if the absolute floor were not
   in the formula?

If you can answer all five, chapter 03's PSI-plus-calibration argument
is internalised and you are ready to apply it in exercise-02's
`drift/distances.py`, `drift/calibrate.py`, and the three-shape
synthetic-incident test suite.

## Where to go next

- **`exercise-02` ("Data drift detection")** turns the single-feature
  PSI from this lab into a full `drift/distances.py` (with KS,
  chi-squared, JSD, Wasserstein), a `drift/multivariate.py` (with a
  domain classifier and PCA reconstruction), a scheduled job with CLI,
  and a `calibrate.py` that applies the step-5 procedure to every
  feature in the reference profile at once.
- **`lab-02` ("Prediction-distribution shift from a new-segment
  ramp")** moves from "a single feature drifted" to "every feature
  looks fine but `P(ŷ)` has moved" — chapter 04's Scenario A, and the
  trap that makes retraining on prediction shift alone (chapter 06
  pitfall #1) a regression.
- **`lab-03` ("Debounce, hysteresis, cooldown on a noisy drift
  signal")** takes the single-window drift score you just calibrated
  and feeds a 60-day noisy time series through chapter 06's three
  anti-noise patterns, showing how the retrain count changes with and
  without each.
- **`exercise-01` ("ML metrics instrumentation")** writes the
  `reference_profile.json` this lab assumed was already there, so
  that the backfill in step 4 reads from a real artifact instead of
  a hand-rolled `ref_A` / `ref_B`.
