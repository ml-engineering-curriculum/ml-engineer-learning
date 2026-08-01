# 03 — Data drift detection

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01, 02.

## Motivation

Chapter 02 gave you the reference profile and the live
feature histograms. This chapter turns them into a number
you can alert on: **how far has the live distribution
moved from the training distribution, per feature and in
aggregate?**

The single most useful property of data drift monitoring
is that it works **without any labels**. Every other
model-quality signal in this module either waits for
ground truth (chapter 05) or is an indirect proxy
(chapter 04). Data drift is the leading indicator you get
for free from the input stream, and it explains a large
fraction of production regressions before they show up in
accuracy.

The failure mode this chapter is preventing is not "no
signal" — it is **a noisy signal that everyone learns to
ignore**. A drift dashboard where every feature is red
every day is worse than none at all. Calibration is at
least as important as the choice of test.

## The frame — reference, current, distance, threshold

Every drift measurement has the same four pieces:

1. **Reference distribution** `P_ref` — the training-time
   distribution of a feature (or of predictions, or of
   the joint distribution of all features). Chapter 02's
   reference profile.
2. **Current distribution** `P_cur` — the same feature
   over a recent window: last hour, last day, last week.
   Window size is a knob (see below).
3. **A distance** `d(P_ref, P_cur)` — a scalar that grows
   with dissimilarity. Different tests suit different
   data types.
4. **A threshold** `τ` — the value of `d` above which the
   signal counts as "drifted enough to act on." Set
   empirically; not from theory.

The rest of the chapter is picking the right `d` for each
data type, sizing the window, and calibrating `τ`.

## Univariate tests — what to use when

Univariate means one feature at a time. Cheap, easy to
attribute ("`income` drifted"), and covers most real
drift because most real drift is upstream on a single
column.

### Population Stability Index (PSI) — the workhorse

PSI is the metric the credit-scoring industry has used
for decades to answer "has this variable's distribution
shifted between two populations?". It is:

- Symmetric-ish in practice (people report it either way;
  either is defensible),
- Interpretable on a fixed scale that generalises
  across features,
- Bin-based, so it handles continuous and categorical
  features with the same formula.

The formula: for a set of `k` bins with reference
proportions `p_i` and current proportions `q_i`,

```
PSI = Σ_i (q_i − p_i) · ln(q_i / p_i)
```

Rules of thumb widely cited in the industry:

- `PSI < 0.1` — no meaningful shift.
- `0.1 ≤ PSI < 0.2` — moderate shift, investigate.
- `PSI ≥ 0.2` — significant shift, act.

<!-- needs-research: the 0.1 / 0.2 thresholds are folk wisdom from credit scoring rather than a peer-reviewed cutoff — cite a specific primary source (Karakoulas or the SAS whitepapers if you can trace them) or drop the specific numbers in favour of "calibrate empirically." -->

Two implementation gotchas:

- **Bin edges come from the reference distribution.**
  Same edges applied to both reference and current.
  Compute them once at reference-profile time.
- **Add a small epsilon to zero bins.** `ln(0)` is
  undefined; `ln(q/0)` is undefined. Standard practice
  is to replace any zero proportion with `1e-6` before
  computing the log.

```python
import numpy as np

def psi(reference: np.ndarray, current: np.ndarray,
        bins: np.ndarray) -> float:
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    p = ref_hist / max(ref_hist.sum(), 1)
    q = cur_hist / max(cur_hist.sum(), 1)
    p = np.clip(p, 1e-6, None)
    q = np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))
```

For a categorical feature the bins are the categories
themselves.

### Kolmogorov–Smirnov (KS) for continuous features

The two-sample KS test compares two empirical CDFs and
reports the maximum vertical distance between them plus a
p-value.

```python
from scipy.stats import ks_2samp

result = ks_2samp(reference_values, current_values)
# result.statistic  ∈ [0, 1]  — the D statistic
# result.pvalue     — probability of D under H0 of equal distributions
```

Use the **statistic**, not the p-value, as the drift
signal. With production traffic volumes, any real
difference produces a vanishingly small p-value; alerting
on p-values guarantees noise.

Two constraints:

- **Continuous features only.** KS assumes a continuous
  distribution; on discrete features it is anti-
  conservative.
- **Sensitive at the tails.** The D statistic can be
  dominated by mass in the middle. If you care about
  tail drift specifically, prefer Wasserstein or a
  quantile-based check.

### Chi-squared for categorical features

For categoricals, a two-sample chi-squared test on the
category counts (`scipy.stats.chi2_contingency`) is the
classical choice. Same warning as KS: the p-value goes
to zero with volume; use the statistic (or, better, use
PSI, which reports on a scale that is comparable across
features).

**A subtlety:** if a *new* category appears in the current
window that was not in the reference, chi-squared cannot
score it — the reference count is zero. Treat "new
category" as a separate signal (chapter 02's
`unknown_category` counter) rather than folding it into
the same test.

### Wasserstein / Earth Mover's Distance

For continuous features where you specifically care about
*how far* mass moved (not just that it moved), the
Wasserstein-1 distance is:

```python
from scipy.stats import wasserstein_distance

d = wasserstein_distance(reference_values, current_values)
```

`d` is in the units of the feature. That is a feature and
a bug — interpretable per feature, but not comparable
across features. Wasserstein is the right choice when a
"1% shift in the median" of a specific feature has a
domain meaning ("median income moved by $2k").

### Jensen–Shannon divergence

For probability distributions (a natural fit for the
prediction distribution), JSD is a bounded, symmetric
version of KL divergence in `[0, ln 2]` (or `[0, 1]` with
the base-2 log). Bounded is nice for setting thresholds.

```python
from scipy.spatial.distance import jensenshannon

# jensenshannon returns the *square root* of the divergence
d = jensenshannon(p_ref, p_cur, base=2) ** 2  # actual JSD
```

Widely used for prediction drift where the histogram is
already a probability mass function.

### Which test for which data type — a shortlist

| Data type | Recommended | Also fine |
|-----------|-------------|-----------|
| Continuous, general purpose | KS statistic *or* PSI | Wasserstein |
| Continuous, tail-sensitive | Wasserstein | Quantile deltas |
| Categorical | PSI | Chi-squared statistic |
| Probability distribution | JSD | PSI |
| Prediction score histogram | PSI or JSD | KS |

You do not have to pick one across the whole system.
Different features can use different tests; the drift
score is comparable across features only if the test is
the same.

## Multivariate drift — when univariate is not enough

Univariate tests miss the case where every marginal
distribution looks identical to the reference but the
*joint* distribution has moved. Two customers used to be
correlated; now they are not. Both features are still
`N(0, 1)` on the marginal.

Multivariate drift detectors report a single scalar for
the joint distribution. Three common approaches:

### Domain classifier (a.k.a. "Classifier Two-Sample
Test")

Train a small classifier to distinguish reference samples
from current samples. If it succeeds, the two are
distinguishable (drift). If it fails (AUC ≈ 0.5), they
are not (no drift).

```python
# Pseudocode
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score

X = np.vstack([reference_sample, current_sample])
y = np.concatenate([np.zeros(len(reference_sample)),
                    np.ones(len(current_sample))])
auc = cross_val_score(lgb.LGBMClassifier(), X, y,
                      scoring="roc_auc", cv=3).mean()
drift_score = 2 * abs(auc - 0.5)  # ∈ [0, 1]
```

Cheap, model-agnostic, works on mixed dtypes, and — the
best part — the classifier's feature importances tell
you *which* features carried the drift.

<!-- needs-research: attach the Lopez-Paz & Oquab (2017) "Revisiting Classifier Two-Sample Tests" citation and the arXiv URL to strengthen this attribution. -->

### Maximum Mean Discrepancy (MMD)

A kernel-based test that measures the distance between
two distributions in a reproducing kernel Hilbert space.
Sensitive to differences in mean, variance, and higher
moments; useful for high-dimensional or embedding data.

The `alibi-detect` library ships an implementation
(`MMDDrift`); rolling your own is possible but easy to
get wrong.

### Principal Component Analysis (PCA) reconstruction
error

Project the current window onto the reference PCA basis;
measure the reconstruction error. Rising error indicates
the current data lies in a subspace the training data did
not span.

Simple and interpretable; less sensitive than the domain
classifier or MMD, but a useful sanity check that
requires no extra library.

### Which multivariate detector — a shortlist

- **Domain classifier** — first choice. Cheap,
  interpretable via feature importances.
- **MMD** — when the features are embeddings or the
  data is high-dimensional and the domain classifier
  overfits.
- **PCA reconstruction error** — a cheap complementary
  signal; runs in one pass.

Do not run all three and average. Pick one and calibrate
it; the second and third become confirmatory checks in an
incident.

## Windows — what "current" means

The `current` window is a knob with real consequences.
Two rules pin down the choice:

- **The window should be large enough that random noise
  does not dominate.** Very small windows (100 requests)
  make every test twitchy; PSI numbers bounce wildly.
- **The window should be short enough to detect the
  drift you care about.** A 30-day window smooths over a
  sudden shift; an hourly window catches it inside the
  hour.

A common compromise:

- **Fast window** — 1 hour or 10k requests, whichever
  comes first. Alerts on sudden shifts (an ETL job that
  changed shape, a new client that started sending
  malformed data).
- **Slow window** — 1 day or 7 days. Alerts on gradual
  drift (a market moving, a season changing).

Run both. They surface different failure modes and
generate roughly independent alerts.

**Do not** slide the reference. The reference is the
training distribution, immutable. The window that moves
is always the current one.

## Thresholds — calibrate, don't guess

Every drift test spits out a scalar. `PSI < 0.1` is a
useful default; **it is not a safe default for every
feature in every product.** A feature that is naturally
volatile in the training set (a category with 3% mass
plus/minus 2%) will bounce over `0.1` every day without
anything being wrong.

Calibration procedure:

1. **Backfill the current-window test on historical
   data.** Take the last 90 days of production traffic
   (or, if unavailable, the last 90 days of the training
   set's temporal splits). Slide the current window
   through it. Compute the drift score at each step
   against the reference.
2. **Look at the distribution of the score.** What is the
   99th percentile? The maximum? Feature `X` might
   normally sit at PSI ≈ 0.15 with a 99th percentile of
   0.22. Setting the threshold at 0.1 would page every
   day.
3. **Set the threshold to catch clear anomalies, not the
   99th percentile of normal.** A common heuristic is
   `max(baseline_99p * 1.5, absolute_floor)`. The
   absolute floor stops a naturally stable feature from
   having a threshold of 0.001 that trips constantly.
4. **Re-calibrate when the model version changes.** New
   version → new reference → new calibration. Six months
   later, re-calibrate again as the world moves.

Alerting on statistical significance (a p-value below
0.05) will page you every hour once traffic is large. Use
effect size (PSI, KS statistic, Wasserstein, JSD) with a
calibrated threshold, always.

## The drift job — architecture

Drift monitoring is not part of the serving path. Do not
compute PSI in the request handler. Instead:

- **A batch job** — Airflow / Dagster / Prefect / a cron
  job — that runs on a schedule (hourly or daily).
- **Reads:** the reference profile and the last window
  of the prediction table (chapter 02).
- **Computes:** per-feature drift scores + one
  multivariate score.
- **Writes:** a `drift_scores` table with one row per
  (feature, window, score, threshold, breached?).
- **Emits:** a metric per feature (a Prometheus gauge is
  fine here because feature count is bounded) and a log
  line per window.
- **Alerts:** on `breached=True` with a debounce (e.g.,
  three consecutive windows) to avoid a single spike
  paging.

A minimal shape:

```python
def run_drift_job(reference: ReferenceProfile,
                  window_df: pd.DataFrame) -> DriftReport:
    scores = {}
    for feature in reference.features:
        if feature.dtype in ("float", "int"):
            scores[feature.name] = psi(
                reference.sample[feature.name],
                window_df[feature.name],
                bins=reference.bins[feature.name],
            )
        else:  # categorical
            scores[feature.name] = psi_categorical(
                reference.categories[feature.name],
                window_df[feature.name].value_counts(normalize=True),
            )

    multivariate_score = domain_classifier_auc(
        reference.sample, window_df[reference.feature_names]
    )

    return DriftReport(
        window_id=window_df.attrs["window_id"],
        univariate=scores,
        multivariate=multivariate_score,
        thresholds=reference.thresholds,
    )
```

Chapter 06 turns `DriftReport` into a retraining trigger;
chapter 07 turns it into a debugging entry point.

## Off-the-shelf libraries

You do not have to build this from scratch. A few widely
used, actively maintained open-source options:

- **Evidently** (`evidently-ai`) — Python library that
  computes a large catalogue of univariate and
  multivariate drift tests and renders HTML/JSON
  reports. Good for teams that want a batteries-included
  starting point.
  <https://docs.evidentlyai.com/>
- **whylogs** (WhyLabs) — profile-based
  observability; you emit a profile per window, and
  drift is measured between profiles. Handles very
  large-cardinality features well.
  <https://whylogs.readthedocs.io/>
- **NannyML** — focuses on performance estimation
  without labels (CBPE, DLE) and also ships drift
  detectors. Useful once you get to chapter 04.
  <https://nannyml.readthedocs.io/>
- **Alibi Detect** (Seldon) — a broader drift-detection
  toolkit with kernel-based methods (MMD), classifier
  tests, and drift on embeddings. Well suited to
  unstructured data.
  <https://docs.seldon.io/projects/alibi-detect/>

For a first drift job, pick one and use its defaults.
Roll your own once you know exactly what you want.

## What "drift" is *not* by itself

Two disciplines this chapter deliberately does not cover
but which people frequently conflate with drift:

- **Data quality.** Nulls, out-of-range values, schema
  violations, duplicate rows. `mod-103` (data
  engineering) owns this. Tools like Great Expectations,
  dbt tests, or TFDV validate schema and value
  constraints. Chapter 02's `feature_anomaly` counter is
  the ML-side shim; the real data-quality regime lives
  upstream.
- **A change in what you're modelling** (concept drift).
  Same `P(X)`, different `P(y|X)`. Requires labels;
  chapter 04.

If your drift job also does data-quality validation, name
it that. Do not muddle the two.

## Common pitfalls

1. **Alerting on p-values.** At scale, every test is
   "significant." Use effect size.
2. **A single global drift score.** Loses attribution.
   Per-feature drift is the useful unit.
3. **Sliding the reference window.** Drift monotonically
   trends to zero; you find out the model is broken from
   an accuracy alert, weeks late.
4. **Reference and current using different bin edges.**
   The tests silently return nonsense. Compute bin edges
   once, from the reference.
5. **Zero-bin `ln(0)`.** Add epsilon. Every serious
   PSI implementation does this.
6. **Recomputing PSI on the reference every day.** The
   reference does not move. Save the histogram; reload
   it; compare.
7. **Uncalibrated thresholds copied from a blog post.**
   The 0.1 / 0.2 PSI thresholds are a starting point,
   not a law. Backfill and calibrate per feature.
8. **Ignoring `unknown_category`.** New categories are
   drift the tests are structurally bad at scoring.
   Alert on the counter directly (chapter 02).
9. **Drift on features that were never in the model.**
   Wasted signal — the model doesn't use them, so
   drift on them cannot affect predictions. Score only
   the features in the model's feature list.
10. **Drift on the prediction distribution alone.**
    Prediction drift is a downstream symptom. Pair it
    with per-feature drift so you can attribute cause.

## Summary

- Drift measurement has four pieces: reference, current
  window, distance, threshold. Get all four right and
  the rest is tuning.
- Univariate: PSI as the workhorse; KS for continuous;
  chi-squared or PSI for categorical; Wasserstein when
  units matter; JSD for probability distributions.
- Multivariate: domain classifier first (interpretable
  via importances), MMD for embeddings, PCA
  reconstruction as a cheap sanity check.
- Windows: run a fast window and a slow window in
  parallel; the reference never slides.
- Thresholds are calibrated per feature from a historical
  backfill. Not copied from a table.
- Off-the-shelf libraries (Evidently, whylogs, NannyML,
  Alibi Detect) accelerate the first pass; roll your own
  once you know the shape.

## Where this goes next

Chapter 04 picks up the label-free thread and asks the
harder question: **what do you do when `P(X)` looks
identical but the model is still wrong?** Concept drift,
prediction-distribution shift as a proxy, and the
performance-estimation-without-labels methods that let
you say *something* about accuracy while you wait for
ground truth to arrive.
