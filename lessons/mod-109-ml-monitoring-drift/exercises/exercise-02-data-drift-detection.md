# exercise-02 — Data drift detection

> **Estimated effort:** 3–4 hours.
> **Companion chapters:** [03 — Data drift detection](../03-data-drift-detection.md),
> [04 — Concept drift and prediction-distribution shift](../04-concept-and-prediction-drift.md).

## Objective

Build a scheduled drift-detection job that consumes the
reference profile and prediction table from exercise-01,
computes univariate and multivariate drift scores over
sliding windows, and calibrates alert thresholds from a
historical backfill so the resulting alert channel is
actually useful.

By the end you should have:

- A pure-Python `drift.py` module with implementations of
  PSI, KS, chi-squared, JSD, and a domain-classifier
  multivariate score — each covered by unit tests
  against synthetic distributions with known drift.
- A batch job runnable from CLI and importable as a
  library that produces a `DriftReport` per window and
  writes a `drift_scores` table.
- A calibration script that backfills scores over
  historical windows and emits per-feature thresholds.
- An `ALERTS.md` with your calibration methodology,
  final thresholds, and a synthetic-incident test.

You are **not** using an off-the-shelf library
end-to-end — the point is to see the internals. Compare
your outputs against one library (Evidently, whylogs,
NannyML, or Alibi Detect) as a sanity check.

## Prerequisites

- Exercise-01 completed (or equivalent: a reference
  profile and a stream of prediction records).
- Python 3.11+; `numpy`, `pandas`, `scipy`,
  `scikit-learn` (or `lightgbm`), `pytest`,
  `pyarrow`.
- A synthetic or real dataset that you can inject drift
  into. Suggested starter: reuse the churn / tabular
  dataset from `mod-104`; generate synthetic prediction
  logs by scoring a shifted copy of the validation set.

## Problem statement

### Part A — The distance functions (≈ 60 min)

In `drift/distances.py`, implement each of the following
with a clear signature and docstring:

```python
def psi(p_ref: np.ndarray, p_cur: np.ndarray,
        bins: np.ndarray) -> float: ...

def psi_categorical(ref_counts: dict[str, float],
                    cur_counts: dict[str, float]) -> float: ...

def ks_statistic(ref: np.ndarray, cur: np.ndarray) -> float: ...

def chi_squared_statistic(ref_counts: dict[str, int],
                          cur_counts: dict[str, int]) -> float: ...

def jsd(p_ref: np.ndarray, p_cur: np.ndarray) -> float: ...

def wasserstein(ref: np.ndarray, cur: np.ndarray) -> float: ...
```

Behavioural requirements:

- Handle zero bins in PSI without producing `inf` /
  `nan` (add `1e-6` to zero proportions, or use a
  small epsilon of your choice — document it).
- Handle a new category in `psi_categorical` /
  `chi_squared_statistic` — either raise a specific
  exception (`UnknownCategoryError`) or return a
  documented "unknown handling" score with a warning
  log.
- All functions return non-negative floats.
- Bin edges for PSI are inputs, not inferred from data
  — you compute them once from the reference profile.

`tests/test_distances.py` should include, at minimum:

- Identical distributions → each metric returns ≤
  small epsilon.
- Shifted mean → each metric grows monotonically as
  the shift grows (test on a synthetic Gaussian).
- Reflexivity where applicable: `d(P, P) ≈ 0`.
- KS statistic increases as two beta distributions
  are pulled apart.
- PSI on a known credit-scoring toy example (you can
  synthesise one — e.g., 10 bins, 5% mass migrating
  from bin 1 to bin 9 should give a PSI in a
  documented range).

### Part B — The multivariate score (≈ 45 min)

In `drift/multivariate.py`, implement the domain-
classifier drift score from chapter 03:

```python
def domain_classifier_score(
    reference_sample: pd.DataFrame,
    current_sample: pd.DataFrame,
    feature_names: list[str],
) -> MultivariateResult:
    """
    Returns:
        score:     2 * |AUC - 0.5|, in [0, 1]
        top_features:  list of (feature, importance) — the
                       features the classifier used most to
                       distinguish current from reference.
    """
```

Use a small model (LightGBM, sklearn's
`HistGradientBoostingClassifier`, or logistic regression
on standardised features). Train with cross-validation;
report the mean AUC. The classifier itself does not
need to be tuned — you want a coarse "can the model
tell them apart at all?" signal.

Tests:

- Two draws from the same distribution → score close
  to 0.
- Two draws from clearly different distributions →
  score close to 1.
- Top-feature attribution: inject drift into a single
  known column; assert that column is the top
  importance.

Also implement a cheap fallback: **PCA reconstruction
error**. Fit PCA on the reference; report the mean
reconstruction error on the current window. Include it
as a secondary signal in the report.

### Part C — The drift job (≈ 60 min)

`drift/job.py` — the batch runner:

```python
@dataclass
class DriftReport:
    window_id: str
    window_start: datetime
    window_end: datetime
    univariate: dict[str, FeatureDrift]     # per-feature score + threshold + breached
    multivariate: MultivariateResult
    prediction_shift: FeatureDrift           # on P(ŷ)
    n_rows: int
    reference_profile_id: str

def run_drift_job(
    reference: ReferenceProfile,
    window_df: pd.DataFrame,
    thresholds: DriftThresholds,
) -> DriftReport: ...
```

Behavioural requirements:

- Skip windows below a minimum sample size (`n_rows <
  min_samples`) and log the skip — a 20-row window
  produces meaningless drift scores.
- For each feature, pick the test based on the
  reference profile's dtype (numeric → PSI + KS,
  categorical → PSI + chi-squared). Store both the
  primary and secondary score; alert on the primary.
- Emit the report as a JSON row appended to
  `drift_scores.parquet` (or your warehouse table of
  choice) with columns: `window_id`, `feature`,
  `test`, `score`, `threshold`, `breached`,
  `reference_profile_id`, `model_version`.

CLI:

```
python -m drift.job \
  --reference-profile artifact/reference_profile.json \
  --predictions ./predictions/2026-06-14 \
  --output drift_scores.parquet \
  --thresholds thresholds.yaml
```

### Part D — The calibration script (≈ 45 min)

`drift/calibrate.py`:

- Reads a large window of historical prediction data
  (from an on-disk sample if you don't have a real
  warehouse).
- Slides the current-window through it; runs
  `run_drift_job` with a placeholder "alert on
  everything" threshold; records the score per feature
  per window.
- Emits `thresholds.yaml`:

```yaml
thresholds:
  plan_tier:      { primary: psi, threshold: 0.18 }
  months_active:  { primary: psi, threshold: 0.14 }
  country:        { primary: psi, threshold: 0.15 }
  # ...
  multivariate:   { threshold: 0.25 }
  prediction_shift: { primary: jsd, threshold: 0.03 }
```

- The threshold-picking rule is up to you but must be
  documented in `ALERTS.md`. A reasonable default is
  `max(1.5 * baseline_p99, absolute_floor)`; you can
  pick another as long as you justify it.

### Part E — The synthetic-incident test (≈ 30 min)

Write `tests/test_end_to_end.py` that simulates a real
incident and asserts your pipeline catches it:

1. Take the reference dataset; copy it.
2. In the copy, introduce a specific drift — e.g.,
   shift `income` upward by 15%, or add a new
   `plan_tier` value to 10% of rows.
3. Run `run_drift_job` on the drifted copy against the
   reference profile with your calibrated thresholds.
4. Assert:
   - `univariate[<injected feature>].breached == True`.
   - `multivariate.score > multivariate_threshold`.
   - At least one *un-drifted* feature is **not**
     breached (false-positive check).

Do this for at least three incident shapes:

- **Sudden schema change** (new category value).
- **Gradual distributional shift** (feature mean
  moves 15%).
- **Multivariate-only drift** (each marginal is stable
  but two features' correlation flips sign; the
  domain classifier should catch it; univariate tests
  should not).

### Part F — Cross-check against an off-the-shelf library (≈ 30 min)

Pick one of Evidently, whylogs, NannyML, or Alibi Detect.
Run its drift detector on the same reference + current
sample as one of your synthetic-incident tests.
Compare the top-drifted features and the aggregate
score. Write up the comparison in `ALERTS.md`:

- Do the two agree on which features drifted?
- Where they disagree, why? (Different test, different
  bucketing, different sensitivity to sample size?)
- Would you use the library or your own for this
  service in production? Why?

The point is not "your implementation must match" —
the point is to develop taste about what a drift
number *means*.

## Acceptance criteria

- `drift/distances.py` implements all six functions and
  passes the unit tests in `tests/test_distances.py`.
- `drift/multivariate.py` implements the domain
  classifier and PCA reconstruction; feature
  attribution surfaces the correct injected feature in
  tests.
- `drift/job.py` produces a `DriftReport` for a
  window; the CLI runs end-to-end from prediction
  files.
- `drift/calibrate.py` reads historical data and emits
  a `thresholds.yaml`. `ALERTS.md` documents the
  calibration methodology.
- The three synthetic-incident tests pass: each
  injected drift is caught and un-drifted features
  are not falsely flagged.
- One off-the-shelf library has been run on the same
  data and the results are compared in `ALERTS.md`.
- `pytest -q` reports zero failures.

## Stretch goals (optional)

- **Streaming windows.** Rewrite the job to consume
  from a stream (Kafka, an in-memory queue, or
  `pyarrow` batches from a directory) and emit a
  `DriftReport` per rolling window.
- **Slice-aware drift.** Add slice metadata to the
  prediction table (e.g., `country`, `cohort`); the
  drift job produces per-slice drift scores. Verify
  a slice-only drift is caught in a slice score even
  when the aggregate is clean.
- **Alert wiring.** Publish drift-breach events to a
  webhook or Slack channel (test with a mock);
  include the runbook link from chapter 07.
- **A drift-explanation report.** When the multivariate
  score breaches, generate an HTML page listing the
  top-K classifier features and side-by-side
  histograms of reference vs. current. Evidently and
  whylogs both do this — write your own from
  matplotlib/plotly and compare.
- **Backfill from `mod-108` batch inference logs.**
  Reuse batch predictions from the offline path
  (`mod-108` chapter 08) to build the historical
  window; verify the calibrated thresholds hold on
  that shape too.

## What to hand in

```
exercise-02/
├── drift/
│   ├── __init__.py
│   ├── distances.py                   # PSI, KS, chi-sq, JSD, Wasserstein
│   ├── multivariate.py                # domain classifier + PCA
│   ├── job.py                         # DriftReport pipeline + CLI
│   └── calibrate.py                   # threshold calibration script
├── tests/
│   ├── test_distances.py
│   ├── test_multivariate.py
│   ├── test_job.py
│   └── test_end_to_end.py             # synthetic incidents
├── artifact/
│   ├── reference_profile.json         # from exercise-01
│   └── reference_sample.parquet
├── thresholds.yaml                    # generated by calibrate.py
├── drift_scores.parquet               # generated by the job
├── pyproject.toml
├── README.md                          # how to run each script
└── ALERTS.md                          # calibration methodology + library cross-check
```

Solutions live in the paired `ml-engineer-solutions`
repo — do not peek until you have submitted yours.
