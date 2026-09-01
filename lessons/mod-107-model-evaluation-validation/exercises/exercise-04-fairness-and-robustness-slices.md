# exercise-04 — Fairness and robustness slices

> **Estimated effort:** 3 hours.
> **Companion chapters:** [10 — Error analysis](../10-error-analysis.md), [11 — Fairness and robustness slices](../11-fairness-and-robustness-slices.md).

## Objective

Turn the model you evaluated in exercise-03 into a
report that a stakeholder can use to answer the two
questions the aggregate metric hides:

- **Which subgroups does this model fail?**
- **What happens when the input shifts a little?**

By the end you should have:

- A **`slice_report.md`** — the level-20 report from
  chapter 11: an aggregate metric at the top and a
  table of slices ordered by their gap, covering all
  three dimensions (subgroup performance, perturbation
  robustness, distribution shift).
- A **`slice_reporter.py`** module that regenerates the
  table from a `(y_true, y_pred_or_proba, feature_df)`
  triple and a slice specification, so the report is a
  reproducible artefact rather than a one-off
  spreadsheet.
- A **worst-slice call** — a one-paragraph judgement in
  `NOTES.md` on whether the model is fit to deploy,
  what you would ask a stakeholder before shipping,
  and which slice is the deployment risk.

The point of the exercise is *not* to invent new
fairness metrics or run a full audit. It is to build
the reporting discipline that surfaces the worst case
before someone else has to.

## Prerequisites

- Chapters 10 and 11 of this module.
- **A trained model with held-out predictions** from
  exercise-03 (or `mod-104` / `mod-105` if you skipped
  ahead). The exercise is written for a classification
  model; regression variants are called out where they
  differ.
- Python 3.11+; `scikit-learn ≥ 1.4`; `pandas`, `numpy`,
  `matplotlib`.
- Optional: `fairlearn ≥ 0.10` for the standard fairness
  metric names; the exercise does not require it.

## The dataset

Pick a dataset that has:

- **At least one plausibly sensitive or operationally
  important attribute** (age band, gender, region,
  customer tier — whatever your model's stakeholders
  would ask about). The attribute does not have to be
  a training feature; it just has to be attached to
  the held-out rows.
- **A meaningful time or geography axis** you can hold
  out as a distribution-shift probe.

Two good defaults, both publicly available and used in
earlier modules:

- **UCI Adult (Census Income).** Sensitive attributes:
  `sex`, `race`, `age`. Distribution shift can be
  constructed by holding out one country subset or one
  age band. Loadable via `sklearn.datasets.fetch_openml
  ("adult", version=2)`.
- **NYC Taxi trip data** (any month subset). Operational
  slices: pickup borough, hour-of-day bucket,
  weekend / weekday. Distribution shift: hold out the
  last week or a specific borough.

State your dataset choice, the model you are evaluating,
and the primary + companion metrics (chapter 03's rule)
up front in `NOTES.md`.

## Problem statement

### Part A — Subgroup performance (≈ 45 min)

Produce a subgroup-performance table for **at least
three attributes**. At least one must be a plausibly
sensitive attribute (as above); at least one must be an
operational subgroup (business-important but not
regulated).

For each attribute:

- Compute the **primary metric** and the **companion
  metric** on each subgroup, using the chapter 11
  `min_n` pattern (default `min_n = 100`; state your
  threshold and any exclusions).
- Report the **positive prediction rate** per subgroup.
- Report the **largest pairwise gap** in the primary
  metric across subgroups.

Author the code as `slice_reporter.py::subgroup_report`
using roughly this interface:

```python
def subgroup_report(
    df,
    *,
    attribute: str,
    y_true: str,
    y_score: str,
    metric_fn,           # e.g. roc_auc_score
    companion_fn=None,   # e.g. average_precision_score
    min_n: int = 100,
) -> pd.DataFrame:
    """
    Return one row per subgroup with:
      - n
      - positive_rate
      - primary
      - companion (if provided)
    plus a `.attrs["gap"]` scalar for the largest
    primary-metric gap and a `.attrs["excluded"]` list
    of subgroups below `min_n`.
    """
```

State in `NOTES.md`, for each attribute:

- **The largest gap and its direction.** Which subgroup
  is worst served? Is that consistent with known bias
  in the training data or the label collection process?
- **Any subgroup you dropped for `n < min_n`.** Size and
  fraction of the population. A subgroup you cannot
  measure is a subgroup you cannot vouch for.
- **The reporting convention.** State whether you are
  reporting *demographic parity* (gap in positive
  prediction rate) or *equalised odds* (gap in TPR /
  FPR) — chapter 11's language — and why.

### Part B — Perturbation robustness (≈ 45 min)

Apply **at least two task-preserving perturbations**
to the held-out inputs and re-score.

Pick perturbations that a deployed system would
plausibly encounter. Suggestions by modality:

- **Tabular.** Add Gaussian noise to numerical features
  at σ = 0.05, 0.10, 0.20 times each feature's standard
  deviation; flip 1%, 5%, 10% of categorical values to
  another observed category.
- **Text.** Character-level typo injection at
  1%, 3%, 5% (libraries: `nlpaug`, `TextAttack`); case
  perturbation; punctuation removal.
- **Image.** Small rotations (±5°), brightness /
  contrast shifts (±10%), JPEG quality drop
  (85 → 60 → 40) (libraries: `albumentations`,
  `imgaug`).

For each perturbation:

- Report the **clean metric**, the **perturbed metric**,
  and the **drop** at each intensity.
- Document the **exact perturbation function** (seed,
  σ, flip probability) — the number is only meaningful
  if someone else can reproduce it.
- **Sanity-check the label.** Spot-read a sample of
  perturbed rows; if the perturbation is aggressive
  enough that a human would relabel the row, the
  drop is not a robustness finding.

Extend `slice_reporter.py` with:

```python
def perturbation_report(
    predict_fn,
    X, y,
    metric_fn,
    perturbations: list[tuple[str, Callable]],
) -> pd.DataFrame:
    """
    Return one row per perturbation with clean, perturbed,
    drop, and the perturbation label.
    """
```

`predict_fn(X) -> y_pred_or_proba` isolates the model
from the loop, so this module works against any
trained estimator.

### Part C — Distribution shift (≈ 30 min)

Construct **at least one distribution-shift slice**.
The rule of thumb from chapter 11: the shift must be
one the deployment will *actually see*, not an
adversarial worst-case you can invent.

Pick from:

- **Time shift.** Hold out the latest week / month /
  quarter and evaluate on it as the "future" the model
  will predict on.
- **Geographic shift.** If the training data skews to
  one region, hold out a different region as the
  shifted set.
- **Version / device shift.** Held-out iOS-only rows
  when Android-heavy in training, or held-out mobile
  when desktop-heavy.
- **Rare-event shift.** Up-sample the positive class
  in a subset to simulate a higher-prevalence regime
  (for fraud / anomaly work).

Report:

- **The metric on the shifted subset** and the drop vs
  aggregate.
- **One shift diagnostic** — the Population Stability
  Index (PSI) on the primary feature the model
  depends on, *or* a Kolmogorov–Smirnov statistic on
  a continuous feature, comparing the training and
  shift distributions. State the diagnostic and its
  value; do not over-interpret it (chapter 11: PSI is
  about the data, not the model).

### Part D — Compose the report (≈ 30 min)

Author `slice_report.md` following the chapter 11
table format:

| Slice | Kind | n | Primary | Companion | Δ vs aggregate |
|---|---|---|---|---|---|
| Aggregate | overall | ... | ... | ... | 0.000 |
| Subgroup — `<attribute> = <value>` | fairness | ... | ... | ... | ... |
| Perturbation — `<name>` at intensity `<i>` | robustness | ... | ... | ... | ... |
| Shift — `<name>` | distribution | ... | ... | ... | ... |

Ordering:

- Aggregate at the top.
- Slices below, sorted by |Δ| descending. The **worst
  slice appears second**, so a reader who only reads
  two rows sees the deployment risk.

Include a short prose section per dimension explaining
what you picked and why. Chapter 11's rule: publish the
aggregate *and* the worst slice; the gap is the risk.

### Part E — Reflection (≈ 15 min)

In `NOTES.md`:

- **The worst slice call.** One paragraph: is this
  model fit to deploy? If not, what would need to be
  true first? Which subgroup / perturbation /
  shift is the deployment risk?
- **A slice you expected to matter that did not.**
  What did that null result teach you about the
  model?
- **A slice you did not measure but wish you could.**
  What data would you need? Chapter 11's honest-
  response: if you cannot measure it, the report says
  "not measured" and the deployment plan owns a
  monitoring alert for it (a `mod-109` hand-off).
- **The intersection you looked at.** Chapter 11
  points at combined slices (worst subgroup ×
  worst perturbation) as where worst-case behaviour
  hides. Pick one intersection, report its metric,
  and comment on whether it agrees with or
  contradicts the marginal slices.

## Starter guidance

- **Use the same predictions everywhere.** Load the
  held-out `y_true`, `y_score`, and `feature_df` once;
  every slice call operates on the same arrays.
  Recomputing predictions per slice is a bug surface.
- **Fix the metric before you slice.** The primary and
  companion metric come from exercise-01's rubric;
  bring them across so the report is comparable to
  every other artefact for this model.
- **Reuse chapter-11 code as skeletons.** The
  `subgroup_metrics` and `robustness_slice` functions
  in the chapter are starting points; extend them,
  don't reinvent them.
- **Perturbations that change the label are not
  robustness perturbations.** If Gaussian noise moves
  most rows across the decision boundary, either the
  noise is too aggressive or the model is genuinely
  brittle at that operating point. Chapter 11's fifth
  pitfall.
- **State every choice.** `min_n`, perturbation seed,
  σ, PSI bin count, which fairness definition you
  are reporting. Every unstated choice becomes a
  question in the review.

## Acceptance criteria

- `slice_reporter.py` implements `subgroup_report`,
  `perturbation_report`, and a `shift_report` (or an
  equivalent decomposition), each returning a
  `pandas.DataFrame` and each callable independently.
- `slice_report.md` contains the aggregate row and at
  least **seven slice rows** total: ≥ 3 subgroup rows
  (from ≥ 2 attributes), ≥ 2 perturbation rows (from
  ≥ 2 perturbations), ≥ 2 distribution-shift rows
  (one shift diagnostic value plus at least one
  shifted-subset metric).
- The slice table is ordered by |Δ| descending, with
  aggregate at the top and the worst slice visible
  in the first two rows.
- Fairness definition (parity vs equalised odds vs
  other) is stated explicitly in the report, not left
  implicit.
- `NOTES.md` answers all four reflection questions
  and states the worst-slice call.
- All code committed under `exercise-04/`.

## Stretch goals (optional)

- **Confidence intervals on the slice metrics.**
  Bootstrap each slice with
  `sklearn.utils.resample` (chapter 06) and report a
  95% CI alongside the point estimate. A "worst
  slice" whose CI overlaps the aggregate is a weaker
  finding than a "worst slice" whose CI is strictly
  below.
- **Fairlearn-native report.** Re-implement the
  subgroup table using `fairlearn.metrics.MetricFrame`
  and `demographic_parity_difference` /
  `equalized_odds_difference`. Compare the numbers
  to your hand-rolled version; document any
  discrepancies.
- **Adversarial-perturbation appendix.** For a
  classification model that accepts numerical
  features, generate an FGSM-style perturbation
  (chapter 11 explicitly says this is out of scope
  for the level-20 report; it is a valid stretch
  goal). Report the drop as an *appendix*, clearly
  labelled adversarial, not confused with the
  naturally-occurring perturbations in Part B.
- **Slice-threshold calibration.** For a subgroup
  with a large gap, refit a per-group decision
  threshold using chapter 03's threshold-tuning
  pattern and report the equalised-odds metrics
  before and after. This is *not* de-biasing — it is
  operating-point tuning — and the report should
  state that distinction.
- **Automate a slice-regression check.** Save the
  current slice metrics. Write a `pytest` test that
  fails if a future model regresses the metric on
  any slice by more than a stated tolerance (say
  2 metric points). This is the first primitive of
  monitoring; `mod-109` picks it up.

## What to hand in

```
exercise-04/
├── src/
│   └── slice_reporter.py
├── tests/
│   └── test_slice_reporter.py
├── notebooks/
│   └── slice_report_demo.ipynb           # regenerates slice_report.md
├── slice_report.md
├── artefacts/
│   ├── subgroup_metrics.csv
│   ├── perturbation_metrics.csv
│   └── shift_metrics.csv
├── pyproject.toml
└── NOTES.md
```

Solutions live in the paired `ml-engineer-solutions` repo
— do not peek until you have submitted yours.
