# exercise-01 — Metric selection rubric

> **Estimated effort:** 3 hours.
> **Companion chapters:** [02 — Metrics for regression](../02-metrics-for-regression.md), [03 — Metrics for classification](../03-metrics-for-classification.md), [04 — Calibration](../04-calibration.md), [05 — Metrics for ranking](../05-metrics-for-ranking.md).

## Objective

Build the muscle to pick a defensible metric from a
one-paragraph problem statement. By the end you should
have:

- A **`rubric.md`** — a decision tree you can point at
  when a stakeholder asks "which number should we
  report?" — with a worked example for each branch.
- A **`metric_report.py`** module that, given a task
  type and a `(y_true, y_pred_or_proba)` pair, prints a
  primary metric, a companion metric, and any baselines
  applicable for the task.
- **Case studies** — four problem statements from the
  set below, each with the metric you chose, the metric
  you rejected, and the sentence you would put in a PR
  description defending the choice.

The point of the exercise is not writing metric
functions (scikit-learn provides them); it is the
disciplined reasoning that makes the reported number
defensible under a review.

## Prerequisites

- Chapters 02–05 of this module.
- Python 3.11+; `scikit-learn ≥ 1.4`; `numpy`, `pandas`.
- Optional: `matplotlib` for reliability plots.

## Problem statement

### Part A — The rubric (≈ 45 min)

Author `rubric.md` as a decision tree covering the four
task families in the module:

- **Regression** — MAE / RMSE / R² / RMSLE / MAPE /
  MASE / quantile / Huber. When to reach for each.
- **Classification** — accuracy vs precision / recall /
  F$_\beta$ vs ROC-AUC vs PR-AUC vs log loss. Threshold
  handling. Multi-class averaging strategies.
- **Calibration** — reliability diagram, ECE, Brier,
  when to add a recalibrator.
- **Ranking** — precision@k / recall@k / hit@k / MAP /
  MRR / nDCG. Binary vs graded relevance.

For each branch, include:

- **The metric name and formula.**
- **The one-sentence "when to reach for it".**
- **The one-sentence "when it misleads".**
- **A worked example** — a single-paragraph problem
  statement and the metric you would pick for it, with
  justification.

Use tables where they are clearer than prose. The
document should be usable as a lookup — not read cover
to cover.

### Part B — The metric-report module (≈ 60 min)

Implement `metric_report.py` with this interface:

```python
def metric_report(
    task: Literal["regression", "classification",
                  "ranking", "calibration"],
    y_true,
    y_pred_or_proba,
    *,
    threshold: float | None = None,
    k: int | None = None,
    groups: np.ndarray | None = None,
) -> dict:
    """
    Return a dict with:
      - primary_metric   (name, value)
      - companion_metric (name, value) — a metric from a
        different family (e.g. F1 companion to ROC-AUC)
      - baselines        list of (name, metric_value) tuples
      - notes            list[str] with any caveats
        (e.g. threshold used, averaging strategy, min_n
        exclusions, whether the model was calibrated)
    """
```

Requirements:

- **The primary metric is task-appropriate** according
  to your rubric.
- **A companion metric is always reported** — chapter 02's
  and chapter 03's rule.
- **Trivial baselines are reported** —
  `DummyClassifier` / `DummyRegressor` values wherever
  the task admits them. Compare against the class
  balance / target mean of `y_true`.
- **For classification, if `y_pred_or_proba` is 2-D
  (probabilities), the report includes a reliability
  diagnostic** — at minimum ECE with a stated `n_bins`.
- **For ranking, `k` must be specified.** Raise
  `ValueError` if not.
- **The notes list** captures every choice not obvious
  from the numbers: threshold value, averaging strategy,
  binning, sample-size exclusions.

Test the module against at least three publicly-available
datasets — one per applicable task family. Include the
tests in `tests/test_metric_report.py`.

### Part C — Four case studies (≈ 60 min)

Pick four problem statements from the list below (mix
tasks). For each, write a short markdown file
`case-studies/<slug>.md` with:

1. The problem statement (copy verbatim).
2. **Chosen primary metric.** Which and why.
3. **Chosen companion metric.** Which and why.
4. **Rejected metric.** One metric you considered and
   ruled out, with the reason.
5. **Baselines.** What trivial and strong baselines you
   would compare against.
6. **PR-description sentence.** The one line a code
   reviewer would read.

Problem statements:

- **A.** "Predict the number of scooters that will be
  rented at each station in the next hour. Fleet
  managers plan rebalancing trips from the prediction;
  an under-forecast leaves users without scooters, an
  over-forecast wastes a rebalance trip."
- **B.** "Given a search query, rank the store's product
  catalogue so that the item the shopper eventually
  buys appears as close to the top as possible. Only
  the first ten results are shown."
- **C.** "Given a credit-card transaction, predict
  whether it is fraudulent. Roughly 0.2% of transactions
  are fraud. Flagged transactions are held for manual
  review; the review team can process at most 500
  transactions per hour."
- **D.** "Predict the probability that a patient
  readmits within 30 days of discharge. The probability
  is used as an input to a downstream expected-cost
  calculation that decides which patients receive a
  post-discharge follow-up call. Patient outcomes are
  weakly correlated within hospital."
- **E.** "Forecast the daily electricity consumption of
  a data centre for the next 14 days. Over-supply is
  cheap; under-supply forces expensive spot-market
  purchases at 5× the base rate."
- **F.** "Recommend up to five items to each user on the
  home page. Users see the ranking; some users click,
  some do not; graded relevance based on dwell time is
  available."
- **G.** "Classify support tickets into one of 50
  categories so they can be routed to the right team.
  The categories are imbalanced (top-5 categories cover
  60% of volume). Mis-routing wastes reviewer time; the
  routing decision is per-ticket."
- **H.** "Predict the sale price of a house from listing
  features. Real-estate agents use the prediction as a
  reference point when advising sellers on pricing.
  The market has occasional outliers (celebrity homes,
  distressed sales)."

Pick two case studies where the "obvious" metric is
wrong, and defend the alternative. Part of the exercise
is developing the taste to notice when the default
answer is the wrong answer.

### Part D — Notes and reflection (≈ 15 min)

Write `NOTES.md` with:

- **The one rubric branch you found hardest to
  articulate.** What was the ambiguity?
- **A case study where you changed your mind midway.**
  What did you initially pick, and what shifted it?
- **A metric from your rubric you don't fully believe.**
  Which is the least defensible entry in your own
  rubric, and why?

## Starter guidance

- **Read chapters 02–05 before writing the rubric.**
  Their decision trees are the skeleton yours should
  extend.
- **The rubric is a *decision tree*, not a comprehensive
  metric encyclopedia.** Every branch ends in a
  concrete metric choice for a concrete situation. If a
  branch says "it depends", it needs another level of
  question.
- **The companion metric matters as much as the
  primary.** A ROC-AUC alone is missing a threshold
  story; an F1 alone is missing a threshold story. Pair
  them.
- **Sanity-check your `metric_report.py` on the same
  data you would put in an MLflow run.** Metrics that
  disagree with the MLflow-logged ones are a red flag —
  find the difference (`log_loss` clipping, threshold
  default, averaging strategy) and document it.

## Acceptance criteria

- `rubric.md` covers all four task families with at
  least three branches per family, each ending in a
  concrete metric choice with a worked example.
- `metric_report.py` runs on three real datasets and
  returns dicts matching the schema in Part B; passes
  the test suite you wrote.
- Four case-study files exist, each with all six
  required sections, and at least two of them defend a
  non-default metric.
- `NOTES.md` includes the three reflection answers.
- Everything committed to a single `exercise-01/`
  directory.

## Stretch goals (optional)

- **Add a fifth task family** — density estimation,
  clustering, or anomaly detection. Which metrics
  belong? Which are its "accuracy is the wrong number"
  traps? Extend the rubric.
- **Metric under label noise.** Simulate label noise
  (flip 5% of labels) on one classification dataset
  and observe how ROC-AUC, PR-AUC, F1, and log loss
  respond. Which is most robust to label noise? Add a
  note to the rubric.
- **Metric-vs-cost calculator.** For case study C
  (fraud with reviewer capacity 500 / hour), implement
  a small utility that, given a predicted probability
  vector and the reviewer capacity, computes the
  precision, recall, and total-cost curves. Which
  operating point maximises net saved fraud dollars?
- **Contribute one rubric branch** to the module's
  future `resources.md` as a candidate reference — for
  a metric you had to look up because the module did
  not cover it well.

## What to hand in

```
exercise-01/
├── rubric.md
├── src/
│   ├── metric_report.py
│   └── __init__.py
├── tests/
│   └── test_metric_report.py
├── case-studies/
│   ├── <slug-1>.md
│   ├── <slug-2>.md
│   ├── <slug-3>.md
│   └── <slug-4>.md
├── notebooks/
│   └── metric_report_demo.ipynb      # runs metric_report against 3 datasets
├── pyproject.toml
└── NOTES.md
```

Solutions live in the paired `ml-engineer-solutions` repo
— do not peek until you have submitted yours.
