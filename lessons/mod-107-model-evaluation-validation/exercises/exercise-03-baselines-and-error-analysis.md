# exercise-03 — Baselines and error analysis

> **Estimated effort:** 3 hours.
> **Companion chapters:** [09 — Baselines](../09-baselines.md), [10 — Error analysis](../10-error-analysis.md).

## Objective

Take a model you already trained — from `mod-104` (a
classical scikit-learn pipeline) or `mod-105` (a PyTorch
model) — and produce the two artefacts that make a metric
defensible:

- **A baselines table** — the trivial and strong-classical
  baselines the model is compared against, computed on
  the same held-out set as the model.
- **An error-analysis report** — the five moves from
  chapter 10 (confusion matrix, top-K worst errors,
  per-slice metrics, error clusters, cross-model
  comparison) applied to the model, with a ranked list
  of hypotheses that would inform the next iteration.

The point of the exercise is to convert an aggregate
metric into a report that a stakeholder can read and
act on. If the reader cannot answer "should we ship
this?" after reading your report, the report is
incomplete.

## Prerequisites

- Chapters 09 and 10.
- A trained model with a held-out set from `mod-104` or
  `mod-105`. If you do not have one, fit a
  `HistGradientBoostingClassifier` on the UCI Adult
  income dataset (used in `mod-106` chapter 02) — the
  imbalance and rich feature set make it a good
  target.
- `mod-106` (experiment tracking) — you will log
  baselines and error-analysis artefacts as MLflow
  metrics / artefacts.

## Problem statement

### Part A — Add the baselines (≈ 45 min)

Extend the training script so that every run also fits
and scores baselines on the same held-out set as the
model. Report:

- **At least two trivial baselines** using
  `sklearn.dummy.DummyClassifier` / `DummyRegressor`
  (most_frequent / stratified for classification;
  mean / median for regression).
- **At least one strong-classical baseline** — a
  well-tuned logistic regression / linear regression
  for the linear reference, and a well-tuned GBDT
  (`HistGradientBoostingClassifier` /
  `HistGradientBoostingRegressor` with a small
  hyperparameter sweep) for the tree reference.
- **The persistence baseline** if the task is
  time-series.
- **The popularity baseline** if the task is retrieval /
  recommendation.

For each baseline, log the same set of metrics you log
for the model — including the companion metric from
chapter 03. Use `mlflow.log_metric` with names like
`baseline_majority_f1`, `baseline_logreg_roc_auc`, etc.,
so a single run's UI shows the model metric next to
each baseline metric.

Produce a `baselines.md` (in the run's artefact store
too) with the table from chapter 09:

| Model | Primary | Companion |
|---|---|---|
| Trivial (majority) | ... | ... |
| Trivial (stratified) | ... | ... |
| LogReg (tuned) | ... | ... |
| GBDT (tuned) | ... | ... |
| **Your model** | ... | ... |

Highlight in bold or with a marker the model that
"wins" on the primary metric. Include the *increment*
of your model over the strongest classical baseline.

### Part B — Error analysis (≈ 90 min)

Author `error_analysis.py` producing the five moves
from chapter 10 for the held-out predictions:

1. **Confusion matrix** (classification) or **residual-
   vs-x plot** (regression). Save as an artefact.
2. **Top-20 highest-loss rows.** Save as a CSV
   artefact; include the true label, predicted label,
   predicted probability / value, and all input
   features. Sort by per-row loss.
3. **Per-slice metrics** for at least three sensible
   feature slices from your dataset (numerical
   buckets, categorical values, time bucket). Use
   `min_n = 50`. Save as a CSV.
4. **Cluster the errors.** Use `KMeans` on the error
   rows' features (or on a learned embedding for
   text / image data). Produce 5–10 clusters; for
   each, log the cluster centroid summary and a
   representative example.
5. **Cross-model comparison.** Pick one baseline model
   from Part A (the strongest classical one is a good
   choice). Compute the sets `errors_only_baseline`,
   `errors_only_model`, `errors_both`; report their
   sizes and inspect representative examples from each.
   The `errors_only_model` set is your regression list
   — the aggregate metric hides these but the
   deployment does not.

Log the outputs as MLflow artefacts under an
`error_analysis/` folder in the run's artefact store.

### Part C — Hypotheses (≈ 30 min)

Turn the analysis into concrete claims. Produce
`hypotheses.md` with a ranked list of at least **three**
hypotheses, each formatted:

```
### Hypothesis N: <one-line title>

- Slice / cluster: <where in the data>
- Volume: <fraction of validation traffic>
- Current metric on the slice: <number>
- Population metric: <number>
- Estimated impact on aggregate if closed to
  parity: <back-of-envelope calculation>
- Proposed intervention: <feature? more data? model
  change? loss re-weighting?>
- How you would measure whether the intervention
  worked: <named metric on the named slice>
```

Rank by `volume × gap`. Chapter 10's rule: work the
highest-leverage hypothesis first.

### Part D — Reflection (≈ 15 min)

In `NOTES.md`:

- **Did any of the top-20 highest-loss rows turn out to
  be label errors?** How many? What is the implied
  label-noise floor on the metric?
- **Did per-slice metrics reveal a slice you did not
  expect to matter?**
- **What did the cross-model comparison show?** Is your
  model strictly better than the classical baseline
  or are there real regressions?
- **Which hypothesis from Part C do you actually
  believe will move the metric the most, and why?**

## Starter guidance

- **Compute per-row loss the same way MLflow does.**
  Use `sklearn.metrics.log_loss` with the same
  clipping so the row losses sum to the reported log
  loss.
- **Save the analysis dataframe as an artefact.** A CSV
  of `(row_id, true, pred, proba, per_row_loss,
  key_features)` lets you re-run the analysis against
  a future model without re-computing predictions.
- **`min_n` matters.** State it in `NOTES.md`; a
  slice with 20 rows is noisy, not a finding.
- **Do not fix anything in this exercise.** The output
  is *hypotheses*, not model changes. Iterating on the
  model is where you go next.

## Acceptance criteria

- Every run logs baselines alongside the model, in the
  same MLflow experiment, with metric names prefixed
  `baseline_`.
- `baselines.md` and the run UI both show a table with
  at least four rows (2 trivial + 1 strong-classical
  + model). The increment over the strongest classical
  baseline is explicit.
- `error_analysis/` artefacts include: confusion
  matrix / residual plot, top-20 CSV, per-slice CSV
  covering ≥ 3 slices, cluster summary, cross-model
  set comparison.
- `hypotheses.md` has ≥ 3 hypotheses, ranked by
  `volume × gap`, each with the six required fields.
- `NOTES.md` has answers to all four reflection
  questions.

## Stretch goals (optional)

- **Human-agreement ceiling.** If your dataset has
  multiple annotations per item, compute
  `cohen_kappa_score` between annotators as the
  ceiling. Add a row to `baselines.md`.
- **Interactive error viewer.** A tiny Streamlit or
  Dash app that lets a reviewer page through the
  top-K errors with the input rendered nicely.
- **Slice-metric heatmap.** For pairs of features,
  compute the metric on the 2-D bucketed slice.
  Heatmap the result — the darkest cells are the
  worst-performing feature combinations.
- **Automatic regression test.** Save the current
  errors-per-slice numbers. In CI, fail the build if
  a future model regresses the metric on any slice
  by more than a stated tolerance. This turns error
  analysis into a monitored gate.
- **Address one hypothesis** — actually make the
  change, re-run everything, and compare error sets
  before/after. This is the whole loop.

## What to hand in

```
exercise-03/
├── src/
│   ├── train_with_baselines.py
│   ├── error_analysis.py
│   └── slice_report.py
├── artefacts/
│   ├── confusion_matrix.png             # or residual_plot.png
│   ├── top20_errors.csv
│   ├── per_slice_metrics.csv
│   ├── error_clusters.csv
│   └── cross_model_comparison.md
├── baselines.md
├── hypotheses.md
├── pyproject.toml
└── NOTES.md
```

Solutions live in the paired `ml-engineer-solutions` repo
— do not peek until you have submitted yours.
