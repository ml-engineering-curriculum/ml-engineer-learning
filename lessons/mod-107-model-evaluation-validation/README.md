# mod-107 — Model Evaluation & Validation

> **Track:** Machine Learning Engineer (level 20)
> **Estimated effort:** 12 hours

`mod-104` (classical ML) and `mod-105` (deep learning) trained
models that produce predictions. `mod-106` tracked the runs so
the numbers are reproducible. This module answers the question
everyone eventually asks about those numbers: **can we defend
them?**

Two failure modes account for most "we shipped a regression"
stories in production ML, and neither is caught by unit tests
or by tracking discipline alone:

1. **The metric was wrong for the task.** Accuracy on a 99%-
   negative dataset. R² on a task where a fat-tailed residual
   is what actually costs money. ROC-AUC on a retrieval task
   where only the top-10 positions matter.
2. **The validation split was wrong for the task.** Future
   information bled into training. Two rows for the same
   patient / customer / vehicle landed on opposite sides of
   the split. A `StandardScaler` was fit on all the data
   before the split, then re-used inside the CV loop.

This module is the discipline for catching both — pick a
metric that mirrors the decision the model will drive, design
a split that mimics the way the model will be used in
production, then break the aggregate open with baselines,
error analysis, and slice reports so the worst case is
visible before it ships.

## Learning objectives

- Pick the right metric for the task — regression,
  classification, ranking, and calibration — and pair it with
  an informative companion metric.
- Design validation strategies that prevent leakage:
  time-based (walk-forward, expanding-window),
  group-based (`GroupKFold`, `StratifiedGroupKFold`), and
  nested CV for honest hyperparameter selection.
- Build informative baselines — trivial (`DummyClassifier` /
  `DummyRegressor`, persistence, popularity) and strong-
  classical (logistic regression, GBDT) — and conduct
  structured error analysis on the residuals.
- Run lightweight fairness / robustness slice reports
  appropriate for a level-20 evaluation: subgroup performance
  gaps, perturbation robustness, and distribution-shift
  probes.

## Chapters

| # | Chapter | Maps to objective |
|---|---|---|
| 01 | [Why evaluation and validation](01-why-evaluation-and-validation.md) | orientation |
| 02 | [Metrics for regression](02-metrics-for-regression.md) | metrics |
| 03 | [Metrics for classification](03-metrics-for-classification.md) | metrics |
| 04 | [Calibration](04-calibration.md) | metrics |
| 05 | [Metrics for ranking](05-metrics-for-ranking.md) | metrics |
| 06 | [Splits, leakage, and cross-validation](06-splits-leakage-and-cross-validation.md) | validation |
| 07 | [Time-based validation](07-time-based-validation.md) | validation |
| 08 | [Group-based and nested cross-validation](08-group-based-and-nested-cv.md) | validation |
| 09 | [Baselines that tell you something](09-baselines.md) | baselines & error analysis |
| 10 | [Error analysis](10-error-analysis.md) | baselines & error analysis |
| 11 | [Fairness and robustness slices](11-fairness-and-robustness-slices.md) | slices |

## Exercises

| # | Exercise | Hours |
|---|---|---|
| 01 | [Metric selection rubric](exercises/exercise-01-metric-selection-rubric.md) | 3 |
| 02 | [Leakage-resistant validation](exercises/exercise-02-leakage-resistant-validation.md) | 3 |
| 03 | [Baselines and error analysis](exercises/exercise-03-baselines-and-error-analysis.md) | 3 |
| 04 | [Fairness and robustness slices](exercises/exercise-04-fairness-and-robustness-slices.md) | 3 |

The four exercises follow the module's arc: pick the metric,
guard the split, break the aggregate apart with baselines and
error analysis, then slice the result along the axes that
matter. Do them in order.

Plus one lab (`labs/`) and one quiz (`quizzes/`) — scaffolded
on subsequent authoring cycles.

## How to use this module

Read the chapters in order. Three are worth slowing down for:

- **03 (Metrics for classification).** The confusion-matrix
  vocabulary and the ROC-vs-PR trade-off under class imbalance
  are the load-bearing reference for the rest of the module.
  Chapters 04, 05, 09, 10, and 11 all assume you have it.
- **06 (Splits, leakage, and cross-validation).** The five-
  kind leakage taxonomy in this chapter is what you carry into
  every future PR review. A wrong split silently invalidates
  every number downstream, including the ones this module
  teaches you to compute.
- **11 (Fairness and robustness slices).** The level-20 bar:
  what a competent ML engineer reports about worst-case
  behaviour before a model ships. Short chapter, high leverage.

Chapters 02–05 are the metrics half; chapters 06–08 are the
validation half; chapters 09–11 turn a single aggregate number
into a defensible report. Every exercise pulls from two or
three chapters — the mapping is in the exercise headers.

## What this module does **not** cover

- **Statistical significance testing** as a first-class topic
  (paired t-tests, bootstrap CIs, McNemar's test). The
  vocabulary appears where it is directly useful — the
  underlying inference course is out of scope.
- **Full fairness audits.** Chapter 11 delivers the level-20
  slice report. Full audits require domain ethicists, legal
  review, and often specialised modelling techniques
  (adversarial de-biasing, causal fairness). We cite where
  to go next.
- **Adversarial robustness** (FGSM, PGD, certified defences).
  The perturbations in chapter 11 are naturally occurring, not
  adversarially chosen. Adversarial ML is its own research
  area.
- **Online / production monitoring.** Detecting drift and
  performance decay after a model ships is `mod-109`. This
  module produces the pre-launch report; the next monitors
  what actually happens.
- **Deployment concerns** — latency budgets, serving-time
  metrics, staged rollouts. That is `mod-108`.
- **Bandit / interleaving experiments** for ranking. Offline
  ranking metrics are in chapter 05; the online-experiment
  literature is a separate track.

## Prerequisites

- `mod-104` (classical ML) *or* `mod-105` (deep learning) —
  a trained model that produces predictions on a held-out
  set. Every example works with either a scikit-learn
  `Pipeline` or a PyTorch model wrapped in an inference
  function.
- `mod-103` (data engineering for ML): the dataset with a
  documented schema and a plausible train/dev/test partition.
- `mod-106` (experiment tracking): the metrics you compute in
  this module get logged as MLflow metrics; exercise-03
  writes error-analysis artefacts to the MLflow run store.
- Python 3.11+; `scikit-learn ≥ 1.4`; `numpy`, `pandas`,
  `matplotlib`.

See [`resources.md`](resources.md) for official documentation
and curated references behind each chapter.
