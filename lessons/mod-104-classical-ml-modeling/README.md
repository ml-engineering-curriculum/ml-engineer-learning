# mod-104 — Classical ML Modeling: Regression, Trees, Ensembles

> **Track:** Machine Learning Engineer (level 20)
> **Estimated effort:** 16 hours

`mod-103` handed you an ML-ready DataFrame — typed, cleaned,
split, and schema-validated. This module fits models on it.

For most tabular supervised-learning problems in production, the
model that wins is a linear model with regularisation, a random
forest, or a gradient-boosted tree ensemble. This module trains
those, tunes them honestly with cross-validation and modern
hyperparameter search, and explains the winner well enough that a
stakeholder — or a regulator — can follow the reasoning.

## Learning objectives

- Train and tune linear/logistic regression with regularisation
  (Ridge, Lasso, elastic-net).
- Train decision trees, random forests, and gradient-boosted trees
  (XGBoost / LightGBM / CatBoost / `HistGradientBoosting`).
- Apply k-fold cross-validation and hyperparameter search
  (grid / random / halving / Optuna).
- Interpret models with feature importance and SHAP, and avoid the
  common feature-selection pitfalls.

## Chapters

| # | Chapter | Maps to objective |
|---|---|---|
| 01 | [Why classical ML still wins](01-why-classical-ml-still-wins.md) | orientation |
| 02 | [Linear and logistic regression](02-linear-and-logistic-regression.md) | regression |
| 03 | [Regularisation: Ridge, Lasso, and elastic-net](03-regularisation-ridge-lasso-elasticnet.md) | regression |
| 04 | [Decision trees](04-decision-trees.md) | trees & ensembles |
| 05 | [Random forests and bagging](05-random-forests-and-bagging.md) | trees & ensembles |
| 06 | [Gradient-boosted trees: XGBoost, LightGBM, CatBoost](06-gradient-boosted-trees.md) | trees & ensembles |
| 07 | [Cross-validation strategies](07-cross-validation-strategies.md) | CV & tuning |
| 08 | [Hyperparameter search: grid, random, and Optuna](08-hyperparameter-search.md) | CV & tuning |
| 09 | [Feature importance and selection pitfalls](09-feature-importance-and-selection-pitfalls.md) | interpretation |
| 10 | [Model interpretation with SHAP](10-shap-and-model-interpretation.md) | interpretation |

## Exercises

| # | Exercise | Hours |
|---|---|---|
| 01 | [Regression with regularisation](exercises/exercise-01-regression-with-regularisation.md) | 3 |
| 02 | [Trees and ensembles](exercises/exercise-02-trees-and-ensembles.md) | 3 |
| 03 | [XGBoost or LightGBM tuning with Optuna](exercises/exercise-03-xgboost-or-lightgbm-tuning.md) | 4 |
| 04 | [Model explainability with SHAP](exercises/exercise-04-model-explainability-with-shap.md) | 3 |

The four exercises follow the module's arc: fit a regularised
linear model, compare the tree family, tune a gradient booster with
Optuna, and explain the winner with SHAP. Do them in order.

Plus one lab (`labs/`) and one quiz (`quizzes/`) — scaffolded on
subsequent authoring cycles.

## How to use this module

Read the chapters in order — later chapters build on earlier ones.
Three are worth slowing down for:

- **06 (Gradient-boosted trees).** The single highest-return chapter
  for tabular practitioners. The learning-rate / tree-depth /
  early-stopping vocabulary you learn here transfers across all four
  boosting libraries.
- **07 (Cross-validation strategies).** Short and non-negotiable.
  A wrong CV split silently invalidates every number downstream.
- **10 (SHAP).** The current best answer to "why did the model
  predict what it did?" Everything you report to a stakeholder about
  feature effects lives or dies on this chapter's discipline.

Chapters 04–05 (decision trees, random forests) are prerequisites
for 06. If you already know how CART works, chapter 04 is a fast
skim.

## What this module does **not** cover

- **Deep learning.** `mod-105` covers MLPs, CNNs, transformers, and
  the neural-network toolchain.
- **Full model-evaluation methodology.** ROC/PR curves, calibration,
  cost-sensitive metrics, and fairness slicing are `mod-107`. This
  module reports scores; the next argues about which score.
- **Experiment tracking.** MLflow / Weights & Biases / DVC are
  `mod-106`. Runs here log to `stdout` and `REPORT.md`.
- **Deployment and serving.** How to load a fitted `Pipeline` into
  a FastAPI endpoint, batch vs. online serving, latency budgets —
  all of that lives in `mod-108`.
- **Feature engineering.** `mod-103`. We assume you have
  `(X, y)` and train on it.
- **Unsupervised learning at depth.** Clustering, PCA, and matrix
  factorisation are touched only where they support the supervised
  workflow, not as first-class topics.
- **Time-series forecasting as a specialty.** Time-ordered CV
  appears in chapter 07 because every practitioner needs it;
  ARIMA / Prophet / TFT do not.

## Prerequisites

- `mod-101` (ML foundations), `mod-102` (Python ML toolchain), and
  `mod-103` (data engineering for ML). Chapters here assume you can
  build a scikit-learn `Pipeline`, split without leakage, and write
  a `pytest` test.

See [`resources.md`](resources.md) for official documentation and
curated references behind each chapter.
