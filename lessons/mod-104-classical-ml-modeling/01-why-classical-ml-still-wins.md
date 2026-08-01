# 01 — Why classical ML still wins

> **Reading time:** ~15 minutes.
> **Prerequisite modules:** `mod-101` (foundations), `mod-102` (Python
> toolchain), `mod-103` (data engineering for ML).

## Motivation

`mod-103` closed with an ML-ready DataFrame: correctly typed, cleanly
joined, leakage-free, reproducibly split, and validated by a schema.
This module trains a model on it.

The "model" in most production tabular problems is a linear model with
regularisation, a random forest, or a gradient-boosted tree ensemble.
It is not a neural network. It is not a large language model. The
reason is boring: on structured tabular data, gradient-boosted trees
still win, and linear models still hold their own as baselines,
serve-time-cheap alternatives, and calibration references. Empirical
comparisons between deep-learning approaches for tabular data (TabNet,
FT-Transformer, SAINT, and others) and mature gradient-boosting
libraries (XGBoost, LightGBM, CatBoost) have repeatedly found the
tree-based methods competitive or superior on standard tabular
benchmarks.<!-- needs-research: verify against the Shwartz-Ziv &
Armon 2022 "Tabular Data: Deep Learning is Not All You Need" paper
(https://arxiv.org/abs/2106.03253) and the Grinsztajn et al. 2022
"Why do tree-based models still outperform deep learning on tabular
data?" paper (https://arxiv.org/abs/2207.08815). Update wording if
the current consensus has shifted. -->

Deep learning is not off the table — `mod-105` is about it, and for
images, text, and audio it is the right tool. But for the workflow
that most ML engineers spend most of their time on — a supervised
learning problem on tabular data from a warehouse — classical ML is
still the pragmatic default.

## What "classical" means here

For the purposes of this module, **classical ML** means the family
of estimators that predate deep learning and still dominate tabular
supervised learning:

- **Linear / logistic regression** — with L1, L2, or elastic-net
  regularisation. The workhorses of interpretable, cheap-to-serve
  models. Also the calibration reference (their probability outputs
  are famously well-behaved after Platt scaling / isotonic
  regression, see `mod-107`).
- **Decision trees** — CART, the recursive greedy splitter that
  underlies every ensemble below.
- **Random forests** — bagging over decorrelated trees. Strong
  baseline, minimal tuning, gives free out-of-bag error estimates.
- **Gradient-boosted trees** — the ensemble family that ships in
  the majority of Kaggle-winning tabular solutions and in most
  production tabular models. XGBoost, LightGBM, and CatBoost are
  the three implementations you will actually reach for.

Not covered here: k-nearest-neighbours (largely superseded), naive
Bayes (still useful in NLP contexts, but the modality is text and
belongs closer to `mod-105`), support vector machines (usable, but
slower than gradient boosting and rarely competitive on modern
tabular data), and unsupervised methods (clustering, PCA, matrix
factorisation — a separate discipline that we touch only where it
supports supervised learning).

## What this module teaches

The four learning objectives:

1. **Train and tune linear/logistic regression with
   regularisation.** L1 (Lasso), L2 (Ridge), and elastic-net. When
   each is appropriate, how the regularisation strength interacts
   with feature scaling, and the coordinate-descent / iterative
   solvers behind `scikit-learn`'s implementations.
2. **Train decision trees, random forests, and gradient-boosted
   trees.** The recursive-partitioning algorithm; bagging and
   random-subspace decorrelation; boosting as sequential residual-
   fitting. Practical use of `RandomForestClassifier`,
   `HistGradientBoostingClassifier`, `xgboost.XGBClassifier`,
   `lightgbm.LGBMClassifier`.
3. **Apply k-fold cross-validation and hyperparameter search.**
   Stratified, grouped, and time-series CV; grid vs. random vs.
   Bayesian (Optuna) search; the nested-CV trap and the
   halving-search family for expensive estimators.
4. **Interpret models with feature importance and SHAP.** Impurity-
   based importance and its biases; permutation importance and its
   caveats under correlated features; TreeSHAP for the tree
   family; the feature-selection pitfalls that survive every one of
   these methods.

Roughly one chapter per sub-topic:

| # | Chapter | Objective |
|---|---|---|
| 01 | This chapter | orientation |
| 02 | [Linear and logistic regression](02-linear-and-logistic-regression.md) | 1 |
| 03 | [Regularisation: Ridge, Lasso, and elastic-net](03-regularisation-ridge-lasso-elasticnet.md) | 1 |
| 04 | [Decision trees](04-decision-trees.md) | 2 |
| 05 | [Random forests and bagging](05-random-forests-and-bagging.md) | 2 |
| 06 | [Gradient-boosted trees: XGBoost, LightGBM, CatBoost](06-gradient-boosted-trees.md) | 2 |
| 07 | [Cross-validation strategies](07-cross-validation-strategies.md) | 3 |
| 08 | [Hyperparameter search: grid, random, and Optuna](08-hyperparameter-search.md) | 3 |
| 09 | [Feature importance and selection pitfalls](09-feature-importance-and-selection-pitfalls.md) | 4 |
| 10 | [Model interpretation with SHAP](10-shap-and-model-interpretation.md) | 4 |

## What this module does *not* cover

- **Deep learning.** `mod-105` is the neural-network module — MLPs,
  CNNs, transformers, and the toolchain around them.
- **Unsupervised learning at depth.** Clustering, dimensionality
  reduction, matrix factorisation. We use PCA and simple clustering
  where they support supervised workflows but do not teach them as
  first-class topics.
- **Time-series forecasting as a specialty.** Time-ordered CV
  appears in chapter 07 because it is a CV discipline every
  practitioner needs; ARIMA / Prophet / TFT and the forecasting-
  specific vocabulary do not.
- **Full evaluation methodology.** The metrics themselves — ROC/PR
  curves, calibration, cost-sensitive evaluation, fairness slicing
  — are the topic of `mod-107`. This module reports scores; it does
  not deeply argue about which score.
- **Deployment and serving.** How to `joblib.load` a fitted
  `Pipeline` into a FastAPI endpoint, batch vs. online serving,
  latency budgets — all of that lives in `mod-108`.
- **Experiment tracking.** MLflow / Weights & Biases / DVC are
  `mod-106`. This module writes to `stdout` and to `REPORT.md`;
  next module writes to a tracker.
- **Feature engineering.** `mod-103`. We assume you have already
  produced `(X, y)`; we train on it.

## The bias–variance frame you will keep coming back to

Every algorithmic choice in this module is, at some level, a choice
about the bias–variance trade-off.

- A **linear model** has high bias (it assumes linearity) and low
  variance (small changes in the training data change the
  coefficients only a little). Regularisation trades a little more
  bias for less variance.
- A **single decision tree** grown to purity has low bias (it can
  fit any training set exactly) and enormous variance (change one
  row and the tree restructures). Pruning trades variance for bias.
- A **random forest** averages many high-variance trees whose errors
  partially cancel. Bias is roughly the same as a single deep tree;
  variance is much lower.
- A **gradient-boosted ensemble** builds trees sequentially, each
  fitting the previous ensemble's residuals. Both bias and variance
  can be controlled through learning rate, tree depth, and the
  number of boosting rounds, and the right hyperparameters usually
  put it at the current state of the art on tabular data.

Chapter 04 spells out the trade-off formally; the ensemble chapters
show how bagging and boosting move different terms.

## Where models sit in the ML lifecycle

To keep the modelling work honest, remember what it does *not*
handle. In particular:

- The **input distribution** is the domain of `mod-103` (feature
  engineering and leakage) and `mod-109` (drift monitoring). A
  perfectly tuned model on a leaky dataset is worse than an
  untuned model on a clean one.
- The **loss function and evaluation metric** decide what "good"
  means. `mod-107` covers when accuracy is a lie, when AUC is a
  distraction, and when calibration matters more than
  discrimination.
- The **serving code** is a separate artefact. `mod-108` covers the
  path from a fitted `Pipeline` to a served endpoint; every
  divergence between the two pipelines is a bug.
- The **monitoring layer** decides when the model has gone stale.
  `mod-109` builds the tripwires.

This module lives between `mod-103` and `mod-107`: given a clean
dataset, produce a model whose CV-reported performance is honest and
whose behaviour a human can inspect.

## How to read the module

The chapters are ordered so each builds on the last, but they are
not equally load-bearing:

- Chapters **02–03** (linear and regularisation) are the shortest
  and the most foundational. Read them even if you plan to reach
  straight for gradient boosting; the regularisation vocabulary
  transfers.
- Chapters **04–05** are prerequisites for chapter 06. If you
  already know how a decision tree works, chapter 04 is a fast
  skim.
- Chapter **06** (gradient-boosted trees) is the highest-return
  chapter in the module for tabular practitioners. Slow down.
- Chapters **07–08** are the CV and tuning discipline. Chapter 07
  is short and non-negotiable; chapter 08 is where most engineers
  add or subtract weeks of engineering time depending on how well
  they understand the search space.
- Chapters **09–10** are the interpretation arc. Chapter 09
  catalogues the failure modes of naive feature importance and
  selection; chapter 10 shows what SHAP actually does — enough that
  you can defend the numbers when a stakeholder asks.

The four exercises follow the same arc: fit a regularised linear
model, fit and compare tree ensembles, tune a gradient-boosting
model with Optuna, and explain the winner with SHAP.

## Summary

- Classical ML — linear models with regularisation, random forests,
  gradient-boosted trees — is still the pragmatic default for
  tabular supervised learning.
- Every choice in the module maps to a bias–variance move.
- The module lives between `mod-103` (data) and `mod-107`
  (evaluation): given a clean dataset, produce a model whose
  CV-reported performance is honest and whose behaviour is
  inspectable.
- Ten chapters, four objectives, four exercises. Chapters 06
  (gradient boosting), 07 (CV strategies), and 10 (SHAP) are the
  three you cannot skip.

## Where this goes next

Chapter 02 opens the linear-model arc with `LinearRegression` and
`LogisticRegression` — the interfaces you almost certainly already
know, viewed through the lens of loss functions, solvers, and
serving cost.
