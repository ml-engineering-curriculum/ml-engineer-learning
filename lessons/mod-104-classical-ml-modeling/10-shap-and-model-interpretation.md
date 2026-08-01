# 10 — Model interpretation with SHAP

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 06 (gradient boosting), 09
> (importance methods and their pitfalls).

## Motivation

Chapter 09 catalogued the ways that "feature importance" lies to
you: impurity bias toward high-cardinality features, permutation
importance diluted by correlations, filter methods that miss
interactions. SHAP is the current best-in-class answer to the
underlying question — *why did the model predict what it did?* — and
this chapter is about what SHAP actually computes, when it is
reliable, and how to defend the numbers to a stakeholder.

SHAP (SHapley Additive exPlanations) is a *unified* framework
because it recasts many earlier interpretation methods — LIME, DeepLIFT,
Layer-Wise Relevance Propagation, integrated gradients — as
approximations of the same underlying quantity: the Shapley value
from cooperative game theory. Lundberg and Lee (2017) introduced the
framework; the same authors' TreeSHAP paper (2020) gave the fast
exact algorithm for tree ensembles that made SHAP practical in
production.<!-- needs-research: cite Lundberg & Lee 2017 "A Unified
Approach to Interpreting Model Predictions" (NeurIPS,
https://arxiv.org/abs/1705.07874) and Lundberg et al. 2020 "From local
explanations to global understanding with explainable AI for trees"
(Nature Machine Intelligence, https://www.nature.com/articles/s42256-019-0138-9). -->

The `shap` Python library implements all of this. It is not
maintained by scikit-learn; it is a third-party dependency, but it
is the standard.

## Shapley values, in one page

Shapley values come from cooperative game theory (Shapley, 1953).
Given a set of `n` players and a value function `v(S)` — how much
value a coalition `S ⊆ {1, ..., n}` produces — the Shapley value of
player `j` is their fair share of the total. Fair means it is the
unique attribution that satisfies four axioms:

1. **Efficiency.** The Shapley values sum to `v({1, ..., n})`.
2. **Symmetry.** Interchangeable players get equal attribution.
3. **Dummy.** A player who contributes nothing to any coalition
   gets zero attribution.
4. **Additivity.** For two games, the Shapley values sum.

Formally:

```
φ_j = Σ_{S ⊆ N \ {j}}  (|S|! · (n − |S| − 1)! / n!) · [v(S ∪ {j}) − v(S)]
```

For ML, the "players" are features, and `v(S)` is the model's
prediction using only the features in `S`. The Shapley value `φ_j`
is the average marginal contribution of feature `j` across all
possible orderings of the features. Positive means "this feature
pushed the prediction up"; negative means "this feature pushed it
down".

The computational cost: `2^n` coalitions per prediction. For any
non-trivial `n`, that is impossible. TreeSHAP is the algorithm that
makes it tractable for tree ensembles by exploiting the tree
structure to compute exact Shapley values in polynomial time.

## What SHAP gives you

For a fitted model and an input `x`, SHAP returns a per-feature
attribution `φ_j(x)` such that:

```
model_prediction(x) = expected_prediction + Σ_j φ_j(x)
```

That is: the model's output equals its average output plus the sum
of feature contributions for this specific row.

For classification, "prediction" is the raw log-odds (for tree
ensembles) or the log-odds contribution. Convert with `sigmoid` to
get probability contributions.

## The three explainer choices

`shap.Explainer` autoselects; you can be explicit:

- **`shap.TreeExplainer(model)`** — fast, exact, for tree ensembles
  (XGBoost, LightGBM, CatBoost, sklearn's tree-based models). This
  is the one to use for anything in chapters 04–06.
- **`shap.LinearExplainer(model, X_background)`** — exact for linear
  models. Fast.
- **`shap.KernelExplainer(model.predict, X_background)`** — model-
  agnostic, samples `2^n` coalitions or approximates. Slow. Use
  when nothing else fits.
- **`shap.PermutationExplainer(...)`** — model-agnostic, uses
  permutation-based approximation. Faster than `KernelExplainer`
  and often the right model-agnostic default in modern `shap`.
- **`shap.DeepExplainer`** / **`shap.GradientExplainer`** — for
  neural networks. `mod-105` territory.

For a gradient-boosted classifier fit in chapter 06:

```python
import shap

explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer(X_sample)          # shap.Explanation object

# shap_values.values.shape == (n_samples, n_features)
# shap_values.base_values          -> the expected prediction
# shap_values.data                 -> the corresponding X_sample values
```

The result is a `shap.Explanation` object that holds the values, the
base value, the raw features, and the feature names — enough to
render any of the built-in plots.

## The five plots you will actually use

### 1. Waterfall plot: one prediction, decomposed

```python
shap.plots.waterfall(shap_values[0])
```

Shows how each feature moves the prediction from the base value
(the model's average) to the actual prediction for one row. The
best single visualisation for "why did the model say what it said
about this specific case?"

Use when a stakeholder points at a specific customer / patient /
transaction and asks why.

### 2. Force plot: waterfall, compact form

```python
shap.plots.force(shap_values[0])
```

Same information as a waterfall, packed into a horizontal ribbon.
Useful for embedding in a report.

### 3. Beeswarm plot: global importance + direction + distribution

```python
shap.plots.beeswarm(shap_values)
```

For each feature (Y-axis), plot each row's SHAP value (X-axis)
coloured by the feature's raw value. Features are ordered top-to-
bottom by mean absolute SHAP.

Reads like: "high values of `age` (red dots on the right) push
predictions up; low values (blue dots on the left) push down —
strong monotone effect. `country` shows a broad spread with
no clear direction — importance without monotonicity."

This is the plot to include in the `REPORT.md`.

### 4. Bar plot: global importance summary

```python
shap.plots.bar(shap_values)
```

Mean absolute SHAP per feature, sorted. The `feature_importances_`
equivalent, but based on out-of-training predictions and unbiased
by cardinality. This is the number you actually want when someone
asks "top 10 features".

### 5. Dependence plot: one feature's effect

```python
shap.plots.scatter(shap_values[:, "age"], color=shap_values[:, "income"])
```

For a single feature, plot its SHAP value against its raw value.
Colouring by another feature reveals interactions:

- A monotone red-then-blue pattern along X: `age` has a large
  interaction with `income` — high income amplifies the `age` effect.
- No colour pattern: the effect of `age` is independent of `income`.

Dependence plots are how you find nonlinearities and interactions
worth naming in your report.

## Aggregating SHAP to global importance

The right global importance is `mean(|φ_j(x)|)` over a representative
sample:

```python
import numpy as np
import pandas as pd

global_importance = pd.Series(
    np.abs(shap_values.values).mean(axis=0),
    index=X.columns,
).sort_values(ascending=False)
```

Compare this to the impurity-based `feature_importances_` from
chapter 09. The two typically agree on the top features, disagree
on the middle, and disagree strongly on features that look
important to impurity because of high cardinality. Trust the SHAP
number.

## Speed and background sets

For TreeSHAP on tree ensembles: exact and fast, no background set
needed (or use a small representative sample as `X_sample` for the
explanations themselves).

For `KernelExplainer` / `PermutationExplainer` on any model:

- **The background set** is the reference distribution against
  which "feature not present" is defined (typically the mean or
  median of that feature). Small representative background: 50–500
  rows. Too large = slow; too small = noisy.
- **`shap.sample(X, n)` or `shap.kmeans(X, n)`** — utilities to
  build a representative background.

```python
background = shap.sample(X_train, 200, random_state=0)
explainer = shap.KernelExplainer(model.predict_proba, background)
shap_values = explainer.shap_values(X_test.iloc[:100])
```

Expect kernel SHAP to take seconds per prediction on medium-sized
models. Not suitable for scoring millions of rows.

## Interventional vs. tree-path SHAP

TreeSHAP has two variants, differing in how they handle correlated
features:

- **`feature_perturbation="tree_path_dependent"`** (default) — uses
  the tree structure to compute Shapley values *conditioning on the
  observed tree paths*. Fast, no background dataset needed. When
  features are correlated, some of the attribution can be credited
  to features the tree did not actually split on but that are
  correlated with ones it did.
- **`feature_perturbation="interventional"`** — uses a background
  dataset to define "feature absent" via marginalisation. Slower,
  requires a background sample, more faithful to what the model
  actually uses on truly unseen combinations of features.

For most practical use, the default is fine. When you have highly
correlated features and interpretation matters, `interventional`
with a well-chosen background is more defensible.

```python
explainer = shap.TreeExplainer(
    model,
    data=shap.sample(X_train, 200, random_state=0),
    feature_perturbation="interventional",
)
```

## SHAP on categorical / one-hot features

If categoricals are one-hot encoded, each one-hot column gets its
own SHAP value. That is technically correct but confusing to read
("`country_US` = +0.02, `country_UK` = -0.01, `country_FR` = 0"
when only one is 1). Two options:

1. **Sum SHAP values across the one-hot columns per category.**
   `shap_values_country = shap_values[:, country_cols].values.sum(axis=1)`.
   The sum is the SHAP value for the *original* categorical.
2. **Use a model with native categorical handling** (LightGBM,
   CatBoost, `HistGradientBoosting`) so SHAP operates on a single
   categorical column per feature.

The second option is cleaner and matches the model's actual
decision structure.

## Interpreting SHAP for classification

For a binary classifier, TreeSHAP returns log-odds contributions.
Convert to probability changes by centring on the base value and
applying `sigmoid`:

```python
import numpy as np
p_base   = 1 / (1 + np.exp(-shap_values.base_values[0]))
p_actual = 1 / (1 + np.exp(-(shap_values.base_values[0] + shap_values.values[0].sum())))
```

Or use `shap.explainers.Tree(model, model_output="probability")` to
get probability-scale SHAP values directly (slower; requires
background data).

## The pitfalls that SHAP does not solve

SHAP is more rigorous than the alternatives, but it is not magic.

1. **Correlated features still share credit.** Shapley values
   distribute the joint credit "fairly" (per the axioms), but two
   perfectly correlated features will each get about half the
   credit even if only one is causal. SHAP tells you the model's
   dependence structure, not the world's.
2. **SHAP explains the model, not the ground truth.** If the model
   is wrong, the SHAP explanation of it is a faithful explanation of
   a wrong model. Do not confuse "the model predicts churn because
   of `days_since_last_login`" with "the customer will churn
   because of `days_since_last_login`".
3. **SHAP is expensive on non-tree models.** `KernelExplainer` on a
   1M-row test set is not going to run in your evening. For online
   serving of explanations at scale, use TreeSHAP (or precompute).
4. **SHAP + heavily-leaky features = beautiful attribution to
   nonsense.** SHAP will happily attribute the model's near-perfect
   accuracy to the leaking feature. That is the correct
   explanation. It is also a diagnostic — a feature that
   dominates the SHAP summary should be checked for leakage.
5. **Individual SHAP values are noisy for small samples.** Global
   importance (mean absolute over many rows) is stable; a single
   row's decomposition can swing based on the local approximation.
   Report a distribution, not a point value, when you can.

## SHAP inside a `Pipeline`

The one place SHAP is fussier than the rest of the module: the
`Explainer` needs to see the *model*, not the entire `Pipeline`.
Pull the pieces apart:

```python
from sklearn.pipeline import Pipeline

pipe.fit(X_train, y_train)
preprocessor = pipe[:-1]
model        = pipe.named_steps["clf"]

X_train_processed = preprocessor.transform(X_train)
X_test_processed  = preprocessor.transform(X_test.iloc[:100])

# Preserve feature names for the plots
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_processed)
shap_values.feature_names = list(feature_names)

shap.plots.beeswarm(shap_values)
```

Since `shap` 0.42, `shap.Explainer(pipe, background)` handles some
pipeline cases end-to-end, but for tree ensembles the manual
`preprocessor.transform(...)` + `TreeExplainer` pattern is the
most reliable.

## A complete example: fit → explain → decide

```python
import shap
from lightgbm import LGBMClassifier, early_stopping
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
y = (y == ">50K").astype(int)
for c in X.select_dtypes(include="object").columns:
    X[c] = X[c].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0,
)

model = LGBMClassifier(
    n_estimators=2000, learning_rate=0.05,
    num_leaves=63, random_state=0, n_jobs=-1, verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stopping(50)],
)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test.sample(2000, random_state=0))

# Global: bar and beeswarm
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)

# Local: one row's decomposition
shap.plots.waterfall(shap_values[0])

# Interaction: age vs. capital-gain
shap.plots.scatter(shap_values[:, "age"], color=shap_values[:, "capital-gain"])
```

Reading this output, in order:

1. Bar plot — which features dominate the model on average.
2. Beeswarm — direction and magnitude per feature.
3. Waterfall — how a specific row's prediction was built.
4. Dependence — non-linearities and interactions to flag.

Everything else in `shap.plots` is a variation on these four.

## When SHAP is worth the effort

SHAP is not free (`shap` is a new dependency; computing values
takes wall-clock; plots consume screen space). Fits it earns its
keep on:

- **Regulated decisions.** Credit scoring, insurance pricing,
  medical decision support. Regulators explicitly ask "why did the
  model reject this application?" and SHAP produces a defensible
  answer.
- **Debugging.** A model that scores well on CV but does something
  weird on a specific segment — SHAP on that segment usually
  surfaces the reason.
- **Feature-set decisions.** Global importance ranks features by
  actual model use, not by impurity bias. Use it to argue for
  dropping features or investing in better ones.
- **Stakeholder communication.** A waterfall plot for a single
  decision beats a paragraph of prose about "the model considered
  age and income."

Where you can skip SHAP:

- **Pure prediction problems with no interpretability requirement.**
  Score, deploy, monitor. No one is going to look at the SHAP
  plots.
- **Very cost-sensitive inference.** SHAP at inference time is
  expensive; if you need per-request explanations at high QPS,
  precompute or move to a linear model with coefficient-based
  explanation.

## Pitfalls collected

1. **Passing a `Pipeline` to `TreeExplainer`.** Depends on version;
   often silently wrong. Extract the model and preprocessed data,
   then explain.
2. **Explaining on the training set.** Reflects fitted-model
   internals, not generalisation. Explain on a held-out sample.
3. **Interpreting one-hot columns individually.** Sum them per
   original categorical, or use native categoricals.
4. **Treating SHAP as causal.** SHAP is an explanation of what the
   model does, not of what causes the target.
5. **`KernelExplainer` on a large dataset.** It will finish
   eventually. Prefer `TreeExplainer` for trees, `LinearExplainer`
   for linear, `PermutationExplainer` for other.
6. **Comparing SHAP values across models with different base
   values.** SHAP values are additive to a specific `base_value`;
   two different models have different base values. Compare
   *rankings* or normalised SHAP, not raw magnitudes.
7. **Reading a beeswarm coloured by a feature with a very
   non-monotone effect** and concluding "there is no interaction."
   Dependence plots (`shap.plots.scatter`) are better for
   confirming interactions.

## Summary

- SHAP is the current standard for model interpretation. Shapley
  values from cooperative game theory give each feature its fair
  share of the prediction, satisfying the axioms (efficiency,
  symmetry, dummy, additivity).
- TreeSHAP makes exact SHAP fast for tree ensembles — this is the
  95%-of-cases explainer for tabular ML.
- Use `beeswarm` for global direction+magnitude, `bar` for global
  ranking, `waterfall` for individual decisions, and `scatter`
  (dependence) for interactions.
- Mean absolute SHAP is the unbiased alternative to
  `feature_importances_` — use it in your final reporting.
- SHAP does not solve correlated features, does not explain
  causality, and does not rescue you from a leaky model. It
  faithfully explains the model as trained, no more.

## Where this goes next

You have finished the modelling arc: linear and regularised, tree
and ensemble, cross-validated, tuned, and interpreted. `mod-107`
takes up model evaluation and validation at greater depth — the
metrics, calibration, error analysis, and the discipline of proving
that a model is genuinely better than a baseline before shipping it.
