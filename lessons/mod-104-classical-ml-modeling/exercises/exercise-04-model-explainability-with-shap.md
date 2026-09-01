# exercise-04 — Model explainability with SHAP

> **Estimated effort:** 3 hours.
> **Companion chapters:** [09 — Feature importance and selection pitfalls](../09-feature-importance-and-selection-pitfalls.md), [10 — Model interpretation with SHAP](../10-shap-and-model-interpretation.md).

## Objective

Take the tuned gradient-booster from exercise-03 (or a fresh one
you fit here) and produce an explanation report a stakeholder could
read. You will:

- Compute SHAP values for a held-out sample with `TreeExplainer`.
- Compare SHAP-based global importance against impurity-based
  `feature_importances_` and permutation importance.
- Read the four canonical SHAP plots — beeswarm, bar, waterfall,
  dependence — and write a one-sentence interpretation of each.
- Explain three individual predictions (one confident-correct, one
  confident-wrong, one near the decision boundary) with waterfall
  plots.
- Sum SHAP values across one-hot / categorical columns so each
  reported number corresponds to an *original* feature.
- Use SHAP as a leakage diagnostic: any feature that dominates the
  summary is a suspect until proven innocent.

The goal is not "SHAP ran". The goal is a `REPORT.md` that answers,
in plain language, *which features the model uses, in which
direction, and why the top prediction on three specific rows came
out the way it did* — with numbers a reader can verify.

## Prerequisites

- Chapters 09 and 10 of this module.
- Exercise-03 (a tuned XGBoost or LightGBM model) — reuse the
  fitted model and CV split if you have them.
- Python 3.11+, scikit-learn ≥ 1.4, Pandas ≥ 2.1, `matplotlib`,
  `pytest`, `shap` ≥ 0.44.
- One of `xgboost` ≥ 2.0 or `lightgbm` ≥ 4.0.

## Dataset

Use one of:

- **UCI Adult** (`sklearn.datasets.fetch_openml("adult", version=2,
  as_frame=True)`) — the default; matches exercises 02 and 03. The
  categorical mix makes the "SHAP on one-hot vs. native categoricals"
  point land.
- **Your model from exercise-03** — if you already have a tuned
  gradient booster on a binary or multiclass dataset, reuse it.
- **Credit-default** or **Bank Marketing** from OpenML — either
  works.

Pick one and stick with it.

## Problem statement

### Part A — set up the model and sample (≈ 20 min)

If you are continuing from exercise-03, load the saved model and the
train / test split. Otherwise, fit a `LGBMClassifier` or
`XGBClassifier` on the dataset with reasonable defaults (e.g. the
tuned config from your Optuna study, or `learning_rate=0.05`,
`n_estimators=1000`, early stopping on a validation split).

Draw a **fixed** SHAP sample from `X_test`:

```python
X_shap = X_test.sample(2000, random_state=0)
y_shap = y_test.loc[X_shap.index]
```

Reuse this sample everywhere so plot-to-plot comparisons are
apples-to-apples. Reserve the rest of `X_test` for the honest
accuracy numbers in `REPORT.md`.

**In `REPORT.md`:** state the model (library, best hyperparameters,
number of boosting iterations actually used), the test AUC on the
full `X_test`, and the size of the SHAP sample.

### Part B — global explanations (≈ 40 min)

Build a `TreeExplainer` on your fitted model and compute SHAP
values on `X_shap`:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_shap)          # shap.Explanation object
```

Produce these three plots and save to `plots/`:

1. `shap.plots.bar(shap_values, max_display=15)` — global importance,
   mean absolute SHAP per feature.
2. `shap.plots.beeswarm(shap_values, max_display=15)` — direction
   and magnitude per feature.
3. `shap.plots.scatter(shap_values[:, top_feature], color=shap_values[:, second_feature])`
   for the top-two features by mean absolute SHAP — a dependence
   plot revealing non-linearity and interactions.

**In `REPORT.md`:**

- Paste each plot with a one-sentence caption. "The plot shows
  global importance" does not count; something like "`age` is the
  top feature; high `age` (red) pushes predictions up almost
  monotonically; `education-num` has similar magnitude but a wider
  spread suggesting interaction effects" does.
- List the top 10 features by mean absolute SHAP as a table
  (feature, mean |SHAP|).

### Part C — three importance methods, side by side (≈ 30 min)

Compare, on the same held-out sample, three importance rankings:

1. `model.feature_importances_` — impurity-based (fast, biased —
   chapter 09).
2. `sklearn.inspection.permutation_importance(model, X_shap, y_shap,
   n_repeats=20, random_state=0, scoring="roc_auc")` — hold-out
   permutation importance.
3. `np.abs(shap_values.values).mean(axis=0)` — mean absolute SHAP.

Build a comparison DataFrame:

```
                  impurity   perm_mean   shap_mean   imp_rank   perm_rank   shap_rank
feature_A            0.180       0.032       0.412           1           3           1
feature_B            0.045       0.061       0.297           7           1           2
...
```

**In `REPORT.md`:**

- The top-15 rows of this table.
- Which features have the largest rank disagreements. A feature
  ranked top by impurity but low by SHAP / permutation is usually
  a high-cardinality artefact (chapter 09) — name any you find.
- A `pandas.DataFrame.corr(method="spearman")` heatmap of the three
  rank columns. Ideally SHAP and permutation agree strongly (~0.9+)
  and both disagree at least mildly with impurity.

### Part D — individual explanations (≈ 40 min)

Pick three rows from `X_shap` and generate a waterfall plot for
each:

1. **A confident-correct positive prediction.** `y_shap == 1` and
   `model.predict_proba(row)[:, 1] > 0.9`.
2. **A confident-wrong negative prediction.** `y_shap == 0` but
   `model.predict_proba(row)[:, 1] > 0.9` (a false positive with
   high confidence).
3. **A near-boundary prediction.** `0.45 < proba < 0.55`.

For each:

```python
i = <chosen index>
shap.plots.waterfall(shap_values[i])
```

Save to `plots/waterfall_*.png`.

**In `REPORT.md`, per row:**

- The predicted probability and the true label.
- The top 3 features pushing the prediction toward positive
  (features with the largest positive SHAP contribution).
- The top 3 features pushing toward negative.
- A one-paragraph plain-English narrative. "The model predicted a
  92% probability of `>50K` because `capital-gain=15024` (SHAP
  +2.3) and `education-num=13` (SHAP +0.8) both push strongly
  upward; `age=24` (SHAP -0.4) pushes down modestly but does not
  overcome the other two."

The **confident-wrong** case is the interesting one. A stakeholder
will ask about false positives; your report should show one.

### Part E — categorical / one-hot aggregation (≈ 20 min)

If your model uses native categoricals (LightGBM with `category`
dtype, or `HistGradientBoosting` with `categorical_features="from_dtype"`),
each categorical feature already has one SHAP value per row — skip
to the second half of this part.

If your preprocessing one-hot-encoded the categoricals, each
one-hot column has its own SHAP value. Aggregate: for each original
categorical, sum the SHAP contributions across its one-hot columns
per row.

```python
country_cols = [c for c in X_shap.columns if c.startswith("country_")]
country_shap = shap_values.values[:, [X_shap.columns.get_loc(c) for c in country_cols]].sum(axis=1)
```

The sum is the SHAP value for the *original* `country` feature on
each row.

**In `REPORT.md`:**

- The updated global-importance ranking after aggregation — how does
  the top-10 shift when categorical contributions are summed rather
  than reported per one-hot column?
- If you have native-categorical support, re-run TreeSHAP on the
  native model and paste both bar plots side by side. Discuss any
  differences.

### Part F — SHAP as a leakage smoke test (≈ 20 min)

The chapter-10 pitfall: a leaky feature will dominate the SHAP
summary because it dominates the model. Turn that into an active
check.

1. Sort features by mean absolute SHAP.
2. For the top-3 features, answer in `REPORT.md`:
   - Is this feature available at prediction time in production,
     with the same distribution as in training? (If the dataset is
     synthetic like Adult, answer "yes, this is a static UCI
     dataset" and move on — but state it explicitly.)
   - Is any part of the label leaking through this feature (post-
     outcome data, time-of-outcome timestamps, aggregations
     computed after the target event)?
3. **Add a deliberately-leaky feature** to a *copy* of the training
   set (e.g. `X_train["leak"] = y_train + np.random.normal(0, 0.01,
   size=len(y_train))`) and retrain. Compute SHAP on the leaky
   model. Confirm that `leak` dominates the beeswarm and every
   other feature is pushed toward zero. Paste the plot.

The point is not that leakage is bad (you knew that). The point is
that SHAP will *cheerfully* explain the leaky model as if the
prediction were legitimate — the beautiful attribution to `leak` is
the diagnostic. Any time a single feature dominates a SHAP summary
by an order of magnitude over the runner-up, treat it as a leakage
suspect until you have ruled it out.

## Starter guidance

- **Pass the model, not the `Pipeline`, to `TreeExplainer`.** If
  your pipeline has preprocessing steps, transform the data first
  and pass the transformed matrix. Chapter 10 has the pattern.
- **Explain on held-out data, not on the training set.** SHAP on
  training-set rows describes the fitted model's internals, not
  its generalisation behaviour.
- **Fix `random_state=0` on the `X_shap` sample.** Different samples
  give slightly different importance rankings; a reproducible
  ordering is worth the two seconds it costs.
- **`shap.plots.beeswarm` is slow on > 5000 rows.** 2000 is the
  sweet spot: enough for stable rankings, fast enough to iterate.
- **For multiclass, `shap_values` has an extra class axis.** Slice
  the class you care about (`shap_values[..., 1]` for the positive
  class in binary; index the specific class for multiclass).
- **Log-odds vs. probability.** TreeSHAP on classifiers returns
  log-odds contributions by default. That is fine for global
  ranking and waterfall plots, but if a stakeholder asks "how much
  did age change the probability?", either convert manually with
  `sigmoid` (chapter 10) or fit with `model_output="probability"`
  and a background dataset.
- **`shap.plots.scatter(shap_values[:, "col"], color=shap_values[:, "other"])`.**
  The `color=` argument is where interactions become visible; do
  not skip it.
- **When plots look wrong, check `shap_values.feature_names`.** If
  it is `None`, your global plots will label features by index.
  Set it manually from `X_shap.columns`.

## Acceptance criteria

- A fitted gradient-booster (from exercise-03 or freshly trained)
  and a fixed 2000-row SHAP sample from `X_test`.
- SHAP values computed with `TreeExplainer` and saved to disk
  (`shap_values.pkl` or an equivalent).
- `plots/` contains: `bar.png`, `beeswarm.png`, `dependence_*.png`,
  three `waterfall_*.png` (confident-correct, confident-wrong,
  boundary), and `beeswarm_leaky.png` from Part F.
- The three-method comparison table (impurity / permutation / SHAP)
  is in `REPORT.md`, with rank-correlation heatmap and a discussion
  of the biggest disagreements.
- Three individual predictions each have a plain-English narrative
  (predicted probability, true label, top-3 up-pushers, top-3
  down-pushers) in `REPORT.md`.
- Categorical SHAP is aggregated to the original feature level (by
  summing one-hot contributions, or by using native categoricals),
  and the aggregated ranking is reported.
- Leakage smoke test: the leaky-feature model's SHAP summary is
  shown; a paragraph explains what the plot proves.
- `REPORT.md` opens with a one-paragraph plain-language summary of
  what the model does — the kind of paragraph you would put at the
  top of a decision memo.
- No `fit` call touches the held-out `X_test` after Part A; the
  SHAP sample is drawn once and never re-drawn.

## Stretch goals (optional)

- **Cohort SHAP.** Split `X_shap` by a demographic slice (e.g.
  `sex` on Adult) and compute mean absolute SHAP per feature per
  cohort. Are the top features the same? If they differ
  substantially, the model uses different signals for different
  groups — usually worth naming in a fairness review.
- **Interaction values.** `explainer(X_shap, interactions=True)`
  or `explainer.shap_interaction_values(X_shap)` returns a
  `(n, n_features, n_features)` array. Plot the top-5 interactions
  as heat maps; do they match the interactions you inferred from
  the dependence plots?
- **Kernel vs. Tree SHAP.** On a 200-row sample, compute SHAP with
  `KernelExplainer` as well and correlate the two rankings. They
  should agree closely; a large disagreement suggests numerical
  issues (usually background-set size).
- **Interventional TreeSHAP with a background sample.** Re-run
  Part B with `feature_perturbation="interventional"` and a
  200-row background. Do the top features change? On correlated
  features, they usually do.
- **A LinearExplainer sanity check.** Fit a `LogisticRegression`
  as a baseline; explain with `shap.LinearExplainer` and compare
  the SHAP-derived importance ranking against the standardised
  coefficient magnitudes. They should agree exactly (up to sign).
- **Serving-time SHAP.** Time `explainer(one_row)` versus
  `model.predict_proba(one_row)`. If SHAP is 100× slower than
  prediction, discuss whether real-time explanations are feasible
  in your context or whether precomputation / batching is needed.

## What to hand in

```
exercise-04/
├── model.py              # load / fit the gradient booster
├── explain.py            # TreeExplainer + shap_values on X_shap
├── compare.py            # impurity vs. permutation vs. SHAP table
├── individuals.py        # waterfall plots for the three chosen rows
├── categorical.py        # one-hot aggregation (or native-cat path)
├── leakage_check.py      # deliberately-leaky model + SHAP summary
├── tests/
│   ├── test_shap_shape.py   # shap_values.values.shape matches X_shap
│   └── test_ranking.py       # SHAP + permutation rankings correlate > 0.7
├── plots/                # PNG output for REPORT.md
├── shap_values.pkl       # persisted so REPORT reruns are fast
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
