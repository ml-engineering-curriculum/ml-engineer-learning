# exercise-01 — Regression with regularisation

> **Estimated effort:** 3 hours.
> **Companion chapters:** [02 — Linear and logistic regression](../02-linear-and-logistic-regression.md), [03 — Regularisation: Ridge, Lasso, and elastic-net](../03-regularisation-ridge-lasso-elasticnet.md).

## Objective

Fit and compare four linear models on a regression problem: an
unregularised baseline (`LinearRegression`), Ridge, Lasso, and
elastic-net. Along the way you will:

- Wire every model inside a `Pipeline` with a `StandardScaler` and
  demonstrate why (chapter 03).
- Use the `*CV` variants (`RidgeCV`, `LassoCV`, `ElasticNetCV`) to
  pick regularisation strength efficiently.
- Compare the four models honestly on a held-out test set and on
  5-fold cross-validation.
- Bootstrap Lasso to measure *selection stability* — which features
  survive across resamples, not just in one fit.
- Report the regularisation path — how each feature's coefficient
  moves as `alpha` grows.

The goal is not the last percentage point of R². It is a set of
models whose CV numbers you can defend, and a stability report you
would be willing to hand to a stakeholder who cares which features
your Lasso selected.

## Prerequisites

- Chapters 02 and 03 of this module.
- `mod-102` chapter 06 (Pipelines) and `mod-103` chapter 06
  (leakage rules).
- Python 3.11+, scikit-learn ≥ 1.4, Pandas ≥ 2.1, `matplotlib`,
  `pytest`.

## Dataset

Use one of:

- **California Housing** (`sklearn.datasets.fetch_california_housing`)
  — the default. ~20k rows, 8 numeric features, continuous target.
  Well-suited to Ridge (dense numerics, mild correlation).
- **Ames Housing** (via `sklearn.datasets.fetch_openml("house_prices",
  as_frame=True)` or the `Kaggle` original) — larger, mixed types,
  more interesting for Lasso (many features, some likely useless).
  Requires a bit of preprocessing.
- **Diabetes** (`sklearn.datasets.load_diabetes`) — small (~442 rows),
  10 features, useful if you want the whole exercise to run in
  seconds.

Pick one and stick with it. The exercise generalises.

## Problem statement

### Part A — split and inspect (≈ 20 min)

In `data.py`, load the dataset and split into
`(X_train, X_test, y_train, y_test)` using `train_test_split(
test_size=0.2, random_state=0)`. Save the split so subsequent
steps use the same one.

Write a short `describe.py` that prints:

- Shape of `X_train`, `X_test`.
- The 5-number summary of `y`.
- The pairwise correlation matrix of `X` (`X.corr()`).
- The condition number of `Xᵀ X` (`np.linalg.cond(X.T @ X)` — a
  proxy for how ill-conditioned the OLS problem is).

Put the correlation heatmap and condition number in `REPORT.md`.
Note any pairs with `|corr| > 0.7`; those are the features Ridge
will handle better than OLS and Lasso will pick between arbitrarily.

### Part B — unregularised baseline (≈ 20 min)

In `pipeline.py`, build `build_ols_pipeline() -> Pipeline`:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("ols",   LinearRegression()),
])
```

Fit on `X_train`, report `R²` and `RMSE` on both the training set
and `X_test`. Report `cross_val_score` mean and std (5-fold).

In `REPORT.md`:

- The four numbers (train R², test R², CV R² mean, CV R² std).
- The coefficient vector, sorted by `|β|` descending.
- Answer: is there a large gap between train and test? Which
  feature has the largest coefficient?

### Part C — Ridge (≈ 30 min)

Build `build_ridge_pipeline(alpha)` and, more usefully,
`build_ridgecv_pipeline(alphas)` using `RidgeCV`:

```python
alphas = np.logspace(-4, 4, 25)
Pipeline([
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=alphas, cv=5)),
])
```

Fit on `X_train`. Report the selected `alpha_`, R², and RMSE on both
sets and via 5-fold CV.

**In `REPORT.md`:**

- Selected `alpha_`.
- Compare Ridge's test R² and coefficient stability to OLS. On a
  dataset with correlated features, Ridge's coefficients should be
  smaller and more sensible.
- Plot the regularisation path (coefficient magnitude vs. `alpha`)
  and paste the plot. Which coefficients get shrunk fastest?

Hint: `sklearn.linear_model.Ridge(alphas=...)` can be swept over a
list by fitting one at each and stacking `coef_` vectors.
`sklearn.linear_model.ridge_regression` exposes a lower-level call
if you want it.

### Part D — Lasso (≈ 40 min)

Build `build_lassocv_pipeline`:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("lasso", LassoCV(cv=5, max_iter=10_000, random_state=0)),
])
```

Fit and report as above. Additionally, report:

- The **number of non-zero features** at the selected `alpha_`.
- The list of surviving features and their coefficients.

**Plot the Lasso path** using `sklearn.linear_model.lasso_path`:

```python
alphas, coefs, _ = lasso_path(X_scaled, y, alphas=np.logspace(-3, 1, 50))
# coefs.shape == (n_features, len(alphas))
```

Paste the plot in `REPORT.md`. Which features die first as `alpha`
grows? Which persist to the largest `alpha`?

### Part E — selection stability (≈ 40 min)

The point of the exercise. Lasso selects features, but the
selection is unstable under correlated features. Measure how
unstable.

Write `stability.py` that:

1. Draws 200 bootstrap resamples of `X_train, y_train` (with
   replacement) using `sklearn.utils.resample`.
2. On each, fits the `LassoCV` pipeline.
3. Records the boolean vector "coefficient non-zero" per feature.
4. Reports, per feature, the **fraction of bootstraps in which it
   was selected**.

Output a DataFrame like:

```
              selection_freq   mean_coef   std_coef
feature_A            1.00        0.482      0.031
feature_B            0.97       -0.153      0.028
feature_C            0.54        0.041      0.038   <-- unstable
feature_D            0.12        0.002      0.020   <-- almost never selected
...
```

**In `REPORT.md`:**

- The full stability table for the top 15 features by frequency.
- Which features are selected in ≥ 80% of resamples ("stable
  selection").
- Which features are between 40% and 60% (coin-flip stable) —
  flag them as unreliable.
- If two features are highly correlated and Lasso alternates
  between them, name the pair.

### Part F — elastic-net (≈ 20 min)

Repeat Part D and Part E with `ElasticNetCV`:

```python
ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    cv=5,
    max_iter=10_000,
    random_state=0,
)
```

Report:

- Selected `alpha_` and `l1_ratio_`.
- CV R², test R².
- Number of non-zero features vs. Lasso.
- Stability comparison: are elastic-net's selections more stable
  than Lasso's? (They usually are.)

### Part G — the comparison table (≈ 20 min)

Assemble the four models' results in a single table in `REPORT.md`:

| Model | CV R² | Test R² | Test RMSE | # non-zero coef | Notes |
|---|---|---|---|---|---|
| OLS | | | | (all) | |
| Ridge | | | | (all) | |
| Lasso | | | | | |
| Elastic-net | | | | | |

Answer in prose (a paragraph each):

1. Which model would you ship, and why?
2. Which model would you use to *explain* the problem to a
   stakeholder, and why?
3. If a colleague reports "the important features are A, B, C" from
   a single Lasso fit, what would you say based on your stability
   analysis?

## Starter guidance

- **Scaling is not optional for regularised models.** The `Pipeline`
  is what makes it correct — the scaler refits per CV fold. If you
  scale outside the pipeline, your CV numbers are slightly
  optimistic. Chapter 03 has the argument at length.
- **Use `numpy.logspace(-4, 4, 25)` (or a similar log grid) for
  `alpha` searches.** Log-scale is essential; linear grids miss the
  interesting range.
- **`LassoCV` sometimes warns about not converging.** Bump
  `max_iter` to `20_000`. Or, if convergence is genuinely stuck,
  the input scale may be off.
- **The bootstrap in Part E should use `resample(X_train, y_train,
  random_state=None)`.** Different `random_state` per iteration
  (i.e. omit it). Otherwise every bootstrap is identical.
- **For the regularisation-path plot, plot on a `logx` axis.** The
  interesting behaviour is in orders of magnitude.

## Acceptance criteria

- All four `Pipeline` objects (OLS, Ridge, Lasso, elastic-net) fit
  on `X_train` and score on `X_test`. Any preprocessing (scaling)
  is inside the pipeline.
- `RidgeCV`, `LassoCV`, `ElasticNetCV` are used to select `alpha`
  (and `l1_ratio` for elastic-net) via 5-fold CV; the selected
  values are reported.
- The regularisation paths for Ridge and Lasso are plotted and
  included in `REPORT.md`.
- The Lasso stability analysis (200 bootstraps) is complete;
  selection frequencies for each feature are in `REPORT.md`.
- Elastic-net stability is reported alongside Lasso's; a
  comparison sentence explains the difference.
- The four-model comparison table is filled in.
- `REPORT.md` answers the three prose questions in Part G.
- No `fit` call touches `X_test`. `git grep X_test` shows only
  `score`, `predict`, and print statements.

## Stretch goals (optional)

- **`RidgeClassifier` vs. `LogisticRegression(penalty="l2")`.** On
  a binary classification version of your dataset (or the
  `sklearn.datasets.load_breast_cancer` set), fit both. Compare
  hinge-loss vs. log-loss regularised solutions.
- **Coefficient CIs via bootstrap.** For your best Ridge, report a
  95% bootstrap CI for each coefficient. Which coefficients have
  intervals that exclude zero? How does that ranking compare to
  the "selected" features from Lasso?
- **`ElasticNetCV` with an `l1_ratio` sweep, plotted as a heat map
  of (alpha, l1_ratio) → CV score.** The heat map shows the whole
  regularisation surface at once.
- **Nested CV for the "final" model.** Wrap `LassoCV` inside a
  `cross_val_score(..., cv=outer)` and report the nested CV
  estimate. Compare to `LassoCV`'s reported best score. The gap is
  the winner's curse.

## What to hand in

```
exercise-01/
├── data.py               # load() + split; returns (X_train, X_test, y_train, y_test)
├── describe.py           # correlations, condition number
├── pipeline.py           # build_*_pipeline() for OLS, Ridge, Lasso, elastic-net
├── stability.py          # bootstrap Lasso / elastic-net for selection frequency
├── paths.py              # regularisation-path plots
├── tests/
│   ├── test_pipeline.py  # each pipeline fits and scores without warnings
│   └── test_stability.py # selection frequencies sum to a plausible range
├── plots/                # PNG output for REPORT.md
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
