# 06 — Gradient-boosted trees: XGBoost, LightGBM, CatBoost

> **Reading time:** ~50 minutes.
> **Prerequisite chapters:** 04, 05.

## Motivation

If you have to bet on which single model family will win on a new
tabular dataset today, bet on gradient-boosted trees. Multi-year
benchmarks and Kaggle post-mortems from the tabular era through the
present continue to show gradient boosting — XGBoost, LightGBM,
CatBoost, or scikit-learn's `HistGradientBoosting*` — at or near the
top of the leaderboard for structured supervised learning.<!-- needs-research:
verify with the Grinsztajn et al. 2022 paper
(https://arxiv.org/abs/2207.08815) "Why do tree-based models still
outperform deep learning on tabular data?" and the Kaggle "State of
Machine Learning and Data Science" survey — cite the most recent
year's numbers. -->

This chapter covers the algorithm (gradient boosting), the four
implementations you will actually use, and the practical decisions —
tree depth, learning rate, early stopping, categorical handling —
that decide whether you get their real accuracy or a mediocre first
draft.

## Boosting: the idea

Bagging trains many models independently and averages them. Boosting
does the opposite:

1. Start with a constant model (mean of `y` for regression, class
   log-odds for classification).
2. Compute the residuals (or, more generally, the negative gradient
   of the loss) at every training point.
3. Fit a small tree to predict those residuals.
4. Add a fraction (`learning_rate`) of the new tree's prediction to
   the ensemble.
5. Repeat.

Formally, at iteration `m`, the current model is `F_{m-1}(x)`. Fit
a tree `h_m` to the negative gradient of the loss w.r.t.
`F_{m-1}(x)`, and update `F_m(x) = F_{m-1}(x) + η · h_m(x)`, where
`η` is the learning rate.

For MSE loss, "negative gradient" is exactly the residual `y −
F_{m-1}(x)`, so gradient boosting reduces to fitting trees on
residuals — the historical name.

The base learner is nearly always a shallow tree (depth 3–8). Each
tree corrects the ensemble's remaining errors on the training set.
Because each tree is fit *given the current ensemble*, the trees are
sequential and not independent — you cannot parallelise across trees
the way you can with a random forest, though modern implementations
parallelise across features and rows within each tree.

### Why "gradient" boosting

The name generalises the residual-fitting idea to any differentiable
loss. Squared error gives you the original algorithm; log-loss gives
you a classifier; quantile loss gives you a quantile regressor;
custom losses (Cox for survival, Poisson for counts) all fit the
same framework. You always fit the next tree to the negative
gradient of the loss at the current ensemble's predictions.

## The four implementations

You have a choice of four:

| Library | First release | Underlying idea | Best for |
|---|---|---|---|
| `sklearn.ensemble.HistGradientBoosting*` | 2019 | Histogram-based, ships with sklearn | Zero-dependency baseline; native missing / categorical |
| XGBoost (`xgboost`) | 2014 | Level-wise trees + regularised objective + histogram (since 1.0) | Well-tuned production models; strong ecosystem |
| LightGBM (`lightgbm`) | 2016 | Leaf-wise growth + histogram + GOSS + EFB | Very large datasets; fastest training |
| CatBoost (`catboost`) | 2017 | Ordered boosting + native categorical encoding | Datasets rich in categoricals |

All four implement gradient boosting; the differences are engineering
choices (tree growth strategy, missing-value handling, categorical
encoding, distributed training). For a first project any of the four
will give you a competitive model. The paragraphs below cover the
differences that matter in practice.

### `HistGradientBoosting*`: the sklearn default

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`
were added to scikit-learn in 2019 (formerly experimental, promoted
in 0.21). They are a histogram-based gradient booster, inspired by
LightGBM.

Advantages of the sklearn-native version:

- **Zero extra dependency.** Ships with `sklearn`. Do not underestimate
  this in enterprise CI.
- **Handles missing values natively.** No `SimpleImputer` needed;
  missings go to whichever child branch reduces loss.
- **Handles categorical features natively.** Since scikit-learn 1.4,
  `categorical_features="from_dtype"` reads any column with dtype
  `category` as categorical. No `OneHotEncoder` needed.
- **Early stopping and monitoring built in.** `early_stopping="auto"`
  (default since 1.3) automatically holds out 10% and stops when
  validation stops improving.

Downsides:

- Slightly slower than LightGBM on the largest datasets.
- Fewer knobs than XGBoost / LightGBM. This is often a feature.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    max_iter=1000,
    learning_rate=0.05,
    max_depth=None,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping="auto",
    validation_fraction=0.1,
    n_iter_no_change=25,
    random_state=0,
).fit(X_train, y_train)
```

For a first modelling pass, this is often the right choice.

### XGBoost

The original modern gradient-boosting library (Chen and Guestrin,
2016). Widely deployed; the reference against which the others
compare. Since 2.0 it has both level-wise (`grow_policy="depthwise"`)
and leaf-wise (`grow_policy="lossguide"`) tree growth.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,        # L1 on leaf weights
    reg_lambda=1.0,       # L2 on leaf weights
    tree_method="hist",   # histogram; default since 2.0
    enable_categorical=True,
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=0,
    n_jobs=-1,
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
```

Notable knobs:

- **`max_depth`** — level-wise depth. `6` is the historic default;
  common range 3–10.
- **`min_child_weight`** — minimum sum-of-Hessians per leaf. In
  effect, minimum leaf size. Larger = stronger regularisation.
- **`subsample`** — bagging fraction per tree. Adds decorrelation.
- **`colsample_bytree` / `colsample_bylevel` / `colsample_bynode`**
  — feature subsampling à la random forest.
- **`reg_alpha` / `reg_lambda`** — L1 / L2 penalties on leaf
  weights. `reg_lambda=1.0` default is usually fine.
- **`tree_method`** — `"hist"` (histogram), `"exact"` (deprecated
  for practical use), `"gpu_hist"` (GPU; renamed to `device="cuda"`
  in 2.0). Use `"hist"`.

XGBoost handles missing values natively (learns a default direction
per split). Categorical support was added in 1.5 (`enable_categorical=True`);
the column dtype must be `category` in Pandas.

### LightGBM

Faster than XGBoost on large datasets and produces slightly different
trees due to two engineering choices:

- **Leaf-wise growth.** Instead of adding a full level at each
  iteration, LightGBM adds the single leaf that reduces the loss
  most. Faster to reach a given loss, easier to overfit — hence
  the `num_leaves` knob (upper bound on leaves) instead of
  `max_depth`.
- **GOSS (Gradient-based One-Side Sampling).** Keeps all rows with
  large gradients and subsamples the small-gradient rows for speed
  with minimal accuracy loss.
- **EFB (Exclusive Feature Bundling).** Bundles mutually exclusive
  sparse features (typical of one-hot-encoded categoricals) into
  fewer effective columns.

```python
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,           # unlimited when num_leaves is set
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,       # apply bagging every iteration
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=0,
    n_jobs=-1,
).fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[__import__("lightgbm").early_stopping(50)],
)
```

Native categorical support: cast the column to `category` in Pandas
and pass `categorical_feature="auto"`. LightGBM implements the
Fisher (1958) algorithm for the optimal split of a categorical
feature under many-value settings — faster and more accurate than
one-hot for high cardinality.

### CatBoost

Developed by Yandex, differentiated by:

- **Ordered boosting.** A permutation-based scheme that prevents a
  subtle form of prediction shift when target-statistics are used
  for categorical encoding.
- **Native categorical handling.** The library computes leakage-safe
  target statistics per fold internally. You pass raw string
  categories.
- **Reasonable defaults.** Often competitive out of the box with
  less tuning than XGBoost / LightGBM.

```python
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    cat_features=["country", "device", "product_id"],
    eval_metric="Logloss",
    early_stopping_rounds=50,
    verbose=False,
    random_seed=0,
).fit(X_train, y_train, eval_set=(X_val, y_val))
```

CatBoost is often the least-tuning-effort option when the dataset is
rich in high-cardinality categoricals. It is also slower to train
than LightGBM on very large numeric datasets.

## The five hyperparameters that decide the outcome

Across all four implementations, the five parameters that matter
most:

### 1. `learning_rate` (a.k.a. `eta`, `shrinkage`)

The step size — how much of each new tree's prediction is added.
Small values (0.01–0.1) mean many trees each with a small
contribution: more accurate, slower to train. Large values (0.2+)
mean fewer trees: faster to train, risk of overshooting.

**Default:** 0.1 in XGBoost / LightGBM / CatBoost. Reduce to
0.05 or 0.03 as you increase `n_estimators`.

### 2. `n_estimators` / `num_iterations` / `iterations`

The number of boosting rounds. Should be set high (2000–10 000) and
controlled by early stopping. Do not tune this by grid search — use
early stopping instead (see below).

### 3. `max_depth` (XGBoost, CatBoost) / `num_leaves` (LightGBM) / `max_leaf_nodes` (HGB)

The complexity of each tree. Shallow trees (depth 3–5) generalise
better and are the "canonical" gradient boosting setup. Deeper trees
(6–10) can capture more complex interactions but risk overfitting.

- XGBoost / CatBoost: `max_depth=6` is the standard default.
- LightGBM: `num_leaves=31` (i.e. `2^5 - 1`, a 5-deep balanced tree)
  is the default; keep in mind LightGBM's leaf-wise growth means
  `num_leaves=127` can be far deeper than a depth-7 balanced tree.

### 4. Regularisation (`reg_alpha`, `reg_lambda`, `min_child_weight`, `min_child_samples`)

Prevents leaves from being too small or too aggressive.

- `min_child_weight` (XGBoost) — minimum sum of Hessians per leaf.
  Bump to 5–100 on noisy targets.
- `min_child_samples` (LightGBM) — minimum row count per leaf.
  Bump for the same reason.
- `reg_alpha` (L1), `reg_lambda` (L2) — penalise leaf-weight
  magnitude. Rarely needed; try if the model is clearly overfitting
  and `min_child_*` is not enough.

### 5. Subsampling (`subsample`, `colsample_bytree`)

Row and feature subsampling per tree, borrowed from random forests.
0.7–0.9 for both is a common range. Adds decorrelation between
trees, similar in spirit to bagging.

## Early stopping: how you actually tune `n_estimators`

Never `for n in [100, 500, 1000, ...]: fit(n)`. Fit once with a
large `n_estimators` and an `eval_set`, and let the library track
validation loss round-by-round. When validation loss stops improving
for `early_stopping_rounds` iterations, stop and use the best-iteration
model.

```python
from lightgbm import LGBMClassifier, early_stopping

lgb = LGBMClassifier(n_estimators=5000, learning_rate=0.05)
lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50)],
)
print(f"Best iteration: {lgb.best_iteration_}")
```

XGBoost: `early_stopping_rounds=50` in the constructor (since 2.0).

`HistGradientBoosting*`: `early_stopping="auto"` (default since 1.3)
uses an internal 10% split; pass `validation_fraction=0.1`,
`n_iter_no_change=25`.

CatBoost: `early_stopping_rounds=50`.

**Two gotchas:**

- **Do not use the test set as the `eval_set`.** That is
  hyperparameter tuning on the test set. Use a validation split
  (or, better, use CV — chapter 07).
- **Early stopping needs a metric.** Set `eval_metric` explicitly to
  match your objective. Default is often `logloss` for
  classification, which is what you want; explicit is better than
  implicit.

## Categorical features: three strategies

Ranked from most-to-least accurate for high-cardinality features:

### 1. Native handling

`HistGradientBoosting*` with `categorical_features="from_dtype"`,
LightGBM with `categorical_feature="auto"` on Pandas `category`
dtype, CatBoost with `cat_features=[...]`. All three use algorithms
purpose-built for splitting categoricals, and all handle unseen
values gracefully.

**Use this when you can.** It is faster to fit, more memory-
efficient, and usually more accurate than pre-encoding.

### 2. Target encoding (`sklearn.preprocessing.TargetEncoder`)

Cross-fitted target encoding, wrapped in a `Pipeline`. See `mod-103`
chapter 07. Works with any gradient booster, including ones that do
not support categoricals natively (XGBoost before 1.5, or when you
need to serialise to a format that doesn't preserve category dtype).

### 3. Ordinal encoding

`OrdinalEncoder`. Fast; trees do not care about the ordering.
Loses information at high cardinality (the tree can only split at
integer boundaries), but usually good enough for cardinalities up
to a few hundred.

### 4. One-hot encoding

Only if cardinality is small (~20 or fewer per column). Explodes
in the number of features otherwise. Trees still work — but each
split can only peel off one category at a time.

## Class imbalance in gradient boosting

- **`scale_pos_weight`** (XGBoost, LightGBM) — multiplier on the
  positive-class loss. Set to `n_negative / n_positive` for
  balanced training.
- **`class_weight="balanced"`** (`HistGradientBoosting*` since 1.4).
- **CatBoost:** `auto_class_weights="Balanced"`.

All of these are preferable to `SMOTE`-style oversampling with
gradient boosters — the boosting objective already handles the
weighted loss directly.

## Regression, classification, and beyond

The default objectives:

- **Regression:** `squared_error` / `MSE`, `absolute_error` /
  `MAE`, `poisson` for counts, `gamma` for positive skewed
  targets, `quantile` for quantile regression, `huber` for
  outlier-robust regression.
- **Binary classification:** `binary` / `binary_logistic`.
- **Multiclass classification:** `multi_softmax` /
  `multi_logloss`.
- **Ranking:** `rank_xendcg` (LightGBM), `pairwise` /
  `ndcg` (XGBoost) — for learning-to-rank / recommender problems.

All four libraries also support user-defined losses via a callable
gradient/Hessian function, when the built-in list is not enough.

## A worked comparison

```python
from lightgbm import LGBMClassifier, early_stopping
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
y = (y == ">50K").astype(int)

# Cast object columns to `category` so LightGBM handles them natively.
for c in X.select_dtypes(include="object").columns:
    X[c] = X[c].astype("category")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=0,
)

lgb = LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=-1,
)
lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(50)],
)
proba = lgb.predict_proba(X_test)[:, 1]
print(f"Best iter: {lgb.best_iteration_}")
print(f"Test AUC:  {roc_auc_score(y_test, proba):.4f}")
```

Swap `LGBMClassifier` for `XGBClassifier`, `CatBoostClassifier`, or
`HistGradientBoostingClassifier` for a side-by-side comparison. On
most tabular datasets the four will land within ~1 percentage point
of each other in AUC; which one wins is dataset-dependent.

## GPU training

Since 2.0, XGBoost uses `device="cuda"` (formerly `tree_method="gpu_hist"`).
LightGBM: `device="gpu"` and compile with GPU support. CatBoost:
`task_type="GPU"`. For datasets under a few million rows the CPU is
usually fast enough; GPU pays off on tens of millions of rows.

## Pitfalls collected

1. **Tuning `n_estimators` by grid search.** Use early stopping.
   Grid search over `n_estimators` wastes the tuning budget on a
   parameter you can determine automatically.
2. **Reading the training loss curve to decide when to stop.**
   Training loss goes to zero. Watch the validation loss.
3. **Setting `learning_rate=0.01` and `n_estimators=100`.** You
   never train long enough to converge. Rule of thumb: as you
   halve `learning_rate`, roughly double `n_estimators` (with
   early stopping, this is just a starting cap).
4. **Using the test set as `eval_set` for early stopping.** You
   are now selecting the number of trees on the test set —
   optimistic by construction.
5. **`fit(..., eval_set=...)` without `early_stopping_rounds`.**
   The library trains all `n_estimators` rounds; the eval set is
   only used for reporting. Both have to be set.
6. **Using one-hot encoding for a 10 000-category column with
   XGBoost.** You have created 10 000 columns and are about to
   OOM. Ordinal-encode, target-encode, or use LightGBM /
   CatBoost's native handling.
7. **Comparing tuned XGBoost to un-tuned LightGBM and concluding
   XGBoost is better.** All four libraries default to reasonable
   but not-tuned settings. Compare fairly — tune each with the
   same effort.
8. **Not setting `random_state` / `random_seed`.** Fits are not
   deterministic without it, and CV variance eats into your
   ability to reason about hyperparameter effects.
9. **Leaving `subsample=1.0`, `colsample_bytree=1.0`.** No
   decorrelation between trees. Bump both to 0.7–0.9 for free
   regularisation.
10. **Using accuracy as the `eval_metric` for a probabilistic
    problem.** Accuracy is a step function; early stopping on it
    is noisy. Use log-loss or AUC.

## Summary

- Gradient boosting fits trees sequentially, each on the negative
  gradient of the loss w.r.t. the current ensemble. It typically
  wins on tabular data.
- Four implementations you will encounter: `HistGradientBoosting*`
  (sklearn native, zero-dependency default), XGBoost (ubiquitous,
  most tunable), LightGBM (fastest, leaf-wise growth), CatBoost
  (native categoricals, ordered boosting). All four are
  competitive; choose by ecosystem fit.
- Five hyperparameters do most of the work: `learning_rate`,
  `n_estimators` (via early stopping), tree size (`max_depth` /
  `num_leaves`), leaf regularisation (`min_child_weight` /
  `min_child_samples`), and subsampling.
- Use early stopping. Never grid-search over `n_estimators`.
- Prefer native categorical handling to pre-encoding; use
  `TargetEncoder` inside a `Pipeline` when your library does not
  handle categoricals directly.

## Where this goes next

Chapters 07 and 08 close the loop: 07 covers the cross-validation
strategies that give you an honest generalisation estimate for any
of the models in chapters 02–06, and 08 covers the hyperparameter
search machinery (grid, random, halving, Optuna) that finds the
best configuration efficiently.
