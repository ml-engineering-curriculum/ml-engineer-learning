# 05 — Random forests and bagging

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 04.

## Motivation

A single decision tree is high-variance: change one row and the
top split can flip. If we could somehow average many trees whose
errors are only partially correlated, the averaged prediction would
be much more stable — and, empirically, much more accurate.

Bagging (bootstrap aggregating) does exactly that. A random forest
adds one more twist — a random feature subsample at each split — to
push the trees' errors further apart. Together they produce a
model that is:

- Nearly parameter-free (three or four knobs you have to think
  about).
- Robust to overfitting (adding more trees never hurts).
- Free of preprocessing (no scaling, no imputation for the modern
  histogram variants).
- Trivially parallel to train and predict.
- Equipped with a free hold-out set (out-of-bag scoring).

For a first non-linear model on tabular data, a random forest with
sensible defaults is a strong, honest baseline. Chapter 06's
gradient boosters will beat it on most well-tuned problems, but the
gap is often smaller than you would expect, and the random forest
takes about one tenth the tuning effort.

## Bagging: the general idea

Given a training set of `n` rows, bagging (Breiman, 1996) works as
follows:

1. Draw a bootstrap sample of `n` rows *with replacement* from the
   training set. On average, each bootstrap sample contains ~63.2%
   of the unique training rows (the rest are duplicates), and
   ~36.8% of the training rows are omitted — the *out-of-bag* (OOB)
   set for that tree.
2. Fit a base learner on the bootstrap sample.
3. Repeat `n_estimators` times.
4. To predict: **average** the predicted probabilities (classification)
   or means (regression) across all base learners.

If the base learner is a decision tree grown to purity (high
variance, low bias), the ensemble's variance is roughly `Var(single) /
n_estimators` when the trees are independent — and lower bias than
the trees would have if we tried to prevent overfitting by pruning.
Trees are rarely fully independent (they see overlapping data), but
bagging still reduces variance substantially.

`sklearn.ensemble.BaggingClassifier` and `BaggingRegressor` implement
generic bagging over any base estimator. `RandomForestClassifier` /
`RandomForestRegressor` is the specialised case that everyone uses.

## Random forests: bagging + random subspace

Breiman's random forest (2001) makes two changes beyond generic
bagging:

- **Each tree is trained on a bootstrap sample** (the bagging part).
- **At each split, only a random subset of `max_features` features
  is considered** (the *random subspace* part).

The second change decorrelates the trees. Without it, a very
predictive feature dominates the top split of every tree, and the
trees make correlated errors. With it, the strong feature is only
available at ~`max_features / n_features` of the splits, forcing
other features to compete and the trees to explore different
partitions.

The result: individual trees are slightly worse (each sees fewer
features per split), but their errors are much less correlated, and
the average is better than pure bagging.

## The scikit-learn interface

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,          # let each tree grow deep
    min_samples_leaf=5,
    max_features="sqrt",     # sqrt(n_features) considered per split
    bootstrap=True,
    oob_score=True,          # compute OOB score after fit
    n_jobs=-1,               # use all cores
    random_state=0,
).fit(X_train, y_train)

print(f"OOB score: {rf.oob_score_:.4f}")
print(f"Test acc:  {rf.score(X_test, y_test):.4f}")
```

Key hyperparameters, with the ones that matter first:

- **`n_estimators`** — number of trees. **More is always at least
  as good** (in expectation); the cost is training and prediction
  time. Rule of thumb: 500 for a baseline, 1000+ if you can
  afford it, more only if you can measure the improvement.
- **`max_features`** — features per split. `"sqrt"` (default for
  classification) and `"log2"` are the two you will use;
  regression defaults to `1.0` (all features) which is *not*
  Breiman's original recommendation of `n_features / 3`. Set it
  explicitly for regression: `max_features="sqrt"` or `max_features=0.33`.
- **`max_depth`** — tree depth. In a random forest, `None` (unlimited)
  is the *right* default — bagging handles the overfitting.
- **`min_samples_leaf`** — protects against very small leaves. 1
  (default) is fine on clean data; bump to 5–20 if you have noisy
  targets.
- **`bootstrap`** — whether to sample with replacement. True by
  default. If False, each tree sees the whole training set and you
  are relying entirely on `max_features` for decorrelation
  (Extremely Randomised Trees, aka Extra Trees, take this further —
  see below).
- **`class_weight="balanced"`** — for imbalanced classification.
  Reweights the loss without oversampling.

The reason `max_depth=None` is safe here — and not in a single
tree — is that the ensemble average protects you. A single
depth-`None` tree memorises. Five hundred depth-`None` trees
trained on different bootstraps memorise *different* rows, and
their disagreements cancel on average.

## Out-of-bag scoring: a free hold-out set

For each training row, roughly 37% of the trees did not see it
during training. That subset of trees is a valid, held-out model
for that row. `oob_score=True` computes the ensemble prediction on
each training row using only the trees that did not train on it,
and reports the resulting score.

Why care?

- **You get a hold-out score without setting one aside.** For a
  quick baseline on a small dataset, OOB is a decent proxy for
  cross-validated accuracy at no additional cost.
- **You can watch it grow with `n_estimators`.** Refit at
  `n=100, 200, 500, 1000, ...` and see when OOB plateaus.

Cautions:

- OOB is per-row; it is not the same as a cross-validated
  estimate for hyperparameter selection. For picking between
  models, use `cross_val_score` (chapter 07).
- OOB assumes rows are IID. If you have grouped or time-ordered
  data, the bootstrap breaks the grouping and OOB overstates
  generalisation.
- `oob_score` accepts a scoring function since scikit-learn 1.4;
  before that it was fixed to accuracy for classification and R²
  for regression.

## Extra Trees: bagging without the bootstrap

`ExtraTreesClassifier` / `ExtraTreesRegressor` push the randomisation
further:

- No bootstrap — every tree sees the whole training set.
- At each split, a random threshold per candidate feature is
  chosen, and the *best of those random splits* is picked (instead
  of the exact best split).

Trees are individually weaker, ensemble variance is even lower,
and training is faster (no threshold search). Sometimes marginally
better than random forests, sometimes marginally worse. Worth
trying as a second baseline.

```python
from sklearn.ensemble import ExtraTreesClassifier
rf = ExtraTreesClassifier(n_estimators=500, random_state=0, n_jobs=-1)
```

## When random forests shine

- **First non-linear baseline.** Almost always beats a single
  tree, often within a few percentage points of a well-tuned
  gradient booster, with far less tuning effort.
- **Robust to hyperparameters.** The defaults are good on most
  problems. `n_estimators=500`, `max_features="sqrt"`,
  everything else at default is a reasonable submission.
- **Mixed feature types, missing values, no scaling needed.**
  Trees don't care.
- **Fast, parallel training and inference.** `n_jobs=-1` uses all
  cores; each tree is independent.
- **Comes with an OOB estimate.** A free hold-out.
- **Comes with a feature importance vector.** Chapter 09 covers
  the caveats.

## Where random forests hurt

- **They cannot extrapolate.** Every prediction is an average of
  training-target values in some leaf; the model never predicts
  outside `[min(y_train), max(y_train)]`. Regression on trending
  time-series time is a bad match.
- **They over-smooth.** The averaging softens sharp decision
  boundaries. A gradient booster (chapter 06) can preserve them.
- **Prediction is not cheap at high `n_estimators`.** 1000 trees
  of depth 20 is ~20000 comparisons per prediction. If serving
  latency matters, this can be too slow — a smaller, well-tuned
  gradient booster often serves faster with better accuracy.
- **Memory scales with `n_estimators × n_leaves`.** A random forest
  on a large training set can easily be a gigabyte on disk. Trim
  `max_depth` and `min_samples_leaf` if this matters.
- **Impurity-based feature importance is biased** (chapter 09).
  Use permutation importance or SHAP instead of `.feature_importances_`
  for anything you report.

## Class imbalance in random forests

`class_weight="balanced"` reweights the loss per tree.
`class_weight="balanced_subsample"` recomputes weights per bootstrap
sample, which is subtly better when the imbalance is severe.

For extreme imbalance (positive rate < 1%), consider
`BalancedRandomForestClassifier` from `imbalanced-learn`, which
under-samples the majority class in each bootstrap.

## A worked example: OOB vs. test

```python
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
y = (y == ">50K").astype(int)

# Random forests need categoricals encoded; a minimal OneHot inline:
X = X.copy()
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = X[col].astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0,
)

for n in [50, 100, 200, 500, 1000]:
    rf = RandomForestClassifier(
        n_estimators=n,
        max_features="sqrt",
        oob_score=True,
        n_jobs=-1,
        random_state=0,
    ).fit(X_train, y_train)
    print(f"n={n:4d}  OOB={rf.oob_score_:.4f}  test={rf.score(X_test, y_test):.4f}")
```

Two things to notice as you run this:

- OOB tracks test accuracy closely, within a few tenths of a
  percentage point. That is the "free hold-out" property.
- The improvement per additional tree gets smaller after ~200.
  Marginal accuracy gains from tree 500 to 1000 are usually
  fractions of a percent — cost the tuning budget accordingly.

## Random forest inside a Pipeline

Even though random forests do not need scaling or imputation, they
still belong in a `Pipeline`:

```python
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

preprocess = ColumnTransformer([
    ("num", "passthrough", make_column_selector(dtype_include="number")),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            make_column_selector(dtype_include=["object", "category"])),
])

pipe = Pipeline([
    ("features", preprocess),
    ("rf",       RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)),
])
```

The Pipeline is what makes CV, hyperparameter search, and serving
work identically to training. `OrdinalEncoder` is enough for trees
(one-hot only helps if you have very few categories and a naïve
splitter); random forests handle the numerics passed through
without scaling.

For handling categoricals natively — with no encoding needed —
`HistGradientBoostingClassifier` and LightGBM (chapter 06) are
usually the better choice.

## Pitfalls collected

1. **Reading `.feature_importances_` from a random forest and
   shipping it as "the important features".** Impurity-based
   importance biases toward high-cardinality and continuous
   features (chapter 09). Use permutation importance or SHAP
   instead.
2. **Setting `max_features=1.0` on a regression `RandomForestRegressor`
   and being disappointed by the accuracy.** The regression
   default is `1.0` (all features), which turns off decorrelation
   and reduces the ensemble to plain bagging. Set `max_features="sqrt"`
   or `max_features=0.33`.
3. **Trusting `oob_score_` on time-series or grouped data.** The
   bootstrap breaks temporal / group structure; OOB is optimistic.
   Use a time-ordered or grouped CV instead (chapter 07).
4. **`n_estimators=50` "to save time".** You will pay for it in
   variance on the validation score, and in downstream tuning
   confusion. Start at 200–500.
5. **Random forest on high-cardinality categoricals with one-hot
   encoding.** The one-hot columns are almost all zeros; each
   split can only "peel off" one category at a time. Ordinal
   encode, target encode (`mod-103` chapter 07), or use LightGBM
   (chapter 06).
6. **Confusing bagging's independence with cross-validation
   independence.** Bagging reduces variance across trees; it does
   not give you an honest generalisation estimate for
   hyperparameter selection. `cross_val_score` still required.

## Summary

- Bagging trains many high-variance base learners on bootstrap
  samples and averages them; the ensemble's variance is much lower
  than a single learner's.
- Random forests add a random feature subsample per split, which
  decorrelates the trees further and improves the average.
- `n_estimators`, `max_features`, `max_depth`, and `min_samples_leaf`
  are the four knobs; the first two matter most.
- OOB scoring is a free approximation to CV — useful for tracking
  learning curves, not for hyperparameter tuning on non-IID data.
- Random forests are a strong non-linear baseline. On tuned tabular
  problems, gradient boosting (chapter 06) usually beats them, but
  the gap is small and the tuning cost is high.

## Where this goes next

Chapter 06 changes the ensembling strategy from "many independent
trees, averaged" to "many sequential trees, each fitting the
residual of the previous ensemble". That change — boosting instead
of bagging — is the current state of the art for tabular data.
