# 04 — Decision trees

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01.

## Motivation

A decision tree is a sequence of if/else questions on the features
that eventually leads to a prediction. It is the most human-readable
non-linear model in ML: you can trace any prediction back to the
five or ten questions that produced it.

By itself a decision tree is rarely the model you deploy. It
overfits eagerly, its predictions have step-function discontinuities,
and its accuracy is usually beaten by a random forest with the same
tuning effort. But it is the essential building block: random forests
(chapter 05) average trees, and gradient boosting (chapter 06) adds
them sequentially. If you cannot reason about one tree, you cannot
reason about a thousand of them.

This chapter covers the algorithm (CART), the split criteria, the
hyperparameters that control complexity, and the ways a single tree
goes wrong.

## The algorithm: recursive greedy partitioning

The CART (Classification And Regression Trees) algorithm, which
`scikit-learn` implements, builds a binary tree recursively:

1. Start with all training rows in the root node.
2. For each feature and each candidate split point, score how much
   the split reduces impurity (classification) or variance
   (regression).
3. Pick the (feature, threshold) pair that gives the best split.
4. Partition the rows into two child nodes and recurse on each.
5. Stop when a stopping criterion is met (max depth, min samples
   per leaf, no impurity gain, etc.).

Predictions:

- **Classification.** The predicted class of a leaf is the majority
  class of the training rows that fell into it. The predicted
  probability is the empirical class distribution.
- **Regression.** The prediction is the mean of the target values
  of the training rows in the leaf.

This is "greedy" because it optimises one split at a time, never
revisiting an earlier decision. It is not optimal — finding the
globally optimal tree is NP-hard — but it is fast and, with enough
depth, arbitrarily flexible.

## Split criteria

At each candidate split, the algorithm scores the resulting child
nodes. The "impurity" of a node summarises how mixed the classes (or
how spread the target) is within it.

### Classification

For a node with class proportions `p_1, ..., p_K`:

- **Gini impurity:** `1 − Σ p_k²`. Zero at a pure node, `1 − 1/K`
  at a uniformly mixed one. Default in `sklearn`. Cheap to compute.
- **Entropy:** `−Σ p_k log p_k`. Zero at a pure node, `log K` at
  uniform. Slightly slower (log). Both criteria produce very
  similar trees in practice.
- **Log-loss** (`sklearn` ≥ 1.1) — equivalent to entropy; renamed
  for clarity.

The impurity *reduction* from a split is:

```
ΔI = I(parent) − (n_L / n) · I(left) − (n_R / n) · I(right)
```

The best split maximises `ΔI`.

### Regression

- **Squared error (MSE):** default. Variance within the node. The
  prediction at a leaf minimises MSE ⇒ it is the mean of the
  training targets.
- **Absolute error (MAE):** median-based prediction; more robust
  to outliers, roughly an order of magnitude slower to fit.
- **Friedman MSE:** a variant used inside gradient boosting; gives
  slightly better split scores for `GradientBoostingRegressor`.
- **Poisson deviance** (`sklearn` ≥ 1.0) — for count data. Requires
  non-negative targets.

### Choosing

For 99% of use cases the default (`gini` for classification, `squared_error`
for regression) is fine. The choice of split criterion is far less
important than the depth of the tree.

## The hyperparameters that actually matter

`DecisionTreeClassifier` and `DecisionTreeRegressor` have a dozen
knobs. Four of them do 90% of the work.

### `max_depth`

The maximum depth of the tree. `None` (default) means "grow until
every leaf is pure or has fewer than `min_samples_split` samples".
On non-trivial data this produces a tree with a leaf per training
row and 100% training accuracy — the textbook overfitting example.

Cap it. Depths in the 4–10 range are typical for a stand-alone
tree; 3–5 for trees inside a boosting ensemble; unlimited for
random-forest members (see chapter 05 for why).

### `min_samples_leaf`

The minimum number of training samples required for a node to be a
leaf. Nodes with fewer samples cannot exist. Values in the 5–50
range prevent leaves that are just a memorised outlier.

Use this instead of `max_depth` when the depth varies naturally
across branches — one region of the input space has more structure
than another.

### `min_samples_split`

The minimum number of samples required to *consider* splitting a
node. A stricter version of `min_samples_leaf`. Usually `min_samples_split
= 2 * min_samples_leaf` or thereabouts; set one or the other, not
both aggressively.

### `max_features`

The number of features considered at each split. `None` (default)
means all features. Setting `max_features="sqrt"` or a small integer
adds randomness — each split is chosen from a random subset — and is
the trick that makes random forests work (chapter 05). For a single
tree, leave it at `None`.

### `ccp_alpha`: cost-complexity pruning

An alternative to depth-based control. After the tree is fully
grown, `ccp_alpha` prunes back subtrees whose leaves do not
justify their complexity. Set via cross-validation:

```python
from sklearn.tree import DecisionTreeClassifier

path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas
# Fit a tree for each alpha and pick the best by CV.
```

Rarely worth the effort for tabular ML — bumping `max_depth` or
`min_samples_leaf` is simpler and produces comparable results.

## A minimal example

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
X = X.select_dtypes(include="number")   # keep it simple for the illustration
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=25,
    random_state=0,
).fit(X_train, y_train)

print(f"Train acc: {tree.score(X_train, y_train):.3f}")
print(f"Test  acc: {tree.score(X_test,  y_test):.3f}")

plt.figure(figsize=(14, 6))
plot_tree(tree, feature_names=X.columns, class_names=tree.classes_, filled=True)
plt.show()
```

A depth-5 tree on the Adult dataset has ~32 leaves at most, which is
about the upper end of "still readable in a screenshot".

## What trees are good at

- **Non-linear relationships out of the box.** Age is non-monotonic
  with conversion? A tree finds and encodes the shape without any
  feature engineering.
- **Feature interactions.** Every path from root to leaf is an
  implicit `AND` of the conditions along it. Two features that
  together predict the target — but neither predicts it alone —
  will be found.
- **Mixed-type features.** Numerics, ordinals, and binary-encoded
  categoricals coexist without scaling. No `StandardScaler`
  needed.
- **Missing values, natively.** Since scikit-learn 1.3,
  `DecisionTree*` (and the histogram-based ensembles) route
  missing values to the child branch that reduces loss most. No
  `SimpleImputer` needed.

## What trees are bad at

- **Smooth relationships.** A tree approximates a smooth curve as
  a staircase. Linear signals are best served by linear models;
  cyclical signals need explicit encoding (`hour_sin`, `hour_cos`)
  before a tree can find them.
- **Extrapolation.** A tree only predicts values in the range of
  the training target. A regression tree trained on `y ∈ [0, 100]`
  can never predict 150, no matter how obviously extrapolatable the
  input.
- **Stability.** Change a handful of training rows and the top split
  can change entirely, restructuring the whole tree below it. This
  is the *high variance* that makes single trees a weak model and
  ensembles a strong one.
- **High-cardinality categoricals.** With hundreds of levels per
  feature, the number of candidate binary splits explodes and
  performance degrades. Use ordinal or target encoding first
  (`mod-103` chapter 07), or use LightGBM's native categorical
  handling (chapter 06).

## Regression trees, briefly

Same algorithm, different loss. `DecisionTreeRegressor` splits to
minimise variance within children; leaves predict the mean.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=0)
```

The step-function nature is more obvious for regression: predictions
are one of at most `n_leaves` distinct values. If your target lives
on a continuous scale, this alone is a reason to prefer an
ensemble.

## The variance problem: a warm-up for chapter 05

Fit ten trees on ten bootstrap samples of the same training data:

```python
from sklearn.utils import resample
import numpy as np

trees = []
for _ in range(10):
    X_b, y_b = resample(X_train, y_train, random_state=None)
    trees.append(DecisionTreeClassifier(max_depth=5, random_state=None).fit(X_b, y_b))

preds = np.stack([t.predict_proba(X_test)[:, 1] for t in trees])
print(f"Prediction std across trees: {preds.std(axis=0).mean():.3f}")
```

The typical row will have a prediction that varies by 5–15 percentage
points across the ten trees. Now average them:

```python
avg_pred = preds.mean(axis=0)
```

The averaged prediction is more accurate than any individual tree.
This is *bagging* — bootstrap aggregating — and it is the entire
motivating idea of chapter 05.

## Reading a tree

`plot_tree` gives you the picture; `export_text` gives you the
rules:

```python
from sklearn.tree import export_text
print(export_text(tree, feature_names=list(X.columns)))
```

Sample output (fictional shape, real format):

```
|--- capital-gain <= 7073.50
|   |--- education-num <= 12.50
|   |   |--- class: <=50K
|   |--- education-num >  12.50
|   |   |--- age <= 27.50
|   |   |   |--- class: <=50K
|   |   |--- age >  27.50
|   |   |   |--- class: >50K
|--- capital-gain >  7073.50
|   |--- class: >50K
```

Every row is a training rule. This is the single strongest argument
for keeping a shallow tree in your toolkit even when your production
model is a gradient booster — you can hand this text to a
non-technical stakeholder and they will follow it.

## Pitfalls collected

1. **Growing an unlimited tree "to see how well it fits".** It
   will fit training data perfectly and generalise terribly. Set
   `max_depth` (or `min_samples_leaf`) up front.
2. **Interpreting a single tree's feature importance and shipping
   that as a stakeholder deliverable.** A single tree's importance
   depends on which features happened to win the top splits, which
   is high-variance. Use a random-forest (chapter 05) or
   permutation importance (chapter 09) for stable importance.
3. **Not setting `random_state`.** Trees are deterministic given a
   dataset, but ties in split scores are broken arbitrarily. Set
   `random_state=0` (or any fixed integer) for reproducible fits.
4. **Assuming a tree "handles high cardinality" because it does not
   need one-hot.** It handles it slowly. Bin, target-encode, or
   let LightGBM's native categorical handler take it (chapter 06).
5. **Ordinal-encoding a nominal categorical for a tree "because it
   only cares about splits".** True but subtle: with `country`
   ordinal-encoded, the tree can only make splits like `country_id
   <= 42`, which forces artificial groupings. `HistGradientBoosting*`
   and LightGBM's `categorical_features=` handle this correctly;
   plain `DecisionTree*` do not.

## Summary

- A decision tree recursively partitions the feature space by
  greedily maximising impurity reduction (Gini / entropy for
  classification; variance for regression).
- `max_depth`, `min_samples_leaf`, `min_samples_split`, and
  `max_features` are the four hyperparameters that matter; the
  rest are footnotes.
- Trees handle non-linearity, interactions, mixed types, and
  missing values natively. They struggle with smooth relationships,
  extrapolation, high-cardinality categoricals, and — most
  importantly — variance.
- The variance problem is what motivates ensembling. Chapter 05
  averages trees to reduce it; chapter 06 boosts them to reduce
  bias as well.

## Where this goes next

Chapter 05 introduces bagging and random forests: many trees, each
trained on a bootstrap sample, each considering a random subset of
features per split. This is the smallest change to the previous
chapter's algorithm that produces a competitive tabular model.
