# exercise-02 — Trees and ensembles

> **Estimated effort:** 3 hours.
> **Companion chapters:** [04 — Decision trees](../04-decision-trees.md), [05 — Random forests and bagging](../05-random-forests-and-bagging.md), [06 — Gradient-boosted trees](../06-gradient-boosted-trees.md), and [07 — Cross-validation strategies](../07-cross-validation-strategies.md).

## Objective

Build and compare the tree family on a classification problem: a
single `DecisionTreeClassifier`, a `RandomForestClassifier`, and
`HistGradientBoostingClassifier`. You will:

- See a single tree overfit — and fix it with `max_depth` and
  `min_samples_leaf`.
- Confirm that bagging reduces variance by measuring prediction
  disagreement across trees.
- Compare a random forest's OOB score to a proper CV score.
- Move to gradient boosting with early stopping and measure the
  accuracy / training-time trade-off.
- Report each model's training curve (accuracy vs. `n_estimators`)
  and pick a stopping point that pays for itself.

The goal is empirical intuition: you should finish this exercise
able to *predict*, before running the code, roughly how much a
random forest will beat a single tree by, and roughly how much a
tuned gradient booster will beat both.

## Prerequisites

- Chapters 04–07 of this module.
- `mod-102` chapter 06 (Pipelines) and `mod-103` chapter 07
  (feature engineering, for the categorical handling).
- Python 3.11+, scikit-learn ≥ 1.4, Pandas ≥ 2.1, `matplotlib`,
  `pytest`.

## Dataset

Use one of:

- **UCI Adult** (`sklearn.datasets.fetch_openml("adult", version=2,
  as_frame=True)`) — the default. Binary classification, mixed
  numerics and categoricals, no time axis, no groups. Well-suited
  to demonstrating the model comparison.
- **Credit-default** or **Bank Marketing** from OpenML — either
  works if you want a different domain.
- **Your own from exercise-03 of `mod-103`** — if you already have
  a cleaned tabular classification dataset from that exercise,
  reuse it.

Pick one and stick with it.

## Problem statement

### Part A — the setup (≈ 20 min)

Load the dataset in `data.py`. For Adult:

```python
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
y = (y == ">50K").astype(int)
```

Split with `train_test_split(test_size=0.2, stratify=y,
random_state=0)`.

Categorical handling: for this exercise, use `OrdinalEncoder` on
the categorical branch (trees do not need one-hot):

```python
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder

preprocess = ColumnTransformer([
    ("num", "passthrough", make_column_selector(dtype_include="number")),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            make_column_selector(dtype_include=["object", "category"])),
])
```

Build a `Pipeline([("features", preprocess), ("clf", ...)])` shape
that only swaps the classifier as you move through the exercise.

Baseline: a `DummyClassifier(strategy="most_frequent")` and a
`DummyClassifier(strategy="stratified")`. Report both accuracies —
anything below either is a bug, not a model.

### Part B — one decision tree, unpruned vs. pruned (≈ 30 min)

Fit `DecisionTreeClassifier(max_depth=None, random_state=0)` on
`X_train`. Report:

- Training accuracy.
- Test accuracy.
- Depth of the fitted tree (`.get_depth()`), number of leaves
  (`.get_n_leaves()`).

Now do a small grid over `max_depth ∈ [2, 4, 6, 8, 10, 20, None]`
and `min_samples_leaf ∈ [1, 5, 25, 100]` — 5-fold CV. Report the
best combination and its CV accuracy.

**In `REPORT.md`:**

- The gap between train and test for the unpruned tree.
- The best pruned configuration.
- Plot: for `min_samples_leaf=25`, a train/CV curve as
  `max_depth` grows from 2 to 20. The "hockey-stick" divergence is
  the overfitting story chapter 04 named.

### Part C — random forest and bagging variance (≈ 40 min)

Fit `RandomForestClassifier(n_estimators=500, max_features="sqrt",
oob_score=True, n_jobs=-1, random_state=0)`.

Report:

- Training accuracy.
- OOB accuracy.
- Test accuracy.
- 5-fold CV accuracy.

**Bagging-variance experiment.** For 20 individual decision trees
(each grown with `max_depth=None` on a bootstrap sample of
`X_train`), record the test-set predicted probability per row. Plot
a histogram of the *standard deviation* of predictions across the
20 trees for each test row.

Then repeat with the random forest's individual tree predictions
(access via `[t.predict_proba(X_test)[:, 1] for t in rf.estimators_]`).
Confirm the per-row std is comparable — it is bagging + subspace
that produces the low ensemble variance, not any single tree.

Finally, plot **OOB accuracy vs. `n_estimators`** for `n ∈ [50,
100, 200, 500, 1000]`. Note where OOB plateaus. That is roughly
where "more trees" stops paying off.

**In `REPORT.md`:**

- The four accuracy numbers (train, OOB, test, CV).
- The bagging-variance histogram.
- The OOB-vs-n_estimators plot.
- Answer: is OOB accuracy an honest proxy for test accuracy on
  this dataset? (For Adult, yes; for a dataset with groups or
  time it would not be — see chapter 07.)

### Part D — `HistGradientBoostingClassifier` (≈ 40 min)

Swap the estimator for `HistGradientBoostingClassifier`. For the
categorical branch, cast to `category` dtype and set
`categorical_features="from_dtype"` on the classifier so it handles
categoricals natively (`sklearn` ≥ 1.4):

```python
X = X.copy()
for c in X.select_dtypes(include=["object"]).columns:
    X[c] = X[c].astype("category")

pipe = Pipeline([
    ("features", "passthrough"),                 # HGB handles it natively
    ("clf",      HistGradientBoostingClassifier(
        max_iter=2000,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping="auto",
        n_iter_no_change=25,
        validation_fraction=0.1,
        random_state=0,
        categorical_features="from_dtype",
    )),
])
```

Fit, report train / test / CV accuracy. Report the number of
iterations actually taken (`clf.n_iter_`) — should be substantially
less than `max_iter=2000` because of early stopping.

Run a small `RandomizedSearchCV` (`n_iter=30`) over:

- `clf__learning_rate` ∈ `loguniform(0.01, 0.3)`
- `clf__max_leaf_nodes` ∈ `randint(15, 128)`
- `clf__min_samples_leaf` ∈ `randint(5, 50)`
- `clf__l2_regularization` ∈ `loguniform(1e-8, 1)`

Report `search.best_params_`, best CV score, and the test-set score
of the winner. Compare to the un-tuned defaults.

**In `REPORT.md`:**

- Training accuracy, test accuracy, CV accuracy for the un-tuned
  HGB.
- Same for the tuned HGB.
- Wall-clock training time for the un-tuned HGB fit and for the
  full random search.
- The number of boosting iterations actually taken.

### Part E — the comparison (≈ 30 min)

Assemble a single results table in `REPORT.md`:

| Model | CV acc | Test acc | Test AUC | Fit time (s) | Notes |
|---|---|---|---|---|---|
| Dummy (majority) | | | | | |
| Dummy (stratified) | | | | | |
| Tree (best-pruned) | | | | | |
| Random forest (500 trees) | | | | | |
| HGB (defaults) | | | | | |
| HGB (tuned) | | | | | |

Include AUC as well as accuracy — accuracy alone hides the
class-imbalance behaviour.

Prose in `REPORT.md`:

1. How large is the gap between the single pruned tree and the
   random forest? Between the random forest and the tuned HGB?
   Between the tuned HGB and un-tuned HGB?
2. If you had to ship *one* model, which and why? Consider
   accuracy, tuning cost, serving cost, and interpretability.
3. Were there any surprises? (For example, HGB with `l2` heavy
   regularisation sometimes underperforms the RF baseline on
   Adult — if this happens to you, name the reason.)

### Part F — cross-validate the whole comparison (≈ 20 min)

The results above use a single train/test split. Repeat the
comparison — every model — with `cross_val_score` using
`StratifiedKFold(5, shuffle=True, random_state=0)`, reporting
mean ± std per model.

**In `REPORT.md`:**

- Updated table with CV mean and std.
- Answer: is the ranking from the single-split comparison the same
  as the CV comparison? If a model looked best on one split but
  isn't consistently best across folds, its win was probably noise.

## Starter guidance

- **Use `n_jobs=-1` everywhere** it is supported. Random forests,
  HGB, `RandomizedSearchCV` all parallelise.
- **`HistGradientBoosting*` with early stopping is your friend.**
  Do not tune `max_iter` by grid search; let early stopping pick
  it. Report `clf.n_iter_` afterwards.
- **For the bagging-variance experiment**, use *different* random
  states across the 20 hand-rolled bootstrap trees so their
  disagreements are real (not just from the deterministic random
  tie-breaking).
- **When cross-validating HGB with early stopping**, the estimator
  uses an internal validation split by default. That is separate
  from the outer CV fold, so it does not leak — but note that this
  means each CV fold's HGB uses a slightly smaller training set.
- **The `RandomizedSearchCV` with `n_iter=30`** should run in a few
  minutes on the Adult dataset with `n_jobs=-1`. If it takes more
  than half an hour, reduce `max_iter` on HGB or reduce
  `n_iter`.

## Acceptance criteria

- All models fit inside a `Pipeline` (or, for HGB with native
  categoricals, an equivalent shape); no preprocessing leaks
  outside the pipeline.
- Baseline dummy classifiers reported.
- Single-tree overfitting demonstrated (train >> test on unpruned;
  matched on pruned) with a plot.
- Random forest: OOB, CV, and test accuracy all reported and
  discussed. Bagging-variance histogram included.
- HGB: un-tuned and tuned variants both reported. Best hyperparams
  and search wall-clock included.
- Full comparison table filled in (per-model accuracy, AUC, fit
  time). CV mean ± std also reported.
- Three prose questions in Part E answered.
- `X_test` not touched by any `fit`, `fit_transform`, or CV loop
  before the final scoring.

## Stretch goals (optional)

- **Add `ExtraTreesClassifier`** to the comparison. Two extra
  lines of code; sometimes marginally better than `RandomForest`,
  sometimes marginally worse.
- **Add `XGBClassifier` and `LGBMClassifier`.** Compare against
  HGB. The three should land within a percentage point of each
  other on Adult; the differences come from tuning effort.
- **Learning curve.** For the tuned HGB, plot test accuracy vs.
  training-set size (10%, 25%, 50%, 100%). A flat curve at 50%+
  means more data won't help; a still-rising curve at 100% means
  more data would.
- **Class-imbalance experiment.** Downsample the positive class to
  5% of the training set. Re-fit each model with
  `class_weight="balanced"` (`RandomForest`) or
  `sample_weight` (HGB) and report the AUC / PR-AUC. Which model
  degrades most gracefully?
- **Serving-cost comparison.** Time
  `model.predict(X_test.iloc[[0]])` for each model, repeat 1000
  times, and report mean latency per prediction. The 500-tree
  random forest is often 10–100× slower than a compact tuned
  gradient booster.

## What to hand in

```
exercise-02/
├── data.py                    # load + split
├── models.py                  # build_*_pipeline() for each classifier
├── variance_experiment.py     # bagging-variance histograms
├── search.py                  # RandomizedSearchCV for HGB
├── evaluate.py                # comparison table + CV runs
├── tests/
│   ├── test_pipelines.py      # each pipeline fits and predicts
│   └── test_leakage.py        # X_test not touched during CV
├── plots/                     # PNG output for REPORT.md
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
