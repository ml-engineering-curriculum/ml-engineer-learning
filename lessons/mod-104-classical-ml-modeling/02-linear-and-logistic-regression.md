# 02 — Linear and logistic regression

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** `mod-101` chapter on supervised learning,
> `mod-102` chapter 06 (Pipelines).

## Motivation

Linear regression is the model you can carry with you into any
meeting: it fits in one line of code, its coefficients read like
sentences ("a one-unit increase in `x` corresponds to a `β` change in
`y`"), and its errors are cheap to compute. Logistic regression does
the same for classification.

Neither will win a Kaggle competition on tabular data — gradient
boosting will (chapter 06). But both are the baseline you compare
against, the calibration reference other models are measured on, and
the model you deploy when serve-time latency is measured in
microseconds instead of milliseconds. If your first submission of any
tabular problem is not a regularised linear model, you have skipped
the step that tells you whether the problem is hard.

This chapter covers the API, the underlying loss functions, and the
solver choices that decide whether your fit converges. Chapter 03
covers the regularisation term.

## Linear regression: the model

Given a design matrix `X ∈ ℝⁿˣᵈ` and a target vector `y ∈ ℝⁿ`, ordinary
least squares (OLS) finds the coefficient vector `β ∈ ℝᵈ` that
minimises the sum of squared residuals:

```
L(β) = ‖y − Xβ‖²
```

Closed-form solution:

```
β̂ = (Xᵀ X)⁻¹ Xᵀ y
```

`scikit-learn`'s `LinearRegression` implements this via SVD by
default, which is more numerically stable than forming `(XᵀX)⁻¹`
directly.

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)   # R² on the test set
```

`model.coef_` is the vector `β̂`; `model.intercept_` is the bias term.
Positive coefficients mean the feature moves the prediction up.

### Assumptions, in decreasing order of load-bearing

OLS is unbiased and BLUE (Best Linear Unbiased Estimator) under the
Gauss–Markov assumptions: linearity of the relationship, independent
errors, homoscedasticity, and no perfect multicollinearity. The
strict Gauss–Markov theorem also assumes zero-mean errors uncorrelated
with the features.

In practice, and roughly in order of how much they matter for
prediction (as distinct from inference):

1. **Linearity.** The relationship between each feature and the
   target is linear. Chapter 07 of `mod-103` (feature engineering)
   is how you get non-linear signals into a linear model —
   `log1p`, binning, interactions.
2. **No perfect multicollinearity.** Two features that are exact
   linear combinations of each other make `XᵀX` singular. `sklearn`
   will still fit (SVD absorbs it) but the individual coefficients
   become uninterpretable. Ridge regression (chapter 03) is the
   pragmatic fix.
3. **Approximately Gaussian residuals.** OLS is not sensitive to
   Gaussian *features*, only to the residuals being roughly
   symmetric and unimodal. When they are not — a heavy right tail on
   `y` is the common case — a log-transform of the target often
   fixes it. See `TransformedTargetRegressor`.
4. **Homoscedasticity.** Constant variance of residuals across the
   fitted values. Rarely holds in practice; matters for inference
   (confidence intervals) more than for prediction.

For pure prediction, only #1 and #2 are load-bearing. #3 and #4
matter when you want to say something about *why* the model predicts
what it does.

### Metrics for regression

For point predictions on a continuous target, three metrics cover
almost everything:

- **MSE / RMSE** — mean (or root-mean) squared error. Penalises
  large errors quadratically. The loss OLS minimises.
- **MAE** — mean absolute error. Penalises linearly. Robust to
  outliers.
- **R²** — proportion of variance explained. `1.0` is perfect,
  `0.0` is "as good as predicting the mean", negative means "worse
  than predicting the mean". `sklearn`'s `.score()` default for
  regressors.

All three live in `sklearn.metrics`:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

`mod-107` covers the metric-selection discipline in depth (choose
the metric that matches the cost of a wrong prediction).

## Logistic regression: the model

For a binary target `y ∈ {0, 1}`, logistic regression fits the
same linear function `Xβ` and passes it through the logistic
(sigmoid) function to produce a probability:

```
P(y = 1 | x) = σ(xᵀβ) = 1 / (1 + e^(−xᵀβ))
```

The loss is the negative log-likelihood, which for binary targets is
the binary cross-entropy:

```
L(β) = −Σᵢ [ yᵢ log σ(xᵢᵀβ) + (1 − yᵢ) log(1 − σ(xᵢᵀβ)) ]
```

There is no closed form. `scikit-learn` uses one of several
iterative solvers, discussed below.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
# ... clean categoricals with your Pipeline from mod-103 ...

model = LogisticRegression(max_iter=1000, random_state=0)
model.fit(X_train, y_train)
model.predict_proba(X_test)[:5]   # (n, 2) — column 1 is P(y=1)
```

**Key methods:**

- `predict(X)` → hard labels (thresholded at 0.5).
- `predict_proba(X)` → per-class probabilities. **Use this whenever
  you care about ranking or calibration.**
- `decision_function(X)` → the raw `xᵀβ` (log-odds). Useful for
  computing custom thresholds or for AUC.
- `coef_` → shape `(n_classes − 1, n_features)` for binary
  (`n_classes = 2` gives shape `(1, n_features)`), or `(n_classes,
  n_features)` for multiclass.

### Multiclass

Two strategies:

- **`multi_class="ovr"`** — one-vs-rest. `n_classes` independent
  binary classifiers. Simple; well-defined even when classes are
  rare.
- **`multi_class="multinomial"`** — a single softmax classifier
  fitting all classes jointly. Requires a compatible solver
  (`lbfgs`, `newton-cg`, `sag`, `saga`, but not `liblinear`).
  Usually gives slightly better probabilities.

Since scikit-learn 1.5 the default is `"multinomial"` when the
solver supports it. Do not rely on the default silently; be
explicit for reproducibility.

### Metrics for classification

For binary classification, the metrics that matter here:

- **Accuracy** — fraction correct. Misleading when classes are
  imbalanced.
- **ROC-AUC / PR-AUC** — the ranking metrics that ignore the
  probability threshold.
- **Log loss (binary cross-entropy)** — the loss the model
  minimises. What you want to look at if you care about
  calibration.
- **F1 / precision / recall** — at a fixed threshold. Use when the
  operating threshold is fixed by the product decision.

`mod-107` handles metric selection at length. For the exercises in
this module, `roc_auc_score` and `log_loss` are enough.

## Fitting: solvers and when to use each

`LinearRegression` uses SVD; there is nothing to tune. `LogisticRegression`
is where the solver choice matters. As of scikit-learn 1.5 the
options are:

| Solver | Handles L1? | L2? | Elastic-net? | Multinomial? | Sparse `X`? | Best for |
|---|---|---|---|---|---|---|
| `lbfgs` (default) | ✗ | ✓ | ✗ | ✓ | ✓ | small–medium, dense |
| `liblinear` | ✓ | ✓ | ✗ | ✗ (OvR only) | ✓ | small, sparse; L1 |
| `newton-cg` | ✗ | ✓ | ✗ | ✓ | ✓ | large, dense |
| `newton-cholesky` | ✗ | ✓ | ✗ | ✓ | ✗ | `n_samples ≫ n_features` |
| `sag` | ✗ | ✓ | ✗ | ✓ | ✓ | very large `n_samples` |
| `saga` | ✓ | ✓ | ✓ | ✓ | ✓ | large + L1 or elastic-net |

Rules of thumb:

- Default (`lbfgs`) is fine for datasets up to a few hundred
  thousand rows and a few hundred features.
- **Use `saga`** when you want L1 or elastic-net *and* your dataset
  is bigger than `liblinear` can handle. `saga` needs standardised
  features to converge quickly (chapter 03).
- **Use `newton-cholesky`** when you have millions of rows and few
  features and only L2 regularisation. Fastest converger for that
  shape.
- If you see a `ConvergenceWarning`, increase `max_iter` before
  reaching for a different solver.

Since scikit-learn 1.3, `LogisticRegression`'s convergence tolerance
is `tol=1e-4` by default. That is fine for prediction; tighten it
(e.g. `tol=1e-6`) if you are inspecting individual coefficients.

## Feature scaling: not optional for penalised models

Linear regression without regularisation is *scale-invariant* — you
can multiply a column by 1000 and the fitted `β` for that column will
be divided by 1000, and predictions will be unchanged.

**As soon as you add regularisation (chapter 03), scale matters
enormously.** The L2 penalty `‖β‖²` penalises coefficients equally,
so a large-magnitude coefficient on a small-scale feature gets
shrunk more than a small-magnitude coefficient on a large-scale
feature. This is almost never what you want.

Same for L1: the penalty threshold at which a feature is zeroed out
is measured in coefficient magnitude, which depends on scale.

For gradient-based solvers (`lbfgs`, `sag`, `saga`), unscaled
features also hurt convergence. `saga` can take 10–100× more
iterations without scaling.

Consequence: **wrap `LogisticRegression` (and any regularised linear
model) in a `Pipeline` with `StandardScaler` as the first step.**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler(with_mean=True)),
    ("clf",   LogisticRegression(penalty="l2", C=1.0, max_iter=1000)),
])
```

If your features are sparse (one-hot encoded), use
`StandardScaler(with_mean=False)` — subtracting the mean destroys
sparsity. Or skip scaling for the one-hot branch entirely; sparse
one-hots are already binary and scale-comparable.

## Reading the coefficients

For a fitted linear model, `model.coef_` is where interpretation
starts. Two rules:

1. **Coefficients are on the scale of the *scaled* features.** If
   you standardised, a coefficient of `+0.3` means "a one-standard-
   deviation increase in this feature moves the prediction by `+0.3`
   units of `y`". This is often what you want for comparability.
2. **Coefficients are conditional.** `β_j` is the effect of `x_j`
   *holding all other features constant*. In the presence of
   correlated features, individual coefficients can be misleading;
   see chapter 03 (Ridge) and chapter 09 (interpretation pitfalls).

For logistic regression, `β_j` is on the *log-odds* scale. `exp(β_j)`
gives the odds ratio: "a one-unit increase in `x_j` multiplies the
odds of the positive class by `exp(β_j)`". Small coefficients: `β ≈
0.05 ⇒ ~5% odds increase`. Large ones: `β = 1.0 ⇒ e ≈ 2.7× odds`.

```python
import pandas as pd

pipe.fit(X_train, y_train)
coefs = pd.Series(
    pipe.named_steps["clf"].coef_.ravel(),
    index=pipe[:-1].get_feature_names_out(),
).sort_values(key=abs, ascending=False)
print(coefs.head(20))
```

`get_feature_names_out()` (added in scikit-learn 1.0 and comprehensive
across preprocessors in 1.2) is what makes this readable — the column
names on the transformed matrix, in order.

## Class imbalance

For classification, imbalanced classes bias `LogisticRegression`
toward the majority class. Two knobs:

- **`class_weight="balanced"`** — reweights the loss by inverse
  class frequency: `w_c = n_samples / (n_classes · n_c)`. This is
  usually the right first move.
- **`sample_weight=`** in `.fit(X, y, sample_weight=...)` — per-row
  weights. Use when you have a domain-specific cost structure (e.g.
  false negatives cost 10× more than false positives).

Do **not** use `RandomOverSampler` / `SMOTE` before a `Pipeline` —
oversampling on the whole dataset before splitting leaks the
minority class from test into train (`mod-103` chapter 06). If you
need synthetic oversampling, use `imblearn.pipeline.Pipeline` and
put the sampler *inside* the pipeline so it only runs on training
folds.

## A worked baseline

For the UCI Adult dataset (`fetch_openml("adult", version=2)`), the
end-to-end baseline in one script:

```python
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
y = (y == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, stratify=y,
)

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
])

preprocess = ColumnTransformer([
    ("num",  numeric,     make_column_selector(dtype_include="number")),
    ("cat",  categorical, make_column_selector(dtype_include=["object", "category"])),
])

pipe = Pipeline([
    ("features", preprocess),
    ("clf",      LogisticRegression(max_iter=1000, class_weight="balanced")),
])

pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
```

This is the shape every subsequent model in the module will take:
a `ColumnTransformer` that handles preprocessing, and a final
estimator. Only the estimator will change.

## Pitfalls collected

1. **Fitting `LogisticRegression` on unscaled features and
   wondering why it doesn't converge.** Always scale for penalised
   linear models.
2. **Reading raw coefficients when features have wildly different
   scales.** The "largest coefficient" belongs to the smallest-scale
   feature. Standardise, then compare.
3. **`predict()` on an imbalanced problem.** Silent 0.5 threshold
   on a 5% positive class returns almost all negatives, and
   accuracy is misleadingly high. Look at `predict_proba` and pick
   the threshold deliberately (`mod-107`).
4. **Trusting `p_hat` from an unregularised model on wide,
   correlated data.** Coefficients (and hence probabilities) can be
   unstable. Ridge (chapter 03) is the first fix.
5. **`multi_class` default drift.** The default has shifted across
   scikit-learn versions. If your pinned version differs from CI or
   from a colleague's, be explicit: `multi_class="multinomial"` or
   `"ovr"`.
6. **Using `SMOTE` before splitting.** Textbook target-fold leakage.
   Sampler goes inside `imblearn.pipeline.Pipeline`.
7. **`LinearRegression().fit(X, y).predict(X_test)` when `y` is
   log-transformed but you forgot to `expm1` on the way back.**
   `TransformedTargetRegressor(regressor=LinearRegression(),
   func=np.log1p, inverse_func=np.expm1)` handles it in one step.

## Summary

- Linear and logistic regression are the baseline every tabular ML
  project should start with — cheap to fit, cheap to serve, easy to
  interpret.
- `LinearRegression` is closed-form via SVD; `LogisticRegression` is
  iterative and the solver choice matters (`lbfgs` default;
  `saga` for L1 / elastic-net; `newton-cholesky` for tall-thin
  L2-only problems).
- Scale features before regularised linear models; every penalty
  term is scale-sensitive.
- Coefficients are on the scaled feature space and are conditional
  on the other features. For log-odds interpretation on logistic
  regression, take `exp(β)`.
- Handle class imbalance with `class_weight="balanced"` and
  keep any sampler *inside* an `imblearn` pipeline.

## Where this goes next

Chapter 03 introduces the regularisation term that made all of the
above robust on real data with dozens of correlated features:
Ridge, Lasso, and elastic-net.
