# 03 — Regularisation: Ridge, Lasso, and elastic-net

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 02.

## Motivation

Chapter 02 fit a linear model by minimising the training loss. In
theory that gives the best in-sample fit. In practice, on tabular
data with dozens of features — some correlated, some noise, some
scaled in millions and some in units — the unregularised fit is
almost never the model you want to deploy.

Two symptoms make this concrete:

1. **Coefficients that flip sign or swing wildly** when you
   re-fit on a different random sample. This is variance in the
   coefficient estimates. Correlated features are the usual cause:
   if `x_1` and `x_2` are highly correlated, the loss surface has a
   long, thin trough along the line `β_1 + β_2 = c`, and any point
   on it is roughly as good.
2. **A model that scores well on training and poorly on the held-
   out set.** Classic overfitting. With enough features, the model
   memorises noise in the training targets.

Regularisation is the single-parameter fix. Add a penalty to the
loss that shrinks coefficients toward zero. Trade a little training
error for a lot of generalisation error, and — in the case of L1 — get
implicit feature selection for free.

This chapter covers the three penalties you will actually reach for:
Ridge (L2), Lasso (L1), and elastic-net (a mix), plus the practical
details that decide whether they work: scaling, choosing the
regularisation strength, and the CV variants that pick it for you.

## The penalised objective

For linear regression, the penalised loss is:

```
L(β) = ‖y − Xβ‖² + α · P(β)
```

where `P(β)` is the penalty and `α ≥ 0` is the regularisation
strength.

| Method | Penalty `P(β)` | Effect |
|---|---|---|
| Ridge (L2) | `Σ β_j² = ‖β‖²` | shrinks all coefficients smoothly toward 0 |
| Lasso (L1) | `Σ |β_j| = ‖β‖₁` | shrinks and *zeros out* small coefficients |
| Elastic-net | `ρ ‖β‖₁ + (1 − ρ) ‖β‖²` | mixture: L1 selection with L2 stability |

`α = 0` recovers OLS. `α → ∞` shrinks every coefficient to zero (and
the model always predicts the mean).

For logistic regression, `sklearn` uses the reciprocal convention:
`C = 1 / α`. Small `C` means strong regularisation, large `C` means
weak. This is a common source of confusion — Ridge and Lasso classes
use `alpha`; `LogisticRegression` uses `C`. They mean opposite things.

## Ridge (L2): the safe default

Ridge shrinks each coefficient smoothly toward zero. Nothing is
zeroed exactly — every feature keeps a (usually small) coefficient.
Two properties make it the sensible default:

- **Stability under correlated features.** Where OLS's coefficients
  can swing wildly along a correlation ridge, Ridge collapses them
  onto a stable interior point of the trough.
- **Closed-form solution.** `β̂ = (XᵀX + αI)⁻¹ Xᵀy`. `sklearn`
  implements it via one of several solvers (`svd`, `cholesky`,
  `lsqr`, `sparse_cg`, `sag`, `saga`, `lbfgs`); default is `auto`.

```python
from sklearn.linear_model import Ridge, RidgeClassifier
model = Ridge(alpha=1.0)   # regression
```

For classification, `RidgeClassifier` is a hinge-loss-adjacent
formulation — it does not produce probabilities. If you want
Ridge-style regularisation on a probabilistic classifier, use
`LogisticRegression(penalty="l2", C=1/alpha)`, which is what
`sklearn` normally means by "logistic regression with L2".

### Bayesian view (one paragraph)

Ridge is equivalent to maximum-a-posteriori estimation of the
coefficients under a Gaussian prior with zero mean and variance
`σ² / α`. Small `α` = wide prior = little shrinkage. Large `α` =
narrow prior = strong shrinkage. Useful for reasoning about
priors when you have domain knowledge; not required for using it.

### When Ridge shines

- Dense feature sets (few zero features expected).
- Many correlated features you do not want to force-choose between.
- You want stable coefficients for interpretation.
- You have almost no time to tune — Ridge with `alpha=1` on scaled
  features is a good starting point on almost any tabular problem.

## Lasso (L1): shrinkage with feature selection

Lasso uses `‖β‖₁`. The geometry of the L1 ball has corners on the
axes; when the loss surface's minimum lies near a corner, the
solution snaps to it and one or more coefficients become exactly
zero. This gives **implicit feature selection**: the fitted model
uses a subset of the features and reports the rest as unused.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1, max_iter=10_000)
model.fit(X_train, y_train)

selected = X.columns[model.coef_ != 0]
```

Two knobs beyond `alpha`:

- **`max_iter`** — Lasso is solved by coordinate descent, which
  needs more iterations than OLS. Bump to `10_000`+ if you see a
  `ConvergenceWarning`.
- **`selection="random"`** — random coordinate order. Faster on
  wide problems; still gets to the same optimum.

### When Lasso shines

- You suspect most features are irrelevant and want the model to
  tell you which.
- You need a compact model for serving (fewer coefficients to look
  up).
- You want an interpretable list of "the features the model
  actually used".

### Where Lasso hurts

- **Correlated features.** With two nearly identical columns,
  Lasso picks one and zeros the other, essentially at random. The
  chosen column changes across bootstrap samples. If which feature
  survives is a decision you care about, Lasso by itself is not
  enough. Elastic-net (below) is the standard fix; a *stability
  path* (fitting Lasso on many bootstrap samples and reporting the
  selection frequency for each feature) is another.
- **`n_samples < n_features`** with more than `n_samples` truly
  non-zero features. Lasso can only select at most `n_samples`
  features in that regime. Rarely matters on tabular ML, matters
  in genomics / bioinformatics settings.

## Elastic-net: L1 and L2 together

Elastic-net penalises with `ρ ‖β‖₁ + (1 − ρ) ‖β‖²`. It keeps the
sparsity of Lasso while inheriting Ridge's stability under
correlated features: correlated features share their coefficient
rather than fight over it.

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10_000)
```

Two hyperparameters instead of one:

- **`alpha`** — overall strength, as before.
- **`l1_ratio`** — how much of the penalty is L1 vs. L2.
  - `l1_ratio=0.0` is pure Ridge.
  - `l1_ratio=1.0` is pure Lasso.
  - `l1_ratio ∈ (0, 1)` is a mix. `l1_ratio=0.5` is a common
    starting point.

If your dataset has correlated features and you want *some*
selection, elastic-net with `l1_ratio` around `0.3–0.7` typically
beats pure Lasso.

## Choosing `alpha` (and `l1_ratio`)

Three routes, in increasing power:

### 1. Fixed value

Use `alpha=1.0` on scaled features. Fine for a baseline. Do not
ship this to production without evaluating it against something
tuned.

### 2. The built-in `CV` variants (regularisation path)

`RidgeCV`, `LassoCV`, `ElasticNetCV`, `LogisticRegressionCV` fit
along an entire path of `alpha` values efficiently and pick the
best by cross-validation. They exploit the fact that solutions for
adjacent `alpha` values are similar (warm-starting), so the whole
path costs only slightly more than a single fit.

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import numpy as np

alphas = np.logspace(-4, 2, 25)
ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
print(ridge.alpha_)
```

For elastic-net you also search `l1_ratio`:

```python
enet = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=None,          # let it pick a path per l1_ratio
    cv=5,
    max_iter=10_000,
).fit(X_train, y_train)
print(enet.alpha_, enet.l1_ratio_)
```

These are almost always what you should reach for on regression
problems. On classification, `LogisticRegressionCV` plays the same
role.

### 3. `GridSearchCV` / `RandomizedSearchCV` / Optuna

The general search machinery from chapter 08 works. It is more
flexible (you can tune `alpha` alongside model-agnostic pipeline
hyperparameters like `min_frequency` on `OneHotEncoder`) but slower
than the built-in path CV variants for pure `alpha` tuning.

## Scaling: the load-bearing step

`mod-102` chapter 06 and chapter 02 above both said this. It bears
repeating in the regularisation chapter because it is where
regularisation goes wrong most often.

**Every regularised linear model must be fit on standardised
features.** The penalty `‖β‖²` or `‖β‖₁` is a magnitude-based penalty.
Features on different scales get penalised unequally:

- A feature measured in dollars (say, `income` ∈ [10⁴, 10⁶]) will
  have a naturally small coefficient (say, `10⁻⁴`).
- A feature measured in units of 0–1 (say, `is_married` ∈ {0, 1})
  will have a coefficient with larger magnitude (say, `0.3`).

Without scaling, the L2 penalty shrinks `is_married` far more
aggressively than `income`. That is not a feature-selection
decision; that is a units mistake.

Wrap in a `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("lasso", LassoCV(cv=5, max_iter=10_000)),
])
pipe.fit(X_train, y_train)
```

For sparse features (one-hot), `StandardScaler(with_mean=False)`.
Better: use a `ColumnTransformer` that scales the numeric branch
and passes the one-hot branch through untouched.

## The regularisation path, visualised

The set of `β̂(α)` values across a range of `α` is the
*regularisation path*. It answers the question "which features get
shrunk to zero first as regularisation strengthens?" For Ridge, no
coefficient ever reaches zero; the path smoothly compresses. For
Lasso, coefficients hit zero one by one and the resulting sequence
of surviving features is a natural feature-importance ranking.

```python
from sklearn.linear_model import lasso_path
import numpy as np

alphas, coefs, _ = lasso_path(X_scaled, y, alphas=np.logspace(-3, 1, 50))
# coefs.shape == (n_features, len(alphas))
```

Plot each feature's coefficient trajectory across `alpha`. The
features whose coefficients persist at high `alpha` are the ones
Lasso considers most important. This is a much more useful
importance signal than "raw coefficient magnitude at a single
`alpha`".

## Coefficient stability: bootstrap and refit

Even with regularisation, individual coefficients on correlated
features can be unstable. If you plan to *interpret* the fitted
coefficients (rather than just use them for prediction), check that
the sign and magnitude are stable under resampling:

```python
from sklearn.utils import resample
import pandas as pd

coefs = []
for _ in range(200):
    X_b, y_b = resample(X_train, y_train, random_state=None)
    pipe.fit(X_b, y_b)
    coefs.append(pipe.named_steps["lasso"].coef_)

coefs = pd.DataFrame(coefs, columns=X_train.columns)
summary = pd.DataFrame({
    "mean":    coefs.mean(),
    "std":     coefs.std(),
    "nonzero": (coefs != 0).mean(),
})
```

`nonzero` — the fraction of bootstraps in which a feature was
selected — is the *selection stability*. A feature selected in <50%
of bootstraps is a coin flip; do not report it as "selected".

## Regularisation on logistic regression

Everything above works for classification, with the `alpha ↔ C`
convention flip.

```python
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import numpy as np

# Fixed strength
clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)

# CV-tuned strength on the path
clf_cv = LogisticRegressionCV(
    Cs=np.logspace(-4, 4, 20),
    penalty="l2",
    cv=5,
    max_iter=1000,
    scoring="roc_auc",
).fit(X_scaled, y)
```

For L1 or elastic-net on logistic regression, use
`solver="saga"` and (for elastic-net) `penalty="elasticnet",
l1_ratios=[...]`.

## Pitfalls collected

1. **Forgetting to scale.** The single most common regularisation
   bug. Standardise numeric features; put the scaler inside a
   `Pipeline`.
2. **Mixing up `alpha` and `C`.** `Ridge(alpha=1)` and
   `LogisticRegression(C=1)` regularise very differently at the
   same numeric value. Small `C` = strong regularisation.
3. **`LassoCV` on a wide, correlated feature set and reading the
   surviving features as "important".** With correlated features,
   which member of a correlated group survives is unstable. Bootstrap
   before you claim.
4. **Lasso for probability calibration.** Lasso's aggressive shrinkage
   biases probabilities toward the mean. Ridge (or unregularised
   logistic + Platt scaling) calibrates better. `mod-107` covers
   calibration in detail.
5. **Optimising `alpha` on the test set.** Use `RidgeCV` /
   `LassoCV`'s built-in CV, or use `GridSearchCV`. Do not do a
   sweep and pick by hand looking at held-out performance — you have
   just made the test set part of training.
6. **Fitting Ridge without a `max_iter` bump on ill-conditioned
   data.** The `sag` and `saga` solvers can silently fail to
   converge on unscaled or degenerate `X`. `ConvergenceWarning` is
   a real warning; fix it.
7. **Fitting Lasso and forgetting `max_iter`.** The default (1000)
   is too low for many real datasets. Bump to 10_000+ for anything
   that warns.

## Choosing between the three

| Situation | First choice |
|---|---|
| No idea yet, baseline | Ridge with `alpha=1.0` |
| Many features, most likely noise | Lasso; expect ~5–20 to survive |
| Many correlated features, want some sparsity | Elastic-net, `l1_ratio ≈ 0.5` |
| Extremely wide (`n_features ≫ n_samples`) | Lasso or elastic-net; expect selection |
| Need well-calibrated probabilities | Ridge (or logistic + Platt) |
| Need a compact deployed model | Lasso (fewer non-zero coefficients) |
| Feature importance is the deliverable | Elastic-net with bootstrap stability check |

None of these is a one-way door. `RidgeCV`, `LassoCV`, and
`ElasticNetCV` all fit in seconds on typical tabular problems; run
all three, compare on CV, ship the winner.

## Summary

- Regularisation trades training error for generalisation and
  coefficient stability. It is the single-parameter fix for
  overfitting in linear models.
- Ridge (L2) shrinks smoothly and is the safe default. Lasso (L1)
  gives implicit feature selection but is unstable under correlated
  features. Elastic-net mixes them and is often the best of both.
- **Scale first.** Every penalty is scale-sensitive; the `Pipeline`
  with `StandardScaler` is not optional.
- Use the `*CV` estimators (`RidgeCV`, `LassoCV`, `ElasticNetCV`,
  `LogisticRegressionCV`) for path-based selection of the
  regularisation strength — they are dramatically faster than
  `GridSearchCV` and give the same answer.
- If you plan to interpret the coefficients, bootstrap the fit and
  report selection stability, not a single point estimate.

## Where this goes next

Chapters 04–06 leave linear-model territory for the ensemble family
that dominates tabular ML in practice. Chapter 04 starts with the
single decision tree — the building block that bagging (chapter 05)
and boosting (chapter 06) both use.
