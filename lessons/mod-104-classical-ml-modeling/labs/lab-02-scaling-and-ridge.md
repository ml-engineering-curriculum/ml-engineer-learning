# lab-02 — Why regularised linear models need a `StandardScaler`

> **Duration:** ~30–45 minutes.
> **Companion chapters:** [02 — Linear and logistic regression](../02-linear-and-logistic-regression.md), [03 — Regularisation: Ridge, Lasso, and elastic-net](../03-regularisation-ridge-lasso-elasticnet.md).

## Objective

Reproduce chapter 03's "scaling: the load-bearing step" argument on a
hand-built two-feature problem small enough to inspect in a single
print statement. By the end you will have (a) built a synthetic
regression problem where two features are equally predictive but live
on wildly different scales, (b) fit Ridge *without* scaling and watched
the penalty crush the small-scale feature's coefficient while leaving
the large-scale feature barely shrunk, (c) wrapped Ridge in a
`Pipeline` with `StandardScaler` and seen the two coefficients become
comparable, and (d) encoded the "scaling habit" as an assertion.

This lab is a targeted drill: just the scaling point, isolated from
everything else exercise-01 covers (the regularisation path, `*CV`
variants, bootstrap selection stability). The exercise expects you to
*already have* the intuition this lab builds.

## Prerequisites

- Python 3.11+, scikit-learn ≥ 1.4, Pandas ≥ 2.1, NumPy ≥ 1.26.
- Chapters 02 and 03 read through once.
- Any virtual environment (`uv`, `poetry`, or a plain venv).

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-02-scaling && cd lab-02-scaling
touch scaling.py
```

Everything below goes in `scaling.py`. Run with `python scaling.py`
after each step.

Start with imports:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

## Steps

### Step 1 — A two-feature problem with wildly different scales (≈ 5 min)

Chapter 03's canonical setup: one feature in the units of dollars
(roughly `10^4`–`10^6`), one feature as a 0/1 flag. Both equally
predictive *per standard deviation* — i.e. shift `income` by one sd and
`is_member` by one sd and the target moves by the same amount.

Add to `scaling.py`:

```python
RNG = np.random.default_rng(seed=0)

def build_dataset(n: int = 2000) -> tuple[pd.DataFrame, np.ndarray]:
    income    = RNG.uniform(10_000, 1_000_000, size=n)    # ~ [10^4, 10^6]
    is_member = RNG.integers(0, 2, size=n).astype(float)  # 0 or 1

    # Standardised versions — equally predictive by construction.
    income_z    = (income    - income.mean())    / income.std()
    is_member_z = (is_member - is_member.mean()) / is_member.std()

    noise = RNG.normal(scale=0.5, size=n)
    y = 2.0 * income_z + 2.0 * is_member_z + noise

    X = pd.DataFrame({"income": income, "is_member": is_member})
    return X, y

X, y = build_dataset()
print(X.describe().round(1))
print(f"\n|corr(income, y)|    = {X['income'].corr(pd.Series(y)):.3f}")
print(f"|corr(is_member, y)| = {X['is_member'].corr(pd.Series(y)):.3f}")
```

**Expected output.** `income` has a range roughly `[1e4, 1e6]`,
`is_member` is {0, 1}. The absolute correlations of the two features
with `y` are roughly equal (each near `0.6`–`0.7`). That is the whole
point — the two features carry the same amount of signal, just in
wildly different units.

**Checkpoint.** Both features are equally predictive on a
*standardised* basis. If any model prefers one over the other, that
preference is an artefact of the fitting procedure, not of the data.

### Step 2 — Fit Ridge *without* scaling (≈ 10 min)

Chapter 03's "Scaling: the load-bearing step" argument says:

> A feature measured in dollars (say, `income ∈ [10⁴, 10⁶]`) will have
> a naturally small coefficient (say, `10⁻⁴`). A feature measured in
> units of 0–1 (say, `is_married ∈ {0, 1}`) will have a coefficient
> with larger magnitude (say, `0.3`). Without scaling, the L2 penalty
> shrinks `is_married` far more aggressively than `income`.

Reproduce it. Add to `scaling.py`:

```python
ridge_raw = Ridge(alpha=1.0).fit(X, y)
coefs_raw = pd.Series(ridge_raw.coef_, index=X.columns)
print("\nRidge on unscaled features (alpha=1.0):")
print(coefs_raw.to_string())
print(f"  |income_coef|     = {abs(coefs_raw['income']):.3e}")
print(f"  |is_member_coef|  = {abs(coefs_raw['is_member']):.3e}")
print(
    "  ratio             = "
    f"{abs(coefs_raw['is_member']) / abs(coefs_raw['income']):.2e}"
)
```

Run it.

**Expected output.** The `income` coefficient is tiny (something like
`4e-06`) and the `is_member` coefficient is ordinary-sized (something
like `2.0`). The ratio is roughly on the order of the ratio of the two
features' standard deviations — i.e. around `10^5`.

**Checkpoint.** Both features are equally predictive, yet their fitted
coefficients differ by five orders of magnitude. A reader who ranks
features by `|coef|` and concludes "`is_member` matters a million times
more than `income`" is looking at a units mistake, not a model
finding. Chapter 03's pitfall #1 and chapter 09's "coefficient
magnitude" caveat both name this.

Now show that the L2 penalty is doing exactly what the chapter said —
shrinking one coefficient far more than the other. Compare the fit at
`alpha=1.0` to the fit at `alpha=1e-8` (effectively unpenalised, to
reveal what the OLS solution would be):

```python
ridge_weak = Ridge(alpha=1e-8).fit(X, y)
coefs_weak = pd.Series(ridge_weak.coef_, index=X.columns)
print("\nRidge on unscaled features (alpha≈0, i.e. ~OLS):")
print(coefs_weak.to_string())
print(
    "\nShrinkage from alpha=1e-8 → alpha=1.0, per feature:\n"
    f"  income    coefficient × "
    f"{coefs_raw['income']    / coefs_weak['income']:.6f}\n"
    f"  is_member coefficient × "
    f"{coefs_raw['is_member'] / coefs_weak['is_member']:.6f}"
)
```

**Expected output.** `income`'s coefficient barely moves between
`alpha=1e-8` and `alpha=1.0` — the ratio is very close to `1.0`.
`is_member`'s coefficient is shrunk substantially — the ratio is
noticeably less than `1.0`.

**Checkpoint.** The L2 penalty `‖β‖²` penalises coefficients in
*magnitude*. A coefficient that is naturally `10⁻⁵` barely registers
against the penalty, so it is hardly shrunk. A coefficient that is
naturally `2.0` is squared to `4.0` and takes most of the penalty
budget. That is why the small-scale coefficient collapses and the
large-scale one survives — not because it is "more important".

### Step 3 — Fit Ridge *inside* a `Pipeline` with `StandardScaler` (≈ 10 min)

Chapter 03's fix: wrap the regularised estimator in a `Pipeline` so the
scaler refits inside every CV fold.

Add to `scaling.py`:

```python
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
pipe.fit(X, y)

coefs_scaled = pd.Series(pipe.named_steps["ridge"].coef_, index=X.columns)
print("\nRidge on standardised features (alpha=1.0):")
print(coefs_scaled.to_string())
print(
    f"  |income_coef|    = {abs(coefs_scaled['income']):.3f}\n"
    f"  |is_member_coef| = {abs(coefs_scaled['is_member']):.3f}\n"
    f"  ratio            = "
    f"{abs(coefs_scaled['is_member']) / abs(coefs_scaled['income']):.3f}"
)
```

Run it.

**Expected output.** Both coefficients are now of the same order of
magnitude — each close to `2.0`, which is the per-standard-deviation
effect you encoded in Step 1 when building `y`. The ratio is near
`1.0`.

**Checkpoint.** Two things changed. The coefficients recovered the
true per-sd effects you used to generate the data. And the L2 penalty
now treats the two features comparably, because after standardisation
both features have variance `1.0` and the magnitude of each coefficient
reflects actual predictive weight, not units.

### Step 4 — Compare CV scores, scaled vs. unscaled (≈ 5 min)

Coefficient interpretation is one half of the argument. The other half
is that on a dataset with any meaningful difference in scales, the
unscaled Ridge *also loses accuracy* — because the penalty is being
spent in the wrong place.

Add to `scaling.py`:

```python
cv_raw = cross_val_score(
    Ridge(alpha=1.0), X, y, cv=5, scoring="r2",
)
cv_scaled = cross_val_score(
    pipe, X, y, cv=5, scoring="r2",
)

print("\n5-fold CV R² (higher is better):")
print(f"  Ridge, unscaled:  {cv_raw.mean():.4f} ± {cv_raw.std():.4f}")
print(f"  Ridge, scaled:    {cv_scaled.mean():.4f} ± {cv_scaled.std():.4f}")
print(f"  gap:              {cv_scaled.mean() - cv_raw.mean():+.4f}")
```

Run it.

**Expected output.** The scaled Ridge beats the unscaled Ridge by a
visible margin. On this synthetic problem the gap is typically
somewhere in the `0.01`–`0.05` R² range; on wider real datasets with
more features of mixed scale the gap tends to be larger.

**Checkpoint.** The unscaled model is literally a worse predictor on
cross-validation. Chapter 03 is not just about interpretability — the
penalty's misallocation also costs you accuracy. Scaling is both a
correctness fix and a performance fix.

### Step 5 — Encode the invariant as a test (≈ 5 min)

Chapter 03's pitfall #1 ("Forgetting to scale") is the kind of mistake
that reappears every time a new teammate adds a feature. Turn the
scaling rule into a one-line assertion against the *fitted* Ridge
coefficients.

Append to `scaling.py`:

```python
def test_scaled_coefs_are_comparable() -> None:
    """After standardisation, two equally-predictive features should have
    coefficients of the same order of magnitude. If this fails, either the
    scaler is not actually in the pipeline or the test data has changed."""
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ]).fit(X, y)
    income_c    = abs(pipe.named_steps["ridge"].coef_[0])
    is_member_c = abs(pipe.named_steps["ridge"].coef_[1])
    ratio       = max(income_c, is_member_c) / max(min(income_c, is_member_c), 1e-12)
    assert ratio < 2.0, (
        f"coefficients disagree by {ratio:.2f}×; expected roughly comparable "
        f"after standardisation (got income={income_c:.3f}, "
        f"is_member={is_member_c:.3f})"
    )
    print("OK: scaled Ridge coefficients are of comparable magnitude.")

def test_unscaled_coefs_differ_by_orders_of_magnitude() -> None:
    """The failure mode we are protecting against: unscaled Ridge produces
    coefficients that differ by orders of magnitude on equally-predictive
    features. This test documents the pitfall; delete it only if you have a
    good reason."""
    model = Ridge(alpha=1.0).fit(X, y)
    income_c    = abs(model.coef_[0])
    is_member_c = abs(model.coef_[1])
    ratio       = max(income_c, is_member_c) / max(min(income_c, is_member_c), 1e-12)
    assert ratio > 1_000, (
        f"expected unscaled Ridge to produce wildly uneven coefficients, "
        f"got ratio {ratio:.1f}"
    )
    print(f"OK (expected): unscaled Ridge coefficients differ by {ratio:.0f}×.")

test_scaled_coefs_are_comparable()
test_unscaled_coefs_differ_by_orders_of_magnitude()
```

Run it.

**Expected output.** Two `OK` lines. The first confirms that the
scaled pipeline produces comparable coefficients; the second confirms
that the unscaled fit produces the pathological behaviour the lab is
demonstrating.

**Checkpoint.** Each assertion is a load-bearing habit: the first
protects against someone silently removing the `StandardScaler` from a
`Pipeline`, and the second documents *why* the pipeline is there, so a
future reader cannot "simplify" it away without the test suite yelling.

## Expected final artefacts

A single file `lab-02-scaling/scaling.py` containing the five step
blocks above. Running `python scaling.py` end-to-end should take a
few seconds and print (in order) the dataset describe output and
correlations, the unscaled Ridge coefficients and their shrinkage
ratios, the scaled Ridge coefficients, the two CV R² lines with the
gap, and the two `OK` lines from the assertions.

## Final checkpoint

Without re-reading chapter 03, answer:

1. In Step 2, the unscaled Ridge gives `income` a tiny coefficient and
   `is_member` a large one, even though both features are equally
   predictive. State in your own words what the L2 penalty `‖β‖²` is
   doing — and why the per-sd effect, not the per-unit effect, is
   what Ridge is "trying" to shrink.
2. If you replaced `Ridge(alpha=1.0)` with `LinearRegression()`
   (unpenalised OLS) in Step 2, would the two coefficients still differ
   by five orders of magnitude on the raw-scale fit? Why or why not —
   and what does that imply about which models absolutely need a
   `StandardScaler` versus which ones can skip it?
3. The equivalent fix for a `Lasso` is also "wrap in a `Pipeline` with
   `StandardScaler`". But on a 500k × 2000 *sparse* feature matrix,
   chapter 02 says to use `StandardScaler(with_mean=False)` instead of
   the default. What property of the default scaler is the problem,
   and why does it not apply to the dense two-feature case in this lab?
4. In Step 4, the CV gap between scaled and unscaled Ridge is small on
   this synthetic problem. If the dataset had twenty features spanning
   `10⁻³` to `10⁶`, would you expect the gap to grow, shrink, or stay
   the same? Which other regulariser from chapter 03 would widen the
   gap even further?

If you can answer all four, the scaling habit is in your fingers and
you are ready for exercise-01's four-model comparison (OLS, Ridge,
Lasso, elastic-net) without wondering whether a misbehaving
coefficient is a model issue or a units issue.

## Where to go next

- `exercise-01` ("Regression with regularisation") does the same
  scaling discipline on a real dataset (California Housing, Ames, or
  Diabetes) and adds the pieces this lab deliberately left out: the
  `*CV` variants that pick `alpha` for you, the regularisation path,
  and bootstrap-based selection stability for Lasso.
- Chapter 06 ("Gradient-boosted trees") is the point at which
  `StandardScaler` finally stops being required — trees do not care
  about feature magnitude. Your `Pipeline` shape there can drop the
  scaler on the numeric branch; keep the one-hot / categorical branch
  intact. The habit from this lab is to notice the change deliberately,
  not to forget it.
