# lab-03 — Random vs grid search on the same budget

> **Duration:** ~30–45 minutes.
> **Companion chapter:** [08 — Hyperparameter search: grid, random, and Optuna](../08-hyperparameter-search.md).

## Objective

Reproduce Bergstra & Bengio's result — cited by chapter 08 — on a
small, fast, concrete problem: with the **same** total trial count,
`RandomizedSearchCV` typically beats `GridSearchCV` on hyperparameters
that span orders of magnitude. By the end you will have (a) built a
tuning problem where one hyperparameter matters a lot and two matter
very little, (b) run a grid search and a random search against the
same budget, (c) seen the grid miss the sweet spot that random lands on
(because grid spent most of its trials varying hyperparameters that do
nothing), and (d) read the `cv_results_` tables to see *why* random
won.

This lab is a focused drill on just the method-choice argument from
chapter 08 — it deliberately does not touch Optuna, which exercise-03
covers end-to-end. The question here is more basic: given a fixed
tuning budget, should you be running a grid or a random search?

## Prerequisites

- Python 3.11+, scikit-learn ≥ 1.4, Pandas ≥ 2.1, NumPy ≥ 1.26,
  SciPy ≥ 1.10.
- Chapter 08 read through once.
- Any virtual environment (`uv`, `poetry`, or a plain venv).

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-03-random-vs-grid && cd lab-03-random-vs-grid
touch search.py
```

Everything below goes in `search.py`. Run with `python search.py`
after each step.

Start with imports:

```python
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
```

## Steps

### Step 1 — A small, fast tuning problem (≈ 5 min)

Chapter 08's "tuning budgets" table puts gradient-boosting tuning in
the tens-of-minutes range on a medium dataset. For this lab we want
something that finishes in under a minute end-to-end so you can rerun
quickly — use `make_classification` to build a synthetic binary
problem, then tune `HistGradientBoostingClassifier` (sklearn-native, no
extra dependency per chapter 06).

Add to `search.py`:

```python
RANDOM_STATE = 0

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    class_sep=0.8,
    random_state=RANDOM_STATE,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def make_estimator() -> HistGradientBoostingClassifier:
    # max_iter kept modest with early stopping; the point of the lab is the
    # *search* method, not the booster's internal training loop.
    return HistGradientBoostingClassifier(
        max_iter=300,
        early_stopping="auto",
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )

print(f"Dataset: {X.shape[0]} rows × {X.shape[1]} features, "
      f"class rate = {y.mean():.3f}")
```

**Expected output.** 5000 rows, 20 features, class rate near `0.5`.
This is deliberately small so both searches finish in under a minute on
a laptop — the point of the lab is the comparison between methods, not
raw accuracy.

**Checkpoint.** Keep the dataset, CV splitter, and estimator factory
in one place so both searches use the same data and the same base
model. The only variable in the experiment is the *search strategy*.

### Step 2 — Fix the shared tuning budget (≈ 5 min)

Both searches should get the same number of CV fits. Pick a number
that is a perfect cube so grid can divide it evenly across three
hyperparameters, and that is large enough that random sampling gives
decent coverage.

Add to `search.py`:

```python
TOTAL_TRIALS = 27  # 3^3 for the grid; also the n_iter for the random search

print(f"\nBudget: {TOTAL_TRIALS} CV trials each "
      f"(× {cv.get_n_splits()} folds = {TOTAL_TRIALS * cv.get_n_splits()} fits)")
```

**Checkpoint.** `27 = 3^3` gives grid exactly three values per
hyperparameter when searching three hyperparameters — the only shape
that makes grid "fair" against random on a three-dimensional problem.
Raise this to `64 = 4^3` or `125 = 5^3` if you want richer plots at the
cost of a longer runtime.

### Step 3 — Grid search (≈ 10 min)

Chapter 08's warning about grid: "Linear grid over a log-scale
hyperparameter. `[0.001, 0.01, 0.1, 1, 10]` linearly-spaced misses
everything in the low end; `loguniform` finds it."

Grid still picks points — it just picks them from a list. The fairest
log-aware grid for `learning_rate` on this problem is three points
spanning the plausible range on a log scale.

Add to `search.py`:

```python
grid_param_grid = {
    "learning_rate":     np.logspace(-2.0, -0.5, 3).tolist(),  # ~[0.01, 0.056, 0.316]
    "max_leaf_nodes":    [15, 63, 255],
    "min_samples_leaf":  [5, 25, 100],
}

grid_search = GridSearchCV(
    make_estimator(),
    grid_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    refit=False,
)
grid_search.fit(X, y)

print("\nGrid search")
print(f"  trials run:     {len(grid_search.cv_results_['mean_test_score'])}")
print(f"  best CV AUC:    {grid_search.best_score_:.4f}")
print(f"  best params:    {grid_search.best_params_}")
```

Run it.

**Expected output.** 27 trials exactly, a best CV AUC somewhere in the
0.93–0.97 range, and a `best_params` dict with one value picked from
each of the three three-element lists.

**Checkpoint.** Grid searched exactly three `learning_rate` values. If
the sweet spot for `learning_rate` on this problem sits *between* two
of the three — say, at `0.03` — grid has no way to find it. All 27
trials are variations over the other two hyperparameters at those three
fixed learning-rate values.

### Step 4 — Random search (≈ 10 min)

Same budget, same hyperparameter ranges, but drawn from
`loguniform` / `randint` distributions. Each trial gets a fresh,
continuous sample along each axis.

Add to `search.py`:

```python
random_param_distributions = {
    "learning_rate":     loguniform(0.01, 0.316),   # same range as the grid
    "max_leaf_nodes":    randint(15, 256),          # same endpoints
    "min_samples_leaf":  randint(5, 101),
}

random_search = RandomizedSearchCV(
    make_estimator(),
    random_param_distributions,
    n_iter=TOTAL_TRIALS,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    refit=False,
)
random_search.fit(X, y)

print("\nRandomized search")
print(f"  trials run:     {len(random_search.cv_results_['mean_test_score'])}")
print(f"  best CV AUC:    {random_search.best_score_:.4f}")
print(f"  best params:    {random_search.best_params_}")
```

Run it.

**Expected output.** 27 trials. The best CV AUC is typically higher
than grid's; the gap on this problem is often `0.002`–`0.01` AUC.
`best_params["learning_rate"]` will almost certainly be a value that
is *not* one of the three grid points — that is the whole point.

**Checkpoint.** With the same 27 trials, random sampled 27 different
`learning_rate` values from a continuous distribution. Grid sampled
exactly three. On any hyperparameter where the sweet spot is not one
of grid's three preset points, random has a structural advantage.

### Step 5 — Read `cv_results_` to see *why* random won (≈ 10 min)

Chapter 08's explanation is that most hyperparameters have a long tail
of importance — a few matter a lot, most matter little. Grid wastes
trials varying the unimportant ones; random does not. Confirm that on
this dataset by inspecting the per-hyperparameter marginals.

Add to `search.py`:

```python
def marginal_effect(cv_results: dict, param: str) -> pd.Series:
    """Mean CV AUC grouped by one hyperparameter — the 'marginal effect'
    of that hyperparameter while the others vary."""
    df = pd.DataFrame({
        "value": cv_results[f"param_{param}"],
        "score": cv_results["mean_test_score"],
    })
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.groupby(pd.qcut(df["value"], q=3, duplicates="drop"))["score"].mean()

print("\nGrid: mean CV AUC, grouped into terciles of each hyperparameter")
for p in grid_param_grid:
    print(f"  {p:18s}\n{marginal_effect(grid_search.cv_results_, p).to_string()}")

print("\nRandom: mean CV AUC, grouped into terciles of each hyperparameter")
for p in random_param_distributions:
    print(f"  {p:18s}\n{marginal_effect(random_search.cv_results_, p).to_string()}")
```

Run it.

**Expected output.** For `learning_rate` the terciles show a clear
monotone (or U-shaped) pattern — the hyperparameter actually matters.
For `max_leaf_nodes` and `min_samples_leaf` the terciles are much
closer together — those hyperparameters have smaller effect on this
dataset at the budgets tried.

**Checkpoint.** This is the empirical form of chapter 08's claim.
`learning_rate` is doing most of the work; the other two
hyperparameters are mostly along for the ride. Grid with 3 values per
axis spends 9 of its 27 trials at each of three `learning_rate` values;
random spends 27 trials at 27 different `learning_rate` values — a 9×
higher-resolution scan along the axis that actually matters.

### Step 6 — Encode the result as a comparison and a check (≈ 5 min)

Chapter 08 does not promise random beats grid on *every* run; it
promises random does at least as well on average, over many
replications, and does better when the important axis is continuous.
Make the lab's conclusion survive seed drift by averaging over three
random-search seeds (grid is deterministic).

Append to `search.py`:

```python
def random_search_with_seed(seed: int) -> float:
    rs = RandomizedSearchCV(
        make_estimator(),
        random_param_distributions,
        n_iter=TOTAL_TRIALS,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=seed,
        refit=False,
    ).fit(X, y)
    return rs.best_score_

random_scores_by_seed = [random_search_with_seed(s) for s in (0, 1, 2)]
random_mean           = float(np.mean(random_scores_by_seed))

print("\nComparison summary")
print(f"  Grid                   best CV AUC = {grid_search.best_score_:.4f}")
print(f"  Random (seeds 0,1,2)   per-seed    = "
      f"{[round(s, 4) for s in random_scores_by_seed]}")
print(f"  Random, mean           best CV AUC = {random_mean:.4f}")
print(f"  gap (random − grid)                = "
      f"{random_mean - grid_search.best_score_:+.4f}")

def test_random_at_least_matches_grid_on_average() -> None:
    """Over three seeds, average-case random search should not lose to grid
    on this deliberately-designed example. On a single seed either can win;
    the claim is about expected behaviour, not per-seed outcomes."""
    assert random_mean >= grid_search.best_score_ - 1e-4, (
        f"random lost to grid on average: random={random_mean:.4f} vs "
        f"grid={grid_search.best_score_:.4f}. Investigate before trusting "
        f"this lab — chapter 08 predicts random ≥ grid in expectation."
    )
    print("OK: random search matched or beat grid, averaged over 3 seeds.")

test_random_at_least_matches_grid_on_average()
```

Run it.

**Expected output.** Three per-seed random-search AUCs, their mean,
the signed gap against grid (usually positive), and one `OK` line.

**Checkpoint.** The assertion is deliberately stated as an *expected*
claim, not a per-seed guarantee. On a single seed, random can lose to
grid; averaged over several seeds on a problem where the important
axis is continuous, random wins. This is exactly the shape of
Bergstra & Bengio's result.

## Expected final artefacts

A single file `lab-03-random-vs-grid/search.py` containing the six
step blocks above. Running `python search.py` end-to-end should take
under a couple of minutes on a laptop and print (in order) the dataset
summary, the shared budget, the grid-search result, the random-search
result, the per-hyperparameter tercile tables for both, the
multi-seed comparison summary, and one `OK` line.

## Final checkpoint

Without re-reading chapter 08, answer:

1. In Step 3, grid searched exactly three values of `learning_rate`.
   If you doubled the grid to `np.logspace(-2.0, -0.5, 6)` (six
   values) while keeping the total budget at 27 trials, what would
   you have to give up in the other two hyperparameters? What does
   that trade-off tell you about why grid search does not scale past
   two or three hyperparameters?
2. In Step 4, the random distributions use `loguniform` for
   `learning_rate` but `randint` for the two integer hyperparameters.
   Why is a log-scale sampler right for `learning_rate` and not for
   `min_samples_leaf`? What does chapter 08's pitfall #2 say about
   using a linear grid over a log-scale hyperparameter?
3. In Step 5, `learning_rate`'s tercile means vary a lot and
   `min_samples_leaf`'s tercile means barely move. On a dataset
   where you *already know* `min_samples_leaf` is unimportant, what
   does chapter 08 recommend doing with the trials you would
   otherwise spend on it?
4. If you repeated this lab with `TOTAL_TRIALS = 1000` instead of
   `27`, would you expect the gap between grid and random to shrink,
   grow, or flip sign? Why? And at what budget would chapter 08
   suggest you reach for a smarter strategy (halving / Optuna)
   instead of either?

If you can answer all four, you have the method-selection reflex from
chapter 08 in your fingers — random first, grid for small refinements,
Optuna for serious budgets.

## Where to go next

- `exercise-03` ("XGBoost or LightGBM tuning with Optuna") replaces
  the random baseline here with a 50-trial `RandomizedSearchCV` and
  then adds Optuna at 200+ trials on top. The grid-vs-random reflex
  from this lab is the control case; the exercise demonstrates when
  Bayesian search earns its complexity budget.
- Chapter 08's `HalvingRandomSearchCV` section covers the next
  level: when fits are expensive, kill bad trials at low resource
  budgets and keep only the promising ones for full-budget evaluation.
  The lab's `make_estimator()` is cheap enough that halving does not
  pay off here, but the same budget split on a 100k-row LightGBM fit
  easily does.
