# 08 — Hyperparameter search: grid, random, and Optuna

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 06, 07.

## Motivation

Chapter 06 named the five hyperparameters that decide whether a
gradient booster reaches its potential. Chapter 07 defined the CV
loop that scores a configuration honestly. This chapter is the
outer loop: which configurations do you try, in what order, and
when do you stop?

The naïve answer — grid search over a Cartesian product of every
plausible value — is what most tutorials show. It is also what
wastes the most engineer time. This chapter argues for three moves,
in increasing sophistication:

1. **Random search** for the first exploration (cheap, embarrassingly
   parallel, often good enough).
2. **Halving search** when fits are expensive and the search space
   is large.
3. **Optuna** (or another Bayesian / TPE-based search) when you
   have a serious compute budget and want efficient use of it.

Grid search still has its place — for orthogonal 2-D scans and for
final refinement — but "grid search over 5 hyperparameters" is a
plan that will spend a week for a result random search would have
found in an afternoon.

## The tuning contract

Every method in this chapter fits the same contract:

- Given an estimator (or a `Pipeline`), a parameter space, a CV
  strategy, and a scoring metric, produce the best configuration
  and its score.
- Report the CV score for the winning configuration, and — if you
  are being honest — a separately-locked test-set score for the
  winner.
- The winning configuration's CV score is slightly optimistic
  because you have selected on the same data. The larger the
  search space and the more configurations tried, the more
  optimistic (this is called the *winner's curse* or *optimism
  bias*).

Two consequences:

- The point of a search is *finding a good configuration*, not
  *estimating its performance*. Estimate performance on a held-out
  set the search never touched.
- If your search has 50 000 configurations and 50 folds, some
  configuration will look great by chance. Prefer smaller search
  spaces over larger ones with more folds.

## Grid search: use sparingly

`GridSearchCV` tries every combination in a Cartesian product:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

param_grid = {
    "clf__C":          [0.01, 0.1, 1, 10, 100],
    "clf__penalty":    ["l1", "l2"],
    "clf__solver":     ["liblinear", "saga"],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

Cost: `len(grid) × cv` fits. Cheap on 2–3 hyperparameters, exponential
in the rest. The right use cases:

- **Small orthogonal 2-D scans** (e.g. `alpha` × `l1_ratio` for
  elastic-net, `n_neighbors` × `weights` for KNN). Grid gives you
  the full surface for plotting.
- **Discrete choice combinations** where every combination is
  meaningful (e.g. `penalty × solver`, since some solvers don't
  support all penalties).
- **Final refinement** around the best random-search config.

Avoid grid for:

- More than three continuous hyperparameters.
- Wide continuous ranges (`learning_rate ∈ [0.001, 0.5]`); log-
  spaced grids sample too coarsely.
- Anything you would call "exploration".

## Random search: the default first pass

Bergstra and Bengio (2012) showed that for most problems, random
search over the same total number of trials as grid search finds
better configurations. The reason: most hyperparameters have a
"long tail" of importance — a few matter a lot, most matter little.
Grid search wastes trials varying the unimportant ones; random
search does not.<!-- needs-research: verify against Bergstra &
Bengio 2012, "Random Search for Hyper-Parameter Optimization",
JMLR: https://www.jmlr.org/papers/v13/bergstra12a.html. -->

`RandomizedSearchCV` samples from distributions:

```python
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "clf__C":               loguniform(1e-4, 1e2),
    "clf__l1_ratio":        [0.0, 0.25, 0.5, 0.75, 1.0],
    "features__num__impute__strategy": ["median", "mean"],
    "features__lowc__ohe__min_frequency": randint(5, 100),
}
search = RandomizedSearchCV(
    pipe,
    param_distributions,
    n_iter=50,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=0,
    verbose=1,
).fit(X_train, y_train)
```

Two habits:

- **Use `loguniform` for scales that span orders of magnitude.**
  `C`, `alpha`, `learning_rate`, `reg_lambda` — none of these vary
  linearly. Sampling `loguniform(1e-4, 1e2)` spreads trials evenly
  across log-space.
- **Set `random_state` and log the trials.** Reproducibility;
  makes it easy to resume a search or share a result.

`n_iter=50` is a reasonable first pass; bump to 200 if the fits
are cheap. Almost always better than an equivalently-large grid.

## Successive halving: for expensive estimators

Halving search fits many configurations on a small subset of the
data (or a small number of boosting rounds), keeps the top few,
and re-runs them on more data. Cheap trials early, expensive
trials only for the promising configurations.

`sklearn` ships `HalvingGridSearchCV` and `HalvingRandomSearchCV`
(experimental since 0.24, promoted since 1.4).

```python
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import loguniform, randint

search = HalvingRandomSearchCV(
    pipe,
    param_distributions,
    factor=3,                # keep top 1/factor at each iteration
    resource="n_samples",    # or a hyperparameter like "clf__n_estimators"
    max_resources="auto",
    n_candidates="exhaust",  # spend the full budget
    cv=5,
    scoring="roc_auc",
    random_state=0,
    n_jobs=-1,
).fit(X_train, y_train)
```

Key options:

- **`resource`** — what to scale up. `"n_samples"` (subset the
  training data), or the name of a hyperparameter that measures
  compute (e.g. `"clf__n_estimators"` for a booster).
- **`factor`** — how aggressively to eliminate. `factor=3` keeps
  the top 1/3 of candidates each round. Default 3 is a good
  balance.
- **`min_resources` / `max_resources`** — the starting and ending
  budget. `"auto"` picks sensible defaults from the dataset size
  or the estimator.

When to reach for it:

- Fits are slow (large data, complex estimator).
- Search space is large (100+ candidates).
- You do not need pinpoint accuracy on the score.

When to skip:

- Fits are cheap. The scheduling overhead is not worth it.
- Small dataset. Subsampling further defeats the purpose.

## Optuna: Bayesian / TPE-based search

`Optuna` is a general hyperparameter optimisation library. It samples
configurations using an adaptive strategy (default: TPE, Tree-
structured Parzen Estimator), which uses the history of past trials
to bias sampling toward promising regions. On problems where the
score surface has structure, TPE finds better configurations in
fewer trials than random search.

Basic pattern:

```python
import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import cross_val_score, StratifiedKFold

def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
    }
    model = LGBMClassifier(
        **params,
        n_estimators=5000,
        n_jobs=-1,
        random_state=0,
        verbose=-1,
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    return scores.mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(study.best_params, study.best_value)
```

The `trial.suggest_*` functions define the search space *in code*,
which lets you use conditional logic (e.g. only suggest
`num_leaves` if `boosting_type == "gbdt"`), enforce constraints,
and integrate with early stopping. This is the biggest single
ergonomic advantage over `sklearn`'s search classes.

### Pruning

Optuna's `Pruner` API kills bad trials early. For iterative
estimators, this can be a factor of 3–10 speedup:

```python
from optuna.integration import LightGBMPruningCallback

def objective(trial):
    ...
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(50),
            LightGBMPruningCallback(trial, "auc"),
        ],
    )
    ...

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
)
```

The pruner reports each boosting round's score and stops the trial
if it looks unlikely to beat the median of running trials at the
same step. Combines well with early stopping.

Analogous pruning callbacks exist for XGBoost, CatBoost, and PyTorch.

### Storage and resumption

`Optuna` studies can be persisted to SQLite or Postgres:

```python
study = optuna.create_study(
    study_name="lgbm-adult",
    storage="sqlite:///optuna.db",
    load_if_exists=True,
    direction="maximize",
)
study.optimize(objective, n_trials=100)
```

Resume by re-running the same script; new trials append to the
study. Multiple workers can `.optimize` the same storage
concurrently for distributed search.

### Alternatives to Optuna

- **`hyperopt`** — the original TPE library. Similar API; less
  maintained.
- **`scikit-optimize`** — Gaussian-process-based search. Great for
  cheap objectives, slower to converge on 20+ hyperparameters.
- **`ray[tune]`** — distributed hyperparameter search; a good
  choice when you need to tune across many GPU workers. Chapter
  8's territory more than this module's.

For a single-machine tabular problem, Optuna is the default.

## Searching over preprocessing too

The pipeline-level search — where hyperparameters live in both the
preprocessor and the estimator — is what makes CV honest. Use the
`__` (double underscore) notation to reach into a pipeline step:

```python
param_distributions = {
    "features__num__impute__strategy":         ["median", "mean", "most_frequent"],
    "features__lowc__ohe__min_frequency":      randint(5, 100),
    "features__hic__te__smooth":               loguniform(1, 100),
    "clf__C":                                  loguniform(1e-4, 1e2),
    "clf__penalty":                            ["l1", "l2"],
}
```

This is where you catch things like "median imputation happens to
work much better than mean on this dataset", which no amount of
model-only tuning would find.

## Nested CV, once more with feeling

For the tuning setup above, the fitted `search.best_score_` is the
best CV score across the search — optimistic by construction. The
honest way to estimate generalisation is nested CV: an outer CV
loop where each fold's training set is passed to the search, and
the returned best model is scored on the outer validation fold.

```python
from sklearn.model_selection import cross_val_score

outer_cv = StratifiedKFold(5, shuffle=True, random_state=42)
nested_scores = cross_val_score(
    search, X_train, y_train, cv=outer_cv, scoring="roc_auc",
    n_jobs=-1,
)
print(f"Nested CV AUC: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

Cost: `n_outer × n_trials × n_inner` fits. Very expensive; usually
not run in production tuning cycles. The pragmatic compromise:

1. Do the search on a training set.
2. Report the search's CV score, acknowledging that it is slightly
   optimistic.
3. Report the *test-set* score of the best model on a locked
   hold-out that neither the search nor its CV loop touched.

The gap between those two numbers is your winner's curse. If it is
small (< 1 percentage point), the search is well-calibrated. If it
is large, either the search space was too big or the CV was noisy.

## Tuning budgets: rough numbers

For a medium-sized tabular problem (~100k rows, ~50 features), the
budget that usually pays off:

| Model | Method | Trials | Wall clock (rough) |
|---|---|---|---|
| Logistic regression | `LogisticRegressionCV` | (path) | seconds |
| Ridge / Lasso | `*CV` variants | (path) | seconds |
| Random forest | Random search | 20–50 | minutes |
| Gradient boosting (first pass) | Optuna TPE + pruning | 100–300 | tens of minutes |
| Gradient boosting (production) | Optuna TPE + pruning | 500–2000 | hours |

The right number is not a formula. Stop when a doubling of the
budget improves the CV score by less than the standard error of the
CV estimate. If your CV std is 0.005 and the last 100 trials
improved the score by 0.002, you are burning compute for noise.

## Reporting the search

Log at minimum:

- The parameter space (as code, not prose).
- The number of trials and the CV strategy.
- The best configuration and its CV score.
- The held-out test score of the best configuration.
- A histogram / table of the top-10 configurations. If the best is
  a lucky outlier, that will show up here.

For Optuna, `optuna.visualization.plot_optimization_history(study)`
and `plot_param_importances(study)` are two-line diagnostics worth
having in every `REPORT.md`.

## Pitfalls collected

1. **Grid over five continuous hyperparameters.** `5^5 = 3125`
   configurations before you even set `cv=5`. Random or halving
   or Optuna.
2. **Linear grid over a log-scale hyperparameter.** `[0.001, 0.01,
   0.1, 1, 10]` linearly-spaced misses everything in the low end;
   `loguniform` finds it.
3. **Tuning on the test set.** Any workflow where the test set
   score influences your choice of hyperparameters is
   hyperparameter tuning on the test set. Lock the test set at
   split time and open it once, at the end.
4. **CV inside the search using the same random state as the
   outer split.** Deterministic overlap, silently optimistic. Set
   independent random states.
5. **Reporting `search.best_score_` as your model's performance.**
   That is a selected-best CV score, not a hold-out estimate. See
   above for nested CV / hold-out.
6. **Optuna without a `study_name` or `storage`.** You cannot
   resume, share, or restart. Set both.
7. **Optuna with `n_jobs > 1` in the objective and again in
   `optimize`.** You have `n_jobs^2` parallelism, and your CPU
   scheduler is thrashing. Pick one level to parallelise.
8. **Ignoring `study.best_params` and trying the "one you liked".**
   If a search picked a configuration you disagree with, the fix
   is a smaller / better-scoped search space, not overriding the
   result.

## Summary

- Grid search: for small orthogonal scans and final refinement.
- Random search: default first pass. Cheap, embarrassingly
  parallel, usually beats grid on the same budget.
- Halving search: for expensive estimators with large search
  spaces. Cheap trials early, expensive trials only for the
  promising.
- Optuna: for serious budgets. TPE + pruning + persistent storage
  make it the sharpest single-machine tool.
- Search on log-scaled distributions for anything spanning orders
  of magnitude. Search over the whole `Pipeline`, not just the
  estimator.
- The tuning score is optimistic; the honest number is the held-out
  test set. Report both.

## Where this goes next

Chapters 09 and 10 are the interpretation arc. Chapter 09 catalogues
the failure modes of naive feature importance and feature
selection. Chapter 10 covers SHAP — what it computes, when it is
reliable, and how to defend the numbers to a stakeholder.
