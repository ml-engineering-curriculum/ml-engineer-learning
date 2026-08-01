# exercise-03 — XGBoost or LightGBM tuning with Optuna

> **Estimated effort:** 4 hours.
> **Companion chapters:** [06 — Gradient-boosted trees](../06-gradient-boosted-trees.md), [07 — Cross-validation strategies](../07-cross-validation-strategies.md), [08 — Hyperparameter search](../08-hyperparameter-search.md).

## Objective

Take one gradient-boosting model — your choice of XGBoost or
LightGBM — from "default settings" to "tuned to convergence" using
Optuna. You will:

- Establish a defaults-only baseline with early stopping.
- Run a random-search baseline as a comparison for Optuna.
- Design an Optuna study with a well-scoped search space and
  pruning.
- Track and plot the optimisation history and per-hyperparameter
  importance.
- Report the tuning outcome honestly — the winner's-curse gap
  between the best CV score and the held-out test score.

The goal is a tuned model *and* a defensible tuning report. "I ran
Optuna for 500 trials and this is the number" is not enough. You
should be able to answer, from the study, which hyperparameters
actually mattered and where the model settled inside your search
space (near the boundary? near the middle?).

## Prerequisites

- Chapters 06, 07, 08 of this module.
- Python 3.11+, scikit-learn ≥ 1.4, Pandas ≥ 2.1, Optuna ≥ 3.5,
  `matplotlib`, `pytest`.
- Either `xgboost` ≥ 2.0 or `lightgbm` ≥ 4.0.
- Optional but recommended: SQLite (for Optuna study persistence).

## Dataset

Use one of:

- **UCI Adult** — same as exercise-02. Modest size (~48k rows),
  binary classification.
- **HIGGS** subset — larger, useful if you want to see the runtime
  matter more.
- **Your own** — any binary or multiclass classification with
  100k+ rows is fine.

Pick one and split into three parts: `X_train`, `X_val`, `X_test`.
Use `X_val` for early stopping and `X_train + X_val` for CV during
the tuning. Reserve `X_test` for the final held-out score, opened
exactly once.

## Problem statement

### Part A — the defaults baseline (≈ 30 min)

Load and split the data 60/20/20 into train / val / test.

Fit your chosen model with **library defaults** except:

- `n_estimators=5000` (a large cap; early stopping will trim).
- Early stopping on `(X_val, y_val)` with `stopping_rounds=50`.
- `random_state=0`.
- `eval_metric="auc"` (or the natural default for your library).

Report:

- Best iteration (`.best_iteration_` / `.best_iteration`).
- Fit time.
- CV AUC on `X_train + X_val` (5-fold stratified).
- Test AUC on `X_test`.

This is your defaults baseline. Every tuned result must beat it,
or the tuning was a waste of compute.

### Part B — the random-search baseline (≈ 40 min)

Before Optuna, run a `RandomizedSearchCV` with 50 trials over the
same search space you will use in Optuna (spec below in Part C).
Report:

- Best CV AUC.
- Best hyperparameters.
- Test AUC of the best config.
- Wall-clock time.

This is the "control" you compare Optuna against. TPE should
usually beat random search at the same budget, but not always —
and knowing the gap matters when you decide whether the extra
complexity of Optuna is worth it in your project.

### Part C — design the search space (≈ 30 min)

Before running Optuna, write out the search space in
`search_space.py` as a function `def suggest_params(trial) -> dict`.
Include, at minimum:

**For LightGBM:**

- `learning_rate` — `loguniform(1e-3, 0.3)`
- `num_leaves` — `int(loguniform(15, 255))`
- `min_child_samples` — `int(uniform(5, 100))`
- `subsample` — `uniform(0.5, 1.0)` with `subsample_freq=1`
- `colsample_bytree` — `uniform(0.5, 1.0)`
- `reg_alpha` — `loguniform(1e-8, 10)`
- `reg_lambda` — `loguniform(1e-8, 10)`

**For XGBoost:**

- `learning_rate` — `loguniform(1e-3, 0.3)`
- `max_depth` — `int(uniform(3, 12))`
- `min_child_weight` — `loguniform(1, 100)`
- `subsample` — `uniform(0.5, 1.0)`
- `colsample_bytree` — `uniform(0.5, 1.0)`
- `reg_alpha` — `loguniform(1e-8, 10)`
- `reg_lambda` — `loguniform(1e-8, 10)`
- `gamma` — `loguniform(1e-8, 10)`

`n_estimators` is *not* in the search space — it is set high
(5000) and controlled by early stopping inside each trial.

In `REPORT.md`, argue for each range. Why the bounds you picked?
Where does the default sit inside your range? (If the default is
at the edge of your range, expand the range.)

### Part D — the Optuna study (≈ 60 min)

Write `tune.py`:

```python
import optuna
from lightgbm import LGBMClassifier, early_stopping
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold, cross_val_score

def objective(trial):
    params = suggest_params(trial)
    model = LGBMClassifier(
        **params,
        n_estimators=5000,
        random_state=0,
        n_jobs=-1,
        verbose=-1,
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(
        model,
        X_trainval,
        y_trainval,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,        # inner uses n_jobs=-1
    )
    return scores.mean()

study = optuna.create_study(
    study_name="mod104-ex03",
    storage="sqlite:///optuna.db",
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=0),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
)
study.optimize(objective, n_trials=200, show_progress_bar=True)
```

Run for **at least 200 trials**. If time permits, extend to 500 or
1000 (the SQLite storage means you can stop and resume freely).

Report:

- Best CV AUC.
- Best hyperparameters (as a dict).
- Number of trials completed.
- Number of trials pruned (if using a pruner).
- Test AUC of the best config (retrained on `X_train + X_val`).

**Reproducibility check.** Fix `random_state` throughout and confirm
that a second run of `study.optimize` from an empty study reproduces
the top-5 configurations bit-for-bit. TPE with a fixed seed should
be deterministic given the same objective.

### Part E — diagnostics from the study (≈ 45 min)

The point of Optuna over random search is not just the final
score — it's the story of the search. Produce these plots
(`optuna.visualization` module):

1. **Optimisation history.** `optuna.visualization.plot_optimization_history(study)`
   — best-so-far score vs. trial number. Is the score still
   improving at trial 200? If yes, more trials will help.
2. **Parameter importance.** `optuna.visualization.plot_param_importances(study)`
   — which hyperparameters mattered? Anything scoring near zero
   can be removed from the search space next time (freeing budget
   for the ones that matter).
3. **Slice plot.** `optuna.visualization.plot_slice(study)` — CV
   score as a function of each hyperparameter individually, holding
   others aside. Reveals where the sweet spot is for each.
4. **Parallel coordinates plot.**
   `optuna.visualization.plot_parallel_coordinate(study,
   params=["learning_rate", "num_leaves", "min_child_samples"])` —
   shows the joint distribution of top configurations.

Paste each plot in `REPORT.md`. For each, write **one sentence**
about what it tells you. Example bad answers: "the plot shows
optimisation history." Example good: "The best score plateaus after
trial ~120; further trials likely give < 0.001 AUC improvement."

### Part F — the honest comparison (≈ 20 min)

Assemble in `REPORT.md`:

| Configuration | Best iter | CV AUC | Test AUC | Test AUC – CV | Wall clock |
|---|---|---|---|---|---|
| Defaults | | | | | |
| RandomizedSearchCV (50) | | | | | |
| Optuna (200 trials) | | | | | |

The "Test AUC – CV" column is the winner's curse. On a healthy
search, it should be small (< 0.005). If it is large (> 0.02), the
search space was too big for the tuning budget or the CV was noisy.

Answer in prose:

1. Was Optuna worth it on this dataset, relative to random search
   at the same budget?
2. Which hyperparameter drove the biggest improvement — the one
   Optuna concentrated its trials on? Would you have guessed?
3. If you had another 5x the compute budget, where would you spend
   it: more Optuna trials, more CV folds, or a bigger search space?

## Starter guidance

- **`n_jobs` conflict.** If your Optuna objective uses
  `cross_val_score(..., n_jobs=-1)` *and* Optuna uses parallel
  trials (`n_jobs=-1` in `optimize`), you thrash the CPU. Pick
  one level to parallelise. Cross-validation across folds is
  usually the safer choice.
- **Pruning callbacks.** `LightGBMPruningCallback(trial, "auc")` and
  `XGBoostPruningCallback(trial, "validation_0-auc")` report each
  boosting round's validation score to the pruner. Only used inside
  a `.fit()` call that has an `eval_set`; combine with early
  stopping.
- **Save the study.** SQLite storage lets you kill and resume. If
  you're on a machine you can't leave running for hours, use
  `storage="sqlite:///optuna.db"` from the start.
- **Optuna's TPE is deterministic with a seed.** But
  `LGBMClassifier`'s bootstrapping (`subsample`) is not, unless
  you also seed it (`random_state=0`). Fix both.
- **Do not tune `n_estimators` in the search space.** Use early
  stopping. Otherwise Optuna will spend most of its budget
  searching a parameter you can determine automatically.

## Acceptance criteria

- Data split into train / val / test with `X_test` untouched until
  the very end.
- Defaults baseline, random-search baseline, and Optuna study all
  run and reported.
- Optuna study has at least 200 trials and uses TPE + pruning.
- Search space is defined as a callable and documented in
  `REPORT.md` with ranges justified.
- All four `optuna.visualization` plots (history, importance,
  slice, parallel coords) are included in `REPORT.md` with a
  one-sentence interpretation each.
- Final table filled in with defaults, random search, and Optuna
  entries.
- Winner's-curse gap (test − CV) reported and discussed.
- Prose in Part F answered.
- Optuna study is persisted (`optuna.db` or equivalent) and can be
  resumed by running the script again.

## Stretch goals (optional)

- **Multi-objective tuning.** Use `optuna.create_study(directions=["maximize",
  "minimize"])` to optimise AUC *and* the number of leaves (a
  proxy for model size / serving cost). Report the Pareto front.
- **Search the categorical-handling strategy.** Add
  `trial.suggest_categorical("cat_strategy", ["native", "target_encoded",
  "one_hot"])` and let Optuna decide. The winning strategy is often
  dataset-dependent.
- **Cross-model tuning.** Extend the search to choose *which
  gradient-booster* (XGBoost / LightGBM / CatBoost) to use. A
  conditional suggest based on `trial.suggest_categorical("model",
  [...])`. Do the three converge to similar CV scores when equally
  tuned?
- **Halving as a warm start.** Run `HalvingRandomSearchCV` first
  to eliminate obviously-bad configurations, then Optuna on the
  survivors. Faster to a good result than Optuna from scratch.
- **Nested CV.** Wrap Optuna's `.optimize` inside a
  `cross_val_score(outer)` for a fully-honest estimate. Very
  expensive (`n_outer × n_trials × n_inner` fits), but the
  resulting number has no winner's curse. Report the gap between
  nested-CV and the single-CV Optuna score.

## What to hand in

```
exercise-03/
├── data.py                # load + train/val/test split
├── search_space.py        # suggest_params(trial) -> dict
├── baselines.py           # defaults + RandomizedSearchCV
├── tune.py                # Optuna study
├── diagnostics.py         # generate the four visualisation plots
├── evaluate.py            # comparison table
├── tests/
│   ├── test_search_space.py  # suggest_params returns a valid dict
│   └── test_reproducibility.py  # study reproduces top-5 from fixed seed
├── plots/                 # PNG output for REPORT.md
├── optuna.db              # persisted study (not committed if large)
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
