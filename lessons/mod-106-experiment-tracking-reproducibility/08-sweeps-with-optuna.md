# 08 — Hyperparameter sweeps with Optuna

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 02 (MLflow basics), 04 (seeds).
> **Cross-reference:** `mod-104` chapter 08 (`hyperparameter
> search`) covers the classical intuition.

## Motivation

You have one instrumented training run. You have a lockfile,
a data version, a seed you trust. Now you want to know:
which combination of hyperparameters gives the best model
you can build on this data?

Three answers scale with the problem:

- **Grid search** — try every combination. Great for two
  hyperparameters with three values each. Combinatorial
  explosion beyond that.
- **Random search** — sample random combinations. Beats
  grid search when only some hyperparameters matter (which
  is usually the case). Bergstra & Bengio's canonical result.
- **Sequential model-based optimisation** — learn from the
  trials you have run to propose the next ones. Optuna's
  default sampler (TPE) is this.

Optuna is the tool. MLflow tracks every trial. The output is
a *sweep* — a group of runs comparable on the same axes —
plus a winning configuration, ready for the registry
(chapter 07) and the report (chapter 09).

## What Optuna gives you

Two abstractions:

- **Study** — a named collection of trials for one
  optimisation problem. Studies live in a storage backend
  (in-memory, SQLite, PostgreSQL, MySQL, or Redis).
- **Trial** — one attempt with a specific set of
  hyperparameters. Each trial:
  1. Receives a `trial` object with `suggest_*` methods.
  2. Runs your training function.
  3. Reports a scalar objective (the "value" to optimise).
  4. Optionally reports intermediate values so the sampler
     can decide to prune early.

The sampler (`TPESampler` by default) chooses the next
trial's hyperparameters based on the outcomes of prior
trials. The pruner (`MedianPruner` by default) kills
under-performing trials early so budget goes to promising
ones.

## Hello, Optuna

```python
# tune.py
import optuna
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA = fetch_openml("adult", version=2, as_frame=True)
X, y = DATA.data, (DATA.target == ">50K").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=0
)


def objective(trial: optuna.Trial) -> float:
    params = dict(
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        max_depth     = trial.suggest_int("max_depth", 3, 12),
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        l2_regularization = trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        max_iter      = trial.suggest_int("max_iter", 100, 500, step=50),
    )
    clf = HistGradientBoostingClassifier(random_state=0, **params)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te))


study = optuna.create_study(
    study_name="adult-hgb",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=0),
    storage="sqlite:///optuna.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=50)

print("best:", study.best_trial.number, study.best_trial.value)
print(study.best_trial.params)
```

Run it once and you have a sweep. Read the key ideas:

- **`suggest_float(name, low, high, log=True)`** — sample
  on a log scale. Use log for things whose *ratio* matters
  (learning rate, regularisation).
- **`suggest_int(name, low, high, step=...)`** — discretised
  integer.
- **`suggest_categorical(name, choices)`** — one of a list.
- **`direction="maximize"`** — accuracy up, loss down would
  be `"minimize"`.
- **`sampler=TPESampler(seed=0)`** — deterministic sampler.
  Without a seed, two runs of the same sweep pick different
  hyperparameters. Chapter 04's rule applies to samplers too.
- **`storage="sqlite:///optuna.db"`** — persists the study
  across process restarts. In-memory is fine for a script;
  SQLite is the smallest step to a resumable sweep.

## The search space — practical rules

Chapter 8 of `mod-104` covers the intuition; here are the
Optuna specifics you will use every day:

- **Log-scale for learning rate, L1/L2 regularisation, and
  anything spanning orders of magnitude.** Passing
  `log=True` is worth being deliberate about — it changes
  the sampler's density.
- **Range endpoints matter.** `learning_rate ∈ [1e-3, 1.0]`
  is fine for tree boosters; `[1e-6, 1e-2]` is the deep-
  learning range. Setting the wrong range means the sampler
  spends its budget in a place the optimum isn't.
- **Conditional hyperparameters** — `suggest_*` inside an
  `if trial.suggest_categorical(...)` branch. Optuna handles
  this correctly; the search tree adapts to the categorical
  choice. Use when e.g. only the `Adam` branch has a
  `beta1` parameter.
- **Fixed hyperparameters** — anything you know you will not
  vary this sweep, hard-code outside `objective`. Don't
  waste a suggest_* call recording a constant.

## Pruning — stopping bad trials early

For deep-learning sweeps where each trial takes many
epochs, pruning is the single biggest cost saver. Report an
intermediate metric each epoch; the pruner can decide the
trial is not competitive with prior trials at the same step
and stop it:

```python
def objective(trial: optuna.Trial) -> float:
    model = build_model(trial)
    for epoch in range(EPOCHS):
        train_one_epoch(model, ...)
        val = evaluate(model, val_loader)
        trial.report(val, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
    ),
    sampler=optuna.samplers.TPESampler(seed=0),
)
```

`n_startup_trials=5` says "don't prune the first 5 trials —
we don't have enough baseline to judge against". `n_warmup_steps=3`
says "don't prune within the first 3 epochs of any trial —
performance is too noisy that early".

Prune aggressively when a trial takes minutes. Prune
conservatively (or not at all) when a trial takes seconds —
the overhead is not worth it.

## Wiring MLflow into the sweep

The pattern: one parent MLflow run for the whole study, one
child MLflow run per Optuna trial. Optuna ships an
`MLflowCallback` for the standard shape:

```python
from optuna.integration.mlflow import MLflowCallback

mlflow_cb = MLflowCallback(
    tracking_uri="http://mlflow.internal:5000",
    metric_name="val_accuracy",
    create_experiment=True,
    mlflow_kwargs={"nested": False},
)

study.optimize(objective, n_trials=50, callbacks=[mlflow_cb])
```

`MLflowCallback` writes one MLflow run per trial with:

- All the `suggest_*` values as `mlflow.log_params`.
- The trial's final objective as `mlflow.log_metric`.
- The trial number as a tag.

For richer instrumentation — logging every epoch's loss,
attaching the trained model artefact — call MLflow yourself
inside `objective`:

```python
def objective(trial):
    with mlflow.start_run(nested=True) as run:
        params = { "learning_rate": trial.suggest_float(...), ... }
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        clf = fit_model(**params)
        acc = accuracy_score(y_te, clf.predict(X_te))
        mlflow.log_metric("val_accuracy", acc)

        # Only log a model for the current best-so-far
        if trial.study.best_trial is None or acc > trial.study.best_value:
            mlflow.sklearn.log_model(clf, "model")

        return acc
```

Wrap the whole optimise call in an outer run to group the
trials:

```python
with mlflow.start_run(run_name="hgb-sweep-2026-08-01"):
    mlflow.log_param("n_trials", 50)
    study.optimize(objective, n_trials=50)
    mlflow.log_metric("best_val_accuracy", study.best_value)
    mlflow.log_dict(study.best_trial.params, "best_params.json")
```

Now the MLflow UI shows one parent run with 50 nested
children, and the parent's `best_params.json` artefact is
what you feed into the training script that produces the
model you register.

## Parallel and distributed sweeps

Optuna's SQL storage supports concurrent workers out of
the box:

```bash
# Terminal 1 (starts the sweep)
python tune.py --n_trials 25

# Terminal 2, 3, 4 (join in, share the same study)
python tune.py --n_trials 25
python tune.py --n_trials 25
python tune.py --n_trials 25
```

All four workers point at the same `storage="sqlite:///optuna.db"`
(or a Postgres URL for real scale). They coordinate via the
storage: each trial is claimed atomically, and the shared
history informs each worker's sampler.

Two practical notes:

- **SQLite works for a handful of local workers.** Beyond
  that, use Postgres or MySQL. SQLite's lock contention
  becomes the bottleneck.
- **Sampler seed with parallel workers.** The seed makes
  the *sampler* deterministic but the *ordering* of trial
  completion is not, because workers finish at different
  times. Reproducing a parallel sweep exactly requires
  serialising it (`n_jobs=1`, one process). For most
  purposes, the seeded sampler + fixed storage is enough
  to make results comparable across reruns.

For truly distributed sweeps (many nodes, GPU-heavy trials),
consider **Ray Tune**, **Katib**, or **W&B Sweeps** as
higher-scale alternatives. Optuna scales well vertically and
to a small number of nodes; those tools scale further.

<!-- needs-research: on the current Optuna release, verify
that the MLflowCallback and PyTorchLightning/scikit-learn
integrations still ship under `optuna.integration.*` or have
migrated to `optuna-integration`. -->

## Analysing a completed sweep

Optuna's built-in plotters cover the two views you will use
first:

```python
import optuna.visualization as vis

fig1 = vis.plot_optimization_history(study)
fig2 = vis.plot_param_importances(study)
fig3 = vis.plot_parallel_coordinate(study)
fig4 = vis.plot_slice(study, params=["learning_rate", "max_depth"])
```

- **Optimisation history.** Best value seen so far vs trial
  number. Confirms the sampler is improving; flatness
  early means you didn't run enough trials, flatness late
  means you converged.
- **Parameter importances.** Which hyperparameter mattered
  most for the objective. Optuna computes this via a small
  random forest fit on trial history. Not gospel — but
  useful for the next sweep's search space.
- **Parallel coordinates.** Every trial as a polyline
  across hyperparameter axes. Colours by objective. You can
  see clusters of good and bad configs.
- **Slice.** For one hyperparameter, the objective at each
  sampled value. Reveals plateaus and multi-modal responses.

Log the plots as MLflow artefacts:

```python
mlflow.log_figure(vis.plot_optimization_history(study), "plots/history.png")
mlflow.log_figure(vis.plot_param_importances(study),   "plots/importances.png")
```

Chapter 09's report includes at least the optimisation
history plot as a required figure.

## Reproducing a sweep

A sweep is reproducible when all of these are true:

- The `sampler` was constructed with a fixed `seed`.
- The `storage` is durable — SQLite or a real DB, not
  in-memory.
- The `objective` is deterministic (chapter 04's harness
  runs inside it).
- The data version is pinned (chapter 06).
- The environment is pinned (chapter 05).

Under those five conditions, running the same sweep script
against a fresh (or resumed) study produces the same trial
outcomes in the same order — with the caveat about parallel
workers above. Every trial's MLflow run has the same
`git_sha`, `data.md5`, and `docker_image_id` tags.

The winning configuration is then rerun *once* under the
registered-model pattern from chapter 07 — a clean final
training run that becomes the version pushed to the
registry.

## Choosing `n_trials`

The honest answer: enough that the last 10% show
diminishing returns. Practical starting points:

- **< 4 hyperparameters, seconds per trial (tree booster,
  sklearn Pipeline):** 100–200 trials.
- **4–8 hyperparameters, minutes per trial (small deep
  model):** 30–60 trials with pruning.
- **> 8 hyperparameters, tens of minutes to hours per
  trial:** 20–40 trials with aggressive pruning; consider
  reducing the search space first.

Watch the optimisation-history plot. If the best value is
still climbing at the end of your budget, you undersized.
If it's flat for the last third, you have a plausible
optimum.

## Seed sweeps — the sanity check on your winner

Chapter 04 flagged this. When Optuna hands you a winner,
retrain it under **3–5 different seeds** (chapter 04's
`seed_everything`) and record mean + std of the
validation metric. If the seed variance is larger than the
gap between the top two Optuna configurations, the "winner"
is within noise of the runner-up.

```python
seeds = [0, 1, 2, 3, 4]
metrics = []
for s in seeds:
    seed_everything(s, deterministic=True)
    clf = HistGradientBoostingClassifier(random_state=s, **best_params)
    clf.fit(X_tr, y_tr)
    metrics.append(accuracy_score(y_te, clf.predict(X_te)))

mean = statistics.mean(metrics)
std  = statistics.stdev(metrics)
print(f"val_accuracy: {mean:.4f} ± {std:.4f}")
```

Report both. Chapter 09's template has a slot for this
number.

## Common pitfalls

1. **In-memory storage in a "real" sweep.** The study
   dies when the process ends. `sqlite:///optuna.db` at
   minimum.
2. **Unseeded sampler.** Two runs of the "same" sweep
   suggest different trials; you cannot reproduce a
   result. Always `TPESampler(seed=...)`.
3. **Pruning without `n_startup_trials`.** The first
   trials get pruned before the pruner has any signal —
   you optimise on noise.
4. **Search space too wide.** `learning_rate ∈ [1e-9, 1e2]`
   wastes budget in the tails. Narrow after the first sweep.
5. **Objective returns a tensor, not a float.** Optuna
   expects a Python `float`. Convert with `.item()`.
6. **Retraining the winner without a fresh seed.** The
   sweep's best trial ran with one seed; retraining with
   the same seed and same data is not new information.
   Sweep multiple seeds; report the distribution.
7. **Nested `mlflow.start_run` without cleanup.** If the
   trial crashes without the context manager finishing, the
   nested run stays `RUNNING`. Always use `with`.

## Summary

- Optuna sweeps hyperparameters with a sampler smarter than
  random (TPE by default) and a pruner that kills bad trials
  early.
- One parent MLflow run + one child per trial. Optuna's
  `MLflowCallback` is the starting point; roll your own
  when you need per-epoch metrics or artefact logging.
- Seed the sampler, use durable storage, and keep the
  objective deterministic — otherwise the sweep isn't
  reproducible even if the training loop is.
- Parallel workers coordinate via shared storage. SQLite for
  a handful of workers; Postgres beyond that. Ray Tune /
  Katib / W&B Sweeps scale further.
- Analyse with `plot_optimization_history`,
  `plot_param_importances`, `plot_parallel_coordinate`, and
  `plot_slice`. Log the plots as MLflow artefacts.
- Confirm the winner with a **seed sweep**; report mean ±
  std. If the gap to runner-up is within one std, they are
  tied.

## Where this goes next

Chapter 09 pulls everything together into the deliverable
that makes the module's work visible: the **reproducibility
report** — the document a teammate uses to rerun your best
result from scratch.
