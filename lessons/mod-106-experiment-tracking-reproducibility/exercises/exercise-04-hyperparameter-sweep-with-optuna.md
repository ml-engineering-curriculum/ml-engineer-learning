# exercise-04 — Hyperparameter sweep with Optuna

> **Estimated effort:** 3 hours.
> **Companion chapters:** [08 — Hyperparameter sweeps with Optuna](../08-sweeps-with-optuna.md), [09 — The reproducibility report](../09-the-reproducibility-report.md).

## Objective

Combine everything from exercises 01–03 into a
production-shaped workflow:

1. Run a small **Optuna sweep** over the training script,
   with every trial captured as a nested MLflow run.
2. Validate the winning configuration with a **seed sweep**
   (chapter 04's discipline) and register the winning model
   in the **Model Registry** (chapter 07).
3. Write a **`REPORT.md`** following the template in
   chapter 09, complete enough that a teammate can rerun
   your best result from git SHA + report alone.

This is the capstone exercise for the module. If a teammate
can execute your `REPORT.md`'s reproduce block and land
within the stated tolerance of your headline metric, the
module has delivered its outcome.

## Prerequisites

- Exercises 01, 02, and 03 complete: an instrumented,
  data-versioned, deterministic training script.
- `uv add optuna optuna-integration` (or `pip install
  optuna optuna-integration`).
- Chapters 08 and 09 of this module.

## Problem statement

### Part A — Refactor `train()` to accept a config (≈ 20 min)

Your training script's core function should take a plain
dict / `dataclass` of hyperparameters, not read them from
the CLI directly. Something like:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    seed:          int   = 0
    learning_rate: float = 0.1
    max_depth:     int   = 6
    max_leaf_nodes: int  = 31
    l2:            float = 0.0
    max_iter:      int   = 200


def train(cfg: TrainConfig, data) -> tuple["Model", float]:
    """Train a model under cfg. Return (model, val_metric)."""
    ...
```

The CLI entry point constructs a `TrainConfig` from
arguments and calls `train(cfg, data)`. Optuna will do the
same, constructing configs from trial suggestions.

Requirements:

1. `train(cfg, data)` is deterministic when
   `seed_everything(cfg.seed, deterministic=True)` is called
   first (exercise-03's guarantee).
2. `train` returns both the model and the validation metric
   as a float.
3. The CLI still works exactly as before; the sweep uses the
   same function, no divergence.

### Part B — The Optuna objective (≈ 25 min)

Create `tune.py`:

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import mlflow

from src.train import TrainConfig, train
from src.data  import load_data
from utils.reproducibility import seed_everything


DATA = load_data()   # cache across trials


def objective(trial: optuna.Trial) -> float:
    cfg = TrainConfig(
        seed          = 0,        # fixed for the sweep — see Part C
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        max_depth     = trial.suggest_int("max_depth", 3, 12),
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        l2            = trial.suggest_float("l2", 1e-8, 10.0, log=True),
        max_iter      = trial.suggest_int("max_iter", 100, 500, step=50),
    )

    with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
        mlflow.log_params(cfg.__dict__)
        mlflow.log_param("trial_number", trial.number)

        seed_everything(cfg.seed, deterministic=True)
        _, val_metric = train(cfg, DATA)

        mlflow.log_metric("val_metric", val_metric)
    return val_metric


if __name__ == "__main__":
    mlflow.set_experiment("mod106-sweep")

    with mlflow.start_run(run_name="optuna-sweep") as parent:
        study = optuna.create_study(
            study_name="mod106-hgb",
            direction="maximize",
            sampler=TPESampler(seed=0),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            storage="sqlite:///optuna.db",
            load_if_exists=True,
        )
        mlflow.log_params({
            "n_trials":    50,
            "sampler":     "TPE",
            "sampler_seed": 0,
            "pruner":      "MedianPruner",
        })

        study.optimize(objective, n_trials=50)

        mlflow.log_metric("best_val_metric", study.best_value)
        mlflow.log_dict(study.best_trial.params, "best_params.json")
```

Requirements:

1. The sweep runs at least **30 trials** — adjust based on
   the cost of a single training run on your machine (see
   chapter 08's "choosing n_trials" section).
2. The sampler is seeded (`TPESampler(seed=0)`).
3. Storage is a durable backend
   (`sqlite:///optuna.db` at minimum; not in-memory).
4. Each trial is a nested MLflow run under one parent run
   named `"optuna-sweep"`.
5. `best_params.json` is logged as an artefact on the
   parent run.

Applicable to deep-learning tasks: add pruning by calling
`trial.report(val, step=epoch)` and
`if trial.should_prune(): raise optuna.TrialPruned()` inside
your training loop.

### Part C — Analyse the sweep (≈ 25 min)

In a notebook or `analyze.py`, produce and log:

- `plot_optimization_history(study)` → parent artefact
  `plots/optimization_history.html`.
- `plot_param_importances(study)`   → parent artefact
  `plots/param_importances.html`.
- `plot_parallel_coordinate(study)` → parent artefact
  `plots/parallel_coordinate.html`.

Optuna's built-in visualisation returns `plotly` figures.
Save as HTML (interactive) and log both HTML and PNG for
convenience:

```python
import optuna.visualization as vis
import mlflow

for name, fig in [
    ("optimization_history", vis.plot_optimization_history(study)),
    ("param_importances",    vis.plot_param_importances(study)),
    ("parallel_coordinate",  vis.plot_parallel_coordinate(study)),
]:
    fig.write_html(f"plots/{name}.html")
    fig.write_image(f"plots/{name}.png")
mlflow.log_artifacts("plots", artifact_path="plots")
```

Include these plots in the report.

### Part D — Seed sweep on the winner (≈ 20 min)

Take `study.best_params` and retrain **5 times** under
different seeds:

```python
import statistics

best_params = study.best_params
seeds = [0, 1, 2, 3, 4]
metrics = []
for s in seeds:
    cfg = TrainConfig(seed=s, **best_params)
    with mlflow.start_run(nested=True, run_name=f"winner-seed-{s}"):
        mlflow.log_params(cfg.__dict__)
        mlflow.log_param("seed_sweep", True)
        seed_everything(cfg.seed, deterministic=True)
        _, val_metric = train(cfg, DATA)
        mlflow.log_metric("val_metric", val_metric)
        metrics.append(val_metric)

mean = statistics.mean(metrics)
std  = statistics.stdev(metrics)

mlflow.log_metric("winner.mean", mean)
mlflow.log_metric("winner.std",  std)
```

If `std > (best - second_best) / 2` — i.e. the seed variance
is larger than half the gap to runner-up — flag this
explicitly in the report. Your "winner" is within noise of
the runner-up; that is a real finding.

### Part E — Register the champion (≈ 20 min)

Reuse the pattern from exercise-02. Take the seed that gave
the *median* validation metric (not the max — the max is
lucky, the median is representative) and retrain **that**
config once more, this time logging + registering the model:

```python
median_seed = sorted(zip(metrics, seeds))[len(seeds)//2][1]
cfg = TrainConfig(seed=median_seed, **best_params)

with mlflow.start_run(run_name="champion-registration") as champ:
    mlflow.log_params(cfg.__dict__)
    seed_everything(cfg.seed, deterministic=True)
    model, val_metric = train(cfg, DATA)
    mlflow.log_metric("val_metric", val_metric)

    signature = mlflow.models.signature.infer_signature(...)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name="mod106-champion",
    )
```

Then set the `@champion` alias on the new version and tag it
with `git_sha`, `data.md5`, and `val_metric` (exercise-02
pattern).

### Part F — Write `REPORT.md` (≈ 60 min)

Follow the template in chapter 09 to the letter. At
minimum, `REPORT.md` must contain:

- All fields from the frontmatter block (author, date, best
  result with mean±std, MLflow run URL, registered model
  URI).
- A **reproduce block** — copy-pasteable commands that
  someone else could run to land within `±std` of your
  best result.
- Data section with DVC hash (from exercise-02) and split
  sizes.
- Model section with framework, versions, hyperparameters,
  and signature URI.
- Training procedure — sweep parameters, winner, seed
  sweep, hardware, wall-clock.
- Environment section — Python, container digest, lockfile
  SHA, GPU info (if any).
- Determinism claim — the tier you achieved and the
  tolerance (reuse from `DETERMINISM.md` in exercise-03).
- Known variance sources you did not eliminate.
- Troubleshooting section.

At least one figure from Part C must be embedded (screenshot
or committed PNG).

### Part G — The rerun test (≈ 20 min)

Prove the report works. Two options — do at least **one**:

- **Colleague test.** Send `REPORT.md` to someone (a
  study partner, a friend, an LLM you have never given the
  project context to) and ask them to follow the reproduce
  block on a fresh machine. Any question they ask goes back
  into the report as a missing sentence.
- **Fresh-checkout test.** Clone the repo to a scratch
  directory, delete `~/.cache/uv`, delete
  `~/.dvc/cache`, follow the reproduce block yourself as if
  you had never seen it. Time yourself; note any command
  you had to reconstruct from memory.

Add a "Rerun test" section to `REPORT.md`:

```markdown
## Rerun test

Performed by: {{colleague name or self}}
On: {{date}}
Hardware: {{describe}}

Reproduce block ran to completion in {{minutes}} minutes.
Final `val_metric` = {{value}} vs. reported {{value}} (Δ =
{{delta}}, within tolerance ✓).

Issues encountered and fixed in the report:
1. {{...}}
2. {{...}}
```

## Starter guidance

- **Cache the data outside `objective`.** Reloading the
  training set for every trial is minutes per sweep. Load
  once at module level.
- **Fix the seed inside the objective for the sweep.**
  Chapter 08 explains why — varying seed *and*
  hyperparameter at once collapses two variance sources
  into one. Vary the seed later (Part D) to isolate.
- **Use log scale for learning rate and regularisation.**
  `trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)`.
  Uniform sampling would spend 99% of trials in
  `[0.1, 1.0]`.
- **Small `n_trials` first.** Run with `n_trials=5` while
  you debug the sweep loop. Scale up once you know it works.
  Optuna will resume from the SQLite storage.
- **The median-seed champion is a deliberate choice.** The
  max is lucky; the median is honest. Explain the choice in
  the report.
- **Chapter 09's checklist is not optional.** Walk it before
  Part G. Missing items surface as questions in the rerun
  test.

## Acceptance criteria

- `python -m tune` runs a sweep of at least 30 trials that
  finishes without errors.
- MLflow shows one parent run `optuna-sweep` with the
  trials nested underneath and a `champion-registration`
  run alongside.
- Optuna's SQLite storage (`optuna.db`) is durable —
  `optuna.load_study(study_name="mod106-hgb",
  storage="sqlite:///optuna.db")` in a fresh Python session
  returns the completed study.
- The registered model `mod106-champion` has at least one
  version tagged with `git_sha`, `data.md5`, and
  `val_metric`, and has the `@champion` alias.
- `REPORT.md` follows the chapter-09 template and contains
  every mandatory field.
- The rerun test in Part G is documented; either a
  colleague or your fresh-checkout self reproduced the
  headline metric within the stated tolerance.
- `pytest` still passes (exercises 01–03's tests must not
  regress).

## Reflection questions (answer in `REPORT.md`)

- How many trials did you need before the optimisation
  history plot flattened? What does that suggest about the
  cost of the sweep on this problem?
- Which hyperparameter had the highest importance in Optuna's
  `plot_param_importances`? Was that expected, and does the
  answer suggest anything about where to narrow the search
  space in a follow-up sweep?
- Was your seed-sweep standard deviation smaller or larger
  than the gap between the best and second-best trial? If
  smaller, the "winner" is real; if larger, it is within
  noise. Which was it, and how does that change how you'd
  describe the result to a stakeholder?
- Chapter 09 argues that the report is the *deliverable* the
  module exists to produce. Now that you have written one,
  what section did you find hardest to write with the
  discipline the chapter demands, and why?

## Stretch goals (optional)

- **Parallel workers.** Point three shells at the same
  SQLite storage and let them share the sweep. Compare
  wall-clock and note whether the winning trial changed.
  Chapter 08 covers the caveats about parallel reproducibility.
- **Pruning aggressiveness.** Rerun the sweep with
  `MedianPruner(n_startup_trials=1, n_warmup_steps=1)` and
  compare: how much budget did aggressive pruning save?
  Did the winner change?
- **Objective with a penalty.** Modify the objective to
  minimise `val_error + alpha * model_size` (encourage
  smaller models). Report how the winner shifts.
- **Alternative sampler.** Rerun with
  `RandomSampler(seed=0)` and compare best value at same
  budget. TPE should beat random on ≥ 4 hyperparameters;
  if it does not in your run, note why.
- **Champion challenger showdown.** After you register
  `@champion`, register `@challenger` as the *second*-best
  Optuna trial's config. Write a small script that runs
  both against a hold-out set and prints a side-by-side.
  Which one you would ship, and why, goes in the report.
- **CI reproduction of the report.** Add a workflow that,
  on every push to `main`, runs a fast subset of the
  reproduce block (data pull + single train, no sweep) and
  asserts the metric stays within tolerance. Chapter 09's
  rerun-test discussion motivates this.

## What to hand in

```
exercise-04/
├── src/
│   ├── train.py                    # takes TrainConfig
│   └── data.py                     # cached loader
├── utils/
│   └── reproducibility.py          # from exercise-03
├── tune.py                         # Optuna sweep entry point
├── analyze.py                      # or notebooks/analyze.ipynb — Part C plots
├── plots/
│   ├── optimization_history.png
│   ├── param_importances.png
│   └── parallel_coordinate.png
├── optuna.db                       # durable SQLite study
├── tests/
│   └── test_determinism.py         # from exercise-03; must still pass
├── DETERMINISM.md                  # from exercise-03
├── REPORT.md                       # the main deliverable
├── dvc.yaml, dvc.lock, params.yaml # from exercise-02
├── pyproject.toml
└── .gitignore
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.
