# exercise-01 — MLflow runs and metrics

> **Estimated effort:** 2 hours.
> **Companion chapters:** [02 — MLflow: runs, metrics, artifacts](../02-mlflow-runs-metrics-and-artifacts.md), [03 — Picking a tracker](../03-picking-a-tracker-mlflow-vs-wandb.md).

## Objective

Take a training script you already have — from `mod-104`
(classical) or `mod-105` (deep learning) — and instrument it
end to end with MLflow. By the end you should have:

- A `train.py` that logs an MLflow run every time it is
  invoked, capturing params, metrics, tags, environment,
  and the trained model.
- At least **five distinct runs** in the MLflow UI, each
  with different hyperparameters, ready to be compared.
- A short `NOTES.md` documenting the tracker choice, the
  URI where runs live, and answers to the reflection
  questions.

The point of this exercise is not the model. It is the
instrumentation muscle memory — after this exercise, you
should be able to bolt MLflow onto any training script in
under twenty minutes without looking anything up.

## Prerequisites

- Python 3.11+, MLflow ≥ 2.10 (`uv add mlflow` or
  `pip install "mlflow>=2.10"`).
- A working training script from `mod-104` (scikit-learn
  pipeline) *or* `mod-105` (PyTorch training loop). Either
  is fine; the exercise is framework-agnostic.
- The dataset that script trains on.
- Chapters 02 and 03 of this module.
- Optional: `matplotlib` if your loss curve is not already
  produced by the base script.

## Problem statement

### Part A — Wire in the tracker (≈ 30 min)

Modify your training script so that every invocation:

1. Sets an experiment name — one per project, e.g.
   `mlflow.set_experiment("adult-income-baseline")`. Do not
   use `Default`.
2. Wraps the training in `with mlflow.start_run(run_name=...)`.
3. Logs **all** hyperparameters as params — even the ones
   you consider defaults. Use `mlflow.log_params({...})`
   in a single call so a stray hyperparameter is easier to
   spot.
4. Logs **at least three** validation metrics with
   `mlflow.log_metric` (or `log_metrics`):
   - The headline metric (accuracy, F1, MAE — whatever the
     task uses).
   - A secondary metric (log loss, ROC-AUC, MSE).
   - The size of the validation set (as a metric — trivial
     but useful when comparing runs across data versions).
5. Sets at least one **tag** with `mlflow.set_tag` — e.g.
   `stage=baseline`, `owner=<your-github-handle>`.
6. Attaches at least one **artefact** with
   `mlflow.log_artifact` — the loss curve for deep learning
   runs, or a confusion-matrix CSV for classification
   otherwise.
7. Logs the model itself with the correct **flavour helper**
   for your framework (`mlflow.sklearn.log_model`,
   `mlflow.pytorch.log_model`, `mlflow.xgboost.log_model`,
   …). Pass a `signature` inferred from your training data.

Do **not** use `mlflow.<framework>.autolog()` for the
required run. Add it in a stretch goal only, after you have
done the manual instrumentation once.

### Part B — Deep-learning per-step metrics (≈ 15 min)

*Skip this part if you are working from a scikit-learn
`mod-104` script.*

For a PyTorch loop from `mod-105`, extend the instrumentation
so **each epoch** logs `train_loss`, `val_loss`, and the
headline validation metric with the correct `step=` value:

```python
for epoch in range(epochs):
    train_loss = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)
    mlflow.log_metrics(
        {"train_loss": train_loss,
         "val_loss":   val_loss,
         "val_acc":    val_acc},
        step=epoch,
    )
```

Open the MLflow UI (`mlflow ui`) and confirm that each of
these metrics renders as a line chart with epochs on the
X-axis.

### Part C — Environment and provenance logging (≈ 20 min)

Add a small helper `env_manifest.py` that captures the
environment and logs it as an artefact. At minimum:

- `python -m pip freeze` output → `env/pip_freeze.txt`.
- Python version, platform, git SHA (via `subprocess`).
- If a GPU is used: `torch.version.cuda`,
  `torch.cuda.get_device_name(0)`, and the NVIDIA driver
  version.

Package the dict and log it:

```python
mlflow.log_dict(env_manifest(), "env/manifest.json")
mlflow.log_text(pip_freeze,      "env/pip_freeze.txt")
```

Also log the current git SHA as a param (redundant with the
system tag `mlflow.source.git.commit` but explicit) and log
the git diff if the working tree is dirty:

```python
import subprocess
sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]).decode().strip()
mlflow.log_param("git_sha", sha)

diff = subprocess.check_output(["git", "diff", "HEAD"]).decode()
if diff:
    mlflow.log_text(diff, "code/uncommitted.diff")
    mlflow.set_tag("git.dirty", "true")
```

### Part D — Run the sweep manually (≈ 25 min)

Run the training script at least **five times** with
different configurations. You will use these runs in later
exercises, so make the differences meaningful, not cosmetic.
Suggested pattern:

| Run | Variation |
|---|---|
| 1 | Your original config (baseline) |
| 2 | Baseline with a substantially larger model / more trees / higher depth |
| 3 | Baseline with a substantially smaller model / shallower tree |
| 4 | Baseline with a different learning rate |
| 5 | Baseline with a different random seed |

Use `mlflow.set_tag("stage", "baseline")` on runs 1–4 and
`mlflow.set_tag("stage", "seed-check")` on run 5 — you will
filter by tag in Part E.

### Part E — Query and compare (≈ 20 min)

Open `mlflow ui` at http://localhost:5000 (or your chosen
tracking URI). Do all of the following, and screenshot each
into `NOTES.md`:

1. **Select all five runs** and use the Compare view to plot
   `val_loss` per epoch on the same axes (or the metrics
   bar chart for sklearn).
2. **Filter** the run list to `tags.stage = 'baseline'`.
3. **Sort** runs by your headline metric.
4. Then in a Python script or notebook, reproduce filter #2
   programmatically:

   ```python
   import mlflow
   df = mlflow.search_runs(
       experiment_names=["adult-income-baseline"],
       filter_string="tags.stage = 'baseline'",
       order_by=["metrics.val_accuracy DESC"],
   )
   print(df[["run_id", "params.learning_rate", "metrics.val_accuracy"]])
   ```

   Save the DataFrame's `to_markdown()` output into
   `NOTES.md` as a table.

### Part F — Notes and reflection (≈ 10 min)

Write `NOTES.md` covering:

- **Tracker & URI.** MLflow + the URI you used
  (`file:./mlruns`, `sqlite:...`, or a server URL).
- **Screenshots** from Part E.
- **Comparison table** from the `search_runs` call.
- Answers to the reflection questions.

## Starter guidance

- **Set `MLFLOW_TRACKING_URI` in your shell**, not in the
  code. `export MLFLOW_TRACKING_URI=file:./mlruns` works
  for local; swap to a server URL later without touching
  the code.
- **Log params in one call.** `mlflow.log_params(dict)` is
  faster and easier to audit than a series of
  `log_param` calls.
- **Match metric names to the plot you want.** Metrics with
  `/` in the name group in the UI. `train/loss`,
  `val/loss`, `val/accuracy` renders as three panels; a
  single `loss` metric with a `phase` tag does not.
- **Do the "does the run finish?" test** before doing the
  five sweeps. Run once; confirm it appears in the UI;
  then vary hyperparameters.
- **Do not commit `mlruns/`** to git. Add it to
  `.gitignore`. Runs live in the tracking backend, not
  the repo.

## Acceptance criteria

- `python -m train` (or equivalent) launches a run that
  appears in the MLflow UI with:
  - At least the required params, ≥ 3 metrics, ≥ 1 tag, ≥ 1
    artefact, and a `model/` artefact with a signature.
  - System tags `mlflow.source.git.commit` and
    `mlflow.user` present (they should appear automatically).
- The `env/pip_freeze.txt` and `env/manifest.json` artefacts
  exist on every run.
- At least five runs are logged in one experiment; at least
  one uses a different seed and is tagged
  `stage=seed-check`.
- The programmatic `search_runs` query in Part E returns
  the four baseline runs sorted by the headline metric.
- `NOTES.md` includes the tracker URI, screenshots from Part
  E, the query table, and the reflection answers.

## Reflection questions (answer in `NOTES.md`)

- Which of the seven required things (params, metrics, tags,
  artefact, model, env, git) added the most value for the
  time it took to instrument? Which added the least?
- If your team already had a lightweight
  `results.md`-in-the-repo tracker, which of the MLflow
  features in this exercise would justify the switch? Which
  would not?
- Suppose two months from now you need to answer "which of
  these five runs used the largest model?" How would you
  find out with only the MLflow UI, and how would you find
  out with the programmatic query in Part E?
- What did the git SHA and dirty-diff logging surface that
  you did not know before?

## Stretch goals (optional)

- **Autologging.** Add `mlflow.<framework>.autolog()` to a
  copy of the script and compare the auto-logged fields
  against the manual ones. Note in `NOTES.md` which fields
  were duplicated and which were missing.
- **W&B mirror.** Rewrite the same training script against
  Weights & Biases (chapter 03). Log to both trackers
  concurrently for one run. Which was easier? Which UI told
  you more with fewer clicks?
- **Datasets API.** Try `mlflow.data.from_pandas` and
  `mlflow.log_input` to record the training set as an input.
  Verify the digest column in the UI matches for two runs
  that used the same DataFrame.
- **Nested runs preview.** Wrap the whole five-run sweep in
  one outer `with mlflow.start_run(run_name="manual-sweep"):`
  and pass `nested=True` to each child. The UI groups them —
  a preview of chapter 08's Optuna pattern.
- **Remote tracking server.** Start a local MLflow server
  with a SQLite backend and a local `./artifacts` root
  (`mlflow server --backend-store-uri sqlite:///mlflow.db
  --default-artifact-root ./artifacts`); point
  `MLFLOW_TRACKING_URI` at it; rerun. Note any behaviour
  changes.

## What to hand in

```
exercise-01/
├── src/
│   ├── train.py                # instrumented training script
│   └── env_manifest.py         # helper from Part C
├── notebooks/
│   └── compare_runs.ipynb      # Part E queries + comparison table
├── screenshots/
│   ├── compare_view.png
│   ├── filter_by_tag.png
│   └── sorted_by_metric.png
├── mlruns/                     # local run store (git-ignored)
├── .gitignore                  # includes mlruns/
├── pyproject.toml
└── NOTES.md
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.
