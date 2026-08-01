# 02 — MLflow: runs, metrics, and artifacts

> **Reading time:** ~35 minutes.
> **Prerequisite chapter:** 01.

## Motivation

Every experiment tracker in production ML uses roughly the
same mental model. Learn it well once and every other tool
becomes a translation exercise.

MLflow is the best tool for this chapter for three reasons:

- **Open source and self-hostable.** No account required to
  start; nothing to migrate away from later.
- **Framework-agnostic.** The core API is
  `log_param`, `log_metric`, `log_artifact`. It does not
  care whether you are training scikit-learn, PyTorch,
  XGBoost, or a shell script.
- **Integrates with the rest of this module.** The Model
  Registry (chapter 07) and Optuna's `MLflowCallback`
  (chapter 08) both live in the same object graph.

Weights & Biases, Neptune, Aim, and Comet all fit the same
mental model. Chapter 03 covers the trade-offs; the API
patterns you learn here transfer.

## The object model

MLflow organises everything under three nouns:

- **Experiment.** A named grouping of runs — usually one
  experiment per problem you are working on ("`churn-v3`",
  "`image-classifier-baseline`"). An experiment has an
  integer `experiment_id` and a `name`.
- **Run.** One execution of the training code. A run has a
  unique `run_id` (a hex string), a status
  (`RUNNING` / `FINISHED` / `FAILED`), a start time, and an
  end time.
- **Run contents.** Four kinds of thing attach to a run:

| Attaches | Type | Written with | Read with |
|---|---|---|---|
| **Param** | Config value known at start | `mlflow.log_param(k, v)` | UI, `mlflow.get_run(id).data.params` |
| **Metric** | Number that evolves during the run | `mlflow.log_metric(k, v, step=i)` | UI, `mlflow.get_run(id).data.metrics` |
| **Tag** | Free-form key/value label | `mlflow.set_tag(k, v)` | UI, `mlflow.get_run(id).data.tags` |
| **Artifact** | Any file (model, plot, requirements) | `mlflow.log_artifact(path)` | UI, `mlflow.artifacts.download_artifacts(...)` |

The distinction that trips new users:

- **Params are immutable.** They are the settings you would
  put in a config file. Logging the same param twice for
  one run is an error.
- **Metrics are a time series.** Every call to `log_metric`
  appends a `(step, value)` point. You will call it many
  times per run — once per epoch or per batch.
- **Tags are for filtering.** `mlflow.set_tag("stage",
  "baseline")` lets you list "all baseline runs" later.
  Tags can be overwritten.
- **Artifacts are files.** A checkpoint, a
  `requirements.txt`, a confusion-matrix PNG.

Get the distinctions right and the UI does the rest.

## A one-file training script, instrumented

Here is a scikit-learn classifier trained on a real dataset,
with the tracker in place. Read it once end to end; the rest
of the chapter unpacks each moving piece.

```python
# train.py
import mlflow
import mlflow.sklearn
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


def main(
    seed:          int = 0,
    learning_rate: float = 0.1,
    max_depth:     int = 6,
    max_iter:      int = 200,
):
    mlflow.set_experiment("adult-income-baseline")

    with mlflow.start_run(run_name="hgb-baseline") as run:
        # 1. Params (immutable settings)
        mlflow.log_params({
            "seed":          seed,
            "learning_rate": learning_rate,
            "max_depth":     max_depth,
            "max_iter":      max_iter,
            "model":         "HistGradientBoostingClassifier",
        })

        # 2. Data
        ds = fetch_openml("adult", version=2, as_frame=True)
        X, y = ds.data, (ds.target == ">50K").astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # 3. Fit
        clf = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)

        # 4. Metrics
        proba_te = clf.predict_proba(X_te)
        preds_te = proba_te.argmax(axis=1)
        mlflow.log_metrics({
            "val_accuracy": float(accuracy_score(y_te, preds_te)),
            "val_log_loss": float(log_loss(y_te, proba_te)),
            "n_train":      float(len(X_tr)),
            "n_val":        float(len(X_te)),
        })

        # 5. Tag for later filtering
        mlflow.set_tag("stage", "baseline")

        # 6. Artefacts
        Path("outputs").mkdir(exist_ok=True)
        np.save("outputs/proba_te.npy", proba_te)
        mlflow.log_artifact("outputs/proba_te.npy")

        # 7. Model (typed artefact via the flavour helper)
        mlflow.sklearn.log_model(clf, artifact_path="model")

        print(f"logged run {run.info.run_id}")


if __name__ == "__main__":
    main()
```

Run it once:

```bash
python train.py
mlflow ui                            # opens on http://localhost:5000
```

You now have a searchable, comparable record of that run.
Change any parameter and re-run — a second row appears in the
UI, and you can select both and compare metrics side by side.

## `mlflow.start_run` and the context manager pattern

Every run must be started and ended. The safe way is a
context manager:

```python
with mlflow.start_run(run_name="hgb-baseline") as run:
    mlflow.log_param(...)
    ...
```

Two properties this pattern gives you:

- **On exception, the run is marked `FAILED`.** Without the
  context manager, a crash leaves the run `RUNNING`
  forever, cluttering the UI.
- **Nested runs work.** Pass `nested=True` to the inner
  `start_run` — useful for a sweep where each trial is a
  child of a parent run (chapter 08).

You will occasionally see the fluent-API pattern without a
`with`:

```python
mlflow.start_run(run_name="...")
try:
    ...
finally:
    mlflow.end_run(status="FINISHED")
```

Use the context manager.

## Where the run data lives — the tracking URI

MLflow has one setting that controls where everything ends up:

```python
mlflow.set_tracking_uri("file:./mlruns")   # local folder (default)
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local SQLite
mlflow.set_tracking_uri("http://mlflow.internal:5000")  # remote server
```

Set it via `MLFLOW_TRACKING_URI` env var in real projects so
that the same code writes to the local store on your laptop
and to the shared server in CI:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.internal:5000
python train.py
```

The default `file:./mlruns` is fine for a first run; graduate
to a real server (SQL backend + object-store artefact root)
the moment more than one person needs to see the results.

## Autologging: `mlflow.<framework>.autolog()`

MLflow ships integrations for scikit-learn, PyTorch,
Lightning, TensorFlow, XGBoost, LightGBM, statsmodels, and
others. Turn on autologging for your framework and MLflow
records the standard things without you writing
`log_param` / `log_metric` calls:

```python
import mlflow

mlflow.sklearn.autolog()      # or mlflow.pytorch.autolog(), etc.

# Now training a model logs params, metrics, model, and
# input signature automatically.
clf = HistGradientBoostingClassifier(...)
clf.fit(X_tr, y_tr)
```

What autologging typically captures:

- All estimator hyperparameters (via `get_params()`).
- The fitted model itself.
- Training metrics reported by the framework (per-epoch
  loss / accuracy for PyTorch Lightning; the CV scores for
  scikit-learn's `GridSearchCV`).
- The input signature (feature schema) inferred from the
  training data.

Autologging is a good default for quick iteration. The
trade-off is that it can hide *what* is being logged, and the
set of things logged changes between MLflow versions. For
serious runs, complement autologging with explicit
`log_metric` calls for the specific validation numbers you
actually care about — the ones that will be compared across
runs and referenced in the report.

<!-- needs-research: verify the current list of MLflow autolog
integrations and any semantic changes per major version. -->

## Logging a metric that changes over time

Deep learning runs have per-epoch or per-step metrics. Pass
`step`:

```python
for epoch in range(epochs):
    train_loss = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)

    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss",   val_loss,   step=epoch)
    mlflow.log_metric("val_acc",    val_acc,    step=epoch)
```

In the UI, each metric renders as a line chart over `step`.
You can plot two runs' `val_loss` on the same axes and see
which one converged faster.

Two rules of thumb:

- **Use `step` consistently.** Either every metric is per-
  epoch, or every metric is per-batch. Mixing steps for the
  same metric produces a mangled chart.
- **Batch metrics with `log_metrics` (plural).** Fewer
  HTTP round-trips against a remote tracking server, and
  ensures all four numbers land at the same timestamp.

## Logging artefacts

Anything that isn't a scalar becomes an artefact — a file
written to the run's artefact store:

```python
mlflow.log_artifact("plots/loss_curve.png")
mlflow.log_artifact("outputs/confusion_matrix.csv")
mlflow.log_artifact("requirements.txt")
mlflow.log_dict({"features": feature_names}, "meta/features.json")
mlflow.log_text(report_markdown, "REPORT.md")
mlflow.log_figure(fig, "plots/loss_curve.png")   # matplotlib Figure
```

Two flavours you will use a lot:

- **`log_artifacts(dir_path)`** (plural) — recursively upload
  a directory. Convenient for a `plots/` folder produced by
  the training script.
- **`log_dict` / `log_text` / `log_figure`** — helpers that
  serialise Python objects to files under the artefact
  store without you naming a temp file.

Under the hood everything is a file. The helpers are
convenience.

## The typed model artefact — `mlflow.<flavour>.log_model`

`mlflow.log_artifact("model.pkl")` records a file. That is
fine for a curiosity model. For a model you intend to
compare, load, or register, use the *flavour* API:

```python
mlflow.sklearn.log_model(clf, artifact_path="model")
mlflow.pytorch.log_model(model, artifact_path="model")
mlflow.xgboost.log_model(booster, artifact_path="model")
```

The flavour helpers write:

- **The serialised model itself** (pickle, TorchScript,
  Booster.save, depending on the flavour).
- **A `MLmodel` file** — a small YAML manifest listing the
  flavour, the Python version, the framework version, and
  the input/output signature.
- **A `conda.yaml` / `requirements.txt`** — the dependencies
  needed to load the model in a fresh environment.
- **A signature and input example** if you pass them.

Signature + input example is the piece worth being explicit
about:

```python
from mlflow.models.signature import infer_signature

signature = infer_signature(X_tr, clf.predict(X_tr))
mlflow.sklearn.log_model(
    clf,
    artifact_path="model",
    signature=signature,
    input_example=X_tr.iloc[:5],
)
```

Now `mlflow.models.serve --model-uri runs:/<run_id>/model`
(and the Model Registry story in chapter 07) both work
without extra setup — the manifest is complete.

## Querying runs programmatically

The UI is not the only interface. Every read you can do in
the UI, you can also do in code:

```python
import mlflow

client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("adult-income-baseline")

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.stage = 'baseline' AND metrics.val_accuracy > 0.85",
    order_by=["metrics.val_accuracy DESC"],
    max_results=20,
)
print(runs[["run_id", "params.learning_rate", "metrics.val_accuracy"]])
```

`search_runs` returns a pandas DataFrame with one row per
run and flattened columns for each param / metric / tag.
Combine this with a Jupyter notebook and you have the
"compare N runs on any axis I like" view without leaving
Python. Chapter 09's reproducibility report reuses this
pattern.

## System tags and provenance

MLflow automatically attaches a handful of `mlflow.*` system
tags to every run. The useful ones:

- **`mlflow.source.name`** — the script or notebook path.
- **`mlflow.source.git.commit`** — the git SHA at the moment
  the run started (if the working directory is a git repo).
- **`mlflow.source.git.branch`** — the branch name.
- **`mlflow.user`** — the OS user who launched the run.
- **`mlflow.runName`** — the human-friendly name (from
  `start_run(run_name=...)` or an auto-generated one).

The git SHA is the single most valuable free field. When
someone shows you a run and asks "what code produced this?",
that tag is the answer.

If your working directory has uncommitted changes, MLflow
records the SHA but not the diff — meaning the run is not
strictly reproducible from the SHA alone. Either commit
before you run, or log the diff explicitly:

```python
import subprocess
diff = subprocess.check_output(["git", "diff", "HEAD"]).decode()
if diff:
    mlflow.log_text(diff, "code/uncommitted.diff")
    mlflow.set_tag("git.dirty", "true")
```

Chapter 05 comes back to this.

## Logging inputs — datasets in MLflow

Recent MLflow versions include first-class dataset logging:

```python
import mlflow.data

dataset = mlflow.data.from_pandas(
    df=X_tr.assign(target=y_tr),
    source="s3://my-bucket/adult/v2/train.parquet",
    name="adult-train",
    targets="target",
)
mlflow.log_input(dataset, context="training")
```

The dataset is content-hashed; two runs that used the same
data get the same hash on the input record. This is a
lightweight alternative to DVC when the dataset lives at a
stable URI already. Chapter 06 covers DVC for the case where
you also need to *version* the underlying bytes.

<!-- needs-research: confirm the mlflow.data API surface and
`source` metadata fields on the latest stable MLflow release. -->

## The tracking backend, in one paragraph

Local `mlruns/` is a file backend. It works for a single
developer. In a team, promote to:

- **Backend store** (params/metrics/tags/run metadata): a SQL
  database — SQLite (single-user), Postgres or MySQL (team).
- **Artefact store** (files): a cloud object store — S3, GCS,
  Azure Blob, or MinIO on-prem.
- **Tracking server**: `mlflow server --backend-store-uri
  postgresql://... --default-artifact-root s3://...`.

The client-side code is unchanged; you point
`MLFLOW_TRACKING_URI` at the server URL. Deployment is a
`mod-108` topic; the API is the same.

## Common pitfalls

1. **Forgetting the context manager.** A crash mid-run leaves
   the run in `RUNNING` forever. Always
   `with mlflow.start_run(...) as run:`.
2. **Logging the same param twice.** MLflow raises. Params
   are immutable; use `mlflow.set_tag` if you need a
   mutable label.
3. **Logging metrics as strings.** `mlflow.log_metric` casts
   to float; if the value cannot be cast (`"n/a"`), the call
   raises. Sanitise upstream.
4. **Naming metrics with `/` inconsistently.** Common
   convention: `train/loss`, `val/loss`, `val/accuracy`. Pick
   a convention early; the UI groups them.
5. **Very high-cardinality tags.** Tags are indexed;
   thousands of unique values slow the UI. If it looks like a
   metric, log it as a metric.
6. **Autologging silently on** — you `import` a library that
   calls `mlflow.autolog()` in its own init and now every
   downstream fit shows up as a run. Grep for `autolog` when
   the UI starts filling with runs you did not intend.
7. **Uploading giant artefacts to a remote store**. A 5 GB
   checkpoint takes real bandwidth. Log a pointer to it
   (S3 URI, DVC hash) instead of the file itself for large
   models.

## Summary

- MLflow's mental model: **experiment → run → (params +
  metrics + tags + artefacts)**. Every other tracker maps
  onto the same shape.
- Wrap training in `with mlflow.start_run(...)`; use
  `log_params`, `log_metric(step=...)`, `set_tag`,
  `log_artifact`. Use the flavour helpers
  (`mlflow.sklearn.log_model`, etc.) for models.
- The tracking URI decides where data goes. `file:./mlruns`
  for solo work; SQL backend + object-store artefact root
  for teams.
- MLflow attaches git SHA, branch, and source path as system
  tags — the free half of the provenance triangle. Commit
  before you run so the SHA is meaningful.
- Autologging is a good starting default; complement with
  explicit calls for the metrics you actually compare.

## Where this goes next

Chapter 03 compares MLflow to Weights & Biases (and briefly
to Neptune, Aim, Comet) so you can pick the right tracker
for a real project instead of defaulting to the one this
module happened to teach.
