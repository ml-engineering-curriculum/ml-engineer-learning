# lab-01 — Dissecting `mlruns/`: what a run actually stores on disk

> **Duration:** ~30–45 minutes.
> **Companion chapter:** [02 — MLflow: runs, metrics, and artifacts](../02-mlflow-runs-metrics-and-artifacts.md).

## Objective

Make chapter 02's object model — **experiment → run → (params + metrics +
tags + artifacts)** — concrete by inspecting the files MLflow writes for a
single trivial run on your local disk. By the end you will be able to point
at the file that stores a specific param, read the time-series metric as a
plain-text append log, name the system tag that carries the git SHA, and
load the same run back out of Python using `MlflowClient` — without ever
opening the UI.

This lab is deliberately narrower than **exercise-01** (which instruments a
real `mod-104` / `mod-105` training script with all seven required logging
calls and compares five runs in the UI). Here the "run" is a two-line toy
and the point is to see MLflow's "database" as what it actually is on a
laptop: a folder, with a few text files and a couple of YAML manifests. If
you ever need to grep an old `mlruns/` directory, debug a missing
artefact, or explain to a teammate why a param can't be re-logged, the
layout you see in this lab is the answer.

## Prerequisites

- Python 3.11+, `mlflow >= 2.10` (`uv add mlflow` or
  `pip install "mlflow>=2.10"`).
- A terminal with a `tree` command on the path (optional but convenient —
  `brew install tree`, `apt install tree`, or substitute
  `find mlruns -type f | sort`).
- A CPU is enough; nothing in this lab needs a GPU, a dataset, or a
  framework beyond MLflow itself.
- Chapter 02 read through once.

## Setup

Create an empty lab directory with a single Python file:

```bash
mkdir -p lab-01-mlruns && cd lab-01-mlruns
touch toy_run.py
```

Confirm MLflow is installed and the tracking URI is the local file store
(the default):

```bash
python -c "import mlflow; print(mlflow.__version__); print(mlflow.get_tracking_uri())"
```

**Expected output.** A version string like `2.17.0` (any `>= 2.10` is
fine) and a tracking URI that is either `file:./mlruns` or the absolute
path to a `mlruns/` folder in the current directory. If it points
somewhere else (e.g. a server URL from a prior session), reset it:

```bash
unset MLFLOW_TRACKING_URI
```

## Steps

### Step 1 — Write a two-line toy run (≈ 5 min)

Put in `toy_run.py`:

```python
# toy_run.py
import time
import mlflow

mlflow.set_experiment("lab-01-dissect")

with mlflow.start_run(run_name="toy") as run:
    # 1. An immutable param.
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("model", "none-just-a-toy")

    # 2. A metric that evolves over time.
    for step in range(5):
        mlflow.log_metric("val_loss", 1.0 / (step + 1), step=step)
        time.sleep(0.01)   # so timestamps are distinct

    # 3. A tag.
    mlflow.set_tag("stage", "lab")

    # 4. A file artefact.
    with open("notes.txt", "w") as fh:
        fh.write("hello from lab-01\n")
    mlflow.log_artifact("notes.txt")

    print("run_id:", run.info.run_id)
    print("experiment_id:", run.info.experiment_id)
```

Run it:

```bash
python toy_run.py
```

**Expected output.** Two printed lines — a 32-character hex `run_id` and a
short integer `experiment_id` (usually `0` for `Default`, or some
small integer for `lab-01-dissect`). No errors. A new `mlruns/` directory
now exists in the current working directory, plus a `notes.txt` file
(the artefact we logged — the original is still on disk; MLflow copied a
version into the run's artefact store).

**Checkpoint.** The output tells you three things about chapter 02:
(a) `experiment_id` is MLflow's handle on the named experiment — the UI
shows the name, but the folder is keyed by the id; (b) `run_id` is the
32-hex handle you would "paste into a bug report" — chapter 02 calls
this "a canonical run ID"; (c) the `with mlflow.start_run(...)` context
manager closed cleanly and the run is now `FINISHED`, not `RUNNING` —
chapter 02's pitfall #1 ("Forgetting the context manager") is what makes
that guarantee.

### Step 2 — Explore the top of `mlruns/` (≈ 5 min)

Run:

```bash
tree -L 3 mlruns
# or: find mlruns -maxdepth 3 -type f -o -maxdepth 3 -type d | sort
```

**Expected output.** Something like:

```
mlruns
├── 0
│   └── meta.yaml                     # the Default experiment metadata
├── <experiment_id>
│   ├── <run_id>
│   │   ├── artifacts
│   │   ├── meta.yaml
│   │   ├── metrics
│   │   ├── params
│   │   └── tags
│   └── meta.yaml
└── models                            # (empty unless models are logged)
```

**Checkpoint.** This is chapter 02's "`file:./mlruns` for solo work"
backend in the flesh. One directory per experiment (named by id, not by
human name — the name is in `meta.yaml`); one directory per run inside;
four subdirectories per run matching the four attachments in chapter
02's table. The entire "database" for the run is a few dozen small text
files. If you delete `mlruns/`, every run you've ever logged with this
backend is gone — a useful thing to know before you add it to
`.gitignore`.

### Step 3 — Where does a param live? (≈ 5 min)

Open the `params/` directory for your run:

```bash
ls mlruns/<experiment_id>/<run_id>/params/
cat mlruns/<experiment_id>/<run_id>/params/learning_rate
cat mlruns/<experiment_id>/<run_id>/params/model
```

**Expected output.** One file per param name (`learning_rate`, `model`),
each containing exactly the stringified value you logged (`0.1` and
`none-just-a-toy`) with no trailing newline guaranteed. Two files total.

Now try to re-log a param from Python:

```bash
python -c "
import mlflow
mlflow.set_experiment('lab-01-dissect')
with mlflow.start_run(run_id='<paste_run_id_here>'):
    mlflow.log_param('learning_rate', 0.2)
"
```

**Expected output.** An `MlflowException` whose message is along the
lines of
`Changing param values is not allowed. Param with key='learning_rate' was already logged with value='0.1' for run ID='<run_id>'`
(the exact wording varies by MLflow version; the mechanism doesn't).

**Checkpoint.** Chapter 02's rule "params are immutable; logging the
same param twice for one run is an error" is enforced by the file
backend itself: once `params/learning_rate` exists, a second
`log_param` with a different value is refused at the API layer. This is
why chapter 02 says "use `mlflow.set_tag` if you need a mutable label."
Tags live in `tags/`, and (as the next step shows) they can be
overwritten.

### Step 4 — Metrics are an append-only log (≈ 5 min)

Open the `metrics/` directory:

```bash
ls mlruns/<experiment_id>/<run_id>/metrics/
cat mlruns/<experiment_id>/<run_id>/metrics/val_loss
```

**Expected output.** One file named `val_loss`. Inside: **five** lines,
one per `log_metric` call, each with the shape
`<unix_timestamp_ms> <value> <step>`:

```
1731324001234 1.0 0
1731324001245 0.5 1
1731324001256 0.3333333333333333 2
1731324001267 0.25 3
1731324001278 0.2 4
```

Timestamps and values will differ on your box. The important thing is
that there are **five lines** — one per `log_metric("val_loss", ...,
step=step)` call — and they are strictly append-order.

**Checkpoint.** Chapter 02 ("Logging a metric that changes over time")
says "Every call to `log_metric` appends a `(step, value)` point." The
file backend is exactly that: an append-only text log. The MLflow UI's
line chart for `val_loss` is just this file plotted. This is also why
the chapter warns to **use `step` consistently** — if one call passes
`step=0` and another passes `step=epoch_number`, the chart stitches
the two together in appended order, which is rarely what you want.
Chapter 02 pitfall #4 (`/`-inconsistent metric naming) is the related
UI-grouping footgun.

### Step 5 — Tags and system tags (≈ 5 min)

Open the `tags/` directory:

```bash
ls mlruns/<experiment_id>/<run_id>/tags/
cat mlruns/<experiment_id>/<run_id>/tags/stage
cat mlruns/<experiment_id>/<run_id>/tags/mlflow.runName
cat mlruns/<experiment_id>/<run_id>/tags/mlflow.user
```

**Expected output.** `stage` contains `lab` (the one you set
explicitly). `mlflow.runName` contains `toy`. `mlflow.user` contains
your OS username. If you ran `python toy_run.py` from inside a git
repository, you should also see:

```bash
cat mlruns/<experiment_id>/<run_id>/tags/mlflow.source.git.commit
cat mlruns/<experiment_id>/<run_id>/tags/mlflow.source.git.branch
cat mlruns/<experiment_id>/<run_id>/tags/mlflow.source.name
```

The first contains the git SHA at the moment the run started, the
second the branch, the third the path to `toy_run.py`. If you ran
outside a git repo, those three files will be absent.

Now overwrite the `stage` tag from a fresh Python session:

```bash
python -c "
import mlflow
mlflow.set_experiment('lab-01-dissect')
with mlflow.start_run(run_id='<paste_run_id_here>'):
    mlflow.set_tag('stage', 'lab-rerun')
"
cat mlruns/<experiment_id>/<run_id>/tags/stage
```

**Expected output.** `lab-rerun`. Tags *can* be re-logged — chapter 02's
table calls them "overwritable" — in contrast to params.

**Checkpoint.** Chapter 02 ("System tags and provenance") names the
mlflow.* tags one by one. The git SHA tag is called out as "the single
most valuable free field. When someone shows you a run and asks 'what
code produced this?', that tag is the answer." Chapter 02 also notes
that MLflow records the SHA but **not** the diff when the working tree
is dirty — chapter 05's env manifest closes that gap with
`code/uncommitted.diff`.

### Step 6 — Artifacts live outside the run's metadata (≈ 5 min)

```bash
ls mlruns/<experiment_id>/<run_id>/artifacts/
cat mlruns/<experiment_id>/<run_id>/artifacts/notes.txt
```

**Expected output.** A single file `notes.txt` whose contents match what
you wrote in step 1 (`hello from lab-01`).

Also inspect the run's `meta.yaml`:

```bash
cat mlruns/<experiment_id>/<run_id>/meta.yaml
```

**Expected output.** A small YAML with fields like `artifact_uri`
(a URI that points at the `artifacts/` directory above — on a file
backend this is a `file://.../artifacts` URL), `start_time`, `end_time`,
`status` (`FINISHED`), `run_name`, `experiment_id`, and the current
`lifecycle_stage` (`active`).

**Checkpoint.** Chapter 02's "The tracking backend, in one paragraph"
distinguishes the **backend store** (params / metrics / tags / run
metadata — SQL in production, flat files locally) from the **artefact
store** (files — S3 / GCS / Azure Blob / MinIO in production, local
folders here). You can see both on disk: `params/`, `metrics/`, `tags/`,
and `meta.yaml` are the backend; `artifacts/` is the artefact store.
Chapter 02 pitfall #7 ("Uploading giant artefacts to a remote store")
is about the fact that on a real deployment, these two stores are on
different machines and a 5 GB checkpoint is a real bandwidth decision.

### Step 7 — Read the run back from Python with `MlflowClient` (≈ 5 min)

Everything you just inspected on disk has a Python equivalent. Add a
second file `read_back.py`:

```python
# read_back.py
import sys
import mlflow
from mlflow import MlflowClient

run_id = sys.argv[1]
client = MlflowClient()
run = client.get_run(run_id)

print("status:         ", run.info.status)
print("run_name:       ", run.info.run_name)
print("experiment_id:  ", run.info.experiment_id)

print("\nparams:")
for k, v in run.data.params.items():
    print(f"  {k} = {v}")

print("\ntags:")
for k, v in sorted(run.data.tags.items()):
    print(f"  {k} = {v}")

print("\nmetrics (last value only):")
for k, v in run.data.metrics.items():
    print(f"  {k} = {v}")

print("\nfull val_loss time series:")
for mh in client.get_metric_history(run_id, "val_loss"):
    print(f"  step={mh.step}  value={mh.value}  timestamp={mh.timestamp}")
```

Run it with the run id you noted in step 1:

```bash
python read_back.py <paste_run_id_here>
```

**Expected output.** Status `FINISHED`; the two params from step 1; the
explicit `stage=lab-rerun` tag plus the system tags
(`mlflow.runName`, `mlflow.user`, `mlflow.source.*` if in a git repo);
one metric `val_loss = 0.2` (MLflow's `run.data.metrics` dict holds
only the last value per metric); and the full five-point time series
from `get_metric_history`, showing the same five
`(step, value, timestamp)` triples you `cat`ted in step 4.

**Checkpoint.** This is chapter 02's "Querying runs programmatically"
API on a single run. The two points to internalise: (1)
`run.data.metrics` gives only the **last** value — you need
`client.get_metric_history` to see the whole time series (same
distinction the UI makes between the summary table and the line chart);
(2) `run.data.params`, `run.data.tags`, and `run.data.metrics` are
plain `dict`s — chapter 02's `mlflow.search_runs(...)` returns a pandas
DataFrame with these dicts flattened into columns, which is why queries
like `metrics.val_accuracy > 0.85` work.

### Step 8 — Point the tracker somewhere else and watch the layout stay the same (≈ 5 min)

Chapter 02's "Where the run data lives — the tracking URI" says the
same client code works against three different backends. Prove it with
the local-SQLite one:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python toy_run.py
```

The script runs identically. There's now a `mlflow.db` file next to
`mlruns/`:

```bash
ls -lh mlflow.db
ls mlruns/.trash/ 2>/dev/null || true
```

**Expected output.** `mlflow.db` exists (SQLite is now storing the
params / metrics / tags / run metadata that was in the flat-file
backend before). **Important:** the `artifacts/` directory still uses
the filesystem — SQLite is the *backend store*, not the *artefact
store*. The `mlflow.db` file does not contain the `notes.txt` artefact;
that lives under an `mlartifacts/` directory (or a configurable
default artefact root) alongside.

Reset the tracking URI before you go:

```bash
unset MLFLOW_TRACKING_URI
```

**Checkpoint.** Chapter 02's three tracking URIs — `file:./mlruns`,
`sqlite:///mlflow.db`, and a remote `http://...` server — are the same
API across three stores. The split between **backend store** (SQL)
and **artefact store** (object store or filesystem) that chapter 02
describes is what you just saw: SQLite handled the metadata, the
filesystem handled the file. On a team deployment both move: SQL
backend + S3 artefact root. The client code stays identical, which is
why chapter 02 insists on setting `MLFLOW_TRACKING_URI` by environment,
not in the code.

## Expected final artefact

A directory `lab-01-mlruns/` containing:

```
lab-01-mlruns/
├── toy_run.py                 # step 1
├── read_back.py               # step 7
├── notes.txt                  # artefact source
├── mlruns/                    # flat-file backend (from steps 1–7)
│   └── <experiment_id>/<run_id>/{params,metrics,tags,artifacts,meta.yaml}
├── mlflow.db                  # from step 8 (SQLite backend)
└── mlartifacts/               # from step 8 (SQLite's default artefact root)
```

Running `python toy_run.py && python read_back.py <run_id>` end-to-end
should complete in a couple of seconds, with no network access
required.

## Final checkpoint

Without re-reading chapter 02, answer:

1. In step 3 you got an `MlflowException` when you tried to re-log
   `learning_rate` with a different value. Which of the four attachment
   types (param / metric / tag / artifact) is responsible, and what is
   the correct attachment type to reach for when you want a value you
   *can* overwrite on a run?
2. In step 4 the `metrics/val_loss` file had five lines but the
   `run.data.metrics["val_loss"]` in step 7 returned only one number.
   Which number is it, and which API returns the full time series?
3. In step 5 `mlflow.source.git.commit` was present if (and only if)
   you launched the script from inside a git repo. What does chapter 02
   recommend you *also* log when the working tree has uncommitted
   changes, and why is the git SHA alone insufficient for a
   byte-identical reproduction in that case?
4. In step 8 you changed `MLFLOW_TRACKING_URI` from the default to
   SQLite and `toy_run.py` ran unchanged. Which store did the SQLite
   backend replace, and which store continued to live on the local
   filesystem? Which chapter-02 pitfall warns about that same split on
   a real deployment with a remote artefact root?

If you can answer all four, you now know what the UI is drawing from
on a local MLflow setup, and the move to a real server (SQL backend +
S3 artefact root) is a configuration change, not a conceptual one.
Exercise-01 reuses every one of these primitives on a real training
script.

## Where to go next

- **`exercise-01` ("MLflow runs and metrics")** instruments a real
  training script with the seven required logging calls (params,
  metrics, tags, artefacts, model, env, git), runs a five-run manual
  sweep, and compares the runs in the UI.
- **`lab-02` ("Non-determinism in a 10-line loop: find the three
  sources")** reproduces chapter 04's claim that `torch.manual_seed`
  alone is not enough — by adding one seed call at a time to a
  deliberately-flaky training loop and seeing which one is actually
  load-bearing.
- **`lab-03` ("A 10-trial Optuna sweep end to end")** runs the smallest
  useful Optuna study against a toy objective and reads the three
  diagnostic plots chapter 08 asks you to produce.
