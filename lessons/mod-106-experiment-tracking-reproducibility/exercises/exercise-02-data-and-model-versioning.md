# exercise-02 — Data and model versioning

> **Estimated effort:** 3 hours.
> **Companion chapters:** [06 — Versioning data with DVC](../06-versioning-data-with-dvc.md), [07 — Model registry and promotion](../07-model-registry-and-promotion.md).

## Objective

Take the instrumented training script from **exercise-01**
and complete the provenance triangle: version the input
data with **DVC**, register the resulting model in **MLflow
Model Registry**, and prove that a teammate could rerun any
past run from just its git SHA.

By the end you should have:

- A repo where `data/` is tracked by DVC and its hash is
  recorded on every MLflow run.
- A `dvc.yaml` pipeline with at least two stages (features
  → train).
- A registered model in MLflow with at least three versions,
  each tagged with git SHA and data hash, and an alias
  (`@champion`) pointing at the winner.
- A working `reproduce.sh` (or documented commands) that
  restores any past run's data + code and reruns it.

## Prerequisites

- Exercise-01 complete: a training script that already logs
  to MLflow.
- `uv add "dvc[s3]"` — or the extra matching whatever remote
  you plan to use (`dvc[gs]`, `dvc[azure]`, `dvc[ssh]`, or
  just `dvc` for a local remote).
- Access to *some* remote storage for the DVC remote — an S3
  bucket, a GCS bucket, or a local directory outside the
  repo (`/tmp/dvc-remote` is fine for solo work).
- Chapters 06 and 07 of this module.

## Problem statement

### Part A — Initialise DVC and track the raw data (≈ 25 min)

From the repo root:

```bash
uv add "dvc[s3]"            # or the extra you need
dvc init
git add .dvc .dvcignore
git commit -m "chore: init DVC"

# Configure a remote (one-time)
dvc remote add -d origin s3://<bucket>/dvc-store
# or, for local:
mkdir -p /tmp/dvc-remote
dvc remote add -d origin /tmp/dvc-remote

git add .dvc/config
git commit -m "chore: configure dvc remote"
```

Track the raw dataset used in exercise-01:

```bash
mkdir -p data/raw
# Move / copy your dataset file into data/raw/
dvc add data/raw/<file>
git add data/raw/<file>.dvc data/raw/.gitignore
git commit -m "data: add <file> v1 (initial)"
dvc push
```

Verify:

- `git status` clean.
- `cat data/raw/<file>.dvc` shows an `md5` and `size`.
- `dvc pull` on a fresh clone (or after `rm -rf
  data/raw/<file>`) restores the file byte-for-byte.

### Part B — Build a `dvc.yaml` pipeline (≈ 45 min)

Refactor your training script into at least two DVC pipeline
stages:

- `features` — takes raw data, produces a cleaned /
  featurised parquet file under `data/features/`.
- `train`     — takes the featurised data and produces a
  model artefact and a `metrics.json`.

`dvc.yaml`:

```yaml
stages:
  features:
    cmd: python -m src.features --in data/raw/<file> --out data/features/train.parquet
    deps:
      - src/features.py
      - data/raw/<file>
    outs:
      - data/features/train.parquet

  train:
    cmd: python -m src.train --data data/features/train.parquet --seed ${seed} --lr ${learning_rate}
    deps:
      - src/train.py
      - data/features/train.parquet
    params:
      - seed
      - learning_rate
    metrics:
      - metrics.json:
          cache: false
    outs:
      - models/model.pkl
```

`params.yaml`:

```yaml
seed: 0
learning_rate: 0.1
```

Requirements:

1. `dvc repro` runs both stages end to end.
2. Changing `params.yaml`'s `learning_rate` re-runs *only*
   the `train` stage on `dvc repro`.
3. `dvc.lock` is committed to git along with the updated
   `.dvc` files.
4. `metrics.json` is a small JSON with your headline metric
   and validation-set size (at minimum). `dvc metrics show`
   should print it.

### Part C — Wire the DVC hash into MLflow (≈ 20 min)

In `src/train.py`, before starting the MLflow run:

```python
import yaml

def read_data_md5(dvc_path: str) -> str:
    ptr = yaml.safe_load(open(dvc_path))
    return ptr["outs"][0]["md5"]

data_md5 = read_data_md5("data/features/train.parquet.dvc")
```

Then inside the run:

```python
mlflow.log_param("data.dvc_path", "data/features/train.parquet")
mlflow.log_param("data.md5",       data_md5)
mlflow.set_tag("data.md5", data_md5)   # tag too, for filtering
```

Also log the raw-data hash so both layers of the pipeline
are recorded:

```python
mlflow.log_param(
    "data.raw_md5",
    read_data_md5("data/raw/<file>.dvc"),
)
```

Rerun `dvc repro` (which invokes your training script) — the
MLflow run should now carry both hashes.

### Part D — Register the model and open a lifecycle (≈ 30 min)

Extend the training script so that, after a successful run
whose validation metric beats a threshold, the model is
registered:

```python
import mlflow.sklearn   # or mlflow.pytorch, etc.

signature = mlflow.models.signature.infer_signature(
    X_train, clf.predict(X_train)
)

mlflow.sklearn.log_model(
    clf,
    artifact_path="model",
    signature=signature,
    input_example=X_train.iloc[:5],
    registered_model_name="mod106-classifier",
)
```

Then manually run the pipeline three times with different
`learning_rate` values (edit `params.yaml`, `dvc repro`).
You should end up with three versions in the registry.

Tag each version programmatically (in a small script or a
notebook cell):

```python
from mlflow import MlflowClient

client = MlflowClient()
for v in client.search_model_versions("name = 'mod106-classifier'"):
    run = client.get_run(v.run_id)
    for tag in ("git_sha", "data.md5"):
        val = run.data.tags.get(tag) or run.data.params.get(tag)
        if val:
            client.set_model_version_tag(
                "mod106-classifier", v.version, tag, val
            )
    client.set_model_version_tag(
        "mod106-classifier", v.version, "val_metric",
        f'{run.data.metrics.get("val_accuracy", "?"):.4f}',
    )
```

Now every registered version carries `git_sha`, `data.md5`,
and `val_metric` as tags visible in the UI.

Pick the best version and set the `@champion` alias:

```python
best = max(
    client.search_model_versions("name = 'mod106-classifier'"),
    key=lambda v: float(v.tags.get("val_metric", "-inf")),
)
client.set_registered_model_alias(
    name="mod106-classifier",
    alias="champion",
    version=best.version,
)
```

Verify:

```python
model = mlflow.pyfunc.load_model("models:/mod106-classifier@champion")
print(model.predict(X_val.iloc[:5]))
```

### Part E — The reproduce script (≈ 30 min)

Write a `scripts/reproduce.sh` that takes an MLflow run ID
and reproduces its inputs and outputs. Rough shape:

```bash
#!/usr/bin/env bash
set -euo pipefail

RUN_ID="$1"

# 1. Pull provenance from MLflow
GIT_SHA=$(mlflow runs describe --run-id "$RUN_ID" | jq -r '.data.tags["mlflow.source.git.commit"]')
DATA_MD5=$(mlflow runs describe --run-id "$RUN_ID" | jq -r '.data.params["data.md5"]')

echo "Reproducing run $RUN_ID"
echo "  git SHA:  $GIT_SHA"
echo "  data md5: $DATA_MD5"

# 2. Check out the code
git checkout "$GIT_SHA"

# 3. Restore the environment
uv sync --frozen

# 4. Restore the data referenced by that commit
dvc pull

# 5. Verify the restored data hash matches what the run recorded
ACTUAL_MD5=$(python -c "import yaml; print(yaml.safe_load(open('data/features/train.parquet.dvc'))['outs'][0]['md5'])")
[ "$ACTUAL_MD5" = "$DATA_MD5" ] || {
  echo "data hash mismatch: $ACTUAL_MD5 vs $DATA_MD5"
  exit 1
}

# 6. Run
python -m src.train --data data/features/train.parquet
```

You are welcome to write this in Python instead if you
prefer.

Verify it works on **all three** registered versions —
`bash scripts/reproduce.sh <run_id>` should produce the same
metric each version originally recorded.

### Part F — `REPORT.md` (≈ 30 min)

Write a short `REPORT.md` covering:

- **The pipeline.** Paste the `dvc dag` output; explain each
  stage in one sentence.
- **The registry state.** Screenshot the registered model's
  versions view; note which is `@champion` and why.
- **The provenance evidence.** For one specific run,
  identify:
  - MLflow run ID.
  - `git_sha` (from system tag).
  - `data.md5` and `data.raw_md5` (from params).
  - The `.dvc` file's `md5` at that git SHA (should match
    `data.md5`).
- **The reproduce demo.** Screenshot or paste-in of the
  `reproduce.sh` output for one run, showing the metric
  matches.

## Starter guidance

- **Do Part A first, in isolation.** Get `dvc init`,
  `dvc add`, `dvc push`, and `dvc pull` working before you
  restructure the training script. Track one raw file; go.
- **`params.yaml` values propagate.** DVC substitutes
  `${seed}` etc. from `params.yaml` into the `cmd`. Wire
  every hyperparameter through `params.yaml`, not
  hard-coded in Python.
- **`dvc.lock` is auto-updated by `dvc repro`.** Do not edit
  it by hand. Do commit it after every meaningful `repro`.
- **Remote credentials via environment.** For S3, export
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. Do not put
  them in `.dvc/config`. For local remotes, no credentials
  needed.
- **Version *features*, not just raw.** The `features/`
  file's hash is what your model actually saw. Log both raw
  and features hashes; they answer different questions
  (raw = "same source data?" vs. features = "same input to
  the model?").
- **Start the registry story small.** One registered model
  with the initial three versions is enough. You do not need
  a challenger alias for the acceptance criteria.

## Acceptance criteria

- `dvc dag` shows at least two stages
  (`features` → `train`), and `dvc repro` runs them end to
  end.
- The `.dvc` files, `dvc.yaml`, `dvc.lock`, and updated
  `.gitignore` entries are committed to git.
- `dvc pull` on a fresh clone (or after
  `rm -rf .dvc/cache/ data/`) restores every tracked file.
- Every MLflow run produced by `dvc repro` carries
  `data.md5` and `data.raw_md5` as params and at least
  `data.md5` as a tag.
- The MLflow registry has a registered model
  `mod106-classifier` with **≥ 3 versions**, each tagged
  with `git_sha`, `data.md5`, and `val_metric`.
- Exactly one version has the `@champion` alias.
- `mlflow.pyfunc.load_model("models:/mod106-classifier@champion")`
  loads and predicts without further setup.
- `scripts/reproduce.sh <run_id>` succeeds for all three
  registered versions and reports a matching metric.
- `REPORT.md` includes the DAG, the registry screenshot,
  the provenance evidence table, and the reproduce demo.

## Reflection questions (answer in `REPORT.md`)

- Where did DVC's `.dvc` pointer + git integration save you
  time compared to your previous "put data in a shared
  drive" workflow? Where did it cost you time?
- If `data.md5` and `data.raw_md5` differ across two runs,
  which of the two is more likely to explain a drop in
  validation accuracy? Why?
- The Model Registry has `stages` (`Staging` / `Production`
  / `Archived`) as a legacy API and `aliases` as the modern
  one. Why do you think MLflow moved from stages to
  aliases? What is one thing you can do with aliases that
  you cannot easily do with stages?
- If you had to promote a new champion once a week and the
  team is 6 people, what parts of the workflow in this
  exercise would you leave manual, and what parts would you
  automate first?

## Stretch goals (optional)

- **A challenger alias.** Register a fourth version. Set
  `@challenger` on it. Write a small script that loads both
  aliases, runs them over a hold-out set, and prints a
  side-by-side comparison. This is the smallest possible
  A/B harness.
- **CI reproduction.** Add a GitHub Actions workflow that
  runs on every push to `main`: `uv sync --frozen`, `dvc
  pull`, `dvc repro`, and asserts the resulting metric is
  within a small tolerance of the last committed
  `metrics.json`. This catches infrastructure rot early.
- **`dvc.data.from_pandas` cross-check.** Log the featurised
  DataFrame via MLflow's Dataset API (chapter 02) with the
  DVC hash as `digest=`. Confirm two runs on the same
  featurised parquet get the same digest.
- **Retention policy.** Configure `dvc gc` (local) and an S3
  lifecycle rule (remote) so that unreferenced cached
  content is aged out. Document your policy in `REPORT.md`.
- **Register from CI, not from the training script.** Move
  the `registered_model_name=` out of `log_model` and add a
  separate `promote.py` script that, given a run ID, only
  registers the model if `val_metric > threshold`. Wire it
  into CI on merges to `main`.

## What to hand in

```
exercise-02/
├── data/
│   ├── raw/
│   │   ├── <file>.dvc
│   │   └── .gitignore
│   └── features/
│       ├── train.parquet.dvc
│       └── .gitignore
├── src/
│   ├── features.py
│   └── train.py                # registers the model
├── scripts/
│   ├── reproduce.sh
│   └── tag_versions.py
├── .dvc/
│   └── config
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── metrics.json
├── pyproject.toml
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.
