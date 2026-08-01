# 06 — Versioning data with DVC

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 05 (environment pinning),
> `mod-103` (something worth versioning).

## Motivation

You have git for code. You have MLflow for run history. You
have a lockfile and a container for the environment. That
leaves the last vertex of the provenance triangle: **the
data**.

Naming the data is not enough. "Customer churn training set,
Q3 2026" is a description; two teammates hearing it will
retrieve different bytes. What we want is:

- **Content-addressed identity.** The dataset's identity is
  its hash, not its name. If the bytes change, the hash
  changes.
- **Git-friendly reference.** A small file in the repo (the
  `.dvc` pointer) encodes the hash. Git tracks the pointer;
  the bytes live elsewhere.
- **Restorable at a past commit.** `git checkout <SHA> && dvc
  pull` gives you exactly the data the model at that SHA
  trained on.

The tool for this in the Python ML world is **DVC** (Data
Version Control). It gets the same treatment MLflow got in
chapter 02: mental model first, then commands, then wiring
into a real pipeline.

## The mental model

DVC layers three pieces on top of git:

- **`.dvc` pointer files.** A tiny YAML in the repo (a few
  hundred bytes) that records the content hash of a real
  file or directory. This is what git tracks.
- **A local cache** (`.dvc/cache/`). The actual bytes,
  content-addressed by their hash. Not in git.
- **A remote** (S3, GCS, Azure Blob, SSH, HTTP, or a shared
  filesystem). The team-shared copy of the cache. `dvc push`
  uploads new content; `dvc pull` downloads what you don't
  have locally.

The result:

- Your git history is small — pointer files only.
- The bytes are versioned by hash: two commits with the same
  pointer refer to the same bytes.
- Anyone with repo access + remote access can restore any
  historical version with `git checkout` + `dvc pull`.

If you have used git-LFS, this will feel familiar: DVC's
scope is broader (see the pipelines section) and its remotes
are more flexible.

## Getting started, in six commands

For a repo that already has git initialised:

```bash
# One-time
uv add "dvc[s3]"           # or dvc[gs], dvc[azure], dvc[ssh], ...
dvc init                   # creates .dvc/ and adds a .gitignore entry

# Configure a remote (once per repo)
dvc remote add -d origin s3://my-bucket/dvc-store
dvc remote modify origin region us-east-1

# Track a data file
dvc add data/raw/adult.csv
git add data/raw/adult.csv.dvc data/raw/.gitignore
git commit -m "data: add UCI Adult v2"
dvc push                    # upload bytes to the remote

# Someone else, later, on a fresh clone
git clone <repo>
uv sync --frozen
dvc pull                    # download bytes referenced by the current commit
```

Two things worth reading carefully in the output of
`dvc add`:

- The `adult.csv.dvc` file. Open it: it's a few lines of
  YAML with a `md5` hash, a size, and the file path.
- The updated `.gitignore`. DVC adds the real file to
  `.gitignore` so that git *does not* track the bytes —
  only the pointer.

## The `.dvc` file — anatomy of the pointer

```yaml
# data/raw/adult.csv.dvc
outs:
  - md5: 5f9a4e5d5b1a9d3f4a1c2e6b7f8a9b0c
    size: 3974305
    hash: md5
    path: adult.csv
```

That is the entire provenance record for the file, as far as
git is concerned. Commit this pointer, and any future
teammate can materialise the exact bytes with `dvc pull`.

Directory tracking works the same way — DVC hashes the
directory contents and stores one pointer for the whole
tree.

## Reproducing a past experiment's data

The killer use case for DVC in this module. Suppose two
weeks ago you produced a run whose MLflow record says
`git_sha = abc123`. To reproduce its inputs today:

```bash
git checkout abc123
dvc pull                    # data files repopulate at the state of abc123
uv sync --frozen            # environment matches lockfile at abc123
python -m src.train         # same seed, same data, same env → same result
```

No email chains, no "does anyone still have the CSV?",
no shared-drive archaeology. The commit is the pointer to
everything.

Return to the current branch when done:

```bash
git checkout main
dvc pull
```

## DVC pipelines — the `dvc.yaml` graph

DVC's `add`/`push`/`pull` is enough for one-off datasets. In
a real project the data is *derived*: raw → cleaned →
featurised → split → trained. DVC's pipelines make each
stage a first-class object with declared inputs and outputs.

Minimal `dvc.yaml`:

```yaml
# dvc.yaml
stages:
  clean:
    cmd: python -m src.data.clean --in data/raw/adult.csv --out data/clean/adult.parquet
    deps:
      - src/data/clean.py
      - data/raw/adult.csv
    outs:
      - data/clean/adult.parquet

  features:
    cmd: python -m src.features.build --in data/clean/adult.parquet --out data/features/adult.parquet
    deps:
      - src/features/build.py
      - data/clean/adult.parquet
    outs:
      - data/features/adult.parquet

  train:
    cmd: python -m src.train --data data/features/adult.parquet --seed ${seed}
    deps:
      - src/train.py
      - data/features/adult.parquet
    params:
      - seed
      - learning_rate
      - max_depth
    metrics:
      - metrics.json:
          cache: false
    outs:
      - models/hgb.joblib
```

`params.yaml` alongside it holds the parameter values:

```yaml
# params.yaml
seed: 0
learning_rate: 0.1
max_depth: 6
```

Run the whole graph:

```bash
dvc repro
```

DVC walks the DAG, hashes each stage's declared deps, and
re-runs only the stages whose inputs changed. Change
`learning_rate` in `params.yaml` and `dvc repro`: the
`clean` and `features` stages skip (unchanged deps), only
`train` re-runs. Commit `dvc.lock` (auto-updated) and now
the pipeline state is versioned in git too.

Three properties this gives you:

- **Deterministic derivation.** Anyone who runs `dvc repro`
  from a fresh checkout produces the same intermediate files
  as you did.
- **Cheap iteration.** Change one thing; only the affected
  stages re-run.
- **A named `metrics.json` output** that MLflow and DVC's
  own comparison tooling both read.

## Where DVC and MLflow fit together

DVC and MLflow overlap slightly and complement mostly. The
mental split most teams settle on:

- **DVC** versions *files* — datasets, intermediate feature
  files, trained model files as bytes.
- **MLflow** versions *runs* — the training executions, the
  hyperparameters, the metrics, the model as a typed
  artefact with a signature (chapter 07).

The bridge:

- Log the data's DVC hash in every MLflow run as a param
  and/or an input:

  ```python
  import yaml, mlflow
  ptr = yaml.safe_load(open("data/features/adult.parquet.dvc"))
  data_hash = ptr["outs"][0]["md5"]
  mlflow.log_param("data.md5", data_hash)
  ```

- Optionally, use MLflow's dataset API (chapter 02) with
  the DVC-tracked file as the `source`:

  ```python
  ds = mlflow.data.from_pandas(
      df, source=f"dvc://data/features/adult.parquet",
      name="adult-features",
      digest=data_hash,   # explicit content hash
  )
  mlflow.log_input(ds, context="training")
  ```

Now the MLflow run says "trained on this hash"; `git blame`
on the `.dvc` file says "this hash landed at commit X"; DVC
remotes have the bytes for that hash. The provenance chain
is complete.

## DVC experiments — a lightweight comparison view

DVC ships an `experiments` subcommand that turns a
`dvc.yaml` pipeline into a comparable sweep without a
tracker:

```bash
dvc exp run --set-param learning_rate=0.05
dvc exp run --set-param learning_rate=0.1
dvc exp run --set-param learning_rate=0.2
dvc exp show                    # table of runs with params + metrics
```

Each `dvc exp run` runs the pipeline with a modified
`params.yaml` and records the metrics from the stage's
`metrics.json`. The runs live under git in a special ref
namespace (`refs/exps/`) — you can promote a good one to a
real branch with `dvc exp branch`.

This is a reasonable choice for a solo project or when you
want the tracking story to live entirely in git. In this
module we use MLflow for run tracking (chapters 02, 08) and
DVC for data and pipeline, because it separates concerns
cleanly and each tool is best in class at its own job.

## Alternatives to DVC

Not every team uses DVC. Know the alternatives so you can
recognise them:

- **git-LFS.** Simpler; tracks large files in git via a
  central server. No pipelines, no remote flexibility.
  Fine for a small number of stable binary assets. Poor
  fit for datasets you regenerate often.
- **LakeFS.** Git-like semantics on top of object storage.
  Branches and commits *for data lakes*. Heavier, aimed at
  team-scale data platforms.
- **Delta Lake / Iceberg / Hudi.** Table-format layers with
  time-travel semantics. Right choice when your dataset is
  already a table in a lakehouse.
- **Content-addressed object store + pointer file.** Roll
  your own: hash the file, upload to S3 keyed by hash, commit
  a `.json` pointer. DVC is essentially this, done right.
- **HuggingFace `datasets` cache.** For datasets already
  hosted on the Hub, the built-in caching + revision-pinning
  is often enough. Pass `revision=<commit-sha>` to
  `load_dataset` for a stable reference.

**Rule of thumb.** DVC is the default for a Python-first ML
project with local training and object-store remotes. Lake-
scale platforms and table-format lakehouses have their own
tools; know which you are in.

## CI patterns

CI should verify that a DVC-tracked repo is coherent:

```yaml
# .github/workflows/dvc-check.yml
name: dvc-check
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: dvc status                         # any missing/moved data?
      - run: dvc dag                            # pipeline graph is valid?
      - run: dvc pull --run-cache               # can we materialise inputs?
      - run: dvc repro --dry                    # what would rerun?
```

`dvc pull` in CI needs credentials for the remote. Use the
CI provider's secret store (never the DVC config file
committed to git). For S3, `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` env vars work out of the box.

## Common pitfalls

1. **Committing the data by mistake.** `dvc add` updates
   `.gitignore`; if you skip that step or override it, the
   file lands in git and the repo balloons. `git status`
   before every commit.
2. **`dvc push` skipped.** Local `dvc add` puts bytes in the
   local cache; without `dvc push`, teammates cannot get
   them. Every workflow that changes tracked data ends in
   `dvc push`.
3. **Uncommitted `dvc.lock`.** The lock records exact hashes
   for pipeline outputs. Without it, `dvc repro` on a
   teammate's machine re-runs everything from scratch. Commit
   both `dvc.yaml` and `dvc.lock`.
4. **`params.yaml` drift from what the code actually uses.**
   Only params declared in a stage's `params:` section
   trigger reruns. A hyperparameter hard-coded in Python
   bypasses DVC. Route every param through `params.yaml`.
5. **Huge remotes without a budget.** DVC will happily keep
   every historical version. If you retrain a
   multi-gigabyte model daily, the remote grows fast. Set
   up a retention policy (`dvc gc` locally, S3 lifecycle
   rules remotely).
6. **Sensitive data in a public remote.** DVC does not
   encrypt data at rest — it defers to the remote. If the
   data is sensitive, the S3 bucket policy is your control
   plane, not DVC.
7. **Renaming a tracked directory without `dvc move`.**
   Break the pointer, break the reproducibility.
   `dvc move src/ dest/` keeps everything wired.

## Summary

- DVC layers content-addressed storage on top of git: small
  `.dvc` pointer files in the repo; the bytes in a cache and
  a shared remote.
- `dvc add` → `git add` the pointer → `dvc push`. Reproduce
  with `git checkout` + `dvc pull`.
- `dvc.yaml` + `dvc.lock` turn the pipeline into a
  first-class DAG. `dvc repro` only reruns stages whose
  inputs changed.
- MLflow logs *runs*; DVC versions *files*. Bridge them by
  logging the DVC hash as an MLflow param or input.
- Alternatives exist (git-LFS, LakeFS, table formats, HF
  datasets). Pick DVC for the default Python + object-store
  ML project.

## Where this goes next

Chapter 07 covers the parallel construct for the *model*
side of the provenance chain: MLflow's **Model Registry**.
DVC handled the input bytes; the registry handles the
output artefacts and their lifecycle.
