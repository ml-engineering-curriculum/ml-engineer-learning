# mod-106 — Experiment Tracking & Reproducibility

> **Track:** Machine Learning Engineer (level 20)
> **Estimated effort:** 10 hours

`mod-105` gave you a training loop that converges. This module
answers the two questions every teammate asks the day after you
run it:

1. **"Which run was that?"** — the tracking question. If you
   cannot point at a UI or a JSON file that lists the exact
   parameters, dataset version, code SHA, and metrics of a
   result, the result does not exist as an experiment. It is
   folklore.
2. **"Can I rerun it and get the same number?"** — the
   reproducibility question. If the answer is "roughly", you
   have no baseline: every future change is measured against
   noise. If the answer is "byte-for-byte, yes", every
   subsequent change is measured against signal.

The tools are **MLflow** (tracking + registry), **DVC** (data
versioning), **Optuna** (sweeps), and disciplined use of the
`random` / `numpy` / `torch` seeding APIs. We also make a fair
tour of **Weights & Biases** as the second tracker most teams
encounter.

This is not a tools tour. It is a workflow module: instrument
one training run end to end, version the data and the model,
prove you can reproduce the numbers, run a sweep, and hand a
teammate a report they can rerun without asking you a
question.

## Learning objectives

- Track experiments with MLflow, and know when to reach for
  Weights & Biases (or another tracker) instead.
- Control randomness and pin environments so a run reproduces
  to the metric — byte-comparable where possible, tolerance-
  bounded where it is not.
- Version datasets and models: DVC for data, MLflow Model
  Registry for model artefacts and lifecycle stages.
- Run a small hyperparameter sweep with Optuna and write a
  reproducibility report that a teammate can execute.

## Chapters

| # | Chapter | Maps to objective |
|---|---|---|
| 01 | [Why tracking and reproducibility](01-why-tracking-and-reproducibility.md) | orientation |
| 02 | [MLflow: runs, metrics, artifacts](02-mlflow-runs-metrics-and-artifacts.md) | tracking |
| 03 | [Picking a tracker: MLflow vs Weights & Biases](03-picking-a-tracker-mlflow-vs-wandb.md) | tracking |
| 04 | [Controlling randomness: seeds and determinism](04-controlling-randomness.md) | reproducibility |
| 05 | [Pinning environments: lockfiles, Docker, hardware](05-pinning-environments.md) | reproducibility |
| 06 | [Versioning data with DVC](06-versioning-data-with-dvc.md) | versioning |
| 07 | [Model registry and promotion](07-model-registry-and-promotion.md) | versioning |
| 08 | [Hyperparameter sweeps with Optuna](08-sweeps-with-optuna.md) | sweeps |
| 09 | [The reproducibility report](09-the-reproducibility-report.md) | sweeps |

## Exercises

| # | Exercise | Hours |
|---|---|---|
| 01 | [MLflow runs and metrics](exercises/exercise-01-mlflow-runs-and-metrics.md) | 2 |
| 02 | [Data and model versioning](exercises/exercise-02-data-and-model-versioning.md) | 3 |
| 03 | [Seeded reruns and determinism](exercises/exercise-03-seeded-reruns-and-determinism.md) | 2 |
| 04 | [Hyperparameter sweep with Optuna](exercises/exercise-04-hyperparameter-sweep-with-optuna.md) | 3 |

Plus one lab (`labs/`) and one quiz (`quizzes/`) — scaffolded on
subsequent authoring cycles.

## How to use this module

Read the chapters in order — chapters 06–09 lean on the
MLflow vocabulary from chapters 02–03. The highest-stakes
chapters, worth slowing down for:

- **02 (MLflow runs, metrics, artifacts).** The mental model
  of *experiment → run → params/metrics/tags/artifacts* is
  shared across MLflow, W&B, Neptune, Aim, and every tracker
  you will meet. Learn it once here; every other tracker is a
  translation.
- **04 (Controlling randomness).** The chapter with the
  highest ratio of "spent hours on this" to "chapter length".
  The distinction between *seed the RNG* and *make the run
  deterministic* is the whole game.
- **09 (Reproducibility report).** The artefact that makes
  the module's value visible. If a teammate can rerun your
  best result from what you write, you have shipped the
  outcome the module exists to teach.

Every chapter maps to an exercise. Exercises are sequenced
so each builds on the last: exercise-01 instruments a
run, exercise-02 versions the data and the resulting model,
exercise-03 proves the run is reproducible, exercise-04 sweeps
and reports. Do them in order.

## What this module does **not** cover

- **A serving stack.** Registering a model is not deploying
  it. `mod-108` (deployment) picks up where chapter 07 leaves
  off.
- **Monitoring in production** — drift detection, alerting,
  online metrics. That is `mod-109`.
- **Full MLOps orchestration** (Airflow, Prefect, Dagster,
  Kubeflow, Metaflow). Chapter 09 shows a `dvc.yaml` pipeline;
  scaling that to a full orchestrator is `mod-110`.
- **Distributed sweep scheduling** (Ray Tune, Katib, SageMaker
  HPO). Chapter 08 uses Optuna's local + parallel storage
  patterns. Distributed scheduling is a track-level topic.
- **Bayesian statistics behind Optuna's TPE.** We use TPE as
  a black-box default and cite where to read.

## Prerequisites

- `mod-102` (Python ML toolchain): lockfiles, `pyproject.toml`,
  the `uv` / Poetry basics chapter 05 assumes.
- `mod-103` (Data engineering for ML): a real dataset to
  version.
- `mod-104` (Classical ML) *or* `mod-105` (Deep learning):
  something worth tracking. Every example in the module works
  with either a scikit-learn pipeline or a PyTorch training
  loop.
- Git, comfortably. Every tool in this module (MLflow, DVC,
  Optuna's storage) leans on git for the code side of the
  provenance triangle.

See [`resources.md`](resources.md) for official documentation
and curated references behind each chapter.
