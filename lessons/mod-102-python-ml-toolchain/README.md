# mod-102 — Python ML Toolchain: NumPy, Pandas, scikit-learn, Jupyter

> **Track:** Machine Learning Engineer (level 20)
> **Estimated effort:** 12 hours

This module teaches the working surface of an ML engineer's day: the
four libraries you will touch on every project (NumPy, Pandas,
scikit-learn, Jupyter), the packaging tooling that makes Python
environments reproducible (`uv`, `poetry`, lockfiles), and the
testing / lint / type-check stack that keeps the model you trained
yesterday behaving the same way today.

`mod-101` taught the maths; this module teaches the tools that turn
that maths into running code. Every later module — `mod-103`'s data
engineering, `mod-104`'s classical models, `mod-105`'s PyTorch loops
— quietly assumes you can write a reproducible, tested scikit-learn
pipeline. Spending time here pays back in every chapter that
follows.

## Learning objectives

- Use NumPy and Pandas idiomatically for vectorised ML data
  manipulation.
- Use scikit-learn pipelines, estimators, and transformers correctly.
- Build a reproducible Python environment with modern tooling (`uv`
  or `poetry`, lockfiles, pinned versions).
- Test ML code: unit-test transformers, golden-test small models,
  lint and type-check.

## Chapters

| # | Chapter | Maps to objective |
|---|---|---|
| 01 | [Why the toolchain matters](01-why-the-toolchain-matters.md) | orientation |
| 02 | [NumPy for ML](02-numpy-for-ml.md) | NumPy idioms |
| 03 | [Pandas for ML](03-pandas-for-ml.md) | Pandas idioms |
| 04 | [Jupyter: useful scratchpad, dangerous deliverable](04-jupyter-as-a-scratchpad.md) | notebook discipline |
| 05 | [scikit-learn: estimators and transformers](05-sklearn-estimators-and-transformers.md) | sklearn API |
| 06 | [Pipelines and `ColumnTransformer`](06-pipelines-and-columntransformer.md) | sklearn pipelines, leakage |
| 07 | [Reproducible Python environments](07-reproducible-python-environments.md) | uv / poetry / lockfiles |
| 08 | [Testing ML code: unit, golden, lint, type-check](08-testing-ml-code.md) | testing ML code |

## Exercises

| # | Exercise | Hours |
|---|---|---|
| 01 | [NumPy / Pandas fluency](exercises/exercise-01-numpy-pandas-fluency.md) | 2 |
| 02 | [scikit-learn pipeline design](exercises/exercise-02-sklearn-pipeline-design.md) | 3 |
| 03 | [Reproducible environment setup](exercises/exercise-03-reproducible-env-setup.md) | 2 |
| 04 | [Testing ML code](exercises/exercise-04-testing-ml-code.md) | 3 |

Plus one lab (`labs/lab-01`) and one quiz (`quizzes/`).

## How to use this module

Each chapter is short and focused on one objective. Read the
chapter, then do the matching exercise — the idioms become muscle
memory in the keyboard, not on the page. If you already use NumPy
and Pandas every day, skim chapters 02 and 03 and spend your time
on 04 (Jupyter discipline), 06 (pipelines), 07 (reproducible
environments), and 08 (testing). Those four are the
highest-leverage content for an experienced practitioner.

## What this module does **not** cover

- Real-world data ingestion, feature engineering, schema validation
  — see `mod-103`.
- Specific tabular model families (trees, ensembles, boosting) —
  see `mod-104`.
- Deep learning frameworks (PyTorch / Lightning / accelerators) —
  see `mod-105`.
- Experiment tracking (MLflow, W&B) — see `mod-106`. Reproducibility
  in this module ends at the lockfile; tracking and dataset
  versioning is `mod-106`'s domain.
- Polars, JAX, and DuckDB — credible alternatives that we point at
  but don't teach. NumPy + Pandas + scikit-learn is the dominant
  stack; learn it first.

See [`resources.md`](resources.md) for the official documentation
and curated references behind each chapter.
