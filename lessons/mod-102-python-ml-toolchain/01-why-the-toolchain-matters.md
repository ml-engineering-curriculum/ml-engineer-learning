# 01 — Why the toolchain matters

> **Reading time:** ~15 minutes.
> **Prerequisite chapters:** none. Comfortable with intermediate
> Python is assumed (see [`PREREQUISITES.md`](../../PREREQUISITES.md)).

## Motivation

`mod-101` taught you the maths an ML engineer cannot avoid. This
module teaches the **tools that turn that maths into running code**.
For a working ML engineer, four libraries dominate the everyday
keyboard:

- **NumPy** — the n-dimensional array. Every model's inputs, weights,
  and outputs are NumPy arrays (or something that quacks like one).
- **Pandas** — the tabular data frame. Almost every ML pipeline starts
  with a CSV / Parquet / SQL result that lands as a `DataFrame`.
- **scikit-learn** — the estimator/transformer API. Even when the
  final model is XGBoost or a PyTorch network, the surrounding
  preprocessing, splitting, and evaluation code is usually scikit-learn.
- **Jupyter** — the exploratory notebook. Good for *thinking*, bad as
  a deliverable. We will treat it as a scratchpad, not a product.

Add the modern packaging tools (`uv`, `poetry`) and the testing /
linting / type-checking stack (`pytest`, `ruff`, `mypy`/`pyright`)
and you have the working surface of an ML engineer's day.

## Why a whole module on this

Two reasons it is worth slowing down for:

1. **Idioms compound.** A two-line vectorised NumPy expression
   replaces a 20-line Python loop *and* runs 100× faster. A
   well-shaped scikit-learn `Pipeline` replaces a 200-line
   imperative preprocessing script *and* eliminates a whole class of
   train/test leakage bugs. Learning the idioms is high-leverage.
2. **Reproducibility is a first-class engineering concern in ML.**
   Models that "worked yesterday" but no longer reproduce because a
   transitive dependency was upgraded are a load-bearing failure mode
   of the field. The packaging and testing parts of this module are
   what prevent that.

## Where this module sits

`mod-101` taught the maths. This module teaches the tools. `mod-103`
applies them to real data engineering. `mod-104` plugs them into
trained models. The skills you build here recur in every later module
and project — the dollar-per-hour return on getting fluent now is
high.

A short orienting map of what each chapter owns:

| Chapter | Owns |
|---|---|
| 02 | NumPy arrays, dtypes, vectorisation, broadcasting at the working-engineer altitude. |
| 03 | Pandas selection, missing data, group-by, joins, and the categorical-dtype escape hatch. |
| 04 | Jupyter notebooks as a scratchpad — what they are good and bad at, and the discipline that keeps them useful. |
| 05 | The scikit-learn API: estimators, transformers, fit / transform / predict, custom transformers. |
| 06 | `Pipeline`, `ColumnTransformer`, and the train/test discipline that prevents leakage. |
| 07 | Reproducible environments: `uv`, `poetry`, lockfiles, pinned versions, deterministic installs. |
| 08 | Testing ML code: unit tests for transformers, golden tests for small models, lint, and type-check. |

## A note on framework choice

This curriculum opinionatedly teaches **NumPy + Pandas + scikit-learn**
for tabular work and **PyTorch** for deep learning (introduced in
`mod-105`). Alternatives exist — Polars for DataFrames, JAX for
arrays/autodiff, etc. — and you should learn them eventually. We
start with the dominant tools because (a) job postings overwhelmingly
ask for them, and (b) the idioms transfer to their alternatives once
you understand the core ideas (vectorisation, the estimator API,
lockfile-based environments).

## How to use this module

Each chapter is short. Read it, then run the matching exercise. The
exercises are calibrated for the working ML engineer altitude:
nothing here is gratuitous; everything reappears in later modules.

If you already use NumPy and Pandas every day at work, skim chapters
02 and 03 and spend your time on 04 (Jupyter discipline), 05–06
(pipelines), 07 (reproducible environments), and 08 (testing). That
is where the highest-leverage idioms hide.

## Summary

- Four libraries dominate everyday ML engineering: NumPy, Pandas,
  scikit-learn, Jupyter.
- Idioms compound, and the packaging/testing tooling is what
  separates "trained a model" from "shipped a model".
- This module is a tour of the working surface — short, dense, and
  reused everywhere later.
