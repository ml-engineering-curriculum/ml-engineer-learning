# 01 — Why tracking and reproducibility

> **Reading time:** ~15 minutes.
> **Prerequisite modules:** `mod-104` or `mod-105` (something
> worth tracking).

## Motivation

Two failure modes account for most of the pain in real ML
projects, and neither has anything to do with the model:

1. **"Which run was that?"** Someone shows a promising number
   in Slack. Two weeks later you cannot find the code that
   produced it. The dataset it trained on has been overwritten.
   The random seed is unknown. The result exists only as a
   screenshot.
2. **"Can I rerun it?"** You want to compare a new idea
   against yesterday's baseline. You re-run yesterday's
   script. You get a different number. Now you are
   comparing two changes at once — your new idea, and
   whatever drifted underneath you — and you cannot separate
   them.

Both failure modes are silent. Neither model performance nor
test coverage catches them. A team can ship broken results
for a year without noticing, because the results *look* fine
on the day they were produced.

This module builds the two habits that eliminate both
failure modes:

- **Tracking.** Every run writes its parameters, metrics,
  code SHA, data version, environment fingerprint, and
  produced artefacts to a central store. When someone points
  at a number, you can point at the run.
- **Reproducibility.** Every run is a function of inputs you
  can name and pin. Given the same inputs, the same run
  produces the same outputs — byte-for-byte where the
  hardware allows, within a stated tolerance where it does
  not.

## The provenance triangle

Every experimental result has three inputs. If any one is
missing, the result cannot be reproduced:

```
              ┌─────────┐
              │  code   │
              └────┬────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
  ┌──────┐    ┌────────┐    ┌────────┐
  │ data │    │ config │    │  env   │
  └──────┘    └────────┘    └────────┘
```

- **Code** — the script, its dependencies, and the git SHA
  at the moment the run started.
- **Data** — the exact bytes fed into the model. A dataset
  name is not enough; two people saying "the customer
  churn dataset" can mean two different tables.
- **Config** — hyperparameters, seed, split, feature list.
- **Env** — the Python interpreter, the resolved dependency
  versions, the CUDA / cuDNN versions, sometimes the GPU
  model.

A tracker (MLflow, W&B, Neptune, Aim, …) records each of
these against a run ID. A versioning tool (DVC, git-LFS,
LakeFS, or a content-addressed store) pins the data so its
recorded fingerprint still resolves months later. A
lockfile + Docker image pins the environment.

Chapter 02 introduces the tracker. Chapters 04–07 pin the
other three vertices.

## What "reproducible" actually means

The word gets stretched to cover three very different bars.
Call each by its name:

- **Repeatable.** Same code, same data, same environment,
  same hardware, same seed → same result. The strictest
  bar. Achievable on CPU-only PyTorch and scikit-learn with
  effort; achievable on GPU only with a specific set of
  deterministic-algorithm flags (chapter 04) and often at a
  performance cost.
- **Reproducible.** Same code, same data, same environment,
  *different* hardware → the same result within a stated
  tolerance (e.g. "top-line accuracy within 0.1%"). This is
  the bar most published results claim.
- **Replicable.** Same experimental *idea*, re-implemented
  from a description → a qualitatively similar result. A
  paper is replicable if a different team can build a
  working version. This is a research-community bar; not our
  target.

This module aims at **repeatable on CPU** and **reproducible
on GPU**. Chapter 04 explains why the stricter bar is worth
paying for on CPU and why it is often not worth chasing on
GPU.

## What a tracker gives you that a spreadsheet does not

Every team's first tracking system is a spreadsheet, a
`results.md`, or a Slack channel called `#experiments`. All
three fail the same way: they capture what the human
remembered to write down and nothing else. When a result
matters six months later, the missing fields are the ones
you needed.

A tracker replaces the discipline of remembering with the
discipline of instrumenting. You write `mlflow.log_metric`
once; it fires every time you train forever. Concretely, a
tracker gives you:

- **A canonical run ID** you can paste into any bug report.
- **Structured params and metrics** you can query
  (`experiment_id = 42 AND params.lr < 1e-3 AND metrics.val_acc > 0.8`).
- **Attached artefacts** — the trained model, the loss
  curve, a confusion matrix, the exact `requirements.txt`
  that resolved for that run.
- **Comparison views** — plot ten runs' loss curves on the
  same axes; sort a table of runs by validation metric.
- **Provenance links** — git SHA, dataset version, parent
  run ID for a resumed or forked run.

The spreadsheet has one advantage: it is trivially portable.
A tracker earns its complexity by making the queries above
one-click instead of one-hour.

## Which parts of ML need this most

Not every script needs a tracker. A quick sanity check plot
in a notebook is fine as-is. Reach for the tooling in this
module when any of these is true:

- **The result is going to be compared against another
  result.** Yesterday's baseline vs. today's variant. Two
  seeds' variance. Two teams' models on the same holdout.
- **The result is going to be shared.** With a teammate, a
  reviewer, a customer. If you cannot point to a run, you
  cannot answer their follow-up questions.
- **The result is going to be re-run.** Because a new dataset
  arrived, because the metric changed, because someone
  wants to see if a bug fix moved the number.
- **The result is going to be promoted.** Someone is going
  to deploy the model. Model registry (chapter 07) is not
  optional when a model reaches production.

A useful heuristic: **if the result will still matter
tomorrow, track it today.**

## The four objectives, one paragraph each

- **Tracking (chapters 02–03).** MLflow is the default: open
  source, self-hostable, integrates cleanly with everything
  you already use. Weights & Biases is the most popular
  hosted alternative and is worth knowing. Both share the
  same mental model — experiment → run → params + metrics +
  artefacts — and translating between them is straightforward
  once you have one in your muscle memory.
- **Reproducibility (chapters 04–05).** Randomness is
  controlled at three layers (Python, NumPy, framework);
  environments are controlled with lockfiles and container
  images pinned by digest; hardware differences are made
  explicit with a manifest. The goal is not "identical
  everywhere" but "identical where it matters, bounded
  where it does not".
- **Versioning (chapters 06–07).** DVC is git for large
  files: it tracks dataset content by hash and stores the
  bytes in an object store, so a commit uniquely identifies
  the dataset it was trained on. MLflow's Model Registry
  is the parallel construct for model artefacts, with named
  versions, aliases (`@champion`, `@challenger`), and a
  promotion workflow that fits neatly onto CI.
- **Sweeps and the report (chapters 08–09).** Optuna
  automates hyperparameter search with samplers smarter than
  random. MLflow captures every trial as a run so the sweep
  itself is a first-class experiment. The report at the end
  is the deliverable: it is what turns "I got X" into "here
  is exactly how you get X".

## Common objections, briefly answered

- **"This is heavyweight."** MLflow's local mode is a single
  `pip install mlflow` and a folder called `mlruns/`. DVC's
  local remote is a folder. Optuna's default storage is
  in-memory. Every tool in this module has a five-line
  "just run it locally" configuration. Scale comes later.
- **"We're moving too fast to bother."** The tracker is the
  reason you can move fast. Every hour spent instrumenting
  is repaid the first time you have to answer "what was that
  run again?" and can do it in twenty seconds.
- **"I only need this for the model that goes to prod."**
  You will not know which model goes to prod until after
  you have run it. By then the useful instrumentation was
  either baked in or not.
- **"We have a Jira ticket instead."** A ticket is a promise
  to write down the answer. Instrumentation *is* the answer,
  written down automatically, every time, in the same
  place.

## Summary

- Two silent failure modes drive most ML-project pain: you
  cannot identify past runs, and you cannot reproduce them.
- A tracker fixes the first; a versioning + environment
  discipline fixes the second. Together they let you compare
  today's variant against yesterday's baseline honestly.
- "Reproducible" has three bars — repeatable, reproducible,
  replicable. This module aims at the first on CPU and the
  second on GPU.
- Instrumentation buys speed, not slowness. The overhead is
  paid once; the returns compound over every subsequent
  question about the run.

## Where this goes next

Chapter 02 introduces the tracker we will use for the rest
of the module: **MLflow**. The three objects to learn are
*experiment*, *run*, and the four things a run carries —
params, metrics, tags, and artefacts.
