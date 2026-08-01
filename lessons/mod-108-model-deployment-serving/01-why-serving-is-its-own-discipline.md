# 01 — Why serving is its own discipline

> **Reading time:** ~15 minutes.
> **Prerequisite modules:** `mod-104` (classical ML) or
> `mod-105` (deep learning), `mod-106` (experiment tracking),
> `mod-107` (evaluation). You need a trained, validated model
> artifact to serve.

## Motivation

Training a model produces a `.pkl` (or `.pt`, or an MLflow
directory). That artifact will happily live inside a notebook
forever. Nothing about it makes it usable to anyone who is not
you, sitting at your machine, with your Python environment.

**Serving is the discipline of turning that artifact into
something a stranger can call.** In practice "stranger" is:

- a mobile app on the other side of a load balancer,
- a backend service written in a different language,
- an analyst running a nightly batch job,
- a data-science teammate who does not want to install your
  environment,
- **you, six months from now**, who no longer remembers which
  columns the model expects.

Each of those callers wants the same three things: a stable
interface, a predictable latency, and a signal for "is this
thing up?". None of them care what framework you trained the
model in.

This module builds the minimum-viable serving stack that
satisfies all three: **a FastAPI service, in a Docker image,
with health checks and structured logs, load-tested and
handed off with a documented contract.** Everything else in the
module is that sentence in more detail.

## The gap between "trains" and "serves"

Two silent gaps kill most first-attempt deployments. Both are
invisible from a notebook and obvious from an incident report.

**Gap 1 — the input contract is implicit.** Training code
lives inside a `Pipeline` that consumed a `DataFrame` where
column dtypes, one-hot categories, and imputed defaults were
inferred from the training set. When a caller sends
`{"age": 42}` over HTTP, none of those inferences happen
automatically. The pipeline that transformed a DataFrame at
training time has to run *the same way* at request time, on a
JSON payload it never saw during fit. Chapter 02 (packaging)
and chapter 03 (the FastAPI service) exist to make the
contract explicit.

**Gap 2 — everything about the environment is Python-shaped.**
The training script imports `pandas`, `xgboost`, and a
half-dozen scikit-learn transformers by version. It ran on
your laptop, or on a CI runner with your `pyproject.toml`. A
caller does not have that environment. A container makes it
portable (chapter 06); a health check makes it observable
(chapter 04); logs make it debuggable (chapter 05).

Neither gap is exotic. They are the two failure modes behind
the majority of "model works locally, breaks in prod"
incidents. The rest of this module systematically closes them.

## What a serving contract looks like

A serving *contract* — the thing a caller can rely on —
consists of exactly four things:

1. **A URL and a request schema.** `POST /predict` with a
   documented JSON body. Type-checked, validated on entry,
   rejected loudly on mismatch.
2. **A response schema.** A predictable JSON body. Versioned.
   The model version that produced the prediction is in the
   response.
3. **A health check.** An HTTP endpoint (`/healthz`,
   `/readyz`) that returns `200` when the service can serve
   traffic and non-`200` when it cannot. Chapter 04 explains
   the split between "is the process alive?" and "is the
   model loaded and warm?".
4. **A latency and throughput budget.** A p95 latency the
   service commits to under a stated request rate. Without
   this number, autoscaling cannot scale, the platform team
   cannot capacity-plan, and no one knows whether an incident
   is a regression or normal noise. Chapters 09–10 make the
   number concrete.

Miss any of the four and downstream teams (platform, mobile,
oncall) will either paper over the gap themselves — usually
badly — or refuse to depend on the service. Chapter 11 (the
handoff) is where the four items become deliverables.

## Two shapes of inference

Not every model is served over HTTP. The module covers **both
of the paths** an ML engineer is expected to build:

- **Online inference.** The caller wants a prediction *now*.
  A user is waiting on a screen; a mobile app has an open
  socket; a service is filling a response. The interface is a
  request/response HTTP call. Latency is measured in
  milliseconds. Chapters 03–07 target this path.
- **Batch inference.** The caller does not need a prediction
  immediately; they need one for **every row in a table** by
  the time a downstream job runs. The interface is a data
  pipeline — read from S3 or the warehouse, score in bulk,
  write back. Throughput per hour matters; per-request
  latency does not. Chapter 08 covers this path.

An ML engineer will typically own both. The interesting design
work — chapter 08 — is choosing which path fits the use case,
and when a hybrid ("nightly batch pre-score, online serve for
misses") is the right answer.

## What this module *does not* cover

Serving at scale is a large field with many well-defined
sub-specialties. This module is deliberately scoped to what
an ML engineer builds *before* handing off. Explicit non-
goals:

- **Kubernetes deployment.** Manifests, Helm charts, service
  meshes, horizontal pod autoscalers, and node pools belong
  to the **AI Infrastructure Engineer** / **ML Platform**
  tracks. You produce a container that satisfies the
  platform's contract (chapter 11); the platform runs it.
- **GPU-accelerated inference and specialist inference
  servers.** TensorRT, Triton, TorchServe, vLLM, and the rest
  become relevant for latency-critical or throughput-critical
  workloads that outgrow a FastAPI + gunicorn stack. They are
  the natural next step and appear in the platform track.
- **LLM-specific serving.** Token streaming, KV-cache
  management, and prefix caching are the concern of a
  separate, LLM-focused module.
- **A/B and canary rollouts.** Traffic splitting, shadow
  deployments, and automated rollback are infrastructure
  concerns; the ML engineer's contribution is a service that
  identifies its model version so those rollouts can reason
  about it (chapter 05 covers this).
- **Model registry mechanics.** Promoting a model from
  staging to production and cutting over serving to a new
  version is `mod-106`'s territory (registry) plus the
  platform team's rollout tooling.

Knowing what you do not own is as important as knowing what
you do. Chapter 11 is a checklist for the handoff.

## The one-sentence altitude of the module

**Ship the smallest thing a platform team can accept: a
containerised FastAPI service that loads a versioned model,
serves both batch and online inference, exposes a health check,
emits structured logs, and has a load-tested latency budget.**

Every chapter is a piece of that sentence.

## Summary

- Serving turns a `.pkl` on your laptop into something a
  stranger can call. Every failure mode covered in this
  module is a way the "stranger" — human or service — is
  going to be disappointed.
- The two silent gaps are the **implicit input contract**
  (the training pipeline's assumptions) and the **implicit
  environment** (the Python stack it needs). Chapters 02–07
  close them.
- A serving contract has four parts: URL/request schema,
  response schema, health check, latency-throughput budget.
  Anything missing pushes work onto downstream teams.
- Two shapes of inference — online and batch — coexist. The
  ML engineer usually owns both. Chapter 08 is when to pick
  which.
- Kubernetes, service meshes, and specialist inference
  servers are out of scope. Chapter 11 makes the handoff
  explicit.

## Where this goes next

Chapter 02 opens the packaging half of the module: **the
model artifact itself**. Pickle vs. joblib vs. ONNX vs. an
MLflow pyfunc — which format survives a Python upgrade, which
runs outside the training framework, and what metadata you
have to save alongside the weights so the caller in six
months' time can still make sense of the file.
