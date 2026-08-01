# 08 — Batch vs online inference

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01–07.

## Motivation

Every ML system runs predictions in one of two shapes:

- **Online** — a caller sends a request and waits for the
  answer. Latency is measured per request; throughput is
  requests per second.
- **Batch** — a job runs on a schedule, scores every row of
  a dataset, and writes the predictions somewhere for
  downstream consumption. Throughput is rows per hour;
  per-row latency is irrelevant.

Chapters 03–07 built the online path. This chapter names the
batch path, contrasts the two, and covers the *hybrid*
patterns most real systems settle into.

The decision "batch or online?" is not primarily a technical
one. It is a product question — *when does the caller need
the prediction?* — and the answer determines the entire
serving architecture. Get it right and the system is small
and cheap; get it wrong and you either miss SLAs (batch when
you needed online) or spend 10× more than necessary (online
when batch would have sufficed).

## The two shapes at a glance

|                      | **Online**                                | **Batch**                                    |
|----------------------|-------------------------------------------|----------------------------------------------|
| Trigger              | HTTP request                              | Schedule (cron, Airflow, DAG)                |
| Latency SLA          | milliseconds to a few seconds             | minutes to hours                             |
| Success metric       | p95 latency, availability                 | wall-clock, job success rate                 |
| Interface            | REST / gRPC                               | file / table / queue                         |
| Scaling model        | per QPS                                   | per row count                                |
| Cost profile         | pay for headroom (always-on capacity)     | pay for burst (rent capacity, shut it down)  |
| Feature freshness    | at request time                           | as fresh as the last batch upstream          |
| Feedback loop        | immediate — the caller uses the prediction | asynchronous — predictions consumed later    |
| Failure mode         | request fails, caller retries             | job fails, someone reruns                    |
| Typical tooling      | FastAPI, gRPC, Triton, KServe             | Spark, Ray, Airflow, dbt, Beam               |

The two are not competing — they are complementary. Any
non-trivial ML system runs both.

## Batch inference — the anatomy

A batch job is a normal data pipeline that happens to call a
model. The primitive is: **read a table, apply the model,
write a table.**

```python
# jobs/nightly_score.py
import argparse
from pathlib import Path

import joblib
import pandas as pd


def main(input_uri: str, output_uri: str, model_uri: Path,
         partition_col: str = "region"):
    model = joblib.load(model_uri)
    df = pd.read_parquet(input_uri)
    # score in chunks to keep memory bounded
    outputs = []
    for chunk in _chunks(df, size=10_000):
        proba = model.predict_proba(chunk.drop(columns=[partition_col]))[:, 1]
        outputs.append(chunk[[partition_col]].assign(score=proba))
    result = pd.concat(outputs)
    result.to_parquet(output_uri, partition_cols=[partition_col])


def _chunks(df: pd.DataFrame, size: int):
    for i in range(0, len(df), size):
        yield df.iloc[i:i + size]
```

Wrap this in an Airflow DAG, a Prefect flow, a `dagster` job,
or a simple `cron` — the choice depends on the platform
around it, not on the ML.

The engineering surface is different from online serving:

- **The unit of work is the job, not the request.** No
  request ID, no per-row error handling (a bad row is
  usually a signal to fail the job, not to serve a garbage
  answer). Structured logging still applies — chapter 05's
  patterns transfer directly to the job.
- **Idempotency matters.** The job may be reruns for the
  same date; the write step should be idempotent — write to
  a versioned path (`s3://scores/2026-06-14/`) rather than
  overwriting.
- **Bounded memory.** A dataframe with 100M rows does not
  fit in RAM. Chunk the read and the score in the same
  loop. If the data is very large, Spark / Ray / Dask
  handle the partitioning for you.
- **Predictable throughput.** The job is deterministic
  given inputs; it either finishes in the window or it
  doesn't. If not, parallelise partitions, distribute across
  workers, or reduce the size of the input.

The success metric is: **the scored table exists in the
target path by the target time.** No p95 latency, no
autoscaling story. That is a simpler operational model.

## Online inference — recap

Chapters 03–07 covered this. The one-line summary: a FastAPI
service loads a model at startup, exposes `/predict` /
`/predict:batch`, hits its latency budget, emits structured
logs, and runs in a container with health checks.

Two properties that constrain what online can do:

- **Latency-bound.** Predictions have to return in the
  caller's timeout. The model can be as complex as the
  latency budget allows, no more.
- **Features have to be available at request time.** If a
  feature requires a table join or an aggregation over
  historical events, that computation has to happen at
  request time — usually via a low-latency feature store.

Both are non-trivial. Chapter 09 unpacks the first; the
second is `mod-103` (data engineering) territory.

## The decision matrix

Pick online when **any** of these is true:

- **A human is waiting.** Interactive UI, in-app
  recommendation, real-time pricing.
- **The prediction is used in a synchronous decision.**
  Fraud check that must approve or decline the transaction
  before it commits; recommendation used to render the
  next page.
- **The feature values are only known at request time.**
  Session state, cart contents, the search query the user
  just typed.

Pick batch when **all** of these are true:

- **The consumer is asynchronous** — a downstream table, a
  message queue, an email campaign that runs at 06:00.
- **The features are known ahead of time** — attributes
  that update on a daily / hourly cadence.
- **A stale-by-hours prediction is acceptable.** The
  customer's churn probability probably didn't change since
  9 pm last night.

If any online condition is true, you probably need online;
if all batch conditions are true, batch is cheaper and
simpler. Most systems have a mix — some predictions online,
some batch — and that is fine.

## Hybrid pattern 1 — pre-compute, serve from cache

The most common hybrid. Batch-score every user overnight;
write the score to a low-latency key-value store (Redis,
DynamoDB, Bigtable); the online service reads by user_id.

```
┌──────────────┐  cron: 02:00     ┌──────────────┐
│ Batch job     │  ──────────────▶ │  KV store    │
│ score every   │   writes         │  (user_id →  │
│ user          │                  │   score)     │
└──────────────┘                  └──────┬───────┘
                                          │ read on request
                                          ▼
                                   ┌──────────────┐
                                   │ Online API    │
                                   │  GET /score   │
                                   └──────┬────────┘
                                          │
                                          ▼
                                         caller
```

**When it's right.** The set of subjects is bounded and
knowable in advance (all users, all products, all documents).
The score changes slowly — daily / hourly. The online path
becomes a *lookup*, not an inference; latency is a KV read
(1–5 ms) rather than a model prediction (10–100 ms).

**Cost story.** The batch job runs once per period, on rented
capacity that shuts down when done. The online path is a
thin cache layer — much cheaper than serving a full model on
every request.

**Failure mode.** New subjects appear (a new user signs up).
The KV store has no entry for them. Choose ahead of time:
default value, fallback to online inference, or reject the
request. Chapter 09 revisits this as a fallback pattern.

## Hybrid pattern 2 — online inference with pre-computed features

The subject varies per request (an arbitrary transaction),
so pre-scoring the world doesn't work. But some *features*
of the subject are stable — the user's account age, their
average transaction size in the last 30 days.

```
┌──────────────┐               ┌──────────────┐
│ Batch job     │   writes      │ Feature       │
│ compute       │  ─────────▶  │ store         │
│ features      │               │ (user_id →   │
└──────────────┘               │  features)   │
                                └──────┬───────┘
                                        │ lookup at request time
                                        ▼
   request                       ┌──────────────┐
   ──────▶ Online API ──▶ lookup ┤ + join with  │──▶ model ──▶ response
                                 │ request feats │
                                 └──────────────┘
```

**When it's right.** Features fall into two families —
"stable, expensive to compute" (30-day aggregates,
embedding vectors, graph features) and "request-time only"
(the current session, the requested item). The batch job
handles the first; the online request handles the second;
the two join at inference time.

This is the *feature store* pattern (Feast, Tecton,
SageMaker Feature Store, Vertex AI Feature Store, and
in-house equivalents). It is the industry-standard way to
serve online inference on rich features; it belongs to
`mod-103` in this curriculum, but the shape is worth
knowing here because it is *the* reason online inference is
tractable at scale.

## Hybrid pattern 3 — micro-batching on the request path

Every request in isolation is one prediction. But **inside
the service**, a coordinator collects incoming requests for
a few milliseconds, batches them into a single model call,
and returns each caller their slice of the output.

```
     request A ─┐
     request B ─┤─▶ [ batch collector, window = 10 ms ] ─▶ one model call ─▶ split
     request C ─┘                                             (batch of 3)   into
                                                                             3 responses
```

**Why bother?** Many models (deep neural nets, GPU-served
LLMs, some tree ensembles) run much faster per-example on a
batch than on a single input. A GPU serving one image at a
time is idle 95% of the time; batching four images together
is essentially free. Micro-batching converts online traffic
into shapes the model likes.

**The trade-off.** You add a small latency floor (the
collection window) in exchange for higher throughput. On
CPU models with millisecond inference, the trade rarely
pays. On GPU models with 10–100 ms inference and
throughput-limited hardware, it can be transformative.

**Do not build this yourself the first time.** Triton
Inference Server, Ray Serve, TorchServe, KServe, and vLLM
all ship a well-tested micro-batching implementation.
Chapter 11 (handoff) is where they show up.

## Choosing chunk / batch size

Whether you are scoring 100M rows overnight or micro-
batching an online stream, "how big is each batch to the
model?" is a knob with real consequences:

- **Too small.** Per-batch overhead (Python-level dispatch,
  DataFrame construction, GPU kernel launch) dominates the
  compute. Throughput is low; the machine is idle waiting.
- **Too large.** Memory footprint spikes. On CPU you swap;
  on GPU you OOM.

Pick empirically. Time single-batch inference at sizes 1,
10, 100, 1 000, 10 000; the throughput usually plateaus
somewhere between 100 and 1 000 for CPU inference, higher
for GPU. For a batch job scoring millions of rows, the
sweet spot is often 10 000 – 100 000 rows per chunk (small
enough to fit in memory, large enough to amortise
overhead).

Log the chosen batch size and the per-batch latency so
tuning is reproducible.

## Latency vs. throughput — the trade-off

The same model, the same hardware, two different
optimisation goals:

- **Latency-optimised** — small batches (often size 1),
  low queue depth, over-provisioned to keep queue empty.
  Serves 1 request in 20 ms. Sits idle otherwise.
- **Throughput-optimised** — large batches, deep queues,
  fully-loaded machines. Serves 1 000 requests in 200 ms;
  each individual request waits ~150 ms in the queue plus
  ~50 ms in the model, but the machine is producing 5 000
  predictions per second.

You cannot simultaneously optimise for both on the same
hardware. Online services usually pick latency; batch jobs
pick throughput; hybrid patterns exist to move workloads
onto the right side of the trade-off.

Chapter 09 formalises this — latency percentiles, queuing,
concurrency — and chapter 10 measures it.

## Batch scoring: the pragmatic patterns

A few operational notes that trip up first-time batch jobs:

- **Idempotent output paths.** Write to
  `s3://scores/dt=2026-06-14/run=<uuid>/`, not to a fixed
  `s3://scores/current/`. Rerunning is safe; downstream
  consumers point at the newest run.
- **A manifest file.** Alongside the output, write a
  `_MANIFEST.json` with the model version, the input path,
  the row count, and the run start / end time. Downstream
  consumers use it to detect "did the job actually run
  today?" without listing partitions.
- **Progressive schema check.** If the input has an
  unexpected column, or a column of the wrong dtype, fail
  the job with a clear error. Don't score with wrong data
  and pretend it succeeded.
- **Recovery from partial failure.** A 4-hour job that
  fails at hour 3 should be resumable from the point of
  failure, not started over. Either partition the job so
  each partition is independent, or checkpoint progress to
  a durable store.
- **Backfills.** Sooner or later someone will ask for
  "score every user for the last 30 days". Write the job so
  it accepts a date range, not a fixed "yesterday".

## What owns which

Which team owns which piece is almost always the source of
the argument. A defensible split:

- **The ML engineer** owns the model, the packaging, the
  scoring code (both batch and online), and the tests.
- **The data engineering team** owns the input tables the
  batch job reads and the output tables downstream
  consumers read.
- **The platform team** owns the online serving
  infrastructure (Kubernetes, autoscaling, ingress) and
  the batch orchestration platform (Airflow, Spark
  cluster).

That split maps onto this curriculum: `mod-103` is your
side of the data engineering interface; `mod-108`
(this module) is the model / serving code; `mod-110`
(systems design) is where all three come together.

## Common pitfalls

1. **Building online for a batch-shaped problem.** A
   nightly email campaign that scores every user once
   should not be an HTTP service that the campaign job
   calls in a loop.
2. **Building batch for an online-shaped problem.** A
   fraud-detection score that has to arrive before the
   transaction commits cannot come from a nightly job.
3. **Serving stale batch scores as online predictions
   without saying so.** Callers get "no data for user X" or,
   worse, yesterday's score for a user whose behaviour
   changed today. Timestamps the score; log its age.
4. **The batch job overwrites yesterday's output.** No
   recovery from a bad run; no diffing between runs. Write
   to a versioned path.
5. **No manifest, no idempotency, no schema check.** Every
   run is a small trust exercise. Adding these three costs
   an hour and pays for itself the first time.
6. **Batch job that scores in a single dataframe.** Blows
   up memory on the first 10× data growth. Chunk from the
   start.
7. **Micro-batching invented in-house.** Complicated to get
   right (adversarial latency spikes, tail-latency
   coupling). Prefer a mature inference server.
8. **Feature drift between batch and online.** The batch
   job computed `age_in_months = today - signup_date`;
   the online service computed `age_in_months = today -
   signup_date + timezone_hack`. The two produce
   subtly different features for the same user. Share the
   code path.

## Summary

- Batch and online are the two shapes of inference; both
  are the ML engineer's responsibility. The decision
  between them is a product question ("when does the
  caller need the answer?"), not a technology question.
- Batch is a data pipeline that calls a model. It is
  cheaper per row and simpler operationally, but only fits
  when a slower-than-instant answer is acceptable.
- Online is a request/response service (chapters 03–07).
  Latency-bound; expensive per prediction; the only option
  when a caller is waiting.
- **Hybrid patterns dominate real systems.** Pre-compute
  and cache; batch features + online model; micro-
  batching on the request path. Each targets a specific
  trade-off.
- Get the chunk / batch size right by measurement. It is
  the single largest tuning knob in both paths.

## Where this goes next

Chapter 09 sharpens the language for online performance:
p50, p95, p99, tail latency, queueing, concurrency models,
the difference between per-request timing and end-to-end
timing. Chapter 10 puts a load generator against the
service and does the measurement.
