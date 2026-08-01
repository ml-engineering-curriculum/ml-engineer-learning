# 02 — Reference architecture of an ML system

> **Reading time:** ~20 minutes.
> **Prerequisite chapter:** 01. The blueprint tells you
> *what* to build; this chapter is the *shape* of the
> thing you end up with.

## Motivation

Every production ML system a team of ML engineers has
ever shipped, whether it lives on one laptop or spans
three cloud accounts, is a small variation of the same
diagram. The same six or seven components appear each
time, in the same relative positions, connected in the
same directions. Once you can draw the diagram from
memory, you can walk any ML system design conversation
without getting lost — including the "we already have
X, we just need Y" conversation, which is where most of
the design decisions actually live.

This chapter is that diagram, chapter by chapter of the
diagram. It is deliberately vendor-neutral. Whether the
feature store is Feast, Tecton, Databricks Feature Store,
a warehouse view, or a Redis cache with a nightly ETL,
the *responsibilities* the box carries are the same.

## The diagram in one paragraph

The **source systems** produce data. A **feature
pipeline** transforms that data into features and lands
them in a **feature store** with two symmetric read
paths (offline for training, online for serving). A
**training pipeline** reads from the offline store,
trains a model, records the run in the **experiment
tracker**, and (if the run is chosen for release)
promotes the artifact in the **model registry**. The
**serving component** — batch job, streaming worker, or
online service — loads the registered artifact, reads
features from the online store, produces predictions, and
writes them to a **predictions table** and a log stream.
An **observability surface** (the monitoring and drift
component from `mod-109`) watches the whole thing. A
**labels loop** joins outcomes back to predictions so
the next training run has fresh ground truth. An
**orchestrator** schedules the pipelines that make all
of it run on time.

That paragraph, drawn on a whiteboard, is enough for
almost any system-design conversation. The rest of the
chapter walks each box.

## The nine components

```
             +----------------------+
             |   Source systems     |
             | (warehouse, events,  |
             |  APIs, third-party)  |
             +----------+-----------+
                        |
                        v
             +----------------------+
             |  Feature pipeline    |
             | (batch + streaming)  |
             +----------+-----------+
                        |
       offline write    |   online write
                        v
        +-------------------------------+
        |         Feature store          |
        |  offline table  |  online KV   |
        +--------+--------------+--------+
                 |              |
       reads for |              | reads at
       training  |              | serving time
                 v              v
     +---------------------+  +----------------------+
     |  Training pipeline  |  |   Serving path       |
     | (retrain scheduled  |  | (batch / stream /    |
     |  or on-trigger)     |  |  online / edge)      |
     +----------+----------+  +----------+-----------+
                |                        |
                v                        v
     +---------------------+  +----------------------+
     |  Experiment tracker |  |  Predictions table   |
     |  Model registry     |  |  Prediction log      |
     +----------+----------+  +----------+-----------+
                |                        |
                +---------+--------------+
                          |
                          v
                +----------------------+
                | Observability surface|
                |  (mod-109 monitor,   |
                |  drift, alerting)    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Labels loop        |
                | (ground truth join,  |
                |  predictions_labeled)|
                +----------+-----------+
                           |
                           v (back into feature pipeline / training)

    All pipeline runs are scheduled and observed by an Orchestrator
    (Airflow, Dagster, Prefect, etc.).
```

Every ML system has all nine — even if a component is a
20-line script instead of a hosted service. Naming them
out loud in a design review lets you spot the missing
one before it becomes an incident.

### 1. Source systems

Where the data actually lives before anyone in the ML
team touches it. Warehouse tables (Snowflake, BigQuery,
Redshift), the events pipeline (Kafka, Kinesis,
Pub/Sub), third-party APIs (Stripe, Segment, HubSpot),
and internal microservices.

Responsibilities:

- **Data availability.** The team that owns the source
  system is on the hook for uptime, schema stability,
  and change notifications.
- **Schema ownership.** Documented, versioned, ideally
  in a data catalog (`mod-103`).

Sharp edges for the ML engineer:

- **Schema changes are the #1 cause of production
  regressions.** Chapter 07 of `mod-109` walks a real
  case. Anything that can subscribe to schema-change
  notifications should.
- **"Real-time" from the source often isn't.** A
  warehouse table may lag by hours; a stream may buffer
  by minutes. Chapter 03 comes back to this as the
  freshness half of training-serving skew.

### 2. Feature pipeline

Code that transforms source data into features. Lives in
two symmetric halves that must produce *identical*
outputs given the same inputs:

- **Batch feature computation.** Runs on the schedule
  (nightly, hourly) and writes to the offline store.
- **Online feature computation.** Runs on the request
  path (or on a stream) and writes to the online store.

The two halves are the training-serving-skew battlefield
and are the subject of chapter 03. The pattern that
prevents most skew is to write the transform *once* and
run it in both contexts (feature stores like Feast and
Tecton are built around this idea; if you're rolling
your own, achieve the same by putting transforms in a
shared library both callers import).

Feature pipelines often include:

- **Aggregations.** "user's average purchase over 30
  days," "impressions in the last hour."
- **Joins.** "the user's plan_tier as of the event
  timestamp."
- **Encodings.** One-hot, target-encoding, embedding
  lookups.
- **Time-window features.** "days since last purchase,"
  "count of events in the trailing 7-day window."

Time-window features are the most common skew source.
Getting them right is a mod of its own; chapter 03
covers the pattern.

### 3. Feature store

The database (or pair of databases) the training and
serving paths both read from. Not strictly required for
small systems — a warehouse table can play the role of
the offline store, and the online store can be omitted
entirely if serving is batch — but the *concept* is
always present.

Two access patterns:

- **Offline access, point-in-time correct.** Training
  queries "give me this user's features *as they were*
  on 2026-04-14 at 09:00 UTC." The feature store must
  return the historical value, not the current value.
  This is the point-in-time-correctness pattern; it is
  what prevents label leakage (`mod-107` chapter 06).
- **Online access, low latency.** Serving queries "give
  me this user's features right now." Latency budget is
  milliseconds. Values are typically materialised into
  a key-value store (Redis, DynamoDB, Bigtable).

If your system doesn't have a feature store, it has one
of two implicit versions:

- A **warehouse** playing both roles, with the online
  path reading through some cache. Fine for batch and
  low-QPS online systems; breaks down at high QPS.
- A **serving-time feature computation** where the
  service re-derives features on every request. Fine
  for stateless features; catastrophic for time-window
  features (you're going to reimplement the offline
  aggregation and it will diverge).

Build vs buy is a real decision here; chapter 06
covers when the managed feature store pays for itself.

### 4. Training pipeline

Code that takes the offline features, joins them to
labels, trains the model, evaluates it, and produces a
candidate artifact.

Responsibilities:

- **Reproducibility.** Same inputs → same artifact.
  `mod-102` chapter on env pinning; `mod-106` on seeded
  reruns.
- **Deterministic data slice.** The training set is
  defined by a query, and that query is committed to
  the repo. The training set is not "whatever the
  warehouse currently returns."
- **Evaluation gates.** The pipeline computes offline
  metrics against a held-out split. A trained model
  that fails the gate does *not* get promoted; the
  pipeline surfaces the failure and stops.
- **Emits every artifact.** The model file, the
  metrics, the feature-importance report, the
  reference profile (`mod-109` chapter 02), the
  fit-time transformers, the training config, and the
  git SHA. All keyed by a run ID from the tracker.

Runs on the orchestrator on a cadence set by the
blueprint (chapter 01 Section 3). Retraining triggered
by drift (`mod-109` chapter 06) or by a scheduled cron;
either way, one pipeline.

### 5. Experiment tracker and model registry

Two components that are logically distinct but usually
implemented together (MLflow, Weights & Biases, Comet).

- **Experiment tracker.** Records every training run —
  parameters, metrics, artifacts, git SHA, dataset
  hash. The searchable history. `mod-106` chapters
  cover the mechanics.
- **Model registry.** A named, versioned index of
  *promoted* models. `churn-model:v1.4.2` maps to the
  specific artifact from a specific training run. The
  serving path resolves the label ("give me the
  currently-serving model") to the artifact.

Two operations that only the registry supports:

- **Promotion.** A trained model earns a version
  number after passing evaluation and (optionally) a
  shadow-mode or canary rollout (chapter 05).
- **Rollback.** The serving path resolves a version
  label; the label can be changed to point at the
  previous version if the new one regresses.

Naming these two together prevents the antipattern of
"the newest run is always the one that's serving." The
tracker records every attempt; the registry names the
one you stand behind.

### 6. Serving path

The `mod-108` deliverable. One of the four shapes from
chapter 01 Section 4: batch, streaming, online, or
embedded. Common responsibilities regardless of shape:

- **Load the artifact by version.** Explicit version
  pinning; never "load the latest."
- **Read features from the online store.** Or compute
  them stateless if the model's features are computable
  from the request payload.
- **Produce and log the prediction.** Structured log
  line carrying `prediction_id`, feature hash,
  `model_version`, latency (`mod-109` chapter 02).
- **Write to the predictions table.** So the labels
  loop can join outcomes back.
- **Expose health / readiness / metrics endpoints.**
  `mod-108` chapters 04–05.

The serving path is the smallest surface in the
diagram in terms of code but the largest in terms of
attention: it is what the user hits and what generates
the log stream everything else in monitoring consumes.

### 7. Predictions table and prediction log

Two names for the same story: **every prediction the
model made is persistent and queryable.**

- **Predictions table** in the warehouse: one row per
  prediction, keyed by `prediction_id`. Columns:
  `predicted_at`, `model_version`, `subject_id`
  (user, session, item), `prediction`, plus enough of
  the input to reconstruct the decision or link back
  to the request. Retained per your data policy
  (30 / 90 / 365 days).
- **Prediction log** in the log stream: same content,
  different retention. Feeds real-time dashboards and
  the incident-debugging workflow (`mod-109`
  chapter 07).

Without these two, you cannot:

- Compare v_new against v_old on the same requests.
- Attribute a business outcome to a prediction (A/B
  test — chapter 05).
- Debug "why did the model say that for this user?"
- Compute accuracy once labels mature.

Build them first. Every other analysis presupposes
them.

### 8. Observability surface

The `mod-109` deliverable. Watches:

- **The service itself** (via generic APM — `mod-108`
  chapter 05).
- **The predictions** (distribution shift, latency,
  volume per version).
- **The features** (per-feature distributions,
  anomaly counters).
- **The labels** (as they arrive; accuracy on
  matured windows).
- **The pipelines** (last run status, freshness,
  time since last successful retrain).

Everything downstream — the retraining trigger, the
incident response — pulls from this surface. Chapter 07
of `mod-109` walks a real incident using it.

### 9. Labels loop

The join between the predictions table and whatever
system holds ground truth (warehouse, human labeler
queue, upstream event). Produces the
`predictions_labeled` table (`mod-109` chapter 05).

Feeds two consumers:

- The **training pipeline**, which uses the labelled
  predictions as its next training set (data flywheel
  — chapter 04 has the details, including when this
  becomes dangerous).
- The **observability surface**, which computes
  post-hoc accuracy metrics once labels mature.

Every model has a labels loop, even if it is currently
just "SQL a warehouse table nightly." Naming it makes
sure someone owns it.

### 10. Orchestrator

Airflow, Dagster, Prefect, or the cloud provider's
equivalent. Runs the feature pipeline, the training
pipeline, the labels loop, the retraining trigger check,
and the reference-profile refresh on their respective
schedules.

Not glamorous; unusually load-bearing. When the
orchestrator quietly fails to schedule the retraining
job, the model silently ages out of date; when the
labels loop fails, the accuracy metric silently reports
the last successful run's numbers forever. Both are
real incidents. Alerts on job success/failure and
freshness are non-negotiable.

## How the boxes wire in practice

The three most common variations of this diagram, in
one paragraph each:

### The batch-only system

Common for retention models, propensity scoring,
inventory forecasting, and internal analytics ML. The
feature store is the warehouse itself. The training and
serving pipelines are both scheduled warehouse jobs.
The predictions table is one warehouse table. The
online store, the online feature-compute path, and the
FastAPI service *do not exist*. Ninety percent of the
`mod-108` complexity vanishes; the labels loop is a
scheduled SQL join. This is the cheapest possible
production ML system, and it fits more use cases than
teams tend to admit.

### The online-serving system with batch features

The system from `mod-108` and `mod-109` combined:
features are batch-computed and materialised nightly
into an online key-value store; the model is behind a
FastAPI service that reads from the online store and
serves per-request predictions. Streaming is not
involved. Latency budget is 50-200 ms. This covers most
personalisation, ranking, and scoring use cases.

### The streaming system

Features are computed on an event stream and updated in
the online store at near-real-time (Feast + Kafka, or
similar). The serving path reads them. Fraud, real-time
recommendations, and dynamic pricing tend to sit here.
Complexity is meaningfully higher — the streaming
platform, the feature-freshness monitoring, the
exactly-once semantics — and it needs to be justified in
the blueprint's Section 4 (chapter 01).

Chapter 06 will revisit build-vs-buy for the components
above; often the answer is "use the warehouse and skip
half the diagram until you can prove the batch story is
insufficient."

## What "the same model" is composed of

A subtle point that a lot of design conversations
misfire on: **the model in production is not just the
`.joblib` file.** It is the file, the featureisation
code that produced its inputs, the reference profile
that defines "normal," the calibration cutoff that
turns a probability into a decision, and the version-
label the registry resolves to. Change any of the five
without the others knowing and you have shipped a
different model without a version bump.

The registry entry records all five together. The
serving path resolves them together. The monitoring
surface labels every metric with `model_version` so
you can attribute post-hoc metrics to *the exact
combination* that was serving. This is how you
distinguish "the model regressed" from "the featurizer
regressed" from "the calibration cutoff was silently
changed."

## Boundaries with adjacent tracks

The reference architecture is the ML engineer's turf,
but three components have shared or foreign ownership
that has to be named explicitly:

- **Source systems and the data warehouse.** Owned by
  the data engineering team (`mod-103` covers the
  scope for the ML engineer's read-side). The ML
  engineer consumes; they do not own SLAs on the
  warehouse or on upstream schemas.
- **Serving infrastructure below the container.**
  Kubernetes, service mesh, autoscaling, ingress, GPU
  scheduling — the **AI Infrastructure Engineer** and
  **ML Platform Engineer** tracks. `mod-108`
  chapter 11 draws the handoff explicitly.
- **The orchestrator's platform layer.** Airflow /
  Dagster / Prefect *the deployment* is often a
  platform-team thing; the ML engineer owns the DAGs,
  not the cluster.

Naming ownership on the diagram before the design
review saves the meeting from turning into a "who does
that?" argument.

## Common pitfalls

1. **A missing box.** No labels loop; no reference
   profile; no experiment tracker. The system runs
   until it doesn't; then nobody can debug it.
2. **A "feature store" that is two different code
   paths pretending to be one.** The batch job in
   Airflow and the request-time transform in FastAPI
   compute "the same" thing. Chapter 03 has the
   incident.
3. **A model registry that is `model.joblib` in an S3
   bucket.** Fine for a solo project; catastrophic
   for a team. There is no version, no promotion, no
   rollback.
4. **Predictions logged but not persisted.** The log
   ages out at 7 days; the labels arrive at 30 days.
   You never compute accuracy.
5. **Monitoring on the service, not on the
   pipelines.** The service is healthy; the retraining
   job hasn't successfully run in three weeks;
   nobody knows.
6. **A "batch or online?" decision that got skipped.**
   Someone built online serving because someone else
   built online serving. The blueprint (chapter 01)
   didn't say it was needed.
7. **The orchestrator not owned by anyone.** The DAG
   fails silently; the team notices when a downstream
   consumer complains.
8. **Two systems that are "the same model" but not the
   same version.** The candidate model is trained
   with a new featurizer; the serving path still uses
   the old one. Chapter 03 again.
9. **A design review that shows the model box only.**
   The rest of the diagram is left off the slide; the
   reviewers can't attach ownership or SLAs to
   anything.

## Summary

- Every production ML system is a variation of the
  same nine-component diagram: source systems, feature
  pipeline, feature store, training pipeline, tracker
  / registry, serving path, predictions table,
  observability, labels loop, orchestrator.
- Even a laptop-scale system has all nine; some are
  just a 20-line script.
- The two halves of the feature pipeline (offline and
  online) *must* produce the same output. Chapter 03
  is about how they don't.
- "The model" is the artifact plus the featureiser
  plus the reference profile plus the calibration plus
  the version label. All five move together.
- Default to batch; only add streaming or online
  serving with a documented justification.
- The labels loop is what turns predictions into
  training data for the next round; without it, the
  system is a one-shot.
- Boundaries with data engineering, platform, and
  infra are on the diagram, in ink.

## Where this goes next

The architecture on paper is the setup for the two
failure modes that will bring it down in production:
training-serving skew (chapter 03) and feedback loops
(chapter 04). Both are consequences of getting the
component boundaries almost-right; both have known
patterns; both are what a good design review will ask
about.
