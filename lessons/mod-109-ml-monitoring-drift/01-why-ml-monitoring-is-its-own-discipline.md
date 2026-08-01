# 01 — Why ML monitoring is its own discipline

> **Reading time:** ~15 minutes.
> **Prerequisite modules:** `mod-107` (evaluation),
> `mod-108` (deployment). You need a model that has been
> served to real traffic — even in a staging environment —
> for this module to feel real.

## Motivation

The service from `mod-108` is live. `/healthz` returns 200.
The pod's CPU sits at 30%. Prometheus says p95 latency is
78 ms. Every generic observability signal is green.

The model is quietly wrong. A shift in the input traffic
last week moved half of the requests into a corner of
feature space the training data barely covered; the model
compensates by predicting the majority class for those
requests; the users who see those predictions bounce off
the product without ever complaining; the business metric
drops 4% before anyone notices.

**No dashboard the platform team runs would have caught
this.** Prometheus does not measure feature distributions.
The Grafana board for the service does not know what the
prediction distribution *should* look like. The oncall
runbook has no entry for "the model was accurate on
Tuesday and inaccurate on Wednesday, but nothing crashed."

This is the gap the module lives in. Generic observability
(APM, logs, metrics, traces) is necessary and does not
cover the ML-native failure modes. **ML monitoring is the
discipline of measuring what a model believes about the
world, and noticing when the world stopped agreeing.**

## What generic observability already gives you

`mod-108` bought you a lot for free. Kubernetes, Prometheus,
your log aggregator, and your APM tool (Datadog, New Relic,
Honeycomb, take your pick) already answer:

- **Is the process alive?** Liveness probe.
- **Is it accepting traffic?** Readiness probe, ingress logs.
- **Is it fast?** Request latency histogram; p50/p95/p99.
- **Is it erroring?** HTTP 5xx rate; exception counter.
- **How busy is it?** CPU, memory, request rate.
- **Which version is running?** Deployment metadata,
  image tag.

These signals catch outages. They catch deploys that break
the request contract. They catch OOMs, slow downstream
dependencies, and pods that fail to schedule.

They do **not** catch:

- A model that returns valid predictions the users find
  useless.
- Feature values that look plausible per-request but have
  shifted in aggregate.
- A retraining pipeline that produced a technically-valid
  artifact whose accuracy quietly dropped 5 points.
- A downstream data source that started sending zeros
  instead of nulls, silently changing what the model sees.
- A concept in the world (fraud patterns, user preferences,
  price elasticity) that moved out from under a model that
  no one had retrained in six months.

Every failure mode in that second list is invisible to a
service dashboard. Every one has to be measured by the ML
engineer, on ML-native signals. That is this module.

## Three kinds of shift you have to distinguish

The literature and the tooling call these different things.
Fix the vocabulary before writing any code. Given inputs
`X`, labels `y`, and model `f(X) ≈ ŷ`:

1. **Data drift** — the input distribution `P(X)` changes.
   Also called **covariate shift** in the statistics
   literature. Users start uploading larger images; a new
   country ramps up and skews the age distribution; a
   sensor firmware update changes the range of a reading.
   You measure this without any labels; the raw request
   stream is enough. Chapter 03 is entirely this.

2. **Concept drift** — the relationship `P(y|X)` changes.
   The inputs may look identical; the *right answer* is
   different. Fraud patterns evolve to evade detection;
   user preferences shift in response to a product change;
   a competitor changes pricing and elasticity moves.
   Measuring this requires ground truth. Chapter 04 covers
   the measurement; chapter 05 covers how you *get* the
   ground truth in production.

3. **Prediction-distribution shift** — `P(ŷ)` changes even
   though nothing about the code has changed. This is the
   proxy signal you get to watch in real time: the model's
   own output distribution moves. Sometimes it moves
   because `P(X)` moved (data drift causing prediction
   drift); sometimes because the world moved (concept
   drift you cannot yet confirm). Prediction-distribution
   shift is what you alert on first because it is cheap
   to measure and correlated with both of the others.

There is a fourth term, **label shift** (`P(y)` changes,
`P(X|y)` stable), that shows up in class-imbalance
literature. For most production ML systems, treating drift
along the three axes above is enough; label shift is a
special case that surfaces the same way — the ratio of
predicted classes moves.

Being sloppy about which one you mean makes conversations
with data scientists frustrating and root-causes slower.
When you say "the model is drifting," someone should be
able to ask "in `P(X)`, `P(y|X)`, or `P(ŷ)`?" and get a
crisp answer.

## What this module builds

By the end of the module you will have added, to a service
that already looks like the one from `mod-108`:

- **An ML-metrics surface** (chapter 02): prediction
  distribution, feature summaries, latency and volume
  labelled by `model_version`, in Prometheus and in your
  log stream. Independent of whether ground truth ever
  arrives.
- **A drift-detection pipeline** (chapter 03): a scheduled
  job that compares recent request windows against a
  reference profile, emits univariate and multivariate
  drift scores, and fires alerts when they cross
  calibrated thresholds.
- **A concept-drift and prediction-shift monitor**
  (chapter 04): the delayed-label story, and the proxy
  signals you use in the meantime.
- **A ground-truth collection loop** (chapter 05): how
  labels flow back from the product, the warehouse, or a
  labeling queue; how you join them to predictions;
  the metrics that fall out.
- **A retraining trigger** (chapter 06): the decision rule
  that turns monitoring signals into a decision to
  retrain, and the guardrails that stop retraining from
  making things worse.
- **A debugging story** (chapter 07): the walkthrough of a
  production regression from the moment the alert fires
  to the post-mortem entry.

Everything is layered on top of the `mod-108` service —
new endpoints, a background job, a table in the warehouse,
and a Grafana board. You do not need Kubernetes expertise
to build any of it.

## The one-sentence altitude of the module

**Instrument the ML-specific signals that generic
observability does not surface, calibrate drift detectors
on them, tie the resulting alerts to a retraining decision
rule, and rehearse the end-to-end debugging story so the
day of the first real regression is not the first time you
have run through it.**

Every chapter is a piece of that sentence.

## What this module is *not*

Some closely related work belongs in other modules and
other tracks. Being explicit about the boundaries prevents
scope creep and finger-pointing:

- **Feature-store schema evolution.** Owned by
  `mod-103` (data engineering) and the feature-store team.
  This module *consumes* the feature schema; it does not
  own it.
- **Model registry and experiment lineage.** `mod-106`
  owns the registry, the run store, and the artifact-to-
  training-run link. This module reads from the registry
  ("what version is deployed?") and writes back
  ("this version regressed") but does not implement it.
- **Serving-infrastructure observability.** Autoscaling
  metrics, pod restarts, ingress error rates, service-mesh
  latency — the AI Infrastructure Engineer / ML Platform
  tracks. You will consume some of these signals in
  chapter 07's debugging; you do not build the platform
  that emits them.
- **LLM- and generative-specific evaluation.** Toxicity
  scoring, hallucination detection, semantic similarity
  drift on prompts and completions — a distinct discipline
  that reuses some machinery from this module and adds a
  lot of its own. Out of scope here.
- **Fairness and bias monitoring.** Overlaps with drift
  monitoring (subgroup performance shifts) but is its own
  regulatory and methodological body of work. Covered
  briefly in `mod-107`; a dedicated topic in specialised
  tracks.

Knowing what you do not own is as important as knowing
what you do.

## Why this is uniquely the ML engineer's job

Two structural reasons:

1. **You are the only role that can name the invariants.**
   The platform team can measure that traffic dropped 3%;
   they cannot measure that `income` has drifted from a
   log-normal centred at $52k to one centred at $47k, or
   that the predicted-churn distribution moved from a mean
   of 0.12 to 0.19. You know the model. You wrote the
   feature pipeline. You picked the loss function. You
   are the person who can turn "we retrained six months
   ago" into a monitoring plan.

2. **You are the only role that can act on the signal.**
   When drift fires, the platform team cannot decide
   whether to retrain, roll back, downweight the new
   segment, or wait a week and see. That is an ML judgment.
   It belongs to the person who trained the model and can
   train the next one.

Both reasons mean that "monitor the model" cannot be
delegated to the platform team, an SRE, or a generic
observability tool. It is on you, and the goal of this
module is to make it a habit rather than a heroism.

## Common pitfalls before you write any code

Skim these once so you notice them when they show up in
later chapters:

1. **"Accuracy in prod" as the only metric.** Accuracy
   requires labels. Labels are delayed, expensive, or
   missing for large classes of production models. If
   your monitoring plan starts and ends with accuracy, you
   have no early-warning signal. Chapters 03–04 build the
   signals you can measure *without* labels.
2. **A single "drift score" for everything.** Aggregating
   drift across features loses the diagnostic value.
   Per-feature drift is what tells you *which* upstream
   thing changed. Chapter 03 belabours this point.
3. **A drift alert with no owner.** Fires at 03:14. Oncall
   pages the platform team. Platform team pages the ML
   team. ML team is not on rotation. Everybody wakes up
   and nobody knows what to do. Chapter 06 and chapter 07
   put an explicit human decision rule behind every alert.
4. **Retraining as the reflex response to every drift
   alert.** Retraining on a drifted-and-unlabelled window
   can make the model worse. Chapter 06 covers the
   guardrails.
5. **No reference window.** Drift is defined *relative to
   something*. If you did not save what "normal" looked
   like at training time, you have no baseline. Chapter 02
   makes the reference profile a required artifact.
6. **Alerting on statistically significant differences.**
   With a large enough traffic stream, everything is
   statistically significant. Alert on effect size and
   calibrated thresholds, not p-values. Chapter 03.
7. **Monitoring the model, not the pipeline.** A
   regression usually starts upstream — a data source
   changed, a feature pipeline had a bug, an ETL job
   failed. If your monitoring only watches the model's
   output, you diagnose the symptom and miss the cause.
   Chapter 07 walks a full drill-down.

## Summary

- Generic observability catches "the service is broken."
  ML monitoring catches "the service is fine but the
  model is wrong." They are complementary, not
  substitutes.
- Fix vocabulary early: **data drift** (`P(X)`), **concept
  drift** (`P(y|X)`), and **prediction-distribution
  shift** (`P(ŷ)`). They have different measurements and
  different remedies.
- Ground truth is delayed or absent for most production
  models. Any monitoring plan that assumes fresh labels
  is a plan for a dashboard nobody looks at.
- The ML engineer is uniquely positioned to name the
  invariants and to act on the signal. It cannot be
  delegated to platform.
- The module builds: an ML-metrics surface, a drift
  pipeline, a ground-truth loop, retraining triggers, and
  a rehearsed debugging story.

## Where this goes next

Chapter 02 opens the instrumentation half of the module:
**what to actually log and count** on top of a
`mod-108`-shaped service, so that everything from chapter
03 onward has real numbers to consume. Prediction
histograms, feature summaries, per-version latency, and a
reference profile you snapshot at training time.
