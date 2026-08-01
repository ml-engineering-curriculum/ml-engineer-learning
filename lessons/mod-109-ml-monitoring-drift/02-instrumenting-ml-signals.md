# 02 — Instrumenting ML-specific signals

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 01, plus `mod-108` chapter 05
> (structured logging) and chapter 09 (Prometheus metrics).

## Motivation

Chapter 01 named the gap: Prometheus knows the request
rate, but not what the model believes. This chapter closes
that gap in the smallest useful way — a handful of extra
metrics, a handful of extra log fields, and one artifact
saved at training time. Everything from chapter 03 onward
consumes what you build here.

Two design constraints shape the whole surface:

- **You will not always have ground truth.** So the
  instrumentation has to be useful *without* labels. The
  primary signals are input distributions, prediction
  distributions, and latency-per-version — not accuracy.
- **You will always have model versions.** So every
  signal carries `model_version` (and ideally `git_sha`)
  as a label, so a rolling deploy or an A/B split does
  not silently muddy the numbers.

## The four surfaces

Four places the model leaves a trace, and one saved
artifact:

1. **Prometheus counters and histograms** — for anything
   you want to alert on or graph continuously. Low
   cardinality, aggregate signals.
2. **Structured log events** — for anything you might
   want to filter, join, or replay later. Higher
   cardinality, per-request detail. `mod-108` chapter 05
   is the pattern.
3. **A prediction table** in the data warehouse — one row
   per prediction (or a sampled subset), joined later
   with ground truth to compute delayed accuracy. This
   is what chapter 05 hangs off.
4. **A reference profile** — a snapshot of what "normal"
   looked like at training time: per-feature summaries,
   the training prediction distribution, class balance,
   a random sample. Written next to the model artifact
   in `mod-108`'s `metadata.json`; consumed by every
   drift check in chapter 03.

Everything else in this chapter is where to plumb each
surface.

## Metrics that pay for themselves

Start with these six. They cover the majority of "why is
the model behaving oddly?" questions and cost almost
nothing to emit.

### 1. Prediction rate per version

```python
from prometheus_client import Counter

PREDICTIONS = Counter(
    "ml_predictions_total",
    "Predictions served, labelled by model version and outcome bucket.",
    labelnames=("model_name", "model_version", "route", "outcome_bucket"),
)
```

`outcome_bucket` is discrete: for classification, the
predicted class; for regression, a coarse bucket (5–10
buckets is plenty). Cardinality stays bounded because you
control the bucket count. The counter answers "what is
the model *saying*?" over time.

### 2. Prediction distribution (histogram)

```python
from prometheus_client import Histogram

PREDICTION_SCORE = Histogram(
    "ml_prediction_score",
    "Predicted score (probability or regression output).",
    labelnames=("model_name", "model_version", "route"),
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
```

For a binary classifier this is a probability histogram;
for regression the buckets should span the plausible
output range. Chapter 03's prediction-shift detector reads
straight off this histogram — a mean or a quantile that
drifts is the earliest warning you get without ground
truth.

Two rules stop this histogram from misbehaving:

- **Pick buckets once and stop.** Prometheus histograms
  cannot be re-bucketed after emission; renaming buckets
  breaks every alert and dashboard.
- **Do not label with `user_id` or `request_id`.**
  Cardinality explodes; the platform team pages you.

### 3. Inference latency per version

```python
LATENCY = Histogram(
    "ml_inference_latency_seconds",
    "Model inference latency (does not include HTTP overhead).",
    labelnames=("model_name", "model_version", "route"),
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
```

Distinct from the service-level request latency `mod-108`
already exposes. This one isolates *model* time so a
change in a preprocessing step does not muddy the model
diagnosis. Also crucial during a canary rollout: comparing
`v1.4.2` to `v1.4.3` at the model level is unambiguous.

### 4. Feature summary histograms (bounded set)

For a handful of features you know matter, expose a
histogram per feature:

```python
FEATURE = Histogram(
    "ml_feature_value",
    "Live distribution of an input feature.",
    labelnames=("feature_name", "model_name", "model_version"),
    buckets=(...),  # per-feature; see notes below
)
```

This one is the trap: emitting a histogram per feature
per version costs cardinality. **Do not** expose
histograms for hundreds of features from Prometheus. The
pattern that works in practice:

- **Prometheus:** histograms for at most ~10 features that
  are the model's top drivers (SHAP top-K from the
  training run in `mod-106`, or the columns you already
  know are volatile).
- **Prediction table + reference profile:** all features,
  compared offline on a schedule. Chapter 03's drift job
  does the heavy lifting there.

Bucket choices are per-feature. A feature bounded to
`[0, 1]` gets 10 evenly-spaced buckets; a heavy-tailed
count feature gets log-spaced buckets (`1, 2, 5, 10, 20,
50, ...`). Copy the bucket edges from the reference
profile — you built them once at training time.

### 5. Null / missing / out-of-range rate

```python
FEATURE_ANOMALY = Counter(
    "ml_feature_anomaly_total",
    "Requests where a feature was null, out-of-range, or unknown-category.",
    labelnames=("feature_name", "kind", "model_name", "model_version"),
)
# kind ∈ {"null", "out_of_range", "unknown_category"}
```

The single most common root cause of a production
regression is an upstream data source that started sending
values the model never saw during training. The counter
catches:

- A newly-seen category (`unknown_category`) — one-hot
  encoding maps it to the "unknown" column or drops it
  silently.
- A numeric feature outside its training range
  (`out_of_range`) — imputation and clipping cover for
  the surprise.
- A `NULL` where the training set had `0.0`
  (`null`) — sometimes the same, often not.

Ranges and category sets come from the reference profile.
The counter is what fires the loudest, earliest alert when
`mod-103`'s upstream table changes shape.

### 6. Model load / warm-up gauges

```python
from prometheus_client import Gauge

MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "1 if a model version is currently loaded and warm.",
    labelnames=("model_name", "model_version"),
)
MODEL_LOAD_TIME = Histogram(
    "ml_model_load_seconds",
    "Time to load the model artifact from cold.",
    labelnames=("model_name", "model_version"),
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
)
```

Not strictly *model quality*, but critical for rollout
attribution. During a rolling deploy the gauge tells the
oncall which version is where; the histogram catches a
model that got much slower to load between versions.

Together, the six give you enough surface to catch most
regressions before ground truth arrives. Everything else
is polish.

## The prediction log — what to write per request

Metrics answer aggregate questions. Individual predictions
have to land somewhere durable so you can join them with
labels weeks later. `mod-108` chapter 05 already logs a
`prediction` event; extend it:

```json
{
  "ts": "2026-06-14T03:14:02.219Z",
  "event": "prediction",
  "request_id": "01HGX...",
  "model_name": "churn",
  "model_version": "1.4.2",
  "git_sha": "9f2b1e",
  "prediction_id": "pred_01HGX...",
  "features_hash": "sha256:...",
  "prediction": 0.87,
  "predicted_class": 1,
  "inference_ms": 34.1,
  "reference_profile_id": "rp_2026_06_01"
}
```

Two fields are new and worth attention:

- **`prediction_id`** — a stable identifier for this
  prediction. It becomes the join key when a label
  arrives later (chapter 05). Generate it in the handler
  and return it in the response so the caller can log it
  too.
- **`features_hash`** — a hash of the ordered feature
  vector. Cheap to compute; catches the case where two
  identical requests produce different predictions (a
  model swap between them). Optional but useful for
  incident replay.

A parallel prediction *table* — one row per prediction,
possibly sampled to 1–10% for volume — is what chapter 05
joins to. Some organisations write it from the service
directly (`INSERT INTO predictions ...` via an async
queue); most stream logs to a warehouse via the log
pipeline and materialise the table there. Either works;
pick the one your data platform already supports.

**Do not** log the raw feature values by default. PII lives
there. If you need the values for debugging, hash them,
sample a small slice offline, or write them to a separate
access-controlled table. The prediction log ships to
everyone with log-read access.

## The reference profile — the artifact "normal" is measured against

Every drift check needs a baseline. The baseline is not
"yesterday's traffic" (that drifts too) — it is a snapshot
of what the training set looked like, written at training
time. Alongside `mod-108`'s `metadata.json`, save a
`reference_profile.json` (or `.parquet`, or a whylogs
profile — the format is negotiable, the content is not):

```json
{
  "reference_profile_id": "rp_2026_06_01",
  "trained_at": "2026-06-01T00:00:00Z",
  "n_rows": 480123,
  "features": {
    "age": {
      "dtype": "int64",
      "nullable": false,
      "range": [18, 92],
      "quantiles": {"0.01": 21, "0.05": 24, "0.5": 41,
                    "0.95": 68, "0.99": 79},
      "mean": 42.6, "std": 14.2
    },
    "plan_tier": {
      "dtype": "category",
      "categories": {"basic": 0.62, "pro": 0.31,
                     "enterprise": 0.07}
    }
  },
  "predictions": {
    "mean": 0.128, "std": 0.194,
    "quantiles": {"0.5": 0.06, "0.95": 0.51, "0.99": 0.83},
    "class_balance": {"0": 0.87, "1": 0.13}
  },
  "sample": "s3://models/churn/1.4.2/reference_sample.parquet"
}
```

A few disciplines make this artifact carry its weight:

- **Include a sample, not just summary statistics.** A
  10k-row sample of the training data is what powers the
  multivariate drift checks in chapter 03 (domain
  classifier, MMD, Wasserstein). Summary stats alone
  cover univariate drift only.
- **Store a `reference_profile_id`.** Every prediction
  log line names the reference profile it was measured
  against. When you rotate profiles (a new training
  run), old alerts don't retroactively change their
  meaning.
- **Freeze the profile with the model version.** New
  version → new profile. Never mutate the profile of a
  deployed model; instead compute drift *between*
  profiles at rollout time (chapter 06).
- **Do not use the last N days of production as the
  reference.** Drift measured against a drifting
  reference reports zero drift right up until the model
  is unusable. The reference is training-time
  distribution, full stop.

The `reference_profile.json` is a `mod-108` deliverable
that this module hard-requires. If the model you inherited
doesn't have one, you build it from the training data
before instrumenting anything else.

## The dashboards worth building

Grafana panels are cheap; opinionated ones are useful.
Four panels per service will earn their space:

1. **Prediction rate by class, stacked, per version.** The
   fastest read on "is the model saying the same thing it
   was saying yesterday?"
2. **Prediction distribution as a heatmap.** Time on
   x-axis, prediction-score bucket on y-axis, colour by
   density. Multimodality, mode collapse, and slow drift
   are visible at a glance.
3. **Feature anomaly rate, top 10.** A stack of
   `ml_feature_anomaly_total` rates by feature. Any bar
   ticking upward is an upstream data change.
4. **p95 inference latency per version.** Distinct from
   request latency; catches a model that got slower
   without breaking correctness.

Add a fifth once ground truth flows (chapter 05):

5. **Delayed accuracy vs. the last N versions.** One line
   per version; accuracy on a rolling window of
   ground-truth-joined predictions.

Everything else is a variant.

## Alert wiring — the minimum useful set

Emitting metrics without alerts is journaling. Three
alerts justify the surface you just built:

- **Feature anomaly rate spike.** `rate(ml_feature_
  anomaly_total[5m])` for any feature crosses a threshold
  set from the reference profile's baseline. Fires within
  minutes; correlates almost 1:1 with upstream breakage.
- **Prediction distribution shift.** A daily or hourly
  job (chapter 03) computes drift on the prediction
  histogram vs. reference and fires when the drift score
  crosses the calibrated threshold. Slower to fire; harder
  to ignore.
- **Model-quality regression.** A ground-truth-joined
  accuracy metric drops by more than a threshold vs. the
  last N-day mean. Slowest to fire (labels are delayed);
  the one that everyone actually cares about.

Chapter 06 makes each of these an input to the retraining
decision.

## Cardinality — the recurring pitfall

The single reason platform teams turn off ML metrics is
cardinality. Every label you add multiplies series count.
Two heuristics keep you out of trouble:

- **Label values should be small enumerable sets.**
  `model_version` (dozens at most), `route`, `class`,
  `feature_name` — fine. `user_id`, `request_id`,
  `session_id`, `image_url` — no. Ever.
- **Prefer log fields to metric labels for anything
  high-cardinality.** Logs handle high cardinality; the
  metrics system does not. If you want per-user
  behaviour, join the log table to the user table
  offline.

If a metric is exploding cardinality, the platform team
sees it in Prometheus's own metrics
(`prometheus_tsdb_head_series`) and will politely ask you
to fix it. Do not force them to be impolite.

## Sampling — when to do it and when not to

Metrics are already sampled implicitly (buckets, rates).
Log volumes and warehouse table sizes can be too big to
keep every prediction row. Some rules of thumb:

- **Metrics: never sample.** Emit every request. The
  counters and histograms are what feed alerts; sampling
  makes rate calculations noisy at low traffic.
- **Prediction log lines: rarely sample.** They are the
  source of truth for later joins. Sample only when
  volume is prohibitive; when you do, keep a stratified
  sample so rare classes remain represented.
- **Prediction table for offline joins: sample
  aggressively if needed.** 1–10% is often enough for
  drift measurement. Keep 100% for regulated domains
  where every prediction has to be auditable.
- **Reference sample: fixed, not streamed.** Sampled once
  from the training set; frozen.

Document the sampling rate. A drift calculation on a
sampled stream needs to know the rate to compute a
confidence interval.

## Common pitfalls

1. **Adding `model_version` to logs but not metrics.**
   Rollouts become impossible to attribute; you cannot
   tell whether the p95 spike is v1.4.2 or v1.4.3.
2. **Emitting a feature histogram per feature per
   version.** With 200 features and 5 concurrent versions,
   1,000 histograms explode into millions of series
   through bucket cardinality. Bounded feature set only.
3. **No reference profile.** Chapter 03 has nothing to
   compare to. Drift is undefined.
4. **Reference profile written by a script but never
   versioned.** When the script changes, the semantics of
   the "no drift" state changes silently. Freeze it with
   the model.
5. **Logging raw feature values by default.** PII in logs.
6. **Prediction histogram buckets that saturate at the
   tail.** A binary classifier whose top bucket is
   `[0.9, 1.0]` cannot distinguish "0.91" from "0.99" —
   fine drift in the extreme tail is invisible. Extra
   buckets at the extremes cost nothing.
7. **No warm-up prediction on startup.** The first live
   prediction pays the cache-miss cost; the p95 histogram
   spikes; you page yourself.
8. **Instrumenting the service without instrumenting the
   batch path.** `mod-108` chapter 08 exists too. Batch
   predictions should populate the same prediction table
   the online path writes to; drift monitoring on batch
   traffic is exactly the same code.

## Summary

- Four surfaces: Prometheus metrics, structured log
  events, a prediction table, a reference profile. Each
  answers a different question.
- Six metrics carry most of the load: prediction rate,
  prediction distribution, inference latency, bounded
  feature histograms, feature-anomaly counters, and
  model-loaded gauges. All labelled by `model_version`.
- A prediction log line per request (or per sampled
  request), with a stable `prediction_id`, is the input
  to the ground-truth join in chapter 05.
- The **reference profile** is the artifact everything
  drift-related reads from. Freeze it with the model
  version, keep a sample alongside the summary stats.
- Cardinality discipline is non-negotiable. Small
  enumerable label sets, and no `user_id` or
  `request_id` ever.

## Where this goes next

Chapter 03 turns the raw signal into a decision. The
per-feature histograms, the prediction distribution, and
the reference sample from this chapter feed a
drift-detection job that runs on a schedule, emits a
score per feature and per prediction, and knows when to
alert. The chapter also picks the specific tests — PSI,
KS, chi-squared, Wasserstein, and the domain-classifier
trick for multivariate drift — and shows how to
calibrate their thresholds so the alert channel is not
just noise.
