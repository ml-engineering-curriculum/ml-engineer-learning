# 05 — Ground-truth collection loops

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 01–04.

## Motivation

Chapter 04's concept-drift detector, chapter 06's
retraining trigger, and every accuracy metric on the
dashboard are downstream of one plumbing question: **how
do labels get from the world back into a table joined to
predictions?**

The answer is not obvious. For a fraud model, the label
is "this transaction was disputed 30 days later." For a
recommendation model, the label is "the user clicked (or
did not click)." For a churn model, the label is "the
account was cancelled seven weeks later." For a medical
imaging model, the label is "a specialist reviewed the
scan and disagreed." Each is a different pipeline, a
different delay, a different failure mode.

This chapter is the shape of that pipeline. Where labels
come from; how they get joined to predictions; the
failure modes that turn a "training loop" into "labels
that never arrive" or, worse, "labels that arrive
biased." The output is a warehouse table (or a stream)
that every downstream module reads from.

## The three sources of labels

Every production label ultimately comes from one of
three places. Systems that mix them combine the failure
modes of each; know which one is which.

### 1. Implicit user feedback

The user's next action *is* the label. A search click, a
recommendation dismissal, a chat "helpful/not helpful"
button, a completed purchase, a return.

- **Cheap and fast.** No human labelers, no explicit
  request. Latency is minutes to hours.
- **Biased by exposure.** You only see feedback for
  items you actually showed. This is the classic
  bandit-feedback / logged-bandit problem: your labels
  underrepresent the "roads not taken." Ignoring the
  bias is one of the most common reasons a
  recommendation model retrains itself into a corner.
- **Sparse for the negatives.** For recommendation, a
  "not clicked" is not the same as "not interested" —
  the user might not have seen it. Reasoning about
  negatives requires impression logs, not just
  click logs.
- **Available even without explicit feedback UI.** The
  telemetry pipeline you already run for product
  analytics is often 80% of what you need.

Implicit feedback dominates web-scale ML systems. The
correction for exposure bias — inverse propensity
scoring, doubly robust estimation, counterfactual
learning — is a whole literature. Even without a full
correction, tracking and reporting the exposure
distribution alongside the labels is enough to keep the
next reviewer honest.

### 2. Explicit business outcomes

The label is a downstream business event with a hard
definition. A chargeback for fraud. A default for
credit. A cancellation for churn. A refund for
recommendation quality. A medical outcome recorded in
the electronic health record.

- **Slow but authoritative.** Delays of days to months.
- **Well-defined.** A dispute has an ID and a date; a
  cancellation has a timestamp. The label schema is
  clean.
- **Owned by another team.** The chargebacks table
  lives in finance's warehouse; the cancellations table
  in growth's. You depend on their SLAs.
- **Prone to label leakage in retraining.** A feature
  that is easy to compute *now* (post-hoc) may not have
  been available at prediction time. Chapter 06 covers
  the guardrail; here the point is that when you
  connect the labels table, you also connect the
  potential for leakage.

Explicit outcomes are what your accuracy metric is
denominated in. They set the cadence of the retraining
loop — labels arrive weeks later, so retraining cannot
be daily.

### 3. Human labeling

A person looks at the input (and sometimes at the
prediction) and produces the label. Radiologists reviewing
scans; a moderation team reviewing flagged content;
domain experts labelling a random sample of predictions;
crowdworkers on Mechanical Turk / Scale / Labelbox.

- **Slow and expensive.** Per-label costs of cents to
  dollars; delays of hours to days.
- **Consistent-quality is work.** Inter-annotator
  agreement, gold-standard tasks, adjudication of
  disagreements. The infrastructure around the
  labelers matters more than the labelers.
- **The only option for some problems.** Content
  moderation, image quality, medical review — no
  implicit signal exists.
- **A sampling knob.** You do not have to label
  everything; you label a stratified sample, or the
  predictions the model was least sure about (active
  learning), or predictions the drift detectors
  flagged.

For any problem where the labels are not free, the
labeling budget is a design constraint. Under-sample and
your accuracy metric has wide error bars; over-sample
and you burn money. Stratified sampling — over-sample
rare classes and high-drift windows, under-sample the
easy majority — buys the most information per dollar.

## The join key — the whole loop hinges on this

Every architecture in this chapter comes down to one
question: **how do you join a prediction row to a label
row weeks later?**

The join key is what chapter 02's `prediction_id` was
for. Rules that keep the join reliable:

- **Generate the ID at prediction time.** Server-side,
  so the caller cannot forget. UUIDv4 or ULID or
  ULID-with-timestamp — anything with negligible
  collision probability.
- **Return it in the response.** The caller logs it in
  their own telemetry; the downstream system stamps it
  on the resulting business event.
- **Store it on the business event.** The chargeback,
  the click, the moderation decision — each carries the
  `prediction_id` of the prediction that was shown.
  When it doesn't, you're joining on time-and-user,
  which loses precision.
- **Do not reuse `request_id` or `user_id`.** A user
  makes many requests; a request produces many
  predictions in batch endpoints. `prediction_id` is
  1:1 with a prediction.

When the join key is missing (a legacy system where
`prediction_id` was never plumbed through), the fallback
is a "best-effort" join on `(user_id, timestamp)` with a
tolerance window. It works; it is lossy; it is a
tech-debt entry against the next release.

## Where the labels table lives

The output of all this is a table:

```
predictions_labeled
-------------------
prediction_id       TEXT PRIMARY KEY
model_name          TEXT
model_version       TEXT
predicted_at        TIMESTAMP
prediction          FLOAT
predicted_class     INT
prediction_score    JSON       -- top-k, raw score, whatever the model returns
features_hash       TEXT
label               ...        -- the ground-truth value(s)
label_source        TEXT       -- 'user_click' | 'chargeback' | 'reviewer_1' | ...
label_at            TIMESTAMP
label_latency_hours FLOAT      -- label_at − predicted_at
weight              FLOAT      -- for stratified sampling
slice_metadata      JSON       -- country, cohort, tier — for slice metrics
```

Two decisions worth making explicitly:

- **Wide vs. narrow.** Chapter 02's prediction log
  already carries prediction metadata; the labels
  table can either duplicate it (wide, easy to query)
  or reference it by ID (narrow, one source of truth).
  Wide wins for the monitoring use case; storage is
  cheap and consumers do not have to remember to join.
- **Where it lives.** In the same warehouse as your
  other business data (BigQuery, Snowflake, Redshift,
  Postgres). Chapter 06's retraining trigger and
  chapter 04's accuracy dashboard both consume this
  table; putting it anywhere else makes them awkward.

The table is the *system of record* for the model's
performance. Its schema evolves with the model; its rows
are immutable once written; every downstream analysis is
one query away.

## Pipeline shapes

Three shapes cover most real systems.

### Shape A — implicit feedback, streaming

The classic recommendation / search loop:

```
serving service ─ writes ─> prediction stream (Kafka/PubSub)
     │                              │
     │                              └─> warehouse: predictions table
     ▼
   product
     │
     └─ user acts ─> event stream
                              │
                              └─> warehouse: events table

nightly (or streaming) job:
    predictions ⨝ events on prediction_id
      ─> predictions_labeled
```

The labels join in near-real-time. The event schema is
owned by product analytics; you consume it. The most
common failure mode is **schema evolution on the event
side** — product renames `click` to `interaction`, your
job silently produces zero labels for a week. Chapter 06
mentions the fix (schema tests on the labels table).

### Shape B — delayed business outcomes, batch

The fraud / credit / churn loop:

```
serving service ─ writes ─> predictions table (batched to warehouse)

business system ─ writes ─> chargebacks table
                            (30-day dispute window)

weekly job:
    predictions[predicted_at ≤ now − 30d]
    ⨝ chargebacks on prediction_id
    ─> predictions_labeled[cohort=<week>]
```

Two twists:

- **The maturity window matters.** A prediction whose
  label window has not yet closed can't have a
  "no chargeback" label — it might get one tomorrow.
  Filter the predictions to `predicted_at ≤ now − W`
  where `W` is the outcome window.
- **The labels table has a "cohort" dimension.**
  Each week's cohort is a distinct denominator; you
  report accuracy by cohort so a slow-arriving late
  cohort doesn't destabilise the current-week metric.

### Shape C — human labeling, sampled

The moderation / medical / rare-class loop:

```
serving service ─ writes ─> predictions table

sampling job (daily):
    stratified sample of predictions[predicted_at ≤ now − 24h]
    ─> labeling queue (Scale / Labelbox / internal tool)

labelers ─ produce ─> labels
                           │
                           └─> warehouse: human_labels table

nightly:
    sampled predictions ⨝ human_labels on prediction_id
    ─> predictions_labeled[sampled=True, weight=1/p]
```

The `weight` column is the inverse sampling probability.
When you compute accuracy from a stratified sample, the
per-row weight is what makes the estimate unbiased with
respect to the source population.

## Stratified sampling — the labeling budget's best friend

Random sampling wastes labels on easy majority-class
predictions. Stratified sampling by prediction bucket
puts labels where they inform the metric most:

- Over-sample the tails of the prediction distribution
  (`prediction ≤ 0.05` and `prediction ≥ 0.95`) — those
  are the confident predictions; a labeler disagreeing
  is high-signal.
- Over-sample the uncertain middle (`0.4 ≤ prediction ≤
  0.6`) — those are the decisions the model is closest
  to being wrong on.
- Over-sample slices that chapter 04 flagged as
  drifted or important — new cohort, new geography,
  high-value customers.
- Under-sample the "confidently normal" bulk.

Per-slice weights come out as `weight = 1 / sampling_
probability`. Downstream metrics use the weights.

For active learning (using the labels to *train*, not
just to measure), the sampling policy gets more
sophisticated — uncertainty sampling, expected model
change, query-by-committee. The measurement use case
here is simpler; a stratified sample is enough.

## The failure modes

Ground-truth loops fail in specific and recoverable
ways. Watch for each.

### Selection bias

The labelled subset is not representative of the
production distribution. Classic examples:

- Only labelling predictions above the alert threshold
  (you never observe the false negatives).
- Only labelling flagged content (you never see the
  content that got through and shouldn't have).
- Only asking users about the recommendations they
  interacted with.

Whenever the labelling gate correlates with the model's
output, the estimated accuracy is biased. The fix is
either (a) label the raw random sample so the estimate
is unbiased, or (b) explicitly weight for the bias
(propensity scoring). Some bias is inevitable; make it
explicit in the metric definition.

### Delay bias

The maturity window is not fully closed for the most
recent cohort. Reporting "current-week accuracy" using
labels that only accumulate over 30 days will
underestimate the negative class (fewer chargebacks have
arrived than will). The fix is chapter 06's cohort
methodology — only report a cohort's accuracy once its
maturity window has closed.

### Concept drift in the labelers

Human labelers change too. A new labelling guideline;
a new team; new adjudication rules. The label-generating
process drifts, and your model looks like it regressed
when it didn't. Fix: version the labelling guidelines;
label a small "gold" sample repeatedly to detect
labeler drift; store the guideline version alongside
the label.

### Missing negatives

For implicit feedback, "not clicked" is not "not
interested" — the user might not have seen it. Impressions
have to be logged alongside clicks; the label is
`clicked | impressed`. Without impressions, positive
rates look artificially high (you count all clicks; you
undercount the impressions denominator).

### Label leakage into retraining

The label is easy to compute after the fact from a
feature that is not available at prediction time. Chapter
06 makes this a guardrail; here, name it and prevent it
at the labels table by not carrying post-hoc-only fields.

### Ghost predictions

A prediction that was made but never shown to the user
(the fallback path took over, an experiment excluded
the user, the request timed out). If the labels-join
doesn't distinguish "shown" from "not shown," the
metrics conflate the two. Log an "outcome" for each
prediction (`shown`, `fallback`, `error`) and filter on
it.

## Latency and cadence — what's realistic

Rough cadences by source type:

| Source                | Latency         | Cadence of the join |
|-----------------------|-----------------|----------------------|
| Streaming impl. feedback | seconds       | streaming or hourly |
| Batch impl. feedback   | hours           | hourly / daily       |
| Business outcomes      | days–weeks     | weekly / monthly     |
| Human labelling (fast) | hours          | daily                |
| Human labelling (slow) | days           | weekly               |

Match your monitoring cadence to your labelling
cadence. A dashboard that updates every minute using a
labels source that fills weekly is a lot of empty
minutes.

## The consumer contract

Once `predictions_labeled` exists, everything downstream
reads from it. Nail the schema; version it; treat it
like an API.

- **`prediction_id` is the primary key.** Enforced.
- **`label` is nullable** for predictions whose
  maturity window has not closed. Consumers filter it
  out.
- **`label_source` is enumerable.** Downstream queries
  can slice by source.
- **`weight` is always populated.** Default `1.0` for
  unsampled rows.
- **Late-arriving labels are UPSERTs**, not INSERTs.
  A prediction can be re-labeled if the source
  updates.

Once the contract stabilises, chapter 04's accuracy
dashboards, chapter 06's retraining trigger, chapter 07's
debugging queries, and any post-hoc analysis all live off
this table.

## Common pitfalls

1. **No `prediction_id`.** Joins collapse to
   best-effort; accuracy metrics are noisy; debugging
   is guesswork.
2. **Labels table without cohort.** Current metrics
   are contaminated by the immature cohort; the trend
   line looks worse than reality.
3. **Random sampling for a heavily-imbalanced label.**
   99.5% of the labels are "not fraud"; your labeling
   budget produces almost no positive examples;
   accuracy metrics are useless. Stratify.
4. **Trusting labeler drift is zero.** Guidelines
   change; adjudication changes; the label
   distribution drifts even when the world doesn't.
   Gold-set replay is the cheap check.
5. **Joining on `(user_id, minute)`.** Reasonable
   fallback; not a strategy. Get `prediction_id`
   plumbed end-to-end.
6. **Missing impression logs.** For implicit feedback,
   "not clicked" without "shown to the user" is
   incomparable. Log impressions.
7. **Post-hoc features in the labels table.** They
   leak into retraining. Only carry fields that were
   available at prediction time (or explicitly labelled
   as `post_hoc=True`).
8. **Labels table lives in a different system than
   predictions.** Every join is a cross-system
   nightmare. Same warehouse.
9. **Not surfacing "no label" as an outcome.** A
   prediction that never got a label (shown but
   ignored, or shown but the user's session was
   terminated) is data too. Log it.
10. **Assuming the labels team owns label quality.**
    They don't — you do. They own the labelling
    process; you own the definition, the guardrails,
    and the audit.

## Summary

- Labels come from three sources: implicit user
  feedback, explicit business outcomes, and human
  labelling. Real systems combine all three, and each
  carries a distinct failure mode.
- Every join hinges on a stable `prediction_id`
  generated at prediction time and propagated to
  labels. Everything else in this chapter is downstream
  of that discipline.
- The `predictions_labeled` table is the system of
  record. Wide, versioned, cohort-aware, weight-aware,
  same warehouse as the business data.
- Stratified sampling makes a small labelling budget
  buy the most information about the metric.
- Selection bias, delay bias, labeler drift, missing
  negatives, and label leakage are the specific,
  named failure modes. Each has a specific mitigation.

## Where this goes next

Chapter 06 turns everything you have measured so far —
data drift, prediction shift, delayed accuracy from
this chapter's labels table — into a **decision rule**.
When does a signal justify retraining? What are the
guardrails that stop a bad retrain from making things
worse? And what does the retraining pipeline itself
look like when it is triggered automatically vs. by a
human review?
