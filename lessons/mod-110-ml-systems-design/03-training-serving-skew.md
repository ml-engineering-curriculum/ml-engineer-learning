# 03 — Training-serving skew

> **Reading time:** ~20 minutes.
> **Prerequisite chapter:** 02.

## Motivation

You retrain the churn model. Offline AUC on the held-out
test set is 0.86 — a real improvement. The model gets
promoted. A/B ramps to 10% traffic. Two days in, the
retention team notices that the intervention list looks
strange: half the top-scored users are ones the team
would never have flagged. You dig in and find that
`support_contacts_30d` — the model's second-most-important
feature — is 0 for 40% of production requests. In the
training data it was 0 for 3% of rows. Same feature name,
same schema, two different distributions.

That is **training-serving skew**: the model saw one
distribution of features during training and a different
one at inference time, without anything breaking loudly.
Metrics stay green. Latency is normal. The prediction
distribution shifts. Users get worse outcomes. Somebody
notices weeks later.

Skew is the most common cause of "the offline model was
great, the online model is not" incidents. Every senior
ML engineer has a story about it. This chapter is the
catalogue of causes, the pattern to prevent each, and the
instrumentation to catch what leaks through.

## What "skew" actually means

Formally, training-serving skew is any systematic
difference between the feature values a model sees during
training and the feature values it sees during inference,
holding the underlying data-generating process fixed.

Distinguish it from adjacent concepts:

- **Data drift** (`mod-109` chapter 03): the *underlying
  distribution* of inputs has shifted between when the
  training data was collected and now. Different problem,
  different fix. Drift is a change in the world; skew is
  a bug in your code.
- **Concept drift** (`mod-109` chapter 04): the
  relationship between `X` and `y` has changed. Also
  different from skew.
- **Label leakage** (`mod-107` chapter 06): a feature
  encodes information from the future or from the label.
  Same code path in training and serving, but the
  training path illegally sees the label. Skew is a
  divergence *between* paths; leakage is a bug in the
  training path.

You can have all four at once. In practice, when the
online model underperforms and offline metrics were
good, skew is the first hypothesis. Chapter 09's
walkthrough on debugging (in `mod-109` chapter 07)
handles the other three; this chapter handles skew.

## The four canonical causes

Every skew incident is one of these four. Learn the shape
of each; the mitigation follows.

### 1. Two implementations of the same transform

The feature pipeline was written twice: once in the
training environment (Python + pandas) and once in the
serving path (Python + FastAPI, or Go, or Java). The two
implementations *almost* agree. A rounding difference in
timezone handling; a slightly different tokenizer; a
different default when a value is missing; a different
order of one-hot columns after `sort()` vs `unique()`.
Offline reports 0.86; online is quietly using a version
that predicts a different thing entirely.

Signs you have this:

- The training pipeline is in a notebook or a Python
  ETL; the serving path reimplements the transforms in
  its own language / framework.
- Nobody can point to a *single* source of truth for the
  transform.
- The feature vectors at training time and at serving
  time have never been compared row-for-row on the same
  inputs.

Pattern to prevent:

- **Write the transform once**, in a shared library,
  and import it from both the training pipeline and the
  serving path. If the serving path is in a different
  language, the transform is behind a service or is
  translated once and locked to the training version.
- **Use a feature store.** Feast, Tecton, Databricks
  Feature Store, and their equivalents exist to solve
  this specific bug. A feature is defined in one place;
  both paths read via the same store's client.
- **Test row-parity.** A test fixture that runs a
  known input through both the training-time and
  serving-time paths and asserts identical output — to
  the byte, not "close enough."

### 2. The time-window feature computed differently

The training data contains `purchases_last_30d` — the
count of purchases in the 30 days before the label date.
The serving path computes `purchases_last_30d` at request
time by counting the last 30 days of activity *up to
now*. Fine when the request time and label time are the
same reference frame; a subtle bug otherwise.

The specific failure modes:

- **Point-in-time incorrectness.** Training's window
  ends at `label_date - horizon`; serving's window ends
  at `now`. If the two are miscalibrated (training used
  `<=`; serving uses `<`, or vice versa; training used
  UTC midnight; serving used local-time midnight), the
  window boundaries diverge by hours or days.
- **Backfill vs streaming aggregation.** Training's
  count was computed by a batch join over the whole
  history; serving's count was accumulated from a
  stream. Late-arriving events count in one and not
  the other.
- **Feature "freshness."** Serving reads yesterday's
  value from an online store that only refreshes
  nightly. Training saw same-day values.

Pattern to prevent:

- **Feature stores with point-in-time-correct joins.**
  The offline read query specifies "as of
  `label_date`," and the online read implicitly uses
  "as of `now`" — but the *definition* of the feature
  is one query. `mod-103` chapter on temporal joins
  is the deep dive.
- **Backfill parity tests.** For any feature that
  depends on a time window, backfill the online value
  for a specific historical date and compare it to
  what the training pipeline would have produced for
  that date. Store as fixture.
- **Explicit freshness SLOs.** For every feature,
  document the maximum staleness of the online value.
  Chapter 02 of `mod-109` monitors this.

### 3. Preprocessing that is fit in training and not
   applied at serving

The training pipeline runs `StandardScaler.fit(X_train)`
and then `.transform(X_train)`. The serving path
instantiates a fresh `StandardScaler()` on the request
payload and calls `.transform(X_request)` — with
whatever the default `mean=0, var=1` behaviour is. The
model was trained on standardised features; it's now
seeing raw features. Predictions are catastrophic;
sometimes the API still returns 200.

More subtle variants:

- **Categorical encodings** where the training-time
  category-to-index map is different from the serving-
  time map. New categories at serving map to
  `unknown_category`; old categories map to shifted
  indices because the map was re-fit on a different
  sample.
- **Target-encoded features** where the mean-per-
  category was computed on training data and needs to
  be shipped alongside the model — but wasn't.
- **Embedding tables** where the vocab was pruned
  between training and serving.

Pattern to prevent:

- **`sklearn.Pipeline`** (or the equivalent in your
  framework) — the fitted preprocessor is part of the
  artifact. Loading the artifact loads the fitted
  transforms. `mod-102` chapter on sklearn pipeline
  design is the primary reference.
- **One artifact contract.** The serialized artifact
  includes every fitted transformer, the fitted
  model, and any fitted encoders. If any of them
  lives in a separate file, they get versioned
  together and loaded together.
- **A smoke test after model load.** Load the
  artifact; call `.predict` on the fixture from the
  training test set; assert the output matches the
  offline record within tolerance. Chapter 04 in
  `mod-108` puts this on the readiness probe.

### 4. Feature source drift

The training data pulled `plan_tier` from the
`subscriptions` warehouse table. The serving path pulls
`plan_tier` from the `users` warehouse table. Both look
right in isolation; the two tables are populated by
different ETLs and drift out of sync at the day-of-
launch of a new plan (see `mod-109` chapter 07's
walkthrough).

Or: training pulled from a snapshot dated
`2026-05-01`; serving pulls live; a schema change
between the two dates changed the meaning of the
column.

Pattern to prevent:

- **A canonical source.** For every feature, the
  registry entry names the source table (or stream)
  and the schema version. Training pulls from that
  source; serving pulls from that source; nothing else
  is allowed.
- **Schema pinning.** The training pipeline records the
  schema version of every source it read; the
  monitoring surface alerts if the serving path is
  reading a different one.
- **Change subscriptions.** If the source table is
  owned by data engineering, subscribe to its schema-
  change notifications. When a new column appears or a
  column type changes, the ML team is on the notice.

## The detection surface

Prevention beats detection, but detection is the
backstop. Three signals to instrument.

### Signal 1: Row parity on a canonical fixture

In the training repo, keep a small canonical fixture:
20-50 rows of realistic input data (with labels for
sanity, if you have them). In CI, on every model
promotion, run the fixture through both:

- The training-time transform pipeline (as saved with
  the artifact).
- The serving-time transform pipeline (as it will run
  in production).

Assert byte-identical (or, for floats, within a strict
tolerance like `1e-9`) feature vectors. If they diverge,
the promotion fails. This is the single highest-value
skew defence in the entire chapter.

```python
def test_serving_transform_matches_training():
    fixture = load_json("tests/fixtures/serving_skew_fixture.json")
    training_pipeline = load_artifact("artifact/pipeline.joblib")
    serving_pipeline = load_serving_pipeline()  # what production loads

    for row in fixture["rows"]:
        x_train = training_pipeline.transform_only(row)
        x_serve = serving_pipeline.transform_only(row)
        assert np.allclose(x_train, x_serve, atol=1e-9), (
            f"Skew detected on row {row['id']}: "
            f"training={x_train} serving={x_serve}"
        )
```

Notes on getting this right:

- **Load the artifact and the serving code from
  production sources.** Not the notebook the data
  scientist trained in.
- **Cover the edge cases.** Fixture includes rows with
  nulls, unusual categories, and boundary values —
  the places skew hides.
- **Fail loudly.** The test failure blocks promotion.
  A warning in a log gets ignored.

### Signal 2: Prediction parity on shadow traffic

Before ramping the new model to any real traffic, run
it in **shadow mode**: for every real request, execute
both models (the currently serving one and the
candidate) and log both predictions. Nothing changes
downstream; the candidate is a spectator.

Two comparisons to make:

- **Prediction distribution.** Aggregate over an hour
  of traffic. Are the distributions statistically
  similar (KS test, JSD)? Wildly different distributions
  are almost always skew, not a legitimate model
  improvement.
- **Feature parity per request.** For a sampled subset,
  log the feature vector each model actually saw. If
  the two vectors differ for the same request, one of
  the two paths is wrong. Usually the candidate; not
  always.

`mod-108` chapter on shadow deployments and `mod-109`
chapter on canary rollouts are the operational
mechanics. This chapter is why they exist.

### Signal 3: Feature-anomaly counters

The `mod-109` chapter 02 counter — for every request,
compare each feature to the reference profile from
training time and count how many are null, out of
range, or unknown-category. In healthy operation the
counter is close to zero. A skew bug lights the
counter up on the specific features that diverged.

The counter runs continuously in production; the
row-parity test runs at promotion time. The pair is
the belt and the braces.

## The skew audit — a checklist to run at design time

Before you promote a new feature or a new model
version, walk this list explicitly. Written in the
first person because that is how the ritual gets
adopted.

- [ ] For every feature, is there exactly one code
      path that computes it?
- [ ] For every fitted transformer, is it part of
      the artifact and loaded together with the
      model?
- [ ] Do the training-time and serving-time paths
      import the transform from the same library, or
      does one of them reimplement it?
- [ ] For every time-window feature, is the point-in-
      time semantics documented and identical between
      training and serving?
- [ ] For every categorical feature, is the category
      vocabulary shipped with the artifact and used at
      serving?
- [ ] For every feature, is the source table (or
      stream) named in the registry entry?
- [ ] Does the CI suite include a row-parity test on a
      fixture that covers null, edge, and boundary
      cases?
- [ ] Is a shadow-mode rollout planned before the
      first real traffic hits the new version?
- [ ] Is `ml_feature_anomaly_total` alerting on this
      model in production?

If the answer to any is "no" or "I'm not sure," you
have work to do before the promotion.

## Two war stories in miniature

Neither is invented; both are anonymised composites of
common real incidents.

### The trailing-space tokenizer

A search-ranking model at a mid-size e-commerce site.
Training used `nltk.word_tokenize` on product titles;
serving used a hand-rolled tokenizer in Go for
latency. On strings without trailing whitespace the two
agreed; on strings with trailing whitespace they
disagreed by one token. Roughly 12% of product titles
in the catalogue had trailing whitespace due to an
upstream ETL bug. The model was quietly serving worse
predictions for that 12% for four months. The offline
metric was fine; the online metric was fine on
aggregate; a business-metric drilldown by product
segment eventually surfaced it.

Cause: two implementations (cause #1). Fix: the Go
service now calls a `sentencepiece` model shipped in
the artifact — one implementation, both paths.

### The overnight refresh that lied

A propensity model at a subscription business. Training
used `activity_last_7d` computed from the events
warehouse via a nightly batch. Serving read
`activity_last_7d` from the online feature store,
which was refreshed nightly by a *different* job that
had been quietly failing for eleven days. Serving was
using an 11-day-stale value. Training used a fresh
value.

Cause: feature-source drift (cause #4) plus a broken
freshness monitor. Fix: the online-store refresh job
now writes a `updated_at` column that both the serving
path and the freshness monitor read; predictions
served with stale features return a warning header;
the online store's freshness SLO is now on the
alerting surface.

The point of the war stories is not the specific bugs.
It is that in both cases the offline model was fine,
the online model was silently wrong, and the incident
took *months* to detect. Skew is patient.

## Common pitfalls

1. **"Our features are simple, we can't have skew."**
   Simple features have skew too. Trailing whitespace,
   integer vs float, UTC vs local time.
2. **"We use a feature store, so no skew."** The
   feature store rules out one class of skew; it does
   not rule out custom transforms in the serving path,
   fitted preprocessors that were forgotten, or source
   drift.
3. **"We tested it end-to-end."** End-to-end tests
   check that the service returns 200. They rarely
   check that the feature vector matches training.
4. **A shadow rollout that only compares aggregate
   distributions.** Two very different models can
   produce similar aggregate distributions. Per-
   request feature parity is the stronger check.
5. **Fixture-parity tests that use a synthetic fixture
   with no edge cases.** The bug is in the edge case.
6. **Not versioning the featurizer with the model.**
   The featurizer is upgraded in isolation; every
   in-flight model version is now serving a different
   thing than it was trained on.
7. **Trusting an OpenAPI schema to prevent skew.** The
   schema says the field is `float`. Training got 0.5;
   serving is sending 5000. Both are floats.
8. **A "one-off ETL fix" that doesn't backfill the
   training data.** The bug is fixed for future
   predictions and still present in the training set;
   the next retrain re-embeds it.
9. **Serving-time features computed lazily and cached
   with a wrong TTL.** The value is stale in
   unpredictable ways and every request looks fine.
10. **No `model_version` label on the metrics that
    fire.** During a rollout you can't tell which
    version is producing the anomalies.

## Summary

- Training-serving skew is any systematic divergence
  between features at training and features at
  inference. It is not drift; it is a code bug.
- Four canonical causes: two implementations, time-
  window mismatch, missing fitted preprocessor,
  feature source drift. Every incident is one of
  them.
- Prevention beats detection: shared transform code,
  fitted preprocessors serialised with the artifact,
  a feature store or its equivalent, and canonical
  source tables.
- Detection is the backstop: CI row-parity tests on a
  fixture, shadow-mode rollout with per-request
  parity, and the feature-anomaly counter in
  production.
- The skew audit is a checklist run at every
  promotion. Ritualise it.
- Skew is patient. It sits under green metrics for
  weeks. Design assuming it exists somewhere and
  build the tests that will find it.

## Where this goes next

Chapter 04 turns to the other systemic failure mode:
feedback loops and data flywheels. Where skew is a
bug the code allowed, a feedback loop is a bug the
system's dynamics create — the model's own predictions
shape the data that trains the next model, and the
loop can go beautifully (Netflix recommendations) or
disastrously (predictive policing). Different fix; same
"design it in on day one, not the day the incident
hits."
