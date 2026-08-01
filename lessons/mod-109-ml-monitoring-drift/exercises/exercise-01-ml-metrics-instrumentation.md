# exercise-01 — ML metrics instrumentation

> **Estimated effort:** 3 hours.
> **Companion chapters:** [01 — Why ML monitoring is its own
> discipline](../01-why-ml-monitoring-is-its-own-discipline.md),
> [02 — Instrumenting ML-specific signals](../02-instrumenting-ml-signals.md).

## Objective

Extend the FastAPI service you built (or would build) in
`mod-108` exercise-01 with the ML-native metrics surface
from chapter 02. By the end you should have:

- Six Prometheus metrics on `/metrics`, all labelled by
  `model_name` and `model_version`.
- Extended structured `prediction` log lines carrying a
  stable `prediction_id`.
- A `reference_profile.json` snapshot alongside the
  model artifact, populated from the training data.
- A small `NOTES.md` documenting a **cardinality
  audit** — how many series each metric produces at
  steady state.

The exercise is not "add a metric." It is "add exactly
the six metrics chapter 02 argues pay for themselves,
label them safely, and prove the surface is useful."

## Prerequisites

- The FastAPI service scaffold from
  `mod-108/exercises/exercise-01-fastapi-prediction-service.md`,
  or any equivalent scikit-learn-model-behind-FastAPI setup.
- Python 3.11+; `prometheus-client`, `pydantic ≥ 2`,
  `numpy`, `pandas`, `scikit-learn`, `pytest`,
  `python-json-logger` (or `structlog`).
- A trained scikit-learn `Pipeline` and the training
  DataFrame it was fit on. If you don't have one,
  reuse the model from `mod-104` or train a fresh churn
  or regression model on any tabular dataset.

## Problem statement

### Part A — The reference profile (≈ 45 min)

Write `training/build_reference_profile.py` that consumes
the training DataFrame and the fitted pipeline and emits
`artifact/reference_profile.json` in the shape from
chapter 02:

```json
{
  "reference_profile_id": "rp_<yyyy_mm_dd>_<git_sha>",
  "trained_at": "...Z",
  "n_rows": ...,
  "features": {
    "<name>": {
      "dtype": "...",
      "nullable": ...,
      "range": [min, max],           // numeric
      "quantiles": {"0.01": ..., "0.5": ..., "0.99": ...},
      "mean": ..., "std": ...,
      "categories": {"...": pct}     // categorical
    }
  },
  "predictions": {
    "mean": ..., "std": ...,
    "quantiles": {"0.5": ..., "0.95": ..., "0.99": ...},
    "class_balance": {...}
  },
  "sample_path": "artifact/reference_sample.parquet"
}
```

Also write `artifact/reference_sample.parquet` — a
uniform random sample of ≤ 10,000 rows from the training
set. It is what chapter 03's multivariate drift check
consumes.

The script should be idempotent for a fixed random seed:
running it twice on the same code and data produces
byte-identical `reference_sample.parquet` and JSON
values (aside from `trained_at`).

### Part B — The six metrics (≈ 60 min)

In `app/metrics.py`, expose all six metrics from
chapter 02 and instrument the request path to update
them:

1. **`ml_predictions_total`** — counter, labelled
   `(model_name, model_version, route, outcome_bucket)`.
2. **`ml_prediction_score`** — histogram, labelled
   `(model_name, model_version, route)`, with buckets
   chosen from the reference profile.
3. **`ml_inference_latency_seconds`** — histogram of
   *model-inference* time (excluding HTTP overhead),
   labelled `(model_name, model_version, route)`.
4. **`ml_feature_value`** — histogram per feature for
   the model's top ~5 features (choose by feature
   importance from the training run, or by
   documented business relevance). Labelled
   `(feature_name, model_name, model_version)` with
   per-feature bucket edges taken from the reference
   profile.
5. **`ml_feature_anomaly_total`** — counter of
   `(feature_name, kind, model_name, model_version)`
   where `kind ∈ {null, out_of_range, unknown_category}`.
   Compare each request's features to the reference
   profile at request time.
6. **`ml_model_loaded`** (gauge) and
   **`ml_model_load_seconds`** (histogram) around the
   lifespan model-load path.

Wire them into the `/predict` handler. Metrics update
happens synchronously; a metric-update failure must not
break a prediction.

### Part C — The prediction log (≈ 30 min)

Extend the `prediction` log event from `mod-108`
chapter 05 with:

- **`prediction_id`** — a stable ID generated at
  prediction time (`ulid`, `uuid.uuid4()`, or the
  library of your choice).
- **`features_hash`** — a sha256 of a canonical string
  representation of the ordered feature vector. Do not
  log the feature values themselves.
- **`reference_profile_id`** — read from the loaded
  profile.
- **`prediction`, `predicted_class`, `inference_ms`,
  `model_name`, `model_version`, `git_sha`** as
  chapter 02 lists.

`prediction_id` also lands in the HTTP response body
(so the caller can log it too) and in a new
`X-Prediction-Id` response header.

### Part D — Tests and a cardinality audit (≈ 45 min)

`tests/test_metrics.py`:

- **`test_metrics_endpoint_shape`** — `GET /metrics`
  returns 200 with the Prometheus text-exposition
  format; verify each of the six metric names is
  present.
- **`test_prediction_counter_increments`** — a
  successful `/predict` call increments
  `ml_predictions_total` for the request's model
  version and outcome bucket.
- **`test_feature_anomaly_on_unknown_category`** — send
  a request with a categorical value that is not in
  the reference profile's category set; assert the
  counter for that feature + `unknown_category`
  incremented.
- **`test_feature_anomaly_on_out_of_range`** — send a
  request with a numeric feature outside the reference
  profile's `range`; assert the corresponding counter
  incremented.
- **`test_prediction_id_propagates_to_response`** — the
  response body's `prediction_id` matches the
  `X-Prediction-Id` header.
- **`test_no_pii_in_metrics_labels`** — parse the
  scraped `/metrics` output; assert no label value
  looks like an email, phone, UUID, or free-form
  string (regex check is enough).

`NOTES.md`:

- **Cardinality audit.** For each of the six metrics,
  compute an upper bound on the series count at
  steady state given your service's expected label
  values. Show the math: `#versions × #features ×
  #routes × #buckets`. Flag any metric whose upper
  bound exceeds ~10k series and explain what you
  would change.
- **Bucket-choice rationale.** For `ml_prediction_
  score` and one `ml_feature_value` histogram, justify
  the bucket edges you chose. Reference the
  distribution from the reference profile.

## Starter guidance

- **Do the reference profile first.** Every other part
  reads from it. If you shortcut it (hardcoded ranges,
  hardcoded buckets), Part C's feature-anomaly counter
  will silently no-op.
- **Test the metric surface with the Prometheus
  client's own text-exposition parser** (`prometheus_
  client.parser.text_string_to_metric_families`) — it
  turns "did the counter increment?" into a one-liner
  and catches label typos.
- **The feature-anomaly check is a hot-path
  computation.** Cache the reference profile's
  category sets and ranges in the module's global
  scope; do not re-parse the JSON per request.
- **Log the `prediction_id` before the handler
  returns.** If the handler raises, the `finally`
  block in the middleware from `mod-108` chapter 05
  still needs the ID for the request-end log line.
- **Do not label metrics with `prediction_id`.** It's
  a per-request ID; cardinality goes to infinity.
  Logs carry it; metrics do not.

## Acceptance criteria

- `artifact/reference_profile.json` and `artifact/
  reference_sample.parquet` exist and match the
  chapter 02 schema.
- `GET /metrics` returns all six metric names, all
  labelled with `model_name` and `model_version`
  (verified by test).
- A `/predict` call produces:
  - An incremented `ml_predictions_total` counter.
  - An observation in `ml_prediction_score` and
    `ml_inference_latency_seconds`.
  - A `prediction` JSON log line with
    `prediction_id`, `features_hash`, and
    `reference_profile_id`.
  - An `X-Prediction-Id` response header matching
    the body.
- A `/predict` with an out-of-range or unknown-category
  feature increments the matching
  `ml_feature_anomaly_total` counter.
- The test suite passes: `pytest -q` reports zero
  failures across the seven tests listed.
- `NOTES.md` contains a cardinality audit that adds
  up.
- No PII, no raw feature values, no unbounded IDs in
  any metric label.

## Stretch goals (optional)

- **Multi-process metrics.** Reconfigure the service
  to use `prometheus_client`'s multiprocess mode
  behind gunicorn with 4 workers; verify counters
  aggregate correctly across workers.
- **A `/reference-profile` endpoint.** Return
  `reference_profile.json` verbatim (small; useful
  during incident response — the on-call can diff
  against what they see in the wild).
- **Adaptive histogram buckets from the reference
  profile.** Instead of hardcoding buckets, compute
  bucket edges from the reference profile's
  quantiles (e.g., 0, 0.05, 0.5, 0.95, 1.0 for a
  probability histogram). Document how you would
  handle re-bucketing across model version changes.
- **Redaction unit test.** Send a request whose
  category name looks like an email; assert the
  metric label value falls into the `unknown_
  category` bucket (not raw).
- **Batch-endpoint instrumentation.** Extend the
  metrics to the `/predict:batch` endpoint from
  `mod-108`; add a `batch_size` label if you can
  keep cardinality bounded (bucketed, not raw).

## What to hand in

```
exercise-01/
├── training/
│   └── build_reference_profile.py
├── artifact/
│   ├── model.joblib                    # from mod-108
│   ├── metadata.json                   # from mod-108
│   ├── reference_profile.json          # new
│   └── reference_sample.parquet        # new
├── app/
│   ├── main.py                         # extended
│   ├── metrics.py                      # NEW
│   ├── feature_check.py                # NEW — anomaly detection
│   ├── logging_config.py               # extended prediction log
│   └── middleware.py                   # from mod-108
├── tests/
│   ├── test_metrics.py                 # NEW
│   └── test_service.py                 # from mod-108
├── pyproject.toml
├── README.md                           # how to run
└── NOTES.md                            # cardinality audit + bucket rationale
```

Solutions live in the paired `ml-engineer-solutions`
repo — do not peek until you have submitted yours.
