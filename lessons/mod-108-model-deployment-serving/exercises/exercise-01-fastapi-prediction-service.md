# exercise-01 — FastAPI prediction service

> **Estimated effort:** 3–4 hours.
> **Companion chapters:** [02 — Packaging the model artifact](../02-packaging-the-model-artifact.md),
> [03 — Anatomy of a FastAPI prediction service](../03-fastapi-prediction-service.md),
> [04 — Health checks and the service lifecycle](../04-health-checks-and-lifecycle.md),
> [05 — Structured logging and request observability](../05-structured-logging-and-observability.md).

## Objective

Take a trained scikit-learn model and put it behind an HTTP
service that a stranger could realistically call. By the end
you should have:

- A **packaged model artifact** — model file plus
  `metadata.json` containing version, git SHA, framework
  version, input/output schema, and eval metrics.
- A **FastAPI service** — `/predict`, `/predict:batch`,
  `/healthz`, `/readyz`, `/model` — with pydantic request
  and response models generated from (or asserted against)
  the artifact's schema.
- **Structured JSON logs** on stdout with per-request
  timings and `request_id` propagation.
- **A minimal test suite** using `TestClient` and dependency
  overrides.

The point of the exercise is not to write a great model
(use anything from `mod-104`); it is to internalise the
"model in prod" contract from chapters 02–05.

## Prerequisites

- Chapters 02–05 of this module.
- Python 3.11+; `fastapi`, `uvicorn`, `pydantic ≥ 2`,
  `scikit-learn`, `joblib`, `python-json-logger` (or
  `structlog`), `pytest`, `httpx`.
- A trained scikit-learn `Pipeline` — reuse the churn or
  regression model from `mod-104`, or train a fresh one in
  a training script. The model itself is not evaluated in
  this exercise; the *serving* is.

## Problem statement

### Part A — Packaging (≈ 45 min)

Write `training/save_artifact.py` that:

1. Trains a scikit-learn `Pipeline` on any tabular dataset
   with at least three input columns and one output.
2. Serialises the pipeline with `joblib.dump` to
   `artifact/model.joblib`.
3. Writes `artifact/metadata.json` with the schema described
   in chapter 02:
   - `model_name`, `model_version`, `git_sha`,
     `trained_at` (UTC ISO-8601).
   - `framework` (name + version), `python_version`.
   - `input_schema` — list of `{name, dtype, nullable,
     categories?}`.
   - `output_schema` — the fields your `/predict` response
     will contain.
   - `eval_metrics` — at least one number computed on a
     held-out split.

The training script should be re-runnable and idempotent:
running it twice with the same code and data should produce
byte-identical metadata (aside from `trained_at` and — if
seeds are correct — the model bytes).

### Part B — The service (≈ 75 min)

Implement `app/main.py` with the shape from chapter 03:

- **`lifespan`** that loads `ModelBundle` from
  `ARTIFACT_DIR` (env var, default `/artifact`), runs a
  warm-up prediction on synthetic input, and holds a
  process-scoped reference.
- **`get_bundle` dependency** used by the route handlers.
- **Pydantic `PredictRequest`** with field-level validation
  derived from `metadata.input_schema` (either generated at
  startup or manually written with an assertion that the
  metadata matches the pydantic model — surface a clear
  error if not).
- **`PredictResponse`** with the model output(s) plus
  `model_version`.
- **`POST /predict`** — single prediction. Uses
  `Depends(get_bundle)`; converts the request to a
  DataFrame in the same column order the training pipeline
  expects; calls `predict_proba` / `predict`; returns the
  response.
- **`POST /predict:batch`** — accepts a list of up to 500
  instances; returns predictions in the same order.
  Enforce the max with pydantic (`max_length=500`).
- **`GET /model`** — returns `bundle.metadata` verbatim.
- **`GET /healthz`** — trivial `{"status": "ok"}`.
- **`GET /readyz`** — returns 200 only when `bundle` is
  loaded and warm; 503 otherwise.

Handle at least these error modes:

- Bad input → 422 (pydantic default is fine).
- Model raises `ValueError` → 400 with a short, non-
  sensitive message.
- Everything else → 500 with a `request_id` echoed in the
  body. Never send a stack trace in the response.

### Part C — Structured logging (≈ 45 min)

Wire up a JSON logger (`python-json-logger` or `structlog`).
Add a `RequestLogMiddleware` (chapter 05) that:

- Reads or generates a `request_id`.
- Logs one line at the *end* of every request with
  `method`, `path`, `status_code`, `duration_ms`,
  `request_id`.
- Echoes `x-request-id` in the response header.

In the `/predict` handler, log an additional structured
event `prediction` with `model_name`, `model_version`,
`git_sha`, `inference_ms`, and the returned probability.
Do **not** log the request body verbatim.

Log to stdout only. UTC ISO-8601 timestamps.

### Part D — Tests (≈ 45 min)

`tests/test_service.py`:

- **`test_healthz`** — 200, body `{"status": "ok"}`.
- **`test_readyz_ready`** — 200 after `lifespan` runs.
- **`test_predict_happy_path`** — using
  `app.dependency_overrides[get_bundle]` to inject a fake
  bundle, verify status 200, valid probability, and
  `model_version` in the response.
- **`test_predict_bad_input`** — invalid payload returns
  422 with an error body naming the offending field.
- **`test_predict_batch_preserves_order`** — send N
  instances with distinguishing values; verify the
  response order matches.
- **`test_model_endpoint_returns_metadata`** — 200; body
  contains at least `model_name` and `model_version`.

`tests/test_logging.py`:

- **`test_request_id_propagates`** — send a request with
  `x-request-id: known-id`; verify the response echoes the
  same header and the log line (captured via `caplog` or
  a JSON-log capture fixture) contains the same value.

## Starter guidance

- **Chapter 03's `main.py` is a strong skeleton.** Do not
  copy it verbatim; adapting it to your dataset teaches
  the shape.
- **Column order is not a suggestion.** If your training
  pipeline was fit on `[age, plan_tier, months]`, the
  DataFrame in `_to_frame` must have those columns in
  that order. Scikit-learn ≥ 1.2 raises on mismatch —
  that raise is your friend.
- **Warm-up requires representative input.** A synthetic
  row that is valid for the pipeline but has no domain
  meaning (all defaults / zeros) is fine.
- **Do not skip Part C.** Structured logging is what
  makes chapter 05 real; every subsequent exercise builds
  on it.
- **Use `pytest -x -q` while iterating** and
  `pytest --cov=app` before you call it done.

## Acceptance criteria

- `training/save_artifact.py` produces `artifact/
  model.joblib` and a `metadata.json` matching chapter
  02's schema.
- The service starts locally with `uvicorn app.main:app`
  and serves all six endpoints described in Part B.
- `GET /readyz` returns 503 until the model is loaded and
  warmed, then 200 for the process lifetime.
- Every request produces at least one JSON log line on
  stdout with a `request_id`; `/predict` produces an
  additional `prediction` log line.
- The test suite passes: `pytest -q` reports zero
  failures across the six functional tests and the
  logging test.
- No PII, no raw request bodies, no stack traces in log
  lines that could reach a log store.
- `curl -X POST http://localhost:8000/predict -H
  'content-type: application/json' -d '<valid body>'`
  returns 200 with a `model_version` field.

## Stretch goals (optional)

- **Generate the pydantic model from `metadata.
  input_schema`.** Instead of hand-writing
  `PredictRequest`, build it at startup with
  `pydantic.create_model` from the metadata. Verify the
  service still validates the same set of payloads.
- **Add an OpenAPI example.** Populate
  `input_example` in the metadata and wire it into the
  route's OpenAPI docs (`Field(..., examples=[...])`).
  Open `/docs` and confirm the "Try it out" pane is
  pre-filled.
- **Model-quality assertion.** In `save_artifact.py`,
  refuse to overwrite the artifact if
  `eval_metrics["roc_auc"]` (or your primary metric) is
  worse than the previous version. Chapter 07's model-
  registry topic anticipates this.
- **`orjson` response.** Swap FastAPI's default JSON
  encoder for `orjson`. Measure the p95 change with a
  small `wrk`/`k6` run. Report the delta.
- **`structlog`** — port the logging module to
  `structlog` with `bind_contextvars(request_id=...)`.
  Simpler in the handler code; different setup.

## What to hand in

```
exercise-01/
├── training/
│   └── save_artifact.py
├── artifact/                     # generated; .gitignore optional
│   ├── model.joblib
│   └── metadata.json
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app
│   ├── config.py                 # env-var settings
│   ├── logging_config.py         # JSON logger + redactor
│   ├── middleware.py             # request-log middleware
│   └── model.py                  # ModelBundle + loader
├── tests/
│   ├── test_service.py
│   └── test_logging.py
├── pyproject.toml
├── README.md                     # how to run + curl example
└── NOTES.md                      # what surprised you
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.
