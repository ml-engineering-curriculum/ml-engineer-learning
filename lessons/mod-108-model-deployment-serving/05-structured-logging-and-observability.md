# 05 — Structured logging and request observability

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 01–04.

## Motivation

An outage happens at 03:14. The oncall engineer runs
`kubectl logs` and gets:

```
INFO: request came in
INFO: predicted 0.87
ERROR: something went wrong
```

Which request? Which model version? Which caller? How long
did prediction take? What was the input? The unstructured
lines above are unactionable — every question the incident
demands to answer has to be answered somewhere else.

Contrast:

```json
{"ts":"2026-06-14T03:14:02.183Z","level":"info","event":"request_start","request_id":"01HGX...","route":"/predict","caller":"mobile-ios/4.2.1"}
{"ts":"2026-06-14T03:14:02.219Z","level":"info","event":"prediction","request_id":"01HGX...","model_name":"churn","model_version":"1.4.2","latency_ms":34.1,"proba":0.87}
{"ts":"2026-06-14T03:14:02.220Z","level":"error","event":"downstream_error","request_id":"01HGX...","dep":"feature-store","error":"timeout","elapsed_ms":1201}
```

Every question the incident demands to answer is a `jq` query
away. That is the whole reason for structured logging.

This chapter is short and prescriptive: what to log, how to
log it, what *not* to log. The recipe is boring on purpose —
the point is to make it so uniform across services that the
platform's log tooling (Loki, Elasticsearch, Datadog, the
GCP / AWS log explorers) can index and search it without any
per-service configuration.

## The logging model

Three principles do most of the work:

1. **Structured over prose.** Every log line is a JSON
   object with typed fields. The message string is one
   field among many, not the whole event.
2. **A request ID stitches events together.** One
   identifier per request, carried through every log line
   the request produces. Callers can pass their own
   (`X-Request-ID` header), or the service generates one on
   entry.
3. **Metadata about the model is on every prediction
   line.** `model_name`, `model_version`, `git_sha` from
   chapter 02's metadata. Now grep-by-version becomes
   trivial, and rolling deploy incidents can be attributed
   to the version, not the timestamp.

## A structured logging setup

Two choices for the actual library:

- **`structlog`** — the widely-used dedicated structured
  logger; feature-rich, ergonomic.
- **`logging` with a JSON formatter** (`python-json-
  logger`) — no new dependency beyond the stdlib
  formatter. Sufficient for this module.

Both work; both produce JSON on stdout. Structured logs are a
format contract, not a library choice — pick either and be
consistent.

A minimum-viable configuration using the stdlib:

```python
# app/logging_config.py
import logging
import sys

from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO"):
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level",
                           "name": "logger"},
            timestamp=True,
        )
    )
    root.addHandler(handler)
```

Two conventions matter:

- **stdout, not a file.** Containers stream stdout to the
  platform's log collector (chapter 06). Writing to a file
  requires log rotation and is invisible to the collector.
- **UTC ISO-8601 timestamps.** Human-local timezones make
  timelines impossible to correlate across services.

Call `configure_logging()` inside your FastAPI `lifespan` or
at import time — before anything else logs.

## The request middleware

The single most useful piece of code in the service:

```python
# app/middleware.py
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "request_end",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "user_agent": request.headers.get("user-agent"),
                },
            )
        response.headers["x-request-id"] = request_id
        return response
```

Wire it up:

```python
app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestLogMiddleware)
```

What this buys you:

- **A single log line per request.** Method, path, status
  code, duration. That is enough for most operational
  questions.
- **A propagated `request_id`.** In the header out to the
  client (useful in error messages) and on `request.state`
  for handler code to read.
- **Correct handling of exceptions.** The `finally` block
  logs the request whether the handler returned or raised.

## Correlating handler logs

Inside the handler, log the *interesting* events (the
prediction, downstream calls, unusual inputs) with the
`request_id` from `request.state`:

```python
@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest,
            bundle: Annotated[ModelBundle, Depends(get_bundle)]):
    request_id = request.state.request_id

    t0 = time.perf_counter()
    features = _to_frame(body)
    proba = float(bundle.model.predict_proba(features)[0, 1])
    duration_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "prediction",
        extra={
            "request_id": request_id,
            "model_name": bundle.metadata["model_name"],
            "model_version": bundle.version,
            "git_sha": bundle.metadata["git_sha"],
            "inference_ms": round(duration_ms, 2),
            "prediction": round(proba, 4),
        },
    )
    return PredictResponse(churn_probability=proba,
                           model_version=bundle.version)
```

Two useful discipline points:

- **Log the prediction, not the input.** The prediction is a
  scalar or a small vector; the input can contain PII.
- **Log the model version.** Every prediction, every line.
  Rolling deploys and A/B tests become greppable.

If you use `structlog`, `structlog.contextvars.bind_
contextvars(request_id=request_id)` in the middleware
attaches the ID to *every* log call in the request scope
automatically — you do not have to pass `extra=` on each
call. Small ergonomic win.

## What to log — a shortlist

**On request entry (middleware):**

- `request_id`, `method`, `path`, `user_agent`, `caller_ip`
  (if you can). The `x-forwarded-for` header, sanitised.

**Per prediction (handler):**

- `request_id`, `model_name`, `model_version`, `git_sha`.
- Inference duration in milliseconds.
- The prediction (probability, class, top-k) — small,
  informative, non-sensitive.
- Batch size for `/predict:batch`.

**On error:**

- `request_id`, exception type, exception message (short),
  the full traceback (as a single `error.traceback` field).
  Never send the traceback in the response body (chapter 03);
  do send it to logs.

**On lifecycle events:**

- Startup: model load time, warm-up time, model version.
- Shutdown: number of in-flight requests drained.
- Downstream failures: which dependency, which operation,
  elapsed time.

## What NOT to log — the PII/PHI/PCI checklist

Serving logs are almost always the highest-risk logs in an
organisation: they see live user requests. Two rules stop
the vast majority of incidents:

1. **Do not log request bodies verbatim.** Log fields you
   have deliberately whitelisted. When in doubt, log the
   *presence* of a field, not its value.
2. **Do not log identifiers that let a support engineer
   pivot from a log line to a person.** Hash user IDs at
   the boundary; log the hash. If you need to reverse the
   hash for legitimate investigations, the mapping lives
   in a separate, access-controlled system — not in the
   logs.

Concrete "do not log":

- Personal identifiers (name, email, phone, SSN, home
  address).
- Health information (diagnoses, medications, lab results).
- Payment information (PAN, CVV, full IBAN).
- Authentication material (bearer tokens, API keys,
  session cookies, refresh tokens, passwords).
- Location fine-grained enough to identify a person (raw
  lat/long, IP address in some jurisdictions).

The GDPR, HIPAA, and PCI-DSS regimes each define specific
categories of data that require additional handling — talk
to legal / compliance before shipping a service that
touches any of them. Your logging config should have PII
redaction *baked in*, not left to individual developers'
memory.

A structured redactor is one useful pattern:

```python
_REDACT_FIELDS = {"email", "phone", "ssn", "authorization"}


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for k in list(record.__dict__):
            if k in _REDACT_FIELDS:
                record.__dict__[k] = "[REDACTED]"
        return True
```

Attach it to the handler in `configure_logging`. Belt and
braces: catches the case where someone adds
`extra={"email": ...}` in a hurry.

## Log levels — the short version

- **DEBUG** — off in production; on when debugging locally.
  Detailed tracing (`features.dtypes`, `model.n_features_
  in_`).
- **INFO** — steady-state events. Every request goes through
  INFO. This is the volume tier.
- **WARN** — something odd but the request succeeded.
  Deprecated field on the request, feature store returned
  stale value, a fallback path activated.
- **ERROR** — a request failed. Include enough context to
  reproduce.
- **CRITICAL** — process-level catastrophe. Model won't
  load, out of memory. Pair with alerts.

Two-way rule: the level shapes both what the log costs and
what alerts fire on it. Overuse of `ERROR` produces
alert-fatigue; overuse of `INFO` on a chatty event fills the
log store. Reserve `ERROR` for things that should be
someone's problem.

## Metrics — a small first step

Structured logs are event-oriented. Metrics are counter- and
histogram-oriented and are the right way to answer questions
like "what's the p95 latency?" or "how many requests per
second?".

For this module, a small `/metrics` endpoint in Prometheus
format is enough:

```python
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response


REQUESTS = Counter(
    "predictions_total", "Number of predictions served",
    labelnames=("route", "model_version", "status"),
)
LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency (s)",
    labelnames=("route", "model_version"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Increment `REQUESTS` and observe `LATENCY` from within the
handler; the labels put model version on every counter. The
Prometheus scraper polls `/metrics`; dashboards and alerts
consume the histogram. Chapter 09 uses these metrics when
we get to latency budgeting.

`prometheus_client` is the reference library; multi-worker
gunicorn requires the multiprocess mode documented in the
library. `mod-109` (monitoring) covers what to *do* with the
metrics — alerts, dashboards, SLO burn rate — in depth.

## The one-request debugging story

The reason to go through all of this: on the day the service
misbehaves, the debugging story is one query.

```sh
# Find a specific request:
kubectl logs deploy/churn -n serving \
  | jq 'select(.request_id == "01HGX...")'

# Find every prediction from a specific model version:
kubectl logs deploy/churn -n serving \
  | jq 'select(.event == "prediction" and .model_version == "1.4.2")'

# 95th percentile latency in the last hour, from logs:
kubectl logs deploy/churn -n serving --since=1h \
  | jq 'select(.event == "prediction") | .inference_ms' \
  | sort -n | awk 'BEGIN {n=0} {a[n++]=$1} END {print a[int(n*0.95)]}'
```

None of these queries work against `INFO: predicted 0.87`.
All of them work against the JSON above.

## Common pitfalls

1. **Free-text log messages.** Grepping for `"error"` in
   free text misses half of them; structured `level=error`
   catches them all.
2. **No `request_id` propagation.** Impossible to correlate
   the middleware line with the handler lines with the
   downstream error line.
3. **Logging the request body.** PII in logs. Every incident
   response protocol starts with "purge the affected log
   entries", which is almost impossible for structured
   append-only stores.
4. **File-based logs in a container.** Invisible to the
   platform's log collector; require rotation. Stdout only.
5. **Local timezone in timestamps.** Timeline reconstruction
   across services becomes guesswork. UTC ISO-8601.
6. **Random `print()` calls.** Bypass the formatter,
   bypass the redactor, arrive in logs unstructured. Ban
   `print` in the service module; use the configured
   logger.
7. **Logging in a hot loop.** A `logger.debug` per row of a
   1000-row batch is 1000 log lines per request. Aggregate
   the loop, log the summary.
8. **Model version in the metadata endpoint but not in
   response headers.** The caller who logs `x-model-
   version` on their side can correlate too. Include it in
   both.

## Summary

- Structured (JSON) logs; a `request_id` that stitches
  every event in a request together; the model name /
  version / git SHA on every prediction line.
- One middleware logs request-in / request-out with
  duration and status. Handlers log the interesting events
  with the same `request_id`.
- Do not log request bodies verbatim, PII, PHI, PCI, or
  authentication material. Redact aggressively; whitelist
  fields explicitly.
- Log to stdout in a container; UTC ISO-8601 timestamps;
  a JSON formatter.
- Metrics are the counter/histogram companion to logs; a
  small `/metrics` endpoint is enough for this module.

## Where this goes next

Chapter 06 wraps the service in a Docker image: a Dockerfile
that plays nicely with `pip` caching, a multi-stage build
that keeps the runtime image small, a `.dockerignore` that
does not accidentally ship your dataset into production, and
the entrypoint that ties the gunicorn+uvicorn recipe from
chapter 03 to the container's process.
