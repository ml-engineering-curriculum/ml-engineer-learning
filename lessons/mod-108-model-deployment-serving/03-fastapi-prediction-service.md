# 03 — Anatomy of a FastAPI prediction service

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 01, 02.
> **Prerequisite installs:** `fastapi`, `uvicorn`,
> `pydantic ≥ 2`, and whichever framework your model uses.

## Motivation

FastAPI is the pragmatic default for an ML serving service in
2026. It is a small ASGI framework built on Starlette and
pydantic; it gives you request validation, dependency
injection, OpenAPI documentation, and async support for
essentially free, and the code you write looks like typed
Python functions rather than a framework.

The other reasonable options — Flask, Django, aiohttp, LitServe,
BentoML — each make sense in specific situations. For this
module we standardise on FastAPI because it is what most ML
platform teams accept as an "ordinary" service, and because
its pydantic-based request/response typing pushes the input
contract from chapter 02 to the boundary of the process, where
it belongs.

This chapter builds the service. Every subsequent chapter —
health checks (04), logging (05), the container (06), the
performance chapters (09–10) — extends what you build here.

## The service in one file

```python
# app/main.py
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.load import ModelBundle, load_bundle


class PredictRequest(BaseModel):
    age: int = Field(ge=0, le=120)
    plan_tier: str = Field(pattern="^(free|pro|enterprise)$")
    months_since_signup: int = Field(ge=0)


class PredictResponse(BaseModel):
    churn_probability: float = Field(ge=0.0, le=1.0)
    model_version: str


_bundle: ModelBundle | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _bundle
    artifact_dir = Path("/artifact")
    _bundle = load_bundle(artifact_dir)
    _bundle.model.predict(_warmup_frame())    # warm the graph
    yield
    _bundle = None


def get_bundle() -> ModelBundle:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return _bundle


app = FastAPI(lifespan=lifespan, title="churn-service")


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    bundle: Annotated[ModelBundle, Depends(get_bundle)],
) -> PredictResponse:
    features = _to_frame(body)
    proba = float(bundle.model.predict_proba(features)[0, 1])
    return PredictResponse(
        churn_probability=proba,
        model_version=bundle.version,
    )
```

Run it locally with `uvicorn app.main:app --host 0.0.0.0 --port
8080`. Visit `http://localhost:8080/docs` for the auto-generated
OpenAPI UI. `POST /predict` with a JSON body that matches
`PredictRequest`; get back a `PredictResponse`.

Every element of that file is deliberate. The rest of the
chapter unpacks it.

## Pydantic request and response models

Pydantic is FastAPI's superpower and the closest thing serving
gets to a training-time typechecker.

**Field-level validation.** The `PredictRequest` above rejects
a payload where `age = -1` (`ge=0`), `plan_tier = "trial"`
(pattern mismatch), or `months_since_signup` is missing
(required). The rejection is a `422 Unprocessable Entity` with
a machine-readable error body — before the request ever
reaches `predict()`.

**Aliases** for JSON payloads that use snake_case *and*
camelCase or shorter names:

```python
class PredictRequest(BaseModel):
    age: int
    plan_tier: str = Field(alias="planTier")
    months_since_signup: int = Field(alias="monthsSinceSignup")

    model_config = {"populate_by_name": True}
```

**Response models** guarantee the shape you promised the
caller. `response_model=PredictResponse` on the route makes
FastAPI validate the return value; if the code accidentally
returns `{"churn_prob": ...}` (typo), the response is
rejected at serialisation and the bug surfaces immediately.

**Extra fields.** Set `model_config = {"extra": "forbid"}`
on `PredictRequest` when you want the service to reject
payloads with unexpected fields — helpful for catching
typos in client code early. `"ignore"` (the default) silently
drops them; `"allow"` echoes them back.

**The input schema comes from the artifact.** Chapter 02
attached an `input_schema` to the metadata; a defensible
service either generates the pydantic model from it at build
time or asserts on startup that the manually-written pydantic
model still matches the metadata schema. That check is what
catches "we retrained with a new feature and forgot to update
the service" long before a caller does.

## The lifespan hook — load once, dispose cleanly

FastAPI's `lifespan` context manager runs once when the
process starts and once when it shuts down. It is the correct
place to load the model.

```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _bundle
    _bundle = load_bundle(Path("/artifact"))
    _warmup(_bundle)
    yield
    _bundle = None
```

Three things happen here:

- **Load** — the joblib / ONNX / MLflow load. This can take
  hundreds of milliseconds to seconds for a large model; it
  is out of the request path.
- **Warm-up** — a dummy prediction on synthetic input. The
  first prediction after load is materially slower than
  subsequent ones (JIT compilation on tree-based models,
  page faulting weights into RAM on neural nets, lazy
  initialisation of ONNX runtime graphs). If the warm-up call
  fails, the service does not start. That is the correct
  behaviour: it is better to fail fast at startup than to
  serve one broken prediction and 500 to the caller.
- **Yield** — the process is now ready to serve. Everything
  above `yield` runs at startup; everything below runs at
  shutdown.

The `_bundle` global is deliberately module-scoped, not
per-request. Chapter 04 will replace the `None`/`HTTPException`
dance with a proper readiness endpoint.

## Dependency injection

`Depends(get_bundle)` is how the route asks FastAPI for the
model. It looks like a small trick and it is doing three
useful jobs:

- **Testability.** In a test, you override the dependency:
  `app.dependency_overrides[get_bundle] = lambda: FakeBundle()`.
  No monkey-patching, no global state fiddling.
- **Composition.** Add `Depends(get_request_id)` for a
  correlation ID, `Depends(get_logger)` for a request-scoped
  logger, `Depends(auth)` for a token check. Each is a
  function; each is testable in isolation.
- **Documentation.** Dependencies with typed parameters show
  up in the OpenAPI spec correctly.

Do not reach for a big DI container. FastAPI's `Depends` is
sufficient for essentially every ML serving service you will
build.

## Feature transformation at the boundary

The pydantic model gives you a validated, typed Python object.
Your training pipeline expected a DataFrame with specific
columns. The `_to_frame` helper is the bridge:

```python
import pandas as pd


def _to_frame(body: PredictRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "age": body.age,
        "plan_tier": body.plan_tier,
        "months_since_signup": body.months_since_signup,
    }])
```

Two rules make this robust:

1. **Column names and order match the training pipeline
   exactly.** The `sklearn.pipeline.Pipeline` from chapter 02
   is column-name aware in scikit-learn ≥ 1.2 and will raise
   on mismatch; that raise is desirable — the wrong
   behaviour is silent misalignment.
2. **The pipeline does everything after that.** No manual
   `StandardScaler.transform`, no manual one-hot in the
   service. If it lives in the training pipeline, it stays
   in the training pipeline.

For non-tabular inputs (an image, a tokenised sequence, an
audio clip) the same rule applies at higher altitude — the
service converts the payload to whatever shape the model
expects (`torch.Tensor`, `np.ndarray`) and delegates
everything else to the packaged model.

## Sync vs async route handlers

FastAPI lets you write `def predict(...)` or `async def
predict(...)`. The rule:

- **`def` (sync)** — runs on a thread from FastAPI's
  threadpool. This is what you want for CPU-bound work like
  scikit-learn prediction, XGBoost prediction, small
  PyTorch inference on CPU.
- **`async def`** — runs on the main asyncio event loop.
  This is what you want when the handler *awaits I/O* — a
  database call for features, an HTTP call to a feature
  store, an `aioboto3` download.

**Do not put CPU-bound prediction in an `async def` handler
that does not `await`.** The event loop blocks for the
duration of the prediction; other connections cannot make
progress; the entire process degenerates to sequential
service. Use `async def` only when there is real awaited I/O,
or wrap the CPU work in `run_in_threadpool`:

```python
from starlette.concurrency import run_in_threadpool


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest,
                  bundle: Annotated[ModelBundle, Depends(get_bundle)]):
    features = _to_frame(body)
    proba = await run_in_threadpool(
        lambda: float(bundle.model.predict_proba(features)[0, 1])
    )
    return PredictResponse(churn_probability=proba,
                           model_version=bundle.version)
```

Chapter 09 revisits this decision under the concurrency and
throughput model.

## Errors — say what happened, without leaking

Three failure modes need explicit handling:

- **Bad input.** Pydantic returns `422` with a structured
  error body; no code from you.
- **Model raises during prediction.** `predict_proba` on a
  scikit-learn pipeline can raise `ValueError` on shape or
  dtype mismatch. Catch these at the route and return a
  `400` with a short, non-sensitive message:

  ```python
  from sklearn.exceptions import NotFittedError

  try:
      proba = float(bundle.model.predict_proba(features)[0, 1])
  except ValueError as e:
      raise HTTPException(status_code=400, detail=f"invalid input: {e}")
  except NotFittedError as e:
      raise HTTPException(status_code=500, detail="model not fitted")
  ```

- **Unhandled exception.** FastAPI returns `500 Internal
  Server Error` with no body by default. Add an exception
  handler that logs the traceback (chapter 05) and returns a
  short error body with a `request_id` the caller can quote:

  ```python
  from fastapi import Request
  from fastapi.responses import JSONResponse


  @app.exception_handler(Exception)
  async def _catchall(request: Request, exc: Exception):
      request_id = request.headers.get("x-request-id", "unknown")
      # (chapter 05) log the traceback with request_id here
      return JSONResponse(
          status_code=500,
          content={"error": "internal", "request_id": request_id},
      )
  ```

**Never** put a stack trace in the response body. It leaks
internal file paths and library versions; it is a small but
real information-disclosure vulnerability.

## Batch predictions in a single request

Callers often want to score more than one row per HTTP call —
it amortises the request overhead. Support both single and
batch by making the request field a list and preserving the
input order:

```python
class BatchPredictRequest(BaseModel):
    instances: list[PredictRequest] = Field(min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str


@app.post("/predict:batch", response_model=BatchPredictResponse)
def predict_batch(
    body: BatchPredictRequest,
    bundle: Annotated[ModelBundle, Depends(get_bundle)],
):
    features = pd.DataFrame([r.model_dump() for r in body.instances])
    proba = bundle.model.predict_proba(features)[:, 1]
    return BatchPredictResponse(
        predictions=[
            PredictResponse(
                churn_probability=float(p),
                model_version=bundle.version,
            )
            for p in proba
        ],
        model_version=bundle.version,
    )
```

Two guardrails: `max_length` caps the batch size so a single
request cannot exhaust memory; the response `predictions`
list is in the *same order* as `instances` (contract with the
caller — do not sort or shuffle). Chapter 08 revisits batch
vs. online at the architectural level.

## Metadata endpoint

A tiny endpoint that returns the bundle's metadata is
underrated. It answers "what model is this service serving?"
without a deploy log:

```python
@app.get("/model")
def model_info(bundle: Annotated[ModelBundle, Depends(get_bundle)]):
    return bundle.metadata
```

Callers can use this to log the version they hit; the
platform team can use it to verify a rollout landed; incident
responders can use it to answer "which git SHA was live at
03:14?" in one curl. Keep it read-only and unauthenticated
— it exposes nothing the caller cannot infer from responses.

## The OpenAPI spec is free

FastAPI serves an OpenAPI 3.1 spec at `/openapi.json` and a
Swagger UI at `/docs`. Two consequences worth internalising:

- **The spec is your interface contract.** Callers can
  auto-generate client SDKs (`openapi-generator`, `orval`)
  from it. If you break the schema, downstream generated
  code breaks. Version routes (`/v1/predict`, `/v2/predict`)
  when you must change the response shape.
- **`/docs` is a live sandbox.** Callers can hit `/predict`
  from a browser without curl. Useful in development; if
  you do not want it exposed in production, pass
  `docs_url=None, redoc_url=None` when constructing the
  `FastAPI` instance.

## The server: uvicorn (development), gunicorn+uvicorn (prod)

For development: `uvicorn app.main:app --reload`.

For production: run gunicorn with uvicorn workers, or
uvicorn's own multi-worker mode:

```
gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    -w 4 \
    -b 0.0.0.0:8080 \
    --timeout 30 \
    --graceful-timeout 30
```

`-w 4` starts four worker processes. Chapter 09 explains how
to pick the number and its interaction with the GIL and
model memory footprint. The container in chapter 06 uses
this command as its entrypoint.

## Common pitfalls

1. **Loading the model in a module-level `load_model()` call
   that runs at import.** Now every worker loads the model
   before it can respond to a signal. Startup times and
   memory allocation become unpredictable. Use `lifespan`.
2. **Skipping the warm-up prediction.** The first live
   request pays the JIT / page-fault cost; that request's
   p95 will look terrible.
3. **`async def` with a blocking prediction.** Blocks the
   event loop; throughput collapses.
4. **No `response_model`.** The service returns whatever
   the code happens to return. Callers write against
   whatever they got the day they tested. Regression risk
   compounds.
5. **Uncaught `ValueError` on bad input.** Returns `500`
   when it should return `400`. Alerts fire; the root cause
   is a client sending the wrong payload.
6. **Stack trace in the response body.** Information
   disclosure. Log the trace; do not send it.
7. **No batch endpoint.** Every caller starts calling
   `/predict` in a loop; the request overhead dominates the
   inference cost. Provide `/predict:batch`.
8. **Serialising the model version only implicitly.** The
   caller cannot tell which model produced which prediction.
   Put it in every response.

## Summary

- FastAPI + pydantic + `lifespan` is the minimum-viable
  serving stack. Every field of the request and response is
  typed; every startup task has a defined home.
- Load the model once in `lifespan`. Warm it. Keep the
  bundle in a module-level global; access it through a
  `Depends` for testability.
- Pydantic request/response models are your input contract.
  Generate them from — or assert them against — the
  `input_schema` in the artifact's metadata (chapter 02).
- Sync `def` for CPU-bound prediction; `async def` only for
  handlers that actually `await`. Never both.
- Expose `/predict`, `/predict:batch`, and `/model` at
  minimum. Chapter 04 adds `/healthz` and `/readyz`.
- Return the model version in every response. It is the
  cheapest observability you will ever add.

## Where this goes next

Chapter 04 adds the two endpoints your platform team is
going to ask for on day one: `/healthz` (is the process
alive?) and `/readyz` (is the model loaded and warm?). It
distinguishes the two rigorously — because Kubernetes and
every other orchestrator treats them as separate signals —
and it covers graceful shutdown so in-flight requests
finish before the process exits.
