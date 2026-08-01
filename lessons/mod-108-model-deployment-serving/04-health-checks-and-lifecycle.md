# 04 — Health checks and the service lifecycle

> **Reading time:** ~20 minutes.
> **Prerequisite chapters:** 01–03.

## Motivation

Two questions the platform is always asking your service:

1. **"Are you alive?"** — is the process running at all?
   Should I restart it?
2. **"Are you ready?"** — can you serve traffic right now?
   Should I send you requests?

They look identical from inside the process. They are not.
Conflating them produces a service that either restarts too
aggressively (a slow model load looks like a crash) or fails
under load (a partially-initialised service accepts traffic
it cannot serve). Every mature orchestrator — Kubernetes,
ECS, Nomad, HashiCorp Consul, most load balancers — treats
liveness and readiness as separate signals with separate
responses.

This chapter is short because the surface area is small.
Getting it wrong is expensive, and the fixes are easy once
the distinction is clear.

## Liveness vs. readiness — the precise distinction

### Liveness: "should this process be killed?"

A liveness probe answers: **is this process making forward
progress?** If it returns non-`200`, the orchestrator kills
the process and restarts it.

The correct behaviour of a liveness endpoint:

- **Trivial.** No dependencies, no I/O, no model reference.
  Return `200 OK` as long as the event loop is spinning.
- **Fast.** Under 100 ms. Ideally under 10.
- **Local.** Never touches the database, never calls the
  model, never talks to another service.

```python
@app.get("/healthz")
def healthz():
    return {"status": "ok"}
```

That is the whole implementation. If this endpoint fails,
something is catastrophically wrong (the process is
deadlocked, out of memory, or entirely hung) and restart is
the correct response.

**Anti-pattern**: putting the model prediction, the database
ping, or a downstream health check into `/healthz`. When
the database has a five-minute hiccup, the liveness probe
fails, every replica of your service is killed and restarted,
and the outage becomes a cascading incident. Downstream
health belongs in *readiness*, not liveness.

### Readiness: "should this process receive traffic?"

A readiness probe answers: **is this instance able to serve
requests successfully right now?** If it returns non-`200`,
the orchestrator stops sending it new requests (removes it
from the load balancer) but does not kill it.

The correct behaviour:

- **Model loaded?** The `lifespan` finished; `_bundle` is
  not `None`.
- **Warm-up succeeded?** The dummy prediction returned
  something sensible.
- **Critical dependencies reachable?** If the service cannot
  serve without the feature store, ping the feature store
  here. If the service cannot serve without the model
  registry, ping it here.
- **Shutdown flag not set?** During graceful shutdown, flip
  a flag so readiness starts returning non-`200`
  immediately (see below).

```python
from fastapi import status
from fastapi.responses import JSONResponse

_shutting_down = False


@app.get("/readyz")
def readyz():
    if _shutting_down:
        return JSONResponse(status_code=503,
                            content={"status": "shutting_down"})
    if _bundle is None:
        return JSONResponse(status_code=503,
                            content={"status": "loading"})
    # optional: ping the feature store / registry
    return {"status": "ready", "model_version": _bundle.version}
```

Readiness is allowed to be moderately expensive (up to a
second) and to include downstream checks. The key property is
that a failing readiness *pauses* traffic; it does not restart
the pod.

### Why the distinction matters in practice

Imagine a model that takes 30 seconds to load. If your
service uses the same endpoint for liveness and readiness
and returns `200` only after load completes:

- Kubernetes' default `initialDelaySeconds=0, periodSeconds=
  10, failureThreshold=3` kills the pod after 30 seconds of
  failed probes — right around when the model finishes
  loading. The pod restarts. The model reloads. Kubernetes
  kills it again. **CrashLoopBackOff.**

With separate probes:

- Liveness returns `200` from t = 0 (process is alive).
- Readiness returns `503` until t = 30 (model loading).
- Kubernetes never restarts the pod. It waits for readiness
  to flip to `200`, then adds the pod to the load balancer.

The mechanics are Kubernetes-specific; the shape is
universal. Every orchestrator you will encounter distinguishes
these two.

## The full lifecycle

```
        startup     warm-up     serving     drain      exit
          │           │            │           │         │
   ┌──────┴───────────┴────────────┴───────────┴─────────┴──────┐
   │ lifespan()      lifespan() │  request  │ signal   │ cleanup│
   │ enter — load    enter —    │  loop     │ SIGTERM  │ close  │
   │ artifact        warm the   │           │ readyz → │ conns  │
   │                 model      │           │ 503      │        │
   └────────────────────────────┴───────────┴──────────┴────────┘
        readyz: 503                readyz: 200   readyz: 503
        healthz: 200                healthz: 200 healthz: 200
```

Five stages, three of them interesting.

### Startup

The process starts. FastAPI's `lifespan` enters. Load the
artifact from chapter 02. This can take seconds. During this
window: **healthz returns `200`, readyz returns `503`**.

### Warm-up

The model is loaded but has never been called. A first
prediction pays for JIT compilation, first-fault memory
paging, and any lazy initialisation the framework does.
Perform a dummy prediction on a representative input, discard
the result. Time it — a warm-up that takes 5 s the first time
and 2 ms afterwards tells you exactly how much cold-start
latency you would have shipped to a real user.

```python
def _warmup(bundle: ModelBundle):
    import time
    t0 = time.perf_counter()
    bundle.model.predict(_synthetic_batch(n=1))
    dt = (time.perf_counter() - t0) * 1000
    logger.info("model_warmup_ms=%.2f", dt)
```

If the warm-up raises, do not let the service go `Ready`.
Failing fast at startup is much better than failing on the
first live request.

### Serving

Steady state. Both endpoints return `200`.

### Drain

The orchestrator wants to shut this instance down (a deploy,
a scale-down, a node drain). It sends `SIGTERM`.

The correct behaviour, ordered:

1. **Flip a `_shutting_down` flag.** Readyz starts returning
   `503`. The load balancer notices and stops sending new
   requests.
2. **Wait for in-flight requests to finish.** Configure
   gunicorn's `--graceful-timeout` (or uvicorn's
   `--timeout-graceful-shutdown`) high enough to cover the
   longest expected request.
3. **Exit.**

```python
_shutting_down = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _bundle
    _bundle = load_bundle(Path("/artifact"))
    _warmup(_bundle)
    yield
    _shutting_down = True
    # (gunicorn/uvicorn drains connections before the process exits;
    #  this yield just lets any cleanup you want happen here)
```

The trick: **readyz has to flip *before* the process starts
declining connections**, so the load balancer stops sending
traffic. If you skip step 1, the load balancer keeps sending
requests for the polling interval (often 10–30 s) and every
one of them fails.

Kubernetes provides a `preStop` hook precisely for this —
run `sleep 15` in preStop so the pod has time to be removed
from the endpoints list *before* SIGTERM triggers the drain.
The platform team may configure this; you should know it
exists so you can ask.

### Exit

Cleanup: flush logs, close database connections, wait for
outstanding tasks. FastAPI's `lifespan` after `yield` is the
place. Under gunicorn+uvicorn workers this runs per worker.

## Startup and readiness probe configuration

The Kubernetes flavour of the tuning knobs — the shape is
mirrored in ECS, Nomad, etc.:

- **`initialDelaySeconds`** — wait this long before starting
  to probe. Don't rely on this alone; it delays the *first*
  probe but does not help if the model load exceeds the
  delay.
- **`periodSeconds`** — how often to probe.
- **`timeoutSeconds`** — how long the probe has to respond.
  Set higher than the endpoint's expected latency plus
  network overhead.
- **`failureThreshold`** — consecutive failed probes before
  action.
- **`startupProbe`** — Kubernetes-specific: a separate probe
  that runs *only* during startup, with its own generous
  thresholds. Once it passes, the normal liveness /
  readiness probes take over. **Use this when model load
  takes more than a handful of seconds** — it decouples
  startup budget from steady-state probe frequency.

You (the ML engineer) don't usually own the manifest, but
you own the numbers on it. The handoff artifact you give
the platform team (chapter 11) should include a *suggested
probe budget*: startup takes ≤ X seconds, warm-up takes ≤ Y,
steady-state readyz responds in ≤ Z ms.

## What readiness should — and should not — check

Include:

- **Model loaded and warmed.** Every service.
- **Downstream dependencies without which no request can
  succeed.** Feature store the service must call every
  request; database of primary keys the model looks up;
  the model registry if the service loads on demand.

Exclude:

- **Downstream dependencies that some requests need and
  others do not.** Include per-request error handling
  instead; don't fail the whole instance.
- **Expensive operations.** A readiness probe that pings a
  slow database on every 10-second poll adds noise to that
  database. If you must, cache the result of the check for
  30 seconds and return the cached value.
- **The prediction itself.** Do it once in warm-up; don't
  repeat on every probe. Prediction is not free.

## `/healthz`, `/livez`, `/readyz` — pick a convention

Names vary by community. The names used consistently in this
module are `/healthz` (liveness) and `/readyz` (readiness),
which mirror the Kubernetes convention. `/livez` is a
synonym for `/healthz` used interchangeably in some places.
Pick one convention per organisation, stay consistent, and
put the two endpoints in the same file next to each other so
that no one confuses them.

A useful additional endpoint:

- **`/startupz`** — mirrors Kubernetes' `startupProbe`.
  Returns `200` only after the model is loaded and warmed.
  On a service with a long startup, you can point the
  Kubernetes `startupProbe` at this and use generous
  `failureThreshold`s (e.g., 60 probes × 5 s = 5 minute
  startup budget).

## The observable side effects of a good health story

- **Rolling deploys are boring.** New pods pass readyz, get
  added to the load balancer, old pods drain and exit.
  Zero-downtime.
- **Alerts fire on the right thing.** "Availability =
  fraction of requests that returned 2xx" is not
  contaminated by traffic sent to a pod that was still
  loading. That number reflects a real user experience.
- **Autoscaling works.** Kubernetes' HPA and the underlying
  metrics-based scalers only add pods that eventually
  become ready. If readiness is honest, scaling behaviour
  is predictable.
- **Incident triage is faster.** "Is the service up?" is a
  well-defined question with a well-defined answer.

## Common pitfalls

1. **Same endpoint for liveness and readiness.** Turns a
   slow start into a crash loop; turns a downstream blip
   into a cascading restart.
2. **Readiness that runs the model.** Adds latency and load
   for no signal beyond "the model didn't crash on this
   input just now". Warm once, trust the warm.
3. **No graceful shutdown.** Deploys drop in-flight
   requests. Users see 502s that alerts blame on the new
   version when the true cause is drop-in-flight from the
   old.
4. **Long-blocking work in `lifespan`.** The event loop is
   frozen until it returns. That is what you *want* for
   model load — but ensure your gunicorn worker timeout is
   set high enough that the worker is not killed while
   loading. `--timeout 60` or higher for large models.
5. **Health endpoints that require auth.** The orchestrator
   will not send a bearer token. Keep `/healthz` and
   `/readyz` unauthenticated; do not put PII in the
   response bodies.
6. **Readiness passes but predictions 500.** The check does
   not exercise the code path that matters. A warm-up
   prediction during startup is the cure — same code path
   as a real request.

## Summary

- Liveness and readiness are *different signals*. Liveness
  says "restart me if I'm broken"; readiness says "send me
  traffic if I'm ready".
- Liveness is trivial. Readiness returns `200` only after
  the model is loaded, warmed, and dependencies are
  reachable.
- Graceful shutdown: flip readyz to `503`, drain in-flight
  requests, then exit. The platform team's `preStop` hook
  helps.
- Warm-up during startup — a real prediction on synthetic
  input — is what makes the first live request behave like
  the hundredth.
- These endpoints are the API you offer the platform team.
  The handoff document (chapter 11) states the probe
  budget explicitly.

## Where this goes next

Chapter 05 makes the service observable. Structured JSON
logs with request IDs, per-request timings, the
`model_version` from chapter 02 in every log line, and a
lightweight metrics endpoint. Then chapter 06 wraps the
whole thing in a Docker image.
