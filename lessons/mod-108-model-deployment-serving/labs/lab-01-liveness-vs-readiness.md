# lab-01 — Liveness vs readiness: watch a slow-starting service survive (or crash-loop)

> **Duration:** ~30–40 minutes.
> **Companion chapter:** [04 — Health checks and the service lifecycle](../04-health-checks-and-lifecycle.md).

## Objective

Reproduce chapter 04's central mechanical claim — "a service that wires the
same endpoint to both the liveness and readiness probes turns a slow
startup into a CrashLoopBackOff" — on a tiny FastAPI app you can run on a
laptop, without needing a Kubernetes cluster. By the end you will have (a)
built a service whose `lifespan` sleeps for a configurable number of
seconds to simulate a slow model load, (b) exposed **two endpoint shapes**
— a single `/health` wired to both probes, and the chapter-04 split
`/healthz` + `/readyz` — (c) run a short bash loop that mimics Kubernetes'
default probe policy (`periodSeconds=5, failureThreshold=3`) against each
shape, (d) watched the single-endpoint service "get killed" repeatedly
while the split-endpoint service passes cleanly, and (e) added the chapter-
04 **warm-up** and the readyz **shutdown flag** so draining works too.

This lab is deliberately narrower than **exercise-01** (which builds a
full FastAPI prediction service with pydantic schemas, structured logs,
and a test suite). The lab is the isolation chamber where you *see* the
liveness/readiness distinction cause a specific restart behaviour; the
exercise then wires correct probe semantics into a real serving service.

## Prerequisites

- Python 3.11+.
- `fastapi`, `uvicorn` installed in a virtual environment of your choice
  (`uv venv && uv pip install fastapi uvicorn` is fine).
- `curl` and `bash` available on your shell.
- Chapter 04 read through once.
- A CPU is enough; nothing in this lab needs a GPU or Docker.

## Setup

Create a lab directory with two files:

```bash
mkdir -p lab-01-liveness-vs-readiness && cd lab-01-liveness-vs-readiness
touch app.py probe_loop.sh
chmod +x probe_loop.sh
```

Everything below goes in those two files. You will run the server in one
terminal and the probe loop in another.

## Steps

### Step 1 — Build a service with a deliberately slow startup (≈ 5 min)

Chapter 04's cold-start story is "the model takes 30 seconds to load; the
probe policy has to respect that." Simulate it with a `lifespan` that
sleeps for `LOAD_SECONDS`, with no real model in sight.

Put this in `app.py`:

```python
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

LOAD_SECONDS = float(os.getenv("LOAD_SECONDS", "20"))
MODE = os.getenv("MODE", "split")  # "single" or "split"

_ready = False
_shutting_down = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ready
    print(f"[lifespan] loading for {LOAD_SECONDS:.1f}s ...", flush=True)
    time.sleep(LOAD_SECONDS)                     # fake the model load
    # chapter 04 warm-up: a dummy prediction on synthetic input.
    # Here it's a no-op call; the point is the code path exists.
    print("[lifespan] warm-up complete", flush=True)
    _ready = True
    yield
    print("[lifespan] shutdown", flush=True)
    _ready = False


app = FastAPI(lifespan=lifespan)


# --- MODE = "single" ----------------------------------------------------
# Chapter 04's anti-pattern: one endpoint wired to BOTH probes.
# It only returns 200 after the model is loaded, so Kubernetes' liveness
# probe fails during loading and the orchestrator restarts the pod.
@app.get("/health")
def health_single():
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


# --- MODE = "split" -----------------------------------------------------
# Chapter 04's recommended shape: separate endpoints.
@app.get("/healthz")
def healthz():
    # Liveness: trivial, fast, local. No model reference.
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if _shutting_down:
        return JSONResponse(status_code=503,
                            content={"status": "shutting_down"})
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready"}
```

Start the server in **terminal 1**:

```bash
LOAD_SECONDS=20 uvicorn app:app --host 127.0.0.1 --port 8080
```

You should see `[lifespan] loading for 20.0s ...` print immediately.
Uvicorn will **not** accept requests yet — chapter 04's point: the
`lifespan` is blocking, which is what you want for the model load, but you
have to budget probe timeouts for it.

**Expected output.** Twenty seconds after startup, `[lifespan] warm-up
complete`, then uvicorn prints `Application startup complete.`. From that
moment on, every endpoint responds.

**Checkpoint.** `curl -s -w "%{http_code}\n" http://127.0.0.1:8080/healthz
-o /dev/null` returns `200` as soon as the application-startup-complete
line appears. Chapter 04's whole claim is that this `200` should have been
visible *from t=0*, not from t=20.

### Step 2 — Simulate the orchestrator's probe policy (≈ 5 min)

Chapter 04 states the Kubernetes default: `periodSeconds=10,
failureThreshold=3`. We'll use `periodSeconds=2, failureThreshold=3` so
the lab fits in a single startup window — 6 seconds to a decision instead
of 30, same shape.

Put this in `probe_loop.sh`:

```bash
#!/usr/bin/env bash
# Usage: ./probe_loop.sh <probe-url> <period-seconds> <failure-threshold>
# Mimics an orchestrator: polls the URL at the given period; counts
# consecutive non-200 responses; after `failure-threshold` in a row,
# logs "RESTART" (we don't actually restart; that's your terminal 1).
set -euo pipefail

URL="${1:-http://127.0.0.1:8080/healthz}"
PERIOD="${2:-2}"
THRESHOLD="${3:-3}"

fails=0
t0=$(date +%s)
while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "${URL}" || echo "000")
  elapsed=$(( $(date +%s) - t0 ))
  if [[ "${code}" == "200" ]]; then
    printf "[t=%3ds] %s -> 200   fails=0\n" "${elapsed}" "${URL}"
    fails=0
  else
    fails=$((fails + 1))
    printf "[t=%3ds] %s -> %s   fails=%d/%d\n" \
      "${elapsed}" "${URL}" "${code}" "${fails}" "${THRESHOLD}"
    if [[ "${fails}" -ge "${THRESHOLD}" ]]; then
      printf "[t=%3ds] !!! orchestrator would RESTART the pod now\n" "${elapsed}"
      fails=0        # fresh pod would restart the counter
    fi
  fi
  sleep "${PERIOD}"
done
```

**Checkpoint.** `bash probe_loop.sh` with the server running should
produce alternating lines; the `RESTART` message triggers after three
consecutive non-200 responses, matching chapter 04's `failureThreshold=3`
example. In Kubernetes the orchestrator would send `SIGKILL` and start a
new pod — on the lab you just observe the message and keep the server
running.

### Step 3 — The anti-pattern: same endpoint for both probes (≈ 10 min)

Chapter 04: "a liveness probe that returns `200` only after the model
load … Kubernetes kills the pod after 30 seconds of failed probes — right
around when the model finishes loading. **CrashLoopBackOff.**" Reproduce
it.

In **terminal 1**, restart the server in `single` mode with a 20-second
load:

```bash
# Ctrl-C the previous run, then:
MODE=single LOAD_SECONDS=20 uvicorn app:app --host 127.0.0.1 --port 8080
```

In **terminal 2**, point the probe loop at `/health` with the shortened
timing so the "restart" decision falls inside the load window:

```bash
./probe_loop.sh http://127.0.0.1:8080/health 2 3
```

**Expected output.** The probe loop prints a few lines like:

```
[t=  2s] http://127.0.0.1:8080/health -> 503   fails=1/3
[t=  4s] http://127.0.0.1:8080/health -> 503   fails=2/3
[t=  6s] http://127.0.0.1:8080/health -> 503   fails=3/3
[t=  6s] !!! orchestrator would RESTART the pod now
[t=  8s] http://127.0.0.1:8080/health -> 503   fails=1/3
[t= 10s] http://127.0.0.1:8080/health -> 503   fails=2/3
[t= 12s] http://127.0.0.1:8080/health -> 503   fails=3/3
[t= 12s] !!! orchestrator would RESTART the pod now
```

The "RESTART" message fires twice *before* the model finishes loading at
t=20. On a real cluster, each of those is an actual pod kill plus a
re-pull plus a fresh 20-second load — the pod never becomes ready, and
Kubernetes transitions it to `CrashLoopBackOff`.

Let the probe continue after the load completes: once `[lifespan] warm-up
complete` prints, the probe starts seeing `200` and `fails=0`. In
Kubernetes that would only happen if *one* restart happened to finish
loading faster than the probe timing allowed — which, with a stable
startup time, it never does.

**Checkpoint.** You have just seen chapter 04's CrashLoopBackOff cause in
seven seconds flat. The probe is honest ("is the service serving
requests? no"), and the orchestrator is honest ("three failures in a row,
restart"), and together they produce a cycle that cannot converge. The
bug lives in the *shape of the signal*, not in either side.

Kill the probe loop (Ctrl-C in terminal 2) and the server (Ctrl-C in
terminal 1).

### Step 4 — The chapter-04 fix: split the probes (≈ 10 min)

In **terminal 1**, start the server in `split` mode with the same slow
load:

```bash
MODE=split LOAD_SECONDS=20 uvicorn app:app --host 127.0.0.1 --port 8080
```

In **terminal 2**, run **two** probe loops in the background so you can
watch both signals at once. Easiest is a single terminal with the output
interleaved:

```bash
./probe_loop.sh http://127.0.0.1:8080/healthz 2 3 &
LIVENESS_PID=$!
./probe_loop.sh http://127.0.0.1:8080/readyz  2 3 &
READINESS_PID=$!
wait
# Ctrl-C ends both; or: kill ${LIVENESS_PID} ${READINESS_PID}
```

**Expected output.** Interleaved lines like:

```
[t=  2s] http://127.0.0.1:8080/healthz -> 200   fails=0
[t=  2s] http://127.0.0.1:8080/readyz  -> 503   fails=1/3
[t=  4s] http://127.0.0.1:8080/healthz -> 200   fails=0
[t=  4s] http://127.0.0.1:8080/readyz  -> 503   fails=2/3
[t=  6s] http://127.0.0.1:8080/healthz -> 200   fails=0
[t=  6s] http://127.0.0.1:8080/readyz  -> 503   fails=3/3
[t=  6s] !!! orchestrator would RESTART the pod now
```

But in real Kubernetes, that "RESTART" line comes from the **readiness**
probe loop, not the liveness one. **Readiness failure does not restart
the pod** — it only removes the pod from the load balancer. The pod keeps
loading; the LB keeps it out of rotation; at t≈20 readyz flips to `200`,
the LB adds the pod back, and the service is up. Zero restarts.

Our tiny simulator doesn't know the difference between the two probes —
it prints "RESTART" on any three-in-a-row — so you have to read it as
"this is what a liveness probe would trigger." For the readyz loop, the
correct interpretation is "the orchestrator would stop sending traffic
during this window."

Let the loops continue past t=20. The readyz line flips to `200` and
`fails=0` from that moment on; the healthz line was `200` from t=2. The
service never had any reason to be restarted.

**Checkpoint.** The two-probe split transforms the same 20-second load
from a crash-loop into a boring, correctly-handled rolling start. The
endpoints are different signals: `/healthz` says "should I kill you?"
(answer: no, the process is alive), and `/readyz` says "should I send you
traffic?" (answer: not yet, the model is still loading). Chapter 04 pitfall
#1 is exactly this bug; this is the before/after.

### Step 5 — Add the drain flag so shutdown is graceful (≈ 5 min)

Chapter 04's drain stage: on SIGTERM, flip a flag so readyz starts
returning `503` *before* the process declines connections. Load balancers
poll on their own cadence (10–30 s typical); if readyz stays `200` right
up to the moment the process exits, every request that landed in that
window gets a 502 from the LB.

Add a signal handler to `app.py` just below the `lifespan` function:

```python
import signal


def _handle_sigterm(_signum, _frame):
    global _shutting_down
    print("[signal] SIGTERM received; flipping readyz to 503", flush=True)
    _shutting_down = True


signal.signal(signal.SIGTERM, _handle_sigterm)
```

Start the service in `split` mode again, wait for warm-up, then in a
separate terminal:

```bash
# Confirm readyz is 200
curl -s -w "%{http_code}\n" http://127.0.0.1:8080/readyz
# -> {"status":"ready"}200
# Send SIGTERM to the uvicorn process (replace PID)
kill -TERM "$(pgrep -f 'uvicorn app:app' | head -1)"
# Immediately re-check readyz — depends on whether uvicorn exits first;
# in a real gunicorn + --graceful-timeout setup, the process stays up long
# enough to drain and this flips to 503.
curl -s -w "%{http_code}\n" http://127.0.0.1:8080/readyz
```

On a vanilla `uvicorn app:app`, SIGTERM exits fast enough that you may
not see the `503` before the process is gone. The *mechanism* you built
(the `_shutting_down` flag, flipped by the handler) is what gunicorn's
`--graceful-timeout` combined with Kubernetes' `preStop: sleep 15` keeps
alive long enough for the load balancer to notice. The lab has made the
flag; production wires the timeouts.

**Expected output.** The `[signal] SIGTERM received; flipping readyz to
503` log line appears in terminal 1 (the server). In a gunicorn
deployment with `--graceful-timeout 30` and a Kubernetes `preStop: sleep
15`, subsequent readyz probes return `503` for ~15 seconds while
in-flight requests drain, and *then* the process exits. Chapter 04's
"The trick: readyz has to flip *before* the process starts declining
connections" is that 15-second window.

**Checkpoint.** You have now implemented chapter 04's full lifecycle:
**startup** (`lifespan` enter, slow load), **warm-up** (the comment in
`lifespan` after `time.sleep`; a real service does a dummy prediction
here), **serving** (healthz=200, readyz=200), **drain** (SIGTERM → flag
flips → readyz=503 so the LB stops sending → in-flight requests finish
→ process exits), and **exit** (`lifespan` after `yield`). Every stage
has a visible signal; every signal maps onto an orchestrator behaviour.

### Step 6 — Write down the one-line "probe budget" (≈ 5 min)

Chapter 11 ("The one-page handoff document") makes the runbook's
`Startup` section the ML engineer's responsibility: "cold start takes
≤ X s; recommended startupProbe `initialDelay=A, periodSeconds=B,
failureThreshold=C` (A + B×C = startup budget)." The exercise asks
for this sentence in a full RUNBOOK. The lab closes with the one-line
form.

For the `LOAD_SECONDS=20` service in this lab, a defensible probe
configuration is:

- `livenessProbe: path=/healthz, periodSeconds=10, failureThreshold=3`
  — kills the pod only if the process stops responding for 30 s, well
  past any legitimate load.
- `readinessProbe: path=/readyz, periodSeconds=5, failureThreshold=2`
  — removes from LB inside 10 s of a problem, puts it back on recovery.
- `startupProbe: path=/readyz, periodSeconds=5, failureThreshold=8`
  — gives up to 40 s for the first successful readiness before the
  readinessProbe takes over, comfortably above the 20 s load.

Write these three lines in your lab directory's `NOTES.md`:

```markdown
# lab-01 — probe budget

Cold start: ~20 s (LOAD_SECONDS, from the lab).
- livenessProbe:  periodSeconds=10, failureThreshold=3
- readinessProbe: periodSeconds=5,  failureThreshold=2
- startupProbe:   periodSeconds=5,  failureThreshold=8   (40 s budget)
```

**Checkpoint.** You now have the three-line answer to chapter 11's
"Startup" and "Shutdown" sections of the RUNBOOK, derived from a number
you measured in step 1, not guessed at. Every parameter has a reason;
every reason comes from a chapter-04 distinction.

## Expected final artefact

A lab directory containing:

- `app.py` — the two-mode FastAPI service (single / split), with the
  SIGTERM handler.
- `probe_loop.sh` — the orchestrator simulator.
- `NOTES.md` — the three-line probe budget.

Running the `single` mode reproduces the CrashLoopBackOff cycle inside
~6 seconds; running the `split` mode runs cleanly to a `200/200` state
at t≈20 and never triggers the "RESTART" warning on the liveness loop.

## Final checkpoint

Without re-reading chapter 04, answer:

1. In step 3, the probe loop logged "RESTART" twice *before* the model
   finished loading. Which specific chapter-04 pitfall does this
   reproduce, and which **two** of the four Kubernetes probe knobs
   (`initialDelaySeconds`, `periodSeconds`, `timeoutSeconds`,
   `failureThreshold`) are responsible for setting the "when" of that
   decision?
2. In step 4, you ran two probe loops but neither triggered a *restart*
   even though one of them hit `fails=3/3` before t=20. State in your
   own words what an orchestrator does differently in response to a
   failing **liveness** probe versus a failing **readiness** probe, and
   why that difference is the whole reason chapter 04 wants them
   separate.
3. The `_shutting_down` flag in step 5 is flipped by a signal handler
   and read by `/readyz`. Why does chapter 04 insist that readyz must
   return `503` *before* the process starts declining connections, and
   what two configuration knobs (one gunicorn, one Kubernetes) keep the
   process alive long enough for that `503` to be noticed by the load
   balancer?
4. For a service whose model takes ~45 seconds to load, write a
   three-line `startupProbe` + `livenessProbe` + `readinessProbe`
   configuration that chapter 04 would call defensible. Explain each
   number as either "matches the load time" or "survives a transient
   blip without pageing".

If you can answer all four, chapter 04's liveness/readiness distinction
is internalised and you are ready to apply it on real serving services
in exercise-01 (where you add the two endpoints correctly from the
start) and later in the RUNBOOK deliverable chapter 11 asks for.

## Where to go next

- **`exercise-01` ("FastAPI prediction service")** turns the two
  endpoints from this lab into the real `/healthz` and `/readyz` on a
  model-serving service, alongside pydantic schemas, structured logs,
  and a test suite.
- **`exercise-02` ("Dockerise the service")** wires a Docker
  `HEALTHCHECK` to `/readyz` with `--start-period` matching the model
  load time — the same probe-budget conversation from step 6 of this
  lab, at the container layer.
- **`lab-02` ("`async def` with a blocking predict call")** moves from
  "is the service ready?" to "is the service serving concurrently?" and
  reproduces the chapter 03 / chapter 09 event-loop-deadlock pattern.
- **`lab-03` ("Reading the latency-throughput curve: arrival-rate vs
  VUs")** takes a ready service and measures its capacity, reproducing
  chapter 10's load-testing pitfall around which k6 executor to use.
