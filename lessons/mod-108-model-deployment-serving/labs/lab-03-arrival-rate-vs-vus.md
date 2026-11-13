# lab-03 — Reading the latency-throughput curve: arrival-rate vs VUs hides saturation

> **Duration:** ~35–45 minutes.
> **Companion chapters:** [09 — Latency, throughput, and concurrency](../09-latency-throughput-and-concurrency.md), [10 — Load testing and tuning](../10-load-testing-and-tuning.md).

## Objective

Reproduce chapter 10's pitfall #4 — "VU-based generators hide saturation
because each virtual user waits for the previous response before sending
the next; use `ramping-arrival-rate` instead" — on a tiny FastAPI service
whose "model" is a controllable sleep. By the end you will have (a) a
FastAPI service with a configurable per-request compute time and a
bounded number of worker threads so saturation is forced at a specific
QPS, (b) run the **same workload** under k6's `ramping-vus` executor and
its `ramping-arrival-rate` executor, (c) watched the two produce
**entirely different latency-throughput curves** — one flat all the way
up, the other hockey-sticking at the knee — (d) identified the
saturation QPS from the arrival-rate run and the "comfortable operating
budget" (25–50% margin, per chapter 10), and (e) written down the one-
line defensible capacity sentence chapter 09 asked for.

The simulated "prediction" is a plain `time.sleep` — an uncontended,
GIL-releasing 50 ms call. That gives the service a predictable
throughput ceiling (1 worker × 20 req/s per handler slot), so the
arrival-rate executor hits saturation at a predictable point and the
VU-executor's hiding behaviour is visible.

This lab is deliberately narrower than **exercise-04** (which builds the
full load-test rig, tunes a lever, writes a REPORT.md, and adds a CI
regression gate). The lab is the isolation chamber where you *see* one
specific executor-choice decision cause saturation to appear or
disappear; the exercise then bakes the correct executor into a real
capacity story.

## Prerequisites

- Python 3.11+, `fastapi` and `uvicorn` installed.
- `k6` installed locally (binary, Homebrew, or the `grafana/k6` Docker
  image). Installation instructions:
  <https://k6.io/docs/get-started/installation/>.
- Chapters 09 (the "Little's Law" and "What to measure" sections) and
  10 (the "Where to run the load", "Producing the latency-throughput
  curve", and "Common pitfalls" sections) read through once.
- A CPU is enough. Run the lab on a machine quiet enough that CPU
  variance doesn't dominate the measurement (chapter 10's "the
  generator is the bottleneck" pitfall applies here too — close the
  browser).

## Setup

Create a lab directory and four files:

```bash
mkdir -p lab-03-arrival-rate-vs-vus && cd lab-03-arrival-rate-vs-vus
touch app.py vu.js arrival.js analyse.py NOTES.md
```

Everything below goes in those files. You will run the server in one
terminal and k6 in another.

## Steps

### Step 1 — A service with a predictable throughput ceiling (≈ 5 min)

The "model" is a 50 ms sleep. The service is sync `def`, one gunicorn
worker, and we constrain FastAPI's threadpool to a small size so the
single-process throughput ceiling is calculable in advance.

Put this in `app.py`:

```python
import asyncio
import os
import time

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

PREDICT_SECONDS = float(os.getenv("PREDICT_SECONDS", "0.05"))

# Cap the sync-handler threadpool so we have a known capacity.
# FastAPI's default is ~40; 10 makes the ceiling obvious.
from anyio import to_thread
to_thread.current_default_thread_limiter().total_tokens = int(
    os.getenv("THREAD_LIMIT", "10")
)

app = FastAPI()


@app.post("/predict")
def predict():
    time.sleep(PREDICT_SECONDS)
    return {"score": 0.42}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
```

Launch:

```bash
PREDICT_SECONDS=0.05 THREAD_LIMIT=10 uvicorn app:app \
    --host 127.0.0.1 --port 8080 --workers 1
```

**Expected output.** One uvicorn worker, `Application startup complete.`

The theoretical ceiling is `THREAD_LIMIT / PREDICT_SECONDS = 10 / 0.05 =
200 req/s` (chapter 09's Little's Law: `λ_max = L / W`, where `L` is
the concurrency the handler permits and `W` is the per-request time).
Below 200 req/s the service has spare threadpool slots; above 200 req/s
requests queue. The exact number will vary with per-request overhead;
160–200 req/s on a laptop is typical.

**Checkpoint.** `curl -s http://127.0.0.1:8080/healthz` returns
`{"status":"ok"}`. A single `curl -X POST http://127.0.0.1:8080/predict`
returns in ~50 ms. The ceiling is the lab's whole point; write it down
in `NOTES.md`:

```markdown
# lab-03 — setup

PREDICT_SECONDS = 0.05  (50 ms simulated inference)
THREAD_LIMIT    = 10    (handler concurrency cap)

Theoretical saturation (Little's Law): λ_max ≈ L / W = 10 / 0.05 = 200 req/s
```

### Step 2 — Warm the service (≈ 2 min)

Chapter 10 reproducibility recipe: warm the workers before measurement.
A short low-QPS run does it. In a second terminal:

```bash
k6 run - <<'EOF'
import http from 'k6/http';
export const options = { vus: 5, duration: '20s' };
export default function () {
  http.post('http://127.0.0.1:8080/predict', '{}',
    { headers: { 'content-type': 'application/json' } });
}
EOF
```

**Expected output.** k6 summary showing `http_req_duration` p50 around
50 ms and no failures. Ignore the numbers — the point is the workers
are now warm.

### Step 3 — The VU executor: saturation is invisible (≈ 10 min)

Chapter 10 pitfall #4: `ramping-vus` means "N virtual users, each
waiting for the previous response before sending the next". When the
service slows down, the VUs slow down too, and the latency curve stays
flat even past the service's real capacity. Reproduce it.

Put this in `vu.js`:

```javascript
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  scenarios: {
    ramp_vus: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '20s', target:  10 },
        { duration: '20s', target:  40 },
        { duration: '20s', target: 100 },
        { duration: '20s', target: 200 },
        { duration: '20s', target: 400 },
      ],
      gracefulRampDown: '0s',
    },
  },
};

export default function () {
  const r = http.post(
    'http://127.0.0.1:8080/predict',
    JSON.stringify({ seed: Math.random() }),
    { headers: { 'content-type': 'application/json' } },
  );
  check(r, { 'status 200': (res) => res.status === 200 });
  // NOTE: no sleep. Each VU immediately sends the next request as
  // soon as the previous one returns. That is the self-throttling
  // behaviour that chapter 10 pitfall #4 warns about.
}
```

Run it:

```bash
k6 run --summary-trend-stats "min,avg,med,p(50),p(95),p(99),max" \
       --out json=results-vu.json vu.js
```

**Expected output.** k6 climbs through 10 → 40 → 100 → 200 → 400 VUs
over 100 seconds. The summary at the end shows something like:

```
http_req_duration: avg=~300ms  med=~250ms  p(95)=~500ms  p(99)=~700ms
http_reqs:         ~16000  (~160/s)
```

The throughput **plateaus** at ~160–200 req/s — exactly the service's
ceiling — but k6's output does not visibly announce "I hit saturation
at QPS = 200". It just reports an average latency that creeps up as
VUs climb. There is **no knee**. If you only read "p95 = 500 ms", you
might conclude "the service is fine, just a bit slow at this VU count"
— and never realise the service has been *queueing* since stage 3.

Save a VU-stage summary in `NOTES.md`:

```markdown
## VU executor (vu.js)

Stages: 10 → 40 → 100 → 200 → 400 VUs
Overall p95 ≈ ___ ms
Overall throughput ≈ ___ req/s
```

**Checkpoint.** The number you want from a load test — "what QPS can
this service sustain at p95 ≤ X ms?" — is **not** derivable from the
VU run. All you see is a monotonically creeping average. You know the
service has a ceiling (step 1 predicted it); the VU executor hides it.
This is chapter 10 pitfall #4 made visible.

### Step 4 — The arrival-rate executor: saturation becomes a cliff (≈ 10 min)

Chapter 10's prescription: use `ramping-arrival-rate` so the generator
keeps injecting N requests per second **regardless of how slow the
service becomes**. Now the queue depth inside the service grows under
saturation; latency hockey-sticks; the knee is visible.

Put this in `arrival.js`:

```javascript
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  scenarios: {
    ramp_arrival: {
      executor: 'ramping-arrival-rate',
      startRate: 20,
      timeUnit: '1s',
      preAllocatedVUs: 400,          // more than enough so VU availability is not the limit
      maxVUs: 800,
      stages: [
        { duration: '20s', target:  50 },
        { duration: '20s', target: 100 },
        { duration: '20s', target: 150 },
        { duration: '20s', target: 200 },
        { duration: '20s', target: 300 },
        { duration: '20s', target: 400 },
      ],
    },
  },
};

export default function () {
  const r = http.post(
    'http://127.0.0.1:8080/predict',
    JSON.stringify({ seed: Math.random() }),
    { headers: { 'content-type': 'application/json' } },
  );
  check(r, { 'status 200': (res) => res.status === 200 });
}
```

Run it:

```bash
k6 run --summary-trend-stats "min,avg,med,p(50),p(95),p(99),max" \
       --out json=results-arrival.json arrival.js
```

**Expected output.** k6 injects 50, 100, 150, 200, 300, 400 requests per
second across the six 20-second stages. The summary at the end shows
p95 dramatically higher than the VU run, because the queue depth in the
service's threadpool grew unboundedly above 200 QPS.

The stage-by-stage breakdown is where the shape lives — that's what
step 5 extracts.

### Step 5 — Bin requests by stage and read the curve (≈ 10 min)

k6's JSON output is per-request; the latency-throughput curve is a
per-target-QPS aggregation. Put this in `analyse.py`:

```python
import argparse
import json
import statistics
from pathlib import Path

# The target-QPS schedule from arrival.js. Each stage ran for 20 s.
STAGES = [
    (0.0,   20.0,  50),
    (20.0,  40.0, 100),
    (40.0,  60.0, 150),
    (60.0,  80.0, 200),
    (80.0, 100.0, 300),
    (100.0, 120.0, 400),
]


def main(path: str) -> None:
    samples: list[tuple[float, float]] = []       # (elapsed_s, duration_ms)
    t0: float | None = None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") != "Point":
            continue
        if row.get("metric") != "http_req_duration":
            continue
        t = row["data"]["time"]           # ISO-8601
        # Simple seconds-since-first-sample
        secs = _epoch(t)
        if t0 is None:
            t0 = secs
        samples.append((secs - t0, float(row["data"]["value"])))

    print(f"{'target_qps':>12} {'n':>8} {'p50_ms':>8} {'p95_ms':>8} "
          f"{'p99_ms':>8} {'max_ms':>8}")
    for lo, hi, target in STAGES:
        bucket = [d for (t, d) in samples if lo <= t < hi]
        if not bucket:
            continue
        bucket.sort()
        p50 = statistics.median(bucket)
        p95 = bucket[int(0.95 * (len(bucket) - 1))]
        p99 = bucket[int(0.99 * (len(bucket) - 1))]
        mx  = bucket[-1]
        print(f"{target:>12} {len(bucket):>8} {p50:>8.1f} {p95:>8.1f} "
              f"{p99:>8.1f} {mx:>8.1f}")


def _epoch(iso: str) -> float:
    # k6 ISO-8601 to epoch seconds. datetime is a stdlib only, no deps.
    from datetime import datetime
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    main(args.path)
```

Run it on both outputs:

```bash
python analyse.py results-arrival.json > arrival-curve.txt
python analyse.py results-vu.json      > vu-curve.txt       # for comparison (stages won't line up)
cat arrival-curve.txt
```

**Expected output.** The arrival-rate table should look like:

```
  target_qps        n   p50_ms   p95_ms   p99_ms   max_ms
          50      995     51.2     59.3     68.7     94.1
         100     1988     52.4     64.1     78.2    120.8
         150     2986     54.7     71.3     95.4    168.9
         200     3980     85.2    310.4    520.1    780.4
         300     5970   1500.2   3200.8   3800.2   4200.3
         400     7920   3000.7   6100.4   7800.1   8500.6
```

The **knee** — where p95 stops being near-constant and starts climbing —
sits between 150 and 200 target QPS, exactly where step 1's Little's-Law
ceiling predicted it would. That is the latency-throughput curve the VU
run could not show you.

Add the table to `NOTES.md` and annotate the knee:

```markdown
## Arrival-rate executor (arrival.js)

Stages: 50 → 100 → 150 → 200 → 300 → 400 req/s target
Knee: between 150 and 200 QPS (p95 jumps from ~70 ms to ~300 ms)

| target_qps | p50_ms | p95_ms | p99_ms |
|-----------:|-------:|-------:|-------:|
|         50 |   51.2 |   59.3 |   68.7 |
|        100 |   52.4 |   64.1 |   78.2 |
|        150 |   54.7 |   71.3 |   95.4 |
|        200 |   85.2 |  310.4 |  520.1 |
|        300 | 1500.2 | 3200.8 | 3800.2 |
|        400 | 3000.7 | 6100.4 | 7800.1 |
```

**Checkpoint.** Compare this table with the VU run's summary from step 3:
the VU summary says "p95 ≈ 500 ms, throughput ≈ 160 req/s" and gives you
one number; the arrival-rate table shows the exact shape of the curve
and identifies the knee. **Both are honest measurements of the same
service.** They disagree because they are measuring *different things* —
the VU run measures "how slowly does the service respond to a fixed
number of eager clients?", the arrival-rate run measures "what happens
to the service under a fixed arrival rate?". The latter is what
autoscaling cares about; the former is not.

### Step 6 — Report the one-line defensible capacity sentence (≈ 5 min)

Chapter 09's "Throughput, precisely": every claim about capacity has to
name a latency SLA. Chapter 10's "Producing the latency-throughput
curve": pick a QPS **a comfortable margin (25–50%) below the knee** so
bursts don't push the service into the steep zone.

From the table above:
- Knee (where p95 leaves the flat region): ~175 QPS.
- Comfortable margin at 25%: 175 × 0.75 ≈ 130 QPS.
- Comfortable margin at 50%: 175 × 0.50 ≈ 90 QPS.

For a service with a p95 budget of 100 ms, the defensible sentence is:

> This service sustains **≈ 130 requests per second at p95 ≤ 100 ms**
> on a 1-worker uvicorn instance with THREAD_LIMIT=10 and
> PREDICT_SECONDS=0.05.

Write that as the final line in `NOTES.md`:

```markdown
## Defended capacity

"This service sustains ≈ 130 req/s at p95 ≤ 100 ms on 1 worker,
THREAD_LIMIT=10, PREDICT_SECONDS=0.05."

That is the sentence chapter 09 said every service must defend before
the handoff; the arrival-rate curve in step 5 is what backs it up.
Without arrival-rate, this sentence cannot be derived — the VU
executor's summary gives a different, uninterpretable number.
```

**Checkpoint.** You now have the exact deliverable chapter 11
("Capacity") asks for: a one-line capacity statement, derived from a
reproducible load-test script, backed by a per-QPS-bin table, with the
knee identified. Chapter 11's runbook template puts this sentence
directly into the "Capacity" section; the arrival-rate script that
produced it belongs in the repo for the platform team's HPA
configuration. The lab has done exactly the "defensible latency-
throughput budget" step the module introduction promised.

## Expected final artefact

A lab directory containing:

- `app.py` — the sync `/predict` service with configurable predict time
  and thread-pool cap.
- `vu.js` — k6 ramping-vus script (the anti-pattern).
- `arrival.js` — k6 ramping-arrival-rate script (the correct shape).
- `analyse.py` — a small per-stage binner that turns k6 JSON into a
  per-QPS percentile table.
- `results-vu.json`, `results-arrival.json` — the raw outputs.
- `vu-curve.txt`, `arrival-curve.txt` — the generated tables.
- `NOTES.md` — setup, both summaries, the arrival-rate table with the
  knee annotated, and the one-line defended capacity sentence.

Running the sequence `uvicorn … --workers 1 → k6 run vu.js → k6 run
arrival.js → python analyse.py …` should reproduce the shape of the
tables regardless of the exact hardware — the absolute knee number
moves with the CPU, but the "VU hides saturation, arrival-rate reveals
it" shape is universal.

## Final checkpoint

Without re-reading chapter 10, answer:

1. In step 3, the VU executor's summary reported `throughput ≈ 160
   req/s` but never produced a "knee QPS". State in your own words what
   the VU executor does when the service slows down, and why that
   behaviour makes the saturation point invisible even though the
   service clearly has a ceiling.
2. In step 5, the arrival-rate table showed p95 jumping from ~70 ms at
   150 QPS to ~310 ms at 200 QPS. Little's Law (chapter 09) is
   `L = λ × W`. Given the service's `L` cap (THREAD_LIMIT) is 10 and
   the per-request compute `W` is 50 ms, what is the theoretical `λ_max`,
   and does the observed knee match it? If not, what does the gap tell
   you about the service (per-request overhead, queue mechanics, kernel
   scheduler, etc.)?
3. The defended capacity sentence in step 6 picked a QPS 25% below the
   knee. State one chapter-10 reason this margin matters — i.e., what
   would happen in production if you set the HPA target at the knee
   itself and a legitimate 10% traffic burst arrived?
4. For a real CPU-bound ML service whose `/predict` takes 20 ms and
   runs with 4 workers on a 4-core node, use Little's Law to predict a
   first-order saturation QPS. If a load test (arrival-rate) reports a
   knee at 60% of your prediction, name one chapter-09 candidate for
   the missing capacity (feature retrieval, GIL contention, async
   mis-handler, GC pauses, ...) and state how you would verify it from
   the per-step log timings chapter 05 asked you to emit.

If you can answer all four, chapter 10's load-testing discipline — the
right executor, the knee reading, the defended capacity sentence — is
internalised and you are ready to run the full rig in exercise-04 with
a real model behind the service.

## Where to go next

- **`exercise-04` ("Load test and tune")** takes a real containerised
  service, builds the full load-test rig, tunes one lever
  (workers / batch size / orjson / etc.), writes a `REPORT.md`, and
  wires a CI regression gate — all using the `ramping-arrival-rate`
  executor from this lab.
- **`exercise-03` ("Batch vs online inference")** uses the capacity
  budget from this lab's shape to argue for one of chapter 08's three
  hybrid patterns per product scenario in its `DECISIONS.md`
  deliverable.
- **`lab-01` ("Liveness vs readiness")** covers the chapter-04 probe
  semantics the service you load-tested here must also satisfy.
- **`lab-02` ("`async def` with a blocking predict call")** is the
  chapter-03 / chapter-09 companion: the handler shape has to be right
  before the load-test numbers are interpretable at all.
