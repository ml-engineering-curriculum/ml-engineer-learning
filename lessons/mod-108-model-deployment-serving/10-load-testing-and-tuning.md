# 10 — Load testing and tuning

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01–09.

## Motivation

The service passes smoke tests. Its p95 on a single curl
call is 25 ms. Ship it, right?

No. Single-request timings only tell you about a warm,
uncontended, best-case service. Real callers arrive in
parallel, from many sockets, at variable rates, with
occasional bursts. The service you tested at rest may
degrade badly under load — through GIL contention, worker
saturation, or a memory pressure spike that triggers GC —
and you cannot know until you have measured it.

**Load testing** is the discipline of driving synthetic
traffic at the service and observing what happens. Done
well it produces the *latency-throughput curve* from
chapter 09, identifies the saturation point, and gives you
a number to defend when the platform team asks "what QPS
can this service sustain at p95 ≤ 100 ms?".

This chapter covers how to run the test, how to read the
numbers, and what to tune when the numbers are wrong.

## The three tools you might reach for

All produce similar numbers; the differences are ergonomics.

### `wrk` — the fast, minimal HTTP benchmark

```
wrk -t4 -c100 -d60s --latency \
    -s post.lua \
    http://localhost:8080/predict
```

- **`-t4`** — 4 threads generating load.
- **`-c100`** — 100 open connections.
- **`-d60s`** — run for 60 seconds.
- **`--latency`** — print per-percentile latency histogram.
- **`-s post.lua`** — a Lua script that constructs the POST
  body.

Written in C; can saturate a service faster than the
service can respond, which is what you want in a load
generator (any latency you see is the service, not the
generator).

Fine for a static POST body. Awkward if requests need to
vary (different payloads per request, per-request auth).

### `k6` — the scriptable, modern choice

```javascript
// load.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 20 },   // ramp to 20 VUs
    { duration: '2m',  target: 100 },  // ramp to 100 VUs
    { duration: '30s', target: 0 },    // ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<100'],  // fail the test if p95 > 100 ms
  },
};

export default function () {
  const body = JSON.stringify({
    age: Math.floor(Math.random() * 80) + 18,
    plan_tier: 'pro',
    months_since_signup: Math.floor(Math.random() * 36),
  });
  const res = http.post('http://localhost:8080/predict',
                         body, { headers: { 'content-type': 'application/json' } });
  check(res, { 'status 200': (r) => r.status === 200 });
  sleep(0.1);
}
```

Run with `k6 run load.js`. JavaScript scripting; ramp
stages; threshold assertions that make the test *fail* if
p95 exceeds a target; JSON output for post-processing.

**Recommendation:** k6 is the default for this module. It
composes well with CI (`k6 run --out json=results.json`),
scripts realistic traffic patterns, and its threshold
feature turns a load test into a *pass/fail* gate.

### Locust — Python-first

```python
# locustfile.py
import random

from locust import HttpUser, between, task


class PredictUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={
                "age": random.randint(18, 80),
                "plan_tier": random.choice(["free", "pro", "enterprise"]),
                "months_since_signup": random.randint(0, 36),
            },
        )
```

Run with `locust -H http://localhost:8080`. Web UI at
`localhost:8089` lets you dial the number of "users" and
watch a live latency chart.

Reach for Locust when the scenario is complex (login →
call feature endpoint → call predict), when non-engineers
need to run it, or when Python is easier than JavaScript
for your team. On a per-request basis it is slower to
generate load than wrk / k6 — the Python generator itself
can bottleneck before the service does.

## Where to run the load

**Not from your laptop, over the internet.** Home network
latency and jitter contaminate the measurement.

**Ideally, from a machine adjacent to the service.** Same
region, same availability zone, same VPC if it's a real
deployment. For local development that means running the
generator and the container on the same host — chapter 07's
`docker compose` is fine.

**Rule the generator out as the bottleneck.** If `wrk` is
saturating a CPU core on the generator machine, its latency
numbers include queueing *inside the generator*. Watch
`top`/`htop` on the generator during the run.

**Warm the service first.** Send 100 requests before you
start the measurement. Chapter 04's warm-up on startup does
most of it; a warm-up phase in the load test finishes the
job.

## Reading the output

k6's output is representative:

```
     data_received..............: 1.7 MB   28 kB/s
     data_sent..................: 2.4 MB   40 kB/s
     http_req_blocked...........: avg=1.4µs   min=1µs   med=1µs   max=210µs   p(90)=2µs   p(95)=3µs
     http_req_connecting........: avg=0s      min=0s    med=0s    max=0s      p(90)=0s    p(95)=0s
     http_req_duration..........: avg=41.2ms  min=8ms   med=32ms  max=402ms   p(90)=88ms  p(95)=112ms
       { expected_response:true }: avg=41.2ms  min=8ms   med=32ms  max=402ms   p(90)=88ms  p(95)=112ms
     http_req_failed............: 0.00%    ✓ 0        ✗ 14320
     http_req_receiving.........: avg=95µs    min=15µs  med=42µs  max=3.9ms   p(90)=112µs p(95)=180µs
     http_req_sending...........: avg=32µs    min=6µs   med=18µs  max=1.2ms   p(90)=42µs  p(95)=68µs
     http_req_waiting...........: avg=41.1ms  min=8ms   med=32ms  max=401ms   p(90)=87ms  p(95)=111ms
     iterations.................: 14320    238/s
```

Read from the bottom:

- **iterations 238/s.** The achieved throughput.
- **http_req_duration p(95) 112 ms.** 95% of requests
  finished within 112 ms.
- **http_req_waiting 111 ms** vs **http_req_duration
  112 ms.** Almost all of the latency is the server
  waiting to respond — network and send/receive are
  negligible. That is what you want; if `receiving` was
  40 ms, the client is the bottleneck.
- **http_req_failed 0.00%.** No 4xx or 5xx.
- **max 402 ms.** One request in that run was very slow.
  Investigate; that is a candidate p99.9 problem.

Compare against `wrk`'s output:

```
Running 1m test @ http://localhost:8080/predict
  4 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    41.20ms   28.60ms   410.03ms   79.11%
    Req/Sec    58.20     10.44     92.00      65.00%
  Latency Distribution
     50%   32.10ms
     75%   58.50ms
     90%   88.90ms
     99%  158.20ms
  13980 requests in 1.00m, 1.68MB read
Requests/sec:    233.00
Transfer/sec:    28.72KB
```

Same story, different presentation. Notice the p99 that
k6's summary omitted; ask for it explicitly.

## Producing the latency-throughput curve

The single measurement above is one data point. The
*curve* is many points, one per target QPS. In k6:

```javascript
export const options = {
  scenarios: {
    ramping: {
      executor: 'ramping-arrival-rate',
      startRate: 10,             // start at 10 req/s
      timeUnit: '1s',
      preAllocatedVUs: 200,
      maxVUs: 500,
      stages: [
        { duration: '30s', target: 50 },
        { duration: '30s', target: 100 },
        { duration: '30s', target: 200 },
        { duration: '30s', target: 400 },
        { duration: '30s', target: 800 },
      ],
    },
  },
};
```

**Key detail:** use `ramping-arrival-rate`, not `ramping-
vus`. Arrival rate means "start N requests per second
regardless of how slow the service becomes"; VUs means "N
virtual users each waiting for the previous response". The
first shape is what you want to see saturation; the second
hides saturation behind self-throttling.

Plot p95 (y) vs. QPS (x). The curve is flat below saturation
and hockey-sticks upward once queueing starts. **The knee of
the curve is your capacity budget.** Report a QPS a
comfortable margin below the knee (25–50%) so bursts do not
push the service into the steep zone.

## Interpreting where the time goes

The service p95 climbed from 30 ms to 300 ms as QPS went
from 100 to 500. Which of chapter 09's steps is
responsible? Three sources of evidence:

1. **The per-step timings from chapter 05's logs.** Chart
   `inference_ms`, `feature_ms`, and `total_ms` over the
   load-test window. If `total_ms` grew but `inference_ms`
   stayed flat, the model is not the bottleneck.
2. **CPU / memory / GC on the service.** `docker stats` or
   `kubectl top pod` during the test. If CPU is at 100%
   before p95 explodes, add workers or bigger nodes. If
   memory is climbing, look for a leak. If GC is
   noticeable — pauses correlate with p99 spikes.
3. **The gunicorn / uvicorn worker health.** Are workers
   crashing under load and restarting? `gunicorn` prints
   worker-fail log lines. Every restart drops in-flight
   requests.

The instinct to "add workers" is often wrong. Add workers
only when CPU is saturated *and* memory allows. Otherwise
you are creating more queueing rather than more capacity.

## Tuning knobs, ranked by impact

Roughly ordered by "what usually matters most":

### 1. Number of workers

`gunicorn -w N`. Start with N = cores. Measure the
latency-throughput curve at N, N−1, N+2. The right N
maximises throughput at the target latency; it is not
always the largest N.

### 2. Batch size (if applicable)

For endpoints that accept `/predict:batch`, the largest
per-request batch you can serve amortises fixed per-request
overhead. Chapter 08's chunk-size search applies here too.
If callers send batches of 1, they pay the overhead every
time; educate them or micro-batch server-side.

### 3. Model complexity

The number that has moved most in real projects. A model
with 30% fewer trees, or half the layers, may be nearly as
accurate and 3× faster to serve. Chapter 09 warned that
model latency is often *not* the bottleneck; measure first,
but when it is, this is where the leverage lives.

### 4. Feature computation

If `feature_ms` is a large fraction of total, the model
isn't the problem. Pre-compute (chapter 08's hybrid
pattern 2), cache, or move the feature store closer.

### 5. Serialisation

Pydantic v2 is fast; JSON encoding a small dict is
essentially free. If your response contains a large
embedding vector, serialisation cost may show up; consider
`orjson` or a binary format (protobuf, MessagePack).

### 6. Framework choice

`torch` on CPU with the `+cpu` wheel; scikit-learn with
`n_jobs=1` at request time (per-worker parallelism is what
the process model is for); `onnxruntime` for many
scikit-learn / small-DL models is measurably faster than
the native framework.

### 7. Base image and kernel

Rarely worth the effort at this level. If you get there,
JEMalloc / TCMalloc for allocator pressure, `PYTHONMALLOC=
malloc`, thread pinning. The gains are single-digit
percentages; the tuning above usually offers 10× more.

## The reproducible load-test recipe

The load test is only useful if the numbers move as
expected when *something* changes. To make it reproducible:

- **Same generator machine, same service host, same
  network path** across runs.
- **Same load profile** — arrival rate, ramp shape, test
  duration, payload distribution.
- **Fresh service, warm workers.** Restart the service
  before each test, then send warm-up traffic before
  starting the measurement window.
- **Same measurement window** — don't include the ramp-up
  in the reported percentiles.
- **Same data plane** — the artifact under test, its
  version, the code SHA, and the environment (workers,
  threads, env vars) are recorded alongside the results.

An entry in the repo's `load-tests/results/` directory,
committed with the code change, gives future you the
counterfactual.

## An end-to-end example: finding the knee

```bash
# Start the service with 4 workers
docker run -d --rm --name svc -p 8080:8080 \
    -e WORKERS=4 churn-service:0.1.0

# Warm-up
k6 run --duration 30s --vus 5 warmup.js

# Baseline curve
k6 run --out json=results-w4.json load-curve.js

# Compare against 8 workers
docker rm -f svc
docker run -d --rm --name svc -p 8080:8080 \
    -e WORKERS=8 churn-service:0.1.0
k6 run --duration 30s --vus 5 warmup.js
k6 run --out json=results-w8.json load-curve.js

# Post-process — extract p95 per QPS bucket
python analyse.py results-w4.json results-w8.json
```

The `analyse.py` script reads the JSON, bins requests by
target QPS, computes p95 per bin, and prints a table:

```
Target QPS |  w=4 p95  |  w=8 p95
        50 |     22 ms |     22 ms
       100 |     28 ms |     26 ms
       200 |     35 ms |     31 ms
       400 |     92 ms |     45 ms
       800 |    412 ms |    120 ms
      1200 |   timeout |    380 ms
```

Reading this table:

- **w=4** knees around 400 QPS; safe operating budget is
  ~200 QPS at p95 ≤ 100 ms.
- **w=8** knees around 800 QPS; safe operating budget is
  ~400 QPS at p95 ≤ 100 ms.

Doubling workers roughly doubled capacity — CPU-bound
service, no other bottleneck. If the increase had been
smaller (e.g., 1.3×), memory pressure or a shared
dependency would be capping the gain, and adding workers
past that point is wasted.

## Load testing in CI

Full load-curve runs are too slow for every PR. Two useful
CI gates:

- **A short "does p95 stay under X ms at Y QPS?" test** on
  every PR — 30 seconds of load, threshold assertion in k6
  that fails the build if p95 regresses.
- **A longer nightly load test** that produces the full
  curve and posts the summary to the PR as a comment on
  merge to main. Regression alerts fire if the knee moves
  the wrong way.

The point of CI load testing is *regression detection*, not
absolute performance validation. The absolute number
depends on the CI runner's CPU — the delta between
successive runs on the *same* runner is meaningful.

## Common pitfalls

1. **Testing from your laptop over Wi-Fi.** Latency and
   jitter are the network's, not the service's. Test
   locally against the container or from an adjacent
   machine.
2. **Reporting only average latency.** As chapter 09
   established, meaningless. Report percentiles.
3. **Skipping the warm-up.** First-request latency shows
   up in the percentiles and pollutes them. Send
   throw-away traffic before measurement.
4. **VUs instead of arrival-rate.** VUs self-throttle;
   saturation is invisible. Use arrival-rate.
5. **The generator is the bottleneck.** `htop` the
   generator; if it's at 100% CPU, add threads / add
   generator instances or switch to a faster tool.
6. **Adding workers past CPU saturation.** More context
   switching, no more throughput. Measure.
7. **`n_jobs=-1` inside a scikit-learn model at request
   time.** Every request spawns per-worker parallelism;
   they collide across workers. `n_jobs=1` at inference;
   let processes provide the parallelism.
8. **Changing multiple variables between runs.** You
   moved from 4 to 8 workers, changed the batch size, and
   upgraded torch — which caused the improvement? Change
   one thing at a time.
9. **Ignoring the tail.** A p50 of 20 ms and a p99.9 of
   30 s is a real incident waiting to happen. Sort by
   tail; investigate; instrument.
10. **No load-test artifact in the PR.** The reviewer has
    to trust your one-off measurement. Commit the k6
    script, the summary output, and the machine spec so
    the number is reproducible.

## Summary

- Load testing produces the latency-throughput curve;
  finding the knee is what turns a service into a
  capacity budget you can defend.
- Use an arrival-rate load generator (k6, wrk with a
  scripted arrival rate) — VU-based generators hide
  saturation.
- Read percentiles (p50, p95, p99), not the mean. Watch
  the max for the tail. Correlate against per-step
  timings from your logs.
- Tune from largest to smallest lever: workers → batch
  size → model complexity → feature retrieval →
  serialisation → framework → allocator. Measure
  between each change.
- Reproducibility is the whole point. Same generator,
  same profile, same data plane, results committed.
- CI: a short regression gate on every PR; a longer
  nightly curve.

## Where this goes next

Chapter 11 wraps the module: the handoff to the platform
team. What "prod-ready" means, what artifacts and numbers
you owe them, and where your responsibility ends and
theirs begins.
