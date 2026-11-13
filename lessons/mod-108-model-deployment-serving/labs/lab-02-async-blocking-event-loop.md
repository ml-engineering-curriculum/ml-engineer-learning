# lab-02 — `async def` with a blocking predict call: watch the event loop deadlock

> **Duration:** ~30–40 minutes.
> **Companion chapters:** [03 — Anatomy of a FastAPI prediction service](../03-fastapi-prediction-service.md), [09 — Latency, throughput, and concurrency](../09-latency-throughput-and-concurrency.md).

## Objective

Reproduce chapter 03's pitfall #3 — "`async def` with a blocking
prediction blocks the event loop; throughput collapses" — and chapter
09's three-axes concurrency framing, on a tiny FastAPI app with no model
at all. By the end you will have (a) built three variants of the same
"predict" handler — `def` (sync on FastAPI's threadpool), `async def`
(blocking the event loop), and `async def` + `run_in_threadpool` (the
chapter 03 bridge) — (b) hit each with 10 concurrent requests from a
small Python async client, (c) measured wall-clock elapsed time for the
whole batch of 10 and watched the blocking-async variant serialise to a
~10× regression, (d) measured p95 latency per variant and seen the
tail-latency amplification chapter 09 warned about, and (e) written
down the one-line decision rule from chapter 09's "Pattern matrix".

The simulated "prediction" is a plain `time.sleep(0.2)` — a CPU-bound
200 ms call that releases no GIL and awaits no I/O. That is exactly the
shape of `scikit-learn` / `XGBoost` / small CPU-`torch` inference as far
as the event loop is concerned, which is what makes the lab
representative without needing a trained model.

This lab is deliberately narrower than **exercise-01** (which builds a
real FastAPI prediction service for a real sklearn pipeline, including
pydantic schemas, warm-up, and tests). The lab is the isolation chamber
where you *see* the three concurrency models produce three very
different numbers on the same workload; the exercise then lets you pick
the right shape for your real handler the first time.

## Prerequisites

- Python 3.11+.
- `fastapi`, `uvicorn`, and `httpx` installed:
  `uv pip install fastapi uvicorn httpx` (or use pip inside a venv).
- Chapters 03 (the "Sync vs async route handlers" section) and 09 (the
  "Concurrency in Python — the three axes" section) read through once.
- A CPU is enough; nothing in this lab needs a GPU.

## Setup

Create a lab directory and three files:

```bash
mkdir -p lab-02-async-blocking && cd lab-02-async-blocking
touch app.py hammer.py NOTES.md
```

- `app.py` holds the three handler variants plus a mode switch.
- `hammer.py` is a small async client that sends N concurrent requests
  and prints wall-clock + p95.
- `NOTES.md` is where you record the numbers you measure.

## Steps

### Step 1 — Three handler variants, one mode switch (≈ 5 min)

Chapter 03's three shapes of handler, all serving the same workload.
Put this in `app.py`:

```python
import os
import time

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

# Simulated per-request CPU-bound work: a 200 ms blocking call.
# This is the shape scikit-learn / XGBoost / CPU torch all have from
# the event loop's point of view: no await, no I/O yield, just compute.
PREDICT_SECONDS = float(os.getenv("PREDICT_SECONDS", "0.2"))

# One of: "sync", "async_blocking", "async_threadpool".
MODE = os.getenv("MODE", "sync")

app = FastAPI()


def _fake_predict():
    time.sleep(PREDICT_SECONDS)                 # the "model"
    return {"score": 0.42}


# --- MODE = "sync" ------------------------------------------------------
# Chapter 03's recommended default for CPU-bound inference.
# FastAPI runs sync `def` handlers on a threadpool; concurrent requests
# overlap. GIL-releasing C extensions (numpy, sklearn, torch) scale here;
# pure Python compute does not (chapter 09's threads axis).
@app.post("/predict-sync")
def predict_sync():
    return _fake_predict()


# --- MODE = "async_blocking" -------------------------------------------
# Chapter 03 pitfall #3: `async def` with a blocking call and no await.
# The handler runs on the main asyncio event loop; `time.sleep(0.2)`
# does not yield; the loop is frozen for the whole 200 ms; no other
# request in this worker can progress until this one returns.
@app.post("/predict-async-blocking")
async def predict_async_blocking():
    return _fake_predict()


# --- MODE = "async_threadpool" -----------------------------------------
# Chapter 03's bridge: `async def` is correct only when you actually
# await something; wrap CPU work with `run_in_threadpool` so the heavy
# call runs off the event loop while the loop stays flowing.
@app.post("/predict-async-threadpool")
async def predict_async_threadpool():
    return await run_in_threadpool(_fake_predict)
```

Run the server with a **single worker** so you can see the event loop
effect clearly — multi-worker gunicorn spreads work across processes and
hides the single-loop pathology:

```bash
uvicorn app:app --host 127.0.0.1 --port 8080 --workers 1
```

**Expected output.** Uvicorn starts and the three endpoints respond. A
single `curl -X POST http://127.0.0.1:8080/predict-sync` returns `{"score":
0.42}` after ~200 ms. Same for the other two — the single-request latency
of all three is identical (~200 ms), which is chapter 09's point about
why mean latency on one request tells you nothing.

**Checkpoint.** All three variants compute the same answer with the same
single-request latency. The differences only appear under concurrency —
which is the whole lab.

### Step 2 — A tiny async hammer (≈ 5 min)

The load generator is one Python file so the measurement is
reproducible without k6 / wrk / Locust. It fires `N` requests
concurrently against the given URL, measures each one's latency, and
prints wall-clock elapsed plus a small percentile summary.

Put this in `hammer.py`:

```python
import argparse
import asyncio
import statistics
import time

import httpx


async def one(client: httpx.AsyncClient, url: str) -> float:
    t0 = time.perf_counter()
    r = await client.post(url, json={})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000.0


async def main(url: str, n: int) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        t0 = time.perf_counter()
        latencies = await asyncio.gather(*(one(client, url) for _ in range(n)))
        wall_ms = (time.perf_counter() - t0) * 1000.0
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * (n - 1))]
    p99 = latencies[int(0.99 * (n - 1))]
    mx  = max(latencies)
    print(f"url:           {url}")
    print(f"concurrency:   {n}")
    print(f"wall_ms:       {wall_ms:8.1f}      (time to complete ALL {n})")
    print(f"p50_ms:        {p50:8.1f}")
    print(f"p95_ms:        {p95:8.1f}")
    print(f"p99_ms:        {p99:8.1f}")
    print(f"max_ms:        {mx:8.1f}")
    print(f"throughput:    {n * 1000.0 / wall_ms:8.1f} req/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    asyncio.run(main(args.url, args.n))
```

**Checkpoint.** `python hammer.py http://127.0.0.1:8080/predict-sync --n 1`
prints a single-request summary with wall ≈ p50 ≈ ~200 ms. The baseline
is set.

### Step 3 — The sync baseline (≈ 5 min)

Chapter 03's recommended default: a sync `def` handler on FastAPI's
threadpool.

With the server running, in a separate terminal:

```bash
python hammer.py http://127.0.0.1:8080/predict-sync --n 10
```

**Expected output.** Wall-clock around 300–500 ms for all 10 requests —
much less than `10 × 200 ms = 2000 ms` because FastAPI's threadpool
(default max 40 worker threads) runs them concurrently. The p50 / p95 /
p99 columns will all be in the 200–400 ms range. Throughput will be
20–40 req/s on a laptop.

Record the row in `NOTES.md`:

```markdown
| handler                           | wall_ms | p50_ms | p95_ms | p99_ms | throughput |
|-----------------------------------|---------|--------|--------|--------|------------|
| sync `def` on threadpool          |   ...   |  ...   |  ...   |  ...   |    ...     |
```

**Checkpoint.** Sync `def` handlers are not single-threaded — FastAPI
runs them on a threadpool. Chapter 09's threads axis notes that pure
Python CPU-bound work does not parallelise under the GIL, but
`time.sleep` releases the GIL, so concurrent sleep *does* overlap. For
numpy / sklearn / torch, the C extension releases the GIL for the
heavy loop — the behaviour you see here generalises directly.

### Step 4 — The anti-pattern: `async def` blocks the loop (≈ 10 min)

Chapter 03 pitfall #3, made numerical.

```bash
python hammer.py http://127.0.0.1:8080/predict-async-blocking --n 10
```

**Expected output.** Wall-clock around **2000 ms** — the 10 requests
were serialised through the single event loop; each one blocked the
next. Throughput drops to ~5 req/s, roughly `1 / PREDICT_SECONDS`. The
latency distribution is particularly ugly: p50 around 1000 ms (the 5th
request waited for 4 before it ran), p95 around 2000 ms (the 10th
request waited for 9), and the max is close to `10 × 200 ms`.

Record the second row:

```markdown
| `async def` with blocking call    |   ~2000 |  ~1000 |  ~2000 |  ~2000 |    ~5      |
```

The ratio of blocking-async wall_ms to sync wall_ms should be between
4× and 10× on a laptop. Chapter 03's "throughput collapses" is literally
that factor.

**Checkpoint.** You just reproduced the single most common FastAPI-ML
performance regression. The handler looks correct — `async def` is "the
modern way" — and a single-request curl gives the same 200 ms you
expected. The pathology only appears under concurrency. That is why
exercise-01's acceptance criteria specifically ban this pattern and why
chapter 03 names it as pitfall #3.

### Step 5 — The chapter-03 bridge: `run_in_threadpool` (≈ 5 min)

Chapter 03's fix when you need `async def` for some reason (an awaited
I/O elsewhere in the handler, a shared context manager): keep the
`async def`, wrap the CPU call in `await run_in_threadpool(...)`. The
CPU work goes to a thread; the event loop stays flowing.

```bash
python hammer.py http://127.0.0.1:8080/predict-async-threadpool --n 10
```

**Expected output.** Wall-clock back to ~300–500 ms, within noise of the
sync baseline. Throughput back to 20–40 req/s. The p50 / p95 numbers
match the sync row.

Record the third row:

```markdown
| `async def` + `run_in_threadpool` |   ~300  |  ~200  |  ~400  |  ~450  |   ~30      |
```

**Checkpoint.** The three variants now show a clear ranking: sync `def`
≈ async `def` + threadpool ≫ async `def` with a blocking call. The fact
that the first two are essentially tied is chapter 03's "pick either,
but if you pick async, keep the event loop flowing".

### Step 6 — Tail latency: p99 under sustained load (≈ 5 min)

Chapter 09's tail-latency warning is harder to see with 10 requests;
re-run with 50 or 100 to let the tail develop.

```bash
python hammer.py http://127.0.0.1:8080/predict-sync             --n 100
python hammer.py http://127.0.0.1:8080/predict-async-blocking   --n 100
python hammer.py http://127.0.0.1:8080/predict-async-threadpool --n 100
```

**Expected output.** The sync and threadpool numbers still track; the
blocking-async run now shows p99 around **18 000–19 000 ms** — the 99th-
percentile request waited for 98 others ahead of it in the loop before
getting its 200 ms slice. The single-worker service's "throughput" at
100 concurrent is essentially `1 / PREDICT_SECONDS = 5 req/s` regardless
of how much traffic you throw at it.

That is Little's Law (chapter 09) in action: `L = λ × W`, so with `W =
0.2 s` and the event loop forcing effective `L = 1`, `λ` is capped at
`5 req/s`. Threads and workers are how you raise `L`; `async def` with
a blocking call forces `L = 1` and no number of workers raises it
because each worker has its own loop with the same problem.

Add the 100-request rows to `NOTES.md`:

```markdown
### N = 100

| handler                           | wall_ms | p50_ms | p95_ms | p99_ms | throughput |
|-----------------------------------|---------|--------|--------|--------|------------|
| sync `def` on threadpool          |   ...   |  ...   |  ...   |  ...   |    ...     |
| `async def` with blocking call    | ~20 000 | ~10 000| ~19 000| ~19 500|    ~5      |
| `async def` + `run_in_threadpool` |   ...   |  ...   |  ...   |  ...   |    ...     |
```

**Checkpoint.** The p99 column in the blocking-async row is the specific
number chapter 09 means by "your p99 is someone else's p50" (the tail-
latency amplifier). On a 10-way fan-out at a downstream caller, the
parent's p50 is the max of ten of these p99 values — so a service with
a 19 000 ms p99 is unusable long before anyone complains about the mean.

### Step 7 — Write down the one-line decision rule (≈ 5 min)

Chapter 09's "Pattern matrix" compresses the whole chapter into five
rows, each matching a service shape. For the service you built in this
lab — "CPU-bound prediction, no downstream I/O" — the matrix prescribes:

- **workers:** k (cores, or k-1)
- **threads:** 1
- **async?:** sync

Add the final block to `NOTES.md`:

```markdown
### Decision rule (chapter 09 "Pattern matrix")

For this lab's workload (CPU-bound predict, no downstream I/O):
  workers=k (CPU cores), threads=1, handler=sync `def`.

Use `async def` + `run_in_threadpool` only if the handler ALSO
needs to `await` real I/O — otherwise sync `def` is equivalent
and simpler.

Never: `async def` with a blocking model call. That is chapter 03
pitfall #3 and chapter 09's "poison for CPU-bound work".
```

**Checkpoint.** You now have three numbers (wall / p95 / throughput) for
each of the three shapes, all from the same workload. Every row of
chapter 09's Pattern matrix has a measured basis for the one labelled
"CPU-bound prediction, no downstream I/O"; the matrix extrapolates to
the other rows (feature-store I/O → threads 2–4; many concurrent
connections on I/O-bound work → async). The lab closes the loop on
chapter 03's "sync for CPU-bound, async only for real I/O" with the
numbers that make the rule hold.

## Expected final artefact

A lab directory containing:

- `app.py` — the three-handler FastAPI service with an env-var mode
  switch (though only the three different routes are needed here).
- `hammer.py` — the async client that fires N concurrent requests and
  prints wall / percentiles / throughput.
- `NOTES.md` — the two tables (N=10 and N=100) plus the one-line
  decision rule.

Running the three hammer commands against a fresh `uvicorn … --workers
1` should reproduce the sync-vs-threadpool tie and the ~10× blocking-
async regression on any CPU.

## Final checkpoint

Without re-reading chapters 03 or 09, answer:

1. In step 4, the blocking-async variant produced a wall-clock of
   ~2000 ms for 10 requests while the sync variant did it in ~300 ms.
   State in your own words what the asyncio event loop does during
   `time.sleep(0.2)` inside an `async def` handler with no `await`, and
   why FastAPI's threadpool makes the sync variant behave so
   differently.
2. The sync and `run_in_threadpool` variants produced numbers within
   noise of each other. State the one scenario chapter 03 says the
   `async def + run_in_threadpool` form is actually necessary for —
   i.e. the case where converting the handler to plain sync `def`
   would be worse.
3. Little's Law (chapter 09) is `L = λ × W`. For the blocking-async
   variant with a single worker, what is the effective `L`, and how
   does that explain the ~5 req/s throughput ceiling regardless of how
   many clients you throw at it? Does adding `--workers 4` to uvicorn
   raise the ceiling, and if so, by how much?
4. For a hypothetical service whose `/predict` handler needs to call a
   feature store (an `await`ed HTTP call, ~50 ms) and then run a CPU
   model prediction (~20 ms), which row of chapter 09's Pattern matrix
   does it match, and what three values (workers, threads, async?)
   would you start with?

If you can answer all four, chapter 03's handler-shape rule and chapter
09's three-axes concurrency framing are internalised and you are ready
to make the right choice on the real handler in exercise-01 and
defend it under the load test in exercise-04.

## Where to go next

- **`exercise-01` ("FastAPI prediction service")** writes the real
  `/predict` handler for a real sklearn pipeline — the acceptance
  criteria assume the sync `def` form from this lab's step 3.
- **`exercise-04` ("Load test and tune")** takes a real service and
  measures a latency-throughput curve; the ~10× regression in step 4
  of this lab is the kind of regression the exercise's `perf-
  regression` CI gate is designed to catch.
- **`lab-01` ("Liveness vs readiness")** covers the *other* chapter-04
  distinction on probe semantics.
- **`lab-03` ("Reading the latency-throughput curve: arrival-rate vs
  VUs")** takes a sync service and measures where its saturation knee
  is, using a real load generator (k6) rather than the hand-rolled
  `hammer.py` here.
