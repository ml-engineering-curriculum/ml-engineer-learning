# 09 — Latency, throughput, and concurrency

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 01–08.

## Motivation

"How fast is the service?" is the most common question a
new ML deployment has to answer, and the most common answer
— a single average latency number — is wrong. Real serving
performance lives in three shapes at once:

- **Latency percentiles**, because the mean hides the tail
  and it is the tail that hurts users.
- **Throughput under load**, because a service that serves
  one request in 20 ms may collapse to 800 ms per request
  under 100 concurrent callers.
- **The concurrency model**, because the number of workers,
  threads, and event-loop tasks determines what the two
  above look like.

This chapter is the vocabulary and the mental models. It
does not measure anything — chapter 10 does that. Read this
first so the numbers chapter 10 produces have somewhere to
land.

## Latency, precisely

**Latency** is *the elapsed time from the caller sending
their request to receiving the response.* Different observers
measure it at different points:

- **In the client.** Includes network, TLS, server
  processing, and the return network hop. This is what the
  user experiences.
- **At the load balancer / ingress.** Server-side, from
  request receipt to response emission. Excludes client
  network but includes server-side networking.
- **In the handler.** From the first line of your `predict()`
  to the return statement. Excludes any queueing before the
  handler ran.
- **The model call alone.** `t0` before `model.predict`,
  `t1` after. This is often what shows up in ML dashboards
  and it is the *smallest* of the four numbers.

They are all "latency". They are all different. When you
report a number, name which of these you measured.

The most common mistake at this level: an ML engineer
reports the 20 ms model call as the latency of the service,
and a downstream team measures 400 ms from the client and
files a bug. Both are right; they measured different things.

## Percentiles, not averages

Latency distributions are heavy-tailed. A service that
serves most requests in 20 ms may serve 1% of them in
800 ms because of a rare page fault, a garbage collection
pause, a lock contention on a shared resource, or a very
occasional model input that triggers a slower path.

Report the **percentiles**, not the mean:

- **p50 (median).** Half the requests are faster; half are
  slower. This is the "typical" experience.
- **p95.** 95% of requests are faster. This is a common
  SLA target — "95% of requests finish within X ms".
- **p99.** 99% of requests are faster. This is the tail
  where high-priority users notice problems. For a system
  doing 1 000 requests per second, 10 requests per second
  are worse than p99.
- **p99.9 and p99.99.** For high-QPS services with tight
  reliability requirements. At 10 000 QPS, p99.9 is 10
  requests per second — still user-facing.

The mean is essentially useless for latency: one 10-second
outlier moves the mean of a thousand 20 ms requests by
10 ms. That "10 ms of degradation" hides that one specific
user waited 10 seconds.

**The rule of thumb:** report p50, p95, p99, and *max*.
Alert on p95 and p99 crossing thresholds. Ignore the mean.

## The tail-latency amplifier

The tail matters more than intuition suggests, for one
non-obvious reason: **a service that calls N downstream
services in parallel sees the *max* of their latencies, not
the average.**

If each downstream call has p99 = 100 ms, then a request
that fans out to 10 downstream calls has approximately
**1 − 0.99¹⁰ ≈ 10%** chance of hitting *at least one*
tail-latency call. The p99 of the parent request is now
worse than the p99 of any single dependency.

That is why the recommendation-engine team obsesses about
p99 rather than p50. Every extra ms in the tail multiplies
across the fan-out.

Serving-side implication: **your service's tail latency is
someone else's average.** A p99 spike on your service
becomes a p50 problem for a downstream caller that fans
out.

## Throughput, precisely

**Throughput** is *requests served per unit of time*, in
steady state, at a specified quality of service.

The "at a specified quality of service" clause matters. A
service can technically process 1 000 requests per second
if you don't care that the 500th one takes 30 seconds.
Report throughput at a stated latency SLA:

- "This service sustains **500 requests per second** at
  **p95 ≤ 100 ms**."

That's a sentence you can defend. "This service does 500
QPS" without the latency budget is meaningless.

The **saturation point** is the QPS above which latency
degrades non-linearly. Below saturation, adding load barely
moves latency (there's spare capacity). At saturation,
requests start queueing (the model can't keep up with the
arrival rate), and per-request latency jumps sharply.
Chapter 10 finds this point empirically.

## Little's Law

The one formula worth remembering:

**L = λ × W**

- **L** — the average number of requests in the system at
  any moment.
- **λ** — the average arrival rate (requests per second).
- **W** — the average time a request spends in the system.

Two useful rearrangements:

- **λ = L / W**. Given a concurrency limit of L and an
  average latency of W, the maximum throughput is L / W.
  A service with 8 workers and average latency of 50 ms
  peaks at 8 / 0.050 = 160 requests per second — no matter
  how much traffic you throw at it.
- **W = L / λ**. Under a target throughput λ and current
  concurrency L, the average time in system is L / λ. If
  λ = 100 QPS and L = 20 (many requests in flight), the
  average latency is 200 ms. That may be much higher than
  the model's actual compute time — the rest is queueing.

Little's Law is the mathematical statement of "you can't
throw more traffic through a fixed-capacity system without
queueing making requests slower". Every autoscaling story
is a story about keeping L / λ within the acceptable W.

## Where latency comes from — the anatomy

For an online serving request, per-request latency is a
sum of contributions:

1. **Network to service** — from the client through the
   load balancer to the pod. Typically 1–20 ms in-cloud;
   10–100 ms cross-region.
2. **Queueing in front of the worker** — how long the
   request waited before a worker picked it up. Zero when
   under saturation; grows fast once saturated.
3. **Request parsing and validation** — pydantic
   validation, JSON parsing. Typically ≤ 1 ms.
4. **Feature retrieval** — feature store lookup, database
   join. Highly variable; the biggest latency contributor
   for real systems with rich features.
5. **Model inference** — `model.predict()`. What the ML
   engineer usually thinks of as "the latency".
6. **Response serialisation** — dumping the response
   pydantic model back to JSON. Typically ≤ 1 ms.
7. **Network back to caller** — the return hop.

Steps 1 and 7 are network; the ML engineer influences them
by choosing where to deploy (in the same region as the
caller, behind the same load balancer, on the same VPC).
Steps 2 and 4 are usually the surprise budget items;
optimising step 5 without touching them can look like it's
having no effect on user-visible latency.

**Instrument each step.** Chapter 05's structured logs are
where the per-step timings land; chapter 10's load tests
are what expose which step is the bottleneck under load.

## Concurrency in Python — the three axes

Python has three concurrency mechanisms and they compose in
non-obvious ways. Understanding them is how you pick
`--workers`, `--threads`, and `async`/`sync` correctly.

### 1. Processes (`gunicorn -w N`)

Each worker is an independent OS process, with its own
Python interpreter, its own GIL, its own memory. Processes
provide *parallelism* for CPU-bound work.

- **Best for CPU-bound inference** (scikit-learn, XGBoost,
  small neural nets on CPU). N cores → N workers, each
  serving requests independently.
- **Memory multiplied by N.** A 500 MB model becomes
  4 × 500 MB = 2 GB of resident memory with 4 workers.
  On memory-constrained nodes, this is the ceiling.
- **Model loaded per worker.** Startup time and memory
  are per-process. Chapter 04's warm-up runs in every
  worker.

For a CPU-bound service on a k-core node, **start with
N = k or k − 1 workers.** That leaves a core for the OS
and system tasks; measurement (chapter 10) tunes further.

### 2. Threads (`gunicorn --threads N` within each worker)

Multiple threads share a process's memory and GIL. Threads
provide *concurrency* for I/O-bound work; the GIL prevents
them from providing parallelism for CPU-bound Python code.

- **Best for I/O-bound handlers** — waiting on a feature
  store, a database, an HTTP call to another service.
  While one thread waits on a socket, another runs.
- **Useless for pure Python CPU work.** The GIL serialises
  Python bytecode; adding threads does not add throughput.
- **Useful for CPU work in C extensions that release the
  GIL.** `numpy`, `scikit-learn`, `torch`, and most C-
  backed ML libraries release the GIL during their heavy
  loops, so threads *can* parallelise the compute. Measure
  before assuming; it depends on the specific operation.

The rule for ML serving on CPU: **workers do the CPU
parallelism (they have separate GILs); threads help only
when the handler waits on I/O.** For a pure inference
service with no downstream calls, `--threads 1` is fine;
for a service that hits a feature store, `--threads 2` to
`--threads 8` may improve throughput.

### 3. Async (`async def` with `await`)

`async def` handlers run on a single event loop; they yield
control at `await` points. Async provides *concurrency* for
I/O-bound work without threads.

- **Best for handlers that make many I/O calls.** One
  async worker can multiplex thousands of concurrent
  connections waiting on network.
- **Poison for CPU-bound work.** A synchronous 50 ms model
  call inside an `async def` blocks the event loop for
  50 ms; no other request in that worker can progress.
- **Use `run_in_threadpool` to bridge** — the pattern from
  chapter 03. CPU work goes into a thread; the event loop
  keeps flowing.

The recommendation: **`def` (sync) handlers for CPU-bound
prediction, `async def` handlers only when the handler
genuinely awaits I/O.** Mixing them badly is the most
common performance regression in FastAPI ML services.

## Pattern matrix — which knob for which service

| Service shape                                     | Workers  | Threads | async? |
|---------------------------------------------------|----------|---------|--------|
| CPU-bound prediction, no downstream I/O           | k        | 1       | sync   |
| CPU-bound prediction + feature store (blocking)   | k        | 2–4     | sync   |
| CPU-bound prediction + feature store (async lib)  | k        | 1       | async + run_in_threadpool for predict |
| Small model, many concurrent connections (I/O)    | 2–4      | high    | async  |
| Large model, memory-constrained node              | 1        | high    | mixed  |

Where **k** is the number of CPU cores available to the pod.
This is a starting point; chapter 10 tunes each choice with
measurements against real load.

## GPU-served models: a special case

Everything above assumes CPU inference. GPU inference has
different mechanics:

- **The GPU is the bottleneck, not Python.** One GPU serves
  one process well; adding process workers does not add
  throughput.
- **Batching matters much more.** GPU kernels have high
  launch overhead; a batch of 32 images is not 32× more
  expensive than a batch of 1. Chapter 08's micro-batching
  pattern is essentially mandatory.
- **Purpose-built inference servers** (Triton, TorchServe,
  vLLM) implement micro-batching, model parallelism, and
  scheduler policies you would otherwise reinvent badly.

For CPU serving, FastAPI + gunicorn is often the right
choice all the way through production. For GPU serving,
your platform team likely already has an inference server;
you package the model to its contract and hand off (chapter
11).

## Memory — the silent constraint

Latency and throughput dominate the conversation. Memory
kills you second, usually in a way that shows up as a pod
OOMKill in the middle of the night.

- **Model memory × workers.** A 500 MB model × 4 workers =
  2 GB of resident memory *before* any request lands.
- **Per-request working memory.** DataFrame construction,
  tensor allocation. Usually small, but a batch endpoint
  with a 1 000-row payload can construct sizeable
  temporary buffers.
- **Framework memory pressure.** `torch` allocates
  aggressively; `xgboost` on many trees is memory-heavy;
  `numpy` mmaps files sometimes.

Set memory limits on the container generously enough that
transient spikes don't kill it, but not so generously that
you paper over a leak. Kubernetes OOMKills at the memory
limit; Docker `--memory` does the same. Chapter 06's
Dockerfile does not set a limit; the platform team does at
deploy time.

## What to measure

Chapter 10 collects the numbers. This is the shortlist:

- **p50, p95, p99, max latency** at a target QPS.
- **Sustained throughput** at a target p95.
- **Saturation point** — the QPS above which p95 explodes.
- **CPU, memory, GC, worker health under load.**
- **Per-step timings** — network, queue, feature retrieval,
  model, serialisation. The traces from chapter 05 are the
  source.

Aim for a **latency-throughput curve**: x-axis = QPS, y-axis
= p95. Below saturation, the curve is flat; above, it
climbs steeply. Where the curve turns is your capacity.

## Common pitfalls

1. **Reporting the mean latency.** Meaningless. Report
   percentiles.
2. **Reporting the model-call latency as service latency.**
   Loses network, queue, and feature-retrieval time.
   Report at each layer explicitly.
3. **Adding workers past the number of CPUs.** N workers
   with N > cores start context-switching; latency degrades
   with no throughput gain. Measure; k or k − 1 is a good
   default.
4. **`async def` with a blocking model call.** Every
   request in that worker serialises through one event
   loop; throughput collapses.
5. **Optimising the model call from 50 ms to 30 ms while
   the feature-store call takes 200 ms.** Total latency
   moved from 250 ms to 230 ms. The user cannot tell. Fix
   the bottleneck, not the visible number.
6. **No warm-up.** First-request latency looks 10× worse
   than steady state. Warm-up is chapter 04.
7. **Ignoring memory in favour of latency.** The service
   is fast, then it OOMs in the middle of the night. Set
   limits; monitor RSS.
8. **Micro-optimising Python.** For most services, the win
   is in the model, the batch size, or the concurrency
   model — not in whether you use a list comprehension or
   a `map`.

## Summary

- Latency is a distribution, not a number. Report p50,
  p95, p99, max. The mean lies.
- Throughput is meaningful only at a specified quality of
  service. "X requests per second at p95 ≤ Y ms."
- Little's Law: L = λ × W. Concurrency, latency, and
  throughput are mathematically linked; you cannot
  independently improve them past a point.
- Python concurrency is three axes — processes (workers),
  threads, async. Workers give parallel CPU; threads help
  I/O and GIL-releasing C code; async multiplexes I/O.
  For CPU-bound ML, start with k workers, 1 thread, sync
  handlers.
- Tail latency amplifies through fan-out. Your p99 is
  someone else's p50.
- Memory is the silent constraint; workers multiply model
  memory.

## Where this goes next

Chapter 10 puts a load generator against the service.
Locust, wrk, or k6 — the specific tool matters less than
the reproducible measurement. Chapter 10 covers reading a
latency-throughput curve, finding the saturation point,
and turning the numbers into a defended latency budget for
the handoff (chapter 11).
