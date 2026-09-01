# exercise-04 — Load test and tune

> **Estimated effort:** 3–4 hours.
> **Companion chapters:** [09 — Latency, throughput, and concurrency](../09-latency-throughput-and-concurrency.md),
> [10 — Load testing and tuning](../10-load-testing-and-tuning.md).
> **Prerequisites:** exercises 01, 02, and 03. A
> containerised FastAPI service you can `docker compose up`
> and hit at `localhost:8080`.

## Objective

Turn the service from the previous exercises into a
capacity budget you can defend. Build a reproducible
k6 load-test rig, produce a **latency-throughput curve**,
locate the **saturation knee**, tune one lever, re-run,
and commit a **REPORT.md** that a platform team would
accept as the handoff artefact for chapter 11.

By the end you should have:

- A **warm-up k6 script** and a **ramping-arrival-rate
  load-curve script**, both under `load-tests/`.
- A **results directory** with the raw k6 output for at
  least two configurations plus a small analyser that
  turns the JSON into a p95-per-QPS table.
- A **REPORT.md** with the curve, the identified
  saturation point, the tuning experiment you ran, and
  a defended "sustained QPS at p95 ≤ target-ms" number.
- A **CI regression gate** — a short k6 script wired
  into GitHub Actions that fails the build if p95 at
  a fixed QPS regresses past a threshold.

The point is to internalise chapter 10's discipline:
you don't ship a service without a *reproducible*
latency-throughput measurement, and every claim about
capacity has a number and a script behind it.

## Prerequisites

- Exercises 01–03 complete.
- `k6` installed locally
  (<https://k6.io/docs/get-started/installation/>). The
  Docker image (`grafana/k6`) is fine if you'd rather
  not install a binary.
- Enough headroom on your test machine to run the
  service and the load generator without either
  starving the other. If you are on a laptop, close
  the browser. Chapter 10's "the generator is the
  bottleneck" pitfall is easy to hit locally.

## Problem statement

### Part A — The scripts (≈ 45 min)

Under `load-tests/`, add:

- **`warmup.js`.** 30 seconds at 5 VUs against
  `/predict`. Ignore its results — this exists to warm
  workers, JIT the model, and page in the artifact.
- **`load-curve.js`.** Uses the `ramping-arrival-rate`
  executor (not `ramping-vus`; chapter 10 pitfall #4).
  Ramp through at least six target rates from below
  saturation to well past it — e.g., 20, 50, 100, 200,
  400, 800 req/s, 60 seconds per stage. Set
  `preAllocatedVUs` high enough that VU availability
  is not the constraint.
- **`regression.js`.** Fixed 30-second run at a single
  QPS (pick a number safely below your identified knee).
  Sets a `thresholds` block that fails the test if
  `http_req_duration p(95)` exceeds a chosen budget
  (e.g., `p(95)<150`).

All three scripts:

- Read `BASE_URL` from an environment variable, default
  `http://localhost:8080`.
- Build a randomised POST body so no two requests are
  byte-identical (chapter 10 pitfall: uniform payloads
  hide worst-case parsing / featurisation cost).
- Assert `status === 200` via `check()`.

### Part B — Run the curve (≈ 45 min)

Bring up the service with the default configuration
(the one committed after exercise 02) using
`docker compose up --build -d`.

Then:

1. Run `warmup.js`.
2. Run `load-curve.js --out json=results/baseline.json`.
3. Capture `docker stats --no-stream churn-service`
   during the peak stage (a separate terminal, or a
   short shell loop that appends to
   `results/baseline-stats.txt`).
4. Tear down.

Write `load-tests/analyse.py` that:

- Reads a k6 JSON output.
- Bins requests by the target arrival rate (from
  scenario metadata or from a small sidecar file you
  write alongside the run — either works).
- Computes p50, p95, p99, and max latency per bin.
- Prints a table and writes a `results/<name>.md`
  with the same table.

Produce the table for the baseline run. The **knee**
is the QPS bin where p95 crosses your latency budget.
Note it in `REPORT.md`.

### Part C — One tuning experiment (≈ 60 min)

Pick one lever from chapter 10's ranked list:

- **Workers** — change `WORKERS` in the compose file.
  This is the default choice; it is the largest lever
  for CPU-bound services.
- **Batch size on `/predict:batch`** — if your callers
  send batches.
- **Model complexity** — a smaller model (e.g., fewer
  trees) served from the same code.
- **`orjson` for response serialisation** — swap
  `JSONResponse` for `ORJSONResponse` and rerun.

Change **one** thing. Rerun `warmup.js` and
`load-curve.js` against the tuned service. Save the
results to `results/tuned.json`. Run `analyse.py`
against both baseline and tuned; produce a side-by-
side table.

If the tuning made things *worse*, do not paper over
it. Report it. "I doubled the workers and p95 got
worse because memory pressure caused GC pauses" is a
valid, learn-something result.

### Part D — The report (≈ 45 min)

`REPORT.md` contains, at minimum:

1. **Test rig.** Machine spec (CPU count, RAM, OS),
   whether the generator and service ran on the same
   host, k6 version, service image tag / git SHA,
   dataset the model was trained on.
2. **Method.** How long the warm-up ran, the ramp
   profile, how you binned the results, how many
   times you ran each configuration (once is fine;
   note it explicitly).
3. **Results.** The p50/p95/p99 vs QPS table for
   both configurations. A brief narrative pointing
   at the knee in each.
4. **Interpretation.** One paragraph mapping the
   observed behaviour to chapter 09's anatomy: was
   the model call the bottleneck? Feature retrieval?
   Queueing? Cite the `docker stats` output or the
   per-step timings from the structured logs
   (exercise 01, Part C).
5. **The defended number.** "This service sustains
   **X requests per second** at **p95 ≤ Y ms** on a
   {CPU-count}-core, {RAM} machine with WORKERS=N."
   That is the sentence chapter 09 asked for; it is
   what the platform team wants in the handoff.
6. **What you would tune next.** In the same ranked
   order as chapter 10. You do not need to actually
   run the next experiment; naming it and predicting
   its impact is enough for this exercise.

### Part E — The CI regression gate (≈ 45 min)

Extend the workflow from exercise 02 with a new job:

- **`perf-regression`**. Depends on the existing
  `build` job. Reuses the built image, starts the
  container with `WORKERS=<the value you tested>`.
  Runs `warmup.js`, then `regression.js`. k6's
  `thresholds` block causes a non-zero exit if
  `p(95)` exceeds the budget; the job fails.

The job should:

- Print the k6 summary to the log even on success.
- Upload the raw JSON as a workflow artifact.
- Take under three minutes end-to-end. Full curves
  belong in a nightly / weekly job, not per-PR
  (chapter 10's "load testing in CI" section).

Verify the gate works both ways on a branch:

1. Push a passing branch — `perf-regression` succeeds.
2. Push a branch that adds `time.sleep(0.5)` to
   `/predict` — `perf-regression` fails; the k6 output
   in the log shows p95 above the budget.
3. Revert.

Note the two run URLs (or screenshots) in `NOTES.md`.

## Starter guidance

- **Chapter 10's scripts are the starting point.** The
  `wrk` and Locust snippets there are equivalent
  alternatives; k6 is the module's default because of
  the CI-friendly thresholds.
- **Watch the generator.** Open a second terminal
  during `load-curve.js` and run `htop` or
  `docker stats` on the machine hosting k6. If k6 is
  pegged, your latency numbers are the generator's,
  not the service's — split them onto two hosts, or
  reduce `preAllocatedVUs` and add stages.
- **Use `ramping-arrival-rate`, not `ramping-vus`.**
  This is chapter 10 pitfall #4; getting it wrong
  makes saturation invisible.
- **Restart the service between runs.** Pod state
  (open connections, warmed caches, worker
  restarts-per-crash) leaks between load tests. Chapter
  10's reproducibility recipe.
- **Change one variable at a time.** If you compare
  baseline (WORKERS=1, small model) against tuned
  (WORKERS=4, big model), you cannot attribute the
  delta to either lever.
- **Do not tune past your evidence.** If p95 improved
  from 100 ms to 30 ms after adding workers, do not
  then quadruple the workers again and claim linear
  scaling. Chapter 10 pitfall #6.
- **`orjson`, if you pick it, is a two-line change**
  in FastAPI — swap `default_response_class` on the
  app to `ORJSONResponse`. Measure its actual impact
  rather than assuming.

## Acceptance criteria

- `k6 run load-tests/load-curve.js --out
  json=results/baseline.json` completes without k6
  errors; the JSON file exists and contains
  `http_req_duration` samples.
- `python load-tests/analyse.py results/baseline.json`
  prints a table with columns:
  `target_qps | p50 | p95 | p99 | max` and at least
  six rows.
- `results/tuned.json` exists for a second
  configuration that changes exactly one lever from
  chapter 10's ranked list.
- `REPORT.md` contains all six sections from Part D;
  the "defended number" sentence names a QPS, a p95
  budget, and the machine spec.
- The GitHub Actions `perf-regression` job passes on
  the current branch and fails on a branch that
  introduces a `time.sleep(0.5)` into `/predict`
  (or an equivalent artificial regression). Both
  run URLs are recorded in `NOTES.md`.
- `NOTES.md` records what was surprising — the biggest
  gap between what you expected and what you measured.

## Stretch goals (optional)

- **The full 4-way sweep.** Run the curve for WORKERS
  values 1, 2, 4, and 8. Plot all four curves on one
  axis (matplotlib is fine; commit the PNG). Report
  the QPS-at-p95 knee for each; comment on where the
  scaling flattens.
- **Sync vs async handler.** Convert `/predict` to
  `async def` with `run_in_threadpool` for the model
  call. Rerun the curve. Compare. Chapter 09 predicts
  this should not help for CPU-bound inference —
  verify or falsify.
- **`onnxruntime`.** Export your model to ONNX and
  serve it with `onnxruntime` behind the same
  endpoint. Rerun the curve. Report the delta.
  Chapter 10 lever #6.
- **Tail-latency deep dive.** For the highest-QPS
  stage in the baseline, extract every request with
  `duration > p99`. Group by any structured-log field
  you have (worker id, request id, batch size).
  Look for a common thread. This is chapter 10's
  "correlate against per-step timings" step made
  literal.
- **Nightly full-curve workflow.** In addition to
  the PR gate, add a scheduled workflow
  (`on: schedule: - cron: '0 3 * * *'`) that runs
  the full `load-curve.js`, invokes `analyse.py`,
  and posts the resulting table as an issue comment
  or a workflow-run summary. Do not upload results
  to a third party.
- **Alternate load generator.** Rerun one
  configuration with `wrk` (chapter 10's example).
  Compare its numbers to k6's. If they differ by
  more than 10%, investigate: is the generator the
  bottleneck? Is one of them measuring something
  the other isn't?

## What to hand in

```
exercise-04/
├── app/                              # from exercise-03, possibly tuned
├── jobs/                             # from exercise-03
├── load-tests/
│   ├── warmup.js
│   ├── load-curve.js
│   ├── regression.js
│   └── analyse.py
├── results/
│   ├── baseline.json
│   ├── baseline.md                   # generated table
│   ├── baseline-stats.txt            # docker stats capture
│   ├── tuned.json
│   ├── tuned.md                      # generated table
│   └── tuned-stats.txt
├── compose.yaml                      # possibly modified for the tuned run
├── .github/workflows/ci.yml          # + perf-regression job
├── REPORT.md                         # the defended number + method
├── NOTES.md                          # surprises, CI run URLs
└── README.md                         # how to reproduce
```

Solutions live in the paired `ml-engineer-solutions` repo.
