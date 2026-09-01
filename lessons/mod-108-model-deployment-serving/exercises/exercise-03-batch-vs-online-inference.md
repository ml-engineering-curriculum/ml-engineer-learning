# exercise-03 — Batch vs online inference

> **Estimated effort:** 3–4 hours.
> **Companion chapter:** [08 — Batch vs online inference](../08-batch-vs-online-inference.md).
> **Prerequisites:** exercises 01 and 02 (a containerised
> FastAPI prediction service exists and can be run locally).

## Objective

Extend the service from exercises 01 / 02 with a **batch
scoring job** and a **hybrid pre-compute + cache** endpoint,
then write a short **decision memo** justifying which path
you would use for each of three concrete product scenarios.

By the end you should have:

- A **batch scoring job** (`jobs/nightly_score.py`) that
  reads a Parquet input, chunks it, calls the same model
  the online service serves, and writes a versioned Parquet
  output plus a `_MANIFEST.json`.
- A **`/score` endpoint** on the online service that reads
  a pre-computed score for a subject from a local key-value
  cache; falls back to online inference on cache miss;
  emits a structured log that records which path was taken.
- A **decision memo** (`DECISIONS.md`) that applies the
  batch-vs-online matrix from chapter 08 to three product
  scenarios you are given below.

The point of the exercise is to *own both shapes of
inference*, keep the model code shared between them, and
be able to argue for one over the other in a design
review.

## Prerequisites

- Exercises 01 and 02 complete: FastAPI service + Docker.
- Python 3.11+ with `pandas`, `pyarrow`, `joblib`,
  `fastapi`, `pytest`.
- A local KV store. `redis` in Docker is the reference
  choice — a `dict` in a JSON file is acceptable if you
  do not want the extra container, but call out the
  simplification in `NOTES.md`.
- A tabular dataset with at least 5 000 rows so the batch
  job's chunking is exercised. Reuse whatever you trained
  on in exercise-01, or synthesise rows from the training
  distribution.

## Problem statement

### Part A — The batch scoring job (≈ 60 min)

Write `jobs/nightly_score.py`. It must:

1. Accept CLI arguments: `--input <parquet-path>`,
   `--output-root <path>`, `--model <artifact-dir>`,
   `--run-date <YYYY-MM-DD>`. No positional args.
2. Load the same `ModelBundle` your online service uses.
   The scoring code MUST call the same function that
   `/predict` calls — no reinvented feature engineering
   for the batch path. This is the "share the code path"
   rule from chapter 08's pitfalls.
3. Read the input Parquet, iterate it in chunks of
   `--chunk-size` rows (default 10 000), score each
   chunk, and accumulate the results.
4. Write to a versioned path:
   `<output-root>/dt=<run-date>/run=<uuid7-or-uuid4>/scores.parquet`.
   Do **not** overwrite `latest/`; if you want a
   `latest` pointer, write a `latest.txt` file next to
   the dated directories with the run's relative path.
5. Alongside the output, write `_MANIFEST.json` with
   at minimum: `model_name`, `model_version`, `git_sha`,
   `input_path`, `input_row_count`, `output_row_count`,
   `chunk_size`, `run_started_at`, `run_finished_at`,
   `duration_seconds`.
6. Fail loudly on schema drift — if the input DataFrame
   is missing columns the model expects, exit non-zero
   with a message naming the missing columns. Do not
   silently coerce or drop rows.
7. Emit structured JSON logs on stdout (the same logger
   configuration from exercise-01) with per-chunk
   progress lines: `chunk_index`, `chunk_rows`,
   `elapsed_ms`.

Run the job at least once end-to-end:

```
python -m jobs.nightly_score \
    --input data/subjects.parquet \
    --output-root out/scores \
    --model artifact \
    --run-date 2026-09-01 \
    --chunk-size 5000
```

### Part B — The hybrid endpoint (≈ 60 min)

Add `POST /score` (or `GET /score/{subject_id}`, your
choice — pick one and stick with it) to the FastAPI app
from exercise-01. It implements chapter 08's **hybrid
pattern 1 — pre-compute, serve from cache**:

- Read the pre-computed score for `subject_id` from the
  cache. On hit: return `{"subject_id", "score",
  "computed_at", "source": "batch",
  "model_version": ...}`.
- On miss: run online inference for that subject
  (features passed in the request body or fetched from
  a lookup — your call), return `{"subject_id",
  "score", "source": "online", "model_version": ...}`.
  Do NOT write the online-computed score back to the
  cache; that would defeat the "batch is the source of
  truth" invariant. Log the miss.

Wire in a small loader that populates the cache from
Part A's output. Provide `scripts/load_cache.py` that
reads `_MANIFEST.json` + `scores.parquet` and writes
`(subject_id -> {score, computed_at, model_version})`
into the KV store.

Requirements:

- `computed_at` in the batch-hit response must be the
  `run_finished_at` timestamp from the manifest, not
  the moment the API served the request. Chapter 08's
  "timestamp the score; log its age" pitfall.
- The response schema is the same shape for hit and
  miss so the caller doesn't have to branch — the
  `source` field is the only cue.
- If the KV store is unreachable, fall back to online
  inference and log a `cache_unavailable` warning. Do
  not fail the request.

### Part C — Micro-batching on `/predict:batch` (≈ 30 min)

Instrument the existing `/predict:batch` endpoint so
that the *server-side* per-request batching is
measurable:

- Add a `batch_size` field to the structured
  `prediction` log line, alongside `inference_ms`.
- Compute and log the *amortised* per-instance cost:
  `inference_ms / batch_size`. Chapter 08's chunk-size
  discussion translates directly.

You do NOT need to implement chapter 08's hybrid
pattern 3 (async collector merging separate calls
into one model call) — that is inference-server
territory (Triton, Ray Serve). Just make the existing
per-request batching legible.

### Part D — The decision memo (≈ 30 min)

Write `DECISIONS.md` at the repo root. For each of the
three scenarios below, state which shape you would
serve — **batch**, **online**, or **hybrid** (which
one) — and give a two-to-four-sentence justification
grounded in the chapter 08 decision matrix (interactive
caller? synchronous decision? features known ahead of
time? staleness tolerance?).

Scenarios:

1. **Homepage recommendations.** A logged-in user
   lands on the homepage. The page needs a
   personalised list of 10 items. The catalogue is
   updated hourly.
2. **Card-present fraud check.** An in-store
   transaction is authorised or declined at the point
   of sale in under 300 ms. The features include the
   customer's 30-day spending pattern plus the
   just-swiped transaction amount / merchant /
   location.
3. **Nightly customer-health email.** A CRM job at
   06:00 sends a "your health score has changed"
   email to customers whose churn probability crossed
   a threshold since yesterday. The email links back
   to a dashboard that shows the same number.

There is not necessarily one right answer for each —
what matters is that the reasoning maps to the chapter's
matrix and that you name the trade-off you are
accepting.

## Starter guidance

- **Share the scoring function.** Extract
  `predict_dataframe(bundle, df) -> np.ndarray` in
  `app/model.py`. Both `/predict` and the batch job
  call it. Neither owns feature engineering.
- **Do not skip `_MANIFEST.json`.** A batch job
  without a manifest cannot be trusted downstream.
  Chapter 08 makes this a pitfall for a reason.
- **`uuid7`** — `uuid.uuid7` on Python 3.13+, or
  `pip install uuid6` for older Pythons. Sortable
  IDs make `latest` reasoning easier. Plain
  `uuid.uuid4` is fine if you don't want another
  dependency.
- **Cache TTL.** Redis: set a TTL of ~30 hours so a
  missed batch run (say, 27 h since last successful
  score) surfaces as a cache miss rather than as
  stale data served forever. Note the choice in
  `NOTES.md`.
- **Do not swap `pandas` for Spark / Ray** on this
  exercise. Chapter 08 mentions them as scale
  options; introducing a distributed framework here
  buries the learning objective.
- **The FastAPI route stays small.** All the work
  (cache read, fallback, logging) fits in ~40 lines.
  If it doesn't, look for a helper to extract.

## Acceptance criteria

- `python -m jobs.nightly_score --input ... --output-root
  ... --model ... --run-date 2026-09-01` produces
  `out/scores/dt=2026-09-01/run=<id>/scores.parquet`
  and `_MANIFEST.json`. The manifest contains every
  field listed in Part A.
- Running the job twice with the same date creates two
  distinct `run=<id>/` directories; the second run does
  not overwrite the first.
- The batch job scores a 50 000-row synthetic input
  without loading the whole DataFrame at once
  (verify by peak-RSS observation — `/usr/bin/time -v`,
  `resource.getrusage`, or `psutil` sampling; note the
  peak in `NOTES.md`).
- `python scripts/load_cache.py --manifest <path>`
  populates the KV store from a completed run.
- Against the loaded cache:
  - `curl <base>/score/known_subject` returns
    `source: "batch"` and the correct `computed_at`.
  - `curl <base>/score/unknown_subject` returns
    `source: "online"`; the log shows a `cache_miss`
    event.
  - Stopping the KV container mid-run causes
    subsequent requests to return online results with a
    `cache_unavailable` warning in the logs, not 5xx.
- `/predict:batch` log lines include `batch_size` and
  `amortised_per_instance_ms`.
- `DECISIONS.md` contains three sections — one per
  scenario — with a decision and a justification that
  refers back to the chapter 08 matrix.
- `pytest -q` still passes; add at least one test each
  for: batch-cache hit path, online fallback on cache
  miss, cache-unavailable fallback.

## Stretch goals (optional)

- **Backfill mode.** Add `--date-range 2026-08-01:2026-08-07`
  that runs the job once per date in the range. Chapter
  08's "sooner or later someone will ask for the last 30
  days" pitfall.
- **Idempotency assertion.** Store an
  `(input_path, model_version, run_date)` key in a
  small sqlite file; refuse to rerun for a key that
  has already produced a manifest, unless `--force`
  is passed.
- **Parallelise chunk processing.** Use
  `concurrent.futures.ProcessPoolExecutor` to score
  chunks in parallel. Measure wall-clock speedup on
  a 100 000-row input. Note where the speedup
  plateaus.
- **Feature-store analogue.** Instead of caching
  scores, cache *features* (the "hybrid pattern 2"
  from chapter 08). The online path fetches
  features from the cache and calls the model at
  request time. Compare end-to-end latency against
  the pre-computed-score version.
- **Airflow / Prefect wrapper.** Wrap the job in a
  simple DAG that runs on a schedule and depends on
  an upstream "input ready" sensor. This is really
  a `mod-103` exercise; do it here if you already
  own that infrastructure.
- **Score-age SLO.** Add a `/score/health` endpoint
  that returns the age of the most recent successful
  batch run as a header, and 503 if the age is
  greater than a configured threshold (default 30 h).
  Chapter 08's "stale scores are worse than a clean
  miss" idea, made mechanical.

## What to hand in

```
exercise-03/
├── app/                          # from exercise-01, extended
│   ├── main.py                   # + /score endpoint
│   ├── model.py                  # predict_dataframe helper
│   ├── cache.py                  # KV client, get/set/miss handling
│   └── logging_config.py         # existing
├── jobs/
│   └── nightly_score.py
├── scripts/
│   └── load_cache.py
├── tests/
│   ├── test_batch_job.py         # schema-drift raise, manifest
│   ├── test_score_hit.py         # batch cache hit
│   ├── test_score_miss.py        # online fallback
│   └── test_cache_unavailable.py
├── compose.yaml                  # + redis service
├── DECISIONS.md                  # three-scenario memo
├── NOTES.md                      # peak RSS, cache TTL rationale,
│                                 # anything surprising
└── README.md
```

Solutions live in the paired `ml-engineer-solutions` repo.
