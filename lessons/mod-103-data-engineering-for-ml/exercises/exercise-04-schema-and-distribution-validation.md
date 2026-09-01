# exercise-04 — Schema and distribution validation

> **Estimated effort:** 3 hours.
> **Companion chapter:** [10 — Dataset validation: `pandera` and Great Expectations](../10-dataset-validation.md).

## Objective

Wrap the dataset you have been building through exercises 01–03 in a
**validation gate** that catches schema drift and distribution drift
before either can silently corrupt a training run.

By the end you will have:

- A `pandera` schema encoding every business invariant of the
  dataset (dtypes, ranges, enums, uniqueness, cross-column rules).
- A **baseline stats** artefact — a versioned snapshot of what
  "healthy" looks like — plus PSI / KS drift checks against it.
- A Great Expectations expectation suite for the ingest-boundary
  view of the same dataset, with auto-generated Data Docs.
- A single `validate_batch(df)` function that gates the training
  pipeline: pass → training continues; fail → the pipeline stops,
  attributes the failure, and snapshots the bad batch.
- Tests that would fail if a colleague accidentally deleted a
  check, and *would* have caught three plausible upstream breakages.

The point is not to hit 100% test coverage. It is the discipline of
turning tacit "we assume `country` is one of five values" beliefs
into executable, versioned code that fails loud when the assumption
stops holding.

## Prerequisites

- Chapter 10 of this module.
- The output of exercise-01 (partitioned Parquet dataset) is the
  natural input. Any tabular dataset with a mix of numeric,
  categorical, and timestamp columns works — reuse Adult, Ames, or
  NYC Taxi from exercise-02 if you skipped 01.
- Python 3.11+, Pandas ≥ 2.1, `pandera[pandas] >= 0.20`,
  `great-expectations >= 1.0`, `scipy` (for KS), `pytest`.

## Problem statement

### Part A — a `pandera` schema for the training input (≈ 40 min)

In `validation/schema.py`, declare a `pa.DataFrameModel` for the
dataset. At a minimum, the schema must encode:

1. **Every column present** with its expected dtype. Use
   `pd.DatetimeTZDtype("ns", "UTC")` for timestamps — no naive
   datetimes.
2. **Non-null on the primary key** and on any column your downstream
   pipeline dereferences without a fallback.
3. **Value ranges** for numeric columns (`ge=`, `le=`) with limits
   grounded in the exercise-02 outlier audit — not arbitrary bounds.
4. **`isin=[...]`** on categorical columns with a fixed vocabulary.
   List them explicitly; do not pull the set from the current batch.
5. **Uniqueness** on the primary key (`unique=True`).
6. **`Config.strict = True`** so an unexpected upstream column is a
   hard failure, not a warning.

Then add at least **two custom checks** covering cross-column or
whole-DataFrame invariants:

- A `@pa.dataframe_check` asserting `event_ts >= user_signup_ts`
  (order-can't-happen-before-signup, or an equivalent invariant
  from your dataset).
- A `@pa.check("col")` asserting a per-column property the schema
  fields can't express — e.g. "the last 24 hours of data are
  present" (`(df["ts"].max() - df["ts"].min()) >= pd.Timedelta("1D")`).

Wire the schema to your training entry point via `@check_input`:

```python
from pandera import check_input

@check_input(TrainingBatch, lazy=True)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

**Explain, in one paragraph in `REPORT.md`,** why `lazy=True` and
what it changes about the failure UX.

### Part B — baseline statistics (≈ 30 min)

A drift check needs a **baseline** to compare against — chapter 10's
"snapshot of what the data looked like when the pipeline was last
known to be healthy". Produce one.

In `validation/baseline.py`, write:

```python
def compute_baseline(df: pd.DataFrame) -> dict:
    """Return a JSON-serialisable per-column stats snapshot."""
```

For each numeric column: `mean`, `std`, `median`, `q01`, `q05`,
`q25`, `q75`, `q95`, `q99`, `n_missing`, and a `hist` (10-bin
histogram, bin edges + counts) — the histogram is what you'll
compare against for PSI.

For each categorical / string column: `top_k` (top-20 value → count
map) and `n_missing`.

For each timestamp column: `min`, `max`, `tz`, `n_missing`.

Write the baseline to a date-stamped directory:

```
baselines/
  training_batch/
    2026-09-01/
      column_stats.json
      pandera_schema.py        # copy of the schema at this version
      row_count.txt
```

Do **not** overwrite `latest.json`. Every baseline is versioned by
date; the pipeline reads whichever one you point it at.

Compute the baseline on your **training partition only** — a
baseline built on train+test conflates natural growth with drift.

### Part C — a drift-check function (≈ 30 min)

In `validation/drift.py`, write:

```python
def check_drift(
    batch: pd.DataFrame,
    baseline: dict,
    *,
    psi_moderate: float = 0.10,
    psi_severe:   float = 0.25,
    ks_alpha:     float = 0.01,
) -> DriftReport:
    ...
```

Requirements:

1. **PSI on every numeric column**, computed against the baseline's
   histogram bin edges (not re-binned each run). Follow the standard
   formula:

   `PSI = Σ (p_batch - p_base) * ln(p_batch / p_base)`

   with a small floor (e.g. 1e-4) on empty bins to avoid log-zero.
2. **KS on every numeric column**, using
   `scipy.stats.ks_2samp`. Record the statistic and p-value.
3. **Chi-square on every categorical column**, comparing the
   observed top-k counts to the baseline's. Use
   `scipy.stats.chisquare` with matched category ordering; drop
   categories with expected count < 5.
4. **Return a `DriftReport`** (a small dataclass or dict) with
   per-column verdicts and an overall severity: `ok`, `moderate`, or
   `severe`. The severity rule is: severe if *any* column has PSI
   ≥ `psi_severe` or a chi-square p-value < 0.001; moderate if any
   column has PSI ≥ `psi_moderate` or a KS p-value < `ks_alpha`;
   otherwise ok.
5. **Do not silently truncate**: if you skip a column because it
   isn't in the baseline (schema drift), that goes in the report as
   `unknown_columns`.

The thresholds above are the ones chapter 10 cites; keep them as
function arguments so they can be tuned per dataset.

### Part D — a Great Expectations suite for the ingest boundary (≈ 40 min)

`pandera` is your in-pipeline validator. Great Expectations is the
one you hand to the data team so they get Data Docs and can run
checks outside your training code. Build both.

In `gx/`, create a minimal GX project rooted at `gx/gx/` (or the
default `great_expectations/`) with:

1. A file-context (`gx.get_context()` in the default local mode).
2. A Pandas data source and a whole-DataFrame batch definition
   named `training_batch`.
3. An expectation suite named `training_batch_suite` covering:
   - `ExpectColumnToExist` for every column in the pandera schema.
   - `ExpectColumnValuesToNotBeNull` on the primary key and any
     non-nullable column.
   - `ExpectColumnValuesToBeInSet` for each `isin=[...]` column.
   - `ExpectColumnValuesToBeBetween` for each numeric range.
   - `ExpectColumnMeanToBeBetween` for at least one numeric column,
     with limits derived from the baseline (± 3 σ, say).
   - `ExpectColumnKlDivergenceToBeLessThan` on one column, using
     the baseline histogram as the reference distribution.
4. A **checkpoint** that runs the suite against `training_batch`
   and, on failure, exits non-zero.

Generate the Data Docs (`context.build_data_docs()`), and commit
**the suite JSON** and the checkpoint config — but **`.gitignore`
the `gx/uncommitted/` and any built HTML**. Chapter 10's pitfall
#4: committed Data Docs balloon the repo.

Write a short paragraph in `REPORT.md` comparing pandera and GX for
this dataset: which check is easier to express in each, and where
you'd draw the boundary between them in a real stack.

### Part E — the training-pipeline gate (≈ 20 min)

Tie it together. In `validation/gate.py`, write:

```python
def validate_batch(
    df: pd.DataFrame,
    *,
    baseline_dir: Path,
    on_failure: Literal["raise", "warn"] = "raise",
) -> None:
    ...
```

Order of operations:

1. **Schema** via `TrainingBatch.validate(df, lazy=True)`. If it
   fails, snapshot and raise.
2. **Drift** via `check_drift(df, baseline)`. If `severe`, snapshot
   and raise. If `moderate`, log a structured warning; do not fail.
3. **Log** the row count, the baseline directory used, the overall
   severity, and (on failure) the path to the snapshot.

**Snapshot on failure.** Save the offending DataFrame (or a
random 10 000-row sample if it is larger) to
`failed_batches/{utc_iso_timestamp}/batch.parquet` alongside a
`failure.json` naming the schema errors / drift findings. Chapter
10's rule 4: the person debugging at 3am should not have to
reproduce the failure to see the data.

Import `validate_batch` into a stub training entry point:

```python
def train(config):
    df = load_dataset(config.data_uri)
    validate_batch(df, baseline_dir=config.baseline_dir)
    # ... exercise-03's pipeline continues here
```

You do not need to run a real training loop; the gate call and a
`print("would train now")` is enough for this exercise.

### Part F — tests (≈ 30 min)

In `tests/`, at minimum:

- `test_schema_accepts_a_known_good_batch` — the schema validates
  a snapshot of a clean training batch (a small fixture file works).
- `test_schema_rejects_missing_required_column` — drop the primary
  key column, expect `SchemaError` with `column_missing`.
- `test_schema_rejects_out_of_range_value` — inject a numeric value
  outside the range, expect `SchemaError`.
- `test_schema_rejects_unexpected_column` — add a mystery column,
  expect `SchemaError` because `strict=True`.
- `test_baseline_is_stable_on_identical_input` — computing the
  baseline twice on the same DataFrame returns byte-identical JSON.
- `test_drift_ok_on_baseline_itself` — running `check_drift(batch,
  baseline)` where `batch` is the exact DataFrame the baseline was
  computed from returns `ok`.
- `test_drift_severe_on_shifted_column` — take the baseline batch,
  add `+3σ` to one numeric column, assert severity == `severe`.
- `test_gate_snapshots_bad_batch` — build a bad batch, call
  `validate_batch(..., on_failure="raise")`, catch the exception,
  and assert the snapshot files exist.

Use fixtures (tiny CSVs or programmatically-constructed DataFrames)
in `tests/fixtures/`. No test should require the real S3 dataset.

## Starter guidance

- **Start with the schema on a fixture, not the whole dataset.**
  A 20-row CSV that you can eyeball is much easier to iterate the
  schema against than the full partition.
- **Read `pandera`'s error messages carefully.** With `lazy=True`
  the `SchemaErrors` object has a `.failure_cases` DataFrame — that
  is your entire debugging surface. Print it.
- **The GX v1.x API changed materially in 2024.** If a tutorial you
  find uses `context.add_expectation_suite(...)` or `BatchRequest`,
  it is v0.18-era; the v1.x equivalents are `context.suites.add(...)`
  and `batch_definition.get_batch(...)`. Trust the official docs
  over Stack Overflow.
- **Use `scipy.stats.ks_2samp`** rather than rolling your own KS.
  Its return is a named tuple `(statistic, pvalue)`.
- **For PSI, watch the log-zero trap.** Any bin with zero count in
  either distribution will make `ln(p_batch/p_base)` blow up. Add a
  small floor (e.g. `max(p, 1e-4)`) and document the floor.
- **Version the baseline by ISO date in a directory name.** Do not
  pickle a dict named `baseline.pkl` and overwrite it — you will
  regret it the first time you want to reproduce a drift alert.
- **Test-driven schema.** Write the "rejects missing column" test
  *first*, then add the column to the schema, then watch the test
  turn green. It clarifies the schema's job.

## Acceptance criteria

- `TrainingBatch` schema exists, has `strict=True`, and encodes
  dtype, non-null, range, enum, and uniqueness constraints for
  every column of the dataset. At least two custom checks
  (`@pa.check` or `@pa.dataframe_check`) are present.
- `compute_baseline` writes a versioned baseline directory with
  per-column stats and a copy of the schema module.
- `check_drift` returns per-column PSI / KS / chi-square results
  and an overall severity. The thresholds are function arguments,
  not magic numbers in the code.
- A GX suite named `training_batch_suite` is committed
  (`gx/gx/expectations/training_batch_suite.json` or equivalent)
  and includes at least one distribution-shape expectation
  (`ExpectColumnMeanToBeBetween` or
  `ExpectColumnKlDivergenceToBeLessThan`). `gx/uncommitted/` is
  in `.gitignore`; no HTML data docs are committed.
- `validate_batch(df, ...)` is the single call the training entry
  point makes. It logs its severity and snapshots any failing
  batch under `failed_batches/{timestamp}/`.
- All eight listed tests pass. `pytest -q` completes in under 30
  seconds on the fixture data.
- `REPORT.md` includes (a) the paragraph on `lazy=True`, (b) the
  pandera-vs-GX comparison paragraph, and (c) the reflection
  answers below.

## Reflection questions (answer in `REPORT.md`)

- Pick two chapter-10 pitfalls (silence, over-tight thresholds,
  committed GX docs, baseline overwrite, coupling to training-only,
  schema-in-a-notebook). For each, describe how your solution avoids
  it.
- Suppose your baseline is 90 days old. Which of your checks would
  begin to false-positive first — schema, PSI, KS, or chi-square?
  What is the operational rule you'd add to prevent that class of
  false alert?
- If this validator ran on every hourly ingest (not just on
  training), which of your checks would you *disable* to keep the
  false-alert rate tolerable? What would you keep, no matter what?
- The failure snapshot in Part E is a promise. Sketch, in three
  bullets, what you would do with `failed_batches/` after a month
  of running the pipeline. (Chapter 10 hints at the answer.)

## Stretch goals (optional)

- **`evidently` or `whylogs` comparison.** Reproduce your PSI / KS
  results with `evidently.report.Report([DataDriftPreset()])`.
  Compare the numbers to yours; investigate any disagreement.
- **CI wiring.** Add a GitHub Actions workflow that runs
  `pytest` on every push and additionally runs `validate_batch`
  against a committed 100-row fixture batch. Fail the build on
  a validation error. Publish the GX Data Docs to
  `gh-pages` or an S3 bucket.
- **Serving-side gate.** Add a `validate_request(row: dict)`
  function that applies the *columnwise* subset of the schema to a
  single row at prediction time. Assert that a batch of these rows
  produces the same schema errors as
  `TrainingBatch.validate(pd.DataFrame(rows))`.
- **Slack / PagerDuty alert.** Wrap `validate_batch`'s severe-
  failure path in a call to a mock alerting sink. Test that a
  severe drift call produces exactly one alert with the schema
  failure list and the snapshot URI.
- **A `revalidate_all_partitions.py` script** that walks the
  partitioned Parquet output from exercise-01 and runs the gate
  against each partition, producing a wall-of-green (or wall-of-
  red) summary. This is the smallest useful drift dashboard.

## What to hand in

```
exercise-04/
├── validation/
│   ├── schema.py               # pa.DataFrameModel + custom checks
│   ├── baseline.py             # compute_baseline(df) -> dict + writer
│   ├── drift.py                # PSI / KS / chi-square + DriftReport
│   └── gate.py                 # validate_batch(df, ...)
├── gx/
│   ├── gx/                     # GX project (suites, checkpoints)
│   └── .gitignore              # ignores uncommitted/ and built docs
├── baselines/
│   └── training_batch/
│       └── 2026-09-01/
│           ├── column_stats.json
│           ├── pandera_schema.py
│           └── row_count.txt
├── failed_batches/             # gitignored; the gate writes here
├── tests/
│   ├── fixtures/               # tiny CSVs
│   ├── test_schema.py
│   ├── test_baseline.py
│   ├── test_drift.py
│   └── test_gate.py
├── pyproject.toml
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
