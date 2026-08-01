# 01 — Why data engineering is most of ML

> **Reading time:** ~15 minutes.
> **Prerequisite modules:** `mod-101` (foundations), `mod-102` (Python
> toolchain).

## Motivation

`mod-102` closed with a working scikit-learn `Pipeline` that trained a
model on a clean, tidy DataFrame someone else had prepared. That
"someone else" was doing the majority of the actual work.

In real ML projects the labour split rarely favours modelling. The
data does not arrive as a tidy DataFrame; it arrives as:

- half a dozen CSVs someone dropped in a bucket, some with a header
  row and some without;
- a production Postgres you should not `SELECT *` from during trading
  hours;
- a Parquet dataset partitioned by day in S3 whose schema drifted at
  some point last quarter and no one remembers when;
- a JSON API you have to page through, retrying on rate limits;
- labels that are 3% wrong, 5% missing, and 100% attached to the
  wrong join key on rows written before a certain deploy.

Turning that into `(X, y)` — reliably, reproducibly, with the same
transformations at training time and at serving time, and with a
tripwire that fires when the upstream data changes shape — is
**data engineering for ML**. It is the topic of this module.

There is a widely repeated industry heuristic that data preparation
takes 80% of a data project's time. The exact percentage is folklore
and varies by survey, but every practitioner survey that has asked
the question puts data work well above modelling.<!-- needs-research:
verify the current AnacondA/Kaggle/Anaconda State of Data Science
survey number and cite it here with year and URL; only cite if a
recent survey (last 3 years) reports it. --> The number matters less
than the ordering: the model is not usually the bottleneck; the data
pipeline is.

## What this module teaches, and what it does not

The four learning objectives:

1. **Ingest** data from files, SQL, and object stores into ML-ready
   datasets.
2. **Profile and clean** messy data — missingness, outliers,
   leakage, label quality.
3. **Engineer features** for tabular, text, and image data.
4. **Validate** datasets with schema and distribution checks
   (`pandera`, Great Expectations).

Roughly one chapter per sub-topic; the map below.

| # | Chapter | Objective |
|---|---|---|
| 01 | This chapter | orientation |
| 02 | Ingesting from files (CSV, JSON, Parquet) | 1 |
| 03 | Ingesting from SQL | 1 |
| 04 | Ingesting from object stores (S3, GCS, Azure Blob) | 1 |
| 05 | Profiling and cleaning: missingness, outliers, duplicates | 2 |
| 06 | Leakage and label quality | 2 |
| 07 | Tabular feature engineering | 3 |
| 08 | Text feature engineering | 3 |
| 09 | Image feature engineering | 3 |
| 10 | Dataset validation: `pandera` and Great Expectations | 4 |

This module is deliberately **not** a data engineering course in the
"batch orchestration and warehouse modelling" sense. It does not
cover Airflow / Dagster / Prefect DAG authoring, Spark or Flink
distributed compute, Kafka streaming, dbt transformations, or
warehouse modelling patterns (dimensional / one-big-table). Those
are the discipline of a *data engineer*; the ML engineer consumes
their output. What we cover here is the layer where their output
meets your `Pipeline`.

Related-but-out-of-scope topics we point at but don't teach:

- **Feature stores** (Feast, Tecton, Databricks Feature Store). A
  production concern that becomes important when the same feature is
  computed by more than one model or served in real time. Covered
  briefly in `mod-108` (deployment).
- **Data versioning** (DVC, LakeFS, Delta Lake, Iceberg). Referenced
  in `mod-106` (experiment tracking and reproducibility).
- **Drift monitoring in production.** Chapter 10 teaches the
  *validators*; `mod-109` teaches the *monitors*.
- **Full data-labelling programs** (Label Studio, Scale, Snorkel).
  Chapter 06 covers the ML-side symptoms of bad labels; running a
  labelling pipeline is a discipline of its own.

## A working definition of "ML-ready"

Throughout this module the goal of ingestion + cleaning + feature
engineering is to produce something that is **ML-ready**. Five
properties, each of which the module works through in turn:

1. **Correctly typed.** Every column has a known, stable dtype.
   Categoricals are `category` or `string`, timestamps are timezone-
   aware `datetime64[ns, UTC]`, integers with missing values use the
   nullable `Int64` — not `object`, not silently upcast to `float64`.
2. **Cleanly joined.** The row grain is documented (one row per
   what?). Join keys are unique on at least one side and validated.
   Cardinality assumptions are enforced by `merge(..., validate=)`
   rather than left to hope.
3. **Free of the labels you are trying to predict.** Nothing in `X`
   was computed using information that would not have been available
   at prediction time — the target itself, downstream summaries of
   it, or features from the future.
4. **Split reproducibly.** The train / validation / test split is
   deterministic given a seed, respects the natural grouping of the
   data (user ID, session, time), and is done *before* any statistic
   is computed on the data.
5. **Verifiable.** Someone reading the code (or an automated check
   in CI) can point at each column and say "here is the type
   expected, here is the range expected, here is what happens if the
   next batch violates it". Chapter 10's territory.

A dataset that satisfies all five is a dataset a scikit-learn
`Pipeline` can chew through without hidden surprises. The rest of
this module is technique for getting there.

## The train / serve skew problem, previewed

The single largest class of production ML incidents is not "the
model was wrong"; it is "the data at serving time did not look like
the data at training time." Google's *Rules of Machine Learning*
(rule #29) puts it plainly: *"the best way to make sure you train
like you serve is to save the set of features used at serving time,
then pipe those features to a log to use them at training time."*

Concretely, the failure modes we will keep coming back to:

- A feature is computed differently in the training notebook than in
  the serving code (e.g. `"USA"` vs `"US"`, timezone-naive vs
  timezone-aware timestamps, `NaN` vs empty string).
- A field's schema changes upstream (new enum value, renamed column,
  units change) and the pipeline silently accepts it.
- A distribution shifts (median session length halves overnight)
  and the trained model degrades without anyone noticing.

Chapter 06 covers leakage — the *training-time* version of skew.
Chapter 10 covers validation — the *runtime* tripwires. Both are
about the same underlying principle: **the code that transforms data
into features must be a single, versioned artefact, and the data
flowing through it must be checked against a schema you can point
at.**

## How to read the module

The chapters are ordered so each builds on the last, but they are
not equally load-bearing:

- Chapters 02–04 are reference material with idioms. Skim if you
  already know how to `pd.read_parquet` from S3 with credentials;
  the interesting pages are 03's "chunking" and 04's "partitioning
  and `pyarrow.dataset`".
- Chapter 05 is a workflow chapter. Read it. Profiling is the step
  that most non-DE engineers skip and then pay for.
- Chapter 06 is the highest-stakes chapter in the module. Leakage
  is the bug that will silently invalidate your metrics, sometimes
  for a whole quarter. Slow down here.
- Chapters 07–09 are technique catalogues per modality. Read the
  one you will use next; the other two can wait.
- Chapter 10 turns everything before it into a check you can wire
  into CI. It is what promotes the module's output from "a
  notebook" to "a pipeline".

The exercises follow the same arc: ingest, audit, engineer, validate.
Do them in order.

## Summary

- Data preparation is the majority of an ML project's time. The
  model is rarely the bottleneck.
- This module produces **ML-ready** data: correctly typed, cleanly
  joined, leakage-free, reproducibly split, and verifiable.
- Train / serve skew is the underlying incident class the module is
  organised to prevent, from both sides — leakage at training time
  and drift at serving time.
- Ten chapters, four objectives, four exercises. Chapter 06
  (leakage) and chapter 10 (validation) are the two you cannot
  skip.

## Where this goes next

Chapter 02 opens the ingestion arc with the most common source of
ML data: files. CSV, JSON, and Parquet each have their own hazards
and idioms.
