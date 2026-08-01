# mod-103 — Data Engineering for ML: Ingestion, Cleaning, Feature Engineering

> **Track:** Machine Learning Engineer (level 20)
> **Estimated effort:** 14 hours

This module covers the layer where "the source of truth" — files,
databases, object stores — meets your scikit-learn `Pipeline`.
`mod-102` gave you the tools; `mod-103` puts them to work on data
that does not arrive as a tidy DataFrame. The four objectives are
ingestion, profiling and cleaning, feature engineering across
modalities, and dataset validation.

Data preparation is the majority of an ML project. The model is
rarely the bottleneck; the data pipeline is. This module builds
the pipeline.

## Learning objectives

- Ingest data from files, SQL, and object stores into ML-ready
  datasets.
- Profile and clean messy data: missingness, outliers, leakage,
  label quality.
- Engineer features for tabular, text, and image data.
- Validate datasets with schema and distribution checks (Great
  Expectations, `pandera`).

## Chapters

| # | Chapter | Maps to objective |
|---|---|---|
| 01 | [Why data engineering is most of ML](01-why-data-engineering-is-most-of-ml.md) | orientation |
| 02 | [Ingesting from files: CSV, JSON, Parquet](02-ingesting-from-files.md) | ingestion |
| 03 | [Ingesting from SQL](03-ingesting-from-sql.md) | ingestion |
| 04 | [Ingesting from object stores](04-ingesting-from-object-stores.md) | ingestion |
| 05 | [Profiling and cleaning: missingness, outliers, duplicates](05-profiling-and-cleaning-messy-data.md) | profile & clean |
| 06 | [Leakage and label quality](06-leakage-and-label-quality.md) | profile & clean |
| 07 | [Tabular feature engineering](07-tabular-feature-engineering.md) | feature engineering |
| 08 | [Text feature engineering](08-text-feature-engineering.md) | feature engineering |
| 09 | [Image feature engineering](09-image-feature-engineering.md) | feature engineering |
| 10 | [Dataset validation: `pandera` and Great Expectations](10-dataset-validation.md) | validation |

## Exercises

| # | Exercise | Hours |
|---|---|---|
| 01 | [Ingest from SQL and S3](exercises/exercise-01-ingest-from-sql-and-s3.md) | 3 |
| 02 | [Data quality audit](exercises/exercise-02-data-quality-audit.md) | 3 |
| 03 | [Tabular feature engineering](exercises/exercise-03-tabular-feature-engineering.md) | 3 |
| 04 | [Schema and distribution validation](exercises/exercise-04-schema-and-distribution-validation.md) | 3 |

Plus one lab (`labs/`) and one quiz (`quizzes/`) — scaffolded on
subsequent authoring cycles.

## How to use this module

Read the chapters in order — later ones assume earlier ones. The
two highest-stakes chapters, which are worth slowing down for, are:

- **06 (Leakage and label quality).** The single largest source of
  "the model looked great in the notebook and doesn't work in
  production" outcomes.
- **10 (Dataset validation).** The mechanism that keeps the module's
  output from silently rotting when upstream data changes.

The three feature-engineering chapters (07, 08, 09) are per-modality
technique catalogues. Read the one you will use next; the other two
can wait.

Each chapter maps to at least one exercise. The exercises are
sequenced to build a working end-to-end data pipeline: ingest from
raw sources, audit the result, engineer features from it, validate
the whole thing. Do them in order.

## What this module does **not** cover

- **Batch orchestration** (Airflow, Dagster, Prefect) and the DAG-
  authoring discipline. That is data engineering proper.
- **Distributed compute** (Spark, Flink, Dask). We stay in
  Pandas / PyArrow / Polars territory and call out where to move
  when you outgrow it.
- **Streaming** (Kafka, Flink, Kinesis). Batch is the shape of
  most ML training data.
- **Data warehousing** (dbt, Snowflake modelling patterns). We
  consume the warehouse; the warehouse team produces it.
- **Feature stores** (Feast, Tecton, Databricks Feature Store) —
  briefly named where the pattern comes up. Deep coverage lives in
  `mod-108`.
- **Data versioning** (DVC, LakeFS, Delta, Iceberg). Named in
  chapter 04; explored in `mod-106`.
- **Production drift monitoring.** Chapter 10 teaches the
  *validators*; `mod-109` teaches the *monitors* and the on-call
  workflow.
- **Full data labelling programs.** Chapter 06 covers the
  ML-side symptoms of bad labels; running a labelling pipeline is
  a discipline of its own.

## Prerequisites

- `mod-101` (ML foundations) and `mod-102` (Python ML toolchain).
  Chapters here assume you can write a scikit-learn `Pipeline` and
  read a Pandas DataFrame.

See [`resources.md`](resources.md) for official documentation and
curated references behind each chapter.
