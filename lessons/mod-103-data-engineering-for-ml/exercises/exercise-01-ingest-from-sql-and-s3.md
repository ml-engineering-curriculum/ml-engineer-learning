# exercise-01 — Ingest from SQL and S3

> **Estimated effort:** 3 hours.
> **Companion chapters:** [02 — Ingesting from files](../02-ingesting-from-files.md), [03 — Ingesting from SQL](../03-ingesting-from-sql.md), [04 — Ingesting from object stores](../04-ingesting-from-object-stores.md).

## Objective

Build a small, reproducible ingestion module that pulls a dataset
from *two* sources — a SQL database and an S3-compatible object
store — joins them on a natural key, and produces a single
partitioned Parquet dataset that downstream exercises can consume.

By the end you should have:

- A `data.py` module with typed, parameterised functions that read
  each source.
- A `pipeline.py` that orchestrates the join and writes the
  resulting dataset.
- Enough tests to catch schema drift on the next run.

The whole thing should run end-to-end in a fresh Python
environment with `uv sync && python -m ingestion.run` (or
equivalent).

## Prerequisites

- Python 3.11+, Pandas ≥ 2.1, PyArrow ≥ 15, SQLAlchemy 2.x,
  `psycopg[binary]` (or the driver of your choice), `fsspec`, and
  `s3fs` (or `gcsfs` / `adlfs` for GCP / Azure), `pytest`.
- Chapters 02–04 of this module.
- Access to *some* SQL database and *some* object store. Options
  in ascending order of production-like realism:
  - **Local:** a Postgres container and a `minio` container
    started via `docker compose`. Recommended for a first pass.
  - **Cloud sandbox:** a free-tier RDS instance and a
    personal-account S3 bucket. Realistic; be mindful of cost.
  - **Warehouse:** Snowflake / BigQuery free trial + a free-tier
    bucket. Closest to a production ML stack.

If you go the local route, a minimal `docker-compose.yml` (Postgres
+ MinIO) is a fair 15 minutes of setup and saves you from cloud
billing for the exercise. Include it in your solution.

## Dataset choice

You need two related sources so a join is meaningful. Two safe
templates:

- **E-commerce.** `users` in Postgres (id, signup_date, country,
  plan). `orders` as NDJSON files in the bucket (order_id, user_id,
  ts, sku, amount). Grain of the joined table: one row per order,
  with user attributes attached.
- **IoT / events.** `devices` in Postgres (device_id, model,
  installed_at). `readings` as Parquet files partitioned by
  `dt=YYYY-MM-DD/` in the bucket (device_id, ts, temperature,
  humidity, battery). Grain: one row per reading, with device
  metadata attached.

Use a real public dataset if you prefer:

- The [Chicago Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
  data (public, large, has trip + taxi metadata) — split it
  yourself into "trips in S3, taxis in Postgres".
- The [NYC Yellow Taxi](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  Parquet dataset in S3 (public, requester-pays disabled).

Whichever you choose, call it out in the top of your
`REPORT.md` — including row counts and time range.

## Problem statement

### Part A — SQL source (≈ 45 min)

In `data/sql.py`, write:

```python
def read_users(
    engine: sqlalchemy.Engine,
    *,
    countries: Sequence[str] | None = None,
    signed_up_after: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ...
```

Requirements:

1. Use `sqlalchemy.text` with **bound parameters** for the filters.
   Do not build the SQL string with f-strings.
2. `SELECT` only the columns you need; never `SELECT *`.
3. Pass `parse_dates={"signup_date": {"utc": True}}` (or the
   equivalent for your driver) so all timestamps come back
   `datetime64[ns, UTC]`.
4. Cast `country` to `"category"` and any string-ID column to
   `"string"` before returning. No `object`-dtype columns.
5. Log the row count returned and the elapsed time.

Credentials must come from environment variables (or a local `.env`
file, gitignored). No password strings in the source.

### Part B — Object-store source (≈ 45 min)

In `data/store.py`, write:

```python
def read_orders(
    root: str,                    # "s3://bucket/orders/" or "s3://.../nyc-taxi/"
    *,
    date_from: str,               # ISO date
    date_to:   str,
    columns:   Sequence[str] | None = None,
    storage_options: dict | None = None,
) -> pd.DataFrame:
    ...
```

Requirements:

1. Use `pyarrow.dataset` (or `pd.read_parquet` on the directory)
   with **partition pruning** on the `date` (or equivalent)
   partition. Assert in a comment that you understand which files
   will be opened.
2. Use **column projection** via `columns=` — only read the
   columns you'll use downstream.
3. Read credentials via the SDK's default chain (env vars,
   instance role, `aws sso login` cache). Do not hardcode.
4. Log the number of files considered vs. actually read (i.e. the
   pruning did something).
5. Return a DataFrame with a documented dtype for every column.
   Prefer `dtype_backend="pyarrow"`.

If your source is NDJSON rather than Parquet, `read_orders` should
first convert to a partitioned Parquet dataset at
`{root}/_cache/parquet/` on first read and use that thereafter.
Log the one-time conversion.

### Part C — Join and validate (≈ 45 min)

In `pipeline.py`, write:

```python
def build_dataset(
    users: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    ...
```

Requirements:

1. Merge with `merge(..., how="left", validate="many_to_one")` on
   the natural key (orders many to users one). Fail loud if the
   cardinality assumption is wrong.
2. Assert **no row loss** on the "many" side (`len(result) ==
   len(orders)`). If there is loss (unmatched keys), report the
   count and a sample of unmatched keys in the log before failing.
3. Handle timezones: every timestamp in the output must be
   `datetime64[ns, UTC]`.
4. Add two derived columns: `join_age_days` (order timestamp minus
   user signup date, in days) and `is_first_order` (bool). These
   are for chapter 07 to build on; use them to exercise your
   understanding of point-in-time correctness.

### Part D — Write partitioned Parquet output (≈ 30 min)

Write the joined dataset to:

```
data/processed/dataset/
  date=YYYY-MM-DD/
    part-00000.parquet
    part-00001.parquet
  ...
  _SUCCESS
```

Requirements:

1. Use `df.to_parquet(..., partition_cols=["date"], engine="pyarrow")`
   or `pyarrow.dataset.write_dataset`.
2. Do not include `date` inside the file bodies — it lives in the
   directory names.
3. Write to a `_tmp/` prefix first, then atomically rename to the
   final path (or write a `_SUCCESS` marker) so a downstream
   consumer never sees a half-written dataset.
4. Log the number of partitions, total row count, and total bytes
   written.

### Part E — Tests (≈ 15 min)

In `tests/`, cover the following minimum set:

- `test_read_users_returns_expected_dtypes`
- `test_read_orders_prunes_by_date` (mock or use a small local
  dataset)
- `test_build_dataset_no_row_loss`
- `test_build_dataset_fails_on_duplicate_users` (constructed
  fixture with a duplicate `user_id`)
- `test_pipeline_writes_success_marker`

For the SQL tests, either spin up a Postgres via `testcontainers`,
or use SQLite as a test-only backend and skip the Postgres-specific
tests.

## Starter guidance

- **Do Part D last.** Get the join right, verified with a print /
  head, before you introduce the write-and-partition step.
- **Log everything.** Ingestion errors are notoriously hard to
  reproduce; a log with row counts, byte counts, filter values,
  and elapsed times is the debugging friend you'll thank yourself
  for.
- **Add a top-level `--dry-run` flag** to `pipeline.py` that runs
  everything except the final write. Speeds up iteration.
- Use `sqlalchemy.URL.create` (not string interpolation) to build
  the DB URL from environment variables. See chapter 03.
- For the S3-local dev loop, MinIO's endpoint URL goes into
  `storage_options={"endpoint_url": "http://localhost:9000",
  "key": "...", "secret": "..."}`.
- Add `pool_pre_ping=True` on your SQLAlchemy engine. Prevents
  the "stale connection" error the second time you run in a
  long-lived REPL.

## Acceptance criteria

- The ingestion runs end-to-end from a fresh clone (given the
  documented environment variables and, if applicable,
  `docker-compose up`).
- No credentials appear in the source, and `git grep -i 'password\|secret\|access_key'`
  returns only test doubles.
- The SQL read uses bound parameters (not string interpolation)
  and returns a DataFrame with an explicit dtype per column.
- The S3 read uses partition pruning and column projection. A
  reasonable `date_from` / `date_to` window opens fewer files
  than the whole dataset (verified in the log).
- The join uses `validate="many_to_one"` (or the appropriate
  cardinality). A constructed fixture with a duplicate primary key
  causes the pipeline to fail loudly.
- The output is a partitioned Parquet dataset with a `_SUCCESS`
  marker (or equivalent atomic-write convention). Reading it with
  `pyarrow.dataset(...)` round-trips to the expected schema and
  row count.
- All five listed tests pass.
- `REPORT.md` records: the dataset chosen, the final row and
  partition counts, one measured wall-clock time for the pipeline,
  and the answer to the reflection questions below.

## Reflection questions (answer in `REPORT.md`)

- Which step was the slowest? Was it the SQL query, the object-
  store list, the JSON→Parquet conversion, or the join?
- If your `orders` table had 100× more rows, which of your steps
  would break first (memory, time, connection pool, S3 rate
  limit)? What is the smallest change that would fix it?
- Where does the `date` in your partition column come from — event
  time or ingestion time? What would break if it were the wrong
  one?

## Stretch goals (optional)

- Add a **schema registry**: write the observed schema (column
  name → dtype) to `schema.json` on the first successful run;
  compare against it on subsequent runs and fail if drifted. This
  is a hand-rolled pandera preview.
- Wrap the whole thing in a **CLI** (`typer` or `click`) with
  `--date-from`, `--date-to`, `--dry-run`, `--output-uri`.
- Add a **manifest** file listing the exact S3 keys read on each
  run, and version the manifest. Chapter 10 will use this for
  reproducibility.
- Swap the S3 backend for **GCS** or **Azure Blob** by changing
  only the URL scheme and `storage_options`. Confirm nothing else
  in your code needs to change. This is what `fsspec` buys you.
- Add a **retry with exponential backoff** around the S3 read for
  transient 5xx errors. See `tenacity` or `backoff`.

## What to hand in

```
exercise-01/
├── data/
│   ├── sql.py                    # read_users(...)
│   └── store.py                  # read_orders(...)
├── pipeline.py                   # build_dataset(...) + orchestration
├── tests/
│   ├── test_sql.py
│   ├── test_store.py
│   └── test_pipeline.py
├── docker-compose.yml            # if using the local option
├── pyproject.toml                # dependencies (mod-102 chapter 07)
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
