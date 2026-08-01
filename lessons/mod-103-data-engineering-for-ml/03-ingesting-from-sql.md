# 03 — Ingesting from SQL

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01, 02.

## Motivation

A large fraction of the data an ML engineer wants lives in a
relational database or a warehouse — Postgres, MySQL, SQL Server,
Snowflake, BigQuery, Redshift, Databricks SQL. The right way to pull
it out is not "run the query, dump to CSV, `read_csv` the result".
The right way is a small, well-instrumented Python function that
holds a connection, parameterises its query, chunks large results,
and hands you a typed DataFrame.

This chapter covers the mechanics — `SQLAlchemy`, `pd.read_sql`,
chunking, and the etiquette of running analytical queries against a
production database.

## The three-layer stack

Working with SQL from Pandas involves three layers, each of which
you will occasionally have to look at directly:

1. **A DB-API 2 driver.** The Python package that speaks the wire
   protocol. Examples: `psycopg` (Postgres), `pymysql`, `pyodbc`
   (SQL Server), `snowflake-connector-python`,
   `google-cloud-bigquery`.
2. **SQLAlchemy.** A connection pool + SQL toolkit that wraps the
   driver behind a URL scheme. Provides the `Engine` object every
   other Python data tool expects.
3. **Pandas** (or Polars / PyArrow / DuckDB). The thing that runs
   the query and returns a DataFrame.

The minimum viable path:

```python
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine("postgresql+psycopg://user:pass@host:5432/db")

with engine.connect() as conn:
    df = pd.read_sql(text("SELECT id, ts, amount FROM events LIMIT 1000"), conn)
```

The URL scheme (`postgresql+psycopg`) is `<dialect>+<driver>` — the
dialect is the SQL flavour SQLAlchemy speaks; the driver is the
DB-API 2 package that carries the bytes. Install both.

Do not build the connection URL by string-concatenating a password
into a literal — see "credentials" below.

## `pd.read_sql`: the three signatures

`pd.read_sql` dispatches based on its first argument:

- **A SQL string.** Runs the query, returns a DataFrame.
- **A table name.** Reads the whole table via `SELECT *`.
- **A `sqlalchemy` `Selectable` object** (a Core query). Runs the
  compiled query, returns a DataFrame.

In practice, always use the first form and always wrap the string in
`sqlalchemy.text(...)`:

```python
from sqlalchemy import text
query = text("""
    SELECT user_id, ts, amount
    FROM   events
    WHERE  ts >= :start AND ts < :end
""")
df = pd.read_sql(query, conn, params={"start": start_ts, "end": end_ts})
```

Two things to notice:

- `:start` / `:end` are **bound parameters**, not string
  substitution. The driver escapes them; SQL injection is not
  possible.
- `SELECT` only the columns you need. `SELECT *` on a warehouse
  fact table pulls hundreds of columns you will immediately drop.

### Parse dates and set types on read

`pd.read_sql` accepts `parse_dates=` and (via `dtype=` in recent
Pandas versions) explicit typing. Use them.

```python
df = pd.read_sql(
    query, conn,
    params={"start": start_ts, "end": end_ts},
    parse_dates={"ts": {"utc": True}},
)
```

Timestamp handling deserves a paragraph of its own: databases
usually store timestamps as `TIMESTAMP` (naive) or `TIMESTAMPTZ`
(zoned) with different behaviours across vendors. Force UTC on read
and treat every downstream timestamp as `datetime64[ns, UTC]`. It
saves a whole class of "off by a day because of DST" bugs.

## Chunking large queries

`pd.read_sql(..., chunksize=N)` returns an iterator of DataFrames of
`N` rows each rather than materialising the entire result set in
memory:

```python
for chunk in pd.read_sql(query, conn, chunksize=100_000):
    write_partition(chunk)
```

Two things to know:

- The database keeps the cursor open for the whole loop. On some
  servers this holds a snapshot / lock; keep the work inside the
  loop small.
- Server-side vs. client-side cursors differ by driver. On
  Postgres, wrap the connection in a server-side cursor
  (`conn.execution_options(stream_results=True)`) or the driver
  will actually buffer the whole result set in the client anyway.
  Similar mechanisms exist for MySQL (`SSCursor`) and MSSQL
  (`fast_executemany`).

For anything larger than "millions of rows", prefer to have the
database itself write to a Parquet dataset via `UNLOAD` (Redshift),
`COPY INTO` (Snowflake), `EXPORT DATA` (BigQuery), or an object-store
sink, and then ingest with chapter 04's tools. Pulling terabytes
through Python one row at a time is nobody's idea of a good time.

## Warehouse-specific paths

Every major warehouse has an optimised Python client that avoids the
row-at-a-time SQLAlchemy path for large reads. Use them.

- **BigQuery.** `pandas_gbq.read_gbq(...)` or the newer
  `google-cloud-bigquery` client's `.result().to_dataframe(bqstorage_client=...)`
  which streams via the BigQuery Storage API.
- **Snowflake.** `snowflake.connector.pandas_tools.fetch_pandas_all()`
  after a `cursor.execute(...)`. Uses Arrow under the hood.
- **Databricks / Spark SQL.** `databricks.sql.connect(...)` returns
  Arrow batches; `.fetch_arrow_all().to_pandas()`.
- **Redshift.** Use `psycopg` or the Redshift Data API; for big
  extracts, `UNLOAD` to S3 and read the resulting Parquet.

If a warehouse offers an Arrow-native read path, it is essentially
always faster than the DB-API row-based path for wide analytical
tables — often by an order of magnitude.

## Credentials, done right

Two rules and one tool.

**Rule 1 — never put a password in your source code, notebook, or
config that ships in the repository.** This includes URLs that
embed the password (`postgresql://user:hunter2@...`). The password
ends up in your shell history, in tracebacks, in log output, and in
that one PR you accidentally opened.

**Rule 2 — never build the connection URL with an f-string that
contains a password.** Pass credentials to `create_engine` as
separate arguments, or store the URL in a secret manager and read
it as a whole.

The tool: your environment's secret manager.

```python
import os
from sqlalchemy import URL, create_engine

url = URL.create(
    drivername="postgresql+psycopg",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],   # from your secret store
    host=os.environ["DB_HOST"],
    database=os.environ["DB_NAME"],
    port=5432,
)
engine = create_engine(url, pool_pre_ping=True)
```

`sqlalchemy.URL.create` URL-encodes the password correctly (a
`hunter#2` password does not become an invalid URL). `pool_pre_ping=True`
issues a lightweight ping before checking out a connection from the
pool, which sidesteps the "stale connection" error that appears
whenever the DB has closed an idle connection out from under you.

For production, the secret manager itself matters:

- **AWS**: Secrets Manager, SSM Parameter Store.
- **GCP**: Secret Manager.
- **Azure**: Key Vault.
- **Kubernetes**: `Secret` resources, ideally provisioned by
  External Secrets Operator from one of the above.
- **Local dev**: `.env` files loaded via `python-dotenv`, listed in
  `.gitignore`.

Your training job should read the secret at startup and never write
it anywhere but its own process memory.

## Being a good neighbour on a shared database

An ML engineer's ad-hoc query on the production OLTP database is a
classic incident cause. Two etiquette rules save weekends.

1. **Read from a replica, not the primary.** Every production
   database of any size has a read replica. Use it. Warehouses
   like Snowflake and BigQuery are always the right answer for
   analytical reads; OLTP replicas are the right answer for
   application-facing data.
2. **Give your queries a statement timeout.** Postgres:
   `SET LOCAL statement_timeout = '30s'`. It turns "I accidentally
   ran a five-hour cross join" into "my query got killed after 30
   seconds and no one else noticed".

The friendlier version of both rules: **do not do analytical work
on an OLTP database at all.** ETL a snapshot to your warehouse (or
object store, chapter 04) and query that.

## SQL patterns that come up in ML ingestion

A short list of query shapes that will show up in the exercise and
in your job.

### Windowed feature aggregations (do them in SQL)

Rolling counts and sums per user over the last 30 days are much
faster in SQL than in Pandas:

```sql
SELECT
  user_id,
  COUNT(*)      OVER w AS events_30d,
  SUM(amount)   OVER w AS spend_30d,
  AVG(amount)   OVER w AS avg_amount_30d
FROM events
WINDOW w AS (
  PARTITION BY user_id
  ORDER BY ts
  RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
)
```

Compute the aggregations in the warehouse; return the aggregates,
not the raw rows.

### Point-in-time joins

The bane of ML feature engineering: for each label row `(user, t)`,
join the features that were current at time `t` — never later, never
computed with data from after `t`. In SQL:

```sql
SELECT l.user_id, l.event_ts, u.plan, u.country
FROM   labels  l
JOIN   users u
  ON   u.user_id = l.user_id
  AND  u.effective_from <= l.event_ts
  AND  (u.effective_to  >  l.event_ts OR u.effective_to IS NULL)
```

Chapter 06 comes back to point-in-time correctness at length; the
SQL pattern above is one of the most important tools in the box.

### Deduplication with a window

Warehouse events tables are often not perfectly unique. To keep the
latest row per key:

```sql
SELECT * FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingested_at DESC) AS rn
  FROM raw.events
) WHERE rn = 1
```

Do this once at the ingestion boundary; downstream consumers should
never see duplicates.

## Pitfalls collected

1. **Trusting inferred dtypes.** SQLAlchemy + Pandas will pick
   reasonable dtypes, but `NUMERIC` columns become `object`-dtype
   `Decimal`, `BIGINT` columns with nulls become `float64`, and
   `DATE` columns come back as `datetime64[ns]` (not `date`). Cast
   after reading, or pass `dtype=` on `read_sql`.
2. **Reading whole tables via `pd.read_sql_table`.** It exists;
   don't use it. Always write a `SELECT` that lists your columns
   and applies a `WHERE`.
3. **Leaving a connection unclosed.** `with engine.connect() as
   conn:` is the safe form. A bare `engine.connect()` returned from
   a helper leaks connections and eventually exhausts the pool.
4. **Timezones on `TIMESTAMP` columns.** A naive `TIMESTAMP` column
   is silently treated as the server's local time by some drivers.
   Force UTC on read; convert once at the boundary.
5. **Query plan sensitivity to bound parameter types.** Warehouses
   have query planners that occasionally do very different things
   for `WHERE ts >= '2026-08-01'` (string) vs. `WHERE ts >=
   TIMESTAMP '2026-08-01'` (typed). If a query is inexplicably
   slow, `EXPLAIN` and check the parameter types.
6. **Locale-dependent identifiers.** MySQL is case-insensitive on
   table names on some platforms, case-sensitive on others. Postgres
   folds unquoted identifiers to lowercase. Snowflake folds to
   upper. When a query works locally and fails in prod, this is
   often why.

## Summary

- The SQL ingestion stack: driver → SQLAlchemy → Pandas. Know all
  three well enough to debug each.
- `pd.read_sql(text(query), conn, params={...})` with explicit
  columns, parameterised filters, and `parse_dates=` is the shape
  of a well-behaved read.
- For large results, use `chunksize=` with a server-side cursor,
  or delegate the extract to the warehouse's own `UNLOAD` / `COPY
  INTO` / `EXPORT DATA` mechanism and ingest the resulting Parquet.
- For the major warehouses, use the Arrow-native client path
  (`bqstorage`, `snowflake.pandas_tools`, Databricks SQL Arrow
  batches). Order-of-magnitude faster than DB-API row iteration.
- Credentials come from a secret manager, never from the source
  tree. Build the URL with `sqlalchemy.URL.create`.
- Never run heavy analytical queries against the OLTP primary; use
  a replica or the warehouse.

## Where this goes next

Chapter 04 covers the third major source of ML data: object stores
— S3, GCS, Azure Blob — where warehouses and data lakes actually
put their bytes.
