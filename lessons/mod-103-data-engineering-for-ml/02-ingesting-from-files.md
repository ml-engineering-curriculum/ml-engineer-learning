# 02 — Ingesting from files: CSV, JSON, Parquet

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 01, and `mod-102` chapter 03 (Pandas).

## Motivation

Most ML datasets first land on your disk as files. Even when the
system of record is a database or a data lake, the practical shape
of "give me a snapshot to model" is usually a directory of files
you `pd.read_*` into memory.

Three formats cover the overwhelming majority of ML file ingestion:

- **CSV** — text, universal, ambiguous. Fine for tiny files;
  hazardous for anything larger.
- **JSON / NDJSON** — text, semi-structured, common for API dumps
  and event streams.
- **Parquet** — binary, columnar, typed, compressed. The default
  choice for anything non-trivial.

This chapter is a working tour of reading and writing each, with the
hazards each one loves to inflict.

## CSV: use it when you must

`pandas.read_csv` is the most-used function in the Pandas API and
the source of a startling number of subtle bugs.

```python
import pandas as pd
df = pd.read_csv("events.csv")
```

That line has already made half a dozen assumptions on your behalf:

- The first row is the header.
- The delimiter is `,`.
- The encoding is UTF-8.
- Every column's dtype should be inferred from its contents.
- The string `"NA"`, `"NaN"`, `"null"`, `""`, and about thirteen
  others mean "missing"; the full list is in the docs under `na_values`.
- Leading zeros on `"01234"` will be silently stripped if the column
  parses as integer.

For an ML pipeline you want to eliminate the guessing:

```python
df = pd.read_csv(
    "events.csv",
    dtype={
        "user_id": "string",     # keep the leading zero on "0042"
        "amount":  "Float64",    # nullable
        "country": "category",
    },
    parse_dates=["ts"],
    keep_default_na=True,        # or list explicitly with na_values=
    na_values=["", "N/A", "unknown"],
    encoding="utf-8",
)
```

Two habits worth building on day one:

1. **Always pass an explicit `dtype=`** for any column whose type is
   load-bearing. Never trust inference for identifiers, phone
   numbers, or postal codes — anything that "looks like a number
   but isn't".
2. **Always specify `parse_dates=`** for timestamp columns. Pandas
   will not guess.

For large CSVs, `chunksize=` returns an iterator of DataFrames
rather than trying to load the whole file:

```python
for chunk in pd.read_csv("huge.csv", chunksize=100_000, dtype={...}):
    process(chunk)
```

If you find yourself reaching for `chunksize=` because the file is
too big for memory, that is a strong signal to switch formats
(Parquet) or tools (Polars, DuckDB, PyArrow). CSV was not designed
for terabyte data.

### CSV pitfalls collected

1. **Quotes inside fields.** A comma or newline inside a
   double-quoted field is legal CSV; a raw newline is not. Not all
   producers respect the quoting rules — the pathological cases
   need `pd.read_csv(..., engine="python", quoting=csv.QUOTE_...)`.
2. **BOM (`﻿`) at the start of the file.** Excel loves to emit
   these. Detect with `encoding="utf-8-sig"`.
3. **Mixed decimal separators.** European CSVs often use `;` as
   delimiter and `,` as the decimal point. `pd.read_csv(sep=";",
   decimal=",")`.
4. **The "1.0" surprise.** Integer columns with even a single
   missing value become `float64` under the legacy dtype. Use
   `dtype="Int64"` (capital `I`, nullable) if you need to preserve
   integer semantics.
5. **Line endings.** A CSV written on Windows and read on Linux with
   `newline=""` handling wrong can look one row shorter than
   expected. Let Pandas handle it (it does, usually); do not open
   the file yourself.

## JSON and NDJSON

Two conventions dominate:

- **JSON array** — a single top-level `[...]` of objects. Small
  payloads, API responses.
- **NDJSON / JSON Lines** — one JSON object per line, no wrapping
  array. Log files, streaming exports, most cloud-service dumps.

Pandas handles both:

```python
df = pd.read_json("events.json", orient="records")            # array of objects
df = pd.read_json("events.ndjson", lines=True)                # NDJSON
```

The gotcha with JSON is **nested fields**. A record like:

```json
{"user": {"id": 42, "country": "US"}, "actions": [{"type": "click"}, {"type": "buy"}]}
```

comes back with a `user` column holding a dict and an `actions`
column holding a list. Neither is directly usable as a feature.
`json_normalize` flattens the object tree:

```python
import json
from pandas import json_normalize

with open("events.ndjson") as fh:
    records = [json.loads(line) for line in fh]

df = json_normalize(
    records,
    record_path="actions",              # explode this array
    meta=[["user", "id"], ["user", "country"]],   # keep these parent fields
    record_prefix="action_",
)
# columns: action_type, user.id, user.country
```

For anything larger than a laptop's memory, or anything you plan to
join across many files, JSON is the wrong format. Convert once to
Parquet and read the Parquet from then on:

```python
pd.read_json("events.ndjson", lines=True).to_parquet("events.parquet")
```

## Parquet: the ML default

Parquet is what CSV should have been. It is:

- **Columnar** — reading three columns of a 200-column file reads
  three columns' bytes, not all 200.
- **Typed** — the schema (per-column dtype, including nested types)
  lives inside the file. No guessing.
- **Compressed** — Snappy by default; Zstd for a better
  ratio-per-CPU-second tradeoff.
- **Splittable** — a large file is stored as several row groups; a
  reader can jump to a specific one without scanning the rest.
- **Widely supported** — Pandas, PyArrow, Polars, DuckDB, Spark,
  BigQuery, Snowflake, Athena all read and write it natively.

```python
df.to_parquet("events.parquet", engine="pyarrow", compression="snappy")
df = pd.read_parquet("events.parquet", columns=["user_id", "ts", "amount"])
```

Note `columns=` on the read side — Parquet only fetches the columns
you ask for. On a wide table this can be a 10–50× IO reduction.

### Row groups and partitioning

A single Parquet file is internally split into **row groups** —
sub-tables of typically 100k–1M rows each. Each row group's columns
are stored contiguously, so a reader can decompress just the ones
you need.

A **partitioned dataset** goes one step further: the data is split
across many files in a directory tree named by column values.

```
events/
├── date=2026-07-30/
│   ├── part-0.parquet
│   └── part-1.parquet
├── date=2026-07-31/
│   └── part-0.parquet
└── date=2026-08-01/
    └── part-0.parquet
```

Readers that understand the partition scheme (PyArrow, Polars,
DuckDB, Pandas via PyArrow) can push a filter on `date` down into
directory pruning — they never open the files that could not match.

```python
import pyarrow.dataset as ds
dataset = ds.dataset("events/", format="parquet", partitioning="hive")
tbl = dataset.to_table(
    filter=(ds.field("date") >= "2026-07-31") & (ds.field("country") == "US"),
    columns=["user_id", "ts", "amount"],
)
df = tbl.to_pandas()
```

The `hive` partitioning is the `col=value/` convention above and is
the de-facto standard. `pyarrow.dataset` is the underlying engine
used by `pd.read_parquet` for directory reads, and directly using it
is the right call whenever you want filter pushdown or lazy schema
inspection.

### Parquet + Pandas dtype gotcha

Pandas' legacy NumPy-backed dtypes (`int64`, `float64`, `object`)
do not perfectly round-trip through Parquet's typed schema. In
particular:

- A `datetime64[ns]` written and re-read may come back as
  `datetime64[ns, UTC]` depending on the writer's timezone
  configuration. Be explicit.
- An `object`-dtype column of strings comes back typed as `string`
  (Arrow-backed) if you pass `dtype_backend="pyarrow"`. This is
  usually what you want.
- Nullable integers (`Int64`) round-trip cleanly; legacy `int64`
  cannot represent nulls.

Adopt Arrow-backed dtypes end-to-end and the whole class of
round-trip surprises disappears:

```python
df = pd.read_parquet("events.parquet", dtype_backend="pyarrow")
```

## Format choice, decided

A short decision guide.

| Situation | Format |
|---|---|
| Handing a small snapshot to a colleague to open in Excel | CSV |
| API response, single request | JSON |
| Log dump, one record per line | NDJSON |
| Anything you will read more than twice | Parquet |
| Anything larger than 100 MB | Parquet |
| Anything with nested types or precise dtypes to preserve | Parquet |
| Anything you plan to query with SQL later | Parquet (Athena, DuckDB, BigQuery external) |

The pattern most teams converge on: **ingest to Parquet as soon as
possible.** If the upstream is CSV/JSON, the first pipeline step
writes a typed, partitioned Parquet dataset and every downstream
consumer reads that. The one-time conversion cost is repaid many
times over in downstream read speed and dtype safety.

## Compression, briefly

Three codecs matter in practice:

- **Snappy** (Parquet default). Fast; moderate compression. Good
  default for training data you will read many times.
- **Zstd** with `compression_level=` between 3 and 9. Better
  compression ratio at similar CPU cost. Increasingly the default in
  new writers.
- **Gzip**. Slower to read/write than the above; use only when
  interoperating with a system that requires it.

Do not compress Parquet with Gzip out of habit; it makes reads
measurably slower without a meaningful storage win over Zstd.

## Reading many files at once

For a directory of files with the same schema, all of the following
work:

```python
# 1. Pandas + glob
import glob
df = pd.concat((pd.read_parquet(f) for f in glob.glob("events/*.parquet")))

# 2. PyArrow dataset (recommended)
import pyarrow.dataset as ds
df = ds.dataset("events/", format="parquet").to_table().to_pandas()

# 3. Polars (fastest for many files)
import polars as pl
df = pl.scan_parquet("events/*.parquet").collect().to_pandas()
```

Prefer (2) or (3) as file counts grow. Pattern (1) reads
sequentially and holds every intermediate DataFrame in memory.

## Reproducibility: pin the read

The version of the file matters. Two habits keep results
reproducible:

1. **Never mutate a file in place.** Write a new, versioned
   filename (`events_v3.parquet`) or a new dated partition. If a
   downstream job depends on the shape of yesterday's data, it must
   be able to still read yesterday's data.
2. **Record the exact source path (and, ideally, an object-store
   version ID or checksum) in your run's config.** Chapter 04
   returns to this for object stores. Chapter 10 wires it into
   validation. `mod-106` builds it into experiment tracking.

## Summary

- CSV is universally supported and universally ambiguous. Pass
  explicit `dtype=` and `parse_dates=`; do not trust inference.
- JSON needs `json_normalize` for anything nested; NDJSON needs
  `lines=True`. Convert to Parquet as soon as you have looked at
  the data once.
- Parquet is the format most ML data should live in: columnar,
  typed, compressed, splittable, and supported by every tool worth
  using. Read only the columns you need with `columns=`; use
  partitioned datasets with `pyarrow.dataset` for filter pushdown.
- Choose Arrow-backed Pandas dtypes (`dtype_backend="pyarrow"`) to
  make round-trips clean and preserve nullability.
- Prefer conversion-once-to-Parquet over reading raw CSV/JSON on
  every job.

## Where this goes next

Chapter 03 covers ingesting from SQL databases and warehouses —
where much of your data actually lives before it becomes a file.
