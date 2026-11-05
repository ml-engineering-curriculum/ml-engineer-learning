# lab-01 — Parquet partition pruning and column projection

> **Duration:** ~30–45 minutes.
> **Companion chapters:** [02 — Ingesting from files](../02-ingesting-from-files.md), [04 — Ingesting from object stores](../04-ingesting-from-object-stores.md).

## Objective

Build muscle memory for the one claim chapters 02 and 04 keep making:
a Hive-partitioned Parquet dataset read with `pyarrow.dataset` combines
partition pruning, row-group filtering, and column projection so a
reader fetches **only the bytes it asked for**. By the end you will
have (a) built a tiny partitioned dataset on local disk, (b) measured
file opens and bytes read under three progressively smarter read
strategies, and (c) seen the "three optimisations compose" claim turn
into a measurable wall-clock difference.

This lab is pure local-disk Parquet. Exercise-01 builds a real
SQL + S3 ingestion module; this lab is the five-minute drill on the
pushdown mechanics that exercise-01 relies on.

## Prerequisites

- Python 3.11+, Pandas ≥ 2.1, PyArrow ≥ 15, NumPy ≥ 1.26 in any
  virtual environment.
- Chapters 02 and 04 read through once.
- A terminal and any Python runner (`python script.py`, `ipython`, or
  a Jupyter notebook — all work).

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-01-partition-pruning && cd lab-01-partition-pruning
touch pruning.py
```

All code below goes in `pruning.py`. Run with `python pruning.py` at
each step.

Start the file with the imports:

```python
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
```

## Steps

### Step 1 — Generate a Hive-partitioned Parquet dataset (≈ 10 min)

Chapter 02 ("Row groups and partitioning") describes the Hive layout:
`col=value/` directories, each holding one or more Parquet files.
Build one with three partition columns and ten million rows total, so
pruning has something to prune.

Add to `pruning.py`:

```python
RNG = np.random.default_rng(seed=0)
ROOT = Path("events/")

def build_dataset(n_rows: int = 10_000_000) -> None:
    if ROOT.exists():
        return
    dates = pd.date_range("2026-07-01", "2026-07-31", freq="D").date
    countries = ["US", "CA", "MX", "UK", "DE"]
    df = pd.DataFrame({
        "user_id":   RNG.integers(0, 1_000_000, size=n_rows).astype("int64"),
        "ts":        pd.to_datetime(
                        RNG.integers(int(pd.Timestamp("2026-07-01").value / 1e9),
                                     int(pd.Timestamp("2026-08-01").value / 1e9),
                                     size=n_rows),
                        unit="s", utc=True),
        "amount":    RNG.normal(50.0, 15.0, size=n_rows).astype("float32"),
        "session_s": RNG.exponential(120.0, size=n_rows).astype("float32"),
        "note":      RNG.choice(["hello", "world", "foo", "bar"], size=n_rows),
        "date":      RNG.choice(dates, size=n_rows),
        "country":   RNG.choice(countries, size=n_rows),
    })
    # Hive-partitioned write. partition_cols must NOT also appear in the
    # file body — the directory name is authoritative (chapter 04).
    df.to_parquet(
        ROOT, partition_cols=["date", "country"],
        engine="pyarrow", compression="snappy", index=False,
    )
    print(f"Wrote {n_rows:,} rows across {len(dates)} × {len(countries)} partitions")

build_dataset()
```

Run `python pruning.py` once. The write takes 15–45 seconds the first
time; subsequent runs skip it because `ROOT.exists()`.

**Expected output.** A line saying `Wrote 10,000,000 rows across 31 × 5
partitions`, and an `events/` directory on disk containing paths like
`events/date=2026-07-15/country=US/part-0.parquet`.

**Checkpoint.** List a few partition directories:

```bash
ls events | head -3
ls "events/date=2026-07-15" | head -3
```

You should see `date=YYYY-MM-DD` directories and `country=XX`
sub-directories. The Hive convention from chapter 04 is literally
`col=value/` — Python tools read that convention out of the
directory names.

### Step 2 — Measure a naive full read (≈ 5 min)

Baseline: read the whole dataset into memory and filter in Pandas.
This is what "reading one tiny file at a time" from chapter 04's
pitfall list ends up costing.

Add to `pruning.py`:

```python
def total_bytes_on_disk(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*.parquet"))

def naive_read() -> tuple[pd.DataFrame, float, int]:
    t0 = time.perf_counter()
    df = pd.read_parquet(ROOT, engine="pyarrow")
    df = df[(df["date"].astype(str) == "2026-07-15") & (df["country"] == "US")]
    df = df[["user_id", "ts", "amount"]]
    return df, time.perf_counter() - t0, total_bytes_on_disk(ROOT)

df_naive, t_naive, bytes_total = naive_read()
print(f"naive:  rows={len(df_naive):>8,}  t={t_naive*1000:7.0f} ms  "
      f"bytes-read≈{bytes_total/1e6:.1f} MB (whole dataset)")
```

Run it.

**Expected output.** A number of rows in the single-digit millions
divided by 31·5 — i.e. tens of thousands — and a wall-clock in the
hundreds of milliseconds to a couple of seconds, depending on your
disk. The bytes-read figure is the **whole dataset's size on disk**,
because that is what `pd.read_parquet(ROOT)` actually reads before
your Pandas filter runs.

**Checkpoint.** The filter applied *after* the read is the source of
the waste. Every byte of every partition you don't need was still
pulled through the Parquet reader, decompressed, and materialised into
a column. Chapter 04's pitfall #6: a filter in Pandas cannot
substitute for one in `pyarrow.dataset`.

### Step 3 — Partition pruning with `pyarrow.dataset` (≈ 10 min)

Now the right shape. Push the filter *into* the dataset reader so it
prunes partition directories before opening any files.

Add to `pruning.py`:

```python
def pruned_read() -> tuple[pd.DataFrame, float, int, int]:
    dataset = ds.dataset(str(ROOT), format="parquet", partitioning="hive")
    all_files = dataset.files

    t0 = time.perf_counter()
    scanner = dataset.scanner(
        filter=(ds.field("date") == "2026-07-15") & (ds.field("country") == "US"),
        columns=["user_id", "ts", "amount"],
    )
    fragments_read = [f for f in scanner.to_reader().read_all().schema.names]  # force materialisation
    tbl = scanner.to_table()
    df = tbl.to_pandas()
    t_pruned = time.perf_counter() - t0

    # Which files did pruning leave on the table?
    kept_files = [
        f for f in dataset.get_fragments(
            filter=(ds.field("date") == "2026-07-15") & (ds.field("country") == "US")
        )
    ]
    kept_bytes = sum(Path(frag.path).stat().st_size for frag in kept_files)
    return df, t_pruned, len(all_files), len(kept_files), kept_bytes

df_pruned, t_pruned, n_total, n_kept, bytes_kept = pruned_read()
print(f"pruned: rows={len(df_pruned):>8,}  t={t_pruned*1000:7.0f} ms  "
      f"files {n_kept}/{n_total}  bytes-on-kept-partitions≈{bytes_kept/1e6:.1f} MB")
```

Run it.

**Expected output.** `rows` matches the naive read exactly. The
**files** fraction drops from `155` (31 days × 5 countries) to `1` —
one file for `date=2026-07-15/country=US/`. The `bytes-on-kept-partitions`
drops to roughly `bytes_total / 155`. Wall-clock is typically 5–20×
faster than the naive read.

**Checkpoint.** Three optimisations composed, exactly as chapter 04's
"Partitioned Parquet datasets" section promises:

- **Partition pruning.** `dataset.get_fragments(filter=...)` returned
  one file out of 155 — the reader never opened the other 154.
- **Column projection.** `columns=["user_id","ts","amount"]` pulled
  three columns' bytes out of the seven in the file.
- **Row-group filtering.** Not visible in this script's single-day
  filter — but if you added e.g. `(ds.field("amount") > 100)`, the
  reader would also skip row groups whose amount min/max statistics
  didn't overlap the predicate.

### Step 4 — Compare and sanity-check (≈ 5 min)

Append a consistency check:

```python
assert len(df_naive) == len(df_pruned), "pruned read lost rows!"
assert set(df_naive.columns) == set(df_pruned.columns), "pruned read shape differs!"

speedup = t_naive / t_pruned
print(f"\nspeedup: {speedup:.1f}×  "
      f"(files opened {n_kept}/{n_total}, bytes read "
      f"{bytes_kept/bytes_total*100:.1f}% of dataset)")
```

Run it.

**Expected output.** `pruned read lost rows!` does not fire; the two
DataFrames have the same number of rows and columns. The speedup is
somewhere between 5× and 50× depending on disk and CPU — not a
promise, but a strong tendency.

**Checkpoint.** You now have three concrete measurements chapter 04
only describes in prose:

1. How many files the pruning saved you from opening (154 of 155).
2. How much of the on-disk byte volume the pruning saved you from
   reading (~99.4%).
3. The wall-clock speedup on a trivial local disk — which would
   *grow* on an object store, where each un-pruned file open is also a
   network round-trip.

### Step 5 — The `columns=` argument matters even without partitioning (≈ 5 min)

One last drill. Confirm column projection carries weight *within* a
file, independent of partition pruning. Read just one partition two
ways:

```python
one_partition = ROOT / "date=2026-07-15" / "country=US"

t0 = time.perf_counter()
df_all_cols = pd.read_parquet(one_partition, engine="pyarrow")
t_all = time.perf_counter() - t0

t0 = time.perf_counter()
df_three_cols = pd.read_parquet(
    one_partition, engine="pyarrow", columns=["user_id", "ts", "amount"],
)
t_three = time.perf_counter() - t0

print(f"\none partition, all cols:    shape={df_all_cols.shape}    t={t_all*1000:6.1f} ms")
print(f"one partition, columns=3:   shape={df_three_cols.shape} t={t_three*1000:6.1f} ms")
```

**Expected output.** Both reads produce the same number of rows. The
`columns=3` read's `shape` is `(N, 3)`; the all-columns read's is
`(N, 5)` (the two partition columns live in directory names, not the
file body — chapter 04). The three-column read is noticeably faster
and drastically smaller in memory.

**Checkpoint.** Chapter 02's "Note `columns=` on the read side —
Parquet only fetches the columns you ask for. On a wide table this
can be a 10–50× IO reduction." You just reproduced the baby version
of that claim on a narrow table; the win scales with column count
and row-group size.

> **Note.** The partition columns (`date`, `country`) do not appear in
> `df_three_cols.columns` because `columns=` restricts to file-body
> columns. If you need them back, add them in a post-step or rely on
> `pyarrow.dataset` (step 3), which materialises partition values
> transparently.

## Expected final artefacts

- `lab-01-partition-pruning/pruning.py` with the five step blocks
  above.
- `lab-01-partition-pruning/events/` — the Hive-partitioned dataset
  on disk. You can delete it after; it's regeneratable.

## Final checkpoint

Without re-reading chapters 02 and 04, answer:

1. In Step 3, how did `pyarrow.dataset` know to open only *one* file
   out of 155 before you ever called `.to_table()`?
2. In Step 4, which of the three optimisations contributed most of
   the wall-clock speedup on your machine — pruning, projection, or
   row-group filtering? How would that ranking change if the dataset
   were on S3 instead of local disk?
3. In Step 5, why did `df_three_cols` not contain the `date` and
   `country` columns, and what is the cheapest fix if you need them?
4. If you repartitioned the same data with 100 000 tiny
   `user_id=NNNNN/` directories instead of 155 larger ones, which
   chapter-04 pitfall would start biting first?

If you can answer all four, the pushdown-reader mental model is in
place.

## Where to go next

- `exercise-01` ("Ingest from SQL and S3") takes these same mechanics
  and wraps them in a real ingestion module with partitioned writes,
  `_SUCCESS` markers, and tests. The pushdown-aware read you drilled
  here is Part B of that exercise.
- Chapter 05 opens the second arc of the module: profiling and
  cleaning the data you just learned to read efficiently.
