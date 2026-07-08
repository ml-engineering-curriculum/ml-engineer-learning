# 03 — Pandas for ML

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 01, 02.

## Motivation

Pandas is where tabular data lives in Python. Almost every ML
pipeline starts with `pd.read_csv` / `pd.read_parquet` / a SQL query
landing as a `DataFrame`, and ends with a `(X, y)` split that NumPy
or scikit-learn can consume. Pandas covers everything in between:
filtering, aggregating, joining, reshaping, handling missing values,
and converting categorical columns into something numerical models
can chew on.

This chapter is a working ML engineer's tour of the operations that
appear in nearly every preprocessing notebook. It does not try to
teach all of Pandas — the
[User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
runs to hundreds of pages. It teaches the subset you will actually
reach for.

## The two objects

- **`Series`** — a one-dimensional, labelled array. Wraps a NumPy
  array (or extension array) plus an `Index` of row labels.
- **`DataFrame`** — a two-dimensional, column-labelled table. Each
  column is a `Series`. Has an `Index` (row labels) and a `columns`
  axis.

```python
import pandas as pd
df = pd.DataFrame({"age": [22, 31, 45], "country": ["US", "UK", "US"]})
df.shape       # (3, 2)
df.dtypes      # age: int64, country: object
df["age"]      # Series of length 3
df.index       # RangeIndex(start=0, stop=3, step=1)
```

Three things to internalise on day one:

1. The `Index` is data, not magic. Resetting it (`reset_index`) and
   setting it (`set_index`) is a normal part of preprocessing.
2. The `dtype` of every column matters — it determines memory, speed,
   and how operations behave on missing data (see "categorical and
   nullable types" below).
3. Pandas operations almost always return a *new* object; they do
   not mutate in place (`inplace=True` exists but is widely
   considered an anti-pattern and is being deprecated for several
   methods).

## Loading data

For ML work, two file formats dominate:

```python
df = pd.read_csv("data.csv", dtype={"user_id": "string"})
df = pd.read_parquet("data.parquet")
```

Two rules of thumb worth adopting:

- **Prefer Parquet over CSV** for anything non-trivial. CSVs are
  text, untyped, and ambiguous (Is `"01"` a number or a string? Is
  `"NA"` the country or a missing value?). Parquet is binary,
  typed, columnar, and compressed — it is the format of choice for
  production data lakes.
- **Pass an explicit `dtype=`** on read when you care. The default
  type inference is fast but lossy — leading zeros disappear, mixed
  types become `object`, integers with missing values silently become
  `float64`.

For very large CSVs that don't fit in memory, see
`pd.read_csv(..., chunksize=...)`. Or — frankly — switch to Polars
or DuckDB. Pandas is not the right tool for >memory data.

## Selection: `.loc`, `.iloc`, and the bracket syntax

Pandas has three overlapping selection APIs and the friction between
them is responsible for a remarkable percentage of new-user
confusion.

- `df["col"]` — select a column. Always works.
- `df[["col_a", "col_b"]]` — select multiple columns; returns a
  `DataFrame`.
- `df.loc[row_label, col_label]` — label-based selection. Use when
  the index has meaningful labels (timestamps, IDs, ...).
- `df.iloc[row_pos, col_pos]` — integer-position selection. Use when
  you want "the 0th row" regardless of label.
- `df["col"][0]` — works for integer-indexed frames but is a
  generic anti-pattern; `df["col"].iloc[0]` is unambiguous.

The rule that saves you grief: **`.loc` is by label, `.iloc` is by
position; pick one and write it explicitly.** Avoid chained
indexing (`df[df.x > 0]["y"] = ...`); it produces the infamous
`SettingWithCopyWarning` and may silently fail to assign. The fix is
a single `.loc`: `df.loc[df.x > 0, "y"] = ...`.

## Missing data

Pandas has three flavours of "missing":

- `np.nan` — IEEE 754 NaN. Used by `float64` columns. Propagates
  through arithmetic.
- `None` — Python's null. Often shows up in `object`-dtype columns.
- `pd.NA` — Pandas' typed missing-value sentinel. Used by nullable
  integer / boolean / string dtypes.

For ML preprocessing, two operations cover 90% of cases:

```python
df["age"].isna().sum()             # how many missing values?
df.dropna(subset=["age"])          # drop rows missing critical features
df["age"].fillna(df["age"].median())  # impute (do NOT do this directly — see chapter 06)
```

> **Important.** Filling missing values with column statistics *on
> the full dataset before splitting train/test* is a classic
> data-leakage bug. The right place to impute is inside a
> `sklearn.pipeline.Pipeline`, where the imputer's `fit` only sees
> the training fold. Chapter 06 covers this in depth.

## Group-by and aggregate

`groupby` is the workhorse of feature engineering. The pattern:

```python
df.groupby("country")["age"].mean()
df.groupby("country").agg(mean_age=("age", "mean"), n=("age", "size"))
df.groupby("country", as_index=False).agg(mean_age=("age", "mean"))
```

Three idioms worth knowing:

1. **Named aggregation.** The
   `agg(name=("col", "func"))` form gives every output column an
   explicit name. Use it; the un-named alternative produces a
   `MultiIndex` that is annoying to work with downstream.
2. **`as_index=False`** keeps the group keys as regular columns
   rather than promoting them to the row index. Useful when the
   output feeds straight into another `merge` or join.
3. **`transform` vs. `agg`.** `agg` collapses each group to one row;
   `transform` returns a value for every input row, broadcasting the
   group statistic back. `transform` is how you compute "this row's
   value minus its group's mean" in one expression.

```python
df["age_demeaned"] = df["age"] - df.groupby("country")["age"].transform("mean")
```

## Joins (`merge`)

Joins in Pandas use `merge`. The mental model is the SQL join you
already know:

```python
joined = users.merge(orders, on="user_id", how="left", validate="one_to_many")
```

Two flags that pay for themselves:

- **`how=`** — `inner` (default), `left`, `right`, `outer`. Almost
  always say what you mean explicitly.
- **`validate=`** — `"one_to_one"`, `"one_to_many"`, `"many_to_one"`,
  `"many_to_many"`. Raises if your assumption about cardinality is
  wrong. This single argument has caught many a "row count exploded
  silently because the key wasn't unique" bug. Use it.

## Reshaping: `pivot`, `melt`, and the long/wide divide

Most ML libraries want **long** data: one observation per row. Most
human-friendly tables are **wide**: one row per entity, one column
per observation.

```python
wide = df.pivot(index="user_id", columns="day", values="clicks")
long = df.melt(id_vars=["user_id"], var_name="day", value_name="clicks")
```

You will pivot/melt more than you expect once you start working with
time-series features.

## Dtypes that pay for themselves

Two dtype families are worth learning explicitly:

- **`category`** for low-cardinality string columns. Stores the
  string once and the values as integer codes. Cuts memory by 10×+
  for a "country" column with 200 unique values across 10M rows,
  and speeds up `groupby` substantially.
- **Nullable types** (`Int64`, `Float64`, `boolean`, `string`).
  Unlike the legacy NumPy-backed `int64`, the capital-`I` `Int64`
  supports `pd.NA`, so you do not have to upcast to `float64` just
  to represent a missing integer.

```python
df["country"] = df["country"].astype("category")
df["count"]   = df["count"].astype("Int64")     # nullable
```

## The chained-method idiom

Pandas code that does not use chained methods tends to read like:

```python
df2 = df[df.country == "US"]
df3 = df2.dropna(subset=["age"])
df4 = df3.groupby("city")["age"].mean()
df4 = df4.reset_index()
```

Four reassignments for one pipeline. The chained-method version
reads top-to-bottom:

```python
result = (
    df
    .loc[df.country == "US"]
    .dropna(subset=["age"])
    .groupby("city", as_index=False)["age"].mean()
)
```

The `pipe` method lets you slot in custom functions while staying in
the chain:

```python
result = (
    df
    .pipe(filter_active_users)
    .pipe(add_recency_features, today=as_of)
    .groupby("segment", as_index=False)
    .agg(spend=("amount", "sum"))
)
```

Build the habit early — readable preprocessing code becomes
verifiable preprocessing code.

## Pitfalls collected

A short list of Pandas behaviours that will surprise you exactly
once each:

1. **`==` on a column with NaNs returns False for the NaN rows**, but
   `.isna()` does not. `NaN != NaN` is True.
2. **`apply` over rows is slow.** It iterates in Python. For column
   operations, use vectorised expressions or `pipe`. `df.apply(f,
   axis=1)` is almost always a code smell.
3. **`reset_index()` without `drop=True`** turns the old index into a
   column called `index`. Usually not what you want.
4. **`pd.read_csv` with mixed types in a column** silently downgrades
   the column to `object`. Set `dtype=` explicitly or do a
   post-read `df.dtypes` check.
5. **`merge` on a non-unique key on both sides** causes a Cartesian
   blow-up. `validate="one_to_one"` would have caught it.
6. **`groupby` excludes NaN keys by default.** Pass `dropna=False` to
   include them.

## Summary

- Pandas wraps NumPy with row/column labels and adds operations
  (`groupby`, `merge`, `pivot`, `melt`) that turn raw tables into
  ML-ready feature matrices.
- The three selection APIs (`[]`, `.loc`, `.iloc`) overlap — pick
  one explicitly and avoid chained indexing.
- Treat missing data deliberately; never impute on the full dataset
  before splitting.
- Use `category` and the nullable types when the data calls for
  them; they pay for themselves in memory and correctness.
- Chained methods + `pipe` produce preprocessing code you can read.

## Where this goes next

Chapter 04 covers Jupyter — the notebook environment in which most
of this exploratory work happens. Chapter 05 introduces the
scikit-learn API, and chapter 06 stitches Pandas DataFrames into
production-grade `Pipeline` and `ColumnTransformer` constructions
that avoid the leakage pitfalls flagged above.
