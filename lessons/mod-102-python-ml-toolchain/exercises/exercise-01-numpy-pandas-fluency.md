# exercise-01 — NumPy / Pandas fluency

> **Estimated effort:** 2 hours.
> **Companion chapters:** [02 — NumPy for ML](../02-numpy-for-ml.md), [03 — Pandas for ML](../03-pandas-for-ml.md).

## Objective

Build the everyday idioms a working ML engineer reaches for: shape-
and-dtype hygiene, vectorisation, deliberate broadcasting, group-by
feature engineering, and joins with cardinality assertions. By the
end you will have a small, tested feature-engineering module that
takes a raw transaction dataset and produces an ML-ready feature
matrix — without a single Python `for` loop over rows.

## Prerequisites

- Python 3.11+ in a virtual environment (uv / poetry / `python -m
  venv` all fine).
- NumPy ≥ 1.26, Pandas ≥ 2.1, `pytest`.
- Chapters 02 and 03 of this module.

## Problem statement

You will work with a synthetic but realistic e-commerce dataset of
transactions. The dataset has three tables that you must load,
clean, join, and aggregate into a single feature matrix suitable for
a churn or spend-prediction model.

You may **not** use a Python `for` loop over rows of any DataFrame
or array. Every per-element operation must be either vectorised or
expressed via `groupby` / `merge` / `apply` over columns. `df.apply(f,
axis=1)` is a code smell and should not appear in your solution.

### The dataset

In `data/`, you have three Parquet files (provide your own synthetic
generator — see Part A — or use a small open dataset of your choice
that has these three shapes):

- `users.parquet` — `user_id (string), signup_date (date), country
  (category), age (Int64, nullable)`
- `products.parquet` — `product_id (string), category (category),
  unit_price (float64)`
- `transactions.parquet` — `transaction_id (string), user_id (string),
  product_id (string), quantity (Int64), timestamp (datetime64[ns])`

Realistic data quality issues to include:

- ~5% of `age` is missing.
- A handful of `transactions` reference unknown `user_id`s.
- One or two `users` have no transactions.
- A small fraction of timestamps are in the future (data error).

## Requirements

### Part A — generate the synthetic dataset (≈ 20 min)

In `generate_data.py`, write a function `generate(seed: int = 0,
out_dir: Path = Path("data"))` that produces the three Parquet files
above. Use `numpy.random.default_rng(seed)` — not the legacy global
API. Make the dataset small enough to fit in memory (e.g. 1 000
users, 50 products, 20 000 transactions).

Bake the data quality issues above into the generator. Document
each one with a short comment.

### Part B — `numpy_warmups.py` (≈ 30 min)

Implement and test the following pure-NumPy functions. No DataFrame
input.

- `rolling_mean(x, w)` — given a 1-D array `x` and a window size
  `w`, return a 1-D array of length `len(x) - w + 1` containing the
  rolling mean. Use `np.convolve` or `np.cumsum`. No Python loops.
- `pairwise_cosine(A, B)` — given `A` of shape `(m, d)` and `B` of
  shape `(n, d)`, return an `(m, n)` matrix of cosine similarities.
  Use broadcasting and `np.linalg.norm`.
- `topk_per_row(scores, k)` — given a 2-D array of scores, return the
  indices of the top `k` entries in each row. Use `np.argpartition`,
  not a sort.
- `safe_log1p(x)` — like `np.log1p`, but raise a `ValueError` if any
  value of `x` is below `-1`. Reasoning: silent NaNs in the feature
  matrix are a debugging nightmare downstream.

For each function, write at least two `pytest` assertions: a shape
check and a correctness check on a tiny hand-computed example.

### Part C — `pandas_features.py` (≈ 60 min)

This is the main course. Implement the following functions, each
operating on the loaded DataFrames from Part A.

1. `load(data_dir)` returning a dict `{"users": ..., "products": ...,
   "transactions": ...}` of DataFrames. Pass explicit `dtype=`s on
   read; cast string columns to `pd.StringDtype()` and category
   columns to `"category"`.

2. `clean_transactions(transactions, *, as_of)` that:
   - drops rows with a future `timestamp` (relative to `as_of`),
   - returns the cleaned frame plus a small `dict` of counts
     describing what was dropped. Use this as a worked example of
     "always report what your cleaning step removed".

3. `enrich(transactions, users, products)` that:
   - left-joins `users` and `products` onto `transactions`,
   - **must** pass `validate="many_to_one"` on each `merge`,
   - returns a single enriched DataFrame.

   If `validate=` raises on your synthetic data, you have a key-
   uniqueness bug in your generator — fix it in Part A.

4. `build_user_features(enriched, *, as_of)` that returns a
   per-user feature `DataFrame` with at least the following
   features:

   - `tenure_days` = (`as_of` − `signup_date`) in days
   - `n_transactions` = count of transactions in the last 90 days
   - `total_spend_90d` = sum of `quantity * unit_price` in the last
     90 days
   - `mean_basket_90d` = mean transaction value in the last 90 days
   - `n_unique_categories_90d` = number of distinct product
     categories purchased in the last 90 days
   - `days_since_last_txn` = days between `as_of` and the user's
     last transaction (NaN if none)
   - `country` (categorical)
   - `age` (nullable Int64)

   You **must** use `groupby` + named aggregation. You **may not**
   use `apply` over groups. Use `transform` if you need to broadcast
   a group statistic back to rows.

5. `final_feature_matrix(features)` that returns:
   - `X` — a NumPy array of float features (cast `age` to float;
     fill missing with median *only inside* a downstream pipeline,
     not here),
   - `y` — your placeholder target. For this exercise you can
     define `y` as a synthetic future-90d-spend variable; see Part
     D's stretch.

### Part D — tests and a short report (≈ 30 min)

In `tests/test_features.py`, write tests that cover at least:

- `load` returns three DataFrames with the expected columns and
  dtypes.
- `clean_transactions` removes future-timestamp rows and reports the
  correct count.
- `enrich` raises if a `validate=` assumption is violated. Simulate
  this by duplicating a `user_id` and confirming the right
  exception.
- `build_user_features` returns one row per active user with all
  required columns and the documented dtypes.
- `tenure_days` is non-negative.
- `n_transactions == 0` ⇒ `mean_basket_90d` is NaN (not 0 — they
  mean different things).

In `REPORT.md` (5–10 sentences), answer:

- Which feature was the trickiest to compute correctly, and why?
- Did `validate=` catch any bugs during development, or did you
  add it at the end? Honest answer.
- One screenshot or markdown table of the first five rows of your
  final feature DataFrame.

## Starter guidance

- Build the synthetic data generator *first*. Half the difficulty
  of pandas exercises is debugging surprising data.
- Add `.dtype` and `.shape` assertions at the top of every function
  body. They cost nothing and catch the most common bug.
- Prefer `to_numpy()` over `np.array(series)` when leaving Pandas;
  it preserves dtype better.
- For time-window aggregations, slice the enriched frame on the
  timestamp condition *before* the `groupby`. Do not try to do it
  inside an `agg`.
- If a `groupby` operation feels like it wants `apply`, you almost
  certainly want `transform` or a join instead.

## Acceptance criteria

- All `pytest` tests pass on a fresh `uv sync && uv run pytest`.
- No Python `for` loop over rows of a DataFrame or array appears in
  `pandas_features.py` or `numpy_warmups.py`.
- No call to `df.apply(..., axis=1)` appears in your solution.
- Every `merge` in `pandas_features.py` passes an explicit
  `validate=` argument.
- `build_user_features` returns at least the seven features named
  above, with the documented dtypes.
- `REPORT.md` answers all three prompts honestly.

## Stretch goals (optional)

- Add a `target_future_spend(enriched, *, as_of, horizon_days=30)`
  function that computes the per-user spend in the *next* 30 days
  after `as_of`. Use this as your `y` in `final_feature_matrix`.
- Replace one of your `groupby`s with the equivalent `polars` or
  `duckdb` SQL query. Compare wall-clock time on a larger seed.
- Cast the feature DataFrame to PyArrow-backed Pandas (`df.convert_dtypes(dtype_backend="pyarrow")`)
  and benchmark the resulting `to_numpy()` against the NumPy-backed
  version.
- Add a `tests/test_no_loops.py` that uses
  [`ast`](https://docs.python.org/3/library/ast.html) to grep your
  source for `for ... in df.iterrows():` and fails if found. A real-
  life lint-against-bad-pandas check.

## What to hand in

```
exercise-01/
├── generate_data.py
├── numpy_warmups.py
├── pandas_features.py
├── data/
│   ├── users.parquet
│   ├── products.parquet
│   └── transactions.parquet
├── tests/
│   ├── test_numpy_warmups.py
│   └── test_features.py
└── REPORT.md
```

Solutions for this exercise live in the paired
`ml-engineer-solutions` repo; do not look at them until you have
submitted yours.
