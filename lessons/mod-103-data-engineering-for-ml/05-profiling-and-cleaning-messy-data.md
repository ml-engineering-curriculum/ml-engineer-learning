# 05 — Profiling and cleaning: missingness, outliers, duplicates

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 02, and `mod-102` chapter 03 (Pandas).

## Motivation

The dataset landed on your machine. Your instinct is to write a
`Pipeline` and train something. Wait. First profile.

Profiling is the act of looking at the data *before* deciding how
to model it. It answers:

- What is the row grain — one row per what?
- What is the shape (rows × columns) and the type of each column?
- Where are the missing values, and are they random or systematic?
- Which columns have outliers, and are the outliers real or errors?
- Are there duplicates, and if so, on which key?
- Does the distribution of the target look sane?

Most model-quality bugs are actually profiling failures: a duplicate
key that inflated CV scores, a missingness pattern that meant "this
event didn't happen" rather than "we don't know", a column that
looked numeric but was actually a bucket ID with 99% zeros. Fifteen
minutes of profiling routinely saves days of debugging.

## The five-minute profile

Every dataset gets the same opening move. Print these six things
before touching a model:

```python
df.shape                       # rows × columns
df.dtypes                      # per-column dtype
df.head()                      # what does one row look like?
df.describe(include="all")     # numeric summaries + categorical top/freq
df.isna().mean()               # fraction missing per column
df.duplicated().sum()          # any exact duplicates?
```

Read the output. Ask yourself: *does anything here surprise me?*
Every surprise is a bug or a misunderstanding to resolve now,
before it becomes a feature you fit on.

For a first-pass automated profile, `ydata-profiling` (formerly
`pandas-profiling`) generates a full HTML report with histograms,
correlations, and missingness matrices in one call:

```python
from ydata_profiling import ProfileReport
ProfileReport(df, title="Events").to_file("profile.html")
```

Convenient for exploration; do not commit the HTML to the repo (it
can be huge, and it contains data).

## Row grain and duplicates

Every dataset has an implicit row grain. Write it down in one
sentence: "one row per event", "one row per (user, day)", "one row
per SKU per store per week". If you cannot, that is a red flag —
find someone who knows.

Once you have a grain, verify it. Duplicates on the natural key are
the single most common ingestion bug.

```python
df["event_id"].duplicated().sum()             # duplicates on a single key
df.duplicated(subset=["user_id", "ts"]).sum() # duplicates on a composite key

# Peek at a few:
dups = df[df.duplicated(subset=["user_id", "ts"], keep=False)]
dups.sort_values(["user_id", "ts"]).head(20)
```

Three responses to duplicates, in decreasing order of preference:

1. **Fix upstream.** If the ingestion pipeline is producing
   duplicates, that is the bug to fix. Silently deduping downstream
   masks a systemic issue.
2. **Keep the newest.** If duplicates come from at-least-once event
   delivery, keep the row with the max `ingested_at` per key.
3. **Aggregate.** If duplicates represent multiple partial rows
   that together form the "true" row, `groupby(key).agg(...)`.

`df.drop_duplicates(keep="last")` is easy to reach for and often
wrong. It picks the last row *as ordered in the current DataFrame*,
which after a `read_parquet` from a partitioned dataset is not
necessarily the most recent event. Sort explicitly first.

## Missing data: three questions

For every column with non-zero missingness, ask three questions.

### 1. How much is missing?

```python
missing = df.isna().mean().sort_values(ascending=False)
print(missing[missing > 0])
```

An arbitrary but useful heuristic:

- **< 5% missing.** Impute; the choice of imputer barely matters.
- **5–30% missing.** Impute carefully; add a "was missing" indicator
  column (see below).
- **> 30% missing.** The column may not be usable; try modelling
  with and without it. If it's a categorical, a
  `"missing"` category is often the right call.
- **> 70% missing.** Almost always drop the column, unless the
  presence of a value is itself informative.

These are heuristics, not laws. A column that is "critical when
present, 90% missing" — like a doctor's diagnosis code — carries
signal precisely in *whether* it is present. Add the indicator.

### 2. Why is it missing?

"Missingness" is not one thing. Statisticians distinguish:

- **MCAR — missing completely at random.** The missingness has no
  relationship to anything, observed or unobserved. Rare in the
  wild. Safe to impute with a column statistic.
- **MAR — missing at random.** Missingness depends on other
  *observed* variables. E.g. "income is missing more often for
  younger users". Imputation conditional on the other variables
  (a model-based imputer) is appropriate.
- **MNAR — missing not at random.** Missingness depends on the
  missing value itself, or on an unobserved variable. E.g.
  "spending is missing precisely when the user hasn't spent
  anything". No imputation strategy is unbiased here — you have to
  encode the missingness itself as a feature.

Practical test: for each column with missingness, plot the fraction
missing against a few candidate variables (time, source, product,
country). If you see structure, you are looking at MAR or MNAR.

### 3. What does missing *mean* in this domain?

The literal question no one asks. In many datasets:

- `NULL` in a `discount` column may mean "no discount was applied",
  which is semantically `0`, not "unknown".
- `NULL` in a `last_purchase_ts` column may mean "the user has
  never purchased" — you want to encode "no purchase" as a
  separate categorical, not as a very-old timestamp.
- A missing feature may mean the pipeline that computes it hasn't
  run yet for that row — a bug, not data.

Before imputing, ask the person who produces the data. If you
cannot, add a `"<col>_was_missing"` indicator column so a downstream
model can learn from the missingness pattern itself:

```python
for col in ["income", "discount", "last_login_ts"]:
    df[f"{col}_was_missing"] = df[col].isna()
```

### Imputation: chapter 07 for how, `Pipeline` for where

This chapter *identifies* what needs imputing. Chapter 07 covers the
tools (`SimpleImputer`, `KNNImputer`, `IterativeImputer`, target
encoding for categoricals). The critical rule from `mod-102`
chapter 06 holds: **impute inside a `Pipeline`, never on the raw
DataFrame before splitting.** Fitting an imputer on the combined
train+test data is textbook leakage.

## Outliers: real, wrong, or rare

An outlier is a value that is very far from the rest. There are
three archetypes and they call for different responses.

### Real but extreme (long-tail data)

Income, page-view counts, session durations: the tail is real. The
99.9th percentile of session duration in an events dataset really
is 12 hours because someone left a tab open. This is not an error;
it is the shape of the distribution.

Responses:

- **Log-transform.** `np.log1p(x)` compresses the tail. Especially
  useful for linear models.
- **Winsorise.** Clip values to the 1st and 99th percentiles. Use
  when the tail is real but the model is sensitive to it.
- **Model with a tree-based method.** Trees are robust to
  monotone transformations of features, so log/no-log rarely
  changes anything for them.

The rule: transform the data to match the assumptions of the model
you plan to use. Do not transform to make the histogram look
prettier.

### Errors (data entry mistakes)

An age of 999, a country of `"ZZ"`, a timestamp in the year 1970: an
implausible value. Some detection strategies:

- **Domain range checks.** `age` should be in `[0, 120]`. Print
  everything outside.
- **The IQR rule.** Values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
  are outliers by Tukey's classic definition. Cheap and often
  correct; wildly wrong on skewed distributions.
- **Z-score.** `|z| > 3` on approximately-normal data. Not
  applicable to counts, prices, or anything log-normal.
- **Model-based** (`IsolationForest`, `LocalOutlierFactor`).
  Overkill for cleaning; better for anomaly-detection as a task.

For known errors, prefer to **NULL them out and let the imputer
handle it** rather than dropping the row. Dropping rows can bias
the dataset (see MAR above).

### Rare-but-real (the tail of a categorical)

A city column with 3 million values where 2.9 million are one of ten
cities and the remaining 100k are 50 000 unique long-tail values.
The tail is not wrong, but it defeats one-hot encoding.

The standard response is bucketing: any category with fewer than
`min_count` occurrences becomes `"__other__"`. Chapter 07 covers
the mechanics inside a `Pipeline`.

## Timestamps: check the range, check the timezone

Timestamps are a special case that deserves its own paragraph.

```python
df["ts"].min(), df["ts"].max()          # what's the range?
df["ts"].dt.tz                          # what timezone?
(df["ts"] < "2020-01-01").sum()         # any suspiciously old values?
(df["ts"] > pd.Timestamp.utcnow()).sum() # any in the future?
```

Common failure modes:

- **Naive timestamps.** No timezone, silently treated as local
  time by whatever code touches them next. Convert to UTC:
  `df["ts"] = pd.to_datetime(df["ts"], utc=True)`.
- **Unix epoch confusion.** Values around `1e9` are seconds since
  epoch (2001). Values around `1e12` are milliseconds. Values
  around `1e15` are microseconds. Values around `1e18` are
  nanoseconds. Pandas' default `pd.to_datetime(unit=)` is
  `"ns"` — passing seconds-since-epoch without specifying gets
  you a date in 1970.
- **Epoch = 0.** A pile of rows at exactly `1970-01-01 00:00:00`
  is almost always a sentinel for "we don't have this timestamp".
  Convert to `NaT`.
- **Future dates.** A row with `ts` after `now()` is either a
  clock-skew issue or a placeholder. Investigate.

## Numeric edge cases

A short list of numeric-column pathologies to check on a first
profile:

- **Infinities.** `df.replace([np.inf, -np.inf], np.nan)`
  frequently at the top of a cleaning function.
- **Impossible negatives.** A `count` or `duration` column
  with negative values is almost always a bug. Investigate before
  clipping.
- **Precision loss.** Currency stored as `float64` is fine for
  aggregation; comparing `float64` values for equality is not.
  Store money in cents as `Int64` if you can, or `Decimal` if you
  need arbitrary precision (Pandas will hold it as `object`).
- **Zero-variance columns.** `df.nunique()[df.nunique() <= 1]` —
  columns with one unique value carry no information and confuse
  some models. Drop.
- **Near-constant columns.** A boolean column with 99.99% `True`
  may or may not be useful; look before dropping.

## Categorical hygiene

Text columns hide their own class of issue.

```python
df["country"].value_counts(dropna=False).head(20)
```

Look for:

- **Whitespace variants.** `"US"`, `"US "`, `" US"` are three
  different values by default. Normalise: `df["country"] =
  df["country"].str.strip()`.
- **Case variants.** `"us"`, `"US"`, `"Us"`. `.str.upper()` or
  `.str.lower()` deliberately.
- **Encoding variants.** `"café"` vs `"café"` are canonically
  different Unicode strings but visually identical. Normalise with
  `unicodedata.normalize("NFC", s)`.
- **Multiple "unknown" spellings.** `"unknown"`, `"UNK"`, `""`,
  `"N/A"`, `"-"`, `"?"`. Standardise on `pd.NA` at ingestion time.
- **Enum drift.** A new value appears in production data that
  wasn't in the training set. Chapter 10's schema validation
  catches this.

## The three questions on the target

If you have labels, profile them separately and profile them first:

1. **What is the label's marginal distribution?** For
   classification, `y.value_counts(normalize=True)`. Highly
   imbalanced targets (e.g. 1% positives) will need specific
   handling (class weights, resampling, threshold tuning) covered
   in `mod-107`.
2. **How is the label distributed over time?** `y.groupby(df["ts"].dt.date).mean()`
   for classification. A label whose base rate drifts is telling
   you something — a business rule changed, a definition changed,
   or the labelling process changed. Fix before modelling.
3. **How reliable is the label?** Chapter 06 covers this at length.
   Cliff notes: labels are often 3–10% wrong; the correct workflow
   is to measure the noise rate and factor it into evaluation.

## A short cleaning-function skeleton

The output of profiling is a short, deterministic cleaning function
that turns raw data into "ready-to-feature-engineer" data. It
lives in its own module and is *tested* (`mod-102` chapter 08). The
skeleton:

```python
import numpy as np
import pandas as pd

def clean_events(raw: pd.DataFrame) -> pd.DataFrame:
    """One-time cleanup that must run identically at train and serve time."""
    df = raw.copy()

    # 1. Fix dtypes.
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["country"] = df["country"].astype("string").str.strip().str.upper()

    # 2. Handle sentinels and impossibles.
    df["amount"] = df["amount"].where(df["amount"] >= 0)  # negatives -> NaN
    df = df.replace({"country": {"": pd.NA, "ZZ": pd.NA}})
    df["ts"] = df["ts"].where(df["ts"].dt.year >= 2000)

    # 3. Drop or dedupe on the row grain.
    df = df.dropna(subset=["event_id", "user_id"])
    df = df.sort_values(["event_id", "ingested_at"]).drop_duplicates(
        subset=["event_id"], keep="last"
    )

    # 4. Add missingness indicators for the tricky columns.
    for col in ["country", "amount"]:
        df[f"{col}_was_missing"] = df[col].isna()

    return df.reset_index(drop=True)
```

Notes:

- **`errors="coerce"`** turns unparseable values into `NaN` instead
  of raising. That is usually what you want on ingest; failures
  become imputable rather than crashy.
- **Idempotent.** Calling `clean_events(clean_events(df))` should
  produce the same result as `clean_events(df)`. Test this.
- **No fitting.** Nothing here computes a statistic from the whole
  DataFrame. That is the job of the `Pipeline` (chapter 07).

## Pitfalls collected

1. **Cleaning inside a Jupyter cell.** The transformations get
   lost. Put cleaning in a function in a module, and import it
   into the notebook. `mod-102` chapter 04 covers the discipline.
2. **Dropping rows silently.** Any `dropna` or `drop_duplicates`
   should log the count before and after. A silent 30% drop in row
   count is a bug most of the time.
3. **`inplace=True`.** Confusing, being deprecated, and hides the
   intent of the code. Use assignment.
4. **`fillna(0)` on a float column that later gets one-hot encoded
   or treated as counts.** You have just injected 30% "zero"
   observations into the distribution.
5. **Detecting outliers in-place on the target.** Doing an IQR
   filter on `y` before splitting biases the test set — you have
   filtered the test set by its label. Only clean features
   pre-split; clean labels only within the training fold.
6. **Trusting the first profile.** The dataset changes over time.
   A monthly re-profile catches drift; chapter 10 automates it.

## Summary

- Profiling is a workflow, not a tool. Six commands (`shape`,
  `dtypes`, `head`, `describe`, `isna().mean()`, `duplicated().sum()`)
  are the first thing that runs on any new dataset.
- Duplicates on the row grain are the most common ingestion bug.
  Verify explicitly; fix upstream when possible; sort before
  deduping.
- Missing data is not one thing. Ask *how much*, *why*, and *what
  does missing mean here*. Add `_was_missing` indicators for
  informative missingness.
- Outliers come in three flavours (long-tail-real, error,
  rare-categorical). Each has its own response; do not just clip.
- Timestamps and categoricals hide their own pathologies (naive
  tz, epoch confusion; whitespace, case, encoding variants).
  Standardise once at ingestion.
- Cleaning is a deterministic, idempotent function in a module,
  with tests. Not a scratch cell in a notebook.

## Where this goes next

Chapter 06 covers the highest-stakes cleaning topic: label quality
and the many faces of data leakage. Skip nothing.
