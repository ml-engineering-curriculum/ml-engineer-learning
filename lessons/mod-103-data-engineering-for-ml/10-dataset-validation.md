# 10 — Dataset validation: pandera and Great Expectations

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 05, 06, 07.

## Motivation

Everything so far has been about producing a good dataset. This
chapter is about making sure the *next* dataset is still good.

Data pipelines rot. An upstream table adds a new enum value; a
percentage that used to be `0-1` becomes `0-100`; a timezone
handling change moves half the timestamps by five hours; a bug
inflates a count column. The pipeline runs successfully — no
exception thrown, no test failed — and the resulting model quietly
degrades in production, sometimes for weeks before anyone notices.

The antidote is a **dataset validator**: a piece of code that, given
a DataFrame, asserts a set of expectations about it and either
passes or produces a legible failure report. Two mature Python
frameworks:

- **`pandera`** — schema-first, DataFrame-shaped. Great for
  in-process, per-pipeline-step validation. Cheap to add; runs on
  every pipeline invocation.
- **Great Expectations (GX)** — data-quality-platform-shaped. Rich
  expectations, batch-oriented, produces HTML docs and dashboards.
  Heavier; earns its keep on shared datasets consumed by many
  teams.

You want both, at different points in the stack: `pandera` inside
the training pipeline, GX at the boundary of the shared data lake.

This chapter covers each, when to reach for which, and how they
plug into a training job.

## `pandera`: schema as code

`pandera` lets you declare a DataFrame schema — column names,
dtypes, per-column value constraints — and then validate a DataFrame
against it in one call.

```python
import pandas as pd
import pandera.pandas as pa

class Events(pa.DataFrameModel):
    event_id:  pa.typing.Series[str]      = pa.Field(unique=True)
    user_id:   pa.typing.Series[str]      = pa.Field(nullable=False)
    ts:        pa.typing.Series[pd.DatetimeTZDtype("ns", "UTC")] = pa.Field()
    amount:    pa.typing.Series[float]    = pa.Field(ge=0, le=1_000_000, nullable=True)
    country:   pa.typing.Series[str]      = pa.Field(isin=["US", "CA", "MX", "UK", "DE"])
    is_active: pa.typing.Series[bool]     = pa.Field(nullable=False)

    class Config:
        strict = True     # error on unexpected columns

df: pd.DataFrame = ...
validated = Events.validate(df)   # returns df if OK; raises SchemaError otherwise
```

Read what the schema encodes:

- Every listed column must exist with the declared dtype.
- `event_id` is unique across rows.
- `user_id` and `is_active` are non-null.
- `amount` is in `[0, 1_000_000]` where present.
- `country` is one of the enum values.
- `strict=True` fails if the DataFrame has columns not in the
  schema — catches "upstream added a mystery column" cleanly.

On failure, `pandera` raises `SchemaError` with a legible message
including the failing rows. In `lazy=True` mode
(`Events.validate(df, lazy=True)`), it collects *all* failures
before raising — much friendlier for CI.

### Custom checks

For anything beyond per-cell constraints, register a check:

```python
import pandera.pandas as pa

class Events(pa.DataFrameModel):
    ts:     pa.typing.Series[pd.DatetimeTZDtype("ns", "UTC")]
    amount: pa.typing.Series[float] = pa.Field(ge=0)

    @pa.dataframe_check
    def ts_monotone_per_user(cls, df):
        return df.sort_values("ts").groupby("user_id")["ts"].is_monotonic_increasing.all()

    @pa.check("amount")
    def amount_median_stable(cls, s):
        return abs(s.median() - 42.0) < 10.0    # today's expected median ± tolerance
```

Two things to notice:

- **`@pa.check`** runs per column; **`@pa.dataframe_check`** runs
  against the whole DataFrame. Use the right decorator for the
  check.
- **Custom checks are Python code.** Anything you can express as a
  DataFrame operation is a valid check. This is `pandera`'s
  strongest feature vs. the more declarative alternatives.

### `pandera` as a `@check_input` / `@check_output` decorator

The idiomatic pattern for validating pipeline function
inputs/outputs:

```python
from pandera import check_input, check_output

@check_input(Events)
@check_output(FeaturesSchema)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

The decorators validate at call time; a bad input or output raises
before contaminating the downstream step. Cheap; opinionated;
excellent for a Python-native training pipeline.

### Distribution-shape checks

`pandera` supports basic distribution checks via `Hypothesis`
integration (`pa.Check.equal_to_within_tolerance`, `pa.Check.two_sample_ttest`),
but its bread and butter is *schema* and *per-value* constraints.
For richer distribution testing (KS tests, chi-square across
categorical distributions, PSI on numeric distributions), reach for
Great Expectations or a dedicated drift-detection library.

## Great Expectations: expectations as a first-class artefact

Great Expectations (GX) treats "what you expect of this dataset" as
a versioned, executable artefact:

- **Expectations.** Individual assertions ("the `country` column's
  values are in `[US, CA, MX, UK, DE]`", "the `amount` column's
  mean is between 30 and 60").
- **Expectation Suites.** Named collections of expectations
  attached to a dataset.
- **Data Docs.** Auto-generated HTML documentation of the suite
  and the last validation run against it.
- **Checkpoints.** Scheduled or event-triggered validation runs
  that produce validation results and (optionally) fail loudly.

Setting up a minimal GX project (as of v1.x):

```python
import great_expectations as gx

context = gx.get_context()                       # local file context

# Register a Pandas DataFrame as a data source
ds = context.data_sources.add_pandas("in_memory")
asset = ds.add_dataframe_asset("events")

# Attach an expectation suite
suite = context.suites.add(gx.ExpectationSuite(name="events_suite"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
    column="country", value_set=["US", "CA", "MX", "UK", "DE"]))
suite.add_expectation(gx.expectations.ExpectColumnMeanToBeBetween(
    column="amount", min_value=30, max_value=60))

# Build a batch request and validate
batch = asset.add_batch_definition_whole_dataframe("today").get_batch(
    batch_parameters={"dataframe": df}
)
result = batch.validate(expect=suite)
if not result.success:
    raise RuntimeError(f"Validation failed: {result}")
```

The API surface is larger than `pandera`'s, but the payoff is
richer:

- **Auto-generated data docs** — an HTML site describing the
  suite and its history of runs.
- **A library of ~50 built-in expectations** — including
  `ExpectColumnKlDivergenceToBeLessThan` (distribution drift),
  `ExpectColumnMostCommonValueToBeInSet`, `ExpectColumnPairValuesAToBeGreaterThanB`.
- **Data-source coverage.** GX validates Pandas, Spark, and SQL
  data sources with the same suite. Same expectations, different
  execution backend.

GX is more work to configure. Use it when a dataset is consumed by
multiple teams / pipelines, when you want dashboards, or when you
want expectations that live outside the model's training code.

## `pandera` vs. GX: how to decide

A rough decision guide.

| Need | Reach for |
|---|---|
| Assert column presence, dtype, per-value constraints inside a training pipeline | `pandera` |
| One decorator per function boundary catches schema drift | `pandera` |
| Validate a shared table in the warehouse that many teams read | Great Expectations |
| Auto-generated data docs the data team can review | Great Expectations |
| Distribution drift detection with rich, off-the-shelf checks | Great Expectations |
| Validation lives in the same repo as the model, versioned with the code | `pandera` |
| Validation lives in a separate data-quality repo and runs on schedule | Great Expectations |
| Both — schema drift *and* distribution drift | both |

In practice, mature ML stacks use `pandera` at every internal
function boundary and Great Expectations (or a drift-detection
service like `evidently` or `whylogs`) at the ingest and serving
boundaries.

## Wiring validation into a training pipeline

The productive pattern: **validate the input dataset before any
expensive work.** If the dataset is wrong, fail loud, fail early.

```python
def train(config):
    raw = load_dataset(config.data_uri)

    # 1. Schema — must be exactly right or we abort.
    events = Events.validate(raw, lazy=True)

    # 2. Distribution — warn if drifted; fail if drift is severe.
    drift_report = check_distribution_drift(events, config.baseline_stats)
    if drift_report.severe:
        raise DriftError(drift_report)
    elif drift_report.moderate:
        logger.warning("Moderate drift detected", extra=drift_report.summary)

    # 3. Business logic — split, feature-engineer, fit.
    X, y = build_features(events), events["target"]
    X_train, X_test, y_train, y_test = time_ordered_split(X, y, cutoff=config.cutoff)

    pipe = build_pipeline(config.model)
    pipe.fit(X_train, y_train)

    # 4. Validate the fitted model against expectations.
    validate_fit(pipe, X_test, y_test, config.min_metric)
    save(pipe, config.model_uri)
```

Four gates: schema, distribution, split correctness (implicit — the
`time_ordered_split` respects chapter 06's rules), and metric
threshold on the held-out set. Each one prevents a specific class
of silent failure.

## Baseline statistics: what "drift" is measured against

You cannot detect drift without a **baseline** — a snapshot of what
the data looked like at some previous point (usually when the model
was trained, or when the pipeline was last known to be healthy).
The baseline is itself a data artefact:

```
baselines/
  events/
    2026-06-01/
      column_stats.parquet    # per-column mean, stdev, quantiles, top-k values
      pandera_schema.py       # schema at training time
      ge_suite.json           # GX expectation suite at training time
```

Version the baseline alongside the model. When retraining, produce
a new baseline; keep the old ones for a rolling window.

### Distribution-drift signals

Three per-column drift metrics show up in practice:

- **Population Stability Index (PSI)** — measures shift in a
  distribution's bin-wise proportions. Rule of thumb: PSI < 0.1
  means no shift, 0.1–0.25 moderate, > 0.25 significant. Simple
  to compute; interpretable.
- **Kolmogorov–Smirnov statistic (KS)** — max difference between
  two empirical CDFs. Statistical test with a p-value; useful for
  numeric columns.
- **Jensen–Shannon divergence / KL divergence** — information-
  theoretic distance between two distributions. Common in GX and
  `evidently`.

For categorical columns, chi-square on category proportions is the
standard test.

`evidently` and `whylogs` implement all of the above with sensible
defaults. `mod-109` (monitoring and drift) goes deep on the
detection side; this chapter's job is to know they exist and to
wire the training pipeline to check them.

## Validating splits and grouping

Some of the most important invariants are properties of the split
itself. Two `pandera` checks worth stealing:

```python
@pa.dataframe_check
def train_test_disjoint_users(cls, df: pd.DataFrame) -> bool:
    train_users = set(df.loc[df.split == "train", "user_id"])
    test_users  = set(df.loc[df.split == "test",  "user_id"])
    return not (train_users & test_users)

@pa.dataframe_check
def test_starts_after_train(cls, df: pd.DataFrame) -> bool:
    train_max = df.loc[df.split == "train", "ts"].max()
    test_min  = df.loc[df.split == "test",  "ts"].min()
    return train_max < test_min
```

These are the chapter-06 leakage rules turned into automated
assertions. Ship them as tests in CI. When someone accidentally
switches from a `TimeSeriesSplit` to a random `train_test_split`,
the check fires before the model is trained.

## Failure UX: what happens when a check fails

The right shape of a validation failure:

1. **Fail loud, don't degrade silently.** The pipeline should raise
   and stop, not produce a "best-effort" output.
2. **Attribute the failure.** Which column? Which check? Which
   rows? `pandera`'s `lazy=True` and GX's validation results both
   give you row-level failure detail.
3. **Route the failure to a human.** In CI, that's a failed check;
   in production, that's a page or a Slack alert. Do not
   `try / except: pass` around a validation failure.
4. **Snapshot the bad batch.** Save the DataFrame (or a sample) to
   an object-store location referenced in the alert. The person
   debugging at 3am should not have to reproduce the failure to
   see what data caused it.
5. **Have a `failed_batches/` directory as a design goal.** The
   accumulated set of past failures is the world's cheapest
   regression test suite — new checks should reject them.

## Pitfalls collected

1. **Schema-drift silence.** No validator is worse than a
   validator that catches nothing. Add checks for every business
   invariant you rely on.
2. **Validating after preprocessing.** A validator that only sees
   the transformed matrix can't catch schema drift in the raw
   data. Validate at the *ingest* boundary; then optionally again
   at the *feature* boundary.
3. **Overly tight distribution thresholds.** PSI < 0.05 on every
   column is achievable in staging and impossible in production;
   your alert will fire constantly and be ignored. Calibrate
   thresholds on historical variance.
4. **Committing GX data docs.** The HTML site can be dozens of MB;
   `.gitignore` it and publish to an internal host or S3 bucket
   instead.
5. **Forgetting to version the baseline.** If your baseline stats
   file lives at `latest.parquet` and gets overwritten, you lose
   your ability to detect drift from a specific model version.
   Date-stamp everything.
6. **Coupling validation to the training run only.** Validation
   should run on every data ingestion, not only when someone
   trains a model. Otherwise drift is discovered on the next
   retrain, which may be months later.
7. **`pandera` schema in a notebook.** The schema is code; it
   belongs in a module with tests, alongside the model code.

## Summary

- Dataset validation is the tripwire that catches silent data
  breakage before it becomes a silent model breakage.
- Use `pandera` for schema and per-value constraints inside the
  training pipeline — decorators on function boundaries are the
  idiomatic form.
- Use Great Expectations for shared datasets, dashboards, and
  richer distribution / expectation coverage across teams and
  data sources.
- Baseline stats are themselves a data artefact — versioned
  alongside the model. Drift is a comparison to a specific,
  named baseline.
- The training pipeline has four validation gates: schema,
  distribution, split correctness, model metric. Fail loud at
  each; snapshot the bad batch; page a human.
- The validation checks are versioned code, tested in CI. They
  are how "data engineering for ML" stays engineered rather than
  regressing to whatever the raw data happens to look like this
  week.

## Where this goes next

You have now walked through the full ML-data lifecycle: ingest,
profile, clean, engineer, validate. `mod-104` picks up with the
first family of models to train on that output — classical
tabular ML (linear models, trees, ensembles). `mod-109` returns
to validation on the *serving* side, where the same
schema-and-distribution frame becomes the basis of production
drift monitoring.

Congratulations — you have arrived at the point where the model
gets to be the star of the show.
