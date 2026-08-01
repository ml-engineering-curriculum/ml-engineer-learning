# exercise-02 — Data quality audit

> **Estimated effort:** 3 hours.
> **Companion chapters:** [05 — Profiling and cleaning](../05-profiling-and-cleaning-messy-data.md), [06 — Leakage and label quality](../06-leakage-and-label-quality.md).

## Objective

Take a real, messy dataset. Do a *rigorous* data quality audit:
document the row grain, quantify missingness, identify outliers,
detect duplicates, hunt for label-quality issues and leakage,
and write a deterministic cleaning function whose output is
demonstrably suitable for modelling.

The point is not just to clean the data. It is to *find* what is
wrong with it. A good data audit routinely surfaces
production-blocking issues in fifteen minutes that a modelling
project would otherwise chase for weeks.

## Prerequisites

- Chapters 05 and 06 of this module.
- The output of exercise-01 is a natural input (a joined,
  partitioned Parquet dataset), but any messy tabular dataset with
  labels works.
- Python 3.11+, Pandas ≥ 2.1, `ydata-profiling` (optional but
  recommended), `matplotlib` or `seaborn`, `pytest`.

## Dataset choice

The following are known to be interestingly messy — good practice
material because they contain real, subtle issues rather than
synthetic ones:

- **UCI Adult (Census Income).** `fetch_openml("adult",
  version=2, as_frame=True)`. Small; several missingness quirks
  (encoded as `"?"`); useful categorical column drift.
- **Ames Housing.** `fetch_openml("house_prices",
  as_frame=True)`. Larger (~80 columns); many missing values with
  domain-specific semantics (`NA` for `PoolQC` means "no pool",
  not "unknown").
- **NYC Taxi (a single month).** Public Parquet dataset.
  Timezones, negative fares, zero-second trips, tips on
  cash-payment rows — a wealth of natural cleaning problems.
- **Kaggle Titanic.** Small; classic but well-worn.
- **Your exercise-01 output.** Most authentic; make sure the joined
  dataset actually has some interesting messiness first.

Pick one. In `REPORT.md`, name the dataset, cite the source URL,
and record the raw row and column counts.

## Problem statement

### Part A — the five-minute profile (≈ 20 min)

In `audit.py`, write a function `initial_profile(df) -> dict` that
computes:

- `n_rows`, `n_cols`
- Per-column dtype
- Per-column fraction missing
- Number of exact duplicate rows
- Number of duplicate rows on the natural key (which you must
  identify)
- For numerics: min, max, mean, median, standard deviation, and
  the 1st / 99th percentiles
- For categoricals: number of unique values and the top-5 values
  with counts
- For timestamps: min, max, timezone, count outside a "sane"
  range (define what sane means)

Persist the profile to `profile.json`. In `REPORT.md`, list at
least five things about the profile that surprised you or that you
wanted to investigate further. If nothing surprised you, the
dataset is either well-known or the profile isn't detailed enough.

### Part B — deep-dive on missingness (≈ 30 min)

For every column with missingness above 1%, answer three questions
in `REPORT.md`:

1. **How much is missing?**
2. **Is it MCAR, MAR, or MNAR?** Test this: for the top-3
   missing columns, plot the fraction missing conditional on 2–3
   other candidate variables. Include the plots in `plots/`.
3. **What does missing mean in this domain?** Read the dataset's
   documentation (if it has one). If not, look at a few rows and
   guess — then write down your guess and move on.

For each missing-value column you plan to use downstream, decide:

- Drop the column entirely (justify).
- Impute with a strategy (specify which and why).
- Add a `_was_missing` indicator column alongside the imputed
  value.

Do **not** actually impute yet — imputation belongs in the
`Pipeline` (chapter 07 / exercise-03). Just decide the policy.

### Part C — outlier hunt (≈ 30 min)

For every numeric column, apply *at least three* outlier detection
strategies from chapter 05:

1. Domain range check (define plausible min/max per column).
2. Tukey's IQR rule.
3. `|z| > 3` on approximately-normal columns.
4. Optionally: `IsolationForest` on the full numeric feature
   space.

For each column, count the outliers by each rule and — this is the
important part — **sample five outliers from each rule and
hand-classify them**:

- Real but extreme (long-tail).
- Data entry error.
- Sentinel value (`-999`, `0` where `NULL` was meant).
- Something else.

Write the classifications into a table in `REPORT.md`. This is
the exercise's central deliverable — the manual work of *looking
at* the outliers is what teaches you the data.

Do **not** clip or drop outliers in a top-level cleaning function
without recording why. A silent 5% drop in row count from an IQR
filter is a bug.

### Part D — duplicate audit and row-grain check (≈ 20 min)

- State the row grain of the dataset in one sentence: "one row per
  _____". If you can't, that's the first finding.
- Verify: `df.duplicated(subset=[natural_key]).sum()`. If non-zero,
  investigate whether these are truly duplicates, near-duplicates
  (same key, different values), or missing key values.
- For any duplicates found, either:
  - Fix (dedup with a documented rule — sort by
    `ingested_at`, keep the last).
  - Aggregate (if the duplicates represent partial rows).
  - Document as an open issue if the correct answer is not clear.

### Part E — label-quality and leakage audit (≈ 45 min)

You have identified a target column. Now audit it.

**Label quality:**

1. Report the target's marginal distribution
   (`y.value_counts(normalize=True)` for classification, histogram
   for regression).
2. Plot the target rate over time if there is a time column. Is
   there drift?
3. Manually inspect **20 randomly-sampled rows** and answer,
   for each: "does the label look right for the features?"
   Record how many you would confidently label the same way, how
   many you disagree with, and how many you cannot decide.
4. If your dataset has any groundwork for label noise estimation
   (multiple annotators, ground-truth column), compute it.

Even without formal noise estimation, the manual eyeball is
valuable — an ML engineer who has never looked at 20 raw label rows
does not yet know their dataset.

**Leakage:**

For every column that will become a feature, ask the two questions
from chapter 06:

- Was this value **known at prediction time**? If the answer is
  "no" or "I'm not sure", the column is a candidate leak. Move it
  to a `LEAKY_COLUMNS` list in your code.
- If the dataset has a time axis: was this value **computed
  using only data available at label time**? If it aggregates
  over the future, it leaks.

Additionally, run the **top-feature ablation** smell test:

1. Fit a quick baseline model (`HistGradientBoostingClassifier`,
   default parameters) on the un-cleaned features.
2. Report feature importances.
3. If any feature has near-perfect correlation with the target,
   investigate.
4. Ablate the top-1 feature (drop it, refit). If the score barely
   moves — the feature was redundant. If the score collapses —
   examine that feature for leakage.

Document your findings in `REPORT.md`. This is the second key
deliverable.

### Part F — the cleaning function (≈ 25 min)

Write `clean.py` with a single function:

```python
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, idempotent cleaning applied at both train and serve time."""
    ...
```

Requirements:

- **Deterministic.** Same input → same output. No randomness, no
  `datetime.now()`.
- **Idempotent.** `clean(clean(df))` equals `clean(df)`.
- **No leakage.** No `groupby(...).transform(mean)` that spans the
  full dataset. No `SimpleImputer` fitted here — imputation lives
  in the pipeline.
- **Logs row-count changes.** Any `dropna` or `drop_duplicates`
  logs the count before and after.
- **Drops the `LEAKY_COLUMNS` from Part E.**

Write tests in `tests/test_clean.py`:

- `test_clean_is_idempotent`
- `test_clean_drops_leaky_columns`
- `test_clean_preserves_rows_with_valid_key`
- `test_clean_normalises_country_strings` (or whichever specific
  normalisation you did)

### Part G — a data-quality report (≈ 15 min)

The final `REPORT.md` should be readable by a colleague who has
never seen the dataset. Structure it as:

1. **Dataset.** Source, row/column counts, time range.
2. **Row grain.** One sentence.
3. **Missingness.** Per-column policy table.
4. **Outliers.** Per-column classification table.
5. **Duplicates.** What was found and what was done.
6. **Labels.** Distribution, drift, manual-review sample summary.
7. **Leakage findings.** Any columns removed and why.
8. **Cleaning policy.** Short prose summary of what `clean()` does.
9. **Open issues.** Anything you found but couldn't resolve.

Length: 2–4 pages. If it is one page, you didn't dig deep enough.
If it is ten, you are writing prose where a table would do.

## Starter guidance

- **`ydata-profiling`** gives you a huge HTML report in one call.
  Use it to *start* the audit, not to end it — the interesting
  findings usually come from looking, not from clicking through a
  report.
- Keep the plots minimal and functional. `df["col"].hist()`,
  `df.groupby("cat")["y"].mean().plot.bar()`. This is not a
  data-viz exercise.
- **Set a random seed** for any sampling you do so a colleague can
  reproduce which rows you inspected.
- If you get lost, come back to the three questions on missingness
  (Part B) and the two questions on leakage (Part E) — they will
  re-focus you on what matters.

## Acceptance criteria

- `profile.json` exists and contains all the fields listed in
  Part A.
- `REPORT.md` follows the eight-section structure above and is
  written for a fresh reader (no assumed context).
- The manual outlier classification (Part C) covers at least three
  detection strategies, with at least five hand-classified
  outliers per strategy per column with non-trivial outliers.
- The manual label review (Part E) inspects at least 20 rows.
- The top-feature ablation is reported: the baseline model's
  top-3 feature importances and the ablation score are in
  `REPORT.md`.
- `clean()` is deterministic, idempotent, and drops any leaky
  columns identified in Part E. The four listed tests pass.
- Total wall-clock time to run the whole audit (profile + clean +
  tests) is under 60 seconds on the chosen dataset.

## Reflection questions (answer in `REPORT.md`)

- What is the single most important thing you would tell someone
  who is about to model this dataset without doing an audit?
- Which of your findings would you have missed if you'd only
  looked at `df.head()` and `df.describe()`?
- If this dataset were being ingested nightly, which two of your
  checks would you want to run as automated tripwires in CI?
  (Foreshadowing exercise-04.)

## Stretch goals (optional)

- **Adversarial validation.** Train a classifier to distinguish
  train rows from test rows (assuming you have a natural split
  like "before 2026-01-01" vs. "after"). If it succeeds
  (AUC > 0.6), your test set does not look like your train set —
  investigate why.
- **Per-subgroup metrics.** Split the dataset by a categorical
  (e.g. country) and report the top three profile findings that
  differ across subgroups. This is often where fairness /
  quality issues live.
- **Data-drift comparison** between two consecutive partitions of
  your dataset (e.g. Jan vs Feb). Which columns' distributions
  changed most? Compute PSI or KS per column.
- **`cleanlab` integration.** Use `cleanlab.filter.find_label_issues`
  on a quick model's out-of-fold predictions to surface
  automatically-detected likely label errors. Compare against your
  manual review.

## What to hand in

```
exercise-02/
├── audit.py                # initial_profile(), plotting helpers
├── clean.py                # deterministic clean()
├── plots/                  # missingness, outlier, label-drift plots
├── profile.json
├── tests/
│   └── test_clean.py
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
