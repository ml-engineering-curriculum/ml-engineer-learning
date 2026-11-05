# quiz-01 — Data Engineering for ML: Ingestion, Cleaning, Feature Engineering

> **Format:** 12 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–10 of this module.
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter the point
> comes from — go re-read that section if you missed the question.

---

## Questions

### 1. (Chapter 01) Chapter 01 calls a dataset "ML-ready" when it has five properties. A colleague hands you a Parquet file where every column has a stable dtype, the row grain is documented, and the train/test split is deterministic with a seed — but they built the features by averaging values across the whole file *before* splitting, and there is no code anywhere that asserts a schema on the next batch. Which ML-ready properties are missing?

A. None — the three the colleague did (types, row grain, reproducible split) are
   the only three that matter; the rest are optional polish.
B. Only "verifiable". The leakage concern is a modelling problem, not a
   data-engineering one.
C. "Free of the labels you are trying to predict" **and** "verifiable".
   Averaging across the full file before splitting is a leakage pathway
   (future statistics bleed into training rows), and without a schema or
   distribution check, nothing fires when the next batch changes shape.
D. Only "cleanly joined" — the file was already Parquet, so by definition
   it is ingestion-ready.

### 2. (Chapter 02) Your `events.csv` has a `zip_code` column with values like `"01234"` and an `amount` column where ~2% of rows are empty. You run `pd.read_csv("events.csv")` and are later surprised that `zip_code` is an integer with the leading zero gone and that `amount` is `float64` even though the source system emits integers. What is the fix chapter 02 recommends?

A. Pre-process the CSV with `sed` to quote the values, then re-run the same
   `read_csv` call.
B. Pass `dtype={"zip_code": "string", "amount": "Int64"}` to `read_csv`;
   the chapter's rule is "always pass an explicit `dtype=` for any column
   whose type is load-bearing", and the capital-`I` `Int64` is the nullable
   integer that survives missing values without upcasting to `float64`.
C. Add `low_memory=False`; that is what preserves integer semantics.
D. Switch to `pd.read_json` on a converted file — JSON is immune to this
   class of issue.

### 3. (Chapter 02) Your Parquet file has 200 columns and billions of rows, partitioned by `date` in S3. You need `user_id`, `ts`, and `amount` for one specific day. Which read does the chapter say is appropriate, and why?

A. `pd.read_parquet("s3://bucket/events/", columns=None)` and then
   `.query("date == '2026-08-01'")[['user_id','ts','amount']]` — the IO
   cost is the same either way because Parquet is columnar.
B. `pd.read_csv` on a materialised export — Parquet is an interchange
   format, not a read format.
C. `pyarrow.dataset.dataset("s3://bucket/events/", format="parquet",
   partitioning="hive").to_table(filter=ds.field("date") == "2026-08-01",
   columns=["user_id","ts","amount"]).to_pandas()` — this combines
   **partition pruning** (listing only `date=2026-08-01/`), **row-group
   filtering**, and **column projection** so only the bytes you asked for
   are downloaded.
D. Download the whole dataset to local disk first with `aws s3 cp`; then
   do the filter in Pandas on the local copy.

### 4. (Chapter 03) A teammate's ingestion script builds a connection URL as `f"postgresql://{user}:{password}@{host}/{db}"`, passes a SQL string to `pd.read_sql` using `f"WHERE ts >= '{start_ts}'"`, and reads straight from the production OLTP primary. Chapter 03 identifies three separate hazards here. Which option captures all three fixes?

A. Keep the f-string URL (it's convenient) but switch to a read replica;
   nothing else is wrong because SQLAlchemy sanitises everything.
B. Use `sqlalchemy.URL.create(...)` so the password is URL-encoded
   correctly and never built by string interpolation; use
   `pd.read_sql(text("... WHERE ts >= :start"), conn, params={"start": start_ts})`
   so the filter is a bound parameter (not string substitution); and
   point the connection at the read replica (or the warehouse) rather than
   the production OLTP primary.
C. Add `pool_pre_ping=True` — that alone addresses credentials, injection,
   and load, because it re-authenticates on every checkout.
D. Replace Pandas with Polars; its SQL reader is immune to credential
   leakage and injection.

### 5. (Chapter 04) You have a partitioned Parquet dataset in S3 at `s3://bucket/events/date=YYYY-MM-DD/country=XX/...`. Which statement about reading it with `pyarrow.dataset` reflects what chapter 04 says actually happens?

A. Partition pruning only works for `date=`; a second level like `country=`
   is ignored and all country sub-directories are listed.
B. Three optimisations compose: **partition pruning** lists only the
   matching `date=` / `country=` prefixes; **row-group filtering** inside
   each Parquet file skips row groups whose min/max statistics don't
   overlap the filter; and **column projection** fetches only the bytes
   for the columns you asked for. The net effect is a "yesterday, three
   columns, one country" query that costs seconds on terabytes of data.
C. `pyarrow.dataset` always reads every byte of every file in the
   directory tree and then filters in memory.
D. Column projection and row-group filtering are mutually exclusive — you
   pick one at dataset creation time.

### 6. (Chapter 05) You profile a dataset and find `last_purchase_ts` is 60% missing. Before imputing, chapter 05 tells you to ask three questions. Which option both names the questions correctly and matches the chapter's recommended response for a column where "missing" in the domain means "the user has never purchased"?

A. The three questions are "how much is missing?", "why is it missing?",
   and "what does missing *mean* in this domain?". For a column where
   missing means "never purchased", the chapter's recommended response is
   **not** to impute with a sentinel timestamp from 1970 — treat "never
   purchased" as a separate categorical state (or add a `_was_missing`
   indicator), because the missingness itself is informative signal.
B. The only question that matters is "how much is missing?"; below 70% you
   impute with the column median and move on.
C. Impute with `pd.Timestamp("1970-01-01")` so the dtype stays
   `datetime64[ns, UTC]` and downstream code doesn't have to handle
   `NaT`.
D. Drop the column — anything above 30% missing is unusable per the
   chapter's heuristic.

### 7. (Chapter 06) You have a user-level dataset where each user appears in many rows. The business goal at prediction time is to score users the model has **never seen before**. Which cross-validation setup does chapter 06 call out as mandatory here, and what does using a plain `KFold` instead buy you?

A. `KFold(shuffle=True, random_state=0)`. There is no difference versus
   `GroupKFold` when the dataset is large enough.
B. `TimeSeriesSplit`. The split must be time-ordered regardless of the
   grouping structure.
C. `GroupKFold(n_splits=5)` with `groups=X["user_id"]`. A plain `KFold`
   puts rows from the same user in both train and test, so the model
   memorises per-user baselines (typical spend, baseline heart rate,
   session length) that will not help at inference because the inference
   user is new — chapter 06 warns this can overstate the online metric
   by 10–30 percentage points.
D. `StratifiedKFold`. Group leakage is just a class-imbalance issue.

### 8. (Chapter 06) Which of the following features, computed on a training row with `label_ts = 2026-04-15`, is a **temporal leak** by chapter 06's "as-of" test?

A. `country`, taken verbatim from the user's profile row that was last
   updated on `2026-02-01`.
B. `avg_session_duration_90d`, computed as the mean over all of this
   user's sessions regardless of their timestamp — including sessions
   from April 20, May, and June — because the training data has them.
C. `age_at_event`, computed as `label_ts.year - birth_year`.
D. `is_weekend`, computed from `label_ts.dayofweek >= 5`.

### 9. (Chapter 07) Your dataset has a high-cardinality `city` column (~20 000 unique values). A colleague's PR computes the per-city mean of `y` on the full training DataFrame once, merges it on as a new `city_te` feature, and then runs `cross_val_score` on the resulting matrix. Chapter 07 calls this out as "textbook target leakage". Why, and what is the recommended fix?

A. It leaks because `GridSearchCV` is not compatible with pre-computed
   features. Fix: use `RandomizedSearchCV`.
B. It leaks because every training row's `city_te` value was computed
   using that row's own `y` — so each CV fold's inner-validation rows
   see target information from themselves. The fix is
   `sklearn.preprocessing.TargetEncoder(smooth="auto", cv=5)` **inside
   a `Pipeline`**, which cross-fits the encoding (each row's encoding
   comes from folds that exclude that row) and smooths small
   categories toward the global mean.
C. It leaks only if the target is categorical; for regression targets
   there is no issue.
D. It does not leak — the merge is a deterministic lookup, which is why
   the chapter recommends this exact pattern.

### 10. (Chapter 08) You train a text classifier with `vec = TfidfVectorizer(min_df=5, ngram_range=(1,2), sublinear_tf=True); X = vec.fit_transform(all_texts); clf.fit(X[train_idx], y[train_idx])` and get a strong CV score. A reviewer says this is leaky. Chapter 08 agrees — what is leaking, and what is the idiomatic fix?

A. Nothing leaks — `TfidfVectorizer` is stateless; `fit_transform` on the
   full corpus is equivalent to fitting on the training slice.
B. The learning-rate schedule leaks; wrap in a `GridSearchCV`.
C. The vocabulary learned on the full corpus and the IDF weights both
   come from `all_texts`, so test-document vocabulary presence and
   test-document frequency bleed into the fitted transformer. The fix is
   to put the vectoriser **inside a `Pipeline`** and only ever call
   `Pipeline.fit` on `X_train`, so the vocabulary and IDF are fit on
   the training portion alone.
D. Remove stop words by hand before fitting — that is what prevents the
   leak.

### 11. (Chapter 09) You use a pretrained ResNet-50 for feature extraction. Training runs great. In production, every prediction is slightly wrong in an oddly consistent way. Which chapter-09 failure mode most plausibly fits, and what single API habit prevents the whole class of bug?

A. Random-seed drift. Use `random_state=0` everywhere.
B. The pretrained weights expire. Re-download them nightly.
C. Train/serve preprocessing skew: one of input size, channel order
   (RGB vs BGR), normalisation stats, dtype (`uint8[0,255]` vs
   `float32[0,1]`), colour space, or interpolation method is subtly
   different between train and serve. The chapter's habit that eliminates
   the whole class: use **`weights.transforms()`** from the torchvision
   weights object (or the equivalent helper for your model) in both
   training and serving, so training and serving load the exact same
   transform pipeline.
D. The ResNet's classification head is not re-trained. Fine-tune it.

### 12. (Chapter 10) Your training script calls `TrainingBatch.validate(df)` from a `pandera` schema. In CI the validation fails against a new batch. A teammate wraps the call in `try: ... except SchemaError: logger.warning(...)` so "the pipeline doesn't block the retrain". Chapter 10 argues this is exactly wrong. Which option names both the right UX for a validation failure and the right `pandera` flag to make the failure debuggable?

A. The teammate is right — warnings are the mature pattern. Add
   `logger.warning` and remove the schema so it stops firing.
B. A failed validator should **fail loud, not degrade silently**:
   raise, stop the pipeline, attribute the failure (which column, which
   rows), route it to a human, and snapshot the bad batch under
   `failed_batches/<timestamp>/`. The `pandera` flag that makes the
   failure debuggable is `lazy=True` on `.validate(...)`, which collects
   **all** schema failures before raising (via `.failure_cases` on the
   resulting `SchemaErrors`) rather than stopping at the first one.
C. Catch the exception and retrain on the raw data — label noise will
   smooth it out.
D. Switch to Great Expectations exclusively; `pandera` has no notion of
   loud failure.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **C.** Chapter 01 ("A working definition of 'ML-ready'") lists five
   properties: correctly typed, cleanly joined, free of the labels
   you are trying to predict, split reproducibly, and verifiable.
   Computing statistics on the whole file before splitting is leakage
   (property 3), and having no schema or distribution check on the
   next batch violates verifiability (property 5). The other three
   properties being satisfied does not rescue the dataset.

2. **B.** Chapter 02 ("CSV: use it when you must") lists both habits
   as "day-one" rules: pass explicit `dtype=` for load-bearing
   columns, and use the nullable `Int64` (capital `I`) when a column
   has missing values. The chapter also calls out the "1.0 surprise"
   — a legacy `int64` column with even one missing value is upcast to
   `float64`.

3. **C.** Chapter 02 ("Row groups and partitioning") and the
   "pyarrow.dataset" section describe this three-way composition:
   Parquet is columnar, Hive partitioning prunes directories, and
   `pyarrow.dataset` is the engine that composes all three. The
   chapter notes this is "the right call whenever you want filter
   pushdown or lazy schema inspection".

4. **B.** Chapter 03 ("Credentials, done right", "`pd.read_sql`: the
   three signatures", and "Being a good neighbour on a shared
   database") each name one of the three hazards. The chapter's
   explicit fixes are `sqlalchemy.URL.create` for URL construction,
   `text(...)` + `params={}` for bound parameters (not string
   interpolation — SQL injection is not possible with bound
   parameters), and "read from a replica, not the primary" / "do not
   do analytical work on an OLTP database at all".

5. **B.** Chapter 04 ("Partitioned Parquet datasets") names the three
   optimisations by name and explicitly states that they compose:
   partition pruning + row-group filtering + column projection. The
   chapter's framing: "a dataset with terabytes across years of
   partitions can serve a 'yesterday, three columns, one country'
   read in seconds".

6. **A.** Chapter 05 ("Missing data: three questions") names them as
   "how much is missing?", "why is it missing?", and "what does
   missing *mean* in this domain?". For the `last_purchase_ts`
   example the chapter uses, the recommended response is to encode
   "no purchase" as a separate categorical state rather than imputing
   with a very-old timestamp — and to add a `_was_missing` indicator
   when the missingness itself is informative.

7. **C.** Chapter 06 ("Group leakage") gives this exact example and
   recommends `GroupKFold` (or `GroupShuffleSplit`) with the group
   column passed via `groups=`. The chapter's quantified warning:
   random splits on grouped data can overstate the online metric by
   "10–30 percentage points".

8. **B.** Chapter 06 ("Temporal leakage: the future in the past")
   gives this exact example — averaging over the full user history on
   a January label row includes February–December sessions that
   weren't knowable in January. The "as-of" mental model says: "if I
   were making this prediction live at `event_ts`, would this feature
   value be the same as what I have in the training set?" The other
   three features all pass that test (the profile was last updated
   before the label; age and day-of-week are computed from the label
   timestamp itself).

9. **B.** Chapter 07 ("Target encoding") and chapter 06
   ("Train/test contamination") both call this out. Hand-rolled
   per-category means on the full training set mean each row's
   encoding was computed with its own `y` included — classic target
   leakage. `sklearn.preprocessing.TargetEncoder(smooth="auto", cv=5)`
   inside a `Pipeline` uses cross-fitting (each row's encoding comes
   from folds that exclude that row) and smooths small-count
   categories toward the global mean. Chapter 07's three
   non-negotiable rules: fit inside a `Pipeline`, use cross-fitting,
   smooth toward the global mean.

10. **C.** Chapter 08 ("Leakage rule" under `CountVectorizer` and
    "Pitfalls collected" #1) names this exactly: `fit` on the
    combined corpus leaks test-vocabulary presence and IDF weights
    into training. The fix is the same pattern as every other
    learned transformer: wrap it in a `Pipeline`, call
    `Pipeline.fit` on `X_train` only.

11. **C.** Chapter 09 ("Serving-time preprocessing must match
    training") enumerates the six classic skew vectors (input size,
    channel order, normalisation stats, dtype, colour space,
    interpolation) and recommends `weights.transforms()` as the
    single habit that eliminates the whole class — it returns "the
    exact preprocessing the pretrained weights expect" for use in
    both training and serving.

12. **B.** Chapter 10 ("Failure UX") lists the five rules: fail loud
    not silently; attribute the failure; route to a human; snapshot
    the bad batch; design for a `failed_batches/` directory. The
    `pandera` flag is `lazy=True`, documented in "`pandera`: schema
    as code" — it "collects *all* failures before raising" and
    exposes them via `.failure_cases` on the resulting
    `SchemaErrors`, which is the entire debugging surface.

</details>
