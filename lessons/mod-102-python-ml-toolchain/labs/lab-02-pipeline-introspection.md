# lab-02 — Pipeline introspection: see what your model actually sees

> **Duration:** ~45–60 minutes.
> **Companion chapters:** [05 — scikit-learn: estimators and transformers](../05-sklearn-estimators-and-transformers.md), [06 — Pipelines and `ColumnTransformer`](../06-pipelines-and-columntransformer.md).

## Objective

Build a tiny `Pipeline` + `ColumnTransformer` on a 20-row toy table
and then **look inside it**. By the end you will have (a) used
`set_output(transform="pandas")` to render the transformed matrix with
readable column names, (b) used `pipe[:-1].fit_transform(X)` to inspect
exactly what the final classifier sees, (c) tuned a nested
hyperparameter with the `step__param` syntax, and (d) seen a
`remainder="drop"` bug on purpose so you recognise it later.

This is a short guided tour of the inspection techniques chapter 06
calls out, on a dataset small enough that you can hand-check every
cell. The longer `exercise-02` builds a real pipeline on a real
dataset with a `GridSearchCV` sweep — this lab is the drill that makes
that exercise easier.

## Prerequisites

- Python 3.11+, scikit-learn ≥ 1.4, Pandas ≥ 2.1, NumPy ≥ 1.26.
- Chapters 05 and 06 read through once.
- Any virtual environment (`uv`, `poetry`, or a plain venv).

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-02-pipeline-introspection && cd lab-02-pipeline-introspection
touch inspect_pipeline.py
```

All code below goes in `inspect_pipeline.py`. Run with `python
inspect_pipeline.py` after each step.

## Steps

### Step 1 — A toy heterogeneous table (≈ 5 min)

Add to `inspect_pipeline.py`:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=0)

df = pd.DataFrame({
    "age":      [22, 31, 45, np.nan, 29, 38, 51, 27, np.nan, 44,
                 33, 48, 25, 36, 42, 30, 55, 23, 39, 47],
    "income":   [40_000, 80_000, 120_000, 95_000, 60_000, 72_000,
                 150_000, 55_000, 88_000, 110_000, 70_000, 135_000,
                 50_000, 78_000, 102_000, 65_000, 160_000, 48_000,
                 85_000, 125_000],
    "country":  ["US", "UK", "US", "FR", "US", "UK", "DE", "FR",
                 "US", "DE", "UK", "US", "FR", "DE", "UK", "US",
                 "DE", "FR", "US", "UK"],
    "plan":     ["basic", "pro", "pro", "basic", "basic", "pro",
                 "pro", "basic", "pro", "pro", "basic", "pro",
                 "basic", "pro", "basic", "pro", "pro", "basic",
                 "pro", "pro"],
    "notes":    ["new"] * 20,                       # a filler column
})

y = (df["income"] > 80_000).astype(int).to_numpy()  # toy binary label

print(df.head())
print("dtypes:\n", df.dtypes)
```

**Expected output.** A 20-row DataFrame printed, with `age` as
`float64` (because of the NaNs), `income` as `int64`, and `country`,
`plan`, `notes` as `object`.

**Checkpoint.** You deliberately have a filler column (`notes`) that
is *not* a useful feature. We will come back to it in step 5 to see a
`remainder="drop"` bug on purpose.

### Step 2 — A first pipeline with `ColumnTransformer` (≈ 10 min)

Add:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

numeric_cols     = ["age", "income"]
categorical_cols = ["country", "plan"]

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe,     numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop",                               # deliberate; see Step 5
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf",        LogisticRegression(max_iter=1000, random_state=0)),
])

pipe.fit(df, y)
print("fitted score on training:", pipe.score(df, y))
```

Run it.

**Expected output.** A single line like `fitted score on training:
0.95` (the exact number will vary slightly; it is a toy problem with a
label that is a smooth function of `income`).

**Checkpoint.** The pipeline has three named steps once you unroll it:
`preprocess.num.impute`, `preprocess.num.scale`, and
`preprocess.cat.ohe`, plus the final `clf`. Hold that structure in
your head — it's the one you address in step 6.

### Step 3 — Render the transformed matrix as a pandas DataFrame (≈ 10 min)

Chapter 06 ("Inspecting the pipeline") flags `set_output(transform=
"pandas")` as a quality-of-life win for debugging. Try it:

```python
preprocess.set_output(transform="pandas")

X_transformed = preprocess.fit_transform(df)
print("\nTransformed matrix (first 5 rows):")
print(X_transformed.head())
print("\nTransformed shape:", X_transformed.shape)
print("Columns:", list(X_transformed.columns))
```

**Expected output.** A 20-row DataFrame with columns like:

```text
num__age  num__income  cat__country_DE  cat__country_FR  cat__country_UK
cat__country_US  cat__plan_basic  cat__plan_pro
```

Exactly which names you see depends on the sklearn version, but the
pattern is `<columntransformer-step>__<original-column>` for numeric
and `<columntransformer-step>__<original-column>_<category>` for the
one-hot outputs.

**Checkpoint.** Before `set_output("pandas")` this same matrix would
have been a positional NumPy array with no names — fine for the
classifier, awful for a human reading SHAP output or chasing a
suspicious feature. The chapter's exact claim is "Once you have seen
a `ColumnTransformer` output displayed as a DataFrame with named
columns, you do not go back." You are now in that camp.

### Step 4 — Inspect what the classifier actually sees (≈ 10 min)

Chapter 06 ("Inspecting the pipeline") flags `pipe[:-1]` as the
"single most useful pipeline-debugging expression". Try it:

```python
# Rebuild a clean pipeline so set_output flows through it too.
pipe_debug = Pipeline([
    ("preprocess", preprocess),
    ("clf",        LogisticRegression(max_iter=1000, random_state=0)),
])
pipe_debug.fit(df, y)

# Slice off the final predictor. What's left is the preprocessing chain.
print("\nWhat the classifier sees (first 3 rows):")
print(pipe_debug[:-1].fit_transform(df.head(3)))

# Access a step by name.
print("\nFitted imputer medians (num branch):",
      pipe_debug.named_steps["preprocess"]
                .named_transformers_["num"]
                .named_steps["impute"].statistics_)
```

**Expected output.** First, a 3-row DataFrame of the preprocessed
matrix (same columns you saw in step 3). Second, a 2-element array:
the learned medians for `age` and `income` on the full training data.

**Checkpoint.** Three things to take away:
- `pipe[:-1]` is a slice that returns "everything except the final
  step". You can call `.fit_transform(X)` on it to see exactly the
  matrix the final estimator consumes.
- `named_steps["name"]` reaches into a step by name.
- Learned state is on attributes ending `_` (chapter 05): the imputer
  stores its learned medians on `.statistics_`.

### Step 5 — The silent `remainder="drop"` bug (≈ 10 min)

Chapter 06 ("Pitfalls collected" #2) calls out
`remainder=` left as default as a common source of "I added a
feature and it had no effect". Reproduce it on purpose.

Notice that your `ColumnTransformer` lists `numeric_cols = ["age",
"income"]` and `categorical_cols = ["country", "plan"]` — but `df`
also has a `notes` column. What happened to it?

Add:

```python
print("\nColumns in df:",            list(df.columns))
print("Columns transformed:",        list(X_transformed.columns))
print("Column dropped by default:",  set(df.columns) - {"age", "income", "country", "plan"})
```

**Expected output.**
```text
Columns in df:            ['age', 'income', 'country', 'plan', 'notes']
Columns transformed:      ['num__age', 'num__income', 'cat__country_DE', ...]
Column dropped by default: {'notes'}
```

The `notes` column is gone. In this lab it's a filler column so you
*wanted* that outcome — but if `notes` had been a real feature that
you forgot to list, your model would train on `notes`-less input and
you would spend an afternoon wondering why the new feature "did
nothing".

Now change `remainder="drop"` to `remainder="passthrough"` and re-fit:

```python
preprocess_pass = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe,     numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="passthrough",
)
preprocess_pass.set_output(transform="pandas")
print("\nWith remainder='passthrough':")
print(preprocess_pass.fit_transform(df).head(3))
```

**Expected output.** A new column appears (something like
`remainder__notes`) carrying the untransformed `notes` values through.

**Checkpoint.** You just reproduced the chapter's pitfall.
`remainder="drop"` is fine *when it is a deliberate choice*; it is
dangerous as a silent default. Always set it explicitly, and name
the columns it will act on.

### Step 6 — Tune a nested hyperparameter with `step__param` (≈ 10 min)

Chapter 06 ("`Pipeline`: a chain of named steps") says the double-
underscore is how scikit-learn flattens nested names. Prove to
yourself that you can address an inner step from the outside.

Add:

```python
from sklearn.model_selection import GridSearchCV

pipe_search = Pipeline([
    ("preprocess", ColumnTransformer(
        transformers=[
            ("num", numeric_pipe,     numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )),
    ("clf", LogisticRegression(max_iter=1000, random_state=0)),
])

param_grid = {
    "preprocess__num__impute__strategy": ["median", "mean"],
    "clf__C":                             [0.1, 1.0, 10.0],
}

search = GridSearchCV(pipe_search, param_grid, cv=5)
search.fit(df, y)

print("\nbest params:", search.best_params_)
print("best CV score:", search.best_score_)
```

**Expected output.** A `best_params_` dict naming one imputer strategy
and one `C`, and a CV score (around `0.6`–`0.85` on this toy
problem — the dataset is too small for the number to be meaningful,
but the *mechanics* are what you're verifying).

**Checkpoint.** Read the parameter name
`preprocess__num__impute__strategy` left to right: it addresses
`pipe_search["preprocess"]["num"]["impute"].strategy`. Each
double-underscore is one level of nesting. If you rename any step
(e.g. `impute` → `imp`), the grid key must change with it; the
chapter calls this out as "step naming collisions in `GridSearchCV`"
(pitfall #4).

## Expected final artefacts

A single file `lab-02-pipeline-introspection/inspect_pipeline.py` with
the six step blocks above. The artefact is in your fingers — three
debugging techniques (`set_output("pandas")`, `pipe[:-1]`,
`named_steps[...]`) and one grid-search syntax (`step__param`).

## Final checkpoint

Without re-reading chapter 06, answer:

1. In step 3, which feature produced the columns `cat__country_*`?
   What would happen to `cat__country_IT` at predict time if
   `handle_unknown="ignore"` were not set on the `OneHotEncoder`?
2. In step 4, if you called `pipe_debug[:-2].fit_transform(df.head(3))`
   instead, what would you get? (Hint: the pipeline has two steps.
   What does `[:-2]` slice to?)
3. In step 5, you changed `remainder="drop"` to
   `remainder="passthrough"`. For which kind of downstream model
   would `"passthrough"` be dangerous — and why would the chapter
   nonetheless still recommend you set it *explicitly*, either way?
4. In step 6, write out the grid-key you would use to tune the
   `OneHotEncoder`'s `min_frequency` parameter inside the
   categorical branch. Check your answer by inspecting the keys of
   `pipe_search.get_params()`.

If you can answer all four, you have internalised the inspection
surface chapter 06 flags as highest-leverage.

## Where to go next

- `exercise-02` ("scikit-learn pipeline design") builds a full
  pipeline on a real dataset (Adult or Ames Housing), adds a custom
  `RareCategoryGrouper`, runs a bigger `GridSearchCV`, and
  demonstrates train/test leakage in a controlled A/B. The
  inspection techniques from this lab are how you will debug that
  exercise.
- Chapter 08 picks up testing — including how to unit-test the
  custom transformers you'll write in exercise-02 and how to golden-
  test an end-to-end pipeline like the one above.
