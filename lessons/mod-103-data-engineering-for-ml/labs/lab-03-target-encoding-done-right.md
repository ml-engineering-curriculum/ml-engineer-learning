# lab-03 — Target encoding done right: hand-roll vs. `TargetEncoder`

> **Duration:** ~30–45 minutes.
> **Companion chapters:** [06 — Leakage and label quality](../06-leakage-and-label-quality.md), [07 — Tabular feature engineering](../07-tabular-feature-engineering.md).

## Objective

Reproduce chapter 07's "textbook target leakage" on a tiny dataset
and watch it inflate a cross-validation score. Then replace the
hand-rolled encoding with `sklearn.preprocessing.TargetEncoder`
inside a `Pipeline` and watch the CV score drop back to what the
signal actually supports. By the end you will have (a) felt why
"merge per-category means onto the training rows" is a leak, (b)
seen cross-fitting do its job in a `cross_val_score` run, and (c)
left with the three non-negotiable rules from chapter 07 as working
muscle memory.

This lab is pure scikit-learn on a 1 000-row synthetic dataset.
Exercise-03 Part C plugs `TargetEncoder` into a full pipeline on a
real dataset; this lab is the drill that makes the "what is actually
leaking" conversation concrete.

## Prerequisites

- Python 3.11+, scikit-learn ≥ 1.4 (for `TargetEncoder`), Pandas
  ≥ 2.1, NumPy ≥ 1.26.
- Chapters 05, 06, and 07 read through once.
- Any virtual environment (`uv`, `poetry`, or a plain venv).

## Setup

Create a lab directory and a single file:

```bash
mkdir -p lab-03-target-encoding && cd lab-03-target-encoding
touch te.py
```

Everything below goes in `te.py`. Run with `python te.py` after
each step.

Start the file with the imports:

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
```

## Steps

### Step 1 — A high-cardinality categorical with noisy labels (≈ 5 min)

Chapter 07 says target encoding is "powerful for high-cardinality
categoricals; fatally leaky if done wrong". Build a dataset with
exactly that shape: a `city` column with ~200 unique values, a label
that is **random noise** given `city`, and nothing else in `X`.

Add to `te.py`:

```python
RNG = np.random.default_rng(seed=0)

def build_data(n_rows: int = 1_000, n_cities: int = 200) -> pd.DataFrame:
    cities = [f"city_{i:03d}" for i in range(n_cities)]
    df = pd.DataFrame({
        "city": RNG.choice(cities, size=n_rows),
        "y":    RNG.integers(0, 2, size=n_rows),   # 50/50, unrelated to city
    })
    return df

df = build_data()
print(df.head())
print(f"\nrows={len(df):,}  unique cities={df['city'].nunique()}  "
      f"label rate={df['y'].mean():.3f}")
```

**Expected output.** 1 000 rows, around 200 unique cities, label rate
near `0.5`. **Importantly: the label is unrelated to the city** —
there is no real signal here. A leak-free encoder should score
roughly `0.5` AUC; any CV score meaningfully above `0.5` is a
leakage artifact.

**Checkpoint.** The reason this dataset was constructed with no
signal is to make the leak legible. On real data, target encoding
can also lift a legitimate signal; on this data, *any* lift is leak.

### Step 2 — Hand-roll the leak (≈ 10 min)

The classic wrong move from chapter 07 pitfall #1: compute per-category
target means on the whole DataFrame once, merge them on as a new
feature, then split and CV. Watch what happens.

Add to `te.py`:

```python
def leaky_target_encode(df: pd.DataFrame) -> pd.DataFrame:
    # The leak: `city_te` for each row is computed INCLUDING that row's
    # own y. Smoothing / cross-fitting are nowhere in sight.
    means = df.groupby("city")["y"].mean().rename("city_te")
    return df.merge(means, left_on="city", right_index=True)

df_leaky = leaky_target_encode(df)
print(df_leaky.head())

X_leaky = df_leaky[["city_te"]].to_numpy()
y       = df_leaky["y"].to_numpy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
auc_leaky = cross_val_score(
    LogisticRegression(max_iter=1000), X_leaky, y, cv=cv, scoring="roc_auc",
).mean()
print(f"\nhand-rolled target encoding  CV AUC = {auc_leaky:.3f}")
```

Run it.

**Expected output.** `CV AUC` is **meaningfully above 0.5** —
typically 0.55 to 0.70 depending on how the random seed rolls. The
label is noise; any AUC above 0.5 is the CV folds reading their own
labels back through the encoder.

**Checkpoint.** Three problems chapter 07 enumerates are all present
here:

1. **No `Pipeline`.** The encoder ran once on the full frame before
   `cross_val_score` ever saw the data.
2. **No cross-fitting.** Each row's `city_te` value includes that
   row's own `y`.
3. **No smoothing.** A category with one observation gets
   `city_te ∈ {0.0, 1.0}` — perfectly correlated with that row's
   label.

Any one of these alone is enough to break the CV. Together they
guarantee it.

### Step 3 — Prove it's a leak with a held-out set (≈ 5 min)

Chapter 06 ("Sanity checks") recommends a strict hold-out as the
backstop CV cannot provide. Simulate it: fit the hand-rolled encoder
on `df_train` only, apply it to `df_test`, and score.

Add to `te.py`:

```python
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, stratify=df["y"])

# Fit the encoder on train only — this is actually the LESS leaky form!
means_train = df_train.groupby("city")["y"].mean()
df_test_scored = df_test.assign(city_te=df_test["city"].map(means_train))
# Any test city not seen in train becomes NaN under .map — fall back to the
# training label rate.
df_test_scored["city_te"] = df_test_scored["city_te"].fillna(df_train["y"].mean())

df_train_scored = df_train.assign(
    city_te=df_train["city"].map(df_train.groupby("city")["y"].mean())
)

lr = LogisticRegression(max_iter=1000).fit(
    df_train_scored[["city_te"]], df_train_scored["y"]
)
from sklearn.metrics import roc_auc_score
auc_heldout = roc_auc_score(df_test_scored["y"], lr.predict_proba(df_test_scored[["city_te"]])[:, 1])
print(f"\nhand-rolled encoder, train-only fit, strict hold-out AUC = {auc_heldout:.3f}")
```

Run it.

**Expected output.** `AUC ≈ 0.50` (within a few hundredths). The
held-out AUC falls roughly to chance because now the test rows'
`city_te` values do **not** encode the test rows' own labels — just
noisy per-category rates from the training subset.

**Checkpoint.** The gap between Step 2's CV AUC and Step 3's hold-out
AUC is the chapter-06 "leakage smell test" fired for real: CV score
substantially better than a strict hold-out ⇒ leakage somewhere. In
this lab the leak is loud; in production it's often subtle
("`groupby.transform` over the whole frame", "feature-selection on
all the data", "`SMOTE` before splitting") and the gap is the only
sign.

### Step 4 — The right way: `TargetEncoder` in a `Pipeline` (≈ 10 min)

Chapter 07 lists three non-negotiable rules for target encoding:

1. **Fit inside a `Pipeline`.** Never merge target means by hand.
2. **Use cross-fitting.** Each training row's encoding comes from a
   fold that does not include that row.
3. **Smooth toward the global mean.** Small categories get pulled
   toward the overall rate.

`sklearn.preprocessing.TargetEncoder(smooth="auto", cv=5)` delivers
all three. Build the pipeline:

```python
preprocess = ColumnTransformer(
    transformers=[
        ("te", TargetEncoder(smooth="auto", cv=5, target_type="binary"),
         ["city"]),
    ],
    remainder="drop",
)

pipe = Pipeline([
    ("features", preprocess),
    ("clf",      LogisticRegression(max_iter=1000)),
])

X = df[["city"]]
y = df["y"].to_numpy()

auc_pipe = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
print(f"\nTargetEncoder in a Pipeline       CV AUC = {auc_pipe:.3f}")
```

Run it.

**Expected output.** `CV AUC ≈ 0.50` (again, within a few hundredths
of chance), because:

- The `Pipeline` re-fits the encoder on the *inner training fold*
  of each CV split, so the inner-validation rows never see
  per-category statistics computed from themselves.
- `TargetEncoder(cv=5)` additionally cross-fits *within* each
  training fold: when it computes the training rows' encoded values
  to pass to the next step, each row's value comes from a sub-fold
  that excludes that row.
- `smooth="auto"` shrinks small-count categories toward the global
  mean, so a city with one observation no longer encodes its own
  label perfectly.

**Checkpoint.** Compare the three numbers:

```python
print(f"\nSummary:")
print(f"  hand-rolled encoder, CV on whole frame:   {auc_leaky:.3f}  (leaky)")
print(f"  hand-rolled encoder, strict hold-out:     {auc_heldout:.3f}  (leak-free; truth)")
print(f"  TargetEncoder inside a Pipeline, CV:      {auc_pipe:.3f}  (matches truth)")
```

The CV score with `TargetEncoder` matches the held-out score. That
is the signature of a leak-free setup: your CV number and your
hold-out number agree, to within sampling noise.

### Step 5 — Peek at what `TargetEncoder` learned (≈ 5 min)

The last drill: inspect the fitted encoder's state and confirm the
smoothing in rule 3 is doing something visible.

Add to `te.py`:

```python
pipe.fit(X, y)

fitted_te = pipe.named_steps["features"].named_transformers_["te"]
# The learned encoding is the final, non-cross-fit version used at serve time.
encoding = pd.Series(
    fitted_te.encodings_[0].ravel(),           # one row per category
    index=fitted_te.categories_[0],
    name="city_te",
).sort_values()

counts = df["city"].value_counts()
peek = pd.DataFrame({
    "encoding": encoding,
    "n_obs":    counts,
    "raw_mean": df.groupby("city")["y"].mean(),
}).sort_values("n_obs")

print("\nSmoothing effect — low-count categories pulled toward the global mean:")
print(f"global mean y = {y.mean():.3f}")
print(peek.head(5))      # rarest categories
print("...")
print(peek.tail(5))      # most common categories
```

Run it.

**Expected output.** For the **rare** categories (those with 1–3
observations), `encoding` is noticeably **closer to the global mean**
than `raw_mean` is — chapter 07's rule 3 in action. For the more
**common** categories, `encoding ≈ raw_mean` because the smoothing
has less to correct against.

**Checkpoint.** The encoding you just inspected is what the pipeline
serialises to disk when you `joblib.dump(pipe, ...)` and what the
serving code looks up at predict time. The `cv=5` cross-fitting was
only used during `fit` to produce training-row encodings for the
downstream model; at serve time there is one lookup, period.

> **Note on `target_type="binary"`.** `TargetEncoder` needs to know
> whether you are doing classification (encoding = class mean) or
> regression (encoding = numeric mean). For binary classification
> the chapter's recommended value is `"binary"`; for regression it is
> `"continuous"`. Getting this wrong is a silent correctness bug
> rather than a loud error — be explicit.

## Expected final artefacts

A single file `lab-03-target-encoding/te.py` with the five step
blocks above. End-to-end runtime should be under five seconds.
The script's final console output is the three-line summary
comparing the hand-rolled leaky CV, the strict hold-out, and the
`TargetEncoder`-in-a-pipeline CV.

## Final checkpoint

Without re-reading chapter 07, answer:

1. In Step 2, which of the **three non-negotiable rules** was being
   violated by the hand-rolled encoding? Name all three.
2. In Step 3, why did training the same hand-rolled encoder on
   `df_train` only (not the full frame) make the held-out AUC
   collapse back to chance, even though `df_train` was still used in
   the encoder?
3. In Step 4, what would change if you replaced `TargetEncoder(cv=5)`
   with `TargetEncoder(cv=None)` or called `fit_transform` on the
   full frame outside a `Pipeline`?
4. In Step 5, which value — `encoding`, `raw_mean`, or the global
   mean — does the fitted encoder use for a `city` it has never seen
   before at serve time? (Hint: look up `TargetEncoder`'s
   `handle_unknown` / unseen-category behaviour.)

If you can answer all four, the target-encoding discipline is in
place and you can slot `TargetEncoder` into the high-cardinality
branch of a `ColumnTransformer` in exercise-03 with confidence.

## Where to go next

- `exercise-03` ("Tabular feature engineering") Part C plugs
  `TargetEncoder` into a real `ColumnTransformer` alongside numeric,
  low-cardinality categorical, and datetime branches. The leak you
  reproduced here is the one its "inside a `Pipeline`" rule
  prevents.
- Chapter 10 then turns the invariant into a validator: a
  `@pa.dataframe_check` that fails CI if any categorical column ends
  up target-encoded outside a `Pipeline`. The rule is the same; the
  enforcement layer is what differs.
