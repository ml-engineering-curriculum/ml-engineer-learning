# 07 — Tabular feature engineering

> **Reading time:** ~45 minutes.
> **Prerequisite chapters:** 05, 06, and `mod-102` chapter 06
> (Pipelines, `ColumnTransformer`).

## Motivation

You have a clean, leakage-audited DataFrame. Now turn it into
numbers a model can consume.

Feature engineering is a catalogue of small, well-understood
transformations: scaling numerics, encoding categoricals, extracting
components of a timestamp, binning a skewed distribution, forming
interactions between two columns. Individually, each is a five-line
transformer. Together — inside a `ColumnTransformer` and a
`Pipeline` — they are what separates a scikit-learn `fit` call from
a working model.

This chapter is a working reference, not a survey. It shows the
transformations you will actually reach for, with the
scikit-learn API for each, and the leakage rules that must hold
for each of them.

## Numeric features

### Scaling

Linear models, neural networks, distance-based models (KNN, k-means,
SVMs with RBF) are sensitive to feature scale. Tree ensembles are
not.

Three scalers cover most cases:

- **`StandardScaler`** — subtract the mean, divide by the standard
  deviation. Default for approximately-normal columns.
- **`RobustScaler`** — subtract the median, divide by the IQR.
  Robust to outliers. Use when the column has a heavy tail you
  didn't Winsorise.
- **`MinMaxScaler`** — rescale to `[0, 1]`. Use when a model
  requires bounded inputs (some neural network activations, image
  pixel-like features).

```python
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
```

**Leakage rule.** All three learn parameters from the training data
(mean, median, min, max). They must live inside a `Pipeline` and
`fit` only on the training portion. `mod-102` chapter 06 covered
the mechanics.

### Log-transform

For heavy-tailed positive columns (spend, view count, session
length), `np.log1p(x)` — `log(1 + x)` — compresses the tail and
makes linear models behave. Zero-safe (unlike `np.log`, which
returns `-inf`).

Wrap it in `FunctionTransformer` to keep it inside a `Pipeline`:

```python
from sklearn.preprocessing import FunctionTransformer
log_scaler = FunctionTransformer(np.log1p, validate=True)
```

For a monotone-preserving transform that handles zeros and
negatives, `PowerTransformer` (Box-Cox for positive-only,
Yeo-Johnson otherwise) fits a parameter to make the distribution
approximately normal. Overkill for most cases; useful when a linear
model's residuals are ugly.

### Binning (discretisation)

Turn a numeric column into categorical bins. Two use cases:

1. **Non-linear signal you want a linear model to see.** Age has a
   non-monotonic relationship with conversion; binning into
   `[<25, 25-40, 40-60, 60+]` lets a linear model fit the shape.
2. **Robustness to outliers.** Extreme values get flattened into
   the top bin.

`sklearn.preprocessing.KBinsDiscretizer` handles it:

```python
from sklearn.preprocessing import KBinsDiscretizer
binner = KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile")
```

`strategy="quantile"` puts equal counts in each bin (safer);
`"uniform"` puts equal widths (interpretable but sensitive to
outliers). `encode="onehot"` returns a one-hot matrix; `"ordinal"`
returns integer bin indices.

Do not bin numerics reflexively — you throw away information. Bin
only when you have a reason.

### Polynomial features and interactions

For linear models, meaningful interactions must be expressed
explicitly:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
```

`interaction_only=True` skips `x²` and includes `x*y`. Use
sparingly — the number of columns explodes as `O(n² / 2)` and you
usually only want *specific* interactions (e.g. `age × income`), not
the full set. For targeted interactions, a small
`FunctionTransformer` is cleaner:

```python
add_age_income = FunctionTransformer(
    lambda df: df.assign(age_income=df["age"] * df["income"]),
    validate=False,
)
```

Tree ensembles and gradient boosting learn interactions natively;
no polynomial expansion needed.

### Imputation, at last

`mod-102` chapter 06 introduced `SimpleImputer`. Two more that show
up in practice:

- **`KNNImputer`** — imputes a missing value using the average of
  its k-nearest neighbours in feature space. Good when missingness
  is MAR and features are correlated. Expensive on large data.
- **`IterativeImputer`** (experimental in scikit-learn) — models
  each column with missing values as a regression on the others.
  More principled; often marginal accuracy gains over `SimpleImputer`
  in practice.

Regardless of imputer, **add the missingness indicator**
(`add_indicator=True` on any imputer) so a model can distinguish
imputed from observed values.

```python
from sklearn.impute import SimpleImputer, KNNImputer
imputer = SimpleImputer(strategy="median", add_indicator=True)
```

## Categorical features

### One-hot encoding

The default for low-cardinality categoricals. Chapter 06 of
`mod-102` covered the mechanics:

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=10)
```

Three flags worth setting deliberately:

- **`handle_unknown="ignore"`** — an unseen category at predict
  time produces an all-zero row instead of raising. This is what
  you almost always want in production.
- **`sparse_output=True`** — sparse matrices are more memory-
  efficient for wide categoricals. Trees and linear models handle
  them natively; some other models don't.
- **`min_frequency=`** — bucket categories below the threshold
  into an implicit "infrequent" group. Since scikit-learn 1.1 this
  removes the need for a custom `RareCategoryGrouper` for many
  cases (though the custom version is still a good exercise).

One-hot encoding is fine up to a few hundred unique categories per
column. Beyond that, memory and model complexity become issues, and
one of the alternatives below is better.

### Ordinal encoding

Some categoricals have a natural order (`"low" < "medium" < "high"`,
education levels, ratings). Ordinal encode them:

```python
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(
    categories=[["low", "medium", "high"]],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
```

Do not ordinal-encode nominal categoricals (`country`, `department`)
— you would be lying to the model about ordering.

Tree models can use ordinal encodings of nominal categoricals well
enough (they only care about splits, not order), and it is more
memory-efficient than one-hot for high-cardinality columns. But be
explicit that this is what you're doing.

### Target encoding

Replace each category with a summary statistic of the target for
that category (mean of `y` for classification, mean of `y` for
regression). Powerful for high-cardinality categoricals; **fatally
leaky if done wrong**.

Since scikit-learn 1.3, `TargetEncoder` handles the leakage rules
correctly out of the box — it uses cross-fitting to compute
encodings from held-out folds:

```python
from sklearn.preprocessing import TargetEncoder
te = TargetEncoder(smooth="auto", cv=5, target_type="binary")
```

**Rules for target encoding, non-negotiable:**

1. **Fit inside a `Pipeline`.** Never compute per-category target
   means on the whole dataset and merge them onto training rows.
2. **Use cross-fitting.** Each training row's encoding should come
   from a fold that does not include that row. `TargetEncoder`
   does this; hand-rolled encoders often don't.
3. **Smooth toward the global mean.** Categories with few
   observations should be pulled toward the overall target rate,
   not encoded as the noisy per-category mean.

If any of the three is missing, you are looking at target leakage.

### Hashing

For very high-cardinality categoricals (millions of URLs, user IDs
you cannot enumerate), the hashing trick maps each category to one
of `n_features` buckets via a hash function. No fit step.

```python
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=2**18, input_type="dict")
```

Trade-offs: collisions inject noise; you cannot invert to see what
each feature means; adding a new category is free. Common in
ad-tech and search-ranking pipelines with billions of features.

### Embeddings (foreshadowing)

For very-high-cardinality columns where the categories have
meaningful similarity structure (words, users, items), a learned
embedding is more powerful than one-hot / hashing. Chapter 08
covers text embeddings; entity embeddings for tabular data are a
`mod-105` topic.

## Datetime features

A raw timestamp is not useful to a model. Extract its meaningful
components.

```python
def add_datetime_features(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    ts = df[col]
    return df.assign(
        **{
            f"{col}_year":         ts.dt.year,
            f"{col}_month":        ts.dt.month,
            f"{col}_day":          ts.dt.day,
            f"{col}_dow":          ts.dt.dayofweek,       # 0 = Monday
            f"{col}_hour":         ts.dt.hour,
            f"{col}_is_weekend":   ts.dt.dayofweek >= 5,
            f"{col}_days_since":  (pd.Timestamp.utcnow() - ts).dt.days,
        }
    )
```

Two more that catch specific patterns:

- **Cyclical encoding.** Hour-of-day is cyclical: 23 and 0 are
  adjacent, not far apart. Encode as
  `(sin(2π · hour / 24), cos(2π · hour / 24))` for models that
  need to see the cycle. Trees don't need this; linear models and
  neural nets do.

  ```python
  df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
  df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
  ```

- **Time-since-event.** `days_since_last_purchase`,
  `days_since_signup`. Trivial when the columns exist; make sure
  they respect point-in-time correctness (chapter 06).

**Leakage rule.** `pd.Timestamp.utcnow()` in a training feature is
almost always wrong — it will change between training and
scoring. Use the label's `event_ts` as the "now" for that row and
compute `days_since = (event_ts - reference_ts).dt.days`.

## Aggregations: entity-level features

Most tabular ML gets substantial lift from summarising an entity's
history into features. "How many purchases has this user made in
the last 30 days" is often more predictive than any single event.

The SQL pattern from chapter 03 (windowed aggregations) is the right
tool at scale. For medium-sized data, Pandas is fine:

```python
last_30d = df[df.ts >= (label_ts - pd.Timedelta("30D"))]
user_30d = (
    last_30d
    .groupby("user_id")
    .agg(
        n_events_30d=("event_id", "count"),
        spend_30d=("amount", "sum"),
        median_amount_30d=("amount", "median"),
    )
    .reset_index()
)
features = labels.merge(user_30d, on="user_id", how="left")
```

**Leakage rule.** The window's upper bound must be the label's time
(strictly before, if the label event itself would be inside the
window). See chapter 06's point-in-time discussion.

### Rolling windows

For time-series-like data with regular observations,
`df.groupby("id")["value"].rolling(...)` gives you per-group rolling
statistics. Be explicit about `closed=` (`"left"` excludes the
current row — often what you want to prevent look-ahead).

## Rare categories: putting Grouper in a Pipeline

The exercise in `mod-102` chapter 06 built a custom
`RareCategoryGrouper`. Since scikit-learn 1.1 you can often skip
the custom transformer by using `OneHotEncoder(min_frequency=N)` or
`OneHotEncoder(max_categories=K)`, both of which bucket infrequent
categories automatically.

For a per-column configurable grouper with explicit `"__rare__"`
semantics — useful when you want the same treatment before other
encoders (`TargetEncoder`, `OrdinalEncoder`), or when you want a
readable rare bucket for auditing — a small custom transformer
still earns its keep.

## Feature interactions in tree models

Gradient-boosted trees (`HistGradientBoostingRegressor`,
`XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`) automatically
learn interactions from raw features. You do not need
`PolynomialFeatures`. In particular:

- `HistGradientBoosting*` handles missing values natively — no
  `SimpleImputer` needed. Missing goes to the branch that
  minimises loss.
- `CatBoost` handles categoricals natively (target encoding with
  ordering-based leakage prevention baked in) — no
  `OneHotEncoder` needed.
- LightGBM handles categoricals natively if you cast the column to
  `category` and set `categorical_feature=`. Faster than one-hot.

This does not mean the `Pipeline` becomes optional. You still need
it for deterministic serving and for the CV discipline. It just
means the preprocessing steps get shorter for tree models.

## A worked example

Assemble the pieces on a hypothetical `users × sessions` dataset:

```python
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, TargetEncoder, FunctionTransformer,
)
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

# Numeric branch for a linear-model-friendly pipeline.
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("log",    FunctionTransformer(np.log1p, validate=True)),
    ("scale",  StandardScaler()),
])

# Low-cardinality categoricals get one-hot with rare-bucketing.
low_card = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
])

# High-cardinality categoricals (user_country, product_id): target-encode.
high_card = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("te",     TargetEncoder(smooth="auto", cv=5)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num",  numeric_pipe, make_column_selector(dtype_include="number")),
        ("lowc", low_card,     ["country", "device"]),
        ("hic",  high_card,    ["city", "product_id"]),
    ],
    remainder="drop",
)

model = Pipeline([
    ("features", preprocess),
    ("clf",      HistGradientBoostingClassifier(random_state=0)),
])
```

Notes:

- `HistGradientBoostingClassifier` doesn't need scaling; keeping it
  here makes swapping in a `LogisticRegression` a one-line change.
- `TargetEncoder` inside a `Pipeline` inside `cross_val_score` uses
  its cross-fitting logic correctly. It is the *only* safe way to
  target-encode.
- `remainder="drop"` is explicit — no silent column-dropping bugs.

## Pitfalls collected

1. **Target encoding done by hand.** Merging per-category target
   means onto the training data before fitting is textbook target
   leakage. Use `TargetEncoder`, always inside a `Pipeline`.
2. **Scaling everything, including trees.** No benefit; adds
   complexity and a fitted parameter to serve.
3. **`OneHotEncoder` without `handle_unknown="ignore"`.** Any
   category not seen at fit time raises at predict time.
   Deploy-time headache.
4. **Cyclical encoding done "just because".** Only cycles are
   cyclical. Applying to `month_of_year` is fine; applying to
   `credit_score` is nonsense.
5. **`PolynomialFeatures(degree=3, interaction_only=False)` on
   50 columns.** You have created ~22,000 features. Your model
   will fit noise beautifully.
6. **Aggregations that use the label row.** A `user_avg_spend`
   feature computed over the user's full history includes the
   transaction you are trying to predict on. Filter to
   `session_ts < label_ts` explicitly (chapter 06).

## Summary

- Numerics: scale for linear/NN/distance models, log-transform
  heavy tails, bin only when you have a reason, add missingness
  indicators.
- Categoricals: one-hot for low cardinality, target encoding
  (`sklearn.preprocessing.TargetEncoder`) for high cardinality,
  hashing for extreme cardinality, embeddings for semantic
  similarity (chapter 08 / `mod-105`).
- Datetimes: extract components, add cyclical encoding for
  cycles, add time-since-event with `event_ts` as the "now".
- Aggregations: summarise entity history into features; respect
  point-in-time correctness.
- Everything inside a `Pipeline` + `ColumnTransformer`. That is
  what enforces the leakage rules from chapter 06 and what makes
  training/serving parity possible.

## Where this goes next

Chapters 08 and 09 extend the same discipline to text and images —
different transformers, same `Pipeline` shape, same leakage rules.
