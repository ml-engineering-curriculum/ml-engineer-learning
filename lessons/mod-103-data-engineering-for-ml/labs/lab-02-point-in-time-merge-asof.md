# lab-02 — Point-in-time features with `merge_asof`

> **Duration:** ~30–45 minutes.
> **Companion chapters:** [06 — Leakage and label quality](../06-leakage-and-label-quality.md), [07 — Tabular feature engineering](../07-tabular-feature-engineering.md).

## Objective

Reproduce temporal leakage on purpose on a hand-built DataFrame small
enough to eyeball, then fix it with the two tools chapter 06 names:
`pd.merge_asof(..., allow_exact_matches=False, direction="backward")`
for the "as-of" join, and a time-ordered split for the CV. By the end
you will have (a) watched a `groupby.rolling` feature leak the label
row's own target into itself, (b) seen that leak show up as an
implausibly good CV score, and (c) replaced it with a provably
point-in-time-correct feature that drops the CV score back to earth.

This lab is pure Pandas + scikit-learn on a 50-row toy table.
Exercise-03 Part D builds the same discipline into a full `Pipeline`
on a real dataset; this lab is the isolation chamber where you can
*see* the leak happen one row at a time.

## Prerequisites

- Python 3.11+, Pandas ≥ 2.1, scikit-learn ≥ 1.4, NumPy ≥ 1.26.
- Chapters 05, 06, and 07 read through once.
- Any virtual environment (`uv`, `poetry`, or a plain venv).

## Setup

Create a lab directory and a single file:

```bash
mkdir -p lab-02-point-in-time && cd lab-02-point-in-time
touch asof.py
```

Everything below goes in `asof.py`. Run with `python asof.py` after
each step.

Start the file with the imports:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
```

## Steps

### Step 1 — A tiny sessions / labels dataset (≈ 5 min)

Build 5 users × 10 sessions each. The label is "will the user buy in
the next session" — the kind of target that is trivially leakable by
features computed over the user's whole history.

Add to `asof.py`:

```python
RNG = np.random.default_rng(seed=0)

def build_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    users = list("ABCDE")
    rows = []
    for u in users:
        base_rate = RNG.uniform(0.2, 0.7)       # per-user propensity
        for i in range(10):
            ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=5 * i + ord(u))
            duration = RNG.exponential(120.0)
            buys_next = int(RNG.uniform() < base_rate)
            rows.append((u, ts, duration, buys_next))
    sessions = pd.DataFrame(rows, columns=["user_id", "ts", "duration_s", "buys_next"])
    sessions = sessions.sort_values(["user_id", "ts"]).reset_index(drop=True)

    labels = sessions[["user_id", "ts", "buys_next"]].rename(columns={"buys_next": "y"})
    sessions = sessions.drop(columns=["buys_next"])
    return sessions, labels

sessions, labels = build_data()
print(sessions.head(8))
print("\nlabels head:\n", labels.head(8))
print("\nlabel rate:", labels["y"].mean().round(3))
```

**Expected output.** A sessions table with 50 rows (5 users × 10
sessions), each row carrying `user_id`, `ts` (UTC timestamps), and
`duration_s`. A paired labels table with the same `user_id` / `ts`
and a `y`. The overall label rate is around `0.45` (random by
construction, with per-user variation).

**Checkpoint.** Two separate frames on purpose. The leakage will
sneak in when we compute a feature on `sessions` and merge it back to
`labels` without being careful about time. The "as-of" mental model
from chapter 06: **at `labels.loc[i]`, only rows of `sessions` with
`ts < labels.loc[i, 'ts']` should be visible**.

### Step 2 — The leaky feature: full-history rolling mean (≈ 10 min)

Chapter 06's canonical leak: a feature computed over the *entire*
user history, used as a feature on a label row that is itself inside
that history. Let's build it exactly the wrong way first.

Add to `asof.py`:

```python
def leaky_feature(sessions: pd.DataFrame) -> pd.Series:
    """Per-user mean duration over the WHOLE history, broadcast back."""
    # `transform("mean")` returns one value per row — the group mean —
    # computed across every row in the group, including the row itself
    # and any future row. This is the leak.
    return sessions.groupby("user_id")["duration_s"].transform("mean")

sessions["mean_duration_leaky"] = leaky_feature(sessions)

# Join to labels on (user_id, ts). The leak is that each label's
# feature was computed using sessions AFTER that label's timestamp.
leaky = labels.merge(
    sessions[["user_id", "ts", "mean_duration_leaky"]],
    on=["user_id", "ts"], how="inner",
)
print(leaky.head(6))
```

**Expected output.** Each row's `mean_duration_leaky` is the same
within a user (the full-history mean is constant per user).

**Checkpoint.** Read that last sentence again. The *same* user has
the *same* feature value on every one of their rows, including on
rows where the feature was computed using sessions chronologically
after that row's timestamp. That is exactly the "future in the past"
leak from chapter 06 — the feature "knows" what the user's future
looks like.

### Step 3 — Measure the leak: implausibly good CV with a time split (≈ 10 min)

Chapter 06 lists the "smell tests" for leakage. The clearest one for
temporal leaks: compare a time-ordered CV score against a baseline.
Here, we don't even need a baseline — the signal is noise by
construction, so a leak-free feature should score near 0.5 AUC.

Add to `asof.py`:

```python
def time_cv_score(df: pd.DataFrame, feature_cols: list[str]) -> float:
    df = df.sort_values("ts").reset_index(drop=True)
    X = df[feature_cols].to_numpy()
    y = df["y"].to_numpy()
    cv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(
        HistGradientBoostingClassifier(random_state=0),
        X, y, cv=cv, scoring="roc_auc",
    )
    return float(scores.mean())

auc_leaky = time_cv_score(leaky, ["mean_duration_leaky"])
print(f"leaky feature  AUC = {auc_leaky:.3f}")
```

Run it.

**Expected output.** An AUC somewhere **well above 0.5** — typically
in the 0.65–0.85 range. The label is pure per-user noise with no
real signal in `duration_s`, so a legitimately leak-free feature
should score `≈ 0.5`. The elevated AUC is the leak: the
full-history mean encodes the user's base rate (which *does* predict
`y`), which only works because the feature saw the future.

**Checkpoint.** You just reproduced the chapter-06 pattern "CV score
is implausibly good → look for leakage". On a real dataset this is
how temporal leakage usually surfaces: the number looks great, and
when the model ships, the online metric collapses because the serving
feature value is computed only from the past and no longer encodes
the future.

### Step 4 — The fix: `merge_asof` with `allow_exact_matches=False` (≈ 10 min)

Chapter 06 names the Pandas form of a point-in-time join:
`pd.merge_asof(direction="backward", allow_exact_matches=False)`. The
`allow_exact_matches=False` is what makes "strictly before" work —
without it, the label row's own session is included in its own
rolling window.

Add to `asof.py`:

```python
def asof_mean_duration(
    labels: pd.DataFrame,
    sessions: pd.DataFrame,
    window: pd.Timedelta,
) -> pd.DataFrame:
    """For each label row, mean session duration over this user's sessions
    with ts in [label_ts - window, label_ts), i.e. strictly before."""
    # Both inputs must be sorted on the join key for merge_asof.
    labels = labels.sort_values("ts").reset_index(drop=True)
    sessions = sessions.sort_values("ts").reset_index(drop=True)

    out = []
    for user_id, g_labels in labels.groupby("user_id", sort=False):
        g_sessions = sessions[sessions["user_id"] == user_id].sort_values("ts")
        # For each label ts, find the latest session ts strictly before it;
        # we actually want a window AGGREGATE, so merge_asof is only half
        # the story — it finds the right slice, we compute the mean manually.
        values = []
        for ts in g_labels["ts"]:
            mask = (g_sessions["ts"] >= ts - window) & (g_sessions["ts"] < ts)
            values.append(g_sessions.loc[mask, "duration_s"].mean())
        out.append(g_labels.assign(mean_duration_30d=values))
    return pd.concat(out).sort_values("ts").reset_index(drop=True)

safe = asof_mean_duration(labels, sessions, window=pd.Timedelta("30D"))
print(safe.head(10))
print("\nNaN count (expected: first row per user has no prior sessions):",
      int(safe["mean_duration_30d"].isna().sum()))
```

**Expected output.** `mean_duration_30d` varies per row within a user
— the feature is now specific to "what was knowable at this label
timestamp". The first row for each user has `NaN` because no sessions
precede it; that's five NaNs, which is the right number.

**Checkpoint.** The NaN count is a correctness signal. On a real
dataset, zero NaNs after a point-in-time join is suspicious — it
usually means the join was not actually looking into the past. Expect
some NaNs; impute them inside a `Pipeline`, never fill them with the
global mean here (chapter 05's "no fitting in the cleaning function"
rule).

> **Note on `merge_asof` vs. the loop.** The loop above is the
> transparent version. The faster form uses
> `pd.merge_asof(labels_sorted, sessions_sorted, on="ts", by="user_id",
> direction="backward", allow_exact_matches=False, tolerance=window)`
> to get the *latest* prior session, and a groupby-rolling window
> on the sessions frame for the rolling aggregate. Try rewriting as
> a stretch exercise; the correctness check in step 5 should still
> pass.

### Step 5 — Measure again: back to ≈ 0.5 AUC (≈ 5 min)

Rerun the CV with the leak-free feature.

Add to `asof.py`:

```python
# Fill the first-row NaNs with a neutral value JUST for the CV (chapter 05
# would route this through a Pipeline; we're doing a quick measurement).
safe["mean_duration_30d"] = safe["mean_duration_30d"].fillna(safe["mean_duration_30d"].mean())
auc_safe = time_cv_score(safe, ["mean_duration_30d"])

print(f"leaky feature  AUC = {auc_leaky:.3f}")
print(f"safe  feature  AUC = {auc_safe:.3f}")
print(f"gap (CV optimism from the leak): {auc_leaky - auc_safe:+.3f}")
```

Run it.

**Expected output.** `safe AUC` is near `0.5` (the random-noise
baseline), while `leaky AUC` is substantially higher. The *gap*
between the two is the CV optimism purchased by the leak — the exact
number you would have seen collapse on deployment.

**Checkpoint.** The gap is the central claim of chapter 06, turned
into a measurement. The specific size will jitter run-to-run (small
dataset, 3-fold CV), but the direction is robust: a point-in-time
feature on noise scores at chance; a full-history feature on noise
scores above chance because it secretly encodes the per-user rate.

### Step 6 — Encode the correctness as a test (≈ 5 min)

Chapter 10 ("Validating splits and grouping") recommends encoding
leakage rules as assertions. Add a direct correctness check for your
`asof_mean_duration`.

Append to `asof.py`:

```python
def test_strictly_before():
    """The point-in-time feature must never include the label row itself."""
    toy_sessions = pd.DataFrame({
        "user_id":   ["A", "A", "A"],
        "ts":        pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
        "duration_s":[10.0, 20.0, 1000.0],
    })
    toy_labels = pd.DataFrame({
        "user_id": ["A"],
        "ts":      pd.to_datetime(["2026-01-03"], utc=True),
        "y":       [1],
    })
    out = asof_mean_duration(toy_labels, toy_sessions, window=pd.Timedelta("30D"))
    # The label row is on 2026-01-03. Prior sessions have durations 10 and 20.
    # If the feature leaked the label row's own session (duration=1000), the
    # mean would be (10 + 20 + 1000) / 3 = 343.3 instead of (10 + 20) / 2 = 15.
    expected = 15.0
    got = float(out["mean_duration_30d"].iloc[0])
    assert abs(got - expected) < 1e-9, f"leakage detected: got {got}, expected {expected}"
    print("OK: point-in-time correctness holds.")

test_strictly_before()
```

Run it.

**Expected output.** `OK: point-in-time correctness holds.` If you
accidentally remove the `< ts` strict inequality (e.g. change to
`<= ts`), the assertion fails with `got 343.333...` — exactly the
diagnostic chapter 06 recommends in its exercise-03-D wiring
("construct two rows with the same key and consecutive timestamps;
the later row's feature must not include the earlier row's target
value").

**Checkpoint.** The assertion is the smallest unit of defence
against the whole class of temporal leak. In a real project this
lives under `tests/test_temporal_feature.py`; in this lab it is one
function at the bottom of your script.

## Expected final artefacts

A single file `lab-02-point-in-time/asof.py` containing the six step
blocks above. Running `python asof.py` end-to-end should take under
ten seconds and print (in order) the sessions / labels preview, the
leaky-feature table, the two AUC numbers with the gap between them,
the `NaN count` for the safe feature, and the `OK:` line from the
assertion.

## Final checkpoint

Without re-reading chapter 06, answer:

1. In Step 2, why does `groupby.transform("mean")` constitute a leak
   for a temporal label? State the "as-of" rule in your own words.
2. In Step 5, what would it mean operationally if the `safe` AUC were
   **also** well above 0.5? Name two places the leak could still
   live.
3. In Step 6, what is the role of `allow_exact_matches=False` (or the
   `< ts` strict inequality in the manual version)? What is the
   bug if you use `≤ ts` instead?
4. If the same lab were on an unbiased *regression* target rather
   than a binary one, which metric would you use in Step 3's CV, and
   would the direction of the leak-induced optimism be the same?

If you can answer all four, the temporal-leakage mental model is in
place and you are ready for the longer end-to-end version in
exercise-03 Part D.

## Where to go next

- `exercise-03` ("Tabular feature engineering") Part D builds an
  entity-level windowed feature inside a `Pipeline`, with a unit
  test, on a real dataset. This lab's `test_strictly_before` is the
  five-line precursor to that test.
- Chapter 10 then automates this class of check — the point-in-time
  invariant becomes a `@pa.dataframe_check` that fires in CI.
