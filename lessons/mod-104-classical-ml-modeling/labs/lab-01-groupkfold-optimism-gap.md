# lab-01 — Measuring the `KFold` vs `GroupKFold` optimism gap

> **Duration:** ~30–45 minutes.
> **Companion chapter:** [07 — Cross-validation strategies](../07-cross-validation-strategies.md).

## Objective

Reproduce chapter 07's central claim — "a random split can inflate an
accuracy metric by tens of percentage points relative to the group-aware
split" — on a hand-built synthetic dataset small enough to eyeball. By
the end you will have (a) built a grouped dataset where each entity has
a distinctive per-entity baseline, (b) watched `KFold(shuffle=True)`
inflate the CV AUC by memorising those baselines across train and
validation folds, (c) replaced it with `GroupKFold` and seen the score
drop to the honest number, and (d) confirmed that the `GroupKFold`
estimate matches what a truly held-out set of *new* entities actually
produces.

This lab is pure scikit-learn on a synthetic 500-row table. It is
deliberately shorter and more focused than exercise-02 (which runs the
full tree-family comparison on the Adult dataset) and exercise-03
(which tunes with Optuna). The lab is the isolation chamber where you
can *see* the group leak happen one fold at a time; the exercises
build the same discipline into a real pipeline.

## Prerequisites

- Python 3.11+, scikit-learn ≥ 1.4, Pandas ≥ 2.1, NumPy ≥ 1.26.
- Chapter 07 read through once.
- Any virtual environment (`uv`, `poetry`, or a plain venv).

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-01-group-cv && cd lab-01-group-cv
touch group_cv.py
```

Everything below goes in `group_cv.py`. Run with `python group_cv.py`
after each step.

Start the file with imports:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
```

## Steps

### Step 1 — Build a grouped synthetic dataset (≈ 5 min)

Chapter 07's grouped-leakage scenario: many rows per entity, inference
sees *new* entities, so each user's baseline should not be available at
scoring time. Make this concrete.

Build 50 users × 10 rows each. Each user has a distinctive per-user
`baseline` drawn once; it is strongly predictive of the label on
rows from users the model has seen before, and completely useless on
rows from new users (because there is nothing else in the features that
correlates with the label).

Add to `group_cv.py`:

```python
RNG = np.random.default_rng(seed=0)

def build_dataset(n_users: int = 50, rows_per_user: int = 10) -> pd.DataFrame:
    """Each user has a per-user baseline that drives the label.

    Features:
      - x1, x2, x3: pure noise, no relationship to y.
      - baseline:  known only through the user_id. We DO NOT put this
                   in the feature matrix — the model has to recover it
                   (if it can) from user_id on training users.
    Label: a noisy threshold on the per-user baseline.
    """
    user_ids  = np.repeat(np.arange(n_users), rows_per_user)
    baselines = RNG.uniform(-1.0, 1.0, size=n_users)
    per_row   = baselines[user_ids] + RNG.normal(0.0, 0.1, size=user_ids.size)

    df = pd.DataFrame({
        "user_id": user_ids,
        "x1":      RNG.normal(size=user_ids.size),
        "x2":      RNG.normal(size=user_ids.size),
        "x3":      RNG.normal(size=user_ids.size),
        "y":       (per_row > 0).astype(int),
    })
    return df

df = build_dataset()
print(df.head(8))
print("\nrows:", len(df), "users:", df["user_id"].nunique())
print("label rate:", df["y"].mean().round(3))
print("rows per user (first 5 users):\n",
      df["user_id"].value_counts().sort_index().head(5))
```

**Expected output.** 500 rows, 50 users, 10 rows each, label rate near
`0.5`. The three `x` features are random noise. Only `user_id` carries
the true signal, through the per-user baseline you drew once.

**Checkpoint.** `user_id` is in the frame on purpose. The model will
be given `[user_id, x1, x2, x3]`; a gradient booster will happily split
on `user_id` and memorise each training user's label rate. That is
exactly the per-entity memorisation chapter 07 warns about.

### Step 2 — The naive CV: `KFold(shuffle=True)` (≈ 10 min)

Chapter 07's default-but-wrong move on grouped data: shuffle the rows
and do a standard k-fold.

Add to `group_cv.py`:

```python
FEATURES = ["user_id", "x1", "x2", "x3"]

def fit_and_score(cv) -> np.ndarray:
    """Return per-fold ROC-AUC scores under the given CV splitter."""
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=31,
        learning_rate=0.1,
        random_state=0,
    )
    return cross_val_score(
        model,
        df[FEATURES].to_numpy(),
        df["y"].to_numpy(),
        cv=cv,
        scoring="roc_auc",
        groups=df["user_id"].to_numpy(),  # ignored by KFold; needed for GroupKFold
    )

scores_kfold = fit_and_score(KFold(n_splits=5, shuffle=True, random_state=0))
print(f"KFold(shuffle=True)   per-fold AUCs: {np.round(scores_kfold, 3)}")
print(f"KFold mean ± std:      {scores_kfold.mean():.3f} ± {scores_kfold.std():.3f}")
```

Run it.

**Expected output.** Per-fold AUCs in the 0.90–1.00 range; the mean is
somewhere around `0.95` or higher. The value jitters a little run-to-run
because of the booster's internal randomness, but it will be well above
`0.5`.

**Checkpoint.** `y` depends only on the per-user baseline (which is not
in the features) plus noise. There is *no* signal in `x1`, `x2`, `x3`,
so a model that cannot see `user_id` would score at chance. The AUC of
~0.95 is the model recovering each training user's baseline through the
`user_id` feature and then being asked to predict for the *same* users
on held-out rows. That is the overstated metric chapter 07 flags.

### Step 3 — The right CV: `GroupKFold` (≈ 10 min)

Chapter 07's rule: if inference will see new entities, pass
`groups=X["user_id"]` and use a group-aware splitter so no user appears
in both train and validation of the same fold.

Add to `group_cv.py`:

```python
scores_group = fit_and_score(GroupKFold(n_splits=5))
print(f"GroupKFold(n_splits=5) per-fold AUCs: {np.round(scores_group, 3)}")
print(f"GroupKFold mean ± std:  {scores_group.mean():.3f} ± {scores_group.std():.3f}")

print(
    f"\nOptimism gap (KFold – GroupKFold): "
    f"{scores_kfold.mean() - scores_group.mean():+.3f}"
)
```

Run it.

**Expected output.** Per-fold AUCs in `GroupKFold` close to `0.5`
(chance); the mean is also near `0.5`. The optimism gap — the
difference between the two CV estimates — is comfortably in the "tens
of percentage points" range chapter 07 quantifies.

**Checkpoint.** With `GroupKFold`, every validation row belongs to a
user that is nowhere in that fold's training set. The model cannot
memorise those users' baselines — it has never seen them — and since
there is no other signal in the features, it predicts at chance. Which
is the honest answer.

### Step 4 — Confirm `GroupKFold` matches the deployed reality (≈ 10 min)

Chapter 07 argues that the right CV should match the deployed model's
behaviour. Prove it: hold out a chunk of users the CV never touches,
then compare against what each CV strategy *claimed* the score would be.

Add to `group_cv.py`:

```python
# True held-out set: 10 users the model will not see at all during CV.
held_out_users = np.arange(40, 50)
train_mask = ~df["user_id"].isin(held_out_users)

df_train = df.loc[train_mask].reset_index(drop=True)
df_hold  = df.loc[~train_mask].reset_index(drop=True)

# Refit on the 40-user training set with the same model.
model = HistGradientBoostingClassifier(
    max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=0,
).fit(df_train[FEATURES].to_numpy(), df_train["y"].to_numpy())

# Score on the 10 new users — the actual deployed-reality number.
proba_hold = model.predict_proba(df_hold[FEATURES].to_numpy())[:, 1]
auc_hold   = roc_auc_score(df_hold["y"].to_numpy(), proba_hold)

print(f"Honest held-out AUC on 10 new users: {auc_hold:.3f}")
print(f"  KFold CV claimed:      {scores_kfold.mean():.3f}  "
      f"(error vs held-out: {scores_kfold.mean() - auc_hold:+.3f})")
print(f"  GroupKFold CV claimed: {scores_group.mean():.3f}  "
      f"(error vs held-out: {scores_group.mean() - auc_hold:+.3f})")
```

Run it.

**Expected output.** The held-out AUC is close to `0.5`. `GroupKFold`'s
mean matches it within a few percentage points. `KFold`'s mean is
dramatically higher than the held-out reality — the exact optimism the
chapter warns about.

**Checkpoint.** The held-out AUC is the gold standard: the model is
scored on users it has truly never seen. `GroupKFold` tracks it;
`KFold` does not. On a real project the held-out number is what the
business will see after deployment, and the CV number is the one you
report in the PR. Ship a `KFold` number and you have committed to a
metric the deployment will not reproduce — usually followed by an
awkward post-mortem.

### Step 5 — Encode the invariant as a test (≈ 5 min)

Chapter 07 treats the group split as a correctness property, not a
stylistic choice. Encode it as an assertion that fails if the splitter
ever puts the same user in both train and validation.

Append to `group_cv.py`:

```python
def test_groupkfold_isolates_groups() -> None:
    """No group may appear in both train and validation of any fold."""
    gkf = GroupKFold(n_splits=5)
    groups = df["user_id"].to_numpy()
    for i, (tr, va) in enumerate(gkf.split(df, df["y"], groups=groups)):
        tr_users = set(groups[tr])
        va_users = set(groups[va])
        overlap  = tr_users & va_users
        assert not overlap, (
            f"fold {i}: {len(overlap)} users appear in both train and val: "
            f"{sorted(overlap)[:5]}..."
        )
    print("OK: GroupKFold isolates users across every fold.")

def test_kfold_leaks_groups() -> None:
    """Show the failure we are protecting against: plain KFold leaks groups."""
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    groups = df["user_id"].to_numpy()
    leaked = 0
    for tr, va in kf.split(df, df["y"]):
        leaked += len(set(groups[tr]) & set(groups[va]))
    assert leaked > 0, "expected KFold to leak groups on this dataset"
    print(f"OK (expected): KFold leaked groups across {leaked} fold-group pairs.")

test_groupkfold_isolates_groups()
test_kfold_leaks_groups()
```

Run it.

**Expected output.** Two `OK` lines. The first says `GroupKFold`
guarantees isolation; the second confirms that `KFold` on this dataset
leaks groups (as expected — that is the whole point of the lab).

**Checkpoint.** The two tests are the smallest unit of defence against
re-introducing the leak. In a real project this lives under
`tests/test_cv_split.py` and runs in CI any time the splitter changes.
A PR that silently swaps `GroupKFold` back to `KFold` will trip the
first assertion the next time the test suite runs.

## Expected final artefacts

A single file `lab-01-group-cv/group_cv.py` containing the five step
blocks above. Running `python group_cv.py` end-to-end should take
under a minute on a laptop and print (in order) the dataset preview,
the two per-fold AUC arrays with their means ± stds, the optimism gap,
the three comparison lines against the held-out AUC, and the two `OK`
lines from the assertions.

## Final checkpoint

Without re-reading chapter 07, answer:

1. In Step 2, the `KFold` AUC is near `0.95` on data where the only
   signal lives in a per-user baseline and the three `x` features are
   pure noise. State in your own words *what information* the model
   is using to predict so well — and why that information is not
   available at inference time on a new user.
2. In Step 3, the `GroupKFold` AUC drops to ~`0.5`. Is that a failure
   of the model, a failure of the CV, or evidence that the task is
   impossible without more features? Chapter 07's answer is one of
   these — which, and why?
3. Suppose rows were ordered by time within each user, and inference
   at serving time will score a *future* row for a *known* user. Is
   `GroupKFold` the right splitter for that problem? Which chapter-07
   splitter would you reach for instead, and why?
4. In Step 5, the second assertion (`test_kfold_leaks_groups`) is
   intentionally proving the failure mode rather than the fix. Why is
   it worth keeping that test in CI, as opposed to deleting it once
   you have switched to `GroupKFold`?

If you can answer all four, the group-leakage mental model is in place
and you are ready to apply it on real datasets in exercise-02 and
exercise-03.

## Where to go next

- `exercise-02` ("Trees and ensembles") uses `StratifiedKFold` on the
  Adult dataset, which does *not* have groups. Chapter 07's decision
  tree is now something you can run yourself: if your next project has
  multiple rows per entity, you know the splitter to reach for.
- `exercise-03` ("XGBoost or LightGBM tuning") uses `cross_val_score`
  inside an Optuna objective. The group-aware CV from this lab plugs
  in directly — pass `groups=` to `cross_val_score` and swap the CV
  class, and every tuning trial respects the group boundary.
- Chapter 10 ("`pandera`-style schema checks" in `mod-103`'s
  neighbourhood) suggests encoding invariants like Step 5's assertion
  as pre-flight checks that run before any model is fit. The pattern
  generalises: the shape of your CV is a property worth asserting.
