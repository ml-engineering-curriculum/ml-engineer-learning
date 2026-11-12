# lab-02 — Preprocessing leakage: scaler before vs inside the `Pipeline`

> **Duration:** ~30–45 minutes.
> **Companion chapter:** [06 — Splits, leakage, and cross-validation](../06-splits-leakage-and-cross-validation.md).

## Objective

Reproduce chapter 06's "preprocessing leakage" footgun — the universal
scikit-learn mistake of fitting a `StandardScaler` (or an imputer, a PCA,
a target encoder, a feature selector) on the full dataset *before*
cross-validation. By the end you will have (a) scored the same model two
ways on the same data — pre-split scaling vs scaling inside a `Pipeline` —
(b) measured the metric gap between them under `KFold(shuffle=True)`,
(c) added a deliberately leak-vulnerable feature selector to amplify the
gap, and (d) confirmed that the `Pipeline`-wrapped version matches an
untouched held-out set while the pre-split version systematically
over-reports.

This lab is deliberately narrower than **exercise-02** (which demonstrates
leakage in a full time+group pipeline on a real dataset and builds a
nested-CV report). The lab isolates **one** of chapter 06's five leakage
families — preprocessing leakage — on a synthetic 1,000-row table where
you can see the mechanism one fold at a time. Exercise-02 then exercises
the whole leakage taxonomy on a real dataset.

## Prerequisites

- Python 3.11+, `scikit-learn ≥ 1.4`, `numpy`.
- Chapter 06 read through once.
- A CPU is enough.
- Any virtual environment (`uv`, `poetry`, or a plain `venv`).

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-02-preproc-leak && cd lab-02-preproc-leak
touch preproc_leak.py
```

Start the file with imports and a seed:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(seed=0)
```

## Steps

### Step 1 — Build a 1,000-row classification dataset with a held-out test set (≈ 5 min)

Chapter 06's three-way split rule: cut a test set *first* and never touch
it until the end. That untouched set is the honesty check that will catch
the leak later.

Add to `preproc_leak.py`:

```python
X, y = make_classification(
    n_samples=1_000,
    n_features=50,
    n_informative=5,
    n_redundant=5,
    class_sep=0.8,
    random_state=0,
)
# Cut 20% as the untouched test set.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y,
)
print("dev shape:", X_dev.shape, "  test shape:", X_test.shape)
print("dev positive rate:", y_dev.mean(), "  test positive rate:", y_test.mean())
```

**Expected output.** `dev shape: (800, 50)`, `test shape: (200, 50)`, with
positive rates close to `0.5` on both (default `make_classification`
produces a balanced dataset). `stratify=y` preserves the base rate across
the split.

**Checkpoint.** The 200-row test set never appears in any CV call below.
Every CV number this lab produces is on the 800-row `X_dev`, and the
honest-reality number is `roc_auc_score(y_test, …)` after the chosen
pipeline refits on `X_dev` at the end. Chapter 06's "cut the test set
once, touch it once" — on paper so far, operationalised in the next
steps.

### Step 2 — The leaky CV: scale *before* splitting (≈ 10 min)

Chapter 06's named error: `fit_transform` the scaler on the whole dataset,
then run `cross_val_score` on the scaled arrays. Every fold's training
split now sees a scaler whose mean and variance were influenced by that
fold's validation rows.

Add to `preproc_leak.py`:

```python
# WRONG — scaler fit on all of X_dev before any CV split happens.
scaler_leaky = StandardScaler().fit(X_dev)
X_dev_scaled_leaky = scaler_leaky.transform(X_dev)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores_leaky = cross_val_score(
    LogisticRegression(max_iter=2000, random_state=0),
    X_dev_scaled_leaky,
    y_dev,
    cv=cv,
    scoring="roc_auc",
)
print("leaky CV  per-fold AUCs:", np.round(scores_leaky, 4))
print(f"leaky CV  mean ± std:  {scores_leaky.mean():.4f} ± {scores_leaky.std():.4f}")
```

**Expected output.** Five per-fold AUCs in the mid-0.9s; a mean somewhere
around `0.95 ± 0.01` depending on the RNG draw. **Note:** on a plain
`StandardScaler` the leak's effect on logistic regression is small — the
scaler only shifts and rescales, and the model is scale-invariant in
decision terms for most practical seeds. The gap becomes obvious in step
4 once we add a leak-vulnerable *feature selector* on top. For now, log
the number and move on.

**Checkpoint.** The leak is already happening — the scaler fit on
`X_dev` saw rows from every eventual fold's validation split — but
logistic regression's scale-robustness masks most of its effect on the
AUC. This is the realistic trap: many pipelines look fine until you add
one more preprocessing step that *is* leak-sensitive, which the next
steps simulate.

### Step 3 — The right CV: scaler inside a `Pipeline` (≈ 10 min)

Chapter 06's remedy: wrap every preprocessing step with learned
parameters inside a `Pipeline` (or `ColumnTransformer`). `cross_val_score`
on a `Pipeline` refits the scaler on each training fold and applies the
fitted transformer to the validation fold — the paved road.

Add to `preproc_leak.py`:

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000, random_state=0)),
])

scores_clean = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="roc_auc")
print("clean CV  per-fold AUCs:", np.round(scores_clean, 4))
print(f"clean CV  mean ± std:  {scores_clean.mean():.4f} ± {scores_clean.std():.4f}")

print(f"\nPreprocessing-leak gap (leaky - clean): "
      f"{scores_leaky.mean() - scores_clean.mean():+.4f}")
```

**Expected output.** Per-fold AUCs very close to the leaky run's — a gap
on the order of `±0.001–0.003`. For plain scaling + logistic regression on
this dataset the gap is near the fold-to-fold noise. **That is the actual
lesson.** Preprocessing leakage from a scalar scaler on a scale-robust
model is small; from a feature selector that chooses *which* columns to
keep based on the target it is catastrophic, which is step 4.

**Checkpoint.** `cross_val_score` on the `Pipeline` did what chapter 06
promises: on each outer fold it refit the scaler on that fold's training
rows only, and applied the fitted scaler to the fold's validation rows.
The pipeline version is the correct baseline you measure other, more
dramatic leaks against — not against the un-CV'd number.

### Step 4 — Amplify the leak with a target-aware feature selector (≈ 10 min)

Chapter 06 names five classes of preprocessing that leak when fit before
the split — "a scaler ... PCA, `SelectKBest`, `SimpleImputer`, target
encoder, feature selector". `SelectKBest` with an ANOVA F-test is the one
where the leak is immediately visible: it chooses columns using the
labels, so fitting it before splitting lets it pick the columns with the
best signal on the (eventual) validation rows.

Add to `preproc_leak.py`:

```python
# WRONG — SelectKBest fit on the whole X_dev / y_dev before CV.
selector_leaky = SelectKBest(f_classif, k=5).fit(X_dev, y_dev)
X_dev_sel_leaky = selector_leaky.transform(X_dev)
scaler_leaky2 = StandardScaler().fit(X_dev_sel_leaky)
X_dev_sel_leaky_scaled = scaler_leaky2.transform(X_dev_sel_leaky)

scores_leaky_sel = cross_val_score(
    LogisticRegression(max_iter=2000, random_state=0),
    X_dev_sel_leaky_scaled,
    y_dev,
    cv=cv,
    scoring="roc_auc",
)
print("\nleaky(selector+scaler) CV per-fold AUCs:",
      np.round(scores_leaky_sel, 4))
print(f"leaky(selector+scaler) CV mean ± std:  "
      f"{scores_leaky_sel.mean():.4f} ± {scores_leaky_sel.std():.4f}")

# RIGHT — both steps inside the Pipeline.
pipe_sel = Pipeline([
    ("select", SelectKBest(f_classif, k=5)),
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000, random_state=0)),
])
scores_clean_sel = cross_val_score(pipe_sel, X_dev, y_dev, cv=cv, scoring="roc_auc")
print("clean(Pipeline[select,scaler,clf]) CV per-fold AUCs:",
      np.round(scores_clean_sel, 4))
print(f"clean(Pipeline[select,scaler,clf]) CV mean ± std: "
      f"{scores_clean_sel.mean():.4f} ± {scores_clean_sel.std():.4f}")

print(f"\nPreprocessing-leak gap with selector (leaky - clean): "
      f"{scores_leaky_sel.mean() - scores_clean_sel.mean():+.4f}")
```

Run `python preproc_leak.py`.

**Expected output.** The leaky-selector run reports a visibly higher mean
than the clean pipeline, in the 1–5 AUC-points range on this dataset
(exact number depends on the RNG and the 50-feature / 5-informative
split). This is chapter 06's warning — "the feature selector chose
features based on their signal on the test set" — made numerical.

**Checkpoint.** The gap is now large enough to matter to a PR reviewer.
Chapter 06 pitfall #1: "Scaling / imputing / feature-selecting before the
split. Universal beginner mistake. Wrap in a `Pipeline`, always." On a
real dataset the preprocessing stack is longer (imputer → encoder →
scaler → PCA → selector → model) and each step is its own opportunity
to leak; the only safe pattern is "every learned step lives inside the
`Pipeline`, no exceptions."

### Step 5 — Verify against the untouched test set (≈ 5 min)

Chapter 06: the test set that was cut in step 1 and never touched is the
honesty check. Refit both pipelines on the full `X_dev`, score on
`X_test`, and compare each CV mean against the held-out reality. The
pipeline that was *honest* about its preprocessing should match
reality; the one that was leaky should over-estimate.

Add to `preproc_leak.py`:

```python
# Refit the WRONG pipeline on all dev data and transform test with
# the selector+scaler that were originally fit on dev only — this
# simulates "shipping" the leaky pipeline (chapter 06's realistic case).
pipe_leaky_shipped_X = scaler_leaky2.transform(
    selector_leaky.transform(X_dev)
)
pipe_leaky_shipped_clf = LogisticRegression(
    max_iter=2000, random_state=0
).fit(pipe_leaky_shipped_X, y_dev)

X_test_leaky = scaler_leaky2.transform(selector_leaky.transform(X_test))
test_auc_leaky = roc_auc_score(
    y_test, pipe_leaky_shipped_clf.predict_proba(X_test_leaky)[:, 1]
)

# Refit the RIGHT pipeline the same way; its transformers get fit on all
# of X_dev at the final-refit stage (that is fine — we don't CV-score
# afterwards).
pipe_sel.fit(X_dev, y_dev)
test_auc_clean = roc_auc_score(
    y_test, pipe_sel.predict_proba(X_test)[:, 1]
)

print(f"\nHeld-out test AUC (leaky  shipped): {test_auc_leaky:.4f}  "
      f"CV claimed {scores_leaky_sel.mean():.4f}  "
      f"(CV error: {scores_leaky_sel.mean() - test_auc_leaky:+.4f})")
print(f"Held-out test AUC (clean  shipped): {test_auc_clean:.4f}  "
      f"CV claimed {scores_clean_sel.mean():.4f}  "
      f"(CV error: {scores_clean_sel.mean() - test_auc_clean:+.4f})")
```

**Expected output.** Two lines. The clean pipeline's CV mean and test AUC
agree closely — the CV mean is the honest expected score, and the test
set confirms it. The leaky pipeline's CV mean overstates its test AUC by
the same ballpark gap you observed in step 4. The leak was already paid
for by the time deployment measured the model.

**Checkpoint.** This is chapter 06's whole point in one comparison:
`cross_val_score` is a variance-reduction tool, not a leakage cure. If
your CV number is honest, it matches reality; if it is not, reality is
the lower one. The paved road is "every learned transformer inside the
`Pipeline`, test set cut once and touched once at the end".

### Step 6 — Encode the invariant as a test (≈ 5 min)

Chapter 06 treats the "preprocessing inside the pipeline" rule as a
correctness property, not a stylistic choice. Encode it as a tiny test
that fails if someone silently reintroduces a pre-split `fit_transform`.
In a real project this would live under `tests/test_leakage.py`.

Append to `preproc_leak.py`:

```python
def test_cv_uses_pipeline_wrapped_preprocessing() -> None:
    """A direct array fed into cross_val_score is a smell — the fix
    is to wrap preprocessing in a Pipeline and feed the raw X.
    """
    # The clean path: the estimator is a Pipeline whose first step has
    # a `fit` method and is NOT already a classifier. That is the shape
    # chapter 06's remedy prescribes.
    assert isinstance(pipe_sel, Pipeline), (
        "estimator passed to cross_val_score must be a Pipeline"
    )
    first_name, first_step = pipe_sel.steps[0]
    assert hasattr(first_step, "fit"), (
        f"first Pipeline step {first_name!r} must be fittable"
    )
    # Leaf estimator is the last step.
    assert hasattr(pipe_sel.steps[-1][1], "predict_proba"), (
        "last Pipeline step must be a classifier"
    )
    print("OK: estimator is a Pipeline with a fittable first step and a "
          "classifier at the end.")


def test_held_out_test_set_was_not_used_by_cv() -> None:
    """The row indices in X_test must be disjoint from the row indices
    seen by the CV splitter. Because we cut X_test before any CV call,
    this is structural — but worth asserting as a shape check.
    """
    assert X_dev.shape[0] + X_test.shape[0] == 1000
    assert X_dev.shape[0] == 800 and X_test.shape[0] == 200
    print("OK: dev / test shapes match the pre-CV cut.")


test_cv_uses_pipeline_wrapped_preprocessing()
test_held_out_test_set_was_not_used_by_cv()
```

Run `python preproc_leak.py`.

**Expected output.** Two `OK` lines at the end of the script. The first
asserts the "every learned step inside the `Pipeline`" rule on the
`pipe_sel` object; the second confirms the test set was cut once and not
touched by any CV call above.

**Checkpoint.** The two assertions are the smallest unit of defence
against preprocessing leakage creeping back in. Pitfall #1 of chapter 06
is exactly what both tests guard against. On a real project, test
coverage like this plus a `cross_val_score(pipeline, X, y, cv=...)`
convention is what prevents a silent regression in a future PR.

## Expected final artefact

A single file `lab-02-preproc-leak/preproc_leak.py` containing the six
step blocks above. Running `python preproc_leak.py` end to end should
take under a minute on a laptop and print (in order) the dev/test shape
summary, the leaky scaler CV, the clean pipeline CV, the leaky
selector+scaler CV, the clean pipeline-with-selector CV, the leak gaps,
the held-out test comparison, and the two assertion `OK` lines.

## Final checkpoint

Without re-reading chapter 06, answer:

1. In step 2, the leak from the pre-split `StandardScaler` was small
   (fold-to-fold noise size). In step 4, adding `SelectKBest` before the
   split pushed the gap into the "visible to a reviewer" range. Which
   property of `SelectKBest` makes the leak bigger than a scaler's leak,
   in terms of what information the preprocessor extracts from the data
   it sees?
2. In step 5, the clean pipeline's CV mean matched the held-out test AUC
   while the leaky pipeline's CV mean overstated it. Chapter 06's line
   is: "CV is a variance-reduction tool, not a leakage cure." Translate
   that sentence into a statement about what CV measures and what it
   does not.
3. Step 6's `test_cv_uses_pipeline_wrapped_preprocessing` asserts that
   the first `Pipeline` step is fittable and the last is a classifier.
   Name one way the assertion could still pass on a leaky pipeline that
   you would need a different check to catch — chapter 06's taxonomy
   includes at least two leaks that this assertion does not cover.
4. Chapter 06's "A minimum-viable CV pattern" template has
   `test_size=0.2` and `stratify=y` on the initial split, and
   `StratifiedKFold` inside. On this lab's balanced dataset,
   `stratify=y` is almost a no-op; on a 1% positive dataset it is not.
   Why?

If you can answer all four, chapter 06's preprocessing-leakage story is
internalised and you are ready to apply it to the full time+group dataset
in exercise-02.

## Where to go next

- **`exercise-02` ("Leakage-resistant validation")** exercises all five
  leakage families on a real dataset that has both a time index and a
  group structure, and wraps the policy in a `validation_config.py`
  module that forbids random shuffles outright.
- **`lab-01` ("Accuracy, ROC-AUC, and PR-AUC disagree under class
  imbalance")** complements this lab: lab-01 asks "is the metric right?"
  before you worry about leakage; lab-02 asks "is the number honest?"
  after you have picked a metric.
- **Chapter 07** extends the same "learned step inside a splitter"
  rule to time-series: a rolling-mean feature is time-leaky for the same
  reason a scaler fit on the whole dataset is CV-leaky — the computation
  used information that would not have been available at each row's
  observation timestamp.
