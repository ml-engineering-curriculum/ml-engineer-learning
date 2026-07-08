# 08 — Testing ML code: unit, golden, lint, type-check

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 05, 06, 07.

## Motivation

Software engineers test code so refactors don't break it. ML
engineers test code so refactors don't *silently* break it —
because "still trains, but on different data" or "still predicts,
but with the wrong scaling" is a worse failure mode than a stack
trace. Tests are how you stay confident that yesterday's pipeline
is doing what it did yesterday.

ML code is testable, despite its reputation. Most of it is
ordinary Python with `numpy` and `pandas` calls; the parts that
*aren't* (the stochastic, slow, or data-dependent parts) need a
slightly different testing toolkit. This chapter covers four
layers:

1. **Unit tests** for deterministic code (transformers, feature
   functions, metric implementations).
2. **Property-based tests** for invariants you cannot fully
   enumerate.
3. **Golden tests** ("snapshot tests") for small models and end-to-
   end pipelines, where the bar is "behaviour did not change" not
   "I can name the right number a priori".
4. **Lint and type-check** as your continuous, cheap, automatic
   review.

## `pytest`: the baseline

[`pytest`](https://docs.pytest.org/) is the default Python test
runner. The interface is small: any function whose name starts with
`test_` in any file that starts with `test_` is a test.

```python
# tests/test_features.py
import numpy as np

from my_ml_project.features import standardise

def test_standardise_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(1000, 4))
    Y = standardise(X)
    assert np.allclose(Y.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(Y.std(axis=0),  1, atol=1e-7)
```

Two patterns to internalise:

- **`np.allclose` and `pd.testing.assert_frame_equal`**, not `==`.
  Floating-point equality almost always loses.
- **Fixtures via `@pytest.fixture`** for shared setup. Build a
  small synthetic dataset once; many tests share it.

```python
import pytest
import pandas as pd

@pytest.fixture
def toy_df():
    return pd.DataFrame({
        "age":     [22, 31, 45, None],
        "country": ["US", "UK", "US", "FR"],
        "income":  [40_000, 80_000, 120_000, 95_000],
    })

def test_pipeline_handles_missing(toy_df):
    pipe = build_preprocessing_pipeline()
    out = pipe.fit_transform(toy_df)
    assert not np.isnan(out).any()
```

## Unit-testing a custom transformer

You wrote a `WinsoriseTransformer` in chapter 05. Now you test it.

The non-obvious thing about transformers is that they have *two*
phases of behaviour — `fit` and `transform` — and you must test
both. A useful checklist:

1. **`fit_transform` is well-defined on a known input.** Hand-compute
   the expected output on a tiny example and assert.
2. **`fit` is idempotent on the same data.** `fit(X).fit(X)` should
   produce the same learned state.
3. **`fit(X).transform(X)` is different from `transform(X)`
   without fit.** The pre-fit transformer should raise (e.g.
   `NotFittedError`).
4. **The transformer respects the train/test split.** Fit on a
   training subset; transform a held-out subset. The held-out
   transformation must not use any held-out statistics.
5. **The transformer round-trips through `clone`.** `clone(tx)`
   produces an unfitted copy with the same hyperparameters; the
   clone should not share fitted state.

A worked example:

```python
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from my_ml_project.features import WinsoriseTransformer

def test_winsorise_clips_to_training_quantiles():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(1000, 2))
    X_test  = np.array([[-100.0, 100.0]])    # extreme values

    tx = WinsoriseTransformer(lower=0.01, upper=0.99).fit(X_train)
    Y = tx.transform(X_test)

    assert (Y >= tx.lower_bounds_).all()
    assert (Y <= tx.upper_bounds_).all()

def test_winsorise_raises_before_fit():
    tx = WinsoriseTransformer()
    with pytest.raises((AttributeError, NotFittedError)):
        tx.transform(np.zeros((1, 2)))

def test_winsorise_clone_drops_fitted_state():
    tx = WinsoriseTransformer().fit(np.random.normal(size=(10, 2)))
    tx2 = clone(tx)
    assert not hasattr(tx2, "lower_bounds_")
```

The same shape of test applies to any custom transformer. Most of
the bugs you'll find live in step 4 — "does the transformer respect
the train/test split". If it doesn't, your pipeline leaks at
predict time.

There is also
[`sklearn.utils.estimator_checks.check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html),
which runs scikit-learn's full conformance suite on your class. It
is heavyweight and a little opinionated — but for a transformer you
intend to publish or share across teams, running it once catches
many API-violation bugs.

## Property-based tests with Hypothesis

[Hypothesis](https://hypothesis.readthedocs.io/) generates inputs
that satisfy a property and looks for one that breaks it. The
mental model: instead of testing *one* input, you state an
*invariant* that should hold for *all* inputs in some space, and
Hypothesis searches.

```python
from hypothesis import given, strategies as st
import numpy as np

from my_ml_project.features import standardise

@given(st.integers(min_value=5, max_value=200),
       st.integers(min_value=1, max_value=20))
def test_standardise_output_shape_matches_input(n, d):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    assert standardise(X).shape == X.shape
```

Hypothesis is overkill for trivial code. It earns its keep for
transformations with subtle shape or numerical invariants —
softmax (rows sum to 1), normalisation (output norm = 1), encoder
round-trips (`decode(encode(x)) == x`).

## Golden / snapshot tests for models and pipelines

A **golden test** asserts that today's output matches a stored
"golden" output. For ML pipelines that means:

1. Train a small model on a fixed seed and a tiny dataset.
2. Predict on a fixed held-out set.
3. Compare the predictions to a stored array (the "golden") within
   tolerance.

Why this matters: many "refactor" bugs in ML pipelines do not break
the *interface* — they break the *numerics*. The pipeline still
returns an array of probabilities, but every probability is now
slightly different because the imputer's median changed, or the
encoder's category order changed, or the scaler's mean is now
computed across the whole input instead of per-column. Unit tests
miss these. Golden tests catch them.

```python
import numpy as np
import pytest

from my_ml_project.train import build_pipeline, load_toy_dataset

GOLDEN_PROBS = np.load("tests/data/baseline_probs.npy")

def test_baseline_pipeline_predictions_unchanged():
    X_train, y_train, X_test = load_toy_dataset(seed=0)
    pipe = build_pipeline(random_state=0).fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    np.testing.assert_allclose(probs, GOLDEN_PROBS, rtol=1e-6, atol=1e-8)
```

Workflow notes:

- The **golden file is data**, not code. Commit it next to the test
  (`tests/data/baseline_probs.npy`). Regenerating it is a deliberate
  act, not an accident.
- When the golden legitimately changes — you intentionally tuned a
  preprocessing step — regenerate the file in the same PR. The diff
  on the data file is small (file size moves) but the *commit
  message* is where you explain what changed and why.
- Keep the dataset *tiny*: 200 rows, 10 columns. Golden tests must
  run in seconds, otherwise they will be skipped.
- Pin every source of randomness — `random_state`, `np.random.default_rng(seed=0)`,
  and any framework-level seed (`torch.manual_seed`, `tf.random.set_seed`).
- For floating-point output, use `np.testing.assert_allclose` with
  an explicit `rtol`/`atol`. Equality is the wrong tool; CPU
  architectures differ in their last bit.

The pattern generalises beyond predictions: hash the fitted
transformer's learned attributes, compare a few rows of the
`ColumnTransformer` output, snapshot a confusion matrix on a held-
out set. The principle is the same — declare what should not
change without your noticing.

## Lint with `ruff`

[`ruff`](https://docs.astral.sh/ruff/) is a fast Python linter (and
auto-formatter) that replaces `flake8`, `isort`, `black`, and a
fistful of other tools with one binary. As of 2026 it is the
field default for new Python projects.

A minimal config in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
# E,F: pycodestyle + pyflakes (the basics)
# I: import sorting
# B: bugbear (likely-bug patterns)
# UP: pyupgrade (modern syntax)
# SIM: simplification suggestions
# RUF: ruff's own rules
```

Run it locally and in CI:

```bash
ruff check .
ruff format .       # autoformat (replaces black)
```

For pre-commit hooks, the
[`pre-commit`](https://pre-commit.com/) framework integrates `ruff`
in one config block. Wire it once and every commit gets a linter
pass for free.

## Type-check with `mypy` or `pyright`

Type hints are not a religion, but the cost is low and the bug-
detection value in ML code is high — especially around shapes,
dtypes, and the `numpy`/`pandas` API.

```python
import numpy as np
import numpy.typing as npt
import pandas as pd

def standardise(X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return (X - X.mean(axis=0)) / X.std(axis=0)

def build_features(df: pd.DataFrame, *, today: pd.Timestamp) -> pd.DataFrame:
    ...
```

Two type-checkers dominate:

- **[`mypy`](https://mypy.readthedocs.io/)** — the original, mature,
  strict-by-default.
- **[`pyright`](https://microsoft.github.io/pyright/)** — Microsoft's,
  faster, the engine behind VS Code's Pylance.

Pick one per project and configure it in `pyproject.toml`. Run it in
CI. Treat type errors with the same severity you treat lint errors.

For NumPy specifically, the modern API surface is
`numpy.typing.NDArray[<dtype>]`. Shape-aware typing (Python 3.12's
`Array[float64, Literal[3]]`-style) is still settling at the
ecosystem level; do not block on it.

## Putting it together: a recommended CI pipeline for an ML repo

A CI run that pays for itself every week:

```yaml
# .github/workflows/ci.yml — schematic; adapt to your provider.
jobs:
  test:
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3       # or set up poetry
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run mypy src tests
      - run: uv run pytest -q
```

In order: install the *exact* environment, lint, type-check, run
tests. The order matters — a lint failure is faster to fix than a
test failure; surface the fast checks first.

## Pitfalls collected

1. **Tests that mutate global state.** A seeded `np.random.default_rng`
   is fine; `np.random.seed(0)` followed by `np.random.randn(...)`
   leaks state to other tests. Use the generator.
2. **Tests that depend on real data.** A test that fails when an S3
   bucket is empty is not a unit test. Use synthetic fixtures.
3. **Tests that are too slow to run.** A 30-second test suite gets
   skipped; a 3-second one gets run. Keep golden datasets tiny.
4. **Floating-point comparison with `==`.** Always allclose, always
   with tolerances you have thought about.
5. **Skipping the type-check.** "We're moving fast" is not a reason
   to skip a 5-second `mypy` run. It's a reason to *want* the type-
   check.
6. **A failing golden test that just gets regenerated.** The point of
   a golden test is to *notice* an unintended change. Regenerating
   without explaining what changed defeats the purpose. Write a
   one-line "Why the golden moved" entry in the PR.

## Summary

- ML code is testable, and the cost of an unhappy refactor is
  higher than in ordinary CRUD code, which means testing pays back
  faster.
- Unit-test transformers along five axes — `fit` correctness,
  idempotency, pre-fit raises, train/test discipline, clone safety.
- Use property-based tests for invariants you cannot enumerate;
  golden tests for end-to-end pipelines whose output you want to
  freeze.
- `ruff` and `mypy`/`pyright` are cheap, fast, automatic
  reviewers — enable both in CI.
- Pin everything stochastic in tests; use tolerances, not equality,
  on floats.

## Where this goes next

That closes the module. Up next: `mod-103` puts these tools onto
real data — ingestion, audits, schema validation, and feature
engineering at the scale a working ML project actually faces.
`mod-104` then plugs the resulting feature matrices into the
classical model families (linear, tree, ensemble). Every chapter in
the rest of the curriculum quietly assumes you can write a
reproducible, tested scikit-learn pipeline.
