# exercise-04 — Testing ML code

> **Estimated effort:** 3 hours.
> **Companion chapter:** [08 — Testing ML code: unit, golden, lint, type-check](../08-testing-ml-code.md).

## Objective

Turn the pipeline you built in exercise-02 (or an equivalent one)
into code you would be willing to ship: unit-tested transformers,
a property-tested invariant, a golden test that pins end-to-end
predictions, `ruff` and `mypy` (or `pyright`) both green, and a CI
job that runs the whole thing on every push.

By the end you should feel the difference between "the model
worked once" and "the model will keep working" — and know which
tests give you which of those guarantees.

## Prerequisites

- The `ml_toolkit` project skeleton from
  [exercise-03](exercise-03-reproducible-env-setup.md), *or* the
  pipeline project from
  [exercise-02](exercise-02-sklearn-pipeline-design.md). If you do
  not have either, start a fresh `uv init --lib`'d project and
  drop in a `RareCategoryGrouper` + preprocessing pipeline as your
  system-under-test.
- Add these dev dependencies:
  `pytest`, `pytest-cov`, `hypothesis`, `ruff`, and one of `mypy` /
  `pyright`.
- Chapters 05, 06, and 08 of this module.

## Problem statement

You will layer four flavours of tests on top of the pipeline, in
order of increasing scope: unit → property → golden → end-to-end
CI.

### Part A — unit-test a custom transformer (≈ 45 min)

Pick a custom transformer to test. Either:

- **`WinsoriseTransformer`** from chapter 05 (numeric,
  quantile-clipping), or
- **`RareCategoryGrouper`** from exercise-02 (categorical, rare-
  bucketing).

In `tests/test_transformer.py`, write tests covering all five
axes from chapter 08:

1. **`fit_transform` correctness on a known input.** Hand-compute
   the expected output on a 5-row toy example and assert with
   `np.testing.assert_allclose` (numeric) or `assert_frame_equal`
   (DataFrame).
2. **`fit` is idempotent.** `tx.fit(X).fit(X)` produces the same
   learned state as `tx.fit(X)`.
3. **Pre-fit `transform` raises.** Use
   `sklearn.utils.validation.check_is_fitted` inside `transform`
   if you have not already; assert it raises `NotFittedError`
   before `fit`.
4. **Train/test discipline.** Fit on `X_train`, transform
   `X_test`. Confirm the transformer's learned state (attributes
   ending `_`) does not change after seeing `X_test`. This is the
   single most important test: it catches leakage bugs at the
   transformer level.
5. **`clone` drops fitted state.** `clone(tx.fit(X))` produces an
   unfitted copy with the same hyperparameters and no
   trailing-underscore attributes.

Optional but recommended: run
`sklearn.utils.estimator_checks.check_estimator(YourTransformer())`
and either resolve or explicitly skip each failing check with a
one-line justification in your test file.

### Part B — one property-based test with Hypothesis (≈ 30 min)

In `tests/test_properties.py`, write a property-based test using
`hypothesis` for **one** invariant of your transformer that would
not be obvious from a hand-crafted example. Suggestions by
transformer:

- **`WinsoriseTransformer`**: for any input `X` and any
  `(lower, upper)` with `0 <= lower < upper <= 1`, the output
  satisfies `output.min(axis=0) >= tx.lower_bounds_` and
  `output.max(axis=0) <= tx.upper_bounds_`, element-wise.
- **`RareCategoryGrouper`**: for any DataFrame `X` and any
  `min_count >= 1`, the set of unique values in each column of the
  transformed output is a subset of the training-set unique
  values plus `{"__rare__"}`.

Use `hypothesis.extra.numpy.arrays` (for NumPy) or
`hypothesis.extra.pandas.data_frames` (for DataFrames) as your
strategies. Constrain shapes to small sizes (rows ≤ 200,
columns ≤ 10) so tests stay fast.

If Hypothesis finds a counterexample, fix the transformer — not
the test. That is the point.

### Part C — a golden test for the full pipeline (≈ 45 min)

In `tests/test_golden.py`, write a golden test that pins the
predictions of your fitted pipeline on a fixed held-out set.

Steps:

1. Build a **tiny** synthetic dataset (200 rows, ~10 columns) with
   `numpy.random.default_rng(0)`. Split with a fixed seed. Commit
   the seed and any generator code alongside the test — the
   dataset must be reproducible from the seed alone.
2. Build the pipeline with every random-state pinned:
   `random_state=0` on the model, `random_state=0` on the split,
   `default_rng(0)` on any data generation. If your pipeline uses
   PyTorch or TensorFlow, seed those too.
3. Fit on the training split; call `predict_proba` (classification)
   or `predict` (regression) on the held-out split.
4. On the first run, save the output to
   `tests/data/pipeline_predictions_golden.npy`:
   ```python
   np.save("tests/data/pipeline_predictions_golden.npy", probs)
   ```
   Commit the file.
5. On every subsequent run, load it and assert:
   ```python
   golden = np.load("tests/data/pipeline_predictions_golden.npy")
   np.testing.assert_allclose(probs, golden, rtol=1e-6, atol=1e-8)
   ```

**Then break it deliberately** to feel the test working:

- Change the imputer from `median` to `mean`.
- Run the test — it must fail with an obvious diff.
- Restore the imputer; confirm the test passes again.

Write in `REPORT.md` (one paragraph): what unit tests miss that
this golden test catches, and vice versa.

**A note on the regeneration workflow.** When you *intentionally*
tune a preprocessing step, regenerate the golden file in the
*same PR* as the code change, and explain in the commit message
what changed and why. Regenerating silently — "the test broke, so
I updated the golden" — defeats the whole purpose. Add a
`REGENERATE_GOLDENS=1 pytest` gate (or a small `scripts/regen_goldens.py`)
that makes regeneration a deliberate act rather than an accident.

### Part D — lint, type-check, and coverage in CI (≈ 30 min)

Extend the CI workflow from exercise-03 (or write a fresh one) so
that every push runs, in this order:

1. `uv sync --frozen` (or the poetry equivalent).
2. `uv run ruff check .`
3. `uv run ruff format --check .` — enforces formatting without
   auto-modifying files in CI.
4. `uv run mypy src tests` (or `pyright`). If your existing code
   has zero type hints, add them to the transformer + pipeline-
   builder module — enough to make `mypy` clean on `src/`.
5. `uv run pytest --cov=src --cov-report=term-missing`.

The build must be red if any step fails. Push, watch it run, and
capture the passing status in `REPORT.md`.

Then — this is the fun part — introduce and revert three
deliberate failures, one at a time:

- **A lint failure**: unused import.
- **A type failure**: pass a `list` where the hint says
  `np.ndarray`.
- **A golden failure**: change one hyperparameter of the pipeline
  without regenerating the golden.

Note in `REPORT.md` which check caught each one and how loud the
error was.

## Starter guidance

- If you use `pytest-cov`, aim for coverage on the transformer and
  pipeline-builder modules to be **≥ 90%**. Do not chase
  100% — coverage below the low 90s is often a real gap, above it
  is diminishing returns and encourages test theatre.
- `Hypothesis`'s default 100 examples per test is fine locally. If
  CI is slow, use `@settings(max_examples=50, deadline=None)`.
- Golden files should stay under a few kilobytes. If yours is
  megabytes, your test set is too big — shrink it.
- `assert_allclose` with `rtol=1e-6, atol=1e-8` is a good default
  for cross-platform-stable numerical output. If a test flakes on
  CI but passes locally, loosen `rtol` to `1e-5` before hunting
  for a heisenbug.
- For NumPy typing: `numpy.typing.NDArray[np.floating]` is the
  modern way; older `np.ndarray` still works but does not carry
  dtype information.
- For scikit-learn typing: annotations are limited. It is fine to
  `# type: ignore[no-any-return]` on `.fit(...)` calls where the
  return type is `Any`.

## Acceptance criteria

- `pytest -q` runs to green.
- **Part A**: the five transformer tests (correctness,
  idempotency, pre-fit raise, train/test discipline, clone) all
  present and passing.
- **Part B**: at least one property-based test runs `hypothesis`
  and finds no counterexample.
- **Part C**: the golden test loads a committed `.npy` file and
  asserts equality within tolerance. The deliberate-break-and-
  restore ritual is documented in `REPORT.md`.
- **Part D**: CI runs `ruff check`, `ruff format --check`, `mypy`
  (or `pyright`), and `pytest --cov`. The three deliberate
  failures were tried and reverted; `REPORT.md` records what
  each check said.
- Test coverage on the transformer + pipeline-builder is ≥ 90%,
  reported by `pytest --cov`.

## Stretch goals (optional)

- Use [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/)
  to add a performance regression test: `pipe.fit(X_train, y_train)`
  on a fixed dataset must complete in under N seconds. Pick N
  generously (2× current runtime) so you catch order-of-magnitude
  regressions, not noise.
- Add a **mutation testing** run with
  [`mutmut`](https://mutmut.readthedocs.io/) or
  [`cosmic-ray`](https://cosmic-ray.readthedocs.io/) against your
  transformer module. Any surviving mutants indicate untested
  behaviour worth writing a test for.
- Replace the golden `.npy` with a hash-based check
  (`hashlib.sha256(probs.tobytes())`), and commit the hex digest
  in the test file itself. Trade-off: no diff on failure, tighter
  git history.
- Wire the `check_estimator` conformance suite into CI, marked
  `slow` and gated behind a `pytest -m slow` flag. Run it nightly,
  not on every push.

## What to hand in

```
ml_toolkit/                           # (evolved from exercise-03)
├── src/
│   └── ml_toolkit/
│       ├── features.py
│       ├── transformers.py           # your tested transformer
│       └── pipeline.py               # your tested pipeline builder
├── tests/
│   ├── test_transformer.py           # Part A — unit tests
│   ├── test_properties.py            # Part B — hypothesis
│   ├── test_golden.py                # Part C — golden
│   └── data/
│       └── pipeline_predictions_golden.npy
├── .github/workflows/ci.yml          # Part D
├── pyproject.toml                    # ruff + mypy config
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
