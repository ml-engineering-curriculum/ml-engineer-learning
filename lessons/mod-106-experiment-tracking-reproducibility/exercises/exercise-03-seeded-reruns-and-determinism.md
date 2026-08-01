# exercise-03 — Seeded reruns and determinism

> **Estimated effort:** 2 hours.
> **Companion chapters:** [04 — Controlling randomness](../04-controlling-randomness.md), [05 — Pinning environments](../05-pinning-environments.md).

## Objective

Make one of your training scripts **repeatable on CPU
byte-for-byte** and **reproducible on GPU within a stated
tolerance**, and prove it with tests. By the end you should
have:

- A `utils/reproducibility.py` seed harness that seeds
  every RNG your training loop touches.
- A `tests/test_determinism.py` file with at least three
  passing tests that lock down the guarantee.
- A `DETERMINISM.md` that names your tier of reproducibility
  and states the tolerance in each case.
- A short throughput measurement showing the cost of running
  under deterministic algorithms on your hardware.

The exercise is deliberately small in code volume and
deliberately strict in tests. This is where reproducibility
either becomes a claim you can defend or a claim you cannot.

## Prerequisites

- Exercise-01 complete: an instrumented training script.
- A training script that uses at least one of: NumPy,
  scikit-learn (which uses NumPy under the hood), PyTorch.
- Ideally: access to a GPU for the second half of the tests.
  If you have no GPU, the exercise still works — you just
  skip the GPU tolerance test and note this in
  `DETERMINISM.md`.
- Chapter 04 of this module (essential); chapter 05
  (background for the environment discussion).

## Problem statement

### Part A — Write the seed harness (≈ 20 min)

Create `utils/reproducibility.py` with the following
functions, matching the chapter-04 harness closely:

```python
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python `random`, NumPy (legacy), PyTorch CPU + all
    GPUs, and PYTHONHASHSEED (for spawned processes).
    If deterministic=True, also enable framework-level
    deterministic algorithms.
    """

def make_dataloader_generator(seed: int) -> "torch.Generator":
    ...

def seed_worker(worker_id: int) -> None:
    ...
```

Requirements:

1. `seed_everything` sets **all** of: `PYTHONHASHSEED` env,
   `random.seed`, `np.random.seed`, `torch.manual_seed`,
   `torch.cuda.manual_seed_all`. Order does not matter, but
   the set is minimum.
2. When `deterministic=True`, also set
   `torch.use_deterministic_algorithms(True, warn_only=True)`,
   `torch.backends.cudnn.deterministic = True`,
   `torch.backends.cudnn.benchmark = False`, and
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
3. `seed_worker` derives per-worker seeds from
   `torch.initial_seed()`.
4. Every function has a docstring stating what it seeds and
   what it does not.

### Part B — Wire the harness into training (≈ 15 min)

Modify your training script so:

1. `seed_everything(seed, deterministic=True)` is the
   **first executable line** of `main()` (after argument
   parsing).
2. Any DataLoader in the code passes both
   `worker_init_fn=seed_worker` and
   `generator=make_dataloader_generator(seed)`.
3. The seed value is a top-level parameter of the training
   function, defaulting to `0`, and it is logged as an
   MLflow param.
4. The `deterministic=True` flag and the resulting
   `hardware` tags (chapter 05's block — CPU, GPU model, CUDA
   / cuDNN versions) are also logged / tagged.

### Part C — CPU byte-identical test (≈ 20 min)

In `tests/test_determinism.py` write:

```python
import torch
from utils.reproducibility import seed_everything
from src.train import train_toy    # a 1-epoch training function


def test_cpu_byte_identical():
    seed_everything(0, deterministic=True)
    m1 = train_toy(epochs=1, device="cpu")

    seed_everything(0, deterministic=True)
    m2 = train_toy(epochs=1, device="cpu")

    sd1, sd2 = m1.state_dict(), m2.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), (
            f"parameter {k} differs on second run — "
            f"seeding or DataLoader determinism is broken"
        )
```

If this test fails, do NOT skip forward. Every subsequent
part depends on it passing. Chapter 04's pitfall list is the
first place to look — the two most common causes are
"forgot `worker_init_fn`" and "used
`torch.manual_seed` but not `np.random.seed`".

If your training script uses only scikit-learn / no PyTorch,
adapt the test:

```python
def test_cpu_byte_identical_sklearn():
    seed_everything(0, deterministic=True)
    clf1 = train_toy_sklearn()

    seed_everything(0, deterministic=True)
    clf2 = train_toy_sklearn()

    # scikit-learn estimators can be pickled and compared byte-wise
    import pickle, hashlib
    h1 = hashlib.sha256(pickle.dumps(clf1)).hexdigest()
    h2 = hashlib.sha256(pickle.dumps(clf2)).hexdigest()
    assert h1 == h2
```

### Part D — Metric-tolerance tests (≈ 15 min)

Add two more tests that lock down the metric:

```python
def test_metric_bit_identical_cpu():
    seed_everything(0, deterministic=True)
    _, val_metric_1 = train_and_evaluate(device="cpu")

    seed_everything(0, deterministic=True)
    _, val_metric_2 = train_and_evaluate(device="cpu")

    assert val_metric_1 == val_metric_2


def test_worker_init_fn_matters():
    """Regression test: removing worker_init_fn must break
    byte-identical training. If this test passes with
    worker_init_fn removed, seeding is over-reaching or
    the DataLoader isn't using workers as expected."""
    seed_everything(0, deterministic=True)
    m_seeded = train_toy(epochs=1, device="cpu", use_worker_init=True)

    seed_everything(0, deterministic=True)
    m_unseeded = train_toy(epochs=1, device="cpu", use_worker_init=False,
                           num_workers=2)

    diffs = 0
    for k in m_seeded.state_dict():
        if not torch.equal(m_seeded.state_dict()[k], m_unseeded.state_dict()[k]):
            diffs += 1
    assert diffs > 0, (
        "Without worker_init_fn, at least one parameter should differ. "
        "If it doesn't, num_workers may be 0 — the test is not exercising "
        "the code path it claims to."
    )
```

The second test is a *negative* test — it confirms your
harness's `worker_init_fn` is actually load-bearing. Without
a negative test, "the CPU test passes" doesn't tell you
whether the harness is doing anything or whether the
workload happens to be deterministic by accident.

### Part E — GPU tolerance test (≈ 20 min)

*Skip this part if you have no GPU. Note the skip in
`DETERMINISM.md`.*

Add:

```python
import pytest, torch

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no GPU available")
def test_gpu_tolerance():
    seed_everything(0, deterministic=True)
    _, metric_1 = train_and_evaluate(device="cuda")

    seed_everything(0, deterministic=True)
    _, metric_2 = train_and_evaluate(device="cuda")

    tolerance = 1e-4
    assert abs(metric_1 - metric_2) < tolerance, (
        f"GPU determinism check: {metric_1} vs {metric_2} "
        f"outside tolerance {tolerance}"
    )
```

Choose a tolerance that reflects **your** hardware, not one
you copied from this file. Run the paired training twice on
your GPU; if it is byte-identical, tighten the tolerance to
`1e-6`. If it is not, loosen enough that the test passes
reliably (across five back-to-back invocations, no
flakiness).

Record whatever number you settle on in `DETERMINISM.md`
with the GPU model.

### Part F — Throughput cost (≈ 10 min)

Measure the cost of `deterministic=True` on your hardware.
Run your training script twice with the same seed and same
hyperparameters:

- Once with `deterministic=True`.
- Once with `deterministic=False` (also disable
  `cudnn.deterministic` and `cudnn.benchmark=False`, and
  turn `use_deterministic_algorithms(False)`).

Record:

- Wall-clock time for each.
- Peak GPU memory (`torch.cuda.max_memory_allocated()`).
- Final validation metric.

If you have no GPU, the cost on CPU is usually small; run
the CPU comparison and report it.

### Part G — `DETERMINISM.md` (≈ 20 min)

Write `DETERMINISM.md` with these sections:

```markdown
# Determinism claim

## Tier
- **Tier 1 (CPU byte-identical):** {{yes/no}}, verified by
  `tests/test_determinism.py::test_cpu_byte_identical`.
- **Tier 2 (GPU tolerance-bounded):** {{yes/no}}. If yes:
  tolerance `{{value}}` on `{{GPU model}}`, verified by
  `tests/test_determinism.py::test_gpu_tolerance`.

## What is seeded
- Python `random`
- NumPy (`np.random.seed`)
- PyTorch CPU (`torch.manual_seed`)
- PyTorch CUDA (`torch.cuda.manual_seed_all`)
- DataLoader workers (`worker_init_fn=seed_worker`)
- DataLoader shuffle order (`generator=`)
- `PYTHONHASHSEED` for spawned processes

## What is NOT seeded (variance sources)
- {{list any augmentation library, tokenizer parallelism,
  BLAS backend threading, etc. that you know is not seeded}}

## Cost of deterministic mode
- CPU: {{time_det}} vs {{time_nondet}} — {{ratio}}× slower
- GPU: {{time_det}} vs {{time_nondet}} — {{ratio}}× slower
- GPU peak memory: {{mem_det}} vs {{mem_nondet}}
- Validation metric under each mode: {{metric_det}} vs {{metric_nondet}}

## Reproduce these tests
```bash
uv sync --frozen
pytest tests/test_determinism.py -v
```

## Cross-hardware notes
- Same GPU model, back-to-back: passes tolerance
  `{{tolerance}}`.
- Different GPU model (if you tried one): metric shifted by
  {{delta}}.
- CPU vs GPU: metric shifted by {{delta}}.
```

## Starter guidance

- **Run the CPU test first, in isolation.** If tier-1 doesn't
  work, tier-2 is pointless. The CPU test is fast and
  catches almost every seeding bug.
- **Set `num_workers > 0` for the byte-identical test.**
  Otherwise, `worker_init_fn` isn't exercised and the test
  is falsely reassuring.
- **`PYTHONHASHSEED` inside `os.environ` does not affect the
  current process.** For strict reproducibility, prefix your
  test invocation with `PYTHONHASHSEED=0 pytest ...` in the
  Makefile or in CI. Document this in `DETERMINISM.md`.
- **The tolerance you pick is a statement of trust.** If
  you cannot demonstrate that back-to-back runs stay within
  it, do not claim that tolerance in the report. Loosen or
  investigate.
- **Do not enable `cudnn.benchmark=True` for speed and
  claim tier 2.** `benchmark=True` opts out of determinism.
  Chapter 04 explains why.

## Acceptance criteria

- `utils/reproducibility.py` exports `seed_everything`,
  `seed_worker`, and `make_dataloader_generator` matching
  the signatures above, with docstrings.
- The training script (from exercise-01) starts with a
  `seed_everything(seed, deterministic=True)` call.
- `pytest tests/test_determinism.py -v` shows:
  - `test_cpu_byte_identical` (or the sklearn variant)
    passing.
  - `test_metric_bit_identical_cpu` passing.
  - `test_worker_init_fn_matters` passing (the negative
    test — proves the harness is load-bearing).
  - `test_gpu_tolerance` passing OR skipped with a reason.
- `DETERMINISM.md` names the tier(s) achieved, lists the
  variance sources, and records the throughput cost.
- The MLflow run written by the instrumented script logs
  `seed`, `deterministic`, and the `hardware.*` tags.

## Reflection questions (answer in `DETERMINISM.md`)

- Chapter 04 lists PYTHONHASHSEED as something you cannot
  seed from inside the current process. In your test setup,
  did you find a case where a `dict`/`set` iteration order
  mattered? If you did not, does that mean it never will?
- Deterministic mode was `{{ratio}}×` slower on your GPU
  (or trivially slower on CPU). Would you turn it on for
  a production final-training run? Under what conditions?
- The negative test (`worker_init_fn`) forced you to prove
  the harness was doing work. Are there other places in
  your training pipeline where a "no-op harness" would
  silently pass a positive test? Name one.
- You state a tolerance in `DETERMINISM.md`. If you got a
  bug report six months from now saying "the metric moved
  by 0.001", would you rerun the tests immediately or ask
  for more context first? Why?

## Stretch goals (optional)

- **Cross-hardware experiment.** If you can run on a
  second GPU model (colab T4, a friend's laptop, a cloud
  instance), record the metric there and add a row to the
  cross-hardware table in `DETERMINISM.md`.
- **NumPy legacy vs new API test.** Add a fourth test
  demonstrating that swapping `np.random.default_rng(0)` for
  `np.random.seed(0)` in one function is enough to break
  determinism if not paired everywhere. Chapter 04's NumPy
  section is the theory.
- **CI wiring.** Add a `.github/workflows/determinism.yml`
  that runs `pytest tests/test_determinism.py` on every
  push, with `PYTHONHASHSEED=0` in the env. When a
  teammate breaks the seeding, CI catches it.
- **Bit-identical checkpoint hashing.** In addition to
  `torch.equal`, compute a SHA-256 hash of the serialised
  checkpoint on each run and assert they match. Useful when
  you want to detect drift in *un-parameterised* state
  (buffers, running mean/var in BatchNorm).
- **Bench the "prod fast" mode.** Add a `--fast` flag that
  disables determinism and enables `cudnn.benchmark=True`.
  Measure the speedup honestly; document which flag you
  would recommend a teammate use for what.

## What to hand in

```
exercise-03/
├── utils/
│   └── reproducibility.py       # seed_everything, seed_worker, generator
├── src/
│   └── train.py                 # calls seed_everything as first line
├── tests/
│   └── test_determinism.py      # CPU byte-identical, negative test, GPU tolerance
├── DETERMINISM.md
├── pyproject.toml
└── (optional) .github/workflows/determinism.yml
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.
