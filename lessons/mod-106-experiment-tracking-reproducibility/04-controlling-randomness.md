# 04 — Controlling randomness: seeds and determinism

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01, and a working training loop
> from `mod-104` or `mod-105`.

## Motivation

You rerun yesterday's training script. The validation
accuracy is off by 0.4%. Did the code change break something,
or is that just noise? Without controlled randomness, you
cannot tell — every subsequent experiment is measured against
a moving baseline.

Two skills matter, and they are not the same:

1. **Seed every RNG your run touches.** So that the sequence
   of pseudo-random numbers your run consumes is a function
   of a single integer.
2. **Force deterministic algorithms where the framework has
   a choice.** So that even with a fixed seed, the *same*
   floating-point result comes out on the *same* hardware.

Miss the first and no seed helps. Miss the second and the
seed helps on Monday but not on Tuesday when the CUDA driver
updates.

This chapter walks through both, honestly labels what each
buys and what each costs, and finishes with a "reproducibility
harness" you will reuse in every training script for the rest
of the module.

## The RNG layers you actually touch

A typical PyTorch training script consumes randomness from
at least four sources. Every one has its own seed:

| Layer | Where it shows up | Seed API |
|---|---|---|
| Python `random` | `random.shuffle`, ad-hoc data code | `random.seed(s)` |
| NumPy | Any `np.random.*` call, scikit-learn splits | `np.random.seed(s)` **and** modern `np.random.default_rng(s)` |
| Framework CPU | Weight init, `torch.randn`, dropout | `torch.manual_seed(s)` |
| Framework GPU | Same, on CUDA | `torch.cuda.manual_seed_all(s)` |
| DataLoader workers | Per-worker shuffling, augmentations | `worker_init_fn` + `torch.Generator` |
| Hash randomisation | `set`/`dict` iteration order, only via `PYTHONHASHSEED` | env var, set before Python starts |

Two things people commonly get wrong:

- **Setting `torch.manual_seed` does not seed NumPy or
  Python `random`.** They are independent RNGs.
- **DataLoader workers get fresh RNGs.** Without a
  `worker_init_fn` they will each produce a different order
  on every run, even with the same top-level seed.

## The one-function reproducibility harness

Here is the function you should put in a `utils/reproducibility.py`
and import at the top of every training script:

```python
# utils/reproducibility.py
import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed every RNG the training stack touches.

    If deterministic=True, also force cuDNN / CUDA to use
    deterministic algorithms (slower, sometimes much slower).
    Call this ONCE at the top of the script, before building
    the model or the dataloader.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Framework-level determinism (see later section).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required by some CUDA reductions when
        # use_deterministic_algorithms is True:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Generator to pass to DataLoader(generator=...)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker_init_fn that gives each worker a
    deterministic-but-distinct seed derived from
    torch.initial_seed(), which the main process already set.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

Wire it in:

```python
from utils.reproducibility import (
    seed_everything, make_dataloader_generator, seed_worker,
)

SEED = 0
seed_everything(SEED, deterministic=True)

g = make_dataloader_generator(SEED)
train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

The rest of this chapter explains *why* each line is there,
and what breaks if you skip one.

## Why `PYTHONHASHSEED`

Python randomises the hash of strings on interpreter start
so that adversarial inputs cannot cause worst-case hash-map
collisions. That randomisation changes the iteration order
of sets and (in older Python) dicts. It rarely matters for
model training — but if you write any code that iterates a
set of feature names, or a dict whose keys are strings, the
order can differ between runs.

Setting `PYTHONHASHSEED` inside the process is a
convention: **it does not affect the current interpreter.**
Python reads the env var at startup only. Set it in the
shell before you launch:

```bash
PYTHONHASHSEED=0 python train.py
```

The line in `seed_everything` is present so *child processes*
(the DataLoader workers, subprocesses you spawn) inherit the
seed. For strict determinism, set it in the shell too.

## NumPy: legacy vs modern APIs

There are two ways to seed NumPy, and both matter:

```python
np.random.seed(seed)              # legacy — global state
rng = np.random.default_rng(seed) # modern — explicit Generator
```

- **Legacy `np.random.seed`** seeds the global process-wide
  RNG. Any `np.random.rand`, `np.random.choice`, etc. call
  in any imported library — including scikit-learn's
  `train_test_split` when `random_state` is omitted — reads
  from this global state.
- **Modern `default_rng`** returns an explicit `Generator`
  object with its own state. Code that receives the
  generator is deterministic; unrelated code that reads the
  global RNG is not.

`seed_everything` sets the legacy one because most existing
code uses it. For code you write yourself, prefer:

```python
rng = np.random.default_rng(seed)
noise = rng.normal(size=(32, 32))
```

Passing a `Generator` around beats mutating global state.
NumPy's own docs (NEP 19) recommend this. `mod-102` chapter
02 has the depth on why.

## Framework: PyTorch CPU vs CUDA

```python
torch.manual_seed(seed)           # seeds the CPU RNG
torch.cuda.manual_seed_all(seed)  # seeds every visible GPU
```

The `all` variant is important on multi-GPU systems: without
it, each device is seeded independently and multi-GPU code
can diverge on the first random op. It is safe to call on
CPU-only systems too — it's a no-op.

`torch.manual_seed` also seeds the CUDA generator on the
current device as a convenience — but not on other devices.
`manual_seed_all` is the belt-and-braces call.

## DataLoader determinism — the piece everyone misses

Take a DataLoader with `num_workers > 0` and try to reproduce
a run. It won't reproduce, even after `torch.manual_seed`.
Why:

- Each worker is a fresh Python process. Its `random` /
  `numpy` state was not seeded by your parent process.
- The shuffling of batches in the main process reads from a
  separate `torch.Generator` if you provide one, and from
  the default (unseeded across restarts unless
  `torch.manual_seed` was called) otherwise.

The fix is two arguments to `DataLoader`:

```python
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,   # seeds each worker
    generator=g,                  # seeds the shuffle order
)
```

`worker_init_fn` runs once per worker on startup; the
canonical body is the `seed_worker` in the harness above.
It derives each worker's seed from `torch.initial_seed()`,
which is set by `torch.manual_seed` in the parent process —
so the worker seeds are deterministic given the parent seed
but distinct per worker.

Without both arguments, two identically-configured runs will
see the training examples in different orders. That is
usually enough to shift the final metric by fractions of a
percent — invisible to a human, fatal to a comparison.

## Framework-level determinism (the expensive part)

Seeding is necessary, not sufficient. Many deep-learning
kernels are not deterministic *by default* even when their
inputs are:

- **cuDNN convolutions** pick an algorithm at run time; the
  picker uses benchmarking (`torch.backends.cudnn.benchmark`)
  and can choose different algorithms across runs.
- **Reductions on GPU** (sum over a tensor, softmax with
  many classes) can accumulate values in a non-deterministic
  order due to floating-point parallelism.
- **Scatter/gather ops** (`index_add_`, some pooling
  backwards) have non-deterministic implementations by
  default.

PyTorch exposes a single opt-in switch:

```python
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

`warn_only=True` warns when you call an op that has no
deterministic implementation instead of raising, so you can
see what would break. Set `warn_only=False` when you want a
hard guarantee — a crash is safer than a silently
non-deterministic run.

**The trade-off is real.** Deterministic mode on GPU:

- Disables cuDNN's algorithm auto-tuner. Convolutions can
  run substantially slower — the exact factor depends on
  model and shapes.
- Uses deterministic-but-slower reductions. Softmax and
  cross-entropy over many classes get slower.
- Sometimes forbids an op entirely — `torch.nn.EmbeddingBag`
  gradient, some scatter ops. You get a `RuntimeError`;
  refactor around it.

On CPU the cost is negligible. On GPU, budget for it: 10–40%
slower is not unusual on convnets.

**Rule of thumb:**

- **Development, sweeps, baselines being compared** — turn
  determinism ON. The speed cost is worth the ability to
  trust variance numbers.
- **Final production training runs** — you may turn it OFF
  once the model is chosen and you are running to squeeze
  final performance. Log the fact that you did.

The env var `CUBLAS_WORKSPACE_CONFIG=:4096:8` is required for
some CUDA operations under `use_deterministic_algorithms`
(cuBLAS-based ops that use workspace memory). Setting it
unconditionally is harmless; without it, you get a
`RuntimeError` from cuBLAS on some kernels.

<!-- needs-research: verify the current list of PyTorch ops
with no deterministic implementation on the latest stable
release; the list narrows over time. -->

## What seeding does *not* control

Even with `seed_everything(0, deterministic=True)` and every
DataLoader argument correct, these can still shift a result:

- **Hardware.** Different GPU generation → potentially
  different floating-point rounding in the same kernel. Even
  A100 vs H100 running the same PyTorch build can produce
  bit-different results on some ops.
- **CUDA / cuDNN / driver versions.** Kernel implementations
  change between releases. Two hosts on the same PyTorch
  version can still differ.
- **BLAS backend** (MKL vs OpenBLAS on CPU). Multi-threaded
  reductions are the classic source.
- **Number of DataLoader workers.** `num_workers=4` shuffles
  data differently from `num_workers=2` even with the same
  seed, because the round-robin over workers changes.
- **Batch size.** Changes gradient accumulation and thus the
  optimiser trajectory. Not really a "reproducibility bug";
  it is a different experiment.
- **Non-determinism outside PyTorch.** Tokenizers with
  parallelism (Hugging Face `tokenizers` in some configs),
  albumentations image augmentations, `numba`-jit'd feature
  code — each has its own seed story.

The honest answer to "why isn't this bit-identical" is
usually one of the above, not a bug in your seeding.

## Two-tier reproducibility: what to claim and what to check

Given the above, aim for a two-tier standard in reports:

- **Tier 1 — CPU byte-identical.** Same seed, same code, same
  data, same CPU box → checkpoints hash-equal, metrics
  bit-identical. This tier is achievable and worth
  demanding for tests and toy runs. It catches
  instrumentation regressions in CI.
- **Tier 2 — GPU tolerance-bounded.** Same seed, same code,
  same data, same GPU model → metrics within a stated
  tolerance (e.g. ±0.001 accuracy). Different GPU model →
  metrics within a looser tolerance you name in the
  reproducibility report (chapter 09).

Bake this into your tests:

```python
def test_deterministic_cpu_run():
    from utils.reproducibility import seed_everything
    seed_everything(0, deterministic=True)

    m1 = train_toy(epochs=1, device="cpu")
    seed_everything(0, deterministic=True)
    m2 = train_toy(epochs=1, device="cpu")

    for k in m1.state_dict():
        assert torch.equal(m1.state_dict()[k], m2.state_dict()[k]), k
```

If this fails, the seeding is wrong before anything else can
be. Chapter 09's report format includes the seed and the
device model as required fields for exactly this reason.

## Seeding sweeps and evaluation

A sweep (chapter 08) trains N models with different
hyperparameters. Two seeding options:

- **Same seed for every trial.** Cleanest — the only source
  of variance across trials is the hyperparameter. Use this
  by default.
- **Different seed per trial**, treated as an
  additional hyperparameter. Use when you need to marginalise
  out seed variance (e.g. sweep 8 seeds × 4 learning rates =
  32 trials, report mean + std over seeds for each LR).

Evaluating a final chosen configuration should always
include a *seed sweep* — retrain the winning configuration
under 3–5 different seeds and report mean ± std. A single
seed's win can be an artefact; three seeds' win is a
signal. Chapter 09's report has a section for this.

## Common pitfalls

1. **Setting only `torch.manual_seed`.** NumPy and Python
   `random` are separate RNGs. Data splits (scikit-learn)
   go through NumPy; the DataLoader workers go through
   Python. Use the harness above.
2. **Seeding *inside* the training loop.** Reseeding every
   epoch collapses the sequence and often causes worse
   results without you knowing why. Seed once, at the top.
3. **Forgetting `worker_init_fn`.** Multi-worker DataLoader
   without one is the most common silent source of
   non-reproducibility.
4. **Assuming `deterministic=True` is free.** It is not free
   on GPU. Measure the throughput cost; log it if you turn
   it off later.
5. **Reproducing on a different GPU.** A100 vs H100 vs T4 all
   produce different bit-level results on some ops. Log the
   GPU model; compare within-model only for the byte-identical
   claim.
6. **Random augmentations from a non-PyTorch library.**
   `albumentations`, `imgaug`, `nlpaug` all have their own
   RNG. Seed them explicitly; grep the pipeline.
7. **Setting `cudnn.benchmark = True` for speed.** It's a
   perfectly reasonable production flag but it opts out of
   determinism; if you set it, you cannot also claim tier-1
   reproducibility.

## Summary

- Seed every RNG your run touches: Python `random`, NumPy
  (legacy and modern), PyTorch CPU, PyTorch CUDA, and the
  DataLoader (via `worker_init_fn` and a `generator`).
- One helper (`seed_everything`) is all the code you need.
  Put it in every training script.
- Seeding gets you a reproducible *sequence of pseudo-random
  numbers*. Framework determinism (`use_deterministic_algorithms`,
  `cudnn.deterministic`) gets you the same *floating-point
  answers* — at a real speed cost on GPU.
- Aim for byte-identical on CPU, tolerance-bounded on GPU.
  Log the seed and the GPU model so a reader knows which
  tier your claim belongs to.
- Sweep multiple seeds to distinguish "this config wins" from
  "this seed on this config wins".

## Where this goes next

Chapter 05 pins the *environment* — the Python interpreter,
the resolved dependency versions, the CUDA build, the
container image. Seeds without a pinned environment are a
lock without a door.
