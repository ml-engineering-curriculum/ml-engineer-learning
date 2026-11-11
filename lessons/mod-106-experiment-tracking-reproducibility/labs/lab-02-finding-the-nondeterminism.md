# lab-02 — Finding the non-determinism: one seed at a time

> **Duration:** ~30–45 minutes.
> **Companion chapter:** [04 — Controlling randomness: seeds and determinism](../04-controlling-randomness.md).

## Objective

Chapter 04's central claim is that a PyTorch training run consumes
randomness from **four or five independent RNG layers**, and that seeding
only one of them — the usual beginner move of `torch.manual_seed(0)` — is
almost never enough. Make that visible the hard way: start from a tiny
training loop that uses Python `random`, NumPy, PyTorch, and a
`DataLoader` with workers; observe that **back-to-back runs diverge**;
then add one seed call at a time and see at which point the final model
state actually becomes byte-identical. The sequence directly mirrors the
chapter's RNG-layer table and `seed_everything` harness.

This lab is deliberately narrower than **exercise-03** (which writes the
full `seed_everything` harness into `utils/reproducibility.py`, a
`test_determinism.py` with a positive **and** a negative test, a GPU
tolerance test, and a `DETERMINISM.md` report). Here you don't produce a
reusable module or a test suite. You produce a single script and a tiny
log table that makes chapter 04's claim "`torch.manual_seed` is not
enough" an experimental finding you can point at, not advice you have
to take on trust.

## Prerequisites

- Python 3.11+, PyTorch ≥ 2.1, NumPy ≥ 1.26.
- A CPU is enough; this lab deliberately runs on CPU so that chapter
  04's **tier-1 (byte-identical CPU)** bar is the target. If you only
  have a GPU box, it still works — just add `device = "cpu"` explicitly.
- Chapter 04 read through once, especially "The RNG layers you actually
  touch" and "DataLoader determinism — the piece everyone misses".
- `lab-01` is helpful but not required.

## Setup

```bash
mkdir -p lab-02-nondeterminism && cd lab-02-nondeterminism
touch flaky_train.py
```

All code below goes into `flaky_train.py`. Start with imports and a
hash helper that reduces a model's state dict to a single SHA-256
fingerprint — one number per run makes the comparisons we need later
trivial:

```python
# flaky_train.py
import hashlib
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def state_fingerprint(model: torch.nn.Module) -> str:
    """SHA-256 of the concatenated raw bytes of every parameter tensor.
    Two models with byte-identical parameters get the same fingerprint;
    one parameter differing by one ULP produces a different fingerprint."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]
```

## Steps

### Step 1 — Build a flaky training function that touches every RNG layer (≈ 5 min)

Chapter 04's table lists **four RNG layers** that a typical PyTorch run
touches: Python `random`, NumPy (legacy + modern), PyTorch CPU, and the
DataLoader (which spawns **worker subprocesses** with their own Python +
NumPy state). We want a training function whose output depends on *all
four* so that leaving any one unseeded shifts the result.

Append to `flaky_train.py`:

```python
class NoisyDataset(Dataset):
    """Each __getitem__ reads from all four RNG layers:
    - the parent process's Python random (via the sampler's shuffle)
    - the worker's numpy (an augmentation-style noise source)
    - the worker's Python random (a random label flip)
    - the parent process's PyTorch RNG (weight init in step 2)
    """
    def __len__(self) -> int:
        return 64

    def __getitem__(self, idx: int):
        # NumPy noise — in the DataLoader this runs inside a worker.
        x_noise = np.random.randn(4).astype("float32")
        # Python random — also inside the worker.
        flip = random.random() < 0.5
        x = torch.tensor([float(idx)] * 4) + torch.tensor(x_noise)
        y = torch.tensor(1 if flip else 0, dtype=torch.long)
        return x, y


def train_once(num_workers: int = 2,
               use_worker_init: bool = False,
               use_generator: bool = False) -> str:
    """One mini training run. Returns the fingerprint of the final model.

    By default: uses num_workers>0 but does NOT pass worker_init_fn or
    a generator. That is the "flaky" baseline. The toggles let later
    steps add those arguments one at a time.
    """
    ds = NoisyDataset()

    g = None
    worker_init_fn = None
    if use_generator:
        g = torch.Generator()
        g.manual_seed(0)
    if use_worker_init:
        def _seed_worker(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        worker_init_fn = _seed_worker

    loader = DataLoader(
        ds, batch_size=8, shuffle=True, num_workers=num_workers,
        worker_init_fn=worker_init_fn, generator=g,
    )

    model = torch.nn.Linear(4, 2)       # weight init touches torch's RNG
    opt   = torch.optim.SGD(model.parameters(), lr=0.1)
    lossf = torch.nn.CrossEntropyLoss()

    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()

    return state_fingerprint(model)
```

Nothing to run yet. The next steps call `train_once` with different
seeding strategies and print the fingerprint.

**Checkpoint.** Three things to notice about the setup, each from
chapter 04:
(a) `NoisyDataset.__getitem__` deliberately touches **both** NumPy
(`np.random.randn`) and Python `random` (`random.random()`), so a run
depends on both layers — this is the "random augmentations from a
non-PyTorch library" shape chapter 04 pitfall #6 names.
(b) The `DataLoader` uses `num_workers=2`. Chapter 04 is explicit:
"Each worker is a fresh Python process. Its `random` / `numpy` state
was not seeded by your parent process." The `use_worker_init` /
`use_generator` toggles let us add chapter 04's two required `DataLoader`
arguments one at a time.
(c) `torch.nn.Linear(4, 2)` initialises its weights by sampling from
PyTorch's CPU RNG. Even with the DataLoader perfectly seeded, if
`torch.manual_seed` is not called the weights change from run to run.

### Step 2 — Baseline: run it twice with no seeding at all (≈ 5 min)

Append:

```python
if __name__ == "__main__":
    print("\n=== Step 2: no seeding ===")
    fp1 = train_once()
    fp2 = train_once()
    print(f"run 1 fingerprint: {fp1}")
    print(f"run 2 fingerprint: {fp2}")
    print(f"identical: {fp1 == fp2}")
```

Run:

```bash
python flaky_train.py
```

**Expected output.** Two different 16-character hex fingerprints and
`identical: False`.

**Checkpoint.** Chapter 04's opening paragraph is "You rerun yesterday's
training script. The validation accuracy is off by 0.4%. Did the code
change break something, or is that just noise? Without controlled
randomness, you cannot tell." You just reproduced that paragraph. Two
back-to-back runs of the same code, same data, same hardware — the
fingerprints disagree, which means every parameter potentially differs.
Any metric this model produced would be a moving baseline.

### Step 3 — Seed only `torch.manual_seed` (the usual beginner mistake) (≈ 5 min)

Replace the `__main__` block (or add a second block below the first) with:

```python
    print("\n=== Step 3: torch.manual_seed only ===")
    torch.manual_seed(0)
    fp1 = train_once()
    torch.manual_seed(0)
    fp2 = train_once()
    print(f"run 1 fingerprint: {fp1}")
    print(f"run 2 fingerprint: {fp2}")
    print(f"identical: {fp1 == fp2}")
```

Run again:

```bash
python flaky_train.py
```

**Expected output.** `identical: False` — still different fingerprints.
If, by remote chance, your two fingerprints happened to agree, bump
`NoisyDataset`'s `__len__` to 128 or `num_workers` to 4 and re-run.
The whole point is that one seed is not enough; the worker processes'
Python and NumPy RNGs are still uninitialised.

**Checkpoint.** Chapter 04 states the rule directly: "**Setting
`torch.manual_seed` does not seed NumPy or Python `random`.** They are
independent RNGs." The worker subprocesses in particular get fresh
Python `random` and NumPy state on every launch; `torch.manual_seed`
in the parent process cannot reach them. This is the single most
common beginner seeding bug and — until you print the fingerprints —
it is silent.

### Step 4 — Add Python `random` and NumPy seeding (parent process) (≈ 5 min)

Append:

```python
    print("\n=== Step 4: torch + random + numpy (parent only) ===")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    fp1 = train_once()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    fp2 = train_once()
    print(f"run 1 fingerprint: {fp1}")
    print(f"run 2 fingerprint: {fp2}")
    print(f"identical: {fp1 == fp2}")
```

Run:

```bash
python flaky_train.py
```

**Expected output.** Still `identical: False` on most boxes with
`num_workers=2` — the parent process is now deterministic, but the
**worker processes** still are not. The weight init (which happens in
the parent) is reproducible; the data the workers produce isn't.

**Checkpoint.** Chapter 04's "DataLoader determinism — the piece
everyone misses": "Each worker is a fresh Python process. Its `random`
/ `numpy` state was not seeded by your parent process. The shuffling
of batches in the main process reads from a separate `torch.Generator`
if you provide one, and from the default (unseeded across restarts
unless `torch.manual_seed` was called) otherwise." You've seeded the
parent process; you have not seeded the workers. This is the "forgot
`worker_init_fn`" bug chapter 04 pitfall #3 names.

### Step 5 — Add `worker_init_fn` (but not the generator) (≈ 5 min)

Append:

```python
    print("\n=== Step 5: + worker_init_fn, NO generator ===")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    fp1 = train_once(use_worker_init=True, use_generator=False)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    fp2 = train_once(use_worker_init=True, use_generator=False)
    print(f"run 1 fingerprint: {fp1}")
    print(f"run 2 fingerprint: {fp2}")
    print(f"identical: {fp1 == fp2}")
```

Run:

```bash
python flaky_train.py
```

**Expected output.** On most setups: `identical: True` finally appears
here, because `worker_init_fn` seeds the workers' Python and NumPy
RNGs from `torch.initial_seed()` (which the parent's
`torch.manual_seed(0)` set), **and** because `torch.manual_seed(0)` in
the parent already seeds the shuffle generator the DataLoader falls
back to. On some PyTorch versions or `num_workers` counts you may
still see a difference — the DataLoader's default behaviour around
shuffling and `generator=None` has shifted across releases. If so, step
6 fixes it.

**Checkpoint.** Chapter 04 shows the canonical `seed_worker`:

```python
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

Note that the per-worker seed is **derived from `torch.initial_seed()`**
rather than being a constant — chapter 04 is careful about this: the
workers get "deterministic-but-distinct" seeds given the parent seed,
"so the worker seeds are deterministic given the parent seed but
distinct per worker". Giving every worker the same constant seed would
make them all produce the same augmentations, which is almost never
what you want.

### Step 6 — Add the explicit `generator=` argument (the full chapter-04 harness) (≈ 5 min)

Append:

```python
    print("\n=== Step 6: + worker_init_fn AND generator= (full chapter-04 harness) ===")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    fp1 = train_once(use_worker_init=True, use_generator=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    fp2 = train_once(use_worker_init=True, use_generator=True)
    print(f"run 1 fingerprint: {fp1}")
    print(f"run 2 fingerprint: {fp2}")
    print(f"identical: {fp1 == fp2}")
```

Run:

```bash
python flaky_train.py
```

**Expected output.** `identical: True`. The pair of fingerprints should
be stable across every re-invocation of `python flaky_train.py` on the
same box. This is chapter 04's tier-1 bar: **CPU byte-identical**
(same seed, same code, same data, same CPU box → model parameters
hash-equal).

**Checkpoint.** You've now reconstructed `seed_everything` by hand, in
the exact order chapter 04 teaches:
1. `PYTHONHASHSEED` — set in the environment so **child processes**
   inherit a fixed hash seed. (Chapter 04: "The line in
   `seed_everything` is present so *child processes* (the DataLoader
   workers, subprocesses you spawn) inherit the seed. For strict
   determinism, set it in the shell too." On the parent process this
   assignment has no effect — Python read `PYTHONHASHSEED` at
   interpreter start.)
2. `random.seed(0)` — Python `random`.
3. `np.random.seed(0)` — NumPy legacy.
4. `torch.manual_seed(0)` — PyTorch CPU + current CUDA device.
   (Chapter 04 recommends also calling
   `torch.cuda.manual_seed_all(0)` for multi-GPU systems; on this
   CPU-only lab it's a no-op but harmless.)
5. `DataLoader(..., worker_init_fn=seed_worker, generator=g)` where
   `g = torch.Generator().manual_seed(0)` — the two load-bearing
   arguments chapter 04 says "everyone misses".

Every step above was a layer that had to be seeded separately. This is
why chapter 04 says "Miss the first and no seed helps" — and why
exercise-03 bundles the whole thing into one `seed_everything(seed)`
you call on the first executable line of training.

### Step 7 — The negative check: prove `worker_init_fn` is load-bearing (≈ 5 min)

A positive result on step 6 could still be an accident — maybe your
training workload doesn't actually consume randomness through the
workers and the test is tautological. Chapter 04 pitfall #3 is exactly
this concern; exercise-03's `test_worker_init_fn_matters` is the
regression test. Add the same check to this lab:

Append:

```python
    print("\n=== Step 7: negative check — removing worker_init_fn must break identity ===")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    fp_with    = train_once(use_worker_init=True,  use_generator=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    fp_without = train_once(use_worker_init=False, use_generator=True)
    print(f"with worker_init_fn:    {fp_with}")
    print(f"without worker_init_fn: {fp_without}")
    print(f"removing worker_init_fn changed the fingerprint: "
          f"{fp_with != fp_without}")
```

Run:

```bash
python flaky_train.py
```

**Expected output.** The last line prints **`True`**. Removing
`worker_init_fn` while keeping everything else the same **must** change
the fingerprint — otherwise the harness isn't exercising the worker
RNG path and the positive result on step 6 was only accidentally
right.

**Checkpoint.** Chapter 04 pitfall #3: "Forgetting `worker_init_fn`.
Multi-worker DataLoader without one is the most common silent source
of non-reproducibility." Exercise-03's `test_worker_init_fn_matters`
reads: "Regression test: removing `worker_init_fn` must break
byte-identical training. If this test passes with `worker_init_fn`
removed, seeding is over-reaching or the DataLoader isn't using
workers as expected." If your step 7 prints `False`, the most likely
cause is `num_workers=0` somewhere in `NoisyDataset`'s flow — go back
and confirm `DataLoader` is actually spawning workers.

### Step 8 — Record the result table (≈ 5 min)

Write a short `NOTES.md` summarising what the script produced:

```markdown
# Lab 02 — Non-determinism result table

| Step | What was seeded                                          | identical? |
|------|----------------------------------------------------------|-----------|
| 2    | (nothing)                                                | False     |
| 3    | torch.manual_seed only                                   | False     |
| 4    | + random.seed + np.random.seed (parent only)             | False     |
| 5    | + worker_init_fn (but no explicit generator)             | True / mixed |
| 6    | + explicit generator= (full chapter-04 harness)          | True      |
| 7    | negative check — removing worker_init_fn broke identity  | expected True |

Final fingerprint on this box: `<paste fp_with from step 7>`

Hardware: {{short description — CPU model, OS, Python, PyTorch version}}
```

Fill in step 5's cell honestly — the "True / mixed" depends on PyTorch
version and `num_workers`. The order **step 5 before step 6** is the
important finding: `worker_init_fn` is the one load-bearing argument
almost nobody knows to add; the explicit `generator=` is belt-and-braces.

**Checkpoint.** Chapter 04's rule: "Seeding gets you a reproducible
*sequence of pseudo-random numbers*." You've empirically derived the
minimum set of seed calls that get you a byte-identical CPU training
run — which is chapter 04's **tier-1 reproducibility** bar. Exercise-03
packages this into `utils/reproducibility.py` so you never have to think
about it again.

## Expected final artefact

```
lab-02-nondeterminism/
├── flaky_train.py             # steps 1–7 in a single file
└── NOTES.md                   # step 8 result table and your box's fingerprint
```

Running `python flaky_train.py` end-to-end should take a few seconds on
CPU and print seven blocks of output (one per step 2–7 plus the step-7
negative check), each with a `run 1 fingerprint`, `run 2 fingerprint`,
and an `identical:` or `changed:` verdict.

## Final checkpoint

Without re-reading chapter 04, answer:

1. In step 4 you seeded Python `random`, NumPy, and PyTorch in the
   parent process and the fingerprints still differed. Which **layer**
   from chapter 04's table was still unseeded, and why does setting
   `PYTHONHASHSEED` via `os.environ[...]` inside `seed_everything` not
   fix this on the parent process itself?
2. In step 6 the `seed_worker` function derives a worker's seed from
   `torch.initial_seed() % 2**32` rather than from a constant. What
   would go wrong with the data pipeline if every worker got the same
   constant seed, and what does the "`% 2**32`" guard against?
3. In step 7 removing only `worker_init_fn` changed the fingerprint. In
   a production training script, which pitfall from chapter 04's list
   does the negative check specifically defend against, and why can't a
   purely **positive** byte-identical test catch that same pitfall?
4. Chapter 04 draws the distinction between **tier 1 (CPU
   byte-identical)** and **tier 2 (GPU tolerance-bounded)**. Which tier
   did this lab clear, which did it not even try, and name one
   additional PyTorch setting
   (`torch.use_deterministic_algorithms`, `cudnn.deterministic`,
   `cudnn.benchmark`) that you would have to add before you could
   honestly claim tier 2 on a GPU box — and honestly state the price
   chapter 04 warns about.

If you can answer all four, chapter 04's harness is now a thing you
know by construction, and exercise-03's requirement to write it into
`utils/reproducibility.py` and lock it down with tests is a transcription
exercise rather than a research project.

## Where to go next

- **`exercise-03` ("Seeded reruns and determinism")** writes the full
  `seed_everything` harness, pins it down with a positive test, a
  metric-tolerance test, and the negative `test_worker_init_fn_matters`
  test you just reproduced by hand, adds a GPU tolerance test, and
  reports throughput cost in `DETERMINISM.md`.
- **`lab-01` ("Dissecting `mlruns/`")** makes chapter 02's object model
  concrete by inspecting the file layout MLflow writes for a single
  trivial run — useful background before you add MLflow logging into
  a reproducible training loop in exercise-01.
- **`lab-03` ("A 10-trial Optuna sweep end to end")** applies chapter
  08's sweep-reproducibility conditions (seeded sampler, durable
  storage) on top of the seeding guarantees you built here.
