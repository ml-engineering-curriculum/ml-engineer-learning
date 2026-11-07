# lab-02 — Overfit a batch of 8: training-loop smoke tests

> **Duration:** ~45–60 minutes.
> **Companion chapters:** [03 — The PyTorch training loop](../03-the-pytorch-training-loop.md), [02 — Tensors, autograd, and devices](../02-tensors-autograd-and-devices.md), [05 — Building an MLP](../05-building-an-mlp.md).

## Objective

Chapter 03 states, in one of the most important paragraphs in the
module: **"if a network cannot memorise 8 examples, it will not learn
8000."** Make that smoke test a reflex. By the end of this lab you will
have (a) implemented chapter 03's overfit-a-batch-of-8 probe as a
reusable function, (b) planted four of chapter 03's named loop bugs
in a deliberately broken training script, (c) *used the probe to find
each one in turn*, and (d) locked the probe into a `pytest` test so
regressions in your own code are caught in CI, not in post-mortems.

This lab is deliberately narrower than exercise-01 (which builds the
full MLP + Dataset + train/val/checkpoint/early-stopping pipeline on a
real dataset and reports metrics against a `mod-104` baseline). Here
the model is the simplest two-layer MLP from chapter 05 and the data
is a 64-row synthetic split: the whole point is to isolate the
loop-is-correct question from everything else. If a bug survives this
lab you have a bug, not a modelling choice, and you catch it in
seconds instead of after a full epoch.

## Prerequisites

- Python 3.11+, PyTorch ≥ 2.1, `pytest`.
- Chapters 02 and 03 read through once; chapter 05's "The anatomy of
  an `nn.Module`" section skimmed.
- A working Python environment (`uv`, `poetry`, `venv` — any of them).
- A CPU is enough; nothing here needs a GPU.
- `lab-01` completed is helpful but not required.

## Setup

Create a lab directory and three files:

```bash
mkdir -p lab-02-overfit && cd lab-02-overfit
touch model.py broken_loop.py test_overfit.py
```

Install `pytest` if it isn't already on the path:

```bash
python -m pip install pytest
```

## Steps

### Step 1 — A minimal, correct model (≈ 5 min)

Chapter 05 ("The anatomy of an `nn.Module`"): the simplest MLP that
*could* be the fault is a two-layer network with one nonlinearity.
That is what we use — any failure to overfit 8 examples must therefore
be in the loop, not the model capacity.

Put in `model.py`:

```python
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """Two Linear layers + ReLU + Dropout. Enough capacity to
    memorise 8 examples; small enough that the loop is the only
    plausible failure mode."""

    def __init__(self, in_features: int = 4, hidden: int = 32,
                 out_features: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**Checkpoint.** `TinyMLP(in_features=4, hidden=32, out_features=3)` has
`4*32 + 32 + 32*3 + 3 = 259` trainable parameters, confirmed with
chapter 05's `sum(p.numel() for p in model.parameters() if
p.requires_grad)` idiom. With 259 free parameters and 8 training
examples × 3 classes, memorising this batch is **trivially achievable**
— if the loss will not go below 1% of its starting value after 200
optimisation steps, the loop is wrong.

### Step 2 — The overfit-a-batch probe as a reusable function (≈ 10 min)

Chapter 03 ("The 'does this training loop learn?' smoke test") gives
the pattern verbatim. Promote it from a snippet to a function.

Put in `broken_loop.py`:

```python
"""A broken training loop and a probe that will find the bugs.

We will introduce chapter-03 bugs one at a time in step 3, use the
probe to find them, and fix them. By the end `overfit_batch_of_eight`
is clean and the probe returns a small loss."""

import torch
import torch.nn as nn

from model import TinyMLP


def make_fixed_batch(seed: int = 0):
    """8 examples, 4 features, 3 classes. Fixed by seed."""
    gen = torch.Generator().manual_seed(seed)
    xb = torch.randn(8, 4, generator=gen)
    yb = torch.randint(0, 3, (8,), generator=gen)
    return xb, yb


def overfit_batch_of_eight(
    train_step,              # a callable: (model, loss_fn, optimizer, xb, yb) -> loss_value
    *,
    steps: int = 200,
    seed: int = 0,
) -> dict:
    """Run a training routine for `steps` optimisation steps on 8
    examples. Return the initial and final loss values; the caller
    decides what counts as 'passed'."""
    torch.manual_seed(seed)
    model     = TinyMLP(in_features=4, hidden=32, out_features=3, dropout=0.2)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    xb, yb    = make_fixed_batch(seed=seed)

    history = []
    for step in range(steps):
        loss = train_step(model, loss_fn, optimizer, xb, yb)
        history.append(loss)

    return {
        "initial_loss": history[0],
        "final_loss":   history[-1],
        "history":      history,
    }
```

**Checkpoint.** `overfit_batch_of_eight` is now the probe chapter 03
names. Note two deliberate design choices: (a) `train_step` is a
**callback**, so we can swap in broken and fixed implementations in
the next step without touching this function; (b) `TinyMLP` and the
8-row batch are both seeded, so "did the loss drop" is a stable, non-
flaky assertion.

### Step 3 — Plant four bugs; use the probe to catch each (≈ 20 min)

Chapter 03 lists nine named pitfalls in "Pitfalls collected". We pick
four that map directly to the four lines of the loop, plus one from
chapter 02 that masquerades as a training bug.

Append each `train_step_v*` below to `broken_loop.py` as you go, run
the probe against each (a tiny driver block at the bottom of
`broken_loop.py` is fine), and read the output before moving on.

#### Bug A — "Forgot `zero_grad()`" (chapter 03 pitfall 1)

```python
def train_step_v1(model, loss_fn, optimizer, xb, yb):
    """BUG A: no optimizer.zero_grad() — gradients accumulate across steps."""
    model.train()
    logits = model(xb)
    loss   = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Run against the probe:

```python
if __name__ == "__main__":
    out = overfit_batch_of_eight(train_step_v1)
    print(f"[v1 no zero_grad]   init={out['initial_loss']:.4f}  final={out['final_loss']:.4f}  ratio={out['final_loss']/out['initial_loss']:+.4f}")
```

**Expected output.** `init ≈ 1.1`, `final ≈ 1.0+` or oscillating; the
ratio never drops below ~0.5. Chapter 03 pitfall 1: "Gradients
accumulate; the model silently learns the sum." Fix it in `v2`.

#### Bug B — "Applied `softmax` before `CrossEntropyLoss`" (chapter 03 pitfall 2)

```python
def train_step_v2(model, loss_fn, optimizer, xb, yb):
    """BUG B: softmax applied before CrossEntropyLoss — double softmax."""
    model.train()
    optimizer.zero_grad()                           # (A) fixed
    logits = model(xb)
    probs  = torch.softmax(logits, dim=-1)          # <-- the bug
    loss   = loss_fn(probs, yb)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Run:

```python
    out = overfit_batch_of_eight(train_step_v2)
    print(f"[v2 double softmax] init={out['initial_loss']:.4f}  final={out['final_loss']:.4f}  ratio={out['final_loss']/out['initial_loss']:+.4f}")
```

**Expected output.** The loss drops but *plateaus*, never getting below
roughly 60–80% of its starting value. Chapter 03 pitfall 2 and the
"Common losses and their expected shapes" table: `CrossEntropyLoss`
fuses `log_softmax` and NLL, so the probabilities are being softmaxed
*twice*; the resulting distribution is flatter than it should be, and
gradients are too weak to memorise even 8 examples. Fix in `v3` by
passing **raw logits**.

#### Bug C — "Doing validation with the model in `train()` mode" (chapter 03 pitfall 4)

For a batch-of-8 probe the equivalent is: calling `model.eval()` to
measure "final loss" but *keeping* `model.eval()` on throughout — so
dropout, which the model relies on for part of its regularisation and
which should be *active* during training, is off the entire time.

```python
def train_step_v3(model, loss_fn, optimizer, xb, yb):
    """BUG C: model is in eval mode during training — dropout off at training."""
    model.eval()                                    # <-- the bug (should be .train())
    optimizer.zero_grad()
    logits = model(xb)
    loss   = loss_fn(logits, yb)                    # (B) fixed: raw logits
    loss.backward()
    optimizer.step()
    return loss.item()
```

Run:

```python
    out = overfit_batch_of_eight(train_step_v3)
    print(f"[v3 wrong mode]     init={out['initial_loss']:.4f}  final={out['final_loss']:.4f}  ratio={out['final_loss']/out['initial_loss']:+.4f}")
```

**Expected output.** Here the loss actually goes to near zero because
8 examples and 259 parameters with dropout *off* is easy to overfit —
**which is the point.** The probe for this specific bug is subtler: in
a *real* training loop, `.eval()` during training disables dropout and
flips BatchNorm to use frozen running stats (chapter 02 "Train mode
vs eval mode"), which means the model you *evaluate* later is not the
model you *trained*. Read the history: if `v3` converges *faster* than
`v4` (next), that is the fingerprint. Record both for comparison.

Fix in `v4` by using `.train()` during training and `.eval()` only in
validation / inference paths.

#### Bug D — "Non-scalar loss" (chapter 03 pitfall 8) — a sneakier failure mode

```python
def train_step_v4(model, loss_fn, optimizer, xb, yb):
    """BUG D: loss is a tensor of shape (B,) instead of a scalar —
    reductions were turned off and the user forgot to reduce."""
    model.train()                                   # (C) fixed
    optimizer.zero_grad()
    logits   = model(xb)
    per_item = nn.functional.cross_entropy(logits, yb, reduction="none")  # (B,)
    loss     = per_item                             # <-- the bug: not reduced
    try:
        loss.backward()
    except RuntimeError as e:
        return f"RuntimeError: {e}"                 # expected failure mode
    optimizer.step()
    return float(loss.sum().item())
```

Run:

```python
    out = overfit_batch_of_eight(train_step_v4)
    print(f"[v4 non-scalar]     first-step report: {out['history'][0]!r}")
```

**Expected output.** The first-step report is a string beginning
`"RuntimeError: grad can be implicitly created only for scalar
outputs"`. Chapter 03 pitfall 8: "Non-scalar loss. `loss_fn(...)`
returned a vector. Reduce first (`.mean()`) or pass `reduction='mean'`
in the loss." Fix: `.mean()` on `per_item` (or drop `reduction="none"`
entirely).

#### The fixed version — all four bugs out

```python
def train_step_fixed(model, loss_fn, optimizer, xb, yb):
    """Chapter 03's four lines, correctly ordered and correctly reduced."""
    model.train()
    optimizer.zero_grad()
    logits = model(xb)
    loss   = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Run:

```python
    out = overfit_batch_of_eight(train_step_fixed)
    print(f"[fixed]             init={out['initial_loss']:.4f}  final={out['final_loss']:.4f}  ratio={out['final_loss']/out['initial_loss']:+.4f}")
```

**Expected output.** The initial loss is near `ln(3) ≈ 1.10` (chapter
05's debugging hint: "uniform predictions give loss ≈ `ln(C)`") and
the final loss is well below 1% of that — comfortably under `0.011`.

**Checkpoint.** Each bug is one chapter-03 pitfall, and each was
discovered by one run of the same probe. The probe is faster than a
real training epoch, and in a real project would have surfaced the
bug before the first `print(f"epoch 0 loss ...")` line was read.

### Step 4 — Encode the probe as a pytest test (≈ 10 min)

Chapter 03 treats the overfit probe as a property of a correct loop,
not as a one-off script. Make it a CI-runnable assertion.

Put in `test_overfit.py`:

```python
"""Pytest tests that lock chapter 03's overfit-a-batch-of-8 probe into CI."""

import pytest

from broken_loop import (
    overfit_batch_of_eight,
    train_step_fixed,
    train_step_v1,
    train_step_v2,
)


def test_fixed_loop_overfits_eight_examples():
    """A correct loop must memorise 8 examples — final loss < 1% of initial."""
    out = overfit_batch_of_eight(train_step_fixed, steps=200)
    assert out["final_loss"] < 0.01 * out["initial_loss"], (
        f"final {out['final_loss']:.4f} vs initial {out['initial_loss']:.4f} "
        f"— loop cannot overfit 8 examples; look at the four lines first."
    )


def test_broken_loops_fail_the_probe():
    """Each planted bug must FAIL the same probe — the test is the probe's test."""
    v1 = overfit_batch_of_eight(train_step_v1, steps=200)
    v2 = overfit_batch_of_eight(train_step_v2, steps=200)

    # v1 ('forgot zero_grad'): gradient sum is directionally wrong, loss does
    #    not collapse.
    assert v1["final_loss"] > 0.1 * v1["initial_loss"], (
        "If the 'forgot zero_grad' bug ever stops failing this probe, the "
        "probe is wrong, not the bug — tighten it."
    )

    # v2 ('double softmax'): loss decreases but plateaus; it never passes the
    #    fixed-loop threshold.
    assert v2["final_loss"] > 0.01 * v2["initial_loss"], (
        "The double-softmax bug should plateau well above the 1%-of-initial "
        "threshold used by test_fixed_loop_overfits_eight_examples."
    )


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_fixed_loop_robust_across_seeds(seed):
    """The probe is not supposed to be seed-dependent; prove it on four seeds."""
    out = overfit_batch_of_eight(train_step_fixed, steps=200, seed=seed)
    assert out["final_loss"] < 0.01 * out["initial_loss"]
```

Run it:

```bash
python -m pytest -q test_overfit.py
```

**Expected output.** Six test items (four parametrised + two others) —
all green in a fraction of a second. If `test_fixed_loop_overfits_eight_examples`
*fails*, something regressed in the four-line loop; if
`test_broken_loops_fail_the_probe` *passes trivially* (i.e. the
broken loop also crosses the 1% threshold), you have an overpowered
model or an underpowered probe — tighten it.

**Checkpoint.** You now have a CI-safe assertion that your training
loop can still overfit 8 examples. It runs in milliseconds, takes
nothing but PyTorch and `pytest`, and is cheaper to run on every PR
than it is to notice two days later that a cherry-pick changed the
sign of a gradient.

### Step 5 — The diagnostic you *actually* reach for in production (≈ 5 min)

Chapter 03 pitfall collection pairs with chapter 05's debugging
checklist: "Check that all weights have gradients after one step."
Add it alongside the probe.

Append to `broken_loop.py`:

```python
def grad_report(model: nn.Module, loss_fn, xb, yb) -> list[tuple[str, bool, float | None]]:
    """After one backward, report (name, has_grad, grad_norm) per parameter.

    Anything in the output where has_grad is False after .backward() is
    disconnected from the loss — the first suspect for 'loss doesn't move'."""
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss = loss_fn(model(xb), yb)
    loss.backward()

    report = []
    for name, p in model.named_parameters():
        g = p.grad
        report.append((name, g is not None, (g.norm().item() if g is not None else None)))
    return report


if __name__ == "__main__":
    from model import TinyMLP

    model   = TinyMLP()
    loss_fn = nn.CrossEntropyLoss()
    xb, yb  = make_fixed_batch(seed=0)
    for name, has, gnorm in grad_report(model, loss_fn, xb, yb):
        print(f"{name:30s}  has_grad={has}  grad_norm={gnorm}")
```

Run it.

**Expected output.** Four lines (two `Linear` layers × `weight` +
`bias`), each with `has_grad=True` and a positive finite `grad_norm`.
If any row had `has_grad=False` after a `.backward()`, chapter 05's
debugging step 3 says: "Any parameter with `grad is None` after
`.backward()` is disconnected from the loss."

**Checkpoint.** With the overfit probe plus the grad-report probe, you
have the two diagnostics chapter 03 and chapter 05 both recommend
*before* you look at the data. On a real project these two scripts
plus pytest are usually enough to triage any "loss doesn't move" bug
inside a coffee break.

## Expected final artefacts

Three files in `lab-02-overfit/`:

- `model.py` — the `TinyMLP` from step 1.
- `broken_loop.py` — the probe, the four buggy `train_step_v*`, the
  fixed `train_step_fixed`, and the `grad_report` helper.
- `test_overfit.py` — three (parametrised: six) passing pytest tests.

Running `python broken_loop.py` end-to-end and `python -m pytest -q`
should both complete in well under 10 seconds on a laptop CPU.

## Final checkpoint

Without re-reading chapter 03, answer:

1. The probe uses `steps=200` on a batch of 8. Chapter 03 says the
   probe should drive the loss "to zero (or as close as the model
   can)." What assumption about `TinyMLP` makes 200 steps enough here,
   and how would you adapt `steps` for a Transformer (chapter 07)
   where you also want to use this probe?
2. In step 3, bug C ("model.eval() during training") *still passed*
   the loss-drops-to-zero criterion. In what kind of model does
   `model.eval()`-during-training become a *loud* failure (loss moves
   much worse than it should) rather than a silent one (loss moves,
   but the evaluated model behaves differently later)? Chapter 06 is
   the right chapter to think about here.
3. The second pytest asserts that two buggy loops *fail* the probe.
   Why is it worth keeping that assertion in CI even after the bugs
   are fixed in `train_step_fixed`? (Compare the reasoning from
   `mod-104`'s `lab-01` for the analogous "KFold leaks groups" test.)
4. The `grad_report` probe would have caught a bug that the overfit
   probe would not — can you name one? (A hint lives in chapter 05
   pitfall 1 and chapter 02's `nn.ModuleList` rule.)

If you can answer all four, chapter 03's "If a network cannot memorise
8 examples, it will not learn 8000." is permanent knowledge, and the
loop is no longer something you *hope* is correct between training
runs — it is something you *assert* is correct on every PR.

## Where to go next

- `exercise-01` ("PyTorch training loop from scratch") uses this
  lab's overfit test as one of its seven required pytest tests
  (Part D). Drop this lab's `test_fixed_loop_overfits_eight_examples`
  into that exercise's `tests/test_loop.py` with minimal edits.
- `exercise-03` ("Small Transformer encoder") replays the same probe
  on a 16-example batch of short sequences — chapter 07 makes the
  mask correctness and positional-encoding presence into additional
  invariants worth asserting at the same tiny scale.
- `lab-03` ("Visualising learning-rate schedules") pairs well: once
  the loop is correct, chapter 08's `scheduler.step()` placement
  footgun is the next-most-common source of a loop that "trains, but
  doesn't learn."
