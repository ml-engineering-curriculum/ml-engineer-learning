# lab-03 — Visualising learning-rate schedules

> **Duration:** ~30–45 minutes.
> **Companion chapters:** [08 — Regularisation and learning-rate schedules](../08-regularisation-and-lr-schedules.md), [03 — The PyTorch training loop](../03-the-pytorch-training-loop.md).

## Objective

Chapter 08 names four schedules ("constant, step, cosine, warmup +
cosine") and spends a whole section on the single most common
scheduler bug: calling `scheduler.step()` in the wrong place in the
training loop. The bug is subtle because the loop keeps training —
"either off-by-one (harmless) or off-by-a-factor-of-100 (silently
ruinous)" — so the habit this lab installs is to always *plot the
schedule* before trusting it on a real run.

By the end of this lab you will have:

- A single Python file that instantiates each of the four chapter-08
  schedulers against a toy optimiser, walks them through a full mock
  training run of `epochs × steps_per_epoch` iterations, and plots the
  per-step learning rate.
- Side-by-side evidence of the per-step-vs-per-epoch footgun: the
  same scheduler called in the wrong place produces a curve so
  different from the intended one that no training run should survive
  it.
- A "learning-rate range test" plot (Smith, 2017) following chapter
  08's description, on the same synthetic problem as `lab-01`, so that
  the recipe "best LR = ~10× below where loss explodes" is an eyeball
  thing rather than a quote.

This lab does not train any real model — that is what exercise-01,
exercise-02, and exercise-03 are for. The point here is to make the
scheduler and its `.step()` placement **visible** one time, so the
chapter-08 footgun becomes impossible to repeat.

## Prerequisites

- Python 3.11+, PyTorch ≥ 2.1, Matplotlib (any plotting library is
  fine — adjust the imports).
- Chapter 08 read through once; chapter 03's "Where the learning rate
  scheduler goes" section skimmed.
- A working Python environment (`uv`, `poetry`, `venv` — any of them).
- A CPU is enough.

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-03-schedules && cd lab-03-schedules
touch schedules.py
mkdir plots
```

Everything below goes in `schedules.py`. Run with
`python schedules.py` after each step.

Start the file with imports:

```python
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    MultiStepLR,
    OneCycleLR,
)
import matplotlib.pyplot as plt

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)
```

## Steps

### Step 1 — A toy optimiser as the scheduler's mount (≈ 5 min)

Chapter 08 names schedulers that wrap a `torch.optim.Optimizer`; the
model attached to that optimiser does not matter for seeing the
schedule — any one parameter is enough. Use the simplest.

Add to `schedules.py`:

```python
def toy_optimizer(base_lr: float = 1e-3) -> torch.optim.Optimizer:
    """One parameter, no model. The scheduler only reads optimizer.param_groups."""
    param = torch.zeros(1, requires_grad=True)
    return torch.optim.SGD([param], lr=base_lr)


def record_lrs(optimizer, scheduler, total_steps: int, step_scheduler_at: str) -> list[float]:
    """Walk through `total_steps` fake training iterations and record the LR
    the optimiser would have used for each step.

    step_scheduler_at:
      - 'per_step':   scheduler.step() right after optimizer.step().
      - 'per_epoch':  scheduler.step() once per epoch (every 100 steps here).
    """
    assert step_scheduler_at in ("per_step", "per_epoch")
    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])        # the LR USED this step
        optimizer.step()                                    # fake update (param is zero, no-op in effect)
        if step_scheduler_at == "per_step":
            scheduler.step()
        elif step_scheduler_at == "per_epoch" and (step + 1) % 100 == 0:
            scheduler.step()
    return lrs
```

**Checkpoint.** `record_lrs` captures the LR the optimiser *reads* on
each step — that is the LR actually applied to any parameter in a real
loop. Which step `scheduler.step()` is called *after* decides what
`param_groups[0]["lr"]` looks like on the next iteration — chapter 03
and chapter 08 both call this out as the single thing that goes wrong
most often.

### Step 2 — Plot the four chapter-08 schedules (≈ 15 min)

The four schedules chapter 08 names in "The four schedules you will
actually use" are: **constant**, **StepLR / MultiStepLR**, **cosine
annealing**, and **warmup + linear or cosine decay**. The chapter says
`OneCycleLR` is a batteries-included alternative to the last one.
Plot each.

Append:

```python
EPOCHS           = 20
STEPS_PER_EPOCH  = 100
TOTAL_STEPS      = EPOCHS * STEPS_PER_EPOCH
PEAK_LR          = 1e-3


def schedule_constant() -> list[float]:
    """No scheduler — the 'skip the scheduler' row in chapter 08's four schedules."""
    opt = toy_optimizer(base_lr=PEAK_LR)
    # Use an identity LambdaLR to hit the same code path as the others.
    sched = LambdaLR(opt, lr_lambda=lambda step: 1.0)
    return record_lrs(opt, sched, TOTAL_STEPS, step_scheduler_at="per_step")


def schedule_multistep() -> list[float]:
    """StepLR / MultiStepLR: drop LR by `gamma` at fixed epochs. PER-EPOCH."""
    opt = toy_optimizer(base_lr=PEAK_LR)
    # Chapter 08's example: milestones in epoch-space.
    sched = MultiStepLR(opt, milestones=[10, 15, 18], gamma=0.2)
    return record_lrs(opt, sched, TOTAL_STEPS, step_scheduler_at="per_epoch")


def schedule_cosine() -> list[float]:
    """CosineAnnealingLR from chapter 08. PER-EPOCH."""
    opt = toy_optimizer(base_lr=PEAK_LR)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    return record_lrs(opt, sched, TOTAL_STEPS, step_scheduler_at="per_epoch")


def warmup_cosine_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Chapter 08's hand-rolled warmup + cosine decay. PER-STEP."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def schedule_warmup_cosine() -> list[float]:
    """Chapter 08's Transformer standard: linear warmup to peak, then cosine decay."""
    opt   = toy_optimizer(base_lr=PEAK_LR)
    sched = LambdaLR(opt, lr_lambda=lambda s: warmup_cosine_lambda(
        s, warmup_steps=int(0.1 * TOTAL_STEPS), total_steps=TOTAL_STEPS,
    ))
    return record_lrs(opt, sched, TOTAL_STEPS, step_scheduler_at="per_step")


def schedule_onecycle() -> list[float]:
    """OneCycleLR — the batteries-included warmup + cosine decay. PER-STEP."""
    opt   = toy_optimizer(base_lr=PEAK_LR)
    sched = OneCycleLR(opt, max_lr=PEAK_LR, total_steps=TOTAL_STEPS, pct_start=0.1)
    return record_lrs(opt, sched, TOTAL_STEPS, step_scheduler_at="per_step")


def plot_all_schedules():
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, fn in [
        ("constant",         schedule_constant),
        ("MultiStepLR",      schedule_multistep),
        ("CosineAnnealingLR",schedule_cosine),
        ("warmup+cosine (LambdaLR)", schedule_warmup_cosine),
        ("OneCycleLR",       schedule_onecycle),
    ]:
        lrs = fn()
        ax.plot(range(len(lrs)), lrs, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title(f"Chapter 08 schedules on {EPOCHS} epochs × {STEPS_PER_EPOCH} steps")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    out = PLOT_DIR / "schedules.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_all_schedules()
```

Run it.

**Expected output.** `wrote plots/schedules.png`. Open the file. You
should see:

- **constant.** Flat line at `1e-3`.
- **MultiStepLR.** Flat at `1e-3` until step 1000 (epoch 10), then a
  sudden drop to `2e-4`, another at step 1500 (epoch 15), another at
  step 1800 (epoch 18). Staircase shape.
- **CosineAnnealingLR.** Smooth drop from `1e-3` toward `eta_min=1e-6`
  following a cosine; but because this scheduler is called *per-epoch*
  here, the drop is **step-wise piecewise-constant within each epoch**
  rather than smooth within an epoch. Chapter 08 says: "Call per-epoch."
- **warmup + cosine (LambdaLR).** Linear ramp from 0 to `1e-3` over
  the first 10% of steps, then a smooth cosine decay to near zero —
  the Transformer standard from chapter 08.
- **OneCycleLR.** Similar shape to the hand-rolled version — brief
  warmup from a small LR up to `max_lr=1e-3` at `pct_start=0.1`,
  then a cosine-like decay the rest of the run.

**Checkpoint.** Chapter 08's table of "which schedule for which
model" (constant for small problems; StepLR for classical image
regimes; cosine for modern training; warmup + cosine for
Transformers) is now visually anchored. The log y-axis matters:
`MultiStepLR` drops by `0.2^3 = 0.008` across the three milestones, so
on a linear axis the last two milestones are barely visible.

### Step 3 — The `scheduler.step()` placement footgun (≈ 10 min)

Chapter 08's "Where the `.step()` goes" and chapter 03's pitfall 9
("`scheduler.step()` before `optimizer.step()`") combine into one
message: per-step schedulers must fire per step, per-epoch schedulers
per epoch. Call it wrong and the curve is nonsense.

Append:

```python
def schedule_warmup_cosine_WRONG_per_epoch() -> list[float]:
    """Warmup+cosine is a PER-STEP schedule — calling it per-epoch is the
    bug chapter 08 warns about. The resulting curve covers the same shape
    over `epochs` points instead of `total_steps` points, which is 100x
    too coarse on this run."""
    opt   = toy_optimizer(base_lr=PEAK_LR)
    sched = LambdaLR(opt, lr_lambda=lambda s: warmup_cosine_lambda(
        s, warmup_steps=int(0.1 * TOTAL_STEPS), total_steps=TOTAL_STEPS,
    ))
    return record_lrs(opt, sched, TOTAL_STEPS, step_scheduler_at="per_epoch")


def schedule_multistep_WRONG_per_step() -> list[float]:
    """MultiStepLR is a PER-EPOCH schedule — calling it per-step blows
    through all the milestones in the first 18 steps."""
    opt   = toy_optimizer(base_lr=PEAK_LR)
    sched = MultiStepLR(opt, milestones=[10, 15, 18], gamma=0.2)
    return record_lrs(opt, sched, TOTAL_STEPS, step_scheduler_at="per_step")


def plot_footgun():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    ax1.plot(schedule_warmup_cosine(),                  label="warmup+cosine (right: per-step)")
    ax1.plot(schedule_warmup_cosine_WRONG_per_epoch(),  label="warmup+cosine (wrong: per-epoch)",
             linestyle="--")
    ax1.set_xlabel("step"); ax1.set_ylabel("learning rate")
    ax1.set_title("Per-step schedule called per-epoch"); ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle=":", alpha=0.5); ax1.legend()

    ax2.plot(schedule_multistep(),                      label="MultiStepLR (right: per-epoch)")
    ax2.plot(schedule_multistep_WRONG_per_step(),       label="MultiStepLR (wrong: per-step)",
             linestyle="--")
    ax2.set_xlabel("step")
    ax2.set_title("Per-epoch schedule called per-step"); ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5); ax2.legend()

    out = PLOT_DIR / "footgun.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_all_schedules()
    plot_footgun()
```

Run it.

**Expected output.** `wrote plots/footgun.png`. The left panel: the
*right* warmup+cosine traces the familiar curve; the *wrong* version
is piecewise-constant with only 20 levels (one per epoch) and barely
warms up at all. The right panel: the *right* `MultiStepLR` makes its
first drop at step 1000; the *wrong* version blows through all three
milestones in the first 18 **steps** and the LR is `0.2^3 × 1e-3 =
8e-6` for the rest of the run — chapter 08's "off-by-a-factor-of-100
(silently ruinous)" literalised.

**Checkpoint.** You now have an artefact you can bookmark and point
at the next time someone asks why their Transformer stopped learning
after 1.8% of training. The two most useful signals from this plot in
practice: (1) the wrong-per-epoch warmup barely reaches peak LR; (2)
the wrong-per-step MultiStepLR collapses to `eta_min`-equivalent
almost immediately. Chapter 08: "Read the scheduler's docstring.
Calling in the wrong place is either off-by-one (harmless) or
off-by-a-factor-of-100 (silently ruinous)."

### Step 4 — A learning-rate range test (≈ 10 min)

Chapter 08's "Choosing a learning rate" section describes Smith
(2017)'s LR range test: "Increase LR linearly from very small to very
large over a few hundred batches while recording loss. Plot loss vs LR
(log scale) — loss will drop, stabilise, and then explode. Pick an LR
about 10× smaller than the one where loss started climbing."

Append (this is a mini version of the exercise-01 stretch goal — same
model, same data from `lab-01`, so the numbers are familiar):

```python
def lr_range_test(n_steps: int = 300, lr_min: float = 1e-7, lr_max: float = 1e-1):
    """Smith-style LR range test on a one-feature regression (lab-01's problem)."""
    torch.manual_seed(0)
    N = 128
    x = torch.linspace(-1.0, 1.0, N).unsqueeze(1)
    y = 2.0 * x + 1.0 + 0.1 * torch.randn(N, 1)

    model     = nn.Linear(1, 1)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)

    # Linear ramp of LR over n_steps.
    gammas = torch.linspace(0.0, 1.0, n_steps)
    lrs    = (lr_min * (lr_max / lr_min) ** gammas).tolist()   # geometric ramp

    recorded_lrs, recorded_losses = [], []
    for step, lr in enumerate(lrs):
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        recorded_lrs.append(lr)
        recorded_losses.append(loss.item())

        # Early-stop if the loss has blown up.
        if not math.isfinite(loss.item()) or loss.item() > 1e3:
            print(f"loss diverged at step {step}, lr={lr:.3g}")
            break

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recorded_lrs, recorded_losses)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate (log)")
    ax.set_ylabel("loss (log)")
    ax.set_title("Smith-style LR range test (chapter 08)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    out = PLOT_DIR / "lr_range_test.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    return recorded_lrs, recorded_losses


if __name__ == "__main__":
    plot_all_schedules()
    plot_footgun()
    lr_range_test()
```

Run it.

**Expected output.** `wrote plots/lr_range_test.png`, possibly
preceded by a `loss diverged at step ..., lr=...` line. Open the
plot. On this one-parameter regression you should see:

- A **flat plateau at very small LR** (nothing is happening).
- A **downward slope** across a middle band of LRs (fast learning).
- An **upward curl** and divergence at the high end.

Chapter 08's rule of thumb: the best LR is ~10× smaller than the LR
where the loss curve started climbing. For this problem that lands in
the `1e-2`–`1e-1` range (matching the LR we used in lab-01 by hand).
The number is not the point — the *shape* is.

**Checkpoint.** You can now do what chapter 08 recommends by hand in
10 lines on any real problem, without pulling in a third-party
dependency. The exercise-01 stretch goal ("Learning-rate range test
(chapter 08). Implement the Smith-style linear ramp from `1e-7` to
`1e-1` over a few hundred steps; plot loss vs. LR. Confirm your
default LR choice.") is a drop-in of this routine onto an MLP + real
dataset.

## Expected final artefacts

A single file `lab-03-schedules/schedules.py` plus three images in
`plots/`:

- `plots/schedules.png` — the four + 1 named schedules overlaid.
- `plots/footgun.png` — the per-step-vs-per-epoch bug, both
  directions.
- `plots/lr_range_test.png` — Smith's LR range test on the lab-01
  regression problem.

Running `python schedules.py` end-to-end should take a few seconds on
a laptop CPU.

## Final checkpoint

Without re-reading chapter 08, answer:

1. In Step 2 you called `CosineAnnealingLR` per-epoch and
   `warmup+cosine (LambdaLR)` per-step. Which clue in each scheduler's
   construction told you the right placement? (Hint: look at what
   `T_max` is measured in versus what `total_steps` is measured in,
   and chapter 08's "Where the `.step()` goes" section is the
   canonical reference.)
2. In Step 3's left panel, the wrong-per-epoch warmup+cosine curve
   barely reaches `PEAK_LR`. On a 100k-step Transformer run called
   per-epoch instead of per-step, what would the effective peak LR
   be (roughly), and what training behaviour does chapter 07 predict
   for "peak LR is a factor of 100 too small"? Which of chapter 07's
   "Debugging a Transformer that won't train" bullets describes it?
3. In Step 4 you used a **geometric** (log-linear) ramp from `1e-7`
   to `1e-1`. Chapter 08 describes a *linear* ramp. For hyperparameters
   that span orders of magnitude, which does `mod-104`'s chapter 08
   (classical ML) prefer and why — and which is Smith's original
   paper closer to for the LR range test? (Compare to classical-ML
   `lab-01`'s discussion of grid vs loguniform if you have done it.)
4. `OneCycleLR` wraps warmup + peak + cosine decay in one call. In
   which real-exercise context (exercise-01, -02, or -03) does the
   module *actually* use it, and why does that exercise prefer it
   over the hand-rolled warmup+cosine from Step 2?

If you can answer all four, chapter 08's scheduler recipes are no
longer a reading-time abstraction. The next time a training run looks
like it "learned too slowly" or "stopped improving early", `.step()`
placement and LR magnitude are the first two things you will look at
— and you will have the plot to prove which one is to blame.

## Where to go next

- `exercise-01` ("PyTorch training loop from scratch") lists an LR
  range test as a stretch goal (Part E / stretch goal #1). This
  lab's `lr_range_test` function is a drop-in for that stretch goal —
  swap the one-parameter regression for the MLP + tabular Dataset
  from that exercise.
- `exercise-02` ("CNN image classifier") uses `OneCycleLR` with
  `pct_start=0.1` for the CIFAR-10 CNN. The step-3 "right vs wrong"
  plot is what prevents the most common regression on that exercise
  (per-epoch `.step()` on a per-step scheduler).
- `exercise-03` ("Small Transformer encoder") requires warmup +
  decay and tests that the warmup actually happens (Part G test 7).
  The step-2 and step-3 plots give you the visual to understand
  what that test is guarding.
- Chapter 09's `torch.compile` and AMP material lands *on top of* a
  scheduler-correct loop. Do not debug both at once; get the schedule
  right first (this lab), then enable AMP (chapter 09).
