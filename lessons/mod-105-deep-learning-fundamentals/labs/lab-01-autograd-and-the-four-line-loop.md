# lab-01 — Autograd and the four-line loop, by hand

> **Duration:** ~30–45 minutes.
> **Companion chapters:** [02 — Tensors, autograd, and devices](../02-tensors-autograd-and-devices.md), [03 — The PyTorch training loop](../03-the-pytorch-training-loop.md).

## Objective

Make chapter 02's three rules of `autograd` ("gradients accumulate",
"only leaf tensors get `.grad`", "`.backward()` needs a scalar") and
chapter 03's four-line loop (`zero_grad → forward+loss → backward →
step`) permanent, hands-on knowledge — by driving a one-parameter
linear regression end-to-end **without any `nn.Module` or optimiser
object**, and then rebuilding the same fit with `nn.Linear` and
`torch.optim.SGD`. Along the way you will reproduce three of the
chapter-02 pitfalls on demand (the "forgot `zero_grad`" bug, the
"validation with autograd on" overhead, and the "non-leaf tensor has no
`.grad`" surprise) so that when you see them in real code they are
instantly recognisable.

This lab is deliberately narrower than exercise-01 (which trains a full
MLP on a real tabular dataset with a test suite and a report): here the
model is `y = w·x + b` on synthetic data, the whole thing fits in one
Python file, and the purpose is to *see* autograd and the loop from
below, before anything is wrapped.

## Prerequisites

- Python 3.11+, PyTorch ≥ 2.1, NumPy ≥ 1.26, Matplotlib (any plotting
  library is fine).
- Chapters 02 and 03 read through once.
- A working Python environment (`uv`, `poetry`, `venv` — any of them).
- A CPU is enough; nothing in this lab needs a GPU.

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-01-autograd && cd lab-01-autograd
touch autograd_loop.py
```

Everything below goes in `autograd_loop.py`. Run with
`python autograd_loop.py` after each step and read the output before
moving on.

Start the file with imports and a seed:

```python
import math
import torch

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Steps

### Step 1 — Build a synthetic one-parameter regression (≈ 5 min)

Chapter 02's framing: tensors are NumPy arrays with `dtype`, `device`,
and (optionally) autograd tracking. Start with a tiny dataset where you
know the answer so you can tell whether the loop learned.

Add to `autograd_loop.py`:

```python
# Ground truth: y = 2.0 * x + 1.0 + small noise.
N = 128
x = torch.linspace(-1.0, 1.0, N, device=device).unsqueeze(1)   # (N, 1)
y = 2.0 * x + 1.0 + 0.1 * torch.randn(N, 1, device=device)     # (N, 1)

print("x", x.shape, x.dtype, x.device)
print("y", y.shape, y.dtype, y.device)
print("x[:3]", x[:3].flatten().tolist())
print("y[:3]", y[:3].flatten().tolist())
```

**Expected output.** Two shape-`(128, 1)` `float32` tensors on your
chosen device, the first three x-values close to `-1.0, -0.984, -0.969`,
and y-values roughly `2*x + 1` with a little jitter.

**Checkpoint.** `x` and `y` are both on the same device and both
`float32`. Chapter 02's rule — "every tensor must be on the same device
as the parameters of the layer it flows through" — becomes real in
step 2 when `w` and `b` join them.

### Step 2 — One gradient step by hand (≈ 10 min)

Chapter 02 says a tensor with `requires_grad=True` is a **leaf** that
participates in the autograd graph; `.backward()` on a scalar populates
`.grad` on every such leaf. Prove it on one parameter.

Append:

```python
# Two learnable scalars; both are LEAF tensors with requires_grad=True.
w = torch.zeros(1, requires_grad=True, device=device)
b = torch.zeros(1, requires_grad=True, device=device)

# Forward pass. preds is NOT a leaf — it has a grad_fn, no persistent .grad.
preds = w * x + b                              # (N, 1)
loss  = ((preds - y) ** 2).mean()              # scalar
print("loss (step 0, w=0, b=0):", loss.item())

# Backward. This populates w.grad and b.grad on the leaves.
loss.backward()
print("w.grad:", w.grad.item(), "   b.grad:", b.grad.item())

# Try to read preds.grad — it's None (preds is not a leaf).
print("preds.grad (non-leaf):", preds.grad)
```

Run it.

**Expected output.** A loss near `5.something` (variance of `y` around
zero plus the bias offset), a negative `w.grad` (to pull `w` up toward
`2.0`), a negative `b.grad` (to pull `b` up toward `1.0`), and
`preds.grad` printed as `None`.

**Checkpoint.** Three things to internalise, all from chapter 02:
(a) `requires_grad=True` on `w` and `b` is what makes them visible to
the graph; (b) `preds` inherits `requires_grad` from `w` and `b` but it
is an intermediate, not a leaf, so its `.grad` is `None` — this is
pitfall #5 ("Reading `.grad` on a non-leaf tensor"); (c) the forward
pass *builds* the graph, `.backward()` *walks* it. If you had instead
done the forward pass inside `torch.no_grad()`, the backward call would
have raised `RuntimeError: element 0 of tensors does not require grad
and does not have a grad_fn` (chapter 03's "graph-shape bug").

### Step 3 — Reproduce the "forgot `zero_grad`" bug on purpose (≈ 10 min)

Chapter 02 rule 1: "Gradients accumulate." Chapter 03 pitfall 1:
"Forgot `zero_grad()`. Gradients accumulate; the model silently learns
the sum." Make it visible.

Append:

```python
# Second backward pass, WITHOUT zeroing grads first.
# PyTorch does NOT let you call .backward() a second time on the same
# scalar by default (the graph was freed). Build a fresh graph:
preds2 = w * x + b
loss2  = ((preds2 - y) ** 2).mean()
loss2.backward()

print("w.grad AFTER second .backward() (no zero_grad):", w.grad.item())

# Now zero and show the delta.
w.grad.zero_()
b.grad.zero_()
preds3 = w * x + b
loss3  = ((preds3 - y) ** 2).mean()
loss3.backward()
print("w.grad AFTER zero_grad + fresh .backward():       ", w.grad.item())
```

Run it.

**Expected output.** The first printed `w.grad` is **twice** the value
you saw at the end of step 2 (two backward passes with no intervening
zero → sum of two identical gradients). The second printed `w.grad`,
after `w.grad.zero_()`, matches the original value from step 2 exactly.

**Checkpoint.** Chapter 03 says "every training run without `zero_grad`
silently learns the wrong thing" — this step is why. The PyTorch 2.x
default `optimizer.zero_grad(set_to_none=True)` is an optimisation: it
sets `.grad` to `None` instead of filling with zeros, which the next
`.backward()` allocates afresh. Here we use `.zero_()` because we are
touching the leaf tensors directly, without an optimiser. The *effect*
on step 3 is the same.

### Step 4 — The four-line loop, by hand, on raw leaves (≈ 10 min)

Now assemble chapter 03's four lines around these raw leaves — no
`optimizer`, no `nn.Module`, no `torch.no_grad()` yet. One epoch of
batch-of-N vanilla SGD.

Append:

```python
# Fresh parameters — start from zero again so we can compare runs.
w = torch.zeros(1, requires_grad=True, device=device)
b = torch.zeros(1, requires_grad=True, device=device)

LR     = 0.1
STEPS  = 200

losses = []
for step in range(STEPS):
    # (1) zero the previous step's grads
    if w.grad is not None: w.grad.zero_()
    if b.grad is not None: b.grad.zero_()

    # (2) forward + loss
    preds = w * x + b
    loss  = ((preds - y) ** 2).mean()

    # (3) backward
    loss.backward()

    # (4) step. Update under no_grad so the update itself doesn't
    # become part of a new autograd graph on the leaf.
    with torch.no_grad():
        w -= LR * w.grad
        b -= LR * b.grad

    losses.append(loss.item())
    if step % 40 == 0:
        print(f"step {step:3d}  loss {loss.item():.5f}  w {w.item():+.4f}  b {b.item():+.4f}")

print(f"\nFinal:  w = {w.item():+.4f}  (target 2.0)   b = {b.item():+.4f}  (target 1.0)")
```

Run it.

**Expected output.** Loss decreases monotonically from `~5` at step 0
toward `~0.01` by step 200; `w` approaches `2.0` and `b` approaches
`1.0` (give or take noise). Four printed epochs at step 0, 40, 80, 120,
160.

**Checkpoint.** You have just written chapter 03's four-line loop with
nothing abstracting it: `zero_grad` → `forward+loss` → `backward` →
`step`. Two points worth absorbing: (1) the `torch.no_grad()` wrapper
on the parameter update exists so the update itself does not get tracked
as a graph operation on the leaf (that would make subsequent
`.backward()` calls behave in confusing ways); `optimizer.step()`
handles the equivalent internally. (2) Chapter 03's canonical loop
replaces the three raw leaves + two in-place updates with
`torch.optim.SGD(model.parameters(), lr=...)` and
`optimizer.step()` — same math, less bookkeeping. Step 5 proves it.

### Step 5 — The same fit using `nn.Linear` and `torch.optim.SGD` (≈ 5 min)

Chapter 02's "end-to-end minimum" and chapter 03's "canonical loop":
same operation, wrapped. Confirm the fit lands in the same place.

Append:

```python
import torch.nn as nn

torch.manual_seed(0)

# Rebuild the same dataset (seed is reset above).
N = 128
x = torch.linspace(-1.0, 1.0, N, device=device).unsqueeze(1)
y = 2.0 * x + 1.0 + 0.1 * torch.randn(N, 1, device=device)

model     = nn.Linear(1, 1).to(device)
loss_fn   = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model.train()
for step in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()

w_hat = model.weight.item()
b_hat = model.bias.item()
print(f"\nnn.Linear fit:  w = {w_hat:+.4f}   b = {b_hat:+.4f}")
```

Run it.

**Expected output.** `w ≈ 2.0`, `b ≈ 1.0`, both close to what you got in
step 4 (the two paths are the same math at `lr=0.1` for 200 steps; the
starting point differs because `nn.Linear`'s default initialisation is
`kaiming_uniform_`, not zeros, so the exact trajectory differs).

**Checkpoint.** Chapter 02's rule 2 ("`nn.Module` is the container;
`nn.Parameter` is the leaf") and chapter 03's "The four lines, one by
one" are two framings of the same thing. The lab's step 4 is what
step 5 does internally. If the loop ever confuses you later, you now
know what to print.

### Step 6 — Validation with `torch.no_grad()` and `inference_mode()` (≈ 5 min)

Chapter 02's rule: "Any code path that does not call `.backward()`
should be inside `torch.no_grad()` or `@torch.inference_mode()`." Prove
the overhead of forgetting it.

Append:

```python
# Hold out a small val set.
x_val = torch.linspace(-1.5, 1.5, 32, device=device).unsqueeze(1)
y_val = 2.0 * x_val + 1.0 + 0.1 * torch.randn(32, 1, device=device)

# Wrong: validation builds a graph you throw away.
model.eval()
val_pred = model(x_val)                    # has grad_fn
print("val_pred.grad_fn (WRONG path):", val_pred.grad_fn)

# Right: wrap in no_grad; no graph is built.
with torch.no_grad():
    val_pred = model(x_val)
    val_loss = loss_fn(val_pred, y_val)
print("val_pred.grad_fn (no_grad path):", val_pred.grad_fn)
print("val_loss:", val_loss.item())

# Also right (and stricter): inference_mode.
@torch.inference_mode()
def predict(model, x):
    return model(x)

val_pred2 = predict(model, x_val)
print("val_pred2.grad_fn (inference_mode):", val_pred2.grad_fn)
```

Run it.

**Expected output.** `val_pred.grad_fn` is `<AddmmBackward0 object>`
(or similar) in the WRONG path — a graph was built even though you are
only validating. In both the `no_grad` and `inference_mode` paths,
`.grad_fn` is `None`. The `val_loss` is a reasonable number near
`0.01`.

**Checkpoint.** This is chapter 02's pitfall #2 ("Doing validation with
autograd on"). On a one-parameter model you cannot see the cost; on a
200-million-parameter one, this doubles memory for no reason.
`inference_mode` is *stricter* than `no_grad` (it also disables version
counters); prefer it for pure inference — chapter 02 is explicit about
that preference.

## Expected final artefact

A single file `lab-01-autograd/autograd_loop.py` containing the six
step blocks above. Running `python autograd_loop.py` end-to-end should
take under five seconds on a laptop CPU and print (in order):

- The dataset shapes and a few preview values.
- The step-2 grads and the non-leaf `preds.grad` `None`.
- The step-3 "forgot zero_grad" doubled gradient and the recovered
  single gradient.
- A monotonically-decreasing per-step loss trace from the step-4 raw
  loop, ending with `w ≈ 2.0, b ≈ 1.0`.
- The `nn.Linear` fit from step 5 matching the raw fit.
- The three `.grad_fn` lines from step 6 (first with a graph, next two
  without).

## Final checkpoint

Without re-reading chapters 02 and 03, answer:

1. In step 2, you printed `preds.grad` and got `None` even though
   `preds.requires_grad` is `True`. Which rule from chapter 02 explains
   this, and what would you call to retain `preds.grad` (and *why* is
   that only useful for debugging, not for training)?
2. In step 3, two consecutive `.backward()` calls doubled `w.grad`. In
   the real training loop (step 5), which call is doing the equivalent
   of the `w.grad.zero_()` from step 3, and what is the practical
   difference between `zero_grad()` and `zero_grad(set_to_none=True)`
   (which is the PyTorch 2.x default)?
3. In step 4, the parameter update is written as
   `with torch.no_grad(): w -= LR * w.grad`. What goes wrong if you
   drop the `torch.no_grad()` wrapper — and why does
   `optimizer.step()` not need you to think about this?
4. In step 6, the WRONG path left `val_pred.grad_fn` set. Which
   chapter-02 pitfall is that, and which chapter-03 pattern (name the
   function or decorator) would you use in a real training script so
   this never happens on the eval pass?

If you can answer all four, chapters 02 and 03's primitives are now
compile-time knowledge rather than reference material, and
exercise-01's training loop will go together without the usual two
hours of debugging the four core lines.

## Where to go next

- `exercise-01` ("PyTorch training loop from scratch") now reuses every
  pattern from this lab at the scale of a real MLP, a real dataset, a
  real `nn.Module`, and a real test suite that asserts each of the
  properties you proved by hand here.
- `lab-02` ("Overfit a batch of 8: training-loop smoke tests") uses
  chapter 03's debugging test to find deliberately-planted bugs in a
  training loop — the "if the loop can't memorise 8 examples, the
  loop is wrong" discipline.
- `lab-03` ("Visualising learning-rate schedules") plots the four
  schedulers chapter 08 names so that the per-step vs per-epoch
  `scheduler.step()` footgun becomes impossible to repeat.
