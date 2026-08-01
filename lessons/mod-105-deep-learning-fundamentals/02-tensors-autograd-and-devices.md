# 02 — Tensors, autograd, and devices

> **Reading time:** ~30 minutes.
> **Prerequisite chapter:** 01.

## Motivation

Every PyTorch program is built out of two things: **tensors**
(multidimensional arrays with dtype and device) and
**autograd** (the engine that computes derivatives of any
scalar with respect to any leaf tensor). Nothing else in the
module makes sense without these two.

This chapter is short and mechanical on purpose. Read it once,
run the snippets, and you will spend the rest of the module
using rather than fighting the API.

## The tensor

`torch.Tensor` is NumPy's `ndarray` with three additions:

1. Every tensor has a **`device`** — `cpu`, `cuda:0`, `mps`
   (Apple Silicon), `xpu` (Intel). Operations require operands
   on the same device.
2. Every tensor has a **`dtype`** — `float32`, `float16`,
   `bfloat16`, `int64`, `bool`, etc.
3. A tensor with `requires_grad=True` participates in the
   autograd graph.

Creating a tensor:

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])           # from data
y = torch.zeros(3, 4)                                # zeros
z = torch.randn(2, 3, device="cpu", dtype=torch.float32)
w = torch.arange(10)                                 # 0..9
```

Everything you would expect from NumPy works: slicing,
broadcasting, reductions, matrix multiplication (`@`),
elementwise ops. The idioms transfer.

```python
a = torch.randn(4, 3)
b = torch.randn(3, 5)
c = a @ b                    # (4, 5)
c.sum(dim=1, keepdim=True)   # (4, 1)
c.mean()                     # scalar tensor
```

### Shape is the interface

The single most common bug in deep learning code is a shape
mismatch. Print shapes early and often:

```python
def forward(self, x):
    print("x     ", x.shape)         # sanity check
    x = self.linear1(x)
    print("after1", x.shape)
    ...
```

Two idioms deserve names:

- **`.view(...)`** — reshape *without* copying (requires
  contiguous memory). Fails loudly if it can't.
- **`.reshape(...)`** — reshape, copying if needed. Almost
  always what you want unless you are optimising.

```python
t = torch.randn(2, 3, 4)
t.view(6, 4)         # OK — contiguous
t.transpose(0, 1).contiguous().view(3, 8)  # non-contiguous first
t.transpose(0, 1).reshape(3, 8)            # or: reshape handles it
```

### Broadcasting

Same rules as NumPy: dimensions are aligned from the right;
size-1 axes are broadcast. This means a single mistake — a
column vector where you meant a row vector — silently
produces the wrong shape and no error:

```python
labels = torch.tensor([1, 0, 1, 0])      # (4,)
logits = torch.randn(4, 2)               # (4, 2)
# Wrong: this broadcasts labels across columns silently.
# Right: compare against logits.argmax(dim=1), which is (4,).
```

The habit that pays off: `assert tensor.shape == (expected,
shape)` at boundaries — the entry to a module, before a loss
call, after a reshape. Comments lie; assertions don't.

## Autograd in one figure

PyTorch builds a **directed acyclic graph** on the fly as you
run operations on tensors that require gradients. Each node
records a `grad_fn`. When you call `.backward()` on a scalar,
PyTorch walks the graph backwards, applying the chain rule at
each node, and accumulates gradients into the `.grad`
attribute of every leaf tensor with `requires_grad=True`.

The minimum example:

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x + 1                 # y = 7, grad_fn=AddBackward0
loss = (y - 5) ** 2           # loss = 4
loss.backward()
print(w.grad)                 # 12.0  (= 2 * (y - 5) * x = 2 * 2 * 3)
```

Three properties to internalise:

1. **Gradients accumulate.** If you call `.backward()` twice,
   `w.grad` sums the two runs. The training loop resets grads
   at the start of each step (chapter 03).
2. **Only leaf tensors get `.grad`.** A tensor you constructed
   with `requires_grad=True` is a leaf. An intermediate
   `y = w * x` is not — it has a `grad_fn` but no persistent
   `.grad`. Set `y.retain_grad()` if you need it (usually for
   debugging).
3. **`.backward()` needs a scalar.** For vector-valued outputs
   you pass a `gradient=` argument (rare) or reduce to a scalar
   (the loss).

### `requires_grad` and how it propagates

Any operation whose output depends on a tensor with
`requires_grad=True` also has `requires_grad=True`. Model
parameters (`nn.Parameter`) default to it. Your input data
does not.

```python
x = torch.randn(4, requires_grad=False)   # data
w = torch.randn(4, requires_grad=True)    # parameter
y = (x * w).sum()
y.requires_grad     # True — inherited from w
```

### Turning autograd off

Autograd has a memory and compute cost — the graph is stored
until `.backward()` runs. Turn it off when you don't need
gradients:

```python
with torch.no_grad():
    preds = model(x_val)               # no graph built
    val_loss = loss_fn(preds, y_val)   # no graph built
```

Or, more concise for inference:

```python
@torch.inference_mode()
def predict(model, x):
    return model(x)
```

`torch.inference_mode` is stricter than `torch.no_grad` (it
also disables version counters); prefer it for pure inference.

**Rule.** Any code path that does not call `.backward()`
should be inside `torch.no_grad()` or `@torch.inference_mode()`.
Validation, test, and inference — all three.

### Detaching

`t.detach()` returns a tensor sharing the same data but with
no autograd history. Use it to break the graph when passing a
value into something that shouldn't be differentiated
through:

```python
running_mean = 0.9 * running_mean + 0.1 * batch_mean.detach()
```

Without `.detach()` the graph would grow across iterations —
memory leak, subtle wrong behaviour.

## Devices

PyTorch supports several accelerators. The idiomatic pattern:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device)
x = x.to(device)
```

For Apple Silicon, `mps`; for Intel GPUs, `xpu`; for CPU-only,
`cpu`. The rest of your code stays device-agnostic.

Every tensor must be on the same device as the parameters of
the layer it flows through. Mixing produces a clear error:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

The two moments this happens:

1. **You forgot to `.to(device)` a new tensor** (a manually
   constructed one-hot, a loaded checkpoint, a
   `torch.zeros(...)` inside the forward pass).
2. **You mixed a NumPy array and a CUDA tensor** at the
   boundary between your data loader and the model.

Fix: move both sides to the same device explicitly at the top
of every forward pass and every train step.

### `.to(device)` copies; `.to(device, non_blocking=True)` can overlap

For CPU-to-GPU transfers of pinned-memory tensors,
`non_blocking=True` lets the copy overlap with computation:

```python
x = batch["x"].to(device, non_blocking=True)
```

`DataLoader(..., pin_memory=True)` (chapter 04) is the
precondition for this to actually help.

## Dtypes

Four dtypes cover 95% of deep learning:

- **`torch.float32`** (default). The reference precision for
  everything unless you have chosen otherwise.
- **`torch.float16`**. Half precision. Range is limited
  (~10⁻⁵ to 10⁴); overflow and underflow are real. Used for
  mixed-precision training on older GPUs; needs `GradScaler`
  (chapter 09).
- **`torch.bfloat16`**. Half precision with `float32`'s
  exponent range. No `GradScaler` needed. Preferred on
  Ampere+ NVIDIA GPUs, TPUs, and newer AMD/Apple hardware.
- **`torch.int64`**. Default integer type. Class labels for
  classification (`CrossEntropyLoss` wants `int64`).

Casts happen with `.to(dtype)` or the type-shortcut methods
(`.float()`, `.half()`, `.bfloat16()`, `.long()`).

Two dtype-adjacent bugs to know now:

- **`CrossEntropyLoss(preds, targets)`** expects `preds` in
  `float32` (or `float16`/`bfloat16` under autocast) and
  `targets` in `int64`. A `float` target here silently does
  the wrong thing on newer versions or raises on older ones.
- **`BCEWithLogitsLoss`** expects both to be floats of the
  same dtype, `targets` in `[0.0, 1.0]`.

Chapter 09 makes the mixed-precision story concrete. For
now: use `float32` everywhere and change one thing at a time.

## Random seeds and determinism

Neural network training is stochastic. Two knobs:

```python
import random, numpy as np, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)                  # CPU + CUDA
torch.cuda.manual_seed_all(0)         # explicit for multi-GPU

# For strict determinism (slower):
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Determinism is not free.** cuDNN's non-deterministic
algorithms are faster; `torch.use_deterministic_algorithms(True)`
disables them and slows training by 10–30% depending on the
workload. Use for reproducibility runs (a regression test,
a bug hunt); not for production training.

Also: **seeding does not eliminate all sources of variation**
across hardware or PyTorch versions. Same code + same seed on
the same GPU with the same PyTorch version → reproducible.
Anything else — only approximately.

## nn.Parameter, nn.Module, and where state lives

The two building blocks:

- **`nn.Parameter`** — a `Tensor` marked as a learnable
  parameter. Automatically has `requires_grad=True`.
  Registered with the parent `Module`.
- **`nn.Module`** — a container for parameters and
  sub-modules. Provides `.parameters()`, `.state_dict()`,
  `.to(device)`, `.train()`, `.eval()`.

You will build custom modules in chapter 05. The one idiom
to know now:

```python
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
```

Anything assigned to `self.something` that is either an
`nn.Parameter` or another `nn.Module` is auto-registered.
Anything else (a Python list, a plain tensor) is not — for
those you need `nn.ParameterList` / `nn.ModuleList`, or the
parameters will be invisible to `optimizer` and
`state_dict()`.

## Train mode vs eval mode

`model.train()` and `model.eval()` toggle behaviours of
specific layers:

- **Dropout** (chapter 08) is active in `train()`, no-op in
  `eval()`.
- **BatchNorm / LayerNorm behaviour differs** —
  BatchNorm uses running statistics in `eval()` and batch
  statistics in `train()`; LayerNorm is unchanged.

Forgetting to call `.eval()` before validation is one of the
top-five deep-learning bugs. Fold it into the training loop
(chapter 03) so it happens automatically.

`.train()` / `.eval()` do **not** affect autograd. That is
what `torch.no_grad()` is for. The two are orthogonal and
you use both.

## The end-to-end minimum

Everything in this chapter, condensed:

```python
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

model = nn.Linear(4, 1).to(device)
opt   = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(32, 4, device=device)
y = torch.randn(32, 1, device=device)

# One training step
model.train()
opt.zero_grad()
pred = model(x)
loss = ((pred - y) ** 2).mean()
loss.backward()
opt.step()

# One validation pass
model.eval()
with torch.no_grad():
    val_pred = model(x)
    val_loss = ((val_pred - y) ** 2).mean()
```

Chapter 03 unpacks that training step into the loop you will
reuse for the rest of the module.

## Pitfalls collected

1. **Forgetting `optimizer.zero_grad()`.** Gradients
   accumulate. Not calling this means your second step's
   gradient is the sum of steps 1 and 2 — silently wrong.
2. **Doing validation with autograd on.** Doubles memory,
   builds a graph you throw away. Wrap in `torch.no_grad()`
   or `@torch.inference_mode()`.
3. **Doing validation with the model in `train()` mode.**
   Dropout keeps dropping; BatchNorm updates running stats
   from validation data. Val metrics look wrong; models
   silently degrade. `.eval()` before, `.train()` after.
4. **Broadcasting bugs from a rogue trailing dimension.**
   `(4,)` and `(4, 1)` broadcast to `(4, 4)`. If a loss
   suddenly explodes for no reason, check shapes first.
5. **Reading `.grad` on a non-leaf tensor.** It is `None`
   unless you called `.retain_grad()`.
6. **Mixing devices.** A `torch.zeros(...)` constructed
   inside `forward` lives on CPU by default. Use
   `torch.zeros(..., device=x.device)`.

## Summary

- Tensors are NumPy arrays with device and dtype awareness
  and (optionally) autograd tracking.
- Autograd builds the graph as you run ops, and
  `.backward()` walks it in reverse. Gradients accumulate on
  leaf tensors; validation code goes inside `torch.no_grad()`.
- Devices and dtypes must match at every op boundary; the
  errors are clear when they don't.
- `nn.Module` is the container; `nn.Parameter` is the leaf.
  Anything you want the optimizer to see must be one or the
  other (or inside `nn.ModuleList` / `nn.ParameterList`).
- `.train()` / `.eval()` toggle layer behaviour; they are
  orthogonal to autograd.

## Where this goes next

Chapter 03 assembles these primitives into the canonical
training loop. Every subsequent chapter reuses that loop.
