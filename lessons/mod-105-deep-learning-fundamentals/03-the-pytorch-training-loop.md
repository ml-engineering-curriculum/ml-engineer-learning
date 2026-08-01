# 03 — The PyTorch training loop from scratch

> **Reading time:** ~35 minutes.
> **Prerequisite chapter:** 02 (tensors, autograd, devices).

## Motivation

The training loop is the beating heart of every deep learning
project. Once you can write it without looking, everything
that follows in this module — MLPs, CNNs, Transformers,
regularisation, mixed precision — is a substitution into one
of its slots.

Frameworks on top of PyTorch (Lightning, `accelerate`,
`transformers.Trainer`, Fabric) all *wrap* this loop. They
save typing, standardise defaults, and add features like
multi-GPU and logging. They also hide the four lines that
matter. This chapter's job is to make sure those four lines
are permanent knowledge before you reach for a wrapper.

The four lines:

```python
optimizer.zero_grad()
loss = loss_fn(model(x), y)
loss.backward()
optimizer.step()
```

Everything else in this chapter is scaffolding around that
core.

## The canonical loop

Here is a complete, correct, minimal training loop that
trains a two-layer network on random data. Read it slowly.
Every subsequent snippet in this chapter is a modification
of this one.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# 1. Data
x = torch.randn(1024, 10)
y = torch.randint(0, 3, (1024,))
train_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

# 2. Model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
).to(device)

# 3. Loss + optimizer
loss_fn  = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Loop
for epoch in range(5):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item() * xb.size(0)
    print(f"epoch {epoch} — loss {running / len(train_loader.dataset):.4f}")
```

Copy this. Run it. Modify it. If nothing else in this chapter
sinks in, this snippet is the deliverable.

## The four lines, one by one

### `optimizer.zero_grad()`

Gradients accumulate in `.grad` across `.backward()` calls
(chapter 02). Without this line, step 2's gradient is the sum
of step 1 and step 2's gradients. Every training run without
it silently learns the wrong thing.

Two flavours worth knowing:

```python
optimizer.zero_grad()                    # zeros the tensors
optimizer.zero_grad(set_to_none=True)    # sets .grad = None
```

`set_to_none=True` (the default in PyTorch 2.x) is slightly
faster because it skips the zero-fill and lets the next
`.backward()` allocate a fresh tensor. Keep the default.

### `loss = loss_fn(model(x), y)`

The forward pass. `model(x)` runs `model.forward(x)` (do not
call `.forward` directly — that skips hooks). `loss_fn`
reduces its inputs to a scalar.

Common losses and their expected shapes:

| Task | Loss | `preds` shape | `targets` shape | `targets` dtype |
|---|---|---|---|---|
| Multiclass classification | `CrossEntropyLoss` | `(B, C)` logits | `(B,)` class indices | `int64` |
| Binary classification (single output) | `BCEWithLogitsLoss` | `(B, 1)` or `(B,)` logits | same as preds | `float32` |
| Multi-label classification | `BCEWithLogitsLoss` | `(B, C)` logits | `(B, C)` `{0, 1}` | `float32` |
| Regression | `MSELoss` / `L1Loss` / `SmoothL1Loss` | `(B, *)` | `(B, *)` | `float32` |

Two things worth being explicit about:

- **`CrossEntropyLoss` combines log-softmax and NLL.** Do not
  apply `softmax` first — you will double-softmax and the
  loss will be numerically wrong. Pass raw logits.
- **`BCEWithLogitsLoss` combines sigmoid and BCE.** Same
  reason, same rule. Pass raw logits, not `sigmoid(logits)`.

### `loss.backward()`

Walks the autograd graph in reverse and populates `.grad` on
every leaf tensor. It expects `loss` to be a scalar.

If it isn't (a batch of losses without a reduction), reduce
first: `loss.mean().backward()`.

If nothing has `requires_grad=True` (you forgot to move the
model to the device *after* the optimiser was created, or
your model outputs a detached tensor), `loss.backward()`
raises `RuntimeError: element 0 of tensors does not require
grad and does not have a grad_fn`. That is a graph-shape
bug, not a loss bug.

### `optimizer.step()`

Applies the update rule: `w ← w - lr * w.grad` for SGD,
adaptive variants for Adam / AdamW. Reads `.grad`; does not
touch the model's forward pass.

Common optimisers:

- **`SGD(lr=..., momentum=0.9)`** — the classical choice.
  Requires more tuning; can be more stable at large batch.
- **`Adam(lr=...)`** — adaptive learning rate per parameter.
  Widely used, often the first thing to try.
- **`AdamW(lr=..., weight_decay=...)`** — Adam with
  decoupled weight decay. Prefer this over `Adam` when using
  weight decay. Chapter 08 explains why.

For most of this module: **`AdamW(lr=1e-3, weight_decay=1e-4)`**
is a safe default. Chapter 08 tunes the pieces.

## A robust loop with validation, metrics, and checkpointing

The minimal loop above trains but does not tell you anything
useful. A production-grade loop needs:

- **A validation pass** every epoch (with `model.eval()` and
  `torch.no_grad()`).
- **Metrics beyond loss** (accuracy, F1, whatever the task
  requires).
- **Best-model checkpointing** based on validation.
- **A logger** — even `print` — so you can see progress.
- **A device string** derived at the top and used
  consistently.
- **A reproducible seed**.

Here is the expanded version. This is the reference you will
copy for the rest of the module:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from copy import deepcopy


def train(
    model:       nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    *,
    loss_fn:     nn.Module,
    optimizer:   torch.optim.Optimizer,
    epochs:      int,
    device:      str,
    ckpt_path:   Path,
) -> dict:
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")

    for epoch in range(epochs):
        # ------- train -------
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ------- validate -------
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:03d} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f}"
        )

        # ------- checkpoint on val loss -------
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss":  val_loss,
                },
                ckpt_path,
            )

    return history


@torch.inference_mode()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    loss_fn:    nn.Module,
    device:     str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        total_loss    += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_correct / n
```

Six paragraphs of prose about what this loop is doing would be
noise — it is the same four lines from before, wrapped in
metric tracking and I/O. What is worth spelling out:

- **`@torch.inference_mode()`** on `evaluate` is stricter and
  faster than `torch.no_grad()` for pure inference.
- **`.item()`** pulls a Python float out of a scalar tensor.
  Doing this *inside* the loop rather than accumulating
  tensors avoids growing the autograd graph.
- **Accumulate weighted by batch size** (`.item() * xb.size(0)`)
  so the mean is correct when the last batch is smaller.
- **Checkpoint the optimiser too.** Resuming training without
  the optimiser state (Adam's momentum, `scheduler` state)
  produces a discontinuity in the loss curve. Save both.

## Reading a loss curve

Print or plot `train_loss` and `val_loss` per epoch. Four
common shapes and what each means:

1. **Both dropping smoothly, val slightly above train.**
   Healthy. Train longer or move on.
2. **Both flat from epoch 1.** Not learning. Check: is the
   learning rate reasonable? Is the loss connected to the
   parameters (are gradients non-zero)? Did you forget to
   move the model to the device?
3. **Loss diverges (nan or inf).** LR is too high, or an
   input has an inf / nan (division by zero in a feature
   transform is a common cause), or numerical instability
   (log of near-zero). LR by 10× lower first.
4. **Train drops, val stays flat or rises.** Overfitting.
   Chapter 08's territory — regularisation, more data, early
   stopping.

Print `loss.item()` every N steps if an epoch is long. Waiting
for an epoch to finish before knowing training is diverging
is wasted time.

## The "does this training loop learn?" smoke test

Before you trust a loop on real data, verify it can learn
*something*.

**Overfit a batch of 8.** Take a single batch, disable
regularisation, disable data augmentation, and train the
model until the loss goes to zero (or as close as the model
can). It should — if a network cannot memorise 8 examples,
it will not learn 8000. This test surfaces:

- Broken losses (wrong shape, wrong dtype, wrong reduction).
- Frozen parameters (a `requires_grad=False` you meant to
  toggle back on).
- Optimizer set up with the wrong `parameters()` iterator
  (empty, so nothing updates).
- Learning rate wildly wrong.

```python
xb, yb = next(iter(train_loader))
xb, yb = xb.to(device), yb.to(device)
for step in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(step, loss.item())
```

If the loss falls to near zero, your loop is wired up. If it
doesn't, fix the loop before you touch the dataset.

## Gradient clipping

For recurrent networks, Transformers, and any model with
occasional very large gradients, add:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

`clip_grad_norm_` rescales the flat gradient vector to have
norm at most `max_norm`. Prevents a single bad batch from
detonating the model. `1.0` is a sensible default for
Transformers; `10.0` for CNNs when you use it at all.

Use it. It is cheap and has saved many training runs.

## Where the learning rate scheduler goes

Chapter 08 covers scheduler *choice*; the mechanic is here.
The `scheduler.step()` call goes at the end of the loop —
after `optimizer.step()`. Two conventions:

**Per-epoch schedulers** (`StepLR`, `CosineAnnealingLR`):

```python
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()
```

**Per-step schedulers** (linear warm-up, `OneCycleLR`):

```python
for epoch in range(epochs):
    for xb, yb in loader:
        ...
        optimizer.step()
        scheduler.step()          # per-batch
```

Read the scheduler docstring; some are one, some the other.
Calling `scheduler.step()` in the wrong place is a common
bug — either the LR barely changes or it saturates in the
first minute of training.

## Where AMP goes

Chapter 09 covers mixed precision in detail. The two-line
modification to the four core lines is:

```python
scaler = torch.amp.GradScaler("cuda")
...
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(xb)
    loss   = loss_fn(logits, yb)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Left as a preview; chapter 09 explains why `GradScaler` is
needed for `float16` but not `bfloat16`, and when the whole
thing is worth the complexity.

## Distributed training, in one paragraph

`torch.nn.parallel.DistributedDataParallel` (DDP) wraps your
model and does gradient averaging across ranks. `torch.distributed`
provides the launcher. Fully-Sharded Data Parallel (FSDP) is
newer and shards optimiser state / parameters as well.
DeepSpeed and Colossal-AI are third-party toolkits.

**This module does not cover distributed training.** The
single-GPU loop in this chapter is the foundation; the
multi-GPU story is a track of its own. When you get there, the
four core lines in the middle of your loop will still be the
same four lines.

## Pitfalls collected

1. **Forgot `zero_grad()`.** Gradients accumulate; the model
   silently learns the sum.
2. **Applied `softmax` before `CrossEntropyLoss`.** Double
   softmax; nonsense loss.
3. **Called `model.forward(x)` directly.** Skips forward
   hooks. Always `model(x)`.
4. **Forgot `.eval()` before validation.** Dropout and
   BatchNorm behave differently; val metrics are wrong.
5. **Forgot `torch.no_grad()` / `inference_mode()` in
   validation.** Wasted memory and compute; potential OOM.
6. **Accumulated raw loss tensors** instead of `.item()`.
   The autograd graph accumulates too; slow-motion memory
   leak.
7. **Constructed the optimiser before moving the model.**
   `AdamW(model.parameters(), lr=1e-3)` before
   `model.to(device)` — the parameters the optimiser holds
   are the CPU copies; updates do nothing. Always
   `.to(device)` first.
8. **Non-scalar loss.** `loss_fn(...)` returned a vector.
   Reduce first (`.mean()`) or pass `reduction="mean"` in the
   loss.
9. **`scheduler.step()` before `optimizer.step()`.** PyTorch
   warns; the schedule is off by one.

## Summary

- The training loop is four lines: `zero_grad`, forward+loss,
  `backward`, `step`. Everything else is scaffolding.
- Wrap it with train/eval-mode toggles, a `no_grad`
  validation pass, per-epoch metrics, and checkpointing on the
  best validation score.
- Smoke-test with "overfit a batch of 8". If it can't, the
  loop is wrong before the data is.
- Gradient clipping, LR schedulers, and AMP are additions to
  the same loop. Chapters 08 and 09 wire them up.
- Distributed training wraps this loop; the loop itself does
  not change.

## Where this goes next

Chapter 04 covers the other half of a training run — the
`Dataset` / `DataLoader` / collate machinery that feeds
tensors into the loop.
