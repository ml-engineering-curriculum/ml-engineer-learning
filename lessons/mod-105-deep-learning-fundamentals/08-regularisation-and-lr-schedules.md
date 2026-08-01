# 08 — Regularisation and learning-rate schedules

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 03, 05–07.

## Motivation

Chapters 05–07 gave you three architectures. This chapter
covers the knobs that turn each of them from "trains" into
"trains well" — the regularisation techniques that stop
the model from memorising the training set and the
learning-rate schedules that keep optimisation moving in the
right direction.

Every regulariser here has a single job: **reduce the gap
between training and validation loss** without giving up
much validation performance. The default posture is to add
them cautiously and remove them if they don't help. More
regularisation is not automatically better.

Every scheduler has a single job: **reduce the learning rate
over the course of training** so that early exploration
gives way to late-training convergence. A fixed learning
rate usually works; a schedule usually works a little
better; the right schedule for the wrong LR still fails.

## Weight decay

Weight decay adds a penalty on the size of the parameters:

```
loss += λ * Σ ‖w‖²
```

The optimiser update becomes `w ← (1 - lr·λ) w - lr · grad`.
Large λ shrinks weights aggressively; small λ has little
effect. Standard values: **`1e-4` to `1e-2`** depending on
architecture and dataset size.

The subtlety: **`Adam(weight_decay=...)` and
`AdamW(weight_decay=...)` implement it differently.** In
`Adam`, the weight-decay term gets folded into the gradient
before Adam's adaptive scaling, which effectively *reduces*
weight decay for parameters with large gradients. In
`AdamW`, weight decay is applied to the parameter update
*after* Adam's scaling — decoupled. `AdamW` is what the
paper "Decoupled Weight Decay Regularization" (Loshchilov &
Hutter, 2019) argued for and what every modern paper uses.

**Use `AdamW` when you want weight decay.** For SGD, the
built-in `weight_decay` argument is fine.

Two more subtleties:

- **Don't decay biases, LayerNorms, or embeddings.** Only
  matrix weights. The convention is to build two parameter
  groups:

  ```python
  decay, no_decay = [], []
  for name, p in model.named_parameters():
      if not p.requires_grad:
          continue
      if p.ndim < 2 or "bias" in name or "norm" in name.lower():
          no_decay.append(p)
      else:
          decay.append(p)

  optimizer = torch.optim.AdamW(
      [
          {"params": decay,    "weight_decay": 1e-2},
          {"params": no_decay, "weight_decay": 0.0 },
      ],
      lr=1e-3,
  )
  ```

- **Weight decay is not the same as L2 regularisation** on
  `SGD` with momentum. Close in vanilla SGD; different once
  momentum is involved. The distinction matters if you are
  trying to reproduce a paper.

## Dropout

Dropout zeros out a random subset of activations during
training (each is set to 0 with probability `p`, and the
remaining ones are scaled by `1 / (1 - p)` so the expected
sum is preserved). At eval time, dropout is a no-op.
`model.eval()` turns it off — one more reason to always
switch modes before validation.

```python
nn.Dropout(p=0.2)                    # generic
nn.Dropout2d(p=0.2)                  # zeros entire feature channels in a CNN
```

Placement conventions:

- **MLP:** between fully-connected layers. `p=0.2`–`0.5`
  typical. Do **not** put it right before the output layer
  in a classification head — it degrades calibration.
- **CNN:** less useful than in MLPs (BatchNorm + data
  augmentation do most of the regularising). A modest
  `nn.Dropout(0.2)` in the classifier head is common; heavy
  dropout in convolutional layers is not.
- **Transformer:** three places — attention weights
  (`attn_dropout`), the feed-forward output (`ff_dropout`),
  and the residual output. `p=0.1` is the paper default.

Increase dropout if training loss is much lower than
validation loss. Decrease it if training loss cannot get
low.

## Early stopping

Stop training when validation loss has not improved for `N`
epochs. Cheap; simple; usually helps. The mechanic:

```python
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs > self.patience   # True → stop
```

Combine with best-checkpoint saving (chapter 03) so the
model you keep is the one at the best epoch, not the one at
the epoch you happened to stop on.

Two rules:

- **Patience should be > 1**, or you will stop on noise.
  `patience=5` is a safe default for short trainings;
  `patience=10` for longer.
- **Watch validation loss, not accuracy.** Loss is smoother
  and moves earlier.

## Label smoothing

Instead of a one-hot target (all mass on the correct class),
distribute a small amount of mass to the other classes:

```
y_smooth = (1 - ε) · one_hot + ε / C
```

Prevents the network from becoming over-confident and often
helps generalisation. Built into `CrossEntropyLoss`:

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Common ε values: 0.1. Widely used in image classification;
less common in text, where it can hurt.

## Data augmentation as regularisation

Chapter 06 introduced image augmentation. It is, in effect,
the strongest regulariser in vision — a random crop is worth
several percentage points of dropout and never over-
regularises. Extend this thinking beyond images:

- **Text:** random deletion, back-translation, synonym
  replacement (moderate returns; often more trouble than
  worth).
- **Audio:** SpecAugment (mask random frequency bands and
  time windows in the spectrogram). Simple and effective.
- **Tabular:** MixUp (interpolate two examples in feature
  space and their labels) — surprisingly effective on some
  problems. Add cautiously.

The rule: **more data always beats more regularisation.**
Augmentation is a way to synthesise more data.

## Choosing a learning rate

The learning rate is the single most important
hyperparameter. Two ways to pick one:

**Grid.** Try `1e-2`, `1e-3`, `1e-4`, `1e-5`. Watch the
training loss for the first few hundred steps. The best
value is the largest LR at which loss decreases smoothly.

**Learning-rate range test** (Smith, 2017). Increase LR
linearly from very small to very large over a few hundred
batches while recording loss. Plot loss vs LR (log scale) —
loss will drop, stabilise, and then explode. Pick an LR
about 10× smaller than the one where loss started climbing.
Chapter's exercise uses this.

Reasonable starting points if you cannot test:

| Model | Optimizer | LR |
|---|---|---|
| Small MLP | AdamW | 1e-3 |
| Small CNN (from scratch) | AdamW | 1e-3 |
| Small Transformer (from scratch) | AdamW | 3e-4 |
| Fine-tuning BERT / ViT / ResNet | AdamW | 2e-5 to 5e-5 |
| Fine-tuning last layer only | AdamW | 1e-3 |

Fine-tuning uses a smaller LR because the pre-trained
weights are already close to a good solution.

## The four schedules you will actually use

### 1. Constant

`torch.optim.lr_scheduler` has no "constant" — just skip the
scheduler. Fine for many small problems.

### 2. StepLR / MultiStepLR

Drop the LR by a factor at fixed epochs.

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[60, 120, 160], gamma=0.2
)
```

Simple, controllable, effective on classic image-
classification training regimes. Call `scheduler.step()`
once per epoch.

### 3. Cosine annealing

Smoothly anneal the LR from the initial value to (near) zero
following a cosine curve over the total training run.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```

Widely used in modern training. Combines well with a short
warm-up prefix (see below). Call per-epoch.

### 4. Warmup + linear or cosine decay

The Transformer standard. Ramp LR from 0 up to the peak
over the first few percent of training, then decay to (near)
zero.

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine(step: int, warmup_steps: int, total_steps: int):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    import math
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda s: warmup_cosine(s, warmup_steps=1000, total_steps=len(train_loader)*epochs),
)
```

Call **per step** (inside the batch loop), not per epoch.
This is what almost every open-source LLM training script
does; when in doubt, use it.

`torch.optim.lr_scheduler.OneCycleLR` is a batteries-
included alternative — one call sets warmup, peak, and
cosine decay in a single scheduler.

### Where the `.step()` goes

Same footgun as chapter 03:

- **Per-epoch schedulers** — `.step()` at the end of the
  epoch, after the validation pass.
- **Per-step schedulers** — `.step()` right after
  `optimizer.step()`, inside the batch loop.

Read the scheduler's docstring. Calling in the wrong place is
either off-by-one (harmless) or off-by-a-factor-of-100
(silently ruinous).

## Gradient clipping revisited

Chapter 03 introduced clipping; it deserves a mention here
as regularisation. `torch.nn.utils.clip_grad_norm_(params,
max_norm)` rescales the flattened gradient to have at most
`max_norm` L2 norm.

- **Transformers:** `max_norm=1.0`. Nearly always on.
- **CNNs from scratch:** usually not needed.
- **Fine-tuning:** `max_norm=1.0` as insurance.

Do not clip too aggressively (say, `max_norm=0.01`) — you
will silently zero out most learning.

## Regularisation, applied

A CIFAR-10 CNN training recipe assembled from this chapter
and chapters 03 and 06:

```python
model = SmallCNN(num_classes=10)

# Weight decay on matrix params only
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]

optimizer = torch.optim.AdamW(
    [
        {"params": decay,    "weight_decay": 5e-4},
        {"params": no_decay, "weight_decay": 0.0 },
    ],
    lr=3e-3,
)

# Cosine schedule with 5-epoch warmup embedded via OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.1,
)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
early   = EarlyStopping(patience=15)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    print(f"epoch {epoch} val_loss={val_loss:.4f} acc={val_acc:.4f}")
    if early.step(val_loss):
        print("early stopping"); break
```

That is what a well-regularised CNN training run looks like.
Change dataset, change model, change LR — the shape does not
change.

## When more regularisation is *not* the answer

Overfitting is a real problem; so is under-fitting, and adding
regularisation makes the latter worse. Diagnose first:

- **Training loss high, val loss high.** Under-fitting. The
  cure is a bigger model, longer training, better features —
  not more regularisation.
- **Training loss low, val loss high.** Over-fitting.
  Regularisation is the right tool.
- **Both curves flat.** The optimiser is not learning.
  Chapter 03's debugging checklist first, not
  regularisation.

## Pitfalls collected

1. **`Adam(weight_decay=...)` where `AdamW` was wanted.**
   Silently under-regularises. Use `AdamW`.
2. **Decaying biases and LayerNorm parameters.** Slightly
   hurts performance; convention is to exclude them.
3. **`scheduler.step()` in the wrong loop position.** Off by
   an epoch or off by a factor of 100. Read the docstring.
4. **Early stopping on `val_acc` instead of `val_loss`.**
   Accuracy is coarser and noisier; loss moves earlier.
5. **Dropout right before the output layer of a classifier.**
   Distorts the logit scale; hurts calibration.
6. **Cosine schedule but training for more epochs than
   `T_max`.** LR goes to `eta_min` and stays there — the last
   half of training does nothing.
7. **Zero warmup on a Transformer.** Loss diverges before it
   ever learns.
8. **Adding regularisation because the accuracy is bad,
   when the diagnosis is under-fitting.** Wrong tool.

## Summary

- Use `AdamW` with weight decay `1e-4`–`1e-2` on matrix
  weights only. Exclude biases, norms, and embeddings from
  decay.
- Dropout `0.2–0.5` between MLP layers; `0.1` in
  Transformers; sparingly in CNNs (augmentation is stronger
  there).
- Early stop on validation loss with `patience ≥ 5`; save
  the best checkpoint.
- Learning-rate range test to pick an LR; safe defaults in
  the table above.
- Four schedules: constant, step, cosine, warmup+cosine.
  The last is the default for Transformers.
- Regularise only when the diagnosis is over-fitting.
  Otherwise fix the loop, the data, or the model capacity.

## Where this goes next

Chapter 09 covers mixed-precision training and the general
"make this train faster" toolkit — the other direction
you can push once training is stable.
