# 09 — Mixed precision and training performance

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 02, 03, 08.

## Motivation

A model that trains in three days when it could train in six
hours is a bug in your engineering, not a fundamental
limitation. This chapter covers the two big wins available
to any PyTorch training loop:

1. **Mixed precision** — run most of the forward and
   backward in 16-bit floats. On a modern NVIDIA GPU this
   gives 2–3× speedup and halves memory use with negligible
   accuracy impact.
2. **General throughput hygiene** — DataLoader tuning
   (chapter 04 revisited), gradient accumulation for a
   larger effective batch, `torch.compile`, and a short tour
   of the profiler.

Both make the loop from chapter 03 faster without changing
what it teaches. Neither is a substitute for training an
efficient model in the first place — but a well-tuned
training run costs less, iterates faster, and lets you try
more experiments.

## The three float precisions

- **`float32` (fp32).** 8-bit exponent, 23-bit mantissa. The
  default. Big dynamic range, plenty of precision. Uses
  4 bytes per number.
- **`float16` (fp16).** 5-bit exponent, 10-bit mantissa.
  Narrow range (~10⁻⁵ to 10⁴); prone to underflow (small
  gradients round to zero) and overflow. Uses 2 bytes.
- **`bfloat16` (bf16).** 8-bit exponent (same as fp32),
  7-bit mantissa. Same range as fp32; less precision. Uses
  2 bytes. Available on NVIDIA Ampere+ (RTX 30-series, A100
  and newer), TPUs, Apple Silicon.

The point of mixed precision: run **matmuls and convs**
(the compute-heavy ops) in fp16 or bf16 for speed, but keep
**loss accumulation, LayerNorm statistics, and gradient
updates** in fp32 for numerical stability. PyTorch's
`torch.autocast` context manager does this automatically for
you.

## bf16 vs fp16: choose bf16 when available

Both are 16-bit. The difference is the exponent range.

- **bf16** has fp32's range, so gradients don't underflow.
  You can drop it into an existing loop with no other
  changes. **No `GradScaler` needed.**
- **fp16** has a narrower range. Small gradients can
  underflow to zero. To prevent that, you multiply the loss
  by a large factor (`GradScaler`) before backward, then
  divide the gradients by the same factor before the
  optimiser step.

**If your hardware supports bf16, use bf16.** It is simpler
and less error-prone. If not (older NVIDIA GPUs), use fp16
with `GradScaler`.

Support quick reference:

| Hardware | fp16 | bf16 |
|---|---|---|
| NVIDIA V100 (Volta) | ✓ | ✗ |
| NVIDIA T4 / RTX 20-series (Turing) | ✓ | ✗ |
| NVIDIA A100 / A40 / RTX 30-series (Ampere) | ✓ | ✓ |
| NVIDIA H100 (Hopper) | ✓ | ✓ (+ fp8) |
| Apple Silicon (MPS) | ✓ | ✓ |
| CPU | limited | limited |

## The autocast + GradScaler pattern (fp16)

The full chapter 03 loop, modified for fp16 AMP:

```python
import torch
from torch import nn

device  = "cuda"
scaler  = torch.amp.GradScaler("cuda")           # only needed for fp16
model   = build_model().to(device)
opt     = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(xb)                    # runs in fp16
            loss   = loss_fn(logits, yb)          # loss stays in fp32

        scaler.scale(loss).backward()             # scaled backward
        scaler.unscale_(opt)                      # for clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)                          # unscales + steps
        scaler.update()                           # adjusts scale factor
```

Three things to note:

- **`autocast` is a context manager.** Everything inside runs
  with the mixed policy: ops that benefit from fp16 use fp16,
  ops that need fp32 (LayerNorm reductions, softmax, some
  losses) stay fp32. PyTorch has an op-by-op allow-list.
- **`scaler.scale(loss).backward()`** multiplies the loss by
  the current scale factor and runs backward. Gradients come
  out large enough that fp16 can represent them.
- **`scaler.unscale_(opt)` before `clip_grad_norm_`** so the
  clipping threshold applies to real gradients, not scaled
  ones. Then `scaler.step(opt)` divides by the scale factor
  and calls `opt.step()` — but only if no gradient contained
  NaN/inf. If it did, the step is skipped and the scaler
  reduces the factor for next time.

## The bf16 pattern

Simpler — no scaler.

```python
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(xb)
        loss   = loss_fn(logits, yb)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

That's the whole change. `bfloat16` has enough range that
gradients don't underflow; loss scaling would only add
overhead.

## What autocast actually does

Inside the `autocast` context, PyTorch checks each op against
a policy:

- **Ops that go to the low-precision dtype:** matmul, conv,
  linear, RNN, attention — the compute-bound ops.
- **Ops that stay fp32:** softmax, log, batch/layer norm's
  running statistics, most losses, sum/mean reductions.
- **Ops that follow the input's dtype:** elementwise,
  concat, split.

If a fp32 tensor and a fp16 tensor meet in an op, the fp32
one is downcast (for compute-bound ops) or the fp16 one is
upcast (for stability-sensitive ops). Most of the time this
is invisible.

Two consequences worth knowing:

- **Casts happen automatically at the boundary of `autocast`.**
  Anything you do *outside* the context is fp32. Anything
  passed *out* of the context retains its autocast dtype
  unless you upcast explicitly.
- **`inference_mode` + `autocast` works for prediction too.**
  Use both for inference to get the speedup with none of the
  gradient concerns.

## Memory: activation checkpointing

Even with bf16, a large model can OOM on a small GPU because
of **activations** — the tensors saved during forward for
use in backward. Activation checkpointing trades compute for
memory: it discards intermediate activations and recomputes
them during backward.

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Recompute this block during backward instead of storing activations:
    x = checkpoint(self.block1, x, use_reentrant=False)
    x = checkpoint(self.block2, x, use_reentrant=False)
    return self.head(x)
```

Typical trade-off: 20–30% slower per step, 30–70% less memory.
Worth it when the alternative is a smaller batch size or
OOM. Standard practice for training large Transformers.

## Gradient accumulation for larger effective batches

You want a batch of 256 but only 64 fits in memory. Solution:
compute the gradient of four batches of 64 and add them
before stepping.

```python
accumulation_steps = 4
opt.zero_grad()
for step, (xb, yb) in enumerate(train_loader):
    xb, yb = xb.to(device), yb.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = loss_fn(model(xb), yb) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
```

Two subtle rules:

- **Divide the loss by `accumulation_steps`**, or the
  effective LR scales up with the number of accumulation
  steps.
- **Don't call `scheduler.step()` inside the mini-batch
  loop** if you're using a per-step scheduler — only after
  the real step, when `opt.step()` fires.

The effective batch size is `batch_size × accumulation_steps
× num_ranks` (in a distributed setting). Track this
explicitly; hyperparameters like LR depend on it.

## `torch.compile`: the free speedup (mostly)

PyTorch 2.0 introduced `torch.compile`, which traces your
model into a graph and compiles it into fused kernels via
TorchInductor.

```python
model = torch.compile(model)
```

Or, more finely:

```python
model = torch.compile(model, mode="reduce-overhead")  # for small models
model = torch.compile(model, mode="max-autotune")     # slower compile, faster runs
```

Wins:

- 10–30% speedup on the training step for typical models.
- Kernel fusion reduces memory traffic.
- Combines with autocast — mostly.

Caveats:

- First call is slow (compiles).
- Shape changes trigger recompilation. Prefer fixed batch
  size and padded sequence length.
- Some custom `nn.Module` code has "graph breaks" — fine, just
  slower. `TORCH_LOGS="graph_breaks"` shows them.
- Debug with compile off first, then enable.

Use it once your training loop is stable. It rewards being
turned on in the final run and off during debugging.

## Fast attention kernels

Chapter 07's `F.scaled_dot_product_attention` picks the
fastest attention kernel available: Flash Attention 2 on
Ampere and Hopper, memory-efficient attention elsewhere,
math attention as a fallback. You get 2–4× speedup on
attention-heavy workloads (Transformers) with no code
change beyond calling that function instead of writing
attention by hand.

If you must, force a backend:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    out = F.scaled_dot_product_attention(q, k, v)
```

Rare in practice; the default selection is usually right.

## DataLoader tuning revisited

Chapter 04 covered the mechanics; the performance-oriented
summary:

- **`num_workers=4`–`8`** for moderate CPU-side transforms;
  higher if the CPU has more cores and you're GPU-bottlenecked.
- **`pin_memory=True`** on CUDA. Always.
- **`persistent_workers=True`.** Avoids the per-epoch worker
  restart tax.
- **`prefetch_factor=2`–`4`.** Each worker prefetches this
  many batches. Raise if the model is fast and the pipeline
  is I/O-bound.
- **`non_blocking=True` on `.to(device)`.** Pairs with
  `pin_memory=True` to overlap transfers with compute.

The failure mode: GPU utilisation (`nvidia-smi`) hovers below
90% while training. That means the data pipeline is the
bottleneck. Fix the pipeline before touching model code.

## Profiling: seeing where time actually goes

The PyTorch profiler emits Chrome-trace-format JSON you can
open in `chrome://tracing`:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./traces"),
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
        if step >= 10:
            break
```

Load the resulting trace in TensorBoard (`tensorboard
--logdir=./traces`) or Chrome. Look for:

- **Long idle gaps between training steps** — data loader
  bottleneck.
- **Kernel launches taking longer than the kernel runs** —
  many small ops; `torch.compile` will help.
- **Softmax, LayerNorm, or GELU dominating** — try a fused
  implementation.

Do not profile prematurely — get the training loop correct
first. But when you *are* optimising, guessing is expensive
and profiles are cheap.

## A note on batch size and learning rate

Doubling the batch size approximately halves the number of
gradient updates per epoch. To maintain training dynamics
you usually need to increase the LR — the "linear scaling
rule" says double the LR when you double the batch size, up
to a point. Beyond a certain batch size (workload
dependent), scaling breaks and you need a **square-root
scaling rule** or a hyperparameter sweep.

For the exercises in this module, keep batch size modest
(64–256) and don't worry about scaling. It becomes important
in distributed training, which the MLOps track covers.

## Pitfalls collected

1. **fp16 without `GradScaler`.** Gradients underflow;
   training doesn't converge. Either add the scaler or
   switch to bf16.
2. **Clipping scaled gradients.** Threshold applies to the
   wrong values. Call `scaler.unscale_(opt)` first.
3. **`scaler.step(opt)` after `opt.step()` was manually
   called.** Double step; corrupts the optimiser state.
   Only `scaler.step(opt)` — never both.
4. **`torch.compile` before the loop is right.** Cryptic
   errors from the compiler when a simpler error message
   would have caught the bug in eager mode. Debug first;
   compile last.
5. **Gradient accumulation without dividing the loss.**
   Effective LR silently scales up.
6. **Autocast forgotten on validation.** Not a correctness
   issue, but you leave 2× on the table. Wrap the eval pass
   in the same autocast context.
7. **Profiler run on the first two steps.** You are
   measuring dataloader warmup, not steady-state training.
   Use the `wait` + `warmup` + `active` schedule.
8. **Assuming AMP works on CPU.** Autocast has a CPU path
   (`device_type="cpu", dtype=torch.bfloat16`) but the wins
   are modest. Do not expect the same 2–3× as GPU.

## Summary

- Mixed-precision training is the single biggest free
  performance win in modern deep learning. Use bf16 if
  hardware supports it; fp16 + `GradScaler` otherwise.
- Activation checkpointing trades compute for memory;
  useful for large models on small GPUs.
- Gradient accumulation lets you train with an effective
  batch larger than fits in memory — mind the LR and the
  loss-scale-by-accumulation-steps rule.
- `torch.compile` is a 10–30% speedup on a stable model;
  disable it while debugging.
- `F.scaled_dot_product_attention` picks the best available
  attention kernel; use it instead of writing attention by
  hand for anything production-ish.
- Tune DataLoader arguments and profile with
  `torch.profiler` before optimising model code.

## Where this goes next

Chapter 10 covers the vocabulary and delegation map for
transformer LLMs — the specialist territory this module
deliberately does not enter.
