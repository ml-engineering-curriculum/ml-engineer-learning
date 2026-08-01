# exercise-01 — PyTorch training loop from scratch

> **Estimated effort:** 4 hours.
> **Companion chapters:** [02 — Tensors, autograd, devices](../02-tensors-autograd-and-devices.md), [03 — The PyTorch training loop](../03-the-pytorch-training-loop.md), [04 — Datasets, DataLoaders, collate](../04-datasets-dataloaders-and-collate.md), [05 — Building an MLP](../05-building-an-mlp.md).

## Objective

Build a **complete, from-scratch** PyTorch training loop
around a small MLP, on a real (small) dataset, without
using any high-level wrapper (no Lightning, no Fabric, no
`accelerate`, no `transformers.Trainer`). By the end you
should have:

- A `model.py`, a `data.py`, and a `train.py` that together
  train a small MLP to a reasonable metric.
- A test suite that verifies the loop is doing what you
  think it is (grads flow, checkpoints round-trip,
  early-stop kicks in).
- A `REPORT.md` with a loss curve, a training-vs-`mod-104`
  baseline comparison, and answers to the reflection
  questions.

The point of writing the loop by hand — even though every
framework has one — is that when it fails in a real project,
you know which of the four lines is wrong.

## Prerequisites

- Python 3.11+, PyTorch ≥ 2.1, `scikit-learn` (for the
  baseline and split), `pandas`, `pytest`.
- Chapters 02, 03, 04, and 05 of this module.
- Any of `matplotlib` / `plotly` / `seaborn` for the loss
  curve. Choose one.
- Ability to run PyTorch on your machine (CPU is fine —
  every part of this exercise finishes in under a couple of
  minutes on CPU).

## Dataset choice

Pick **one** of the following. All are small, tabular, and
have a well-known `mod-104` baseline you can compare against.

- **UCI Adult (Census Income).** Binary classification.
  `sklearn.datasets.fetch_openml("adult", version=2,
  as_frame=True)`. ~48k rows, 14 features (mix of numeric
  and categorical).
- **UCI Wine Quality.** Regression (predict quality score) or
  multi-class (bucket into 3 classes).
  `sklearn.datasets.fetch_openml("wine-quality-red",
  as_frame=True)`.
- **California Housing.** Regression.
  `sklearn.datasets.fetch_california_housing(as_frame=True)`.
- **Kaggle Titanic.** Small (~1k rows); simple binary
  classification if you want the whole thing to run in
  seconds.

Whichever you choose, name it in `REPORT.md` with the source
URL and the raw row / column count.

If you want a stretch, use the **output of `mod-103`
exercise-01** as your dataset. That is the most realistic
option.

## Problem statement

### Part A — Dataset and DataLoader (≈ 30 min)

In `data.py`, write:

```python
def load_split(
    seed: int = 0,
    test_frac: float = 0.2,
) -> tuple[TensorDataset, TensorDataset, dict]:
    """
    Load the chosen dataset, split into train/val, standardise
    numeric features on the training set only, and return
    (train_ds, val_ds, meta).

    `meta` includes at least: n_features, n_classes / task type,
    scaler mean and std, feature names.
    """
```

Requirements:

1. **Split first, transform second.** Fit the scaler on the
   train split only; apply to both. No leakage.
2. **Deterministic given `seed`.** Same seed → same rows in
   each split.
3. **Convert to `torch.Tensor` with explicit dtypes.**
   `float32` for features; `int64` for classification labels;
   `float32` for regression targets. Nothing `object`,
   nothing `float64`.
4. **Handle at least one categorical column** if your
   dataset has one. Either one-hot before tensoring, or set
   up an `nn.Embedding` in the model (chapter 05).

Write a companion:

```python
def make_loaders(train_ds, val_ds, batch_size: int = 128, seed: int = 0):
    """Return (train_loader, val_loader) with sensible defaults."""
```

Requirements:

- `shuffle=True` on train, `shuffle=False` on val.
- `pin_memory=True` (safe on both CPU and CUDA).
- `drop_last=True` on train, `drop_last=False` on val.
- A `torch.Generator` seeded from `seed` for shuffling
  reproducibility.

### Part B — The MLP (≈ 20 min)

In `model.py`, write:

```python
class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        out_features: int,
        dropout: float = 0.2,
    ): ...
```

Requirements:

1. Use `nn.Sequential` internally for the linear stack.
2. Between each pair of `Linear`s: `ReLU` (or `GELU`) and
   `Dropout(dropout)`.
3. **No activation on the final layer.** Output raw logits
   for classification; raw values for regression.
4. Provide a helper method or module-level function
   `count_parameters(model) -> int`.

### Part C — The training loop (≈ 90 min)

In `train.py`, write:

```python
def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    *,
    loss_fn:      nn.Module,
    optimizer:    torch.optim.Optimizer,
    epochs:       int,
    device:       str,
    ckpt_path:    Path,
    scheduler:    torch.optim.lr_scheduler._LRScheduler | None = None,
    early_patience: int | None = None,
    log_every:    int = 50,
) -> dict:
    """Return a history dict with per-epoch train/val metrics."""
```

Requirements:

1. **Move the model to `device` first**, then construct
   nothing new that depends on the parameters.
2. **`optimizer.zero_grad()`, forward, loss, `.backward()`,
   `optimizer.step()`.** In that order, once per batch. If
   `scheduler` is per-step (`OneCycleLR`, `LambdaLR`), call
   it in the batch loop; if per-epoch (`StepLR`,
   `CosineAnnealingLR`), call it after the epoch.
3. **Validation each epoch** with `model.eval()` and
   `@torch.inference_mode()`.
4. **Track:** `train_loss` (per epoch, weighted mean by
   batch size), `val_loss`, `val_metric` (accuracy for
   classification, MAE for regression). Return them in a
   dict.
5. **Save the best checkpoint** on validation loss. Persist:
   `{"epoch", "model": state_dict, "optimizer": state_dict,
   "val_loss", "meta": {...}}`. Load path is a `Path` arg.
6. **Optional early stopping.** If `early_patience` is set,
   stop when val loss has not improved for that many epochs
   (see chapter 08's mechanic).
7. **Log progress** — either to stdout every `log_every`
   batches, or via `logging`. Include: epoch, batch, running
   train loss.
8. **Ban high-level wrappers.** No Lightning, no Fabric, no
   `Trainer`. Only vanilla PyTorch.

Add a CLI entry point (`if __name__ == "__main__":`) that
runs a default configuration end to end. Argparse or `typer`
is fine.

### Part D — Smoke tests and unit tests (≈ 60 min)

In `tests/`, cover the following minimum set. Every test
should complete in under 5 seconds.

1. **`test_overfit_single_batch`.** Take a single batch of 8
   examples. Run 200 optimisation steps on it. Assert final
   loss is at least 100× smaller than initial loss. This is
   the loop-is-correct smoke test.
2. **`test_gradients_flow`.** After one training step,
   assert every trainable parameter has a non-`None`
   `.grad` and a non-zero gradient norm.
3. **`test_zero_grad_resets_grads`.** After
   `optimizer.zero_grad(set_to_none=True)`, assert every
   `.grad` is `None`.
4. **`test_model_train_eval_toggle_dropout`.** In
   `model.train()`, two forward passes of the same input
   produce different outputs (dropout active). In
   `model.eval()`, they are identical.
5. **`test_checkpoint_roundtrip`.** Train for 1 epoch,
   checkpoint, load into a fresh model and optimizer, verify
   `state_dict()`s match key-by-key and tensor-by-tensor.
6. **`test_early_stopping_triggers`.** Feed a monotonically
   increasing series of "val losses" through the early-stop
   helper; assert it fires after `patience + 1` bad epochs.
7. **`test_deterministic_split`.** `load_split(seed=0)`
   returns byte-identical splits across two calls.

Wire up `pytest` in `pyproject.toml`. `pytest -q` should
pass cleanly.

### Part E — Run and report (≈ 40 min)

Run the training end to end. Produce:

- `history.json` — the returned history dict, serialised.
- `plots/loss_curve.png` — a plot of train and validation
  loss per epoch. Y-axis on a log scale if the range is
  wide.
- A `mod-104` baseline: train a
  `HistGradientBoostingClassifier` (or `Regressor`) on the
  same split. Report the same metric.
- `REPORT.md` covering:
  - Dataset, source URL, raw row/column count.
  - Model architecture (hidden dims, dropout), parameter
    count.
  - Training setup: optimizer, LR, batch size, epochs,
    scheduler if any.
  - The loss curve (embed the PNG).
  - Metric comparison: MLP vs. `mod-104` baseline. Include
    training wall-clock time for each.
  - Answers to the reflection questions below.

## Starter guidance

- **Start with random data.** Before you load a real
  dataset, run the loop on `torch.randn(...)` for features
  and `torch.randint(...)` for labels. If the loop is wrong
  on random data, it will be wrong on real data. Once you
  see the loss drop, swap in the real dataset.
- **The "overfit a batch of 8" test is the single most
  useful debugger.** If it fails, do not touch the dataset
  code — the loop is wrong.
- **Print shapes early.** Add
  `assert xb.shape[1] == n_features` at the top of the
  train loop. It's a five-second habit that has saved
  everyone at least an hour once.
- **Use `AdamW(lr=1e-3, weight_decay=1e-4)` as your
  default.** Reach for other optimisers only if you have a
  reason.
- **Don't over-engineer the CLI.** One entry point, a few
  args, defaults that work. This is not the exercise for
  building a config framework.
- **Log the four numbers every epoch** — train loss, val
  loss, val metric, and epoch wall-clock time. That's
  enough to diagnose 90% of issues without dashboards.

## Acceptance criteria

- `python -m train` (or the equivalent invocation) runs end
  to end on a fresh environment and produces a checkpoint
  plus `history.json` and the loss curve.
- No high-level training wrapper is imported (grep for
  `lightning`, `accelerate`, `Trainer` — should return
  nothing).
- All seven listed tests pass. `pytest -q` is green.
- **The overfit-a-batch test brings loss below 1% of the
  initial loss.** If it doesn't, the loop is wrong; fix
  before submitting.
- The MLP reaches at least a *credible* validation metric
  on the chosen dataset (see suggestions per dataset
  below). Beating the `mod-104` baseline is not required —
  matching it is a fair outcome, and being clearly worse is
  a valid finding as long as `REPORT.md` discusses why.
- `REPORT.md` records the dataset, model, training config,
  loss curve, comparison against `mod-104` baseline, and
  the reflection answers.

**Reasonable validation-metric floors**, if you use the
suggested datasets. Not a leaderboard; just a sanity check
that training worked:

| Dataset | Metric | Rough floor |
|---|---|---|
| UCI Adult | Accuracy | > 0.80 |
| Wine Quality (regression) | MAE | < 0.75 |
| California Housing | MAE | < 0.60 (on target scaled to $100k units) |
| Titanic | Accuracy | > 0.75 |

If the MLP is well below the floor, first check for a bug in
the loop; only after that consider tuning.

## Reflection questions (answer in `REPORT.md`)

- Which of the four training-loop lines caused you the
  most trouble on first pass, and what was the actual bug?
- How did the MLP compare to the `mod-104` baseline in
  accuracy and in training time? What does that tell you
  about which model to reach for on this dataset?
- If you doubled the batch size, would you expect
  validation performance to improve, degrade, or stay the
  same, and why? (Chapter 08's hint counts.)
- Which test in Part D would have caught your worst bug the
  fastest? Which would have taken hours to save you?

## Stretch goals (optional)

- **Learning-rate range test** (chapter 08). Implement the
  Smith-style linear ramp from `1e-7` to `1e-1` over a few
  hundred steps; plot loss vs. LR. Confirm your default LR
  choice.
- **Cosine schedule with warmup** (chapter 08). Add
  `OneCycleLR` or a hand-rolled warmup-cosine schedule.
  Re-run and compare the loss curve.
- **Categorical embeddings.** If your dataset has
  categorical features, add `nn.Embedding` for each and
  concatenate with numerics (chapter 05). Compare against
  one-hot encoding.
- **Gradient clipping.** Add `clip_grad_norm_(model, 1.0)`
  before the optimiser step. Verify it makes no measurable
  difference on this small dataset; keep the code path
  for the CNN and Transformer exercises.
- **Confusion matrix and per-class recall** for
  classification. Add to the validation pass; save under
  `plots/`.

## What to hand in

```
exercise-01/
├── data.py                  # load_split, make_loaders
├── model.py                 # MLP, count_parameters
├── train.py                 # train(), evaluate(), CLI entry
├── tests/
│   ├── test_loop.py         # overfit-batch, grad flow, zero_grad
│   ├── test_model.py        # dropout toggle
│   ├── test_data.py         # deterministic split
│   └── test_checkpoint.py   # roundtrip, early stop
├── plots/
│   └── loss_curve.png
├── history.json
├── pyproject.toml
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.
