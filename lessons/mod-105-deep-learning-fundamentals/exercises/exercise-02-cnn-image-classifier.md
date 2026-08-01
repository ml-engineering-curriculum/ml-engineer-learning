# exercise-02 — CNN image classifier

> **Estimated effort:** 4 hours.
> **Companion chapters:** [06 — Convolutional neural networks](../06-convolutional-neural-networks.md), [08 — Regularisation and LR schedules](../08-regularisation-and-lr-schedules.md), [09 — Mixed precision and performance](../09-mixed-precision-and-training-performance.md).

## Objective

Train two CNN image classifiers on **CIFAR-10** and compare
them:

1. **A from-scratch small CNN** built with `nn.Conv2d`,
   `nn.BatchNorm2d`, `nn.MaxPool2d`, and residual blocks —
   ~300k–1M parameters, trained end to end.
2. **A fine-tuned pre-trained backbone** — a ResNet-18 or
   MobileNet-V3 from `torchvision`, with its ImageNet head
   replaced and either fully or partially fine-tuned.

The training loop from exercise-01 is reused unchanged. What
changes is the model, the data pipeline (image augmentation),
and the training tricks from chapters 08 and 09 (schedules,
weight decay, mixed precision).

By the end you should have:

- Both models training end to end and comparable metrics.
- A concrete answer to "how much of the win is from the
  pre-training vs. the architecture?"
- A `REPORT.md` with per-class accuracy, confusion matrix,
  augmentation ablation, and reflection.

## Prerequisites

- Chapters 06, 08, and 09.
- Exercise-01 completed (you will reuse its training loop).
- `torch`, `torchvision`, `pytest`, `matplotlib`.
- Enough disk for the CIFAR-10 download (~170 MB).
- A GPU is recommended but not required. On CPU, budget
  ~30 minutes per epoch for the small CNN and ~10 minutes
  per epoch for the fine-tune (batch of 128).

## Dataset

**CIFAR-10.** 60,000 32×32 RGB images across 10 classes
(50k train, 10k test). Well-understood, small enough for a
laptop, hard enough to actually differentiate models.

```python
from torchvision.datasets import CIFAR10
```

If you would rather use **CIFAR-100** (harder, 100 classes)
or **Imagenette** (a 10-class subset of ImageNet at higher
resolution), those work too — call out the choice in
`REPORT.md` and adjust the metric floors accordingly.

**Do not use ImageNet.** Too large for this exercise's
scope; the fine-tune part is what covers that territory.

## Problem statement

### Part A — Dataset and augmentation (≈ 30 min)

In `data.py`, build the CIFAR-10 pipeline with two
transforms:

```python
def build_transforms(train: bool) -> transforms.Compose: ...
def build_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]: ...
```

Requirements:

1. **Train transform** includes at minimum:
   `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`,
   `ToTensor()`, `Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)`.
   The CIFAR-10 stats are `mean=[0.4914, 0.4822, 0.4465]`,
   `std=[0.2470, 0.2435, 0.2616]`.
2. **Eval transform** contains only `ToTensor()` and
   `Normalize(...)`. No random crops. No flips.
3. `DataLoader` uses `pin_memory=True`,
   `persistent_workers=True`, `drop_last=True` on train,
   `drop_last=False` on val. `num_workers` set to something
   sensible (2–8).
4. **Do not download inside a test.** Cache the dataset
   once locally (`root="./data"`), or use `download=True`
   only on first run.

### Part B — Small CNN (≈ 45 min)

In `models/small_cnn.py`, implement the model from
chapter 06 or a variant of it. Requirements:

1. At least **6 convolutional layers** (three "stages" of
   two convs each), or the equivalent depth using residual
   blocks.
2. Uses `nn.BatchNorm2d` after each conv (with
   `bias=False` on the conv), followed by a `ReLU` or `GELU`.
3. Downsamples spatially at least twice (max pool or strided
   conv). Ends with `nn.AdaptiveAvgPool2d((1, 1))`.
4. Classifier head: `Flatten` → `Dropout` → `Linear` to 10
   classes.
5. Parameter count in the range **200k to 2M**.
   `count_parameters(model)` reports it. Include the number
   in `REPORT.md`.

### Part C — Pre-trained fine-tune (≈ 30 min)

In `models/pretrained.py`, build:

```python
def build_pretrained(num_classes: int = 10, freeze_backbone: bool = False) -> nn.Module: ...
```

Requirements:

1. Load a `torchvision` model with weights, e.g.
   `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`.
2. Replace the classifier head to output `num_classes`
   logits (`model.fc = nn.Linear(model.fc.in_features,
   num_classes)`).
3. If `freeze_backbone=True`, set `requires_grad=False` on
   all parameters *except* the new head.
4. Because CIFAR-10 images are 32×32, either **resize** to
   the model's expected input (224×224) in the transform, or
   pick a model that handles small inputs (MobileNet-V3
   small; ResNet-18 works with resizing).
5. **Use the checkpoint's normalisation stats** for the
   pre-trained model, not CIFAR-10's. Get them via
   `weights.transforms()` and use *those* (or their
   `mean` / `std`) — mismatched normalisation is a silent
   accuracy killer.

### Part D — Training with schedules and AMP (≈ 60 min)

In `train.py`, reuse exercise-01's loop with three
extensions:

1. **Optimizer:** `AdamW(lr=3e-3, weight_decay=5e-4)` for
   the small CNN; `AdamW(lr=1e-3, weight_decay=1e-4)` for
   fine-tuning; if you freeze the backbone, `lr=3e-3` for
   the head alone.
2. **Scheduler:** `OneCycleLR` with `pct_start=0.1`. Confirm
   `scheduler.step()` fires per batch, not per epoch.
3. **Mixed precision** (if you have a GPU). Wrap the
   forward + loss in `torch.autocast(device_type="cuda",
   dtype=torch.bfloat16)` (or `torch.float16` +
   `GradScaler` on older GPUs). Add gradient clipping
   (`clip_grad_norm_(model.parameters(), 1.0)`).
4. **Label smoothing 0.1** in `CrossEntropyLoss`.
5. **Log**: per-batch loss (running mean), per-epoch train
   loss, val loss, val accuracy, epoch wall-clock.

Train **each model** for a reasonable number of epochs:

- **Small CNN:** 40–60 epochs.
- **Pre-trained, frozen backbone:** 10–15 epochs.
- **Pre-trained, full fine-tune:** 10–20 epochs.

Save best checkpoint per model.

### Part E — Evaluation and comparison (≈ 45 min)

Once both models are trained:

1. **Overall test accuracy** on the CIFAR-10 test set.
2. **Per-class accuracy** — 10 rows, one per class.
3. **Confusion matrix** (10×10). Save as `confusion_*.png`.
4. **Top-5 accuracy** (optional, useful if you use
   CIFAR-100).
5. **Sample predictions** — a grid of 16 test images with
   the predicted label and the true label, coloured by
   correctness. Save as `samples_*.png`.
6. **Wall-clock time** to train each model, on your
   hardware. Note the hardware in `REPORT.md`.

### Part F — Augmentation ablation (≈ 20 min)

For the **small CNN only**, run three training conditions
and compare final val accuracy:

- **No augmentation.** Only `ToTensor()` and `Normalize()`,
  even on train.
- **Standard augmentation** (Part A).
- **Strong augmentation** — add `RandAugment` or
  `AutoAugment` from `torchvision.transforms.v2`. (Optional
  cutoff at "stronger than baseline"; you don't need to try
  five variants.)

Report the three accuracies and the difference in
overfitting (train − val loss at end).

### Part G — Tests (≈ 20 min)

In `tests/`, cover:

1. **`test_shapes`.** Feed a batch of `(4, 3, 32, 32)`
   through each model; assert the output is `(4, 10)`.
2. **`test_augmentation_train_only`.** Two eval-mode
   `__getitem__` calls on the same index return identical
   tensors. Two train-mode calls do not.
3. **`test_normalisation_stats_used`.** Check the transform
   Compose contains a `Normalize` with the expected mean and
   std for both models.
4. **`test_frozen_backbone_trainable_params`.** For the
   frozen-backbone version, only the final linear layer's
   parameters have `requires_grad=True`.
5. **`test_overfit_small_batch`.** Take 16 images, train
   the small CNN for 100 steps; assert loss drops
   substantially.

Keep the tests fast (< 30 seconds total).

## Starter guidance

- **Start with the pre-trained fine-tune.** It converges
  fastest, so you get a working end-to-end pipeline in an
  hour and then iterate on the from-scratch model with
  confidence. It also gives you a reference metric.
- **Do the "overfit a tiny batch" test on the from-scratch
  CNN.** If it can't overfit 16 images to near-zero loss,
  the model is wrong before the augmentation is.
- **Use `torchvision`'s `weights.transforms()` for the
  fine-tune model.** Do not hand-code the preprocessing —
  every pre-trained model expects specific stats and
  resizing.
- **`num_workers=0` if you are debugging.** Multi-worker
  DataLoaders are harder to debug when something goes
  wrong; only turn workers on once the pipeline runs.
- **On CPU, halve the epochs and note it in `REPORT.md`.**
  The exercise is about the experiment, not the leaderboard.
- **`torch.compile(model)` is worth trying for the small
  CNN** once training is stable. Might give you 10–30%
  speedup at the cost of a slow first step.

## Acceptance criteria

- Both models train end to end and produce checkpoints.
- The DataLoader pipeline distinguishes train from eval
  augmentation (test verifies).
- Small CNN reaches **≥ 85% test accuracy** on CIFAR-10 at
  the full augmentation setting. If well below that, first
  check the loop, augmentation, and normalisation.
- Pre-trained model (full fine-tune) reaches **≥ 90% test
  accuracy** on CIFAR-10.
- Per-class accuracy and confusion matrix are in the
  report.
- The augmentation ablation is done and the three numbers
  are in the report.
- All five listed tests pass.
- Mixed precision is used on CUDA hardware if available;
  if CPU-only, the AMP code path is written and tested
  (call the autocast context in a test to prove it runs)
  but not used for real training.
- `REPORT.md` records: chosen dataset, model architectures
  and parameter counts, training config, all metrics, the
  ablation table, the wall-clock times, and the reflection
  answers.

## Reflection questions (answer in `REPORT.md`)

- How much of the small-CNN → pre-trained gap is
  "capacity" and how much is "pre-training"? A rough answer
  is fine (the frozen-backbone number vs. the fine-tuned
  number is a data point).
- Look at the confusion matrix. Which pair of classes is
  the model most confused about, and does that surprise
  you? What would you try next to improve it?
- The augmentation ablation. How large was the gap
  between "no augmentation" and "standard"? Between
  "standard" and "strong"? What does the shape of the
  train-vs-val loss curves tell you about
  under-/over-fitting in each condition?
- If you had to serve one of these models in production
  with a latency budget of 20ms per image, which would you
  pick, and what would you change?

## Stretch goals (optional)

- **MixUp.** Interpolate two images and their one-hot
  labels; train on the interpolated example. Add as a fourth
  ablation row.
- **Cutout / RandomErasing.** Another regulariser worth
  ablating.
- **`torch.compile`.** Enable on the training model; measure
  the speedup.
- **Try MobileNet-V3 or EfficientNet-B0** for the
  fine-tune leg. Compare speed and accuracy to ResNet-18.
- **Grad-CAM visualisations.** For 5 test images, plot
  which pixels the model attended to when predicting each.
  Insightful for debugging misclassifications.
- **CIFAR-100** instead of CIFAR-10 for a much harder
  problem. Update the metric floors to something like ≥ 55%
  small CNN, ≥ 70% fine-tuned.
- **Compare Adam vs. SGD-with-momentum** on the small CNN.
  SGD often edges out for from-scratch CNN training; Adam
  is easier to tune.

## What to hand in

```
exercise-02/
├── data.py                  # transforms, loaders
├── models/
│   ├── __init__.py
│   ├── small_cnn.py
│   └── pretrained.py
├── train.py                 # extends exercise-01's loop
├── evaluate.py              # per-class, confusion, samples
├── tests/
│   ├── test_shapes.py
│   ├── test_augmentation.py
│   ├── test_normalisation.py
│   ├── test_frozen.py
│   └── test_overfit.py
├── plots/
│   ├── loss_curve_small_cnn.png
│   ├── loss_curve_pretrained.png
│   ├── confusion_small_cnn.png
│   ├── confusion_pretrained.png
│   └── samples_pretrained.png
├── history_small_cnn.json
├── history_pretrained.json
├── pyproject.toml
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.
