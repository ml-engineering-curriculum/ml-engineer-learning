# 06 — Convolutional neural networks

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 02–05.

## Motivation

A 224×224 RGB image has 150,528 raw pixel values. Feed that
into a fully-connected layer with 512 hidden units and you
have a 77 million-parameter matrix. Do it twice and the model
is bigger than the dataset it will train on.

Convolutional layers replace that matrix with a **small
sliding filter that shares parameters across spatial
positions**. The same 3×3 filter that detects a vertical edge
in the top-left corner of the image also detects it in the
bottom-right. This is the single insight that made deep
learning practical for vision: strong prior + parameter
sharing = a model that generalises from thousands of images
rather than requiring millions.

CNNs also transfer. A model pre-trained on ImageNet has
learned "edges, textures, parts" in its early layers; those
generalise to almost any image task. Fine-tuning a
pre-trained backbone on your own labelled data is usually
the right first move — chapter's end covers it.

## What a convolution does

A 2-D convolution takes a small filter (kernel) of shape
`(C_out, C_in, K, K)` and slides it over the input `(B, C_in,
H, W)`, computing at each spatial position a dot product
between the filter and the input patch beneath it.

For a filter of size 3, an input of shape `(1, 1, 5, 5)`, and
`stride=1, padding=0`, the output is `(1, 1, 3, 3)`. Each
output pixel is a weighted sum of 3×3=9 input pixels. The
same weights are used at every position — that is the
parameter sharing.

```python
import torch.nn as nn
conv = nn.Conv2d(
    in_channels=3, out_channels=16,
    kernel_size=3, stride=1, padding=1
)
```

- **`in_channels`** — number of channels in the input
  (3 for RGB, 1 for grayscale, whatever the previous conv
  emitted for intermediate layers).
- **`out_channels`** — how many filters to learn. Each is a
  separate feature detector.
- **`kernel_size`** — spatial extent. 3 is the modern
  default; 5, 7, 11 appear in older architectures.
- **`stride`** — step size when sliding. `stride=2` halves
  the spatial dimensions.
- **`padding`** — zeros added around the input. `padding=1`
  with `kernel_size=3` keeps `H, W` unchanged (a "same"
  convolution).

Parameter count for `Conv2d(3, 16, kernel_size=3)`:
`16 * 3 * 3 * 3 = 432` weights + 16 biases = **448
parameters**, regardless of image size. Compare to a
fully-connected layer for the same job.

## Receptive field

A single 3×3 conv sees a 3×3 patch of its input. A second
3×3 conv on top of that sees a 5×5 patch of the original
input (each of its 9 inputs came from a 3×3 patch, and
they overlap). A third gets you 7×7. Depth grows the
**receptive field** — the region of the original image that
influences one output activation.

Two implications:

- A deep stack of small filters (3×3) matches the receptive
  field of a shallow stack of large filters (7×7) with fewer
  parameters and more nonlinearities in between. This is why
  modern CNNs use 3×3 almost exclusively.
- To classify a whole image, the receptive field of the
  final feature map must eventually cover the whole input.
  Downsampling (strided conv or pooling) gets you there
  faster than depth alone.

## Pooling

Pooling downsamples a feature map. Two flavours:

- **`nn.MaxPool2d(kernel_size=2, stride=2)`** — take the max
  in each 2×2 patch. Halves `H` and `W`. Classical choice;
  robust to small translations.
- **`nn.AvgPool2d(kernel_size=2, stride=2)`** — mean of each
  patch. Smoother; less common in feature-extraction layers.

Modern architectures often use **strided convolutions**
(`stride=2` inside the `Conv2d`) instead of a separate
pooling layer. Same downsampling effect, but the network
learns the reduction rather than hard-coding it.

At the top of a network, **`nn.AdaptiveAvgPool2d((1, 1))`** is
the modern pattern: whatever the spatial resolution is at
that point, pool to `1×1` per channel. Removes the "input
must be exactly 224×224" constraint that older architectures
had.

## Normalisation

Batch Normalization (`nn.BatchNorm2d`) normalises activations
per-channel over the batch dimension, using batch statistics
during training and running averages during eval. It
smooths the loss landscape and lets you use higher learning
rates. Insert it after the conv and before the activation:

```python
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )
```

Two things worth knowing:

- **`bias=False`** on convs immediately before BatchNorm —
  the BN's own beta parameter subsumes the bias. Saves
  parameters; makes no numerical difference.
- **BatchNorm behaves differently in eval mode** — it uses
  running mean and variance. Forgetting `model.eval()` before
  validation is one of the top-five CNN training bugs.

Two alternatives you will meet:

- **`nn.LayerNorm`** normalises per-sample across features.
  Standard in Transformers (chapter 07); shows up in some
  vision Transformers.
- **`nn.GroupNorm(num_groups, num_channels)`** normalises
  within channel groups. Batch-size independent — good for
  small-batch training or when BN's running-stat behaviour
  is a problem.

## A minimal CNN, built end to end

Enough theory. A working CNN for 32×32 RGB images (CIFAR-10-
shaped):

```python
import torch
import torch.nn as nn

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),                # 32 -> 16

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),                # 16 -> 8

            conv_block(64, 128),
            conv_block(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),   # 8 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

Count the parameters (see chapter 05); you will land around
300k. Compare to a fully-connected model on the same input:
`3 * 32 * 32 = 3,072` inputs times a modest 512 hidden units
is already 1.5M parameters for a single layer, and the model
will learn worse. That is the value of the spatial prior.

## Residual connections

Networks deeper than about 15–20 layers stop training
without residual connections — the gradient signal from the
loss cannot propagate cleanly through that many
transformations. ResNet's contribution was the **skip
connection**: an identity path added to the output of a
block, so gradients can flow around it:

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))          # <-- the skip
```

When the block changes channel count or spatial resolution,
the skip needs a 1×1 conv to match shapes. The general
pattern:

```python
if in_channels != out_channels or stride != 1:
    self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
else:
    self.skip = nn.Identity()
...
def forward(self, x):
    return self.act(self.skip(x) + self.conv(x))
```

Residuals are why ResNet-50 and its descendants (~50–200
layers) train successfully. They are a fundamental
building block, not just a trick — chapter 07's Transformer
uses them too.

## Data augmentation

CNNs learn spatial invariances that are already true of the
world (a cat is still a cat if the image is shifted a few
pixels). Data augmentation gives them examples that make
that invariance explicit. Standard training augmentations
for CIFAR-10-scale images:

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std =[0.2470, 0.2435, 0.2616]),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std =[0.2470, 0.2435, 0.2616]),
])
```

Notes:

- **Augment only training.** Validation and test images get
  the deterministic pipeline (no random crop, no flip).
- **Normalise with dataset statistics.** The numbers above
  are the CIFAR-10 per-channel means and stds. For a
  pre-trained model, use the statistics it was trained with
  (ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229,
  0.224, 0.225]`).
- **Stronger augmentation for larger images / longer
  training.** RandAugment, AutoAugment, and MixUp are the
  next tier when the simple crops-and-flips baseline stops
  improving.
- **`torchvision.transforms.v2`** is the current API. It
  supports GPU-side augmentation and cleaner composition
  than the v1 transforms.

For images loaded lazily from disk, put the augmentation in
your `Dataset.__getitem__`. The DataLoader's workers will run
it in parallel with the previous batch's forward/backward.

## The transfer-learning shortcut

Training a modern CNN from scratch on your dataset is rarely
the right move. A pre-trained model gives you:

- Feature extractors that took hundreds of GPU-days to learn.
- A well-initialised set of weights.
- A shorter fine-tuning schedule (5–30 epochs vs. hundreds).
- Usually higher accuracy than any from-scratch model you
  could train on the same data.

The idiom in `torchvision`:

```python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# Preprocessing that matches the checkpoint:
preprocess = weights.transforms()

# Swap the last layer for your number of classes:
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Optionally freeze the backbone:
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True
```

Three common patterns, in order of "more data / more
tolerance":

- **Feature extraction** — freeze the backbone, train only
  the classifier head. Fastest; underperforms full
  fine-tuning when you have data.
- **Fine-tune the last few blocks** — unfreeze the last 1–2
  ResNet stages, keep earlier layers frozen. Middle ground.
- **Full fine-tune** — everything trainable, small learning
  rate on the backbone (say 10× smaller than the head's LR).
  Best when you have thousands of labelled examples per class.

Whatever the strategy, use the preprocessing that matches
the checkpoint's training data. Mismatched normalisation is
a silent accuracy killer.

## Sizes and layouts

PyTorch defaults to **channels-first** (`NCHW`): shape `(B,
C, H, W)`. Some hardware (Ampere and later, mobile
accelerators) is faster with **channels-last** (`NHWC`); the
model doesn't care about semantics, only the memory layout:

```python
model = model.to(memory_format=torch.channels_last)
x     = x.to(memory_format=torch.channels_last)
```

Optional; measure before assuming it helps. On CPU or older
GPUs, no benefit.

Two other layout gotchas:

- **`transforms.ToTensor()`** converts a PIL image `(H, W,
  C)` uint8 in `[0, 255]` to a `(C, H, W)` float32 in
  `[0.0, 1.0]`. Skip normalisation and your model trains,
  poorly. Don't skip it.
- **Grayscale**: `nn.Conv2d(in_channels=1, ...)`. Loading a
  grayscale image with `.convert("RGB")` is fine (duplicates
  the channel) but wastes memory.

## Debugging CNN training

The failure modes overlap with chapter 03 and 05, plus:

- **Val loss increases from epoch 1 while train loss falls
  cleanly.** Overfitting. Add augmentation, add dropout,
  reduce parameters, or use a pre-trained backbone.
- **Loss goes to NaN.** LR too high, or an unnormalised
  input, or gradient explosion in an unusual initialisation.
  Add gradient clipping (chapter 03), lower the LR, verify
  normalisation.
- **Val accuracy stuck at "random guessing"**. Check the
  label pipeline. A common bug: `torchvision.ImageFolder`
  assigns labels alphabetically by folder name; if you
  interpret the class index differently downstream, your
  metric is computing garbage.

## Pitfalls collected

1. **Fully-connected layers where convs belong.** Millions
   of parameters, no spatial prior, no chance.
2. **Forgetting `bias=False`** before BatchNorm — harmless
   but wasteful.
3. **Applying random augmentation to validation.** Metrics
   become noisy and non-reproducible.
4. **Mismatched normalisation between train and inference.**
   A pre-trained model expects specific stats.
5. **Not calling `.eval()` before validation.** BatchNorm's
   running stats update from val data; performance degrades
   silently.
6. **Training a from-scratch CNN when a pre-trained one is
   available.** Usually a mistake.
7. **Using RGB-normalisation stats on a grayscale image, or
   vice versa.** Silent scale bug.

## Summary

- Convolutions share parameters across spatial positions —
  the same 3×3 filter runs everywhere in the image.
- Deep stacks of small (3×3) convs plus downsampling grow
  the receptive field cheaply.
- BatchNorm smooths training; residual connections make deep
  networks trainable.
- Data augmentation is a strong regulariser for CNNs — crops,
  flips, MixUp; only on training.
- For almost any real task, fine-tune a pre-trained backbone
  rather than training from scratch.
- Chapter 09 (mixed precision) and chapter 08 (LR schedules,
  regularisation) apply straight to CNNs; the loop from
  chapter 03 is unchanged.

## Where this goes next

Chapter 07 introduces the third canonical architecture — the
Transformer encoder — and with it the attention mechanism
that underpins every modern LLM.
