# mod-105 — Deep Learning Fundamentals with PyTorch

> **Track:** Machine Learning Engineer (level 20)
> **Estimated effort:** 18 hours

`mod-104` covered the models that win most tabular problems.
This module covers the ones you reach for when the data is a
grid of pixels, a stream of tokens, or a signal in time — the
architectures that dominate vision, language, and speech.

The tool is **PyTorch**. The workflow is the training loop —
forward, loss, backward, step — assembled by hand so that
nothing is magic when a debugger opens on it. From that
loop we build three architectures (MLP, CNN, Transformer
encoder), layer on regularisation, learning-rate schedules,
and mixed precision, and finish with the vocabulary you need
to read an LLM paper and delegate the specialist work to the
right track.

This is a *fundamentals* module. It does not train GPT.
It teaches you how the pieces fit together so the later
tracks — Applied Deep Learning, LLM Engineering, Vision,
MLOps — land on prepared ground.

## Learning objectives

- Write a PyTorch training loop from scratch (forward, loss,
  backward, step).
- Build an MLP, a CNN, and a small Transformer encoder.
- Apply regularisation, learning-rate schedules, and mixed-
  precision training.
- Gain working familiarity with transformer LLMs and where to
  delegate to specialist tracks.

## Chapters

| # | Chapter | Maps to objective |
|---|---|---|
| 01 | [Why deep learning here, and where classical still wins](01-why-deep-learning-here.md) | orientation |
| 02 | [Tensors, autograd, and devices](02-tensors-autograd-and-devices.md) | training loop |
| 03 | [The PyTorch training loop from scratch](03-the-pytorch-training-loop.md) | training loop |
| 04 | [Datasets, DataLoaders, and collate functions](04-datasets-dataloaders-and-collate.md) | training loop |
| 05 | [Building an MLP](05-building-an-mlp.md) | architectures |
| 06 | [Convolutional neural networks](06-convolutional-neural-networks.md) | architectures |
| 07 | [A small Transformer encoder](07-a-small-transformer-encoder.md) | architectures |
| 08 | [Regularisation and learning-rate schedules](08-regularisation-and-lr-schedules.md) | training tricks |
| 09 | [Mixed precision and training performance](09-mixed-precision-and-training-performance.md) | training tricks |
| 10 | [Transformer LLMs and when to delegate](10-transformer-llms-and-when-to-delegate.md) | LLM literacy |

## Exercises

| # | Exercise | Hours |
|---|---|---|
| 01 | [PyTorch training loop from scratch](exercises/exercise-01-pytorch-training-loop-from-scratch.md) | 4 |
| 02 | [CNN image classifier](exercises/exercise-02-cnn-image-classifier.md) | 4 |
| 03 | [Small Transformer encoder](exercises/exercise-03-small-transformer-encoder.md) | 4 |
| 04 | [LLM vocab and delegation map](exercises/exercise-04-llm-vocab-and-delegation-map.md) | 3 |

Plus one lab (`labs/`) and one quiz (`quizzes/`) — scaffolded on
subsequent authoring cycles.

## How to use this module

Read the chapters in order — later ones assume earlier ones.
The two highest-stakes chapters, which are worth slowing down
for, are:

- **03 (The training loop).** Every subsequent chapter reuses
  the loop from this one. If you skip it, you will paste over
  bugs you don't understand later.
- **07 (Transformer encoder).** The architecture behind every
  modern LLM. Understand it once here, in ~200 lines, and the
  rest of the field opens up.

The three architecture chapters (05, 06, 07) are worked
examples with the same shape: an `nn.Module`, a Dataset, a
training loop, and a metric. Read the one you will use next
first; the other two can wait if time is short.

Each chapter maps to at least one exercise. The exercises are
sequenced to build progressively more sophisticated networks
on top of the same loop, then finish with a written
delegation map for the LLM territory this module deliberately
does not enter. Do them in order.

## What this module does **not** cover

- **Training LLMs from scratch** or fine-tuning frontier
  models. That is the LLM Engineering track. Chapter 10 draws
  the map.
- **Distributed training** (DDP, FSDP, DeepSpeed, tensor /
  pipeline parallelism). Single-GPU or CPU throughout.
- **Production model serving** (TorchServe, Triton, ONNX
  runtime, TensorRT). `mod-108` (deployment) covers it.
- **Advanced computer vision** (detection, segmentation,
  diffusion). Chapter 06 is a foundation; specialist tracks
  build the applications.
- **Reinforcement learning**, **graph neural networks**, or
  **recommender systems**. Each is its own track.
- **Deep learning theory** beyond the working intuition
  needed to debug a training run.

## Prerequisites

- `mod-101` (ML foundations): loss functions, gradient
  descent, train/val/test splits, overfitting.
- `mod-102` (Python ML toolchain): dependency management,
  testing, packaging.
- `mod-103` (Data engineering for ML): building
  a Dataset out of files or an object store.
- `mod-104` (Classical ML): the baseline every deep learning
  model must beat before its complexity is justified.
- A machine with a GPU is helpful but not required. Every
  exercise runs on CPU; chapter 09's mixed-precision content
  is more informative with one but works through without.

See [`resources.md`](resources.md) for official documentation
and curated references behind each chapter.
