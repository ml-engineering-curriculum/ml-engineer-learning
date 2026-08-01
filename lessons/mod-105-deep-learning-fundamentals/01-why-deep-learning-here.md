# 01 — Why deep learning here, and where classical still wins

> **Reading time:** ~15 minutes.
> **Prerequisite modules:** `mod-101` (foundations), `mod-104`
> (classical modelling).

## Motivation

Half the world's ML talk is about deep learning. The other
half of the world's *production* ML is a gradient-boosted
tree. Both statements are true, and holding both in mind is
the point of this chapter.

`mod-104` gave you the models that win most tabular problems:
regularised linear regression, random forests, gradient
boosting. Those models are the correct choice for a wide
class of tasks and they will remain correct after this
module. What they cannot do is learn a good representation
of a grid of pixels, a stream of tokens, a waveform, or a
video. That is what deep learning is for.

This module teaches the fundamentals so that when a problem
lands on your desk you can:

1. Recognise whether it is a deep learning problem at all.
2. If yes, build a working baseline in one of the three
   canonical architectures — MLP, CNN, Transformer encoder —
   with a training loop you understand.
3. If no, do not force deep learning onto it. Reach for
   `mod-104`.

## When to reach for deep learning

Three signals — any one is enough:

- **The input has structure the model must discover.**
  Pixels have local spatial structure; tokens have long-
  range sequential structure; audio frames have both.
  Feature engineering these by hand hits a ceiling. Neural
  networks learn features and predictions jointly.
- **You have a lot of data.** Deep networks reward scale.
  On a few thousand rows with clean features, a
  gradient-boosted tree will match or beat a neural net and
  train in a hundredth of the time. The crossover point
  varies by task but usually starts in the hundreds of
  thousands to millions of examples.
- **A pre-trained model exists for your input modality.**
  Fine-tuning a ResNet, a CLIP encoder, or a BERT-family
  encoder on your task with a thousand labelled examples
  will beat the best classical model you can build on the
  same thousand examples. The pre-training did the
  representation work; you inherit it.

## When to *not* reach for deep learning

Also three signals — any one is enough:

- **The features are already tabular and semantic.**
  `income`, `country`, `is_returning_user`. Trees eat this
  for breakfast. A neural net will match it eventually with
  more parameters, more data, and more of your time.
- **The dataset is small (<10k rows) and expensive to
  grow.** Deep networks overfit small data readily. The
  regularisation techniques in chapter 08 help but do not
  eliminate the disadvantage.
- **The stakes require interpretability that trees or
  linear models provide out of the box.** Credit
  underwriting, medical triage, adverse-action explanations.
  A neural net can be interpreted with tools like SHAP, but
  "the coefficient on income is +0.03" is a simpler story
  than "here is a SHAP plot of a 200k-parameter MLP".

If you are unsure, run the `mod-104` baseline first.
Chapter 05 of this module makes the same case: **beat a
gradient-boosted tree before you commit to a deep network.**

## What deep learning brings that classical does not

Two things that classical ML fundamentally cannot do:

1. **Learn hierarchical features from raw inputs.** A CNN's
   early layers learn edges, its middle layers learn
   textures, its later layers learn parts and objects — all
   from pixels, without a feature engineer specifying any of
   them. A Transformer learns which tokens attend to which
   other tokens from co-occurrence in text. No hand-crafted
   feature can substitute.
2. **Transfer knowledge across tasks via pre-training.** A
   BERT model trained on a general corpus can be fine-tuned
   for a specific classification task with orders of
   magnitude less data than training a bespoke model would
   need. A classical model has no equivalent. Every task
   starts from scratch.

The rest of the module is the machinery you need to exploit
those two capabilities.

## The chapter map

Ten chapters, four objectives:

| # | Chapter | Objective |
|---|---|---|
| 01 | This chapter | orientation |
| 02 | Tensors, autograd, devices | training loop |
| 03 | The PyTorch training loop | training loop |
| 04 | Datasets, DataLoaders, collate | training loop |
| 05 | Building an MLP | architectures |
| 06 | Convolutional neural networks | architectures |
| 07 | A small Transformer encoder | architectures |
| 08 | Regularisation and LR schedules | training tricks |
| 09 | Mixed precision and performance | training tricks |
| 10 | Transformer LLMs and delegation | LLM literacy |

Chapters 02–04 build the *loop*. Chapters 05–07 build the
*models*. Chapters 08–09 make the loop and the models
faster and better-generalising. Chapter 10 draws the line
between what this module teaches and what the LLM
Engineering track teaches, so you know where to send the
next task.

## PyTorch, not TensorFlow, not JAX

We use **PyTorch** as the framework throughout. Three
reasons:

- **Research dominance.** The overwhelming majority of new
  models are released as PyTorch checkpoints on Hugging
  Face. Fluency in PyTorch is fluency with the current
  research output.
- **Eager-mode debugging.** A PyTorch `nn.Module` is a
  plain Python object. You can `print(x.shape)` inside a
  forward pass. This makes learning cheaper.
- **Ecosystem.** `torchvision`, `torchaudio`, `torchtext`,
  Hugging Face `transformers`, Lightning, `torch.compile`,
  and every serving stack has first-class PyTorch support.

Familiarity with JAX or TensorFlow becomes useful in
specialist tracks (JAX for research-scale numerics; TF for
TFX and some production estates). Not here.

## What this module is *not* going to make you

A short list of things this module deliberately does not
promise:

- **An LLM engineer.** Chapter 10 gives you the vocabulary
  and the delegation map. Training an LLM, fine-tuning
  frontier models, RAG pipelines, agentic systems — the LLM
  Engineering track.
- **A vision specialist.** Chapter 06 is CNN foundations.
  Object detection, segmentation, diffusion models — the
  Applied Vision track.
- **An ML researcher.** No original architectures, no new
  optimisers, no theory beyond debugging intuition.
- **A distributed training specialist.** Single GPU (or
  CPU) throughout. DDP, FSDP, tensor parallelism — the MLOps
  and Infra tracks.

What it *does* make you: an ML engineer who can pick up any
of the three canonical architectures, wire it into a
training loop, get it to converge, and know when the
problem in front of you belongs to a specialist.

## A note on hardware

Every chapter's code runs on **CPU**. It will be slow — the
Transformer chapter is the worst offender at a few minutes
per epoch on a modest laptop — but nothing requires a GPU
to *learn from*.

If you have access to a GPU (a personal machine with an
NVIDIA card, a cloud instance, or a free-tier notebook
service like Colab or Kaggle), chapter 09's mixed-precision
material becomes concrete rather than abstract. The
exercises call out which parts benefit.

## Summary

- Deep learning is the right tool for high-structure inputs
  (pixels, tokens, audio, video), for scale (hundreds of
  thousands of examples and up), and for problems where a
  pre-trained model gives you a shortcut.
- Classical ML — the `mod-104` toolkit — remains the right
  tool for tabular, small, or interpretability-critical
  problems.
- This module teaches PyTorch fundamentals in ten chapters:
  the training loop, three canonical architectures, and the
  training tricks that turn a demo into a working model.
- It does not teach LLM training, distributed compute, or
  advanced computer vision. Chapter 10 draws the line and
  points at the specialist tracks.

## Where this goes next

Chapter 02 introduces the two primitives every PyTorch
program is built on: the `Tensor` and its `autograd` engine.
