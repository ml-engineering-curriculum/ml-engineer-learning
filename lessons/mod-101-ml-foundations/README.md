# mod-101 — ML Foundations: Math, Statistics, and Problem Framing

> **Track:** Machine Learning Engineer (level 20)
> **Estimated effort:** 14 hours

This module rebuilds the mathematical and conceptual scaffolding that every
later module assumes. You do not need to become a mathematician here — you
need to recognise the objects that appear in every ML library you will touch
(`numpy.ndarray`, `torch.Tensor`, a loss function, a gradient, a probability
distribution) and be able to think clearly about which of them apply to a new
problem.

By the end you should be able to answer four questions for any new ML task:

1. What are the inputs and outputs as mathematical objects (shape, type,
   distribution)?
2. What is the loss function and what does its gradient look like?
3. What flavour of learning fits — supervised, unsupervised, or
   reinforcement?
4. If supervised, is it regression, classification, or ranking — and why?

## Learning objectives

- Apply the linear-algebra and calculus building blocks that recur in ML
  (vectors, matrices, derivatives, gradients).
- Reason about probability, distributions, and statistical estimation for ML.
- Frame a real-world problem as a supervised / unsupervised / reinforcement
  learning task.
- Choose between regression, classification, and ranking framings and
  articulate the trade-offs.

## Chapters

| # | Chapter | Maps to objective |
|---|---|---|
| 01 | [Why foundations matter](01-why-foundations-matter.md) | orientation |
| 02 | [Vectors and matrices for ML](02-vectors-and-matrices-for-ml.md) | linear algebra |
| 03 | [Derivatives, gradients, and the chain rule](03-derivatives-and-gradients.md) | calculus |
| 04 | [Probability and distributions](04-probability-and-distributions.md) | probability |
| 05 | [Statistical estimation: MLE, MAP, and Bayes](05-statistical-estimation.md) | statistics |
| 06 | [Loss functions and optimisation intuition](06-loss-and-optimization.md) | calculus + statistics |
| 07 | [Framing a problem: supervised, unsupervised, reinforcement](07-framing-an-ml-problem.md) | problem framing |
| 08 | [Regression vs. classification vs. ranking](08-regression-classification-ranking.md) | framing trade-offs |

## Exercises

| # | Exercise | Hours |
|---|---|---|
| 01 | [Vector / matrix / gradient warm-up](exercises/exercise-01-vector-matrix-gradient-warmup.md) | 3 |
| 02 | [Probability and Bayes from scratch](exercises/exercise-02-probability-and-bayes-from-scratch.md) | 3 |
| 03 | [Loss functions and optimisation intuition](exercises/exercise-03-loss-functions-and-optimization-intuition.md) | 3 |
| 04 | [Problem-framing rubric](exercises/exercise-04-problem-framing-rubric.md) | 3 |

Plus one lab (`labs/lab-01`) and one quiz (`quizzes/`).

## How to use this module

Each chapter is short and self-contained. Read it, then do the matching
exercise — the exercises are where the maths becomes muscle memory, not the
prose. If a chapter feels too dense, the prerequisite for it is named in the
chapter intro; drop down to that text and come back.

If you already have a working knowledge of college-level linear algebra,
calculus, and probability, you can skim chapters 02–05 and spend your time
on chapters 06–08 and the framing exercise — that is the content this
curriculum can teach you that a generic maths book cannot.

## What this module does **not** cover

- Numerical implementation tricks (overflow, conditioning, sparse formats) —
  see `mod-102-python-ml-toolchain`.
- Specific model families (trees, ensembles, neural nets) — see `mod-104`
  and `mod-105`.
- Metric design and validation strategy — see `mod-107`.

See [`resources.md`](resources.md) for textbooks, lecture series, and
primary documentation cited by these chapters.
