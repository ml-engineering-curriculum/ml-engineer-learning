# 01 — Why foundations matter

> **Reading time:** ~20 minutes.
> **Prerequisite chapters:** none.

## Motivation

A working ML engineer rarely sits down and proves a theorem. The job is to
move a model from data to a production system that does something useful and
keeps doing it. So why open a curriculum on production ML engineering with a
maths refresher?

Because every loud failure in a working ML team eventually traces back to a
miscommunication with the maths. Examples drawn from common failure modes:

- A model "regresses" after retraining. The fix is not a new model — the
  team is computing accuracy on a class-imbalanced eval set. Understanding
  what a probability distribution is would have led to a different metric.
- A gradient descent run "isn't learning". The cause is not the optimizer —
  the features are on wildly different scales and the gradient direction is
  dominated by one axis. Understanding what a gradient *is* would have led
  to standardising the features first.
- A "regression" model gives nonsense predictions for new customers. The
  task is actually a ranking problem (whoever shows up first wins), and a
  squared-error model is solving the wrong optimisation problem.

In each case the engineer who has internalised the foundations sees the
failure earlier and types fewer wrong fixes.

## What you actually need

You do **not** need to derive backpropagation or prove the convergence of
SGD. You need:

- **Linear algebra at the level of shapes and operations.** What is a
  vector, what is a matrix, what is a dot product, what does a matrix
  multiplication mean, what does a norm measure. You need this every time
  you read a model's docstring or debug a shape mismatch in PyTorch.
- **Calculus at the level of gradients.** What is a partial derivative,
  what does the gradient point at, what does the chain rule do. You need
  this every time you reason about why a model is or isn't learning.
- **Probability at the level of distributions.** What is a probability,
  what is a distribution, what is expectation, what does "i.i.d." mean.
  You need this every time you reason about what a model is predicting and
  whether your evaluation set is representative.
- **Statistical estimation at the level of MLE and Bayes' rule.** What does
  "fitting a model" actually optimise. You need this to understand why
  cross-entropy loss is the natural loss for classification and squared
  loss the natural loss for regression — and when neither is.
- **Problem framing.** Given a vague business request, which of the
  textbook ML tasks does it map to — and how confident are you in that
  mapping?

The first four bullets are tools; the last bullet is the skill that
distinguishes a useful ML engineer from a kaggle-grade model trainer.

## How this module is structured

The next two chapters cover linear algebra (chapter 02) and calculus
(chapter 03) at the ML-engineer's altitude — what you need to recognise,
not what you need to derive. Chapters 04 and 05 do the same for probability
and statistical estimation. Chapter 06 connects the two: a loss function
plus a gradient is the engine that powers almost every supervised model
you will train.

Chapters 07 and 08 are where the maths becomes engineering judgement. They
introduce the framing decision — supervised vs. unsupervised vs.
reinforcement, then regression vs. classification vs. ranking — and give
you a vocabulary for arguing about which framing fits a problem before you
touch a model.

## How to read the rest of the module

Treat each chapter as a tour, not a textbook. Read it, then go and do the
matching exercise. The exercises are short by design: a few hours each,
small enough to fit in an evening, hard enough to expose any place you only
thought you understood.

If a chapter is too compact, look at the prerequisite text it cites in
[`resources.md`](resources.md) and come back. Foundations are worth
spending a little longer on — every later module pays interest on what you
learn here.

## Summary

- The maths in this module is a tool for diagnosis, not a hoop.
- You need fluency with vectors, matrices, gradients, distributions, and
  the basic estimation principles — not theoretical depth.
- The framing skill that lets you turn a vague business problem into a
  precise ML task is the most leveraged thing you will learn in this
  module.
