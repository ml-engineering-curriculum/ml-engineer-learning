# 06 — Loss functions and optimisation intuition

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 03, 05.

## Motivation

Chapters 03 and 05 set up the two halves of the training engine:

- Chapter 03 told you what a gradient is and that the engine of training
  is "step in the direction of the negative gradient".
- Chapter 05 told you that the *function* whose gradient you take comes
  from a probability model — it is the negative log-likelihood plus an
  optional regulariser.

This chapter puts them together: how the loss function and the optimiser
interact in practice, what goes wrong when they don't, and what the
ML-engineering knobs are.

## What a loss function is

A **loss function** $\mathcal{L}(\theta)$ is a scalar measure of how
badly your model with parameters $\theta$ predicts the training data.
Training is the act of finding parameters that minimise it.

For a dataset $\{(x_i, y_i)\}_{i=1}^N$ the loss is typically an average
of per-example losses:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i) + \lambda \cdot R(\theta),$$

where $f_\theta$ is the model, $\ell$ is the per-example loss, $R$ is a
regulariser, and $\lambda \ge 0$ trades fit against complexity.

### Losses you should recognise

| Task | Loss | Why |
|---|---|---|
| Regression | Mean squared error (MSE) | NLL under Gaussian noise |
| Regression (robust) | Mean absolute error (MAE) / Huber | NLL under Laplace / less tail-sensitive |
| Binary classification | Binary cross-entropy (log loss) | NLL under Bernoulli |
| Multi-class classification | Cross-entropy (often with softmax) | NLL under categorical |
| Multi-label classification | Sum of binary cross-entropies | independent Bernoullis per label |
| Ranking (pairwise) | Pairwise hinge / logistic | "the right item should score higher" |
| Count regression | Poisson NLL | NLL under Poisson counts |

Two debugging instincts to internalise:

1. **If the loss does not match the noise model, the predictions will be
   biased in a predictable direction.** MSE on heavy-tailed data
   over-weights outliers; MAE under-weights them.
2. **If your loss can go to negative infinity, you have a numerical
   bug.** Take `log(0)` and you get `-inf`. Always use the library's
   `*_with_logits` variant (e.g. `BCEWithLogitsLoss`) — it fuses
   softmax/sigmoid with cross-entropy in a numerically stable way.

## The optimisation landscape

The loss is a function from parameter space to a scalar. The objects in
that landscape have names:

- **Global minimum** — the smallest value of $\mathcal{L}$ anywhere.
- **Local minimum** — smaller than all immediate neighbours, but not
  necessarily the global minimum.
- **Saddle point** — flat in some directions, sloped in others.
  Notoriously common in high-dimensional non-convex losses.
- **Plateau** — region of (nearly) zero gradient. Slows training.

A **convex** loss has a single minimum (no local-vs-global distinction),
and gradient descent provably converges to it. Linear regression with
MSE and logistic regression with cross-entropy are convex.

A **non-convex** loss — every modern neural network — has many local
minima and saddle points. Empirically, gradient descent still works in
deep learning because most local minima happen to give similar loss
values, and saddle points can be escaped by stochastic noise. This is an
empirical observation more than a theorem; you should know it both
because it explains why deep learning works at all and because it
explains why "my training loss looks weird" is a normal experience.

## Gradient descent

The simplest update rule, given a learning rate $\eta > 0$:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t).$$

In code (toy single-step example):

```python
import numpy as np

def loss_and_grad(w, X, y):
    pred = X @ w
    err = pred - y
    loss = 0.5 * np.mean(err ** 2)
    grad = X.T @ err / len(y)
    return loss, grad

w = np.zeros(X.shape[1])
for step in range(1000):
    loss, grad = loss_and_grad(w, X, y)
    w -= 0.1 * grad
```

Three things determine whether this works:

1. **The learning rate $\eta$.** Too small: training is glacial. Too
   large: the loss oscillates or diverges. There is no universal best
   value — it depends on the loss, the model, the feature scale, and
   the optimiser.
2. **Feature scale.** Gradient descent steps in the same $\eta$ across
   all dimensions. If one feature is in $[0, 10^6]$ and another in
   $[0, 1]$, the gradient direction is dominated by the first and you
   zig-zag. Standardising features (zero mean, unit variance) is the
   single cheapest fix and one of the highest-leverage habits in
   classical ML.
3. **The condition number of the problem.** Even with standardised
   features, if the loss is much steeper in some directions than others,
   plain gradient descent crawls. This is why we have second-order and
   adaptive methods.

## Stochastic and mini-batch gradient descent

Computing the full gradient over the entire training set every step is
expensive. **Stochastic gradient descent (SGD)** estimates the gradient
from a single example; **mini-batch SGD** uses a small batch (32, 64,
256, ...). Almost all real training uses mini-batches.

The trade-offs:

- Smaller batches → noisier gradient estimates → more steps per epoch,
  but each step is cheap. The noise can help escape saddle points.
- Larger batches → smoother gradient → fewer steps per epoch, each more
  expensive. Often needs more memory and a larger learning rate.

Batch size is a hyperparameter; it interacts with the learning rate
(a common rule of thumb: scaling the batch size by $k$ allows scaling
the learning rate by $k$ — proposed in the
[Linear Scaling Rule paper](https://arxiv.org/abs/1706.02677); not a law,
but a starting point).

## Optimisers beyond plain SGD

A whirlwind tour of the optimisers you'll see in `mod-104`/`mod-105`:

- **SGD with momentum.** Accumulate a moving average of past gradients
  to damp oscillation and accelerate along consistent directions.
- **Adagrad / RMSProp.** Divide the learning rate by a running norm of
  past gradients per parameter — effectively per-parameter learning
  rates.
- **Adam / AdamW.** The popular default in deep learning; combines
  momentum with per-parameter rate adaptation. `AdamW` separates weight
  decay (L2 regularisation) from the gradient — usually the better
  choice. See the original
  [Adam paper](https://arxiv.org/abs/1412.6980) and
  [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- **L-BFGS.** A quasi-Newton method useful for small convex problems
  (e.g. fitting a logistic regression). Scales poorly to deep nets.

You do not need to memorise the update rules. You do need to know that
"Adam is the safe default for deep nets, plain SGD with momentum is
sometimes better with a good learning-rate schedule, L-BFGS is for small
convex problems."

## Learning-rate schedules

A constant learning rate works less often than you'd think. The two
schedules you'll see everywhere:

- **Step decay / multistep.** Drop the learning rate by a factor every
  few epochs.
- **Cosine decay (with warmup).** Smoothly decay following a cosine
  curve, often after a few epochs of linear warm-up. Standard in modern
  vision and language pre-training.

Watch the loss curve. If it plateaus and a learning-rate drop restarts
progress, your schedule was too flat. If it oscillates, your schedule
was too aggressive.

## What "convergence" actually looks like

A healthy training run, plotted with loss on the y-axis and step on the
x-axis, looks like a noisy but monotone decrease that flattens out into
a noisy plateau. Common pathologies and what they imply:

- **Loss is `nan` or `inf`.** Numerical blow-up; learning rate too high,
  unstable loss formulation, or unnormalised inputs.
- **Loss decreases on training, then increases.** Overfitting in
  progress; either regularise more or stop training earlier (early
  stopping).
- **Loss is flat from step 1.** Either the gradient is dead (bad
  activation, no gradient flow) or the loss happens to be already at
  the model's expressive limit on this data. Sanity-check with a tiny
  model that can clearly fit a small subset (a memorisation test).
- **Loss oscillates wildly.** Learning rate too high, or batch size too
  small for the noise level. Lower the rate; consider gradient clipping.

You will look at training curves more than at almost anything else in
this curriculum; build the eye for them early.

## Summary

- A loss function comes from a probability model. The match between loss
  and noise model matters.
- Gradient descent is the universal engine. Its three knobs are learning
  rate, batch size, and (for adaptive optimisers) per-parameter
  scaling.
- Convex losses have a unique minimum; non-convex losses don't, but
  gradient descent still works in practice.
- The optimiser zoo is mostly variations on a theme. AdamW is the
  default for deep nets; SGD with momentum + a good schedule sometimes
  wins; L-BFGS is for small convex problems.
- Read the loss curve. It is the cheapest diagnostic you have.

## Where this goes next

You now have the entire training engine in your head. Chapters 07 and 08
zoom out: of all the things a model *could* learn, which one fits your
business problem? That framing decision is what the rest of the module
is about.
