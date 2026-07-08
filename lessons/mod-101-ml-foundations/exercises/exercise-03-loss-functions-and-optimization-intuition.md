# exercise-03 — Loss functions and optimisation intuition

> **Estimated effort:** 3 hours.
> **Companion chapter:** [06 — Loss functions and optimisation intuition](../06-loss-and-optimization.md).

## Objective

Develop the eye for "what is this loss function rewarding?" and "what
does this training curve mean?". You will implement four common
losses, visualise their gradients, run controlled optimisation
experiments, and learn to read loss curves like a working ML engineer.

## Prerequisites

- Python 3.11+, NumPy ≥ 1.26, Matplotlib.
- Chapters 03, 05, and 06 of this module.
- Exercise 01 (you'll reuse the gradient-descent loop).

## Problem statement

Three parts. Part A is implementation, part B is empirical study,
part C is interpretation.

### Part A — four losses and their gradients (≈ 60 min)

In `losses.py`, implement each of the four loss functions below.
Implement the **per-example** scalar loss, the **batch-mean** loss,
and the **gradient with respect to the model output**. Treat the
model output as a single scalar `s` (a "logit" or "score") and the
target as `y`.

1. **Mean squared error.** $\ell(s, y) = (s - y)^2$.
2. **Mean absolute error.** $\ell(s, y) = |s - y|$. (Be careful at
   `s = y`; document your subgradient choice.)
3. **Huber loss** with `delta = 1.0`. Quadratic for `|s - y| < delta`,
   linear outside. Useful for robust regression. State the formula
   in your code's docstring.
4. **Binary cross-entropy with logits.** $\ell(s, y) = -y \cdot
   \log\sigma(s) - (1 - y) \cdot \log(1 - \sigma(s))$ where $y \in
   \{0, 1\}$ and $\sigma$ is the sigmoid. **Implement numerically
   stably** using the
   [log-sum-exp trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/);
   do not call `np.log(sigmoid(s))` directly.

For each loss, write a test that confirms the analytical gradient
matches a numerical (central-difference) gradient on random inputs.

### Part B — loss landscapes and optimiser behaviour (≈ 75 min)

#### B.1 — Visualise losses (≈ 20 min)

For each of the four losses, with `y = 0` held fixed, plot
$\ell(s, 0)$ for $s \in [-5, 5]$. Plot all four on a single chart.
Add a second chart with the four gradients $\partial \ell / \partial s$
on the same axis. Save as `losses/loss_shapes.png` and
`losses/grad_shapes.png`.

Write 3–5 sentences in `REPORT.md`:

- Where is MSE more aggressive than MAE? Where is the reverse?
- What does the Huber loss recover from MSE and MAE?
- What does the binary cross-entropy gradient look like as `s → +∞`
  for `y = 1` vs. `y = 0`?

#### B.2 — Outlier sensitivity (≈ 25 min)

Generate a 1-D regression dataset with 200 inliers from
$y = 2x + 1 + \mathcal{N}(0, 0.3)$ and 10 outliers with `y` drawn
from $\mathcal{N}(20, 2)$ at random `x`. Fit a linear regression
(slope and intercept) by gradient descent under three losses:

- MSE
- MAE
- Huber (delta = 1.0)

Plot the data, the three fitted lines, and the "true" line (`slope=2,
intercept=1`) on one chart. Report the slope and intercept under each
loss. Save as `losses/outlier_fit.png`.

Add to `REPORT.md`: which loss recovered the true line most closely?
Which gave the worst slope? Why?

#### B.3 — Learning rate sweep (≈ 30 min)

Pick MSE on the same dataset (the **inlier-only** version this time).
Run gradient descent for 200 steps with each of the following
learning rates: `1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0`. Plot all six
loss curves on the same chart (log-y axis). Save as
`losses/lr_sweep.png`.

Add to `REPORT.md`: which learning rates diverged, which were too
slow, which were "just right"? What is the consequence of picking
the highest non-divergent learning rate?

### Part C — interpreting a real-looking training curve (≈ 45 min)

In `pathologies.py`, generate four loss curves, each ~200 steps long,
each illustrating *one* of the following pathologies:

1. **Diverged training.** Loss eventually goes to `inf` or `nan`.
2. **Overfitting.** Training loss decreases, validation loss
   eventually rises.
3. **Stuck-at-baseline.** Loss is flat from step 1 (e.g. a zero-gradient
   activation, or a learning rate of 0).
4. **Healthy convergence.** Loss decreases monotonically (modulo
   noise) and plateaus.

You can construct these synthetically — you do not need a real
model. The goal is to be able to **reproduce each shape on demand**
because you understand what causes each one.

For each, save a plot to `pathologies/curve_<n>.png` and write 2–3
sentences in `REPORT.md` answering:

- What does this curve look like, in words?
- In a real training run, what would cause it?
- What is the first thing you would change?

## Starter guidance

- The numerically-stable binary cross-entropy is
  `max(s, 0) - s * y + log(1 + exp(-|s|))`. Derive it; don't memorise it.
- For MAE, plain gradient descent is jittery near the minimum
  because the gradient doesn't vanish. This is a feature of the
  loss, not a bug in your code; tolerate it or use a smaller
  learning rate.
- The lr_sweep chart looks much more informative on a *log-y* axis
  than a linear axis. Use `plt.yscale('log')` after dropping any
  non-positive losses.
- For Part C, you don't need to train an actual model. You can
  synthesise the curves with a few lines of `numpy` and analytical
  shapes; the goal is *recognition*, not realism.

## Acceptance criteria

- All four losses implemented; analytical gradients match numerical
  gradients within `1e-5` (except MAE at `s = y`, where you have
  documented your choice).
- Three plots from Part B (`loss_shapes.png`, `grad_shapes.png`,
  `outlier_fit.png`), one from Part B.3 (`lr_sweep.png`), and four
  from Part C, all saved at the paths above.
- `REPORT.md` answers every required prompt.

## Stretch goals (optional)

- Add **stochastic mini-batch SGD** (batch size 16) to your
  MSE/inlier sweep; compare its loss curve at `lr = 1e-1` to plain
  gradient descent at the same rate. What do you observe?
- Add a **momentum** term to your GD loop and compare convergence
  speed against plain GD on the same problem.
- Add **L2 regularisation** to MSE and re-do the outlier fit with a
  large coefficient. What happens to the slope estimate?
- Reproduce the **"loss stagnates, then a step LR drop revives
  training"** pattern, and chart it.

## What to hand in

```
exercise-03/
├── losses.py
├── pathologies.py
├── tests/
│   └── test_losses.py
├── losses/
│   ├── loss_shapes.png
│   ├── grad_shapes.png
│   ├── outlier_fit.png
│   └── lr_sweep.png
├── pathologies/
│   ├── curve_1.png
│   ├── curve_2.png
│   ├── curve_3.png
│   └── curve_4.png
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo.
