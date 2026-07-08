# exercise-01 — Vector, matrix, and gradient warm-up

> **Estimated effort:** 3 hours.
> **Companion chapters:** [02 — Vectors and matrices for ML](../02-vectors-and-matrices-for-ml.md), [03 — Derivatives and gradients](../03-derivatives-and-gradients.md).

## Objective

Build muscle memory for the four operations you'll touch every day —
matrix multiplication, broadcasting, computing a gradient analytically,
and verifying it numerically. By the end you will have a working
hand-rolled linear regression that trains by gradient descent, and you
will have proved your gradient is right.

## Prerequisites

- Python 3.11+, NumPy ≥ 1.26.
- Chapters 02 and 03 of this module.
- A clean Python environment (`python -m venv` or `uv` works fine).

## Problem statement

You will implement and verify three things, in order:

1. A small library of NumPy "shape gymnastics" functions that exercise
   matrix multiplication, broadcasting, and norms.
2. A hand-derived gradient for the mean squared error loss of linear
   regression, and a gradient-check that proves it agrees with the
   finite-difference estimate.
3. A from-scratch gradient-descent loop that fits the linear regression
   on a synthetic dataset, plots the loss curve, and recovers the true
   parameters within a tolerance.

You may **not** call any function whose name contains `regression`,
`fit`, `optimize`, `minimize`, `gradient`, or `solve` from
`scipy.optimize`, `sklearn`, or `statsmodels`. You may use `numpy`,
`matplotlib`, and standard library.

## Requirements

### Part A — shape gymnastics (≈ 45 min)

In a file `shape_gym.py`, implement and unit-test the following:

- `pairwise_distances(X, Y)` — returns an `(m, n)` matrix where entry
  `(i, j)` is the Euclidean distance between row `i` of `X` (shape
  `(m, d)`) and row `j` of `Y` (shape `(n, d)`). No Python `for`
  loops; use broadcasting and `np.linalg.norm`.
- `row_normalise(X)` — returns `X` with each row scaled to unit L2
  norm. Handle the zero-row case explicitly (document your choice).
- `softmax(Z, axis=-1)` — numerically stable softmax (subtract the
  max along the axis before exponentiating). Verify rows sum to 1.
- `batched_dot(A, B)` — given `A` of shape `(B, d)` and `B` of shape
  `(B, d)`, returns the length-`B` vector of per-row dot products. No
  Python loops.

For each function, write at least two assertions: a shape check and a
correctness check against a tiny hand-computed example.

### Part B — gradient of linear regression (≈ 60 min)

On paper or in a markdown cell, derive the gradient of

$$\mathcal{L}(w, b) = \frac{1}{2N} \sum_{i=1}^N \bigl(w \cdot x_i + b - y_i\bigr)^2$$

with respect to $w$ and $b$. Write both partial derivatives in matrix
form (no element-by-element sums in the final answer).

Then in `linreg.py`:

- Implement `loss(w, b, X, y)` returning the scalar MSE.
- Implement `grad(w, b, X, y)` returning a tuple `(grad_w, grad_b)`
  using your derived formula. Annotate the shape of every intermediate.
- Implement `numerical_grad(w, b, X, y, eps=1e-6)` that uses central
  differences to estimate the same gradients element-wise.
- Write a `test_grad_matches_numerical()` test that draws a small
  random `(X, y)`, calls both, and asserts the max absolute
  per-element difference is below `1e-5`.

### Part C — gradient descent loop (≈ 60 min)

In `train.py`, generate a synthetic dataset:

- `n_samples = 500`, `n_features = 5`, fixed RNG seed.
- Draw `X` from `N(0, 1)`, pick a known `w_true` and `b_true`, set
  `y = X @ w_true + b_true + epsilon` with `epsilon ~ N(0, 0.1)`.

Then:

- Initialise `w = zeros(5)`, `b = 0`.
- Run plain (full-batch) gradient descent for at most 5,000 steps.
- Track the loss every 50 steps.
- Stop early if the L2 norm of the gradient drops below `1e-6`.
- Save the loss curve as `loss_curve.png` using matplotlib.
- Print `||w - w_true||` and `|b - b_true|` at the end.

### Part D — short write-up (≈ 15 min)

Add a `REPORT.md` in your exercise directory answering, in 5–10
sentences total:

- What learning rate did you settle on, and how did you pick it?
- What did your gradient-check find — was your analytical gradient
  right on the first try? If not, what was the bug?
- One plot from your loss curve, and one observation about its shape.

## Starter guidance

- Write your tests first. The hardest part of this exercise is *not*
  the maths; it is being honest about whether you've got it right.
  `assert np.allclose(...)` is your friend.
- Keep every tensor's shape annotated as a comment on the line that
  creates it. You will thank yourself when shapes don't match.
- If your loss explodes to `inf`, your learning rate is too high.
  Start at `1e-2` and divide by 3 until things stop diverging.
- If your gradient check fails by something close to `2x`, you almost
  certainly forgot a `1/N` or have an off-by-2 from the `1/2`
  out-front in the loss.
- Use `np.random.default_rng(0)` rather than the legacy
  `np.random.seed` — it's the supported modern API.

## Acceptance criteria

- All four functions in `shape_gym.py` pass their tests, and none
  uses an explicit Python `for` loop over rows.
- `test_grad_matches_numerical` passes with tolerance `1e-5`.
- `train.py` recovers `w_true` and `b_true` with
  `||w_hat - w_true||_2 < 0.05` and `|b_hat - b_true| < 0.05` on the
  fixed seed.
- `loss_curve.png` exists and shows a monotone (modulo small wiggles)
  decreasing curve.
- `REPORT.md` answers all three prompts honestly and includes the
  plot.

## Stretch goals (optional)

- Replace full-batch gradient descent with mini-batch SGD (batch size
  32). Compare loss curves and final parameter recovery.
- Add an L2 regularisation term, derive its gradient, and verify
  numerically.
- Replace your hand-written `grad` with a `torch.autograd`-based
  version and confirm both agree on a small test case. (This previews
  `mod-105`.)
- Reimplement `pairwise_distances` using the `(a - b)^2 = a^2 - 2ab + b^2`
  trick to avoid materialising the `(m, n, d)` tensor. Compare timings
  with `timeit` for `m = n = 2000`, `d = 64`.

## What to hand in

A directory `exercise-01/` containing:

```
exercise-01/
├── shape_gym.py
├── linreg.py
├── train.py
├── tests/
│   ├── test_shape_gym.py
│   └── test_linreg.py
├── loss_curve.png
└── REPORT.md
```

Solutions for this exercise live in the paired `ml-engineer-solutions`
repo; do not look at them until you have submitted yours.
