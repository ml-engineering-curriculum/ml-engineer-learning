# lab-01 — Gradient descent trajectories on a 2D quadratic

> **Duration:** ~45–60 minutes.
> **Companion chapters:** [03 — Derivatives and gradients](../03-derivatives-and-gradients.md), [06 — Loss functions and optimisation intuition](../06-loss-and-optimization.md).

## Objective

Build an *eye* for what gradient descent actually does on a loss surface
— the zig-zag, the divergence, the effect of feature scale. By the end
you will have a small script that produces contour plots with the GD
trajectory drawn on top, and you will have reproduced on demand three of
the behaviours chapter 06 describes in words: convergence, zig-zag along
an ill-conditioned axis, and outright divergence from a learning rate
that is too high.

This lab is pure visualisation; the gradient derivation and the
linear-regression fit live in `exercise-01` and `exercise-03`, and
should be done before or alongside this lab.

## Prerequisites

- Python 3.11+, NumPy ≥ 1.26, Matplotlib.
- Chapters 03 and 06 read through once.
- A working Python environment (venv / `uv` / conda — any of them).

## Setup

Create a new directory for this lab and a single Python file:

```bash
mkdir -p lab-01-gd-trajectories && cd lab-01-gd-trajectories
touch gd_trajectory.py
```

Open `gd_trajectory.py` in your editor. All code in this lab goes in
this one file; you will run it from the terminal at each step.

## Steps

### Step 1 — Define an ill-conditioned 2D quadratic (≈ 5 min)

Start with the simplest non-trivial loss surface: a 2D quadratic with
very different curvature along the two axes. This makes the
"feature scale" and "condition number" conversation from chapter 06
immediately visible.

Add to `gd_trajectory.py`:

```python
import numpy as np
import matplotlib.pyplot as plt


def loss(w):
    # w is a shape (2,) array. The 10x on the second term
    # makes this axis ten times "steeper" than the first.
    return 0.5 * (w[0] ** 2 + 10.0 * w[1] ** 2)


def grad(w):
    # ∇f at w. Shape (2,).
    return np.array([w[0], 10.0 * w[1]])
```

**Checkpoint.** Call `loss(np.array([3.0, 1.0]))` from a REPL. It should
return `9.5` (`0.5 * (9 + 10 * 1)`). Call `grad(np.array([3.0, 1.0]))` —
it should return `array([3., 10.])`.

### Step 2 — Implement plain gradient descent with trajectory capture (≈ 10 min)

Add a function that returns both the final `w` *and* the full history of
intermediate `w` values, so you can plot it later.

```python
def gradient_descent(w0, lr, n_steps):
    """Returns (final_w, history) where history is an (n_steps+1, 2) array."""
    w = np.array(w0, dtype=float)
    history = [w.copy()]
    for _ in range(n_steps):
        w = w - lr * grad(w)
        history.append(w.copy())
    return w, np.array(history)
```

**Checkpoint.** With `w0 = [1.0, 1.0]`, `lr = 0.01`, `n_steps = 1`, the
history should be shape `(2, 2)` and the second row should be
`[0.99, 0.9]` (one GD step moves `w0` by `-lr * grad(w0)` = `-0.01 *
[1, 10]` = `[-0.01, -0.1]`).

### Step 3 — Plot contours plus trajectory (≈ 10 min)

Write a plotting helper that draws contour lines of the loss and
overlays the GD path.

```python
def plot_trajectory(history, title, out_path):
    xs = np.linspace(-5.0, 5.0, 200)
    ys = np.linspace(-2.0, 2.0, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (X ** 2 + 10.0 * Y ** 2)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.contour(X, Y, Z, levels=20)
    ax.plot(history[:, 0], history[:, 1], marker="o", markersize=3)
    ax.scatter(history[0, 0], history[0, 1], marker="s", label="start")
    ax.scatter(history[-1, 0], history[-1, 1], marker="x", label="end")
    ax.set_xlabel("w[0]")
    ax.set_ylabel("w[1]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

### Step 4 — The healthy run (≈ 5 min)

Add this to the bottom of `gd_trajectory.py` and run it:

```python
if __name__ == "__main__":
    start = [-4.0, 1.5]

    # Healthy run: lr small enough to stay stable.
    _, hist = gradient_descent(start, lr=0.05, n_steps=100)
    plot_trajectory(hist, "lr = 0.05 — healthy", "healthy.png")
```

```bash
python gd_trajectory.py
```

**Expected output.** A file `healthy.png` showing concentric elliptical
contours (long and thin along the `w[0]` axis) and the trajectory
walking from `(-4.0, 1.5)` towards the origin.

**Checkpoint.** Look at the trajectory shape. Does it move more or less
straight to the origin, or does it visibly bend? With `lr = 0.05` on
this problem it should settle fairly directly — the first axis is lazy
and the second is actively corrected.

### Step 5 — The zig-zag run (≈ 10 min)

Push the learning rate up — but not high enough to diverge — and watch
the trajectory oscillate along the steep direction while crawling along
the shallow one.

```python
    _, hist = gradient_descent(start, lr=0.18, n_steps=100)
    plot_trajectory(hist, "lr = 0.18 — zig-zag", "zigzag.png")
```

Re-run `python gd_trajectory.py`.

**Expected output.** `zigzag.png` shows the trajectory bouncing back and
forth across the `w[1] = 0` axis as it slowly makes its way towards the
origin along `w[0]`. This is the picture chapter 06 describes verbally
under "feature scale": the gradient step is too big for the steep axis
and too small for the shallow one.

**Checkpoint.** Can you *point* at the step on the plot where the
trajectory crosses `w[1] = 0`? Count the crossings — each one is a
wasted half-step.

### Step 6 — Divergence (≈ 5 min)

Push `lr` past the stability threshold. For this quadratic the step is
stable when `lr < 2 / L` where `L = 10` is the largest eigenvalue of
the Hessian (so `lr < 0.2`). Try something clearly over that line:

```python
    _, hist = gradient_descent(start, lr=0.21, n_steps=20)
    plot_trajectory(hist, "lr = 0.21 — diverging", "diverging.png")
```

Re-run. **Expected output.** `diverging.png` shows the trajectory
*exploding* outward — each step grows in magnitude. In the console the
final `w[1]` will be enormous (easily `> 10^6`); if you crank
`n_steps`, you'll overflow to `inf`.

**Checkpoint.** You've just reproduced the first pathology in chapter
06's "What convergence actually looks like" list — "`loss` is `nan` or
`inf`, learning rate too high". If your plot does not show divergence,
re-check your `grad` function; the step size being wrong by a factor of
2 is the classic "forgot the `1/2` or doubled it" bug.

### Step 7 — Rescale the problem (≈ 5 min)

The fix chapter 06 recommends for a scale mismatch is to standardise.
Mimic that here: define a *rescaled* loss where the two axes have equal
curvature.

```python
    def loss_scaled(w):
        return 0.5 * (w[0] ** 2 + w[1] ** 2)   # curvature 1, 1

    def grad_scaled(w):
        return np.array([w[0], w[1]])

    def gradient_descent_scaled(w0, lr, n_steps):
        w = np.array(w0, dtype=float)
        history = [w.copy()]
        for _ in range(n_steps):
            w = w - lr * grad_scaled(w)
            history.append(w.copy())
        return w, np.array(history)

    _, hist = gradient_descent_scaled(start, lr=0.18, n_steps=100)
    # Re-use the same contour plot helper, but note contours are now circular.
    xs = np.linspace(-5.0, 5.0, 200)
    ys = np.linspace(-5.0, 5.0, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (X ** 2 + Y ** 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(X, Y, Z, levels=20)
    ax.plot(hist[:, 0], hist[:, 1], marker="o", markersize=3)
    ax.set_title("lr = 0.18 on well-conditioned problem")
    fig.tight_layout()
    fig.savefig("scaled.png", dpi=120)
    plt.close(fig)
```

**Expected output.** `scaled.png` shows circular contours and a straight
line from the start point to the origin — no zig-zag, no divergence.
`lr = 0.18` was unstable on the ill-conditioned problem; here it is
fine.

**Checkpoint.** You have now reproduced, in a toy, the single most
cited diagnosis from chapter 06: *feature scale matters*. A learning
rate that is wrong for one problem is right for the rescaled version of
the same problem.

## Expected final artefacts

By the end of the lab you should have, in `lab-01-gd-trajectories/`:

```
lab-01-gd-trajectories/
├── gd_trajectory.py
├── healthy.png
├── zigzag.png
├── diverging.png
└── scaled.png
```

## Final checkpoint

Answer each of these in your head (or on paper) before closing the
lab:

1. For a quadratic loss with Hessian eigenvalues `λ_1, …, λ_n`, what
   learning rate causes plain gradient descent to diverge? (Hint: the
   chapter doesn't state the formula, but the threshold you observed
   around `lr = 0.2` on `L = 10` is a special case of a general rule;
   you'll come back to this in `mod-104` and `mod-105`.)
2. In `zigzag.png`, which axis is converging slowly and which is
   oscillating? Which of chapter 06's three "gradient descent knobs"
   would fix that without changing the loss function?
3. In `diverging.png`, how many steps does it take before `|w[1]|`
   exceeds its initial value? Does that number depend on the starting
   point?

If you can answer all three without re-reading chapter 06, the lab has
done its job.

## Where to go next

- `exercise-01` turns the one-step gradient descent in Step 2 into a
  full linear-regression fit with a gradient-check.
- `exercise-03` sweeps learning rates on a real 1-D regression problem
  and plots the resulting loss curves, letting you compare what you
  saw here (trajectories on a surface) against what you'll see in
  production (loss vs. step).
