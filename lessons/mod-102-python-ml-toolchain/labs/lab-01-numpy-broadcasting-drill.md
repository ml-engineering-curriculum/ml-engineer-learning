# lab-01 — NumPy broadcasting and vectorisation drill

> **Duration:** ~30–45 minutes.
> **Companion chapter:** [02 — NumPy for ML](../02-numpy-for-ml.md).

## Objective

Build muscle memory for the three NumPy habits chapter 02 calls out as
load-bearing: thinking in shapes, replacing Python loops with vectorised
expressions, and using broadcasting deliberately. By the end you will
have (a) felt the ~100× speed-up vectorisation buys, (b) hit the
`(N, 1)` vs. `(N,)` broadcasting trap on purpose so you recognise it
in the wild, and (c) seen the view-vs-copy distinction behave
differently on two syntactically similar slices.

This lab is pure NumPy. The longer `exercise-01` plugs these idioms
into a Pandas feature-engineering project; this lab is the warm-up
drill.

## Prerequisites

- Python 3.11+, NumPy ≥ 1.26 in any virtual environment.
- Chapter 02 read through once.
- An interactive REPL (`python`, `ipython`, or a Jupyter notebook —
  all work).

## Setup

Create a lab directory and a single file:

```bash
mkdir -p lab-01-numpy-drill && cd lab-01-numpy-drill
touch drill.py
```

All code below goes in `drill.py`. Where the lab asks you to "run it",
you can either `python drill.py` from the terminal or paste the block
into a REPL — both are fine.

Start the file with a single import:

```python
import numpy as np
```

## Steps

### Step 1 — Shape, dtype, and the silent integer (≈ 5 min)

Add to `drill.py`:

```python
a = np.array([1, 2, 3])
b = np.array([1.0, 2.0, 3.0])
print(a.shape, a.dtype)   # expect (3,) int64
print(b.shape, b.dtype)   # expect (3,) float64

big = np.int32(2_000_000_000)
print(big + big)          # look carefully at the sign
```

**Expected output.** `(3,) int64`, `(3,) float64`, and a **negative**
number for `big + big` (on most platforms). NumPy integer types wrap
silently — no `OverflowError`.

**Checkpoint.** You just reproduced chapter 02's "integer overflow
silently" failure. What dtype would have caught it? (Answer: a floating
or wider integer type, e.g. `np.int64`, would not overflow at this
range; a `float64` promotes the result to `4e9`.) The habit worth
leaving with: assert `dtype` at function boundaries when it matters.

### Step 2 — Vectorise the mean squared error (≈ 10 min)

Chapter 02 says: "If you wrote a `for` loop over array elements, you
wrote the wrong code." Prove it to yourself.

```python
rng = np.random.default_rng(seed=0)
y      = rng.normal(size=100_000)
y_hat  = y + rng.normal(scale=0.1, size=100_000)

import time

# Slow: Python loop.
t0 = time.perf_counter()
total = 0.0
for i in range(len(y)):
    total += (y_hat[i] - y[i]) ** 2
mse_loop = total / len(y)
t_loop = time.perf_counter() - t0

# Fast: NumPy expression.
t0 = time.perf_counter()
mse_vec = np.mean((y_hat - y) ** 2)
t_vec = time.perf_counter() - t0

print(f"loop MSE: {mse_loop:.6f} in {t_loop * 1000:.1f} ms")
print(f"vec  MSE: {mse_vec:.6f}  in {t_vec * 1000:.1f} ms")
print(f"speedup:  {t_loop / t_vec:.0f}×")
```

Run it.

**Expected output.** Both values agree to ~six decimal places (float
addition order differs, so don't expect equality beyond that). The
vectorised version runs roughly two orders of magnitude faster — the
chapter's "~100× faster" quote is a working rule of thumb, though your
hardware will push it around.

**Checkpoint.** Chapter 02's exact rule you are reproducing: "every
iteration crosses the Python ↔ C boundary". Try bumping the size to
`1_000_000` — the loop gets *linearly* slower; the vectorised
version does too but from a much lower base.

> **Note on float equality.** `mse_loop == mse_vec` will probably
> return `False` even though the values agree to many decimal places.
> Use `np.isclose(mse_loop, mse_vec)` — chapter 02 pitfall #1.

### Step 3 — Broadcasting a mean/std across a feature matrix (≈ 10 min)

Standardise a `(N, D)` matrix with a `(D,)` mean and std. This is
exactly what `StandardScaler` does under the hood, and it's the most
common broadcasting pattern in ML.

```python
X     = rng.normal(loc=5.0, scale=3.0, size=(1000, 4))
mu    = X.mean(axis=0)              # expect shape (4,)
sigma = X.std(axis=0)               # expect shape (4,)

print("X:",     X.shape)
print("mu:",    mu.shape)
print("sigma:", sigma.shape)

X_std = (X - mu) / sigma
print("X_std.mean(axis=0):", X_std.mean(axis=0))
print("X_std.std(axis=0):",  X_std.std(axis=0))
```

**Expected output.** `X_std.mean(axis=0)` is ~`[0, 0, 0, 0]` and
`X_std.std(axis=0)` is ~`[1, 1, 1, 1]` (up to tiny floating-point
error).

**Checkpoint.** Broadcasting aligned the `(4,)` mean against the
`(1000, 4)` matrix "from the right", applying it per column across
rows. If instead `mu` had shape `(4, 1)`, this expression would error
or do the wrong thing — align *from the right* is the rule to
remember.

### Step 4 — The `(N, 1)` vs. `(N,)` trap (≈ 5 min)

Broadcasting is a sharp tool. The chapter calls out one trap by name:
a 2-D column where you expected a 1-D vector will broadcast
*differently*.

```python
y_1d = np.arange(5)             # shape (5,)
y_2d = y_1d.reshape(5, 1)       # shape (5, 1) — looks the same, broadcasts differently

print("y_1d - y_1d:")
print(y_1d - y_1d)              # shape (5,) — zeros

print("y_2d - y_1d:")
print(y_2d - y_1d)              # shape (5, 5)! — the outer broadcast
```

**Expected output.** The first subtraction is a vector of zeros. The
second produces a **5 × 5 matrix** — because `(5, 1)` and `(5,)`
align to `(5, 1)` against `(1, 5)`, outer-product-style. If you had
expected zeros, you would spend a confused twenty minutes debugging.

**Checkpoint.** The chapter's fix: when you pull a single column out
of a DataFrame, prefer `df["col"].to_numpy()` (1-D) over
`df[["col"]].to_numpy()` (2-D). Or call `.ravel()` on the array to
flatten.

### Step 5 — Views, copies, and a mutation that leaks (≈ 10 min)

The last habit: know what mutates the original.

```python
X = np.arange(12).reshape(3, 4)
print("Original X:\n", X)

view_slice = X[0:2, :]          # basic slicing → view
view_slice[0, 0] = 999

print("After mutating view_slice[0, 0]:\n", X)   # X[0, 0] is now 999

X = np.arange(12).reshape(3, 4)
fancy_slice = X[[0, 1], :]      # fancy indexing → copy
fancy_slice[0, 0] = 999

print("After mutating fancy_slice[0, 0]:\n", X)  # X is unchanged
```

**Expected output.** The first mutation leaks into `X`; the second
does not. Two syntactically similar slices with opposite semantics.

**Checkpoint.** The chapter's defensive rule: **if you intend to
mutate, call `.copy()` explicitly and never rely on the view-vs-copy
behaviour of a particular expression.** Rewrite the first block with
`view_slice = X[0:2, :].copy()` and confirm `X` is now untouched.

### Step 6 — Sanity-check your reductions (≈ 5 min)

Chapter 02 pitfalls #2 and #5: always pass an explicit `axis=`, and
beware of NaN propagation.

```python
M = np.array([[1.0, 2.0, 3.0],
              [4.0, np.nan, 6.0]])

print("M.sum():",           M.sum())                   # one scalar, NaN-tainted
print("M.sum(axis=0):",     M.sum(axis=0))             # per column, still NaN-tainted
print("np.nansum(M, axis=0):", np.nansum(M, axis=0))   # skips NaN
```

**Expected output.**
```text
M.sum(): nan
M.sum(axis=0): [ 5. nan  9.]
np.nansum(M, axis=0): [5. 2. 9.]
```

**Checkpoint.** Three habits to leave with: (1) `axis=` is almost
never optional, (2) a single NaN silently poisons a reduction, (3)
the `np.nan*` family is the opt-in "skip NaNs" variant — use it when
you *mean* to skip, not by default.

## Expected final artefacts

A single file `lab-01-numpy-drill/drill.py` with the six step blocks
above. Nothing to save to disk; the artefact is in your fingers.

## Final checkpoint

Without re-reading chapter 02, answer:

1. In Step 3, why was `(X - mu) / sigma` correct even though `mu` is
   1-D and `X` is 2-D? State the broadcasting rule in your own words.
2. In Step 4, what is the broadcast shape of `(5, 1) - (5,)`? Why?
3. In Step 5, which of the two slices returned a view and why does
   the view mutate the original?
4. If you had written `X.sum()` on a `(1000, 32)` feature matrix,
   expecting a `(32,)` per-column total — what would you have got
   instead?

If you can answer all four, the three chapter-02 habits are in place
and you can move on to chapter 03.

## Where to go next

- `exercise-01` ("NumPy / Pandas fluency") takes these idioms and
  builds a tested feature-engineering module with `groupby`, `merge`,
  and `validate=` — the Pandas side of the same discipline.
- Chapter 05 then introduces the scikit-learn API, which assumes the
  NumPy fluency this lab drilled in.
