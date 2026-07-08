# 02 — NumPy for ML

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 01. The vector / matrix vocabulary from
> [`mod-101` chapter 02](../mod-101-ml-foundations/02-vectors-and-matrices-for-ml.md)
> is assumed.

## Motivation

NumPy is the ndarray library at the bottom of almost every Python ML
stack — Pandas, scikit-learn, SciPy, JAX, and PyTorch all interoperate
with it. Becoming idiomatic with NumPy is therefore not optional; it
is the cost of admission for everything downstream.

This chapter does not re-teach what an array is (see `mod-101`
chapter 02). It teaches the four habits that distinguish a working
ML engineer from someone who "knows NumPy enough to get by":

1. Think in shapes and dtypes.
2. Vectorise — never loop in Python over array elements.
3. Use broadcasting deliberately, not accidentally.
4. Use the right random API.

## Shape and dtype: half the bugs are here

Every NumPy array has a `shape` and a `dtype`. Both matter in
production.

```python
import numpy as np

x = np.array([1, 2, 3])
x.shape          # (3,)
x.dtype          # dtype('int64')  — note: integer, not float!

y = np.array([1.0, 2.0, 3.0])
y.dtype          # dtype('float64')
```

Two failure modes to be aware of:

- **Integer overflow silently.** `np.int32(2_000_000_000) + np.int32(2_000_000_000)` is negative on most platforms. You will not get a Python `OverflowError`; you will get a quiet wrong number.
- **Mixed dtype slows you down.** A `float64` array operated on with a
  `float32` array forces an upcast, doubling memory pressure. If you
  are training on a GPU later, you almost always want `float32`.

A defensive habit: assert the dtype and shape at function boundaries
when you care.

```python
def standardise(X: np.ndarray) -> np.ndarray:
    assert X.ndim == 2, X.shape
    assert np.issubdtype(X.dtype, np.floating), X.dtype
    return (X - X.mean(axis=0)) / X.std(axis=0)
```

## Vectorisation

A Python `for` loop over array elements is roughly 100× slower than
the equivalent NumPy expression, because every iteration crosses the
Python ↔ C boundary. The cardinal NumPy rule:

> **If you wrote a `for` loop over array elements, you wrote the wrong code.**

A working example. Suppose you have predictions `y_hat` and labels
`y`, both shape `(n,)`, and you want the mean squared error.

```python
# The wrong way (slow, also: not how anyone reads it).
total = 0.0
for i in range(len(y)):
    total += (y_hat[i] - y[i]) ** 2
mse_loop = total / len(y)

# The right way.
mse = np.mean((y_hat - y) ** 2)
```

The vectorised version is ~100× faster *and* fits in your head.
Almost every "wrong-shaped" scalar operation has a vectorised
counterpart:

| Want | Use |
|---|---|
| Sum of squared errors | `np.sum((a - b) ** 2)` |
| Element-wise max | `np.maximum(a, b)` |
| Argmax per row | `arr.argmax(axis=1)` |
| Boolean mask | `mask = a > 0; a[mask]` |
| Per-row mean | `arr.mean(axis=1)` |

When you genuinely need a non-vectorisable per-element function (rare
in practice), reach for `np.vectorize` only as a last resort — it is
mostly syntactic sugar over a Python loop and gives almost none of
the speed-up. Better options: `np.where`, `np.select`, or rewriting
your function to take an array directly.

## Broadcasting, deliberately

Broadcasting is the rule that lets NumPy operate on arrays of
different shapes:

> Aligned from the right, two dimensions are compatible when they
> are equal, when one of them is 1, or when one of them is missing
> entirely.

A typical ML use of broadcasting: standardising a `(N, D)` feature
matrix with a `(D,)` mean and standard deviation.

```python
X      # shape (1000, 32)
mu     # shape (32,)
sigma  # shape (32,)
X_std = (X - mu) / sigma   # shape (1000, 32) — broadcasts the (32,) across rows.
```

This is exactly what `sklearn.preprocessing.StandardScaler` does
under the hood. Master the rule and you will recognise it everywhere.

**Two broadcasting traps worth knowing:**

1. **A `(N, 1)` vs. `(N,)` mismatch.** Pandas often returns a 2-D
   column where you expect a 1-D vector. `X - y` where `y` is `(N, 1)`
   broadcasts differently than `(N,)`, and the bug usually shows up
   downstream as a confusing shape error. When you slice a single
   column, use `.ravel()` or `df["col"].to_numpy()` rather than
   `df[["col"]].to_numpy()`.
2. **Memory explosion.** Broadcasting `a[:, None]` of shape `(N, 1)`
   with `b[None, :]` of shape `(1, M)` allocates an `(N, M)` array.
   For `N = M = 100_000` of `float64`, that's 80 GB. Do the maths
   before you write the expression.

Read the [NumPy broadcasting reference](https://numpy.org/doc/stable/user/basics.broadcasting.html)
once carefully — it is the source of half the shape bugs you will
encounter.

## Indexing: views, copies, and the gotcha

NumPy indexing comes in two flavours:

- **Basic slicing** (`X[1:5, :]`) returns a **view**. Mutating the
  view mutates the original.
- **Advanced (fancy) indexing** (`X[[0, 3, 4], :]`, boolean masks)
  returns a **copy**.

This matters because you will write code that looks identical and
behaves differently.

```python
X = np.arange(12).reshape(3, 4)
view = X[0:2, :]      # view — shares memory with X
view[0, 0] = 999      # mutates X[0, 0] too

X = np.arange(12).reshape(3, 4)
copy = X[[0, 1], :]   # copy — independent
copy[0, 0] = 999      # does NOT mutate X
```

The defensive rule: if you intend to mutate, call `.copy()`
explicitly and never rely on the view-vs-copy behaviour of a
particular expression.

## The random API: use `default_rng`, not the legacy global

The modern NumPy random API is the
[`np.random.Generator`](https://numpy.org/doc/stable/reference/random/generator.html)
returned by `np.random.default_rng(seed)`. The legacy
`np.random.seed(0)` / `np.random.randn(...)` global is still around
for backward compatibility but is discouraged for new code because
it relies on hidden global state — a recipe for hard-to-reproduce
runs.

```python
# Good — explicit, threadable, reproducible.
rng = np.random.default_rng(seed=0)
X = rng.normal(size=(100, 5))
idx = rng.choice(100, size=10, replace=False)

# Avoid — uses the legacy global state.
np.random.seed(0)
X = np.random.randn(100, 5)
```

The `default_rng(seed)` form is the canonical way to make your
training data, splits, and initialisations reproducible. We will
re-encounter it in `mod-106` (experiment tracking).

## Useful operations you will reach for daily

Below the surface, NumPy's surface area is huge. The shortlist worth
having in your fingers:

- `np.zeros`, `np.ones`, `np.empty`, `np.arange`, `np.linspace`
- `np.concatenate`, `np.stack`, `np.split`, `np.reshape`,
  `np.transpose`, `arr.T`
- `np.sum`, `np.mean`, `np.std`, `np.var`, `np.median`, `np.percentile`
  (always with the explicit `axis=` you intend)
- `np.argmax`, `np.argmin`, `np.argsort`, `np.unique`,
  `np.bincount`
- `np.clip`, `np.where`, `np.select`, `np.maximum`, `np.minimum`
- `np.isfinite`, `np.isnan`, `np.allclose`
- `np.linalg.norm`, `np.linalg.solve`, `np.linalg.lstsq`,
  `np.linalg.inv` (last resort)

You will not memorise these from a list; you will memorise them from
solving exercises. The reference is one tab-completion away.

## Pitfalls collected

A working-engineer's list of NumPy gotchas, accumulated from real
bugs in real codebases:

1. **`==` comparison on floats** is almost never what you want. Use
   `np.isclose` or `np.allclose`.
2. **`x.sum()` without `axis=`** sums *all* elements, which is rarely
   what you mean when `x` is 2-D. Always pass `axis=`.
3. **Mixing Python `list` and `np.ndarray`** sometimes silently
   demotes you back to a Python container. Convert to an array at
   function entry.
4. **`np.array(some_pandas_series)`** can drop the dtype information
   (e.g. nullable Int64 → object). Prefer `series.to_numpy()` and
   pass an explicit `dtype=`.
5. **NaN propagation.** Most NumPy reductions return `nan` if any
   input is `nan`. Use `np.nanmean`, `np.nansum`, etc., when you mean
   "skip NaNs" — but be deliberate about the choice; silent NaNs are
   a feature, not a bug.

## Summary

- Shape and dtype are first-class objects; check them at function
  boundaries.
- Replace every per-element Python loop with a NumPy expression.
- Broadcast deliberately; know when an expression silently allocates
  a huge intermediate array.
- Know the view-vs-copy distinction; never rely on which one a
  particular slice returns.
- Use `default_rng(seed)` for everything stochastic; the legacy
  global API is a reproducibility hazard.

## Where this goes next

Chapter 03 layers Pandas on top of NumPy — every Pandas operation
ultimately becomes NumPy underneath, but the row-vs-column-vs-index
abstractions buy you something. Chapter 05 picks up the
scikit-learn API, which assumes NumPy fluency.
