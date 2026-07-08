# 02 — Vectors and matrices for ML

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 01.

## Motivation

Every input you feed an ML model is, eventually, a tensor of floats. A
tabular row, a sentence, an image, an audio clip, a graph — each one gets
packed into a vector or stack of vectors before any maths happens. Every
model's internal state is also a stack of tensors. So before you can read
or debug ML code you need to be fluent in the operations on those objects.

This chapter is a working-engineer's tour of the linear algebra that
appears in real ML library code. It is not a complete textbook treatment —
it is the subset you will see in `numpy`, `pandas`, and `torch` every day.

## Vectors

A **vector** is an ordered list of numbers. In ML we almost always treat it
as a point or a direction in some real-valued space:

```python
import numpy as np
x = np.array([1.0, 2.0, 3.0])     # shape (3,) — a vector in R^3
```

The number of entries is the vector's **dimension** (here, 3). A "feature
vector" for one example with $n$ features is a vector in $\mathbb{R}^n$.

### Three operations to internalise

1. **Addition / scaling.** Element-wise; intuitively, "moving" a point or
   stretching a direction.
2. **Dot product.** $\langle x, y \rangle = \sum_i x_i y_i$. Measures
   similarity of direction: maximal when $x$ and $y$ point the same way,
   zero when they are orthogonal, negative when they point opposite ways.
3. **Norm.** $\|x\|_2 = \sqrt{\sum_i x_i^2}$ — the Euclidean length. Other
   norms exist (`L1`, `L∞`) but `L2` is the default. A "unit vector" has
   norm 1; you produce one by dividing by the norm.

```python
y = np.array([4.0, 5.0, 6.0])
np.dot(x, y)               # 32.0
np.linalg.norm(x)          # 3.7416...
x / np.linalg.norm(x)      # unit vector in the direction of x
```

### Why these matter in ML

- The **dot product** is the kernel of linear models: a prediction is
  `w · x + b`. Every linear regression, every logistic regression, every
  attention head, every embedding-similarity score reduces to dot
  products.
- The **norm** appears in regularisation (penalising `||w||²`), in
  gradient clipping, and in cosine similarity (a dot product after
  normalising both vectors to unit length).

## Matrices

A **matrix** is a rectangular grid of numbers — equivalently, a stack of
vectors. Shape `(m, n)` means `m` rows, `n` columns. The convention in ML
is that a **batch** of $m$ examples with $n$ features each is a matrix
$X \in \mathbb{R}^{m \times n}$ — examples down the rows, features across
the columns.

```python
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])    # shape (2, 3): 2 examples, 3 features
```

### Matrix multiplication

Matrix multiplication `A @ B` is defined when the inner dimensions match:
`(m, k) @ (k, n) → (m, n)`. Each output entry is a dot product between a
row of `A` and a column of `B`.

```python
W = np.array([[0.5, -0.1],
              [0.2,  0.3],
              [0.4,  0.6]])         # shape (3, 2)
X @ W                                 # shape (2, 2) — one row per example, one column per output unit
```

In ML terms: this is a **linear layer**. `X` is your batch, `W` is the
weight matrix, and `X @ W` produces a batch of pre-activations. Almost
every neural network layer reduces, at its core, to a matrix multiply.

### Reading shapes is debugging

Half the bugs in production ML code are shape mismatches. Build the habit
of writing the shape of every tensor on the right side of the line — both
in real code (as a comment or a `# (B, T, D)`-style annotation) and in your
head when reading a paper. If you can't say "this is shape `(batch,
sequence, embedding)`" out loud, you don't yet understand the line.

### Broadcasting

NumPy and PyTorch will silently expand mismatched-but-compatible shapes.
This is fast and convenient and the single most common source of
"works on my machine, breaks at scale" bugs.

```python
X        # shape (2, 3)
bias     # shape (3,)
X + bias # OK — bias broadcasts across rows. shape (2, 3).

bad      # shape (2,)
X + bad  # ERROR — last dims don't match.
```

The rule: aligned from the right, dimensions must be equal, one of them
must be 1, or one must be missing entirely. Read the
[NumPy broadcasting docs](https://numpy.org/doc/stable/user/basics.broadcasting.html)
and the
[PyTorch broadcasting semantics page](https://pytorch.org/docs/stable/notes/broadcasting.html)
once, carefully — it pays back forever.

## Special structures worth recognising

- **Identity matrix `I`.** Square matrix with 1s on the diagonal, 0s
  elsewhere. `I @ A == A`. Comes up in regularisation: `(X^T X + λI)`
  in ridge regression.
- **Transpose `A.T`.** Flips rows and columns. Shape `(m, n) → (n, m)`.
  You'll see `X.T @ X` constantly — it produces the symmetric feature
  covariance-shaped matrix used by least squares and PCA.
- **Symmetric matrix.** `A == A.T`. Covariance matrices, gram matrices, and
  many kernels are symmetric. Symmetry buys you faster algorithms and
  nicer eigenvalue properties.
- **Diagonal matrix.** Non-zero only on the diagonal. Multiplying by a
  diagonal matrix is the same as element-wise scaling each column (or
  row). Often stored as a 1-D vector, never as a full matrix.
- **Orthogonal matrix.** `Q.T @ Q == I`. Acts as a rigid rotation /
  reflection — preserves lengths and angles. Foundational for PCA and the
  SVD.

## A worked example: linear regression as one matrix equation

The simplest supervised model is linear regression: given examples
$X \in \mathbb{R}^{m \times n}$ and targets $y \in \mathbb{R}^m$, find
weights $w \in \mathbb{R}^n$ that minimise the squared error
$\|Xw - y\|_2^2$. The closed-form solution is the **normal equation**:

$$w^* = (X^\top X)^{-1} X^\top y$$

Read the shapes: $X^\top X$ is $(n, n)$, $X^\top y$ is $(n,)$, so $w^*$ is
$(n,)$. The expression looks compact only because of the linear-algebra
notation — written out element by element it is a wall of sums. This is
the recurring pattern: linear algebra is the compression layer that lets
ML papers fit on one page.

```python
# Toy example
m, n = 100, 3
rng = np.random.default_rng(0)
X = rng.normal(size=(m, n))
true_w = np.array([1.5, -2.0, 0.7])
y = X @ true_w + rng.normal(scale=0.1, size=m)

w_hat = np.linalg.solve(X.T @ X, X.T @ y)
print(w_hat)   # close to [1.5, -2.0, 0.7]
```

Note the use of `np.linalg.solve` instead of literally computing
`np.linalg.inv(X.T @ X) @ (X.T @ y)`. Both compute the same mathematical
quantity, but `solve` is numerically better-behaved — a hint at the
numerical-implementation concerns that `mod-102` picks up.

## Tensors: a one-line generalisation

A **tensor** in ML usage is just an n-dimensional array — a generalisation
of scalar (0-D), vector (1-D), matrix (2-D) to arbitrary rank. A batch of
RGB images is a 4-D tensor of shape `(batch, channels, height, width)`. A
batch of token sequences with embeddings is a 3-D tensor of shape
`(batch, sequence_length, embedding_dim)`.

Everything in this chapter generalises: dot products become tensor
contractions, matrix multiplications become `einsum` expressions. You can
get a long way before you need that generalisation; for now, recognise
that shapes are still the language.

## Summary

- A vector is a list of numbers; a matrix is a stack of vectors; a tensor
  is the n-D generalisation.
- The three operations to internalise are addition/scaling, the dot
  product, and the norm.
- Matrix multiplication is the workhorse of ML — every linear layer, every
  attention head reduces to it.
- Always know the shape of every tensor on every line. Most ML bugs are
  shape bugs.
- The notation is dense on purpose: it hides nested sums so that papers
  and library APIs fit in the reader's head.

## Where this goes next

Chapter 03 introduces the calculus side — derivatives and gradients —
which combined with matrices gives you the language of training. Chapter
06 then connects the two via the loss-and-optimisation engine.
