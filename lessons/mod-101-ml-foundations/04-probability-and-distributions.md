# 04 — Probability and distributions

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 01.

## Motivation

A model is, almost always, an estimate of a probability distribution.
Classifiers output `P(class | features)`. Regressors implicitly assume a
distribution of the residual. Generative models sample from a learned
$P(x)$. Loss functions like cross-entropy come straight out of
probability theory.

If you treat probability as "uncertainty maths" you will keep making the
same family of mistakes — confusing a probability with a confidence,
treating a 0.9 calibrated probability the same as a 0.9 raw score, or
evaluating a classifier on an unrepresentative split. This chapter is a
short refresher aimed at the specific probability concepts you will rely
on in this curriculum.

## The three rules of probability

You only need three:

1. **Non-negativity.** $P(A) \geq 0$ for any event $A$.
2. **Normalisation.** $P(\Omega) = 1$ — the probability of *some* outcome
   in the sample space $\Omega$ is one.
3. **Additivity.** For disjoint events, $P(A \cup B) = P(A) + P(B)$.

Everything else — conditional probability, independence, Bayes — is a
consequence.

### Conditional probability and independence

The probability of $A$ **given** $B$ is

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} \quad \text{(when } P(B) > 0\text{)}.$$

Events $A$ and $B$ are **independent** when $P(A \cap B) = P(A) \cdot
P(B)$. Equivalently, $P(A \mid B) = P(A)$ — knowing $B$ tells you
nothing new about $A$.

The most useful identity that drops out of these definitions is **Bayes'
rule**:

$$P(H \mid D) = \frac{P(D \mid H) \cdot P(H)}{P(D)}.$$

Read in words: the probability of a hypothesis $H$ given data $D$ is the
likelihood of the data under that hypothesis, times the prior probability
of the hypothesis, divided by the marginal probability of the data. This
is the engine of every Bayesian analysis and the conceptual underpinning
of half the loss functions in this module. Chapter 05 dwells on it.

## Random variables

A **random variable** (RV) is a function from outcomes to numbers. "Roll
a die" is an experiment; "the number that comes up" is a random variable.
In ML, every feature column in your dataset is a realisation of some
underlying random variable, and the model's prediction is a function of
those RVs.

RVs come in two flavours:

- **Discrete** — finitely or countably many values (a class label, a
  count). Characterised by a **probability mass function (PMF)**
  $p(x) = P(X = x)$.
- **Continuous** — values in a continuous range (a length, a temperature).
  Characterised by a **probability density function (PDF)** $p(x)$, where
  $P(a \le X \le b) = \int_a^b p(x)\,dx$. A density at a point is *not*
  a probability — it is a probability per unit length.

### Expectation and variance

The **expectation** $\mathbb{E}[X]$ is the long-run average value of the
RV: $\sum_x x p(x)$ in the discrete case, $\int x p(x)\,dx$ in the
continuous case. Expectation is linear: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$,
even when $X$ and $Y$ are dependent.

The **variance** $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$
measures the spread. Its square root is the **standard deviation**, in
the same units as $X$.

In ML you usually approximate these with empirical estimates over your
dataset:

```python
import numpy as np
x = np.array([...])
mean = x.mean()
var = x.var(ddof=1)        # unbiased estimator
std = x.std(ddof=1)
```

The `ddof=1` divides by `n - 1` instead of `n`, producing the unbiased
estimator of the population variance. The default in NumPy is `ddof=0`
(maximum-likelihood estimator) — a small but worth-knowing footgun.

## Distributions you must recognise

These appear constantly in ML code and papers; you should be able to
name each one on sight, list its parameters, and know one ML place where
it shows up.

### Bernoulli($p$)

Two outcomes, 1 with probability $p$ and 0 with probability $1-p$. PMF:
$p^x(1-p)^{1-x}$. The likelihood of a single binary label under a
classifier's predicted probability — this is the building block of binary
cross-entropy.

### Categorical($\boldsymbol{\pi}$)

$K$ outcomes with probabilities $\pi_1, \dots, \pi_K$ summing to 1. The
likelihood of a single multi-class label — and the building block of
softmax cross-entropy.

### Binomial($n, p$)

Sum of $n$ i.i.d. Bernoulli($p$) trials. Appears in A/B testing math and
in any "k successes out of n" reasoning about classifier metrics.

### Gaussian (Normal) $\mathcal{N}(\mu, \sigma^2)$

The bell curve. PDF $\propto \exp(-(x - \mu)^2 / (2\sigma^2))$.
Foundational because:

- Most regression losses (squared error) correspond to a Gaussian noise
  model.
- The central limit theorem says sums of many i.i.d. RVs are
  approximately Gaussian regardless of their original distribution. That
  is why so many natural measurements look Gaussian.
- It is the most common initialisation distribution for neural-network
  weights.

The **multivariate Gaussian** $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$
generalises to vectors; the covariance matrix $\boldsymbol{\Sigma}$
encodes which dimensions co-vary.

### Uniform, Exponential, Poisson, Beta, Dirichlet

You will see all of these. Skim a reference like the
[SciPy distributions catalogue](https://docs.scipy.org/doc/scipy/reference/stats.html)
when you encounter one in a paper — you don't need to memorise the PDFs,
but you do need to recognise when you're being shown one.

## Independence vs. i.i.d. vs. conditional independence

Three closely related concepts that are constantly conflated:

- **Independent** — two random variables that share no information.
- **i.i.d.** (independent and identically distributed) — a sequence of
  RVs that are pairwise independent *and* drawn from the same
  distribution. Most classical ML theory assumes the data is i.i.d. (it
  often isn't — time series, grouped data, drifting populations).
- **Conditionally independent given Z** — $A$ and $B$ become independent
  once you know $Z$. The naive Bayes classifier assumes features are
  conditionally independent given the label. Causal reasoning lives
  here.

When a model "works in the lab and fails in production", the i.i.d.
assumption being violated (between train and serving distributions) is
the single most common culprit. Module 109 picks this up in detail.

## Joint, marginal, conditional

For two RVs $X$ and $Y$, three distributions are floating around:

- **Joint** $P(X, Y)$ — probabilities over pairs.
- **Marginal** $P(X) = \sum_y P(X, Y = y)$ — probabilities of $X$ ignoring $Y$.
- **Conditional** $P(Y \mid X) = P(X, Y) / P(X)$ — probabilities of $Y$
  given a particular $X$.

Most predictive models target the conditional $P(Y \mid X)$. Most
generative models target the joint $P(X, Y)$ or the marginal $P(X)$.

## Worked example: empirical distribution of a dataset

A **dataset** $\{x_i\}_{i=1}^N$ defines an **empirical distribution** —
the discrete distribution that places probability $1/N$ on each observed
sample. Many ML losses are exactly the expectation under this empirical
distribution of some per-example loss. "Average squared error on the
training set" is a shorthand for "expected squared error under the
training-set empirical distribution."

This is more than terminology — it is why train/validation/test split
matters. The empirical distribution your model is fit to is *not* the
true underlying distribution; the validation set is a different empirical
sample of the same underlying thing; the test set, again. The whole
generalisation story is about whether your fit to one empirical sample
transfers to the others.

## Summary

- Probability is built on non-negativity, normalisation, and additivity.
- Random variables come as discrete (PMF) or continuous (PDF). PDFs are
  not probabilities — they integrate to probabilities.
- Expectation and variance are the workhorse summaries.
- A handful of distributions — Bernoulli, Categorical, Binomial,
  Gaussian — appear over and over in ML losses and models. Learn them
  cold.
- The i.i.d. assumption is everywhere, often violated, and often the
  reason production models fail.

## Where this goes next

Chapter 05 plugs probability into estimation: given data, how do we pick
the parameters of a distribution? That is the principle that defines
almost every supervised loss you will see.
