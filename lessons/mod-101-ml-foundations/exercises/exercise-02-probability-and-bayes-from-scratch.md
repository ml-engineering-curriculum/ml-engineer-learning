# exercise-02 — Probability and Bayes from scratch

> **Estimated effort:** 3 hours.
> **Companion chapters:** [04 — Probability and distributions](../04-probability-and-distributions.md), [05 — Statistical estimation: MLE, MAP, and Bayes](../05-statistical-estimation.md).

## Objective

Make the abstractions from chapters 04 and 05 concrete. You will (a)
simulate distributions and verify their summary statistics, (b)
implement Bayes' rule on a worked diagnostic-testing example to feel
the base-rate trap, and (c) derive — and verify by simulation — the
maximum likelihood and maximum a posteriori estimates for a Bernoulli
parameter.

## Prerequisites

- Python 3.11+, NumPy ≥ 1.26, SciPy ≥ 1.13, Matplotlib.
- Chapters 04 and 05 of this module.

## Problem statement

Three short sub-problems. They build on each other; do them in order.

### Part A — distributions by simulation (≈ 60 min)

In `distributions.py`, for each of the following distributions:

1. **Bernoulli($p = 0.3$)**
2. **Binomial($n = 20, p = 0.3$)**
3. **Gaussian($\mu = 0, \sigma = 1$)**
4. **Gaussian($\mu = 5, \sigma = 2$)**
5. **Mixture: 70% Gaussian($-2, 1$), 30% Gaussian($3, 0.5$)**

do the following:

- Draw 100,000 samples using `numpy.random.default_rng(0)` (and SciPy
  where useful — but **do not use SciPy's analytical mean/variance
  helpers** for the verification step; compute them by hand from the
  PDFs).
- Compute the empirical mean and variance.
- Derive the closed-form mean and variance analytically; write them in
  a markdown cell or docstring.
- Plot the empirical histogram against the analytical PDF/PMF on the
  same axis.
- Assert that empirical and analytical means agree to within `0.05`
  and variances to within `0.05` (you may need a bigger sample for the
  mixture).

Save the five plots to `distributions/<name>.png`.

### Part B — Bayes' rule and the base-rate trap (≈ 60 min)

You are presented with the textbook diagnostic-testing problem:

> A test for a disease has a sensitivity (true positive rate) of 0.99
> and a specificity (true negative rate) of 0.99. The disease has a
> prevalence (prior) of 0.1% (1 in 1000) in the population. A randomly
> chosen person tests positive. What is the probability they have the
> disease?

In `bayes.py`:

- Implement `posterior_positive(prior, sensitivity, specificity)`
  returning $P(\text{disease} \mid \text{test+})$ using Bayes' rule.
  Derive the formula explicitly in a docstring (no copying from
  Wikipedia — derive it).
- Compute the answer for the numbers above. Print it. Compare it to
  the answer most people would intuit (i.e. "the test is 99%
  accurate so probably 99%"). Briefly comment on the gap.
- Re-compute for **prevalences** of `0.001`, `0.01`, `0.1`, `0.5`.
  Plot the posterior as a function of prevalence on a log-x axis.
- Re-compute for `prevalence = 0.001`, varying **specificity** from
  `0.95` to `0.9999`. Plot the posterior as a function of specificity.

Add to `REPORT.md`:

- The posterior for the original numbers.
- The two plots, with one sentence each on what they show.
- One sentence on why "highly accurate test ⇒ trust a positive
  result" is the wrong intuition.

### Part C — MLE and MAP for a Bernoulli (≈ 60 min)

You observe $N$ i.i.d. coin flips with $k$ heads. Three estimators of
$p$ (the head probability):

- **MLE.** Derive $\hat{p}_{\text{MLE}} = k / N$ from the log-likelihood
  (write the derivation).
- **MAP with a Beta($\alpha, \beta$) prior.** Derive
  $\hat{p}_{\text{MAP}} = (k + \alpha - 1) / (N + \alpha + \beta - 2)$.
- **Posterior mean with a Beta prior.** State (no derivation needed —
  this is a standard result for the Beta-Bernoulli conjugate pair)
  $\hat{p}_{\text{post mean}} = (k + \alpha) / (N + \alpha + \beta)$.

In `estimation.py`:

- Implement all three.
- For a **fair-coin** truth ($p = 0.5$), simulate datasets of size
  $N \in \{5, 20, 100, 1000\}$, 500 trials each. For each $N$, plot
  the distribution of $\hat{p}_{\text{MLE}}$ across trials.
- For $N = 5$, additionally plot the distributions of
  $\hat{p}_{\text{MAP}}$ and $\hat{p}_{\text{post mean}}$ for two
  priors: `Beta(1, 1)` (uniform) and `Beta(10, 10)` (strong prior at
  0.5). What do you observe about variance and bias?
- For a **biased-coin** truth ($p = 0.05$), and a strong prior of
  `Beta(50, 50)` (which mistakenly favours fairness), plot how each
  estimator does at $N = 5, 20, 100, 1000$.

Add to `REPORT.md`:

- Your three derivations (MLE, MAP, posterior mean).
- The plots.
- One paragraph on the bias-variance trade-off you see between
  estimators, and one sentence on when you would prefer the MAP /
  posterior-mean estimator over the MLE in real ML work.

## Starter guidance

- A common source of off-by-one errors in the MAP derivation is the
  difference between the mode (the MAP) and the mean of a Beta
  posterior. Don't mix them up.
- For the diagnostic-test problem, the trap is that the marginal
  $P(\text{test+}) = P(+ \mid D) \cdot P(D) + P(+ \mid \neg D) \cdot
  P(\neg D)$ is dominated by the *false* positives when the prior is
  small.
- Plotting the diagnostic posterior on a log-x axis makes the
  base-rate effect dramatic.

## Acceptance criteria

- Part A: empirical and analytical means agree within `0.05` for all
  five distributions; all five plots exist.
- Part B: `posterior_positive(0.001, 0.99, 0.99)` returns a value
  within `1e-6` of `0.0901...` (you may compute and confirm by hand;
  do not just trust this hint). Both plots exist and are correctly
  labelled.
- Part C: all three estimators implemented; six required plots exist;
  derivations included in `REPORT.md`.

## Stretch goals (optional)

- Implement a **Bayesian update with streaming data**: start with a
  `Beta(1, 1)` prior, see flips one at a time, update the posterior
  after each, and animate (or step-plot) the posterior. This makes
  the conjugacy machinery viscerally obvious.
- Re-do Part B with a more realistic medical example you look up
  (e.g. mammography sensitivities and prevalence). Cite the numbers;
  do not invent them.
- Add a **frequentist 95% confidence interval** for the MLE of $p$
  (use the Wilson interval; do not hand-derive — cite the formula),
  and overlay it on your $\hat{p}_{\text{MLE}}$ plots. Compare to the
  Bayesian 95% credible interval from the Beta posterior.

## What to hand in

```
exercise-02/
├── distributions.py
├── bayes.py
├── estimation.py
├── distributions/
│   ├── bernoulli.png
│   ├── binomial.png
│   ├── gaussian_0_1.png
│   ├── gaussian_5_2.png
│   └── mixture.png
├── bayes/
│   ├── posterior_vs_prevalence.png
│   └── posterior_vs_specificity.png
├── estimation/
│   ├── mle_fair_*.png
│   ├── map_n5_*.png
│   └── biased_coin_*.png
├── tests/
│   └── test_estimators.py
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo.
