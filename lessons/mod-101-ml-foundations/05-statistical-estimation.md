# 05 — Statistical estimation: MLE, MAP, and Bayes

> **Reading time:** ~45 minutes.
> **Prerequisite chapters:** 04.

## Motivation

"Fit a model to data" is so familiar it sounds atomic. It isn't. Under
the hood, fitting is **estimation**: given observations, pick the
parameters of a probability model that best explain them. Different
notions of "best" produce different losses, and almost every supervised
loss you will see in this curriculum drops out of one of two principles:
**maximum likelihood** and **maximum a posteriori**.

Understanding this gives you the answer to questions like:

- Why is squared error the "natural" loss for regression?
- Why is cross-entropy the natural loss for classification?
- What exactly is L2 regularisation doing, in probabilistic terms?
- When should I bother computing a confidence interval, and on what?

## Maximum likelihood estimation (MLE)

Suppose your model says "the data came from this family of distributions
$p(x \mid \theta)$ for some parameter $\theta$." Given a dataset
$\{x_1, \dots, x_N\}$ assumed i.i.d., the **likelihood** of the data is

$$L(\theta) = \prod_{i=1}^N p(x_i \mid \theta).$$

The **maximum likelihood estimate** is the $\theta$ that makes the
observed data most probable:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta L(\theta).$$

In practice we maximise the **log-likelihood** instead — a sum is easier
than a product, and the logarithm is monotonic so the argmax is the same:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log p(x_i \mid \theta) = \arg\min_\theta \left[ -\sum_{i=1}^N \log p(x_i \mid \theta) \right].$$

That last form — **negative log-likelihood (NLL)** — is the bridge to
loss functions. Minimising NLL *is* maximising likelihood.

### MLE recovers familiar losses

#### Squared error from a Gaussian noise model

Suppose for each training pair $(x_i, y_i)$ you assume
$y_i = w \cdot x_i + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$.
Then $p(y_i \mid x_i, w) = \mathcal{N}(w \cdot x_i, \sigma^2)$, and

$$-\log p(y_i \mid x_i, w) = \frac{(y_i - w \cdot x_i)^2}{2\sigma^2} + \text{const}.$$

Summing over the dataset and dropping constants, **the negative
log-likelihood is the sum of squared errors, up to a positive constant**.
Maximum likelihood under a Gaussian noise model *is* least squares.

#### Binary cross-entropy from a Bernoulli model

For a binary classifier, assume $y_i \mid x_i \sim \text{Bernoulli}(\hat{y}_i)$
where $\hat{y}_i = \sigma(w \cdot x_i)$. Then

$$-\log p(y_i \mid x_i, w) = -\bigl[y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)\bigr].$$

That is exactly the binary cross-entropy formula. Maximum likelihood
under a Bernoulli model *is* logistic regression.

The pattern is general: **pick a distribution for the noise / labels,
write down the NLL, and you have your loss function**. This is one of the
most leveraged ideas in this curriculum.

## Maximum a posteriori (MAP) estimation

MLE treats $\theta$ as an unknown but fixed quantity. **Bayesian
estimation** treats $\theta$ as a random variable with a **prior**
$p(\theta)$, and looks at the **posterior** after seeing the data:

$$p(\theta \mid \text{data}) \propto p(\text{data} \mid \theta) \cdot p(\theta).$$

The **MAP estimate** is the mode of the posterior — the most probable
value of $\theta$ under it:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \bigl[\log p(\text{data} \mid \theta) + \log p(\theta)\bigr].$$

That extra $\log p(\theta)$ term is the **regulariser**. Concretely:

- A Gaussian prior $p(\theta) \propto \exp(-\|\theta\|_2^2 / (2\tau^2))$
  contributes a term proportional to $\|\theta\|_2^2$ — this is **L2
  (ridge) regularisation**.
- A Laplace prior contributes a term proportional to $\|\theta\|_1$ —
  this is **L1 (lasso) regularisation**.

Read this carefully: **L2 regularisation is a Gaussian prior on the
weights**. L1 is a Laplace prior. The regularisation coefficients you
sweep over in `mod-104` correspond to the variance of those priors.
Once you see this you cannot un-see it.

## Bayesian inference (one paragraph)

Full Bayesian inference does not collapse the posterior to a single
point estimate. It carries the entire posterior around and integrates
over it to make predictions:

$$p(y_{\text{new}} \mid x_{\text{new}}, \text{data}) = \int p(y_{\text{new}} \mid x_{\text{new}}, \theta) \cdot p(\theta \mid \text{data})\, d\theta.$$

This is theoretically lovely and computationally painful — closed-form
posteriors exist only for nice conjugate pairs, and approximate methods
(MCMC, variational inference) are an entire subfield. As a level-20 ML
engineer you should *know this is the principled thing*, *recognise its
artefacts* (Bayesian neural nets, MC dropout, Gaussian processes), and
*default to MLE or MAP unless uncertainty quantification is the actual
business requirement*.

## Estimation has uncertainty

Even MLE point estimates come with uncertainty: with a small dataset, a
different sample would have given a different $\hat{\theta}$. Two
language families exist for talking about this:

### Frequentist: confidence intervals

A 95% confidence interval is a procedure: if you repeated the experiment
many times, 95% of the intervals produced would contain the true
parameter. It is *not* a statement that the true parameter is in this
particular interval with probability 0.95 — a common informal slip.

### Bayesian: credible intervals

A 95% credible interval is a statement about the posterior: there is a
0.95 posterior probability that the parameter lies in this interval. This
is the interpretation people *intuitively* want; it is also the more
demanding one (you must commit to a prior).

In day-to-day ML work you will most often use confidence intervals
around metrics — accuracy, AUC, mean squared error — computed by
bootstrap. The bootstrap (resample the test set with replacement many
times and look at the spread) gives you a working uncertainty estimate
without needing to commit to a parametric distribution.

## Hypothesis testing (just enough)

You will encounter hypothesis tests in two places: A/B test analysis
(does this model beat that one?) and the kind of "is this drift
significant?" check from `mod-109`.

The framework: state a **null hypothesis** $H_0$ (usually "the metric
has not changed" or "the two models are equivalent"), compute a test
statistic, compute the probability of seeing a statistic at least this
extreme under $H_0$ (the **p-value**), and reject $H_0$ if the p-value
falls below a threshold $\alpha$ (commonly 0.05).

Three pitfalls worth flagging:

1. **A p-value is not the probability that $H_0$ is true.** It is a
   conditional probability the other way round.
2. **Multiple comparisons inflate false positives.** If you run 20
   independent tests at $\alpha = 0.05$, you expect one to be
   "significant" by chance. Use a correction (Bonferroni, BH-FDR).
3. **Statistical significance is not practical significance.** With
   enough data, you can detect tiny effects that don't matter.

Module 107 returns to hypothesis testing in the context of model
comparison and metric evaluation; here, the goal is just to make sure
the vocabulary is in your head.

## A unifying picture

Almost every supervised model in this curriculum follows the same
template:

1. Choose a **model family** $p(y \mid x, \theta)$ — what kind of
   distribution will the model output?
2. Pick a **prior** $p(\theta)$ if you want regularisation; skip it if
   you don't.
3. Form the **objective** $-\sum_i \log p(y_i \mid x_i, \theta) -
   \log p(\theta)$.
4. **Minimise** the objective with respect to $\theta$ using gradient
   descent (chapter 06).
5. **Predict** by feeding new $x$ through the trained model.

Every loss function you will write down — squared error, cross-entropy,
hinge, Poisson NLL — fits in step 3. Every regulariser fits in step 2.
Holding this template in your head is the single biggest payoff of this
chapter.

## Summary

- MLE picks $\theta$ to maximise the probability of the data; equivalently
  it minimises the negative log-likelihood, which *is* the loss function.
- Squared error = MLE under Gaussian noise. Cross-entropy = MLE under
  Bernoulli / categorical models.
- MAP adds a prior, which appears as a regulariser. L2 ↔ Gaussian prior,
  L1 ↔ Laplace prior.
- Bayesian inference keeps the whole posterior — principled but
  expensive; use it when uncertainty is the deliverable.
- Estimates have uncertainty. Use bootstrap intervals for metrics. Use
  hypothesis tests carefully, and never confuse statistical significance
  with practical significance.

## Where this goes next

Chapter 06 picks up step 4 of the template above — how do we actually
minimise these objectives, and what should we expect from the
optimisation in practice?
