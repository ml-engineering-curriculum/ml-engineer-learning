# 08 — Regression vs. classification vs. ranking

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 04, 05, 07.

## Motivation

Once you've decided you have a supervised problem, you still have to
pick the *output* framing. The same business question — "which support
tickets should we route to the on-call engineer?" — could be modelled
as:

- a **regression** (predict the urgency score),
- a **classification** (predict the binary "on-call vs. queue" label),
  or
- a **ranking** (sort all open tickets by urgency).

Each choice optimises a different objective and produces a different
failure mode. The wrong choice is a quiet kind of bug — the model
trains, the loss decreases, the demo works, and yet the system
underperforms in production. This chapter gives you the vocabulary to
make the choice deliberately.

## Regression

**Output:** a continuous number $\hat{y} \in \mathbb{R}$ (or sometimes
$\hat{y} \in \mathbb{R}^k$ for multi-output regression).
**Typical loss:** mean squared error (Gaussian noise model) or mean
absolute error (Laplace / more robust to outliers).
**Typical baselines:** predict the mean of $y$; predict the median;
linear regression on a small feature subset.
**Typical metrics:** root mean squared error (RMSE), mean absolute
error (MAE), $R^2$. Module 107 expands on these.

Regression is the right framing when:

- The target is *naturally* continuous (price, latency, temperature).
- You need a calibrated magnitude, not just a category — "expected
  revenue per click", "predicted delivery time in hours".
- The downstream decision depends on the predicted *value*, not just
  whether it crosses a threshold.

### Common regression pitfalls

- **Targets on wildly different scales.** Predicting prices that span
  cents to millions with MSE punishes errors on the high end far more
  than on the low end. Log-transform the target (`log1p(y)` and predict
  in log space).
- **Heavy-tailed targets.** A handful of huge outliers can dominate
  training. Use MAE or Huber loss; consider clipping; consider
  quantile regression if you actually care about specific percentiles
  rather than the mean.
- **Censored or truncated targets.** "Time to event" data where the
  event hasn't happened yet (survival analysis) violates the basic
  regression setup. Specialised models exist.
- **Multi-output regression with correlated outputs.** Predicting each
  output independently ignores the correlation structure. Often okay,
  but be aware.

## Classification

**Output:** a discrete category. Sub-flavours:

- **Binary** — two classes (spam/ham, fraud/not).
- **Multi-class** — one of $K$ mutually exclusive classes (digit
  recognition, document topic).
- **Multi-label** — any subset of $K$ labels can apply (tags on a blog
  post).

**Typical loss:** cross-entropy (binary or categorical). For
multi-label, a sum of binary cross-entropies (each label is its own
yes/no).
**Typical baselines:** predict the majority class; predict the class
prior; logistic regression on a small feature subset.
**Typical metrics:** accuracy, precision, recall, F1, ROC-AUC,
PR-AUC, calibration (Brier score, reliability diagrams), log loss.

A trap right at the start: **a classifier almost always actually
predicts a probability or score, and the "class" is what you get after
applying a threshold.** That threshold is a separate decision —
business-driven, not model-driven. Many models trained as
"classifiers" are best evaluated as scorers, with the threshold tuned
on a validation set against a cost function.

### When binary, when multi-class, when multi-label?

- **Binary** if the decision is yes/no and the two classes are
  meaningful in themselves.
- **Multi-class** if there are $K > 2$ mutually exclusive options.
  Reduce to binary (one-vs-rest) only when forced to.
- **Multi-label** if an example can carry several labels at once.
  Critical distinction: a multi-class model with $K = 5$ outputs one
  label; a multi-label model with 5 labels outputs any subset of them.

### Class imbalance

The biggest classification footgun. If 99% of your examples are class
0, predicting "class 0" always gets you 99% accuracy and a useless
model. Three responses, in increasing order of effort:

1. **Use a better metric.** Accuracy hides imbalance; precision /
   recall / PR-AUC do not.
2. **Adjust the decision threshold.** Most classifiers output a
   probability. A 0.5 threshold is arbitrary; pick a threshold that
   trades off precision and recall the way the business wants.
3. **Resample or reweight.** Oversample the minority, undersample the
   majority, or weight examples in the loss. This can help — and can
   also distort calibration. Module 107 covers this.

### Calibration

A classifier is **well-calibrated** if its predicted probabilities
match observed frequencies — i.e. when it says "70% likely", it is
right about 70% of the time. Calibration matters whenever the
*probability* is the product, not the predicted class. Many
high-accuracy classifiers — boosted trees in particular — are
systematically miscalibrated and benefit from post-hoc calibration
(Platt scaling, isotonic regression).

## Ranking

**Output:** a score (or set of scores) per item; the predicted *order*
is what matters, not the predicted value.
**Typical loss:** pointwise (treat as regression / classification on
relevance), pairwise (e.g. hinge or logistic loss on pairs of items),
or listwise (e.g. LambdaRank / LambdaMART, ListNet, ListMLE).
**Typical baselines:** sort by a popularity signal; sort by a
hand-crafted score; logistic regression treated pointwise.
**Typical metrics:** NDCG@k, MAP@k, MRR, precision-at-k,
recall-at-k.

Ranking is the right framing when:

- You always present a **list** of items to the user (search results,
  recommendations, candidate triage).
- You only have **relative** preference signals (this clicked vs. that
  didn't), not absolute target values.
- The downstream system only cares about the top-$k$ — the absolute
  scores for items at position 100+ don't matter.

### Why ranking is its own thing

You can technically tackle a ranking problem as classification (binary
relevant / not) or regression (predict the relevance score). It often
works. But:

- A regression model can be "wrong" about absolute scores while still
  producing the correct order. Penalising it for absolute errors that
  don't affect the order wastes capacity.
- A pointwise classifier doesn't know that two items are being
  compared at the same query. Pairwise and listwise losses use that
  structure.
- Ranking metrics like NDCG weight the top of the list more heavily.
  Pointwise losses don't naturally do this.

For most simple in-house ranking problems a well-tuned pointwise
logistic regression or gradient-boosted classifier is a strong baseline
and often what gets shipped. Reach for pairwise or listwise losses
when the metric you care about is genuinely ranking-shaped and the
pointwise approach is provably leaving NDCG on the table.

## How to choose: a checklist

For any new supervised problem, work through this list:

1. **What does the downstream decision look like?**
   - Use the predicted *number* → regression.
   - Trigger an action when a probability crosses a threshold →
     classification (often scoring + a tuned threshold).
   - Pick the top few items from many → ranking.
2. **What signal do you actually have?**
   - Continuous targets → regression or classification on a thresholded
     version (rarely a good idea — you lose information).
   - Discrete labels → classification.
   - Pairwise preferences / clicks / ordinal feedback → ranking.
3. **What is the natural noise model?**
   - Symmetric, bell-shaped errors → squared-error regression.
   - Heavy-tailed → MAE or Huber.
   - Counts → Poisson regression.
   - Binary outcome → Bernoulli classification.
4. **What metric will the business actually inspect?**
   - If it's RMSE, you have a regression problem.
   - If it's precision-at-k or NDCG, you have a ranking problem.
   - If it's accuracy or F1, you have a classification problem — but
     check that accuracy isn't hiding imbalance.

If your answers to these questions conflict, that is information: it
usually means the business is asking for one thing and the data
supports another. Surface that tension before you train.

## A worked example: support-ticket triage

> *Stakeholder request:* "Route the most urgent support tickets to
> the on-call engineer."

Same problem, three framings:

- **Regression on urgency score.** Predict a continuous urgency (0–10);
  triage by threshold. Pro: simple, interpretable. Con: needs an
  urgency *value* in the training labels (do we have that?); a
  squared-error loss optimises absolute urgency, not ordering.
- **Binary classification.** Label = `was actually escalated to
  on-call`. Predict `P(escalated)`. Pro: matches the downstream
  decision; cleanest training signal if escalation is logged. Con: the
  label is itself a function of *past* triage decisions — a survivorship-bias-style
  label-leakage trap.
- **Ranking.** Score each open ticket; on-call works the top-$k$ per
  shift. Pro: directly matches the operational reality. Con: needs a
  pairwise / listwise signal that may not exist in current logs.

There is no "right" answer in the abstract. The right answer is the
framing that (a) matches the decision, (b) you have data for, and (c)
you can evaluate honestly. The point of this chapter is to give you
the vocabulary to argue your way to that answer with a stakeholder
*before* you build the model.

## Summary

- Three supervised output framings: regression (predict a number),
  classification (predict a category, usually via a probability + a
  threshold), ranking (predict an order).
- Each comes with characteristic losses, metrics, and failure modes.
- The right framing is determined by the downstream decision, the
  available signal, the noise model, and the business metric — in
  that order.
- A classifier without a calibrated threshold is incomplete. A
  regression model with heavy-tailed targets is fragile. A
  pointwise-loss model on a ranking problem leaves performance on the
  table.
- When framings conflict, the conflict is telling you something. Don't
  paper over it; raise it.

## End of module

Take the framing rubric forward. Exercise 04 turns it into a worksheet
you'll re-use in `mod-104`, `mod-107`, `mod-110`, and the three
end-of-curriculum projects. You will return to this chapter often.
