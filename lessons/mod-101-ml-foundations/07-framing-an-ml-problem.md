# 07 — Framing a problem: supervised, unsupervised, reinforcement

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 04, 05, 06.

## Motivation

A stakeholder walks up to you and says: *"We want to use machine
learning to reduce customer churn."* Before you load `pandas`, you have
to answer:

- What is the **input** ($x$) and what is the **target** ($y$)?
- Do we have **labels** for $y$, and if so, how clean are they?
- Is the system **one-shot** (predict and ship) or **sequential** (act,
  observe, act again)?
- What does **success** look like in measurable terms?

This is **problem framing**, and it is the single highest-leverage
skill in this module. A well-framed problem makes the modelling choice
nearly automatic. A badly-framed problem makes every modelling choice
feel arbitrary.

This chapter walks through the three high-level framings — supervised,
unsupervised, reinforcement — and how to tell which one fits.

## Supervised learning

You have labelled pairs $(x, y)$ and want to learn a function $f$ such
that $f(x) \approx y$ on new examples. This is the dominant flavour of
ML in production today.

Two sub-flavours, chosen by the *type* of $y$:

- **Regression** — $y$ is continuous (price, latency, count).
- **Classification** — $y$ is categorical (spam/ham, fraud/not, one of
  $K$ topics).

A third closely-related variant:

- **Ranking** — you don't need an absolute prediction; you need the
  *order* of items to be right. Search results, recommendations, ad
  placement.

Chapter 08 dives into the regression-vs-classification-vs-ranking choice
specifically.

### When supervised learning fits

Supervised learning works when you can answer "yes" to three questions:

1. **Are there enough labelled examples?** "Enough" depends on the
   problem complexity: maybe hundreds for a clean tabular regression,
   tens of thousands for image classification, millions for an
   end-to-end deep model.
2. **Are the labels reliable?** Noisy labels cap how well any model can
   do. A model can never be more accurate than the inter-annotator
   agreement; if humans disagree 30% of the time, your test-set
   "accuracy" peaks somewhere below 70%.
3. **Will the deployment-time distribution look like the training-time
   distribution?** Supervised learning generalises in the i.i.d. sense
   (chapter 04). The more your serving data drifts from your training
   data, the less of your training-time performance you keep.

### Where labels come from

- **Naturally occurring.** The target *is* the business outcome and you
  just observe it: did the user click? did the loan default? did the
  shipment arrive on time? These labels are cheap but biased — you only
  observe outcomes for the items you actually showed / lent to /
  shipped.
- **Annotated.** Humans label the data. Quality depends entirely on the
  instructions, the annotators, and the disagreement-resolution
  protocol. Always run an inter-annotator agreement check.
- **Programmatic / weak.** Heuristics or distant supervision generate
  noisy labels at scale. Useful when no clean source is available;
  introduces its own systematic errors.
- **Self-supervised.** The model fabricates a label from the structure
  of the data itself (predict the next token, predict a masked image
  patch). Almost all modern foundation models are pre-trained this way
  and *then* fine-tuned with supervised data. Strictly speaking this
  blurs the supervised/unsupervised line.

## Unsupervised learning

You have inputs $x$ but no targets. The goal is to discover structure in
the data itself. Common tasks:

- **Clustering.** Group similar examples together (k-means, DBSCAN,
  hierarchical).
- **Dimensionality reduction.** Project high-dimensional data into a
  lower-dimensional space that preserves something useful (PCA, UMAP,
  autoencoders).
- **Density estimation.** Model $p(x)$ directly — the distribution that
  generated the data.
- **Anomaly detection.** Find examples that look unlike the typical
  ones. Density estimation gives you this for free; isolation forests
  and one-class SVMs are specialised tools.

### When unsupervised learning fits

- You **don't have labels** and can't afford to get them, but you
  believe there is structure in the data worth surfacing.
- You **have labels** but want a structural understanding *before*
  modelling — clustering for EDA, PCA for visualisation, anomaly
  detection for data quality.
- You want a **representation** that downstream supervised tasks can
  exploit (pre-train a representation unsupervised, then fine-tune with
  labels).

### A common pitfall

Unsupervised "results" are easy to over-interpret. If you ask k-means
for 5 clusters, you will get 5 clusters — whether or not there are 5
natural groupings. There is no held-out test set that says "your
clusters are wrong." This is why unsupervised learning is often paired
with a downstream task or with explicit evaluation against domain
knowledge.

## Reinforcement learning (RL)

An **agent** takes **actions** in an **environment**, receives **rewards**
and **observations**, and tries to learn a **policy** that maximises
cumulative reward over time. The training signal is not a label — it is
a reward, often sparse and delayed.

The canonical loop:

1. The agent observes state $s_t$.
2. It picks an action $a_t$ according to its policy.
3. The environment returns a new state $s_{t+1}$ and a reward $r_{t+1}$.
4. The agent updates its policy from the trajectory.

### When RL fits

RL is the right framing when **the data your model sees in deployment
depends on the actions the model takes**. Some honest examples:

- **Robotics / control** — a robot arm picks an action, the world
  responds.
- **Game playing / planning** — chess, Go, poker, AlphaGo and friends.
- **Recommendation when the action distribution shifts** — a
  recommender that drives user behaviour, where the i.i.d. assumption
  fundamentally breaks.

### When RL is the *wrong* framing

Most of the time. RL is famously sample-inefficient (millions of
interactions for non-trivial tasks), brittle to reward specification,
and hard to evaluate offline. A common, painful failure mode is
"framed as RL, would have been fine as supervised". If you can collect
a labelled dataset that captures the right behaviour, supervised
learning will almost always be faster and more reliable.

A useful heuristic: **frame as supervised unless the action your model
takes meaningfully changes the future inputs it sees**. If actions do
shift the distribution, supervised + careful counterfactual evaluation
is still often a better first step than full RL.

This curriculum does not teach deep RL — it is its own specialism, and
peer tracks own it. Your job at level 20 is to know what RL is, when it
genuinely applies, and how to say "this isn't an RL problem" when it
isn't.

## A decision tree for framing

```
1. Do you have labelled targets (y) for the prediction you want to make?
   ├── YES → Supervised learning.
   │         ├── y continuous → regression (chapter 08).
   │         ├── y categorical → classification (chapter 08).
   │         └── you only need the relative order → ranking (chapter 08).
   │
   └── NO → You have only inputs.
              ├── Do you need to act, observe a consequence, and act again?
              │   ├── YES → reinforcement learning.
              │   └── NO  → unsupervised learning.
              │            ├── group similar things → clustering.
              │            ├── compress for visualisation → dim. reduction.
              │            ├── flag unusual things  → anomaly detection.
              │            └── learn a representation for a future
              │              supervised task → self-supervised pre-training.
              └── (consider: can you cheaply *get* labels first?)
```

The final "can you cheaply get labels" question is the most underrated
one. Often the right move is **don't model yet, instrument the system
to collect the labels you'll need.** Module 109 builds out this
ground-truth-collection loop in detail.

## Framing a real problem: a worked example

> *Stakeholder request:* "We want to use ML to reduce customer churn."

A first pass at framing:

- **What is $x$?** Account attributes (tenure, plan, usage, support
  tickets, last login).
- **What is $y$?** "Will this customer cancel in the next 30 days?" —
  binary. Or "what is the predicted lifetime value of this customer?" —
  continuous. The choice changes everything downstream.
- **Are there labels?** Yes — historical cancellation events. But:
  - the label is **delayed** (you don't know cancellation for the last
    30 days yet), which constrains training set construction;
  - the label is **biased** (you only saw outcomes for customers who
    were not already intervened on);
  - the **base rate is low** (maybe 2% of customers churn per month) —
    a class imbalance problem.
- **What flavour?** Supervised classification (or supervised regression
  on probability of churn). RL is tempting because you would actually
  *act* on the predictions (offer a discount, call them), but you can
  decouple the prediction step (supervised) from the action step (a
  separate policy / uplift model). Start with supervised.
- **What is success?** This is the **metric design** question, which
  module 107 owns. At minimum, what proportion of the churners do we
  catch (recall), at what cost (precision and intervention dollars)?

After this 30-minute exercise you have a defensible framing. You can now
plan data collection, feature engineering, modelling, evaluation, and
hand-off — the work of the next eight modules.

## Summary

- The three high-level framings are supervised, unsupervised, and
  reinforcement learning.
- Supervised needs labels, label quality, and i.i.d. assumptions.
  Almost everything you ship will be supervised.
- Unsupervised surfaces structure; it is most useful for representation
  learning, EDA, and anomaly detection, and is easy to over-interpret.
- RL is the right framing when actions shift the input distribution.
  It is hard, sample-hungry, and almost always a worse first choice
  than well-framed supervised learning.
- The framing decision determines almost every downstream choice.
  Spend time on it.

## Where this goes next

Chapter 08 dives into the supervised sub-framings — regression vs.
classification vs. ranking — and gives you the language to argue about
which one fits.
