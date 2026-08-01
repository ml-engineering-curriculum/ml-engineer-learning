# 05 — Metrics for ranking

> **Reading time:** ~25 minutes.
> **Prerequisite chapter:** 03.

## Motivation

Some tasks are not about "is this positive?" or "what is
the numeric value?" — they are about *ordering*. Recommend
the top ten items to a user. Return the ten most relevant
documents for a query. Surface the five most likely fraud
candidates for a reviewer. In each case only the top of the
list is seen, only the relative order matters, and the
classifier / regression metrics from chapters 02–04 answer
the wrong question.

Ranking metrics are what you report when:

- **Only the top-$k$ predictions matter.** A user sees
  five results, not five thousand.
- **The order is user-visible.** Position #1 is more
  valuable than position #5, which is more valuable than
  position #50.
- **Multiple items can be "correct".** Search results,
  recommendations, retrieval — usually there is a set of
  relevant items rather than a single label.

This chapter walks the four families of ranking metrics
you will encounter at level 20: **hit@k / recall@k /
precision@k** (the count-based family), **MAP** (Mean
Average Precision), **MRR** (Mean Reciprocal Rank), and
**nDCG** (Normalised Discounted Cumulative Gain). Each
answers a slightly different question; picking well is a
matter of matching the metric to how the ranking is used.

## The setup: queries, items, and relevance

Every ranking evaluation has three primitives:

- **Query** (or "context", "user", "session"). One unit
  over which you produce a ranking. In search it is a
  query; in recommendations it is a user; in retrieval it
  is an anchor.
- **Items** — the candidate objects being ranked. For
  each query the system produces a ranked list.
- **Relevance labels** — for each (query, item) pair,
  either **binary** (relevant / not relevant) or **graded**
  (score from 0 to 4 for "irrelevant" through "perfectly
  relevant").

The metrics divide cleanly on that binary-vs-graded axis:

| Metric | Relevance | Cares about position? |
|---|---|---|
| Precision@k / Recall@k / Hit@k | binary | position of the cutoff, not within top-k |
| MAP | binary | yes, average across positions |
| MRR | binary, first hit | yes, harshly at the top |
| nDCG | graded (or binary) | yes, log-discounted |

## Precision@k, Recall@k, and Hit@k

The simplest ranking metrics. Take the top $k$ items of
the ranked list; count how many are relevant.

$\text{Precision@}k = \frac{|\{ \text{relevant items in top-}k \}|}{k}$

$\text{Recall@}k = \frac{|\{ \text{relevant items in top-}k \}|}{|\{ \text{all relevant items for this query} \}|}$

$\text{Hit@}k = \mathbb{1}[\text{at least one relevant item in top-}k]$

Aggregate across queries with the arithmetic mean.

**When to reach for them:**

- The product's UI has a hard $k$ (five recommendations,
  ten search results). Precision@5 is what a user
  literally experiences.
- The judgement is binary — item is either relevant or
  not.
- The number is user-communicable. "Our top-10 recall
  went from 42% to 51%" is a sentence a PM understands.

**Two failure modes:**

- **Ignores order within top-$k$.** A ranking that puts
  the one relevant item at position 1 scores the same as
  one that puts it at position $k$. If order-within-top-$k$
  matters, use MAP, MRR, or nDCG.
- **Recall@k is capped by $k$.** A query with 100 relevant
  items and $k = 10$ can never exceed recall@10 = 0.1.
  Recall@k across queries with different relevant-set
  sizes is not always comparable.

## Mean Reciprocal Rank (MRR)

For each query, find the rank $r_q$ of the **first**
relevant item. If none is retrieved, contribute 0.

$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{r_q}$

Reference points:

- **First hit at position 1** → contribution 1.
- **First hit at position 2** → contribution 0.5.
- **First hit at position 10** → contribution 0.1.

The reciprocal falls off very quickly with rank; MRR is
extremely position-sensitive at the top. That is exactly
its point.

**When to reach for it:**

- The user is trying to find **one** answer and stops
  reading as soon as they find it. Autocompletion,
  spell-correction suggestions, "did you mean" boxes.
- Multiple relevant items exist but only the first hit
  matters for the task.

**When to skip it:**

- Multiple relevant items per query all matter — use MAP
  or nDCG instead.
- The task is retrieval into a downstream reranker that
  cares about the top-$k$ set, not just the first hit.

## Mean Average Precision (MAP)

Average Precision for one query is the mean of
`Precision@k` computed at every $k$ where a relevant item
appears:

$\text{AP}(q) = \frac{1}{R_q} \sum_{k=1}^{n} \text{Precision@}k \cdot \text{rel}(k)$

where $R_q$ is the number of relevant items for query $q$
and $\text{rel}(k)$ is $1$ if the item at rank $k$ is
relevant. **MAP** is the arithmetic mean of AP across
queries.

Reading it:

- AP rewards putting relevant items **high** in the
  ranking — every relevant item's contribution is scaled
  by the precision at *its* position.
- Range $[0, 1]$; 1 means every relevant item is at the
  top of the list ahead of every non-relevant item.

**When to reach for it:**

- The task is retrieval with binary relevance and multiple
  relevant items per query — the canonical case in IR
  evaluation (TREC benchmarks use MAP historically).
- You need a single number that captures the whole
  ranking, not just the top-$k$ cutoff.

**When to skip it:**

- Graded relevance — MAP treats "perfectly relevant" and
  "kind of relevant" the same. Use nDCG.
- Only the first hit matters — MRR is the sharper tool.

## Normalised Discounted Cumulative Gain (nDCG)

The workhorse of modern ranking evaluation, and the one
metric to know without needing a lookup.

Two ingredients:

**Discounted Cumulative Gain at $k$** — sum of relevance
scores, discounted by log-rank:

$\text{DCG@}k = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i + 1)}$

(Two conventions exist for the numerator: `rel_i` directly,
or `2^{rel_i} - 1`. The latter emphasises the top
relevance grades and is the default in most modern
implementations, including `sklearn.metrics.ndcg_score`.)

The log-discount is what makes nDCG feel right: position
1 contributes with weight 1, position 5 with weight
$\log_2 6 \approx 2.58$ (so the item there is worth 39%
of a position-1 item), position 10 with weight $\log_2 11
\approx 3.46$.

**Normalisation** — divide by the ideal DCG (the DCG you
would get by sorting the items by their true relevance):

$\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$

Range $[0, 1]$; 1 means the ranking is perfect.

**When to reach for it:**

- Graded relevance (search, ad ranking, recommendations
  with implicit-feedback strength signals).
- Position matters, and you want the top-of-list to
  dominate the score without hard-cutting at $k$.
- Comparing across queries with different relevance
  distributions — the normalisation makes averages
  meaningful.

**When to skip it:**

- Binary relevance with a small ranked list and only the
  first hit matters — MRR is sharper and simpler.
- Very short lists where the log discount barely
  differentiates positions.

Scikit-learn ships `ndcg_score(y_true, y_score, k=None)`;
its default numerator convention and how it handles
missing / negative relevance is worth checking.

<!-- needs-research: verify sklearn.metrics.ndcg_score
numerator convention (`gain_function="exponential"` /
default) and behaviour with negative relevance grades. -->

## A worked example

Score a small hand-built dataset and watch the four
metrics disagree.

```python
import numpy as np
from sklearn.metrics import ndcg_score

# Two queries; four candidates each; binary relevance.
# Perfect ordering would be [1, 1, 0, 0]  and  [1, 0, 0, 0].
y_true = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
])

# Two systems' scores for the same candidates:
scores_A = np.array([
    [0.9, 0.4, 0.8, 0.1],   # puts one relevant at rank 1, other at rank 3
    [0.1, 0.9, 0.8, 0.7],   # puts the relevant item at rank 4
])
scores_B = np.array([
    [0.9, 0.8, 0.1, 0.4],   # puts both relevant at ranks 1 and 2 - perfect
    [0.9, 0.8, 0.7, 0.1],   # puts the relevant item at rank 1 - perfect
])


def rank_metrics(y_true, scores, k=4):
    order = np.argsort(-scores)
    ranked_rel = y_true[np.arange(len(y_true))[:, None], order]
    # Precision@k
    p_at_k = ranked_rel[:, :k].sum(axis=1) / k
    # Recall@k
    r_at_k = ranked_rel[:, :k].sum(axis=1) / y_true.sum(axis=1)
    # Hit@k
    h_at_k = (ranked_rel[:, :k].sum(axis=1) > 0).astype(float)
    # MRR
    first_hits = np.argmax(ranked_rel, axis=1) + 1
    has_hit = ranked_rel.max(axis=1) > 0
    rr = np.where(has_hit, 1.0 / first_hits, 0.0)
    return {
        "precision@k": p_at_k.mean(),
        "recall@k":    r_at_k.mean(),
        "hit@k":       h_at_k.mean(),
        "mrr":         rr.mean(),
        "ndcg":        ndcg_score(y_true, scores),
    }


for name, s in [("A", scores_A), ("B", scores_B)]:
    m = rank_metrics(y_true, s, k=2)
    print(f"{name}: {m}")
```

System B ranks perfectly on both queries. System A
ranks one query well and one poorly. All five metrics
prefer B — but by very different margins. MRR punishes A
hardest (rank-4 first hit on query 2 → 0.25 contribution);
Precision@2 punishes A less; nDCG lies in between. The
metric you pick determines *how strongly* B is preferred,
which affects hyperparameter selection and model choice.

## When to prefer classification metrics over ranking metrics

Ranking metrics are the right choice when the **output is
a list**. If the model is producing a single score that
becomes a single decision, classification / regression
metrics are correct even when the underlying task feels
like ranking. Two examples:

- **A binary fraud classifier** whose output triggers a
  hold action. The decision is per-row, not per-list;
  ROC-AUC and PR-AUC (chapter 03) are the right metrics.
- **A regression model whose predictions are sorted for
  business reasons.** If the downstream use is to
  threshold on the predicted value, RMSE / MAE (chapter
  02) are the right metrics; ranking is a consumer, not
  the objective.

The test: **is the model producing a list per query, or a
score per row?** Ranking metrics are for the first.

## Multi-list aggregation, weighting, and stratification

Two subtleties that matter once you are computing rank
metrics on real data:

**Weighting queries.** Averaging AP across queries with
very different sizes and traffic volumes gives every
query equal weight. That is rarely what you want in
production. Two alternatives:

- **Weight by traffic volume** — an AP for a query issued
  ten million times should count more than a query issued
  once. This is what most industrial ranking systems
  actually optimise.
- **Stratify and report per-segment.** Report nDCG@10 for
  head queries, torso queries, tail queries separately.
  A model that improves head at the expense of tail is
  a different trade-off than a uniform improvement.

**Missing labels.** In most industrial retrieval evaluations
you do not have relevance judgements for every (query,
item) pair — you have judgements for the items your
system (or a competing system) surfaced. Unlabelled items
are typically treated as non-relevant, which underestimates
recall for models that surface novel-but-relevant items.
Address with pooled evaluation (union of candidates from
multiple systems is judged) or off-policy evaluation
techniques beyond level 20.

<!-- needs-research: cite TREC pooling methodology (Harman,
1995) as the standard way of handling unlabelled items in
retrieval evaluation. -->

## Common pitfalls

1. **Computing precision@k without stating k.** "Our
   precision improved" is meaningless without the cutoff.
   Every rank metric names $k$: `precision@10`,
   `nDCG@100`.
2. **Using MAP with graded relevance.** MAP binarises;
   the graded signal is lost. Use nDCG.
3. **Comparing MRR across query sets with different first-
   hit distributions.** MRR is very sensitive to whether
   the first hit exists at all — a query set with more
   "no relevant items" queries drags MRR down without any
   ranking-quality change.
4. **Averaging arithmetically instead of weighting by
   traffic.** Small tail queries dominate; the actual
   production loss looks worse than the metric.
5. **Treating a classification threshold like a ranking
   cutoff.** Ranking metrics do not use a threshold; they
   use position. Reporting "F1 at threshold 0.5" for a
   ranking system misses the point.
6. **Evaluating on the training queries.** Same rule as
   classification: hold out queries (not just items) for
   evaluation. If a query appeared in training, its
   ranking is not unseen — this is a group leakage
   problem, chapter 08's subject.

## Summary

- Ranking metrics are for tasks where the output is a
  ranked *list* per query, not a single score per row.
- Precision@k / Recall@k / Hit@k are the count-based
  family: use them when the UI has a hard $k$ and
  relevance is binary.
- MAP is the position-aware summary for binary relevance;
  MRR is the first-hit metric; nDCG is the graded-
  relevance workhorse with a log discount.
- Match the metric to the decision the ranking supports.
  Report a companion — MAP with a diagnostic Precision@k,
  nDCG with a stratified per-segment breakdown.
- Weight queries by traffic when the deployment does; do
  not evaluate on queries that appeared in training.

## Where this goes next

Chapter 06 opens the validation half of the module. The
first question is not "which metric?" but "on what data?"
— specifically, how to design train / validation / test
splits that do not leak future information, correlated
groups, or preprocessing artefacts into the score.
