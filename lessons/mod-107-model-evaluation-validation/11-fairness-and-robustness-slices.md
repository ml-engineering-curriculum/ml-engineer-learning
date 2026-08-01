# 11 — Fairness and robustness slices

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 03, 10.

## Motivation

An aggregate metric answers "how well does the model
work on average?". Two subpopulations of the metric
answer questions the average conceals:

- **Which subgroups does the model fail?** A model with
  92% aggregate accuracy but 60% accuracy on a specific
  demographic subgroup is not a 92% accurate model —
  it is a discriminatory model that averages to 92%.
- **What happens when the input drifts a little?** A
  model that goes from 95% accuracy on the clean test
  set to 45% accuracy under a small perturbation is
  brittle. Deploying it means the first surprising input
  causes a metric collapse.

Both questions matter *before* the model ships. This
chapter is the level-20 minimum: a small, disciplined
set of slice reports (subgroup performance gaps,
noise / perturbation robustness, distribution-shift
probes) that make worst-case behaviour visible in the
evaluation report. It is not a full fairness audit — those
are their own multi-week engagements — but it is what
distinguishes a competent ML engineer from one who ships
"92% accurate" models that fail loudly.

## The three dimensions

For every model whose deployment matters, report at least
one measurement on each axis:

1. **Subgroup performance.** The metric on slices
   defined by sensitive or operationally important
   attributes.
2. **Perturbation robustness.** The metric under small,
   task-preserving input changes.
3. **Distribution shift.** The metric on data drawn from
   a plausibly different distribution than the training
   data (a later time period, a different geography, a
   different device).

Each is a slice, in the vocabulary of chapter 10 — they
share the mechanics. What is new here is the *choice of
slice* and the *reporting discipline*.

## Subgroup performance — the fairness slice

The task: for a chosen attribute (age, gender, race,
region, socioeconomic proxy — whatever the task's
sensitive attributes are), compute the primary metric on
each subgroup and report the largest gap.

**Pick the attribute:**

- **Directly sensitive attributes** — where the deployment
  has a legal or ethical duty to non-discrimination.
  Gender, race, disability status, age. Check the
  regulatory regime for the deployment jurisdiction.
- **Proxy attributes** — features that correlate with
  sensitive attributes but are not sensitive themselves.
  Postcode as a proxy for race in the UK / US; browser
  agent as a proxy for socioeconomic status.
- **Operational subgroups** — non-sensitive but
  business-important. New vs returning customers, mobile
  vs desktop, low- vs high-value accounts.

**Two definitions of "fair" that matter for reporting:**

- **Demographic parity.** The rate at which the model
  makes a positive prediction is the same across groups:
  $P(\hat{y} = 1 \mid A = a) = P(\hat{y} = 1 \mid A =
  a')$ for all group values $a, a'$. Historically
  demanded in credit decisions, hiring screens.
- **Equalised odds.** The true positive rate and false
  positive rate are the same across groups:
  $P(\hat{y} = 1 \mid Y = y, A = a) = P(\hat{y} = 1 \mid
  Y = y, A = a')$ for $y \in \{0, 1\}$. This is the
  slice-by-recall / slice-by-FPR pair.

Both definitions have their critics. **Barocas, Hardt &
Narayanan (2019)** show they cannot in general be
satisfied simultaneously — a well-known impossibility
result. The choice between them is a value choice, not a
technical one; document which you are reporting and
why.

<!-- needs-research: cite Barocas, Hardt & Narayanan,
"Fairness and Machine Learning" for the incompatibility
result and the vocabulary of fairness definitions. -->

**The minimum viable subgroup report:**

```python
import pandas as pd
from sklearn.metrics import roc_auc_score


def subgroup_metrics(df, group_col, y_true, y_score, min_n=100):
    rows = []
    for group_value, sub in df.groupby(group_col):
        if len(sub) < min_n:
            continue
        rows.append({
            group_col: group_value,
            "n":        len(sub),
            "positive_rate": sub[y_true].mean(),
            "roc_auc":  roc_auc_score(sub[y_true], sub[y_score]),
        })
    df_out = pd.DataFrame(rows).sort_values("roc_auc")
    gap = df_out["roc_auc"].max() - df_out["roc_auc"].min()
    print(df_out)
    print(f"largest {y_true} ROC-AUC gap: {gap:.3f}")
    return df_out
```

Read the output for three things:

- **The largest gap.** A gap greater than 0.05 in ROC-AUC
  (or comparable in F1 / PR-AUC) is worth flagging.
- **The direction of the gap.** Which subgroups are
  underserved? Is the pattern consistent with known bias
  in the training data?
- **The `min_n` exclusion.** Subgroups below the
  threshold are noise; but state which ones you dropped
  and why. A subgroup you cannot measure is a subgroup
  you cannot vouch for.

**Two operational metrics common in fairness reporting:**

- **Statistical parity difference** — the difference in
  positive prediction rate between the most- and least-
  favoured group.
- **Equal opportunity difference** — the difference in
  true positive rate between the most- and least-
  favoured group.

Both are supported by fairness libraries like `fairlearn`
and `aif360`. Level 20 does not require a full library
dependency; report the primary metric per group and the
gaps.

<!-- needs-research: verify fairlearn's canonical metric
names (statistical_parity_difference,
equal_opportunity_difference) match those in the current
Microsoft docs. -->

## Perturbation robustness — the "does it hold under small changes?" slice

Take the test set, apply a task-preserving perturbation,
recompute the metric. A large drop is a robustness
finding.

Task-preserving perturbations depend on the domain:

- **Tabular.** Add small Gaussian noise to numerical
  features; flip a small fraction of categorical values;
  add / drop rare feature values. The perturbations
  should not change the label — if adding noise moves
  most rows across the decision boundary, the model is
  brittle.
- **Text.** Substitute synonyms; introduce typos
  (character insertions / deletions consistent with
  observed typo rates); change casing; add / remove
  punctuation. Libraries: `nlpaug`, `TextAttack`.
- **Images.** Small rotations, brightness / contrast
  shifts, JPEG compression, Gaussian noise, occlusion of
  small patches. Libraries: `albumentations`, `imgaug`.
- **Audio.** Additive noise (speech in noise), pitch
  shifts, small time warps.

The pattern:

```python
def robustness_slice(model, X_test, y_test, perturb_fn, metric_fn):
    X_clean = X_test
    X_perturbed = perturb_fn(X_test)
    m_clean = metric_fn(y_test, model.predict(X_clean))
    m_perturbed = metric_fn(y_test, model.predict(X_perturbed))
    return {
        "clean":      m_clean,
        "perturbed":  m_perturbed,
        "drop":       m_clean - m_perturbed,
    }
```

**What to report:**

- **Metric on clean and perturbed.** Both, not just the
  gap.
- **The perturbation intensity.** "Gaussian noise with
  σ = 0.05 * feature standard deviation" is a claim
  someone else can reproduce.
- **The drop.** A > 5% drop under a plausibly encountered
  perturbation is a flag; investigate before shipping.
- **Per-slice drops.** Combine with the subgroup report:
  which subgroups' metric drops most under perturbation?
  Often the intersection reveals worst-case behaviour
  invisible in either report alone.

## Distribution shift — the "does it hold on new-looking data?" probe

The stricter test: evaluate on data drawn from a
plausibly different distribution than training.

**Sources of shift you can construct in a report:**

- **Time shift.** Hold out the last month / quarter /
  year of data as a shift set — the "future" that the
  model will actually predict on. This is often already
  your test set if you used a time-based split (chapter
  07); if not, construct it.
- **Geographic shift.** If the training data is
  US-heavy, hold out non-US data. Report the metric on
  the shifted subset.
- **Version / device shift.** Held-out iOS-only test set
  when training is Android-heavy; held-out mobile-only
  test when desktop-heavy.
- **Rare-event shift.** For fraud / anomaly detection,
  evaluate on a subset with the tail-event rate elevated
  (up-sample the positives). Reveals how the model
  behaves under a shifted base rate.

**Two metrics that quantify shift itself:**

- **Population Stability Index (PSI).** Compares a
  feature's distribution between the training and shift
  populations. > 0.25 is a "large shift" heuristic in
  credit modelling.
- **KS statistic.** For continuous features, the Kolmogorov-
  Smirnov two-sample statistic quantifies distributional
  difference.

These are diagnostics *for the data*, not the model. A
model can be fine under a large PSI (if the shifted
features are irrelevant) or broken under a small PSI (if
the shifted features are the ones the model depends on).
Chapter `mod-109` (monitoring) uses these as drift
signals in production; here they are context for the
metric report.

<!-- needs-research: cite the original PSI reference
(Kotze / SAS credit modelling literature); the 0.25
threshold is a widely-cited rule of thumb. -->

## When one number is not enough — reporting a slice table

The report artefact that captures all three dimensions:

| Slice | Kind | n | Metric | Δ vs aggregate |
|---|---|---|---|---|
| Aggregate | overall | 20,000 | 0.867 | 0.000 |
| Subgroup — age 18-25 | fairness | 3,120 | 0.812 | -0.055 |
| Subgroup — age 65+ | fairness | 2,400 | 0.891 | +0.024 |
| Perturbation — Gaussian noise σ=0.05 | robustness | 20,000 | 0.831 | -0.036 |
| Perturbation — feature dropout 5% | robustness | 20,000 | 0.804 | -0.063 |
| Shift — held-out last quarter | distribution | 4,500 | 0.789 | -0.078 |
| Shift — non-US | distribution | 1,200 | 0.741 | -0.126 |

Reading it:

- **Aggregate = 0.867** is the headline. Everything below
  is context.
- **The worst slice is "Shift — non-US" at 0.741.** That
  is what the model actually does in the worst case
  that the evaluation could measure. A reviewer with
  responsibility for the deployment cares about that
  number, not the aggregate.
- **The Δ column is signed.** Positive means the model
  is *better* on the slice than on aggregate (rare and
  useful — often points at group-specific overfitting
  that carries over to production well); negative is
  the concerning direction.

## What this chapter is not

The level-20 bar is a report that surfaces the worst
slices. It is not:

- **A full fairness audit.** Which requires domain
  ethicists, legal review, and often specialised
  metrics and modelling techniques (adversarial de-
  biasing, causal fairness, group-aware training).
- **Adversarial robustness.** Which is a distinct
  research area (FGSM, PGD, certified defences) focused
  on worst-case adversarially chosen perturbations. The
  perturbations here are naturally occurring, not
  adversarial.
- **A stress test against distribution shift you cannot
  construct.** If the deployment will see novel input
  types the training data has none of, the report can
  only say "we did not measure this"; the honest
  response is a monitoring plan (`mod-109`).
- **A go/no-go decision.** Reporting a subgroup gap
  triggers a conversation with the stakeholders about
  whether the trade-off is acceptable. It is not a
  technical override.

## Fairness metrics with a caution

Two libraries you will see mentioned:

- **Fairlearn** — Microsoft's Python fairness library.
  Metrics (parity differences, equalised odds
  differences), mitigation algorithms (correlation
  removers, adversarial de-biasers, threshold optimisers).
  MIT-licensed; well-maintained; the default choice for
  fairness work in scikit-learn ecosystems.
- **AIF360** — IBM's counterpart. Broader metric set,
  broader algorithm set, older. Widely cited in academic
  fairness literature.

At level 20, using either as a source of "what metrics
to report" is fine; using their mitigation algorithms
without domain-expert oversight is not. Bias mitigation
is a decision that lands legal and ethical
responsibility on the deploying organisation; it is not
a "and then we ran a de-biasing library over it" step.

## Common pitfalls

1. **Reporting the aggregate only.** The most common
   failure. Any model whose deployment matters at all
   needs at least a subgroup and a perturbation slice.
2. **Dropping small subgroups without stating it.** A
   subgroup with 50 rows is noisy; a subgroup with 50
   rows that is 10% of the population is critical. State
   the exclusion and the size of what was excluded.
3. **Treating fairness metrics as mutually satisfiable.**
   Demographic parity, equalised odds, and predictive
   parity cannot in general all be satisfied at once
   (Barocas et al.). Pick the metric aligned to the
   deployment's ethical / legal duty; do not report
   them all as if they agreed.
4. **Perturbations that change the label.** A
   perturbation that pushes fraud into non-fraud is not
   a "task-preserving" perturbation and the drop is not
   meaningful. Sanity-check by spot-reading a sample.
5. **Distribution shifts contrived to make the model
   look bad.** A held-out slice that is deliberately
   adversarial (e.g., synthetic OOD examples the model
   was never trained on) reports a number that is not
   the deployment reality. Use naturally-arising shifts
   the deployment will actually see.
6. **Confusing the fairness report with the fairness
   audit.** This chapter is the evaluation-report piece.
   A full audit requires domain expertise and often
   additional data collection.
7. **Ignoring the worst slice because the aggregate is
   fine.** The worst slice is often the deployment risk.
   A model that is fine on aggregate and terrible on one
   subgroup is a model that will be pulled from
   production the first time that subgroup complains.

## Summary

- The aggregate metric hides subpopulations. Slice
  reports make the worst case visible.
- Report three dimensions: subgroup performance
  (fairness), perturbation robustness, distribution
  shift.
- Format the report as a table with aggregate at the
  top and slices ordered by their gap. Publish the
  worst slice's metric alongside the aggregate; the gap
  is the risk.
- Fairness definitions (parity vs equalised odds vs
  predictive parity) cannot in general all hold. Pick
  the one aligned to the deployment's duty; document
  the choice.
- This is a level-20 evaluation report, not a full
  fairness audit or a monitoring plan. Escalate when
  the report surfaces a slice gap the deployment
  cannot accept.

## Where this goes next

That closes the module. Chapter 01 promised a validated
model with defensible metrics on defensible splits and a
slice report that says where it fails. If you have
worked through 02–11 you now have the vocabulary and the
mechanics. The exercises put it together into artefacts
you can hand to a teammate.

`mod-108` picks up the validated model and asks how to
serve it. `mod-109` picks up the metric definitions and
asks how to detect when the deployed model starts to
drift. Both start where this module ended.
