# 04 — Feedback loops and data flywheels

> **Reading time:** ~18 minutes.
> **Prerequisite chapters:** 02, 03.

## Motivation

A recommender ships. Users mostly click on what the
model puts on top of the list. Next month's training
data — the labels the model uses to learn what to
recommend — is dominated by items the model itself
already promoted. The next model is *more* confident
about those items. A quarter later, three-quarters of
recommendations come from a shrinking set of items, the
long tail is invisible, and new items never break in.
Offline metrics keep improving. The business metric —
say, session length or add-to-cart rate — slowly
declines.

Nothing broke. No code changed. No feature drifted. The
system did exactly what it was designed to do; the
*dynamics* of doing it created the failure.

This is the second systemic failure mode. Where
training-serving skew is a bug in the code, a feedback
loop is a bug in the loop the code is embedded in. It
is the emergent property of putting a model in a
position to shape the data that trains the next model,
which is the position most production ML systems are
in.

Feedback loops are not always bad — Netflix, Spotify,
and every search engine on the internet run positive
data flywheels that make the product better with more
data. But every flywheel has failure modes, and the
first job of the ML systems designer is to know which
loop you are in and design the correction from day one.

## Vocabulary

Fix the terms before writing about them; they get
conflated in casual conversation.

- **Feedback loop** — the general phenomenon where a
  model's outputs affect the data that trains its
  successor. Neutral term. Can be virtuous or
  degenerate depending on the loop's structure.
- **Data flywheel** — a *virtuous* feedback loop. The
  product uses the model, that generates more data,
  the data makes the model better, the better model
  makes the product better, which drives more use. The
  loop compounds; incumbents get farther ahead.
- **Selection bias** — the training data is not
  representative of the underlying population because
  the model itself decided who / what got observed.
- **Position bias** — a special case of selection bias
  for ranking: items in higher positions get more
  clicks regardless of their intrinsic relevance;
  training on click data attributes the clicks to the
  items instead of to the positions.
- **Filter bubble / echo chamber** — a specific
  degenerate loop in personalisation: the model
  reinforces the user's existing preferences to the
  exclusion of everything else, and the user's
  observed preferences shrink to what the model shows
  them.
- **Runaway loop** — a degenerate loop where a small
  bias amplifies until the system's outputs are
  dominated by it. Common in demand-driven pricing,
  in content moderation, and in credit-risk models
  that get to decide who gets a loan.

Not all four are always at play. In practice, most ML
systems have some form of selection bias; personalisation
systems have position and echo-chamber risks;
scoring/approval systems (credit, hiring, allocation)
have runaway risks. Naming which one you are dealing
with is half the mitigation.

## The dynamics — why the loop is inevitable

For most business-critical models, three facts hold:

1. **The model's output is an action.** A recommendation,
   an approval, a ranking, a price. The action changes
   what the user sees or does.
2. **The label is measured downstream of the action.**
   You cannot observe "would this user have clicked if
   we had shown them the other thing." You only see the
   click on the thing you showed.
3. **The next training set comes from measured
   behaviour.** Which is: behaviour under the current
   model's actions.

Combine the three and every retrain trains on data that
the previous model shaped. This is the reference
architecture from chapter 02 in action: the predictions
table feeds the labels loop, the labels loop feeds the
training pipeline. The loop is the diagram.

You can't get rid of the loop. You *can* design so it
runs in the direction you want.

## Two loops that are virtuous, three that are not

### Virtuous: the recommender flywheel

Netflix, YouTube, Spotify, and every large e-commerce
recommender live here. The loop:

- Model shows items ranked by predicted engagement.
- Users engage more with better-matched items.
- Engagement produces labels (implicit — plays, clicks,
  time-on-item).
- Next model trains on the fresh labels; ranking
  improves; engagement improves.

Why it works:

- The base rate of "trying something new" is high
  enough (either because the user actively seeks
  variety or because the platform mandates
  exploration) that the training set stays diverse.
- The platform *explicitly* injects exploration: some
  fraction of impressions are randomised (see the
  mitigation section below).
- The action set is bounded (finite catalogue), so
  the loop cannot generate unbounded confidence.

### Virtuous: search-quality flywheel

Google, Amazon search, YouTube search. Same shape:
users click on results; ranking learns from clicks;
the top result gets better; users click more
efficiently; ranking gets even better.

Kept virtuous by the same tricks: explicit position-
bias models, exploration, and (importantly) fresh
supply (new documents / products) that force the model
to generalise beyond what it has seen.

### Degenerate: the predictive-policing feedback loop

A predictive-policing model predicts which
neighbourhoods will have high crime. Officers get sent
to high-predicted neighbourhoods, where they observe
and record more incidents (because they are there),
which becomes the training signal for the next model,
which predicts higher crime in the same
neighbourhoods. The loop is not measuring underlying
crime; it is measuring *police presence*. The training
data is not representative of the ground truth.
Documented publicly for years — the Chicago Strategic
Subject List, Los Angeles's PredPol deployment, the
COMPAS risk-scoring investigation by ProPublica in
2016. <!-- needs-research: cite specific ProPublica /
academic references from resources.md when
integrating -->

Why it goes wrong:

- The label ("crime happened here") is measured only
  where the action ("send officers here") was taken.
- No exploration; sending officers to a
  low-predicted neighbourhood is politically and
  operationally hard.
- The bias in the initial training set (historically
  over-policed areas) is amplified rather than
  corrected.

The specific incident matters less than the pattern:
**a scoring system whose action determines whether the
label is even measurable is a runaway loop by
default.**

### Degenerate: credit-scoring exclusion

A credit-risk model predicts default probability.
Applicants above a threshold get loans; the rest are
denied. The training data for the next model contains
outcomes only for the applicants who were approved —
the denied applicants have no outcome to learn from.
The model's estimate of default risk for the "would
have been denied" population is never corrected. Over
successive iterations, the model becomes very
confident about a specific type of good customer and
increasingly blind to the wider space.

Related pattern: the model's confident denials cause
the affected population to seek credit elsewhere,
default there, and be recorded as high-risk in shared
credit-bureau data — feeding back into the training
distribution.

### Degenerate: the filter bubble

Personalised content ranking. The model observes what
the user engages with, ranks more of the same, the
user engages with more of the same, and the model
concludes the user has narrower interests than they
actually do. The user's observed preferences shrink to
the intersection of "what I could plausibly like" and
"what the model shows me." Business metrics (session
length, engagement) can look great; retention over
longer horizons drops as the user runs out of things
they haven't already seen.

## Detecting a loop before it does damage

Loops don't fire alerts. They erode metrics slowly and
invisibly. Design the following four detectors into the
system from day one.

### Detector 1: Diversity metrics

For any ranking or recommendation system, compute:

- **Catalog coverage.** What fraction of the item
  catalogue was recommended at least once, per week?
  Trend line matters more than the absolute number.
- **Entropy of the recommendation distribution.** How
  concentrated is the distribution of what gets shown?
  Falling entropy means the recommender is collapsing
  onto a small set.
- **New-item ramp time.** How long from an item's
  introduction to its first thousand impressions?
  Rising ramp time means the loop is closing on
  incumbents.

Every ranking or recommendation system should have
these three on the same dashboard as CTR and revenue.
If your dashboard has only accuracy and revenue, the
loop can degrade for months before it shows up.

### Detector 2: Untreated control cohort

For any scoring / intervention system (churn outreach,
fraud review, credit approval), permanently hold back
a small fraction — 1-10% depending on cost and ethics
— that is **not** subject to the model's action. This
group's outcomes are the closest thing you get to
ground truth for the population as a whole. The
training pipeline uses this cohort's outcomes as the
gold set for retraining; the observability surface
compares treated vs untreated metrics over time to
detect divergence that would otherwise be invisible.

For interventions with cost or ethical implications
(credit, healthcare, hiring), the untreated cohort has
to be reviewed with the relevant ethics / compliance
partners. The alternative — no untreated cohort —
means you cannot ever measure how the intervention is
performing against the counterfactual.

### Detector 3: Slice performance over time

Chapter 11 of `mod-107` covers slice-based evaluation.
For a feedback-loop diagnostic, monitor the same
metrics on the same slices *over time*, not just at
snapshot points. A slice whose performance drops
consistently over quarters while the aggregate stays
flat is the fingerprint of a loop; the aggregate is
being carried by other slices while a specific
sub-population is degrading.

For personalisation systems, add:

- **Novelty rate per user.** What fraction of items
  shown are new to the user? Declining novelty is a
  filter-bubble smell.
- **Serendipity metric.** Items outside the user's
  historical cluster that the user engaged with. Zero
  serendipity means the model has locked the user in.

### Detector 4: Explicit counterfactual experiments

Occasionally A/B (chapter 05) a *bandit* or a
*random* variant against the model. The bandit tests
whether exploration would improve outcomes; the random
variant is a strong baseline that surfaces how much of
the model's win is real learning vs how much is
capturing already-obvious patterns. Runs should be
short (weeks not quarters) and small (1-5% of
traffic), but they are one of the few ways to
directly detect a runaway loop.

## Mitigating the loop by design

Four patterns to weave into the architecture from
chapter 02.

### Mitigation 1: Exploration budget

The recommender or ranker reserves a fixed fraction of
serving traffic — 1-10% — for **exploration**:
recommending items the model is *not* confident
about. Bandit algorithms (Thompson sampling,
UCB-style approaches) formalise this; simple
uniform-random-sample-from-top-K works surprisingly
well as a starting point.

The exploration traffic is:

- Labelled the same as production traffic (feeds the
  training pipeline).
- Instrumented separately so its performance can be
  measured and reported (usually worse than the
  model's, on purpose).
- Explicitly justified in the blueprint as a *cost*
  the business is paying for training-set diversity.

Without an exploration budget, the loop's data set
narrows to what the model already believes.

### Mitigation 2: Position and selection debiasing

For ranking systems, model the position bias
explicitly. The classic trick is inverse-propensity
weighting: an impression at position `k` gets
downweighted in the training set by its expected
click-through-rate at that position, so higher-ranked
items don't get credit for their position. Netflix,
YouTube, and the search engines all publish papers on
their specific variants — the specific method matters
less than the fact that *some* method is applied.
<!-- needs-research: cite Chen et al. 2019 (YouTube),
Joachims et al. IPS papers when integrating references -->

For scoring / approval systems, the analogue is
selection-adjusted training: the model is trained on
the whole population's outcomes as best you can
observe them, with the untreated cohort (detector 2)
being the crucial input.

### Mitigation 3: Regular full-population retraining

The default retraining schedule uses the labels loop's
output — which is model-shaped. Once every
`N` training cycles, retrain on a *broader* set: the
untreated cohort, historical randomised data,
externally sourced labels, or all of the above. The
model produced this way may be slightly worse on
aggregate metrics; it is the periodic reset that keeps
the loop from tightening indefinitely.

Practically, this is one training-config flag, one
extra pipeline run, and one extra evaluation gate. It
should be as boring as a nightly cron. Skip it and the
loop wins.

### Mitigation 4: Human-in-the-loop calibration for
   high-stakes decisions

For any decision with real-world stakes (credit,
hiring, health, content moderation), a fraction of
model decisions get reviewed by humans, at random or
targeted at low-confidence and high-impact cases. The
review outcomes:

- Become gold labels for retraining.
- Surface systematic errors the offline metrics miss.
- Provide accountability that a fully-automated loop
  cannot.

This is expensive; it is also what separates
regulated-industry ML from consumer-recommender ML.
Chapter 06 covers when this is a hire-a-team decision
vs a delegate-to-existing-review-workflow decision.

## Two design choices that determine your loop's
   direction

Feedback loops are baked in on day one by two
architectural decisions. Get them right up front and
the mitigations above are cheap; get them wrong and
you spend the next year retrofitting.

### Choice 1: How labels are collected

- **Passive-observational labels** (users click on
  what they see): loop is on. Requires exploration
  and debiasing from day one.
- **Randomised-treatment labels** (a fraction of
  traffic is served random or bandit-selected
  outputs): loop is calibrated by construction.
- **Externally-sourced labels** (human labeler queue,
  ground-truth from a downstream system that does
  not depend on the model): loop is broken by
  construction. Expensive but reliable.

For most production ML, the answer is a *combination*:
most labels are passive-observational; a small
fraction is randomised-treatment for calibration; a
smaller fraction is externally sourced for
high-stakes decisions.

### Choice 2: Where the model sits in the decision
   chain

- **The model is the decision** (auto-approve, auto-
  reject): tightest loop. Every mitigation above is
  mandatory.
- **The model is a filter** (top-K into a human
  review): looser loop. The human review provides
  some correction, but only for the top-K; the
  discarded tail is unmeasured.
- **The model is a suggestion** (nudge, rank, hint):
  loosest loop. The user's choice provides an
  additional signal, though position bias still
  applies.

The tighter the model is in the loop, the more
mitigation infrastructure is required. Auto-decision
systems in regulated industries typically require all
four mitigations plus regular external audits.

## The one-page loop analysis

For any ML system in design, produce a short
"loop analysis" — a paragraph in the blueprint
answering:

- What action does the model take, and does the action
  determine whether the label is measurable? (If yes,
  runaway risk.)
- Where do the training labels come from — passive
  observation, random assignment, or an external
  ground truth?
- What fraction of serving traffic is exploration?
- What untreated / control cohort exists to measure
  the counterfactual?
- What diversity metrics are on the dashboard?
- What is the periodic full-population retrain
  cadence?

Six questions; six lines. In a design review, this
paragraph is the first thing an experienced ML
reviewer will look for.

## Common pitfalls

1. **Denying the loop exists.** "We just train on the
   data we have." That's the loop.
2. **Optimising the metric the loop inflates.** CTR
   on a recommendation that is over-served is not
   evidence the recommendation is good; it is
   evidence the loop is tightening.
3. **No exploration budget.** The first year of the
   system's data is what determines the next year's
   model; a homogeneous first year permanently
   narrows the model's view.
4. **No untreated cohort.** Every intervention looks
   like a success because there is no counterfactual.
5. **Aggregate metrics only.** Slice degradation is
   invisible in aggregate; the loop hides in slices.
6. **A "we'll add debiasing later" plan.** By then,
   the training set is contaminated; the loop is set.
7. **Human-in-the-loop as the sole safety
   mechanism.** Humans get overloaded; review rates
   drop; the loop wins by default.
8. **A recommender whose only diversity metric is
   accuracy@K.** Neither catalogue coverage nor
   entropy is on the dashboard; the collapse is
   invisible until the business metric drops.
9. **Auto-decisioning without external audit.** The
   regulator finds out first.
10. **Confusing drift with the loop.** Chapter 03 of
    `mod-109` is about the world changing; this
    chapter is about the *model* changing the world.

## Summary

- Every production ML system has a feedback loop from
  its own outputs into its own training data. The loop
  is inherent, not a bug.
- Loops can be virtuous (recommender / search
  flywheels) or degenerate (predictive-policing-style
  runaways, filter bubbles, exclusion in scoring
  systems).
- Detect loops with diversity metrics, untreated
  cohorts, slice-over-time monitoring, and
  counterfactual A/B tests.
- Mitigate with an exploration budget, position /
  selection debiasing, periodic full-population
  retraining, and human review for high-stakes
  decisions.
- Two design choices determine the loop's direction:
  how labels are collected, and where the model sits
  in the decision chain. Choose consciously.
- Every blueprint has a one-paragraph loop analysis.
  If it doesn't, the design review adds it.

## Where this goes next

Chapter 05 covers the discipline that actually proves
whether a model shipped moved the business number:
the A/B test. A/B tests are the tool that separates
"the model is better" from "we retrained on
model-shaped data and the offline metric moved" —
which is exactly the failure mode this chapter's
loops produce. The tools compose: monitoring detects
the loop's drift; the A/B test proves the fix.
