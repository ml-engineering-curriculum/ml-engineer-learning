# exercise-04 — Problem-framing rubric

> **Estimated effort:** 3 hours.
> **Companion chapters:** [07 — Framing a problem](../07-framing-an-ml-problem.md), [08 — Regression vs. classification vs. ranking](../08-regression-classification-ranking.md).

## Objective

Build a re-usable **problem-framing rubric** that you (and the rest of
your future ML team) can apply to any new "we should use ML for X"
request. Then apply it to three realistic scenarios end-to-end. This
exercise has no code — it is the writing-and-thinking exercise that
chapters 07 and 08 set up. You will use the rubric again in `mod-104`,
`mod-107`, `mod-110`, and the three end-of-curriculum projects.

## Prerequisites

- Chapters 07 and 08 of this module.
- A working brain. Pen and paper, or a markdown editor.

## Problem statement

### Part A — author the rubric (≈ 60 min)

Author `RUBRIC.md` containing a one-page (≤ 800 words) rubric you would
hand a junior teammate to use on day one. The rubric must cover, at
minimum, these eight questions:

1. **What is the business decision this model will drive?** (Stated
   in one sentence, *without* the words "machine learning" or "AI".)
2. **What does success look like in measurable terms?** What metric
   would a sceptic accept as proof the project worked?
3. **What is $x$? What is $y$?** Be concrete about the unit of
   prediction (per user? per session? per ticket?).
4. **Where do the labels come from, how clean are they, and how
   delayed are they?**
5. **Which ML framing does this fit — supervised, unsupervised, or
   reinforcement learning?** Justify in one or two sentences.
6. **If supervised: regression, classification, or ranking?** Justify
   in one or two sentences.
7. **What baseline must we beat?** Random? The current rule-based
   system? Human performance? A previously-deployed model?
8. **What are the most likely ways this project fails?** Pick the
   top three (e.g. distribution drift, label leakage, base-rate
   confusion, scarce labels, ambiguous metric, etc.) and write one
   sentence on what you'd do to mitigate each.

The rubric should be **prescriptive enough that a junior could use
it without help** and **terse enough that it actually fits on a
single page**. Add a short paragraph at the top on *when not to
apply it* (some asks aren't ML problems at all — call that out).

### Part B — apply the rubric to three scenarios (≈ 90 min, 30 min each)

For each scenario, write a `framing_<n>.md` (one to two pages) that
walks through your rubric and lands on a defensible framing
recommendation. You are *not* solving the problem; you are framing
it well enough that the rest of the curriculum could solve it.

#### Scenario 1 — Support-ticket triage

> *Stakeholder:* "Our on-call engineer is drowning in tickets. Can
> ML route the most urgent ones to them and defer the rest to the
> regular queue?"

Available signals: ticket text, account tier, account size, the
on-call's historical "escalated" / "not escalated" actions over the
past two years, average response time per priority level.

Cover, at minimum:

- The supervised/unsupervised/RL choice and why.
- Regression vs. classification vs. ranking — present at least two of
  these as viable framings with their pros/cons before recommending
  one.
- Two label-quality risks specific to this dataset, and the
  mitigation for each.
- What metric a sceptic would accept and why.

#### Scenario 2 — Forecasting weekly ad spend

> *Stakeholder:* "Marketing wants a weekly forecast of how much
> we'll spend on paid ads, broken down by channel and country, with
> enough lead time to set the budget."

Available signals: 4 years of weekly spend by channel × country,
known holidays, known marketing campaigns (calendar), planned
product launches.

Cover, at minimum:

- Why this is fundamentally a *forecasting* problem and what makes
  i.i.d. assumptions awkward here.
- The right output framing — regression, multiple regression,
  quantile regression, distributional forecast — and which the
  business actually needs.
- Why you might *not* use a fancy deep model here.
- Two pitfalls (data leakage from future to past, look-ahead bias,
  level shifts from external events, etc.) and one sentence on each.

#### Scenario 3 — Image-based defect detection on a factory line

> *Stakeholder:* "Replace the human inspector on conveyor belt 4
> with a camera + model that flags defective parts."

Available signals: roughly 50,000 historical photos labelled "OK"
or "defect" by the previous human inspector (with notes that
inspectors sometimes disagreed); newly streaming photos from the
camera, ~10 per second; safety constraint: missing a defect is
much worse than a false positive.

Cover, at minimum:

- The supervised/unsupervised/RL choice.
- The framing as binary classification — *or* as something else
  (e.g. anomaly detection, ranking, calibrated scoring + a tuned
  threshold), and your recommendation.
- The metric the safety constraint implies (think hard about
  precision vs. recall).
- Where this might be the *wrong* problem to apply ML to at all,
  and what that would look like.

### Part C — write a peer review of one (≈ 30 min)

Swap your three framings with a peer (or, if working solo, swap one
of yours with the solutions repo's reference framing for the same
scenario after you've finished). Write a `review.md`:

- Pick one of the peer's framings.
- Find at least one thing that is **stronger than yours** and
  briefly explain why.
- Find at least one thing that is **weaker or missing** and propose
  a specific change.
- Be specific. "I'd add a baseline of <X>" beats "could mention
  baselines".

If working solo, do this against your *own* framings 48 hours after
writing them — distance helps you see your own assumptions.

## Starter guidance

- The rubric is a tool you will actually use. Write it for the
  junior teammate, not for the grader.
- Resist the urge to "decide" on day one. Each framing doc should
  *land* a recommendation, but the value is in the analysis that
  produces it.
- A common trap is to slide into modelling decisions ("we should
  use XGBoost"). Stop and re-read chapter 07. You are framing, not
  modelling.
- If you find yourself writing "we'd need to talk to the
  stakeholder", flag the open question explicitly. That is part of
  the deliverable.

## Acceptance criteria

- `RUBRIC.md` exists, is ≤ 800 words, covers the eight required
  questions, and includes the "when not to apply this" callout.
- Three `framing_<n>.md` docs exist, each applying the rubric and
  recommending a defensible framing.
- Each framing doc explicitly considers **at least two** alternative
  framings before settling on a recommendation.
- `review.md` includes at least one strength and one specific
  improvement.
- No code, models, or actual predictions are produced. (If you find
  yourself writing code, you are misreading the exercise.)

## Stretch goals (optional)

- Add a **fourth scenario of your own choosing**, drawn from a
  domain you know well. The harder it is to frame, the better the
  exercise.
- Translate your rubric into a **one-page worksheet** (PDF or
  markdown table) you can actually hand a stakeholder during a
  scoping meeting.
- For Scenario 1, sketch a brief **A/B test plan** that would
  validate your chosen framing in production (1 paragraph).
- Read the [Google "Rules of Machine Learning"](https://developers.google.com/machine-learning/guides/rules-of-ml)
  document and tag each section of your rubric with the rule(s) it
  most closely encodes.

## What to hand in

```
exercise-04/
├── RUBRIC.md
├── framing_1_support_triage.md
├── framing_2_ad_spend_forecast.md
├── framing_3_defect_detection.md
└── review.md
```

Solutions for this exercise live in the paired `ml-engineer-solutions`
repo. Resist the urge to peek — framing is a skill that only
develops if you struggle through it the first time yourself.
