# exercise-03 — A/B test design for an ML rollout

> **Estimated effort:** 4 hours.
> **Companion chapter:** [05 — A/B testing ML systems](../05-ab-testing-ml-systems.md).

## Objective

Design a defensible A/B test for a specific ML rollout
scenario and produce the pre-registered artifacts a
mature experimentation practice requires:

- A **one-page design doc** (chapter 05 template) that
  fixes the unit, metric, guardrails, power analysis,
  ramp, and decision rule *before* the test runs.
- A **power-analysis script** that produces the sample
  size and duration from the numbers in the design
  doc — reproducibly, not on the back of a napkin.
- A **read-out template** (two pages) that a launch
  review will actually use.
- A written **risk register** for the four ways an A/B
  test lies (novelty, Simpson's paradox, SRM,
  guardrail-shrug) applied to your specific rollout.

The exercise is not "run an A/B test" (that would
require a live platform and real traffic). It is
"produce the pre-registered design artifacts a
launch review will read, in the format the launch
review expects."

## Prerequisites

- Chapter 05 read.
- Python 3.11+; `scipy` (for `scipy.stats`),
  `statsmodels`, `numpy`, `pandas`, `matplotlib` or
  `plotly` for the read-out template's plots.
- Comfort with a power analysis for a two-sample
  proportion or two-sample means test. Chapter 05's
  worked example is the reference.

## Problem statement

You will design the A/B test for **exactly one** of the
following rollout scenarios. Pick before starting.

### Scenario 1 — Churn model v1 (default; matches
  chapter 05's worked example)

The subscription-churn model from chapter 01's worked
example blueprint. Version 1 replaces the rules-based
v0. Retention team acts on the top-N ranked users per
week; the intervention is a discount offer.

- Business metric: 30-day-forward net-revenue-retention
  (NRR) on the Pro plan.
- Population: currently-active Pro-plan subscribers.
- Baseline NRR: 92%. Standard deviation: 3.2%.
- Retention-team capacity: 800 outreach contacts / week.
- Historically validated intervention: 5% relative
  churn reduction on intervened cohort.
- Eligible users per week: ~85,000.
- Cost of the outreach: negligible internal cost.
- Test type: two-sided; new model may be worse.

### Scenario 2 — Search-ranking v2

The e-commerce search-ranking model is being upgraded
from v1 to v2. Users search; ranking model orders the
top 100 candidate products; the top-10 are shown.

- Business metric: sessions with at least one purchase.
- Population: any search session on the site.
- Baseline conversion rate: 3.4%. Highly seasonal.
- Historic A/B win of a comparable magnitude ranking
  change: 30 basis points absolute.
- Eligible sessions per day: ~1.2M.
- Latency SLO: p95 ≤ 250ms end-to-end.
- Test type: two-sided; interference risk (users
  browsing multiple products may compare).

### Scenario 3 — Support-ticket priority scorer

A model scores inbound support tickets by urgency; the
support team's queueing tool uses the score to
reorder. Version 1 is the first launch (no ML
baseline; heuristic ordering by ticket age is v0).

- Business metric: first-response time on the top
  quintile of tickets by score.
- Population: inbound support tickets in the
  enterprise queue.
- Baseline p50 first-response time: 42 minutes.
  p90: 3.5 hours.
- Support-team capacity: fixed — the same total
  headcount handles the same total volume; only the
  ordering changes.
- Eligible tickets per week: ~4,800.
- Test type: one-sided (the model can only be an
  improvement over ticket-age ordering).
- Contamination risk: LOW (each ticket is a distinct
  atom; no user assignment issues).

## Requirements

### Part A — The A/B design doc (≈ 90 min)

Produce `design/ab_design_doc.md`. It must follow
chapter 05 Section 7's template exactly and include
every section:

1. **Hypothesis.** One sentence: "Deploying [system]
   into [population] will increase [primary metric]
   by at least [MDE] vs [control]."
2. **Unit of randomisation.** Named specifically
   (subscriber_id, session_id, ticket_id).
   Discussion of any interference or contamination
   risks.
3. **Variants.** Control and treatment named; any
   held-out cohort called out explicitly (chapter
   04 mitigation 2 — untreated control).
4. **Primary metric.** Definition and how it is
   computed. Reference to where the data lives.
5. **Guardrails.** At least three guardrails, each
   with direction, threshold, and stopping rule.
6. **Power analysis.** All parameters from Part B
   below.
7. **Ramp plan.** Named stages with durations.
8. **Stopping rules.** Explicit conditions that
   halt the test.
9. **Read-out decision rule.** Explicit
   pre-registered rule: what conditions launch
   the treatment.

Length target: 1-2 pages. Match the format of
chapter 05's example; no free-form prose without a
header.

### Part B — The power-analysis script (≈ 60 min)

Write `analysis/power.py`. Given the scenario's
numbers as inputs (either hardcoded constants at
the top of the file or CLI arguments), the script
computes:

- **Sample size per variant** for the primary
  metric at the specified MDE, alpha, and power.
- **Test duration** given the eligible-population
  volume and the ramp allocation.
- **CUPED-adjusted sample size** if a covariate is
  available (chapter 05 mentions the technique;
  `statsmodels` or hand-written implementation is
  fine).
- **Break-even MDE** for a given test duration — if
  the launch has a 3-week deadline, what is the
  smallest effect the test can detect?

The script's output is `analysis/POWER.md`, an
auto-generated markdown table of scenarios:

```
| MDE (abs)  | N per variant | Weeks (at 50/50) | Weeks (at 10/10 canary) |
| 0.001      | ...           | ...              | ...                     |
| 0.002      | ...           | ...              | ...                     |
| 0.003      | ...           | ...              | ...                     |
| 0.005      | ...           | ...              | ...                     |
```

`README.md` in the same directory names the
assumptions the calculation uses (metric
distribution: Bernoulli / normal; variance:
measured or assumed; test type: one- or
two-sided).

### Part C — The risk register (≈ 30 min)

Write `design/RISKS.md`. Four sections, one per
canonical A/B failure mode (chapter 05 Section 6):

1. **Novelty effect.** For your scenario, what
   would a novelty spike look like? What would you
   watch for? How long into the test would you
   distrust the aggregate?
2. **Simpson's paradox.** For your scenario, what
   slices are most likely to move differently from
   the aggregate? What is the pre-registered
   slice list you will monitor?
3. **Sample-ratio mismatch.** What is the SRM
   check? What is the alerting threshold? Who
   investigates when it fires?
4. **Guardrail-shrug.** Which guardrail is the
   most tempting to shrug off (usually the one
   with the least-clear ownership)? What is the
   pre-registered rule that prevents the shrug?

Each section is a short paragraph. The point is
that these four risks were considered *before* the
test runs, so the read-out cannot be surprised.

### Part D — The read-out template (≈ 30 min)

Write `readout/TEMPLATE.md`. This is the document
the launch review will read *after* the test
completes. Chapter 05 Section 8 gives the shape.
Sections:

1. **Primary metric result.** Point estimate, CI,
   p-value, one-sentence summary.
2. **Guardrails table.** Each guardrail with
   observed value, status (green / regressed /
   breached), and stopping rule outcome.
3. **Slice-level table.** Primary metric per
   pre-registered slice.
4. **Novelty check.** A time-series plot of the
   primary metric across the test's duration.
5. **SRM check.** Observed vs designed assignment
   ratio.
6. **Recommendation.** Launch / roll back / hold,
   with reference to the design doc's decision
   rule.

The template should be usable without editing to
present the *hypothetical* result of your test.
Include placeholders (`{point_estimate}`,
`{ci_lower}`, etc.) where the numbers will land.

Also write `readout/simulate.py` — a small script
that fills in the template with a **hypothetical
result** using either (a) random numbers drawn
from a plausible distribution around the null and
around the MDE, or (b) numbers you invent. Two
runs: one where the primary metric hits the MDE
cleanly, one where it doesn't. Both produce a
fully-filled read-out, so a reviewer can see what
"launch" and "no launch" look like end-to-end.

### Part E — The dry-run defense (≈ 30 min)

Simulate the launch review. Write
`review/DRY_RUN.md`: the specific 10-15 questions
a senior reviewer would ask about your design, in
order, with your one-sentence answers. Chapter 07's
Section E questions are the starting point; the
scenario-specific ones are yours.

The goal is that if a reviewer walked in cold and
asked the questions in your dry-run, you would
have a specific answer for each.

## Starter guidance

- **The MDE picks the sample size, and the sample
  size picks the duration.** Do the power analysis
  early; it determines whether the design is even
  feasible in the scenario's timeframe. A design
  that requires four months of testing to detect a
  0.02pp lift is not a design.
- **The guardrails must have owners.** Every
  guardrail in the design doc names the team that
  reads it. If the team isn't identified, the
  guardrail is decoration.
- **Do NOT power on the primary and read on a
  secondary.** If you're tempted to write "we'll
  also look at [other metric]," it is a secondary
  metric with a wider CI and a weaker
  interpretation. Say so explicitly.
- **Interference is real for Scenario 2.** Users
  who browse many products may compare across
  variants. The design doc has to name this and
  say how it will be handled (accept the noise,
  or use a design that reduces it).
- **The read-out template writes itself if the
  design doc is precise.** If the read-out is
  hard to draft, the design doc is missing
  information.
- **Two runs of the simulator, not one.** The
  point is to see both what a launch decision
  looks like and what a no-launch decision looks
  like, in the same format. That's how the
  launch-review muscle memory builds.

## Acceptance criteria

- `design/ab_design_doc.md` exists and contains all
  nine of chapter 05 Section 7's sections.
- The design doc's `Guardrails` section names at
  least three guardrails, each with a direction, a
  threshold, and a stopping rule.
- The design doc's `Read-out decision rule` is
  pre-registered — a specific rule that says
  "launch if X, do not launch if Y, ambiguous
  means no."
- `analysis/power.py` runs and produces a
  `POWER.md` table. Running it with the numbers
  from the design doc reproduces the sample-size
  and duration figures the design doc names.
- `design/RISKS.md` addresses all four canonical
  A/B failure modes.
- `readout/TEMPLATE.md` exists and covers all six
  read-out sections.
- `readout/simulate.py` fills the template with
  two hypothetical results: one launch-worthy,
  one not.
- `review/DRY_RUN.md` has at least ten
  reviewer-simulator questions with one-sentence
  answers.

## Stretch goals

- **A CUPED implementation.** Actually implement
  CUPED variance reduction: pick a pre-period
  covariate (for churn model: prior-30-day NRR;
  for search: prior week's conversion), simulate
  the covariate for the sample, and show the
  variance-reduction ratio in `POWER.md`.
- **A switchback design for Scenario 2.** Sketch
  what a switchback A/B (alternating variants by
  hour or day at the service level) would look
  like for the search-ranking scenario, and how
  it would change the power analysis.
- **A sequential-testing extension.** Add an
  always-valid confidence interval (Howard et al.'s
  mixture supermartingale or a simple SPRT) to
  the read-out template so peeking is safe.
- **An interference simulation.** For Scenario 2,
  simulate a user population where 20% of users
  see multiple search-result pages; measure the
  bias a naive user-level A/B produces vs a
  cluster-based design.
- **A "what if the test result is ambiguous"
  playbook.** A common outcome — the primary
  metric is directionally right but not
  significant. What's the next step? A follow-up
  test with more traffic? A one-tailed extension?
  Write the playbook.

## What to hand in

```
exercise-03/
├── design/
│   ├── ab_design_doc.md          # NEW — chapter 05 template
│   └── RISKS.md                  # NEW — four canonical risks
├── analysis/
│   ├── power.py                  # NEW — reproducible power analysis
│   ├── POWER.md                  # NEW — auto-generated table
│   └── README.md                 # NEW — assumptions doc
├── readout/
│   ├── TEMPLATE.md               # NEW — two-page read-out
│   └── simulate.py               # NEW — fills template with two hypotheticals
├── review/
│   └── DRY_RUN.md                # NEW — reviewer questions + answers
├── pyproject.toml
└── README.md                     # 1 paragraph: which scenario, how to run
```

Solutions live in the paired `ml-engineer-solutions`
repo — do not peek until you have submitted yours.
