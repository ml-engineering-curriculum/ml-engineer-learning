# 05 — A/B testing ML systems

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 01–04. `mod-107` chapter 09
> (baselines) is the closest neighbour.

## Motivation

Chapter 03 argued that offline metrics can lie because
the feature vectors diverged between training and
serving. Chapter 04 argued that offline metrics can lie
because the training data itself is model-shaped. Both
failure modes have the same shape: **the number on the
offline evaluation dashboard is not the number the
business will see.**

The A/B test is the discipline that fixes both. It is a
controlled experiment that compares two variants of the
system — usually "the current model" and "the candidate
model," or "the model" and "no model" — against a live
sample of real users, measures a real business outcome,
and reports whether the difference is real. It is the
only tool in the ML systems designer's kit that gives
an honest answer to "did this ship a win?"

A/B testing is a huge topic. The subject has its own
practitioners, textbooks, and industry conferences. This
chapter's job is not to be the definitive reference; it
is to be the working ML engineer's operating manual.
What you must design before you ramp; what guardrails
must fire; what to watch for during the test; what to
do with the result. Everything else — the deep statistics
of sequential testing, the variance-reduction techniques
like CUPED, the platform engineering of exposure
allocation — is in the reading list.

## The frame

An A/B test on an ML system almost always has this
shape:

- **Unit of randomisation.** The atom being assigned
  to control or treatment. Usually a user; sometimes a
  session, a search query, a request.
- **Control variant.** What is currently serving. For
  the first launch of a model, control is the pre-model
  baseline; for a model refresh, control is the
  currently-deployed version.
- **Treatment variant.** The candidate.
- **Primary metric.** The single number the test is
  powered to move. Usually a business metric; sometimes
  a customer-experience metric; almost never an ML
  metric.
- **Guardrail metrics.** Things that must *not* get
  worse. Latency, error rate, downstream costs,
  fairness slices, revenue on unaffected surfaces.
- **Sample size.** How many units are needed to detect
  the smallest effect worth caring about.
- **Duration.** How long the test runs before you
  read the result.
- **Ramp plan.** The path from 0% to full treatment
  traffic.

Design each of these consciously before writing a line
of experiment-platform code.

## Section 1 — Unit of randomisation

The unit determines what "one observation" means and
therefore what "the sample size is `N`" means. Choose
badly here and every downstream statistic is
misleading.

Common units and when to pick each:

- **User.** Persistent across sessions and requests.
  Prevents a single user from seeing both variants
  (which would contaminate their behaviour). Default
  choice for any product where a user interacts
  repeatedly. Requires a stable user ID (logged-in
  identifier, or a cookie / device ID for logged-out).
- **Session.** One session per assignment; the same
  user could see different variants across sessions.
  Fits when the variants are ephemeral (a specific
  landing-page ranker) and the risk of within-user
  inconsistency is low. Higher sample size for a given
  power because sessions are more numerous and less
  correlated within a unit.
- **Request / query.** Every request gets its own
  assignment. Highest statistical power; highest
  contamination risk (the same user sees both
  variants in rapid succession). Fits when the
  variant's effect is isolated to that request and
  users don't compare across requests (e.g., ranking
  in a search-results page for a very high-QPS
  service).
- **Cluster / geo.** Assignment at the level of a
  city, region, or team. Fits when interference is a
  problem — the treatment cannot be isolated to one
  user because users interact (marketplaces, social
  networks). Sample size shrinks dramatically because
  `N` is now the number of clusters, not the number
  of users.

The two silent traps:

- **Interference / network effects.** If a treatment
  user's behaviour affects a control user's
  experience (marketplace pricing, social feed
  ranking, ride-share matching), user-level
  randomisation gives biased estimates. Cluster or
  switchback designs are the fix; both cost sample
  size. This is the reason marketplace companies
  publish so much about experimentation methodology
  — the naive design produces answers that are
  systematically wrong.
- **Contamination.** A user assigned to treatment
  logs out and comes back on a different device and
  gets assigned to control. Their behaviour blends
  the two. The rate of contamination bounds how
  informative the test can be.

Both traps deserve a paragraph in the design doc.
Don't leave them to be discovered during the read-out.

## Section 2 — Choosing the primary metric

The metric that ends up on the "did the test win?"
slide.

**Rule 1:** the primary metric is a business or
customer metric, not an ML metric. AUC did not move
the number the exec team looks at. Conversion did.
Retention did. Session length did.

**Rule 2:** the primary metric is *upstream of the
model's action, at a horizon shorter than the labels'
maturation.* If the label matures in 30 days and the
test runs for two weeks, the primary metric cannot be
the label. Pick a shorter-horizon proxy that is
mechanistically linked to the label.

**Rule 3:** the primary metric is *sensitive enough to
move* under a plausible model win. A metric that is
correct in direction but too noisy to detect at the
test's sample size is not usable; power calculations
in Section 4 quantify this.

**Rule 4:** the metric has an owner. When it moves,
somebody outside the ML team has a well-formed opinion
about whether the movement is good. If nobody owns
the metric, the test result is a curiosity.

Two examples to anchor:

- **Churn model (from chapter 01).** Primary metric:
  30-day-forward net-revenue-retention for the
  outreach cohort. Proxy for the label ("did the user
  churn?") which matures at 90 days.
- **Search ranking.** Primary metric: sessions with at
  least one successful outcome (click through a
  result, add to cart) per user per week. Proxy for
  long-term retention.

Neither is "AUC." Neither is "prediction accuracy."
Both are what the business already tracks.

## Section 3 — Guardrail metrics

The primary metric is what the test is trying to move;
the guardrail metrics are what must *not* get worse
while it does. Every A/B test on an ML system has
guardrails. If yours doesn't, you didn't design one.

Standard guardrails on any ML rollout:

- **Serving latency.** p50, p95, p99. If the candidate
  model is heavier, latency can regress in ways the
  offline benchmark didn't show.
- **Error rate.** HTTP 5xx, application-level errors.
  A candidate model that occasionally times out is a
  regression regardless of accuracy.
- **Downstream revenue.** For any change to a system
  that touches revenue, an aggregate revenue metric
  is a guardrail even if it isn't the primary.
- **Fairness / slice metrics.** For each protected or
  business-critical slice (region, plan tier,
  customer segment), the primary metric per slice
  should not regress meaningfully. Chapter 04's
  detectors and `mod-107` chapter 11 give the
  vocabulary.
- **Behavioural side effects.** Depending on the
  system: unsubscribe rate, complaint rate, refund
  rate, support-contact rate. If the treatment
  drives complaints up, no accuracy win pays for it.

Each guardrail has:

- A **direction** (must not decrease / must not
  increase).
- A **threshold** (what regression is acceptable —
  often "0" for latency SLOs, sometimes "no worse
  than 2% relative" for revenue).
- An **owner** (the team that reads the guardrail's
  metric and can call the test off).

Guardrail failure is a stop-the-test action. Set the
stopping rule in advance in writing.

## Section 4 — Sample size and duration

The two questions the platform team will ask:

- **How many users do you need?**
- **How long will it take?**

Both come from a **power analysis**. The mechanics are
worth memorising to the point of being able to do them
on the whiteboard.

Given:

- **Baseline metric value** — what the primary metric
  averages today (say, `p₀ = 0.12` conversion rate).
- **Minimum detectable effect (MDE)** — the smallest
  effect you care about (say, `Δ = 0.003` — an
  absolute 30-basis-point lift, or ~2.5% relative).
- **Significance level** — `α`, conventionally 0.05.
- **Statistical power** — `1-β`, conventionally 0.80.
- **Metric variance** — for a proportion, `σ² =
  p(1-p)`. For a continuous metric, measure from
  historical data.

The sample size per variant, for a two-sample z-test
on a proportion, is approximately:

```
n ≈ (z_{α/2} + z_β)² × 2 × σ² / Δ²
    ≈ 7.85 × 2 × 0.12 × 0.88 / 0.003²
    ≈ 184,000
```

So each variant needs ~184k users to detect a 0.3pp
absolute lift on a 12% baseline at 80% power and 5%
significance. Adjust for:

- **Ramp fraction.** If you're running at 10% traffic
  to each variant, you need `184k / 0.1` = 1.84M
  eligible users to see 184k per variant.
- **Test duration.** Divide by daily eligible-user
  count. `1.84M / 250k daily` = ~7.4 days. Round up.
- **Weekly seasonality.** Never run a test for less
  than one week to average across day-of-week
  effects. Two weeks is better; a month is common for
  metrics with weak signal.
- **Multiple testing.** If you're powering for one
  primary metric and reading 15 secondary metrics, at
  least one will be "significant" by chance. Apply
  Bonferroni or an equivalent for secondaries; keep
  the primary interpretation strict.

Practical rules of thumb:

- **A test that says "run for 2 days"** is almost
  always underpowered or has a novelty problem
  waiting to bite. Never read a test in under a week.
- **A test that requires more users than the entire
  eligible population per month** is a signal that
  the MDE is too small; renegotiate scope before
  starting.
- **A test whose primary metric is a rare event**
  (fraud rate, unsubscribes) needs longer than you
  think. Variance dominates.

Two techniques that dramatically reduce required
sample size:

- **CUPED (Controlled experiment Using Pre-Existing
  Data).** Subtract a pre-period covariate from each
  observation to reduce variance without changing the
  estimate's expected value. Publicly documented by
  Microsoft's experimentation team; widely adopted
  in industry.
- **Stratified sampling.** Stratify on known-important
  covariates (segment, region, device) and analyse
  within strata; recombine with weights. Reduces
  variance from covariates the randomisation didn't
  balance.

Both live in the "advanced but standard" bucket for a
mature experimentation platform. If your platform
supports them, use them; if it doesn't, they are the
first requests to make of the platform team.

## Section 5 — Ramp plan

You don't set a test to 50/50 on day one. You ramp.
The pattern:

1. **Shadow / dark launch.** Zero traffic to the
   treatment as a customer-facing variant; the
   candidate model runs alongside the control model
   and its predictions are logged for offline
   comparison. Chapter 03's shadow-mode is exactly
   this — the point is to catch training-serving skew
   before any user is affected. Lasts 1-3 days.
2. **1% canary.** The treatment starts receiving real
   traffic at 1%. Nothing about the test is powered
   yet; the point is that latency, error rate, and
   sanity checks fire before wider exposure. Lasts
   1-2 days.
3. **Test ramp.** Move to the designed test allocation
   (typically 50/50, sometimes an asymmetric split if
   the treatment carries risk). Run for the designed
   duration.
4. **Read-out.** Analyse the test at its designed
   duration. Not before.
5. **Launch or roll back.** If the test wins, ramp
   treatment to 100%. If it loses or is flat, roll
   back. If it's ambiguous, the pre-registered
   decision rule from the design doc applies (usually:
   "no launch unless the primary metric significantly
   improves; a flat result is a no").

Two things the ramp discipline prevents:

- **Peeking.** Watching the test's primary metric
  every hour and stopping it as soon as it looks
  significant is *the* classic way to fool yourself.
  A test read at a random time is meaningfully more
  likely to be false-positive than one read at its
  planned duration. Some platforms support formal
  sequential-testing methods (SPRT, always-valid
  p-values) that make continuous monitoring safe;
  most in-house implementations do not.
- **A launch based on a launch-day spike.** The first
  day of a test often looks different because of
  novelty or bugs. Read the aggregate over the full
  duration, not the trend.

## Section 6 — Novelty, Simpson's paradox, and other
   ways an A/B test lies

Even a well-designed test can produce a misleading
result. Four traps to watch for in the read-out.

### Novelty effects

The treatment produces a spike (positive or negative)
in the first few days that decays over the test's
duration. Users are reacting to the *change*, not the
underlying quality of the treatment. A common shape:
recommender variants that show unfamiliar items get an
initial clickthrough spike as users click to
investigate, then normalise. The long-run effect is
what you want to measure; the short-run is a
distraction. Rule of thumb: if the effect is not
sustained over the last third of the test's duration,
distrust the aggregate.

### Simpson's paradox

The aggregate metric moves one direction; every
sub-group moves the opposite direction. Classical
example: overall conversion goes down under the
treatment, but conversion goes up for both new users
and returning users individually — because the
treatment shifted the mix of who saw the site. In ML
systems, this usually surfaces because the model's
prediction-based traffic routing changes the
composition of who lands on a treated variant. Fix
by looking at slice-level metrics alongside
aggregate; the story is only "true" if aggregate and
slices agree.

### Sample-ratio mismatch (SRM)

The design says 50/50; the actual ratio at read-out
is 51.3/48.7. If the mismatch is statistically
significant, *something is wrong with the assignment*
— usually a bug in the routing layer that
systematically bounces users from one variant back
to the other. SRM invalidates the test's statistical
claims entirely; the fix is to figure out why the
assignment is asymmetric, not to interpret the
result. Every mature experimentation platform runs
an automated SRM check.

### Guardrail regression that gets shrugged off

The primary metric moved up; the latency guardrail
regressed by 6%. The temptation is to launch anyway.
Six months later the platform team notices that
latency has drifted up across all services and the
oncall load has risen; the guardrail regression was
the leading indicator. Guardrails exist because
they get shrugged off; make the pre-registered
stopping rule the arbiter, not the vibe of the
read-out meeting.

## Section 7 — The design doc

A one-page A/B design doc, filed before the test
starts, is the artifact. Sections:

```
# A/B test — churn model v1 rollout
Owner: <you>   Reviewers: retention lead, exp platform PM
Status: pre-launch  Test ID: exp_2026_08_churn_v1

## Hypothesis
Deploying churn model v1 (LightGBM) into the retention
outreach flow increases 30-day-forward
net-revenue-retention on the Pro plan by at least
0.3pp absolute, vs the current heuristic (v0).

## Unit of randomisation
User (stable subscriber_id).

## Variants
- Control (v0): current heuristic outreach list.
- Treatment (v1): churn model v1 outreach list.
- Cohort held out from both: 10% "no outreach at
  all" — see chapter 04 mitigation 2. Not part of
  this test's read-out; used for training data
  cleanliness only.

## Primary metric
30-day-forward NRR for Pro-plan subscribers active at
test-assignment time.

## Guardrails
- p95 outreach-list-generation latency < 500ms (SLO).
- Retention-outreach team weekly capacity ≤ 800
  contacts (system constraint).
- Unsubscribe rate on outreached users must not
  regress by more than 5% relative.
- Slice-level NRR must not regress for any plan-tier
  x country cell with n > 5000.

## Power analysis
Baseline NRR: 92%. Historical std: 3.2%.
MDE: 0.3pp absolute (0.32% relative).
Alpha: 0.05. Power: 0.80. Two-sided.
Per-variant N: 187,500 (using CUPED with prior
30-day NRR as covariate). Reduce to ~120k with
CUPED variance reduction.
Total eligible users per week: ~85k.
Duration: 3 weeks at 50/50 = 127k per variant. OK.

## Ramp plan
- Days -3 to 0: shadow mode. Prediction parity
  checked (chapter 03).
- Day 1: 1% canary (0.5% each variant).
- Day 2: 50/50 test ramp.
- Days 2-23: full duration; no peeking.
- Day 24: read-out.

## Stopping rules
- Latency guardrail breach for >10 minutes: stop.
- Unsubscribe guardrail breach: stop.
- SRM detected: stop, investigate, restart.
- Otherwise: run to designed duration.

## Read-out decision rule
Launch v1 to 100% if:
- Primary metric NRR improves by ≥ 0.3pp with
  p < 0.05, AND
- No guardrail regression, AND
- No slice-level regression per the above rule.
Otherwise: no launch. Ambiguous means no.
```

This doc is the thing you commit to a repo. Everything
in the read-out meeting refers back to it. Deviating
from it is a design decision, not a judgment call.

## The read-out — reporting the result

Two pages, in this order:

1. **The primary metric result.** Point estimate,
   confidence interval, p-value. One sentence:
   "The test moved / did not move the primary
   metric by X (95% CI [A, B]; p = Z)."
2. **The guardrails.** Table with each guardrail's
   result and status (green / regressed / breached).
3. **The slice-level story.** Per-slice primary
   metric, with any regressions called out.
4. **The novelty check.** Time series of the primary
   metric across the test duration.
5. **SRM check.** Assignment ratio observed vs
   designed.
6. **The recommendation.** Launch / roll back / hold
   for further analysis. Refers back to the design
   doc's decision rule.

Anything else — narrative, hypotheses, second-order
metrics — is an appendix. The two-page read-out is what
the launch review reads.

## Common pitfalls

1. **Reading the test early.** Peeking inflates
   false-positive rate. If your platform doesn't
   support sequential testing, you read at the
   designed duration, period.
2. **Powering for the wrong metric.** The test is
   powered on primary; the launch decision is made on
   a secondary that wasn't powered. It will be
   noisy; the launch is a coin flip.
3. **No guardrails.** Latency, error rate, revenue
   on unaffected surfaces — pick the ones that
   matter and put a threshold on each.
4. **Guardrails without stopping rules.** The
   regression fires; the team debates; the test
   keeps running. The regression persists.
5. **Ignoring novelty effects.** Day-one spike is
   read as a win; the launch reverses it a month
   later.
6. **Ignoring interference in marketplaces / social
   networks.** User-level randomisation reports a
   biased estimate; the launch decision is wrong.
7. **No SRM check.** The routing layer is buggy; the
   test's assumptions don't hold; nobody notices.
8. **Reading offline metrics as if they were A/B
   results.** Chapter 03 and chapter 04 covered why
   they lie; the A/B test is the answer.
9. **A test with no pre-registered decision rule.**
   The result is ambiguous; someone launches
   anyway because they built it.
10. **Skipping the shadow-mode step.** Real users
    absorb the training-serving-skew bug that a
    shadow-mode day would have caught for free.
11. **Assuming the same test structure works for
    marketplaces.** Interference invalidates the
    naïve user-split; a switchback or geo design is
    required and requires meaningfully more sample.
12. **Not planning for a re-run.** Some tests are
    genuinely ambiguous. The launch review's answer
    is often "we need one more test," and there
    should be a plan for that.

## Summary

- An A/B test is the only reliable way to measure
  whether an ML change moved a business metric.
  Offline metrics are advisory; A/B is decisive.
- Design every A/B with: unit of randomisation, one
  primary metric, guardrails with stopping rules,
  power analysis, ramp plan, and a pre-registered
  decision rule.
- Pick the primary metric to be a business or
  customer metric measurable within the test
  duration.
- Guardrails exist to protect what you're not
  optimising. Latency, revenue, fairness slices,
  behavioural side effects. Every one gets a
  threshold and a stopping rule.
- Ramp from shadow → canary → test → launch. Never
  read the test before the designed duration.
- Watch for novelty, Simpson's paradox, SRM, and
  interference. Each has a specific check.
- The design doc is the artifact. Deviations from it
  are design decisions, not judgment calls.

## Where this goes next

Chapter 06 covers the meta-question that ought to
have come first: **should you have built this at
all?** The build-vs-buy decision, and the four
peer-specialist tracks (LLM application, RAG,
fine-tuning, evaluation) that own ML problems the
generalist ML engineer should delegate rather than
solve. Get that question wrong on day one and every
chapter above is expensive infrastructure for a
question a $50/month API answered.
