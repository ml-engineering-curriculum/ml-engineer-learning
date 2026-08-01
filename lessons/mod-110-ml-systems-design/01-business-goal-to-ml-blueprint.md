# 01 — From business goal to ML system blueprint

> **Reading time:** ~20 minutes.
> **Prerequisite modules:** the whole track so far — you have
> shipped training pipelines (`mod-104`, `mod-105`), tracked
> experiments (`mod-106`), evaluated defensibly (`mod-107`),
> served a model (`mod-108`), and monitored it (`mod-109`).
> This module composes all of it.

## Motivation

The single most common way an ML project fails is not that
the model was inaccurate. It is that the model was accurate
on the wrong problem — one nobody bothered to translate from
the business goal.

The stakeholder asks for "an AI to reduce churn." Six weeks
later a data scientist demos a model with an AUC of 0.83.
Nobody knows what the model would *do* with a prediction; no
retention team is set up to receive one; no downstream
budget exists to intervene on flagged users; and the model
is trained on last quarter's churn — which, it turns out, is
a different thing from *next* quarter's churn. The demo is
impressive. The project ships to nowhere.

**This chapter is the interview-style walkthrough that
prevents that outcome.** It walks the sequence of questions
an experienced ML engineer asks *before* opening a notebook.
The output is not code; it is a one-page blueprint — a
document specific enough that the next person on the team
could implement it and the same person a quarter later could
defend it in a design review.

The rest of the module builds on this blueprint. Chapter 02
lays out the reference architecture the blueprint fits into.
Chapters 03–04 harden it against the two failure modes that
break more ML systems than anything else. Chapter 05 covers
how you prove, once shipped, that it actually moved the
number. Chapter 06 covers the "should you have built any of
it at all" question that ought to precede the whole exercise.

## The blueprint you are building

Concretely, the artifact you produce by the end of this
chapter is a document with these sections:

1. **Business context and success criteria.** What is the
   goal, whose goal is it, and how will everyone know
   whether it was hit?
2. **The ML formulation.** Task type, target, features,
   inputs, and — as importantly — what the model is *not*
   going to decide.
3. **Data.** What sources exist, what labels exist, what
   the cold-start plan is.
4. **System shape.** Online / batch / streaming; latency
   budget; scale.
5. **Baseline and phasing.** What ships before any model
   does. What version 1 looks like. What version 2 buys.
6. **Risks and non-goals.** What could go wrong, what is
   out of scope on purpose.

Each of the sections below is one of those.

## Section 1 — Business context and success criteria

Ask the questions in this order. The order matters — later
answers depend on earlier ones.

### Q1: What is the business decision this model informs?

Not "what does the model predict?" — **what does the
business do differently because of the prediction?** A
churn model that nobody acts on has zero business value
regardless of its AUC. A fraud model that flags a
transaction but has no downstream reviewer either
auto-blocks (which changes the product) or does nothing
(which is a monitoring dashboard, not an ML product).

Concrete decisions to name explicitly:

- **Automated action.** The prediction directly flips a
  bit. "Approve or deny this transaction." "Show this ad
  or that ad." The model *is* the product surface.
- **Ranking or filtering.** The prediction reorders a
  list. "Which support tickets are most urgent?" "Which
  documents does the search return?"
- **Human-in-the-loop routing.** The prediction sends a
  case to a reviewer. "Flag suspicious accounts for the
  trust team." The reviewer, not the model, is the final
  decision-maker.
- **Suggestion or nudge.** The prediction lands in the UI
  and the user decides. "Users who bought this also
  bought…" "You may be interested in…"
- **Internal analytics.** The prediction never touches a
  user; it feeds a dashboard or a report.

Each of these has a different bar for accuracy, latency,
explainability, and monitoring. A ranking model can be
wrong on 40% of items and be a huge product win because the
top-of-list changes. An auto-decision model that is wrong
on 5% of decisions can be a catastrophe.

### Q2: What is the metric the business already tracks?

Before defining a new metric, find the one already on the
executive dashboard. The stakeholders care about it,
finance can attach a dollar figure to it, and no new
reporting infrastructure needs to be built to move it.

Common ones by pattern:

- **E-commerce.** GMV, conversion rate, AOV, repeat-
  purchase rate, refund rate.
- **Subscription.** MRR / ARR, gross churn, net revenue
  retention, activation rate, engagement DAUs / WAUs.
- **Marketplace.** Take rate, liquidity (time-to-match),
  fill rate.
- **Support / trust.** Handle time, first-contact
  resolution rate, false-positive block rate,
  chargeback rate.
- **Ads / content.** CTR, revenue per session, time on
  site, retention at 30 days.

The ML metric (AUC, RMSE, Recall@K) is a *proxy* for the
business metric. Name the business metric first; then
design an ML metric that credibly moves it (chapter 05
covers how you actually measure the link).

### Q3: What is the smallest change worth the effort?

The **minimum detectable effect (MDE)** frames scope
before you commit engineering time. If churn is 12% and
the retention team says "shaving even 0.3 percentage
points is worth the quarter," you have a budget: the
system has to be at least accurate enough that a
plausible intervention on the top-N flagged users
recovers 0.3pp of churn. Do napkin math *before*
architecture. Common outcomes:

- **The MDE is achievable with a heuristic.** Ship the
  heuristic; declare victory; do not build an ML system.
  Chapter 06 revisits this.
- **The MDE requires a fraction of a percent of accuracy
  at scale.** You need a serious model, careful
  monitoring, and an A/B framework. Budget accordingly.
- **The MDE is delusional.** "A 5x lift in conversion
  from a recommender." Say so, in writing, in this
  section. Do not commit to it.

### Q4: Who is accountable when it goes wrong?

Every production ML system has a runbook, a paging
rotation, and a person who owns the model. If nobody can
answer "who gets paged when the model regresses?" the
system is a demo, not a product. This is a design-review
question, not a launch question — surface it in the
blueprint so the rotation is set up during build, not
scrambled together in the aftermath of the first
incident.

## Section 2 — The ML formulation

The business goal maps to a *specific* ML task. This step
is where inexperienced teams lose weeks.

### The target

Nail three things:

1. **The unit of prediction.** User? Session? Item? Pair
   of (user, item) at some timestamp? Transaction? A
   churn model for `users` predicts one number per user;
   a churn model for `(user, subscription)` predicts one
   per subscription. The choice cascades through the
   entire feature store and evaluation.
2. **The prediction horizon.** "Will churn *when*?" The
   next 7 days? 30 days? The end of the current billing
   cycle? The horizon changes the target definition,
   which labels exist, and how long you must wait to
   know whether the model was right.
3. **The label definition.** "Churn" is not obvious.
   Cancelled subscription? Downgraded plan? Zero
   activity for 14 days? Explicit unsubscribe? Every
   choice is a different target and different label
   pipeline. Pick one and *document it in words a
   non-ML colleague could read*.

The mistake to avoid: **letting the label be whatever the
data warehouse already has.** The warehouse has an
`is_active` flag, and it becomes the label because it's
convenient. Six months in, someone asks "why do we predict
churn for people who obviously churned two weeks ago?" and
the answer is that `is_active` is set by an ETL job that
runs on the 1st of every month. Define the label to fit
the business decision (Q1); then build or buy the label
pipeline. Do not let the pipeline pick the label.

### The task type

- **Binary classification.** Churn / no-churn, fraud /
  legitimate, click / no-click. Evaluate with the metrics
  from `mod-107` (Recall @ operating point, AUC-PR for
  class-imbalanced problems).
- **Multi-class classification.** Which support queue?
  Which product category?
- **Regression.** How much will this order be worth? How
  long until this ticket resolves? Evaluate with RMSE or
  MAE against the business's tolerance for over- vs
  under-prediction (`mod-107` chapter 02).
- **Ranking.** Top-K recommendations, search results.
  Evaluate with Recall@K, NDCG (`mod-107` chapter 05).
- **Sequence / time series.** Demand forecasting, session
  next-event. Different validation splits (`mod-107`
  chapter 07) and different serving patterns.
- **Structured generation.** Extract entities, summarise,
  rewrite. Increasingly LLM-shaped; chapter 06 covers
  when to delegate to that specialist track.

### What the model does not decide

This is easy to skip and expensive to skip. Write down,
explicitly, the decisions the model is *not* making:

- "The model does not decide whether to charge the user;
  the payments team does."
- "The model does not decide which content to remove;
  the trust team does."
- "The model does not decide the discount value; the
  pricing team does."

The list clarifies the human loop, sets the review path
for false positives, and — critically — makes explicit
which teams have to agree to the launch. If four teams
touch the workflow, four teams review the launch. Naming
them here saves the last-week-before-launch surprise.

## Section 3 — Data

### The three questions to answer before feature engineering

1. **Where does the ground truth live?** In a table, in
   a log, in a human labeler queue, or nowhere yet? If
   nowhere, you are building the labeling pipeline
   before you are building the model (`mod-109`
   chapter 05).
2. **What is the label latency?** Days? Weeks? Months?
   A churn label at a 90-day horizon arrives 90 days
   after the prediction is made. That number sets your
   retraining cadence, your monitoring plan, and how
   fast you can iterate.
3. **What features are already available at prediction
   time?** This is the training-serving-skew question
   coming (chapter 03). Any feature the training set
   has that the serving path does not is a bug in the
   making. Any feature that is available offline "as of
   the label date" but only available online "as of
   now" is the same bug in disguise.

### The cold-start plan

If the model is for something the business has never done
before — a new product line, a new region, a new user
segment — the training set is *tiny or zero* on day one.
The blueprint has to say, in writing, how you handle it:

- **Ship without ML first.** A rules-based system
  gathers labels. The next quarter's ML model trains on
  the ruleset's outputs. Chapter 06 revisits this
  ordering.
- **Bootstrap from a related problem.** A model trained
  on adjacent data (US churn transferred to the UK
  launch) with a plan to retrain once the target data
  matures.
- **Human labelers in the loop from day one.** Common
  for trust and content moderation. Expensive; build
  the labeling ops before the model.
- **LLM zero-shot or few-shot classification.** Cheap
  and fast for text-shaped tasks; a lot less accurate
  than a fine-tuned model. Reasonable for the
  cold-start; not a permanent answer. Chapter 06
  discusses when to delegate to the LLM app track.

Not naming a cold-start plan is not a neutral choice. It
means the model will be trained on whatever data leaks in
by accident, which is usually the worst possible split.

## Section 4 — System shape

The service pattern falls out of the business goal and
the latency budget. Four shapes, in decreasing order of
how much you should default to them:

### Batch (default)

The model runs on a schedule against a warehouse table
and writes results back to the warehouse. Downstream
systems read predictions from the warehouse.

- **Fits when** decisions are made periodically (daily
  marketing sends, weekly retention outreach, monthly
  segment reassignment), latency of "same day" or
  slower is fine, and the number of prediction subjects
  is bounded (all active users, all inventory items).
- **Cost.** Cheap. One scheduled job, one warehouse
  table, no serving infrastructure.
- **Trap.** If the business decision needs a
  *just-in-time* answer ("is this the moment to send
  the offer?"), a nightly batch is not it.

**Prefer batch until proven otherwise.** Most models the
business asks for do not need online serving. Chapter 02
comes back to this.

### Streaming

The model runs on an event stream (Kafka, Kinesis,
Pub/Sub) and writes predictions on the same stream or a
downstream table. Latency is seconds to minutes.

- **Fits when** decisions cascade in near-real-time (a
  fraud model that has to score a transaction as it
  happens, a next-best-offer model that reacts to a
  session-in-progress).
- **Cost.** Higher than batch. You are now maintaining a
  streaming platform, feature freshness guarantees, and
  a service that reads them.
- **Trap.** Streaming often gets picked because it
  sounds modern. Ask "would a 60-minute-delayed batch
  produce the same business outcome?" and honestly
  answer.

### Online request/response

The model is behind an HTTP endpoint (or gRPC) and
serves per-request predictions with a latency budget
measured in tens or hundreds of milliseconds. This is
what `mod-108` built.

- **Fits when** the prediction is *part* of the user
  request. Ranking a search-results page. Scoring a
  transaction for approval. Recommending a next
  document.
- **Cost.** Highest of the four. You are running a
  service (`mod-108`), monitoring it (`mod-109`), and
  paying for the availability SLO.
- **Trap.** Reading a stale precomputed prediction from
  a low-latency store is often the right compromise —
  online read of a batch-computed value. Cheaper than
  live inference; hits the same latency budget.

### Embedded / edge

The model runs in the client (browser, mobile, IoT
device). Prediction is local; there is no service.

- **Fits when** the data is sensitive, the network is
  unreliable, or the latency budget is under a few
  milliseconds.
- **Cost.** Model-size and update-cadence constraints
  are the main tax. Distribution is a release, not a
  deploy.
- **Trap.** You lose real-time monitoring. Chapter 07
  in `mod-109` assumes the model is somewhere you can
  query.

The blueprint picks one shape and defends it in one
paragraph. "We chose *batch* because the retention team
runs weekly outreach; nightly refresh is more than fresh
enough; we save ~$40k/yr in serving infrastructure vs
online."

## Section 5 — Baseline and phasing

### The mandatory baseline

Every ML project ships a **non-ML baseline first**. Not
as a courtesy; as a discipline. The baseline serves
three purposes:

1. **It anchors the accuracy conversation.** A model
   with AUC 0.83 sounds impressive until you learn a
   simple heuristic ("has the user logged in this
   week?") hits AUC 0.79.
2. **It de-risks the pipeline.** Building the data
   pipeline, the feature transforms, the serving path,
   and the monitoring surface *without a model in the
   loop* means the ML iteration happens against a
   working system, not a green-field one.
3. **It generates labels for the model.** If the
   baseline is deployed, the outcomes it produces are
   the model's training data. This ties directly into
   the cold-start plan and the data flywheel discussion
   in chapter 04.

The `mod-107` chapter 09 makes this concrete. The
blueprint's baseline paragraph names the heuristic in
one sentence and the metric the model has to beat by a
minimum margin to be worth deploying.

### Phasing

Split the roadmap into versions. The pattern is roughly:

- **v0 — baseline.** Ship the heuristic. Instrument it
  with the same monitoring surface the model will use
  (predictions, distributions, latency, outcomes).
- **v1 — first model.** A defensible model with the
  minimum viable feature set, evaluated against v0,
  A/B tested. Chapter 05 covers the test.
- **v2 — first iteration.** A better model (more
  features, better architecture, targeted error
  analysis from v1). One or two more A/B tests.
- **v3+ — steady-state improvement.** Regularly
  scheduled retrains, drift-triggered retrains, and
  the occasional architectural refresh.

Naming the versions bounds scope and lets the team ship
value at each step. "We'll launch v0 in three weeks and
v1 within the quarter" is more shippable than "we're
building a churn model."

## Section 6 — Risks and non-goals

The last section is the shortest and the most
undervalued.

### Risks

Enumerate two categories:

- **What could go wrong with the model?** Feedback
  loops (chapter 04), drift (chapter 03 + `mod-109`),
  training-serving skew (chapter 03), fairness /
  disparate-impact issues (`mod-107` chapter 11), cold-
  start starvation.
- **What could go wrong with the system?** Data
  source outages, upstream schema changes, retraining
  pipelines silently failing, a stale reference
  profile, a serving path that ignores a feature.

Each risk needs a *mitigation and an owner*. "We
monitor drift on all input features with the surface
from `mod-109` chapter 02; owner: ML on-call rotation."
Vague risks read as decoration; owned risks read as
engineering.

### Non-goals

An explicit list. "This model does not personalise
across sessions." "This model does not use LLMs; if the
task grows to require them, we handoff to the LLM
application track (see chapter 06)." "This model does
not attempt to handle brand-new users; the first-week
experience is rules-based."

Non-goals prevent scope creep. They also give the
design-review audience an easy way to redirect the
discussion when someone asks "have you considered
adding X?" — the answer is often "yes, and we
explicitly chose not to; here is why."

## Worked example — the churn-model blueprint in one page

Putting it all together, a concrete blueprint for a
subscription-churn model:

```
# Churn model v1 — blueprint
Owner: <you>   Reviewers: retention team lead, data-eng lead
Status: proposal
Last updated: 2026-08-01

## Business context
- Goal: reduce voluntary churn on the Pro plan.
- Business metric: net-revenue-retention (NRR) at 90 days.
  Currently 92%. Retention team target: +1pp within Q4.
- Business decision: model output feeds a retention
  outreach queue. Top-N users per week are contacted
  with a discount offer. N is set by outreach-team
  capacity (currently 800/wk).
- Business intervention already validated in an A/B by
  the retention team: 5% relative churn reduction on the
  intervened cohort.

## ML formulation
- Task: binary classification.
- Unit: (user, week).
- Target: user cancels or downgrades within the next 30
  days from the prediction date.
- Label source: `subscriptions.cancelled_at`. Delay: up
  to 30 days after prediction.
- Features (v1, from feature store):
  usage_frequency_30d, plan_tier, months_active,
  support_contacts_30d, feature_adoption_score,
  billing_events_30d. All computable from warehouse.
- Model type: gradient-boosted trees (LightGBM);
  probability output; calibration checked (`mod-107` ch 04).
- Metric: Recall @ top-800/week, evaluated on a rolling
  time-based split (`mod-107` ch 07). Baseline to beat:
  see below.

## Data
- Ground truth in the warehouse; label latency 30 days.
- Training set: 18 months of history, ~2.4M user-weeks,
  ~11% positive rate.
- Cold-start plan: not applicable — the product has
  years of history.
- Refresh cadence: weekly retrain; monthly baseline
  review.

## System shape
- Batch. Nightly job scores all active Pro-plan users;
  writes to `warehouse.churn_predictions_v1`; the
  retention team's outreach tool reads the top-N per
  week. Latency budget: overnight.

## Baseline and phasing
- v0 (this quarter, weeks 1-2): rules-based baseline —
  "users with usage_frequency_30d < 3 AND
  support_contacts_30d > 1", ranked by
  months_active DESC. Instrumented and A/B tested vs
  no-outreach control.
- v1 (this quarter, weeks 3-6): LightGBM model,
  same interface, A/B tested vs v0.
- v2 (next quarter): additional features from
  clickstream; error analysis on v1 misses.

## Risks and mitigations
- Training-serving skew: features computed by same
  warehouse transforms for both training and scoring —
  no online serving skew possible in this batch shape.
  Reviewed with data eng.
- Concept drift: retention team is rolling out a new
  pricing plan in October; retrain on post-launch data
  once we have 60 days; drift monitoring per
  `mod-109` ch 03 in the meantime.
- Feedback loop: users the model flags get outreach and
  are less likely to churn — the label for flagged
  users is a function of the intervention, not the true
  underlying propensity. Mitigation: hold out an
  untreated control every week (10% of top-N), keep as
  the training-label gold set. See chapter 04.
- Fairness: no protected-class features; slice monitor
  on plan_tier and country for prediction-rate parity.

## Non-goals
- This model does not decide the discount value.
- This model does not handle Free-plan or Enterprise-
  plan users (different economics; separate models).
- This model does not personalise the outreach content.
```

That page is enough to green-light the project and enough
to build it. Every downstream design decision — chapter
02's architecture, chapter 05's A/B test — refers back to
it.

## Common pitfalls

1. **Jumping to model architecture in the first
   meeting.** "Should we use XGBoost or a transformer?"
   is the wrong first question; Section 1's questions
   are the first questions.
2. **A metric with no dollar sign.** If nobody in
   product or finance cares about the ML metric, the
   project has no way to prove it worked.
3. **A label pipeline that is the ETL you already
   have.** The label is chosen from what's convenient,
   not what's correct. Six months in, everything
   downstream is subtly wrong.
4. **Skipping the baseline.** Model wins reported
   against `AUC ≈ 0.5` are meaningless; against a
   trained heuristic they are informative.
5. **Assuming online serving because it sounds
   modern.** Chapter 02 defaults to batch; the
   blueprint defends the choice.
6. **No cold-start plan for green-field problems.** The
   training set materialises out of accidents; every
   downstream decision is fitted to a biased sample.
7. **No non-goals.** The scope creeps by the third
   meeting; the launch date slips by two months.
8. **No named owner.** When the model regresses in six
   months, nobody is on the hook to fix it.
9. **A blueprint that requires reading a codebase to
   understand.** The document should stand alone;
   somebody joining the team a year later should be
   able to read it once and know what was built and
   why.
10. **Not writing it down at all.** The blueprint is
    the artifact. Slack threads and half-remembered
    kickoff meetings are not.

## Summary

- The ML engineer's job starts with the business
  decision, not the model.
- A one-page blueprint — context, formulation, data,
  system shape, phasing, risks — is the artifact you
  produce before any code.
- Nail three things about the target: the unit of
  prediction, the horizon, and the label definition
  in plain words.
- Default to batch; defend online serving as an
  exception.
- Ship a non-ML baseline first; every model competes
  against it.
- Every risk has a mitigation and an owner; every
  system has a runbook and someone paged.
- Every ML system has non-goals. Write them down.

## Where this goes next

Chapter 02 lays out the reference architecture the
blueprint lands in: the feature store, the training
pipeline, the model registry, the serving path, the
monitoring surface, the labels loop, and how they wire
together. From there we harden it against training-
serving skew (chapter 03), watch for feedback loops
(chapter 04), and prove it moved the number with an A/B
test (chapter 05).
