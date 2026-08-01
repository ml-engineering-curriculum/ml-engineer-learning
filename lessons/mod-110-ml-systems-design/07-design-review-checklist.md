# 07 — The ML systems design review checklist

> **Reading time:** ~15 minutes.
> **Prerequisite chapters:** 01–06.

## Motivation

Chapters 01–06 built the mental toolkit. This chapter
compresses it into a **checklist** — the questions an
experienced ML systems designer takes into every
design conversation. Not to memorise a script; to
have the right pattern in mind when someone at a
whiteboard says "so we were thinking a model that
predicts…"

Two audiences use the checklist:

- **The author of a design.** Walk it before the
  review; every question that gets a shrug is a
  weak spot to fix.
- **The reviewer.** Walk it during the review; the
  questions that get shrugged in the room become
  the follow-ups.

Both uses are the same list. The point of the
checklist is that a design review is not a debate; it
is an audit of specific claims. The claims are the
answers to these questions.

## How to use the checklist

The list below is grouped by chapter. A design
review typically takes 45-60 minutes; you will not
walk every question, only the ones the design
doesn't already address. The unanswered questions
are the review's real content.

Each question is followed by the shape of a "good"
answer. Vague answers are the follow-up list.

## Section A — Business and framing (chapter 01)

**A1. What is the business decision the model
informs?**
Good answer: "Every prediction becomes a row in the
retention outreach queue; the retention team acts on
the top N per week."
Weak answer: "We want an ML-driven approach to
churn."

**A2. What business metric is this designed to
move, and by how much?**
Good answer: "Net-revenue-retention on the Pro plan;
target +0.3pp absolute over the current baseline."
Weak answer: "Reduce churn."

**A3. What is the smallest change worth building
for?**
Good answer: "Anything less than a 0.2pp NRR lift is
not worth the outreach team's incremental capacity;
the model has to earn at least that."
Weak answer: "As much as we can get."

**A4. Who is on the paging rotation when the model
regresses?**
Good answer: "The ML on-call rotation; runbook link
attached to every alert."
Weak answer: "We'll figure that out at launch."

**A5. What is the target — unit of prediction,
horizon, and label definition?**
Good answer: "Unit is (subscriber, week); horizon is
30 days; label is `cancelled_at IS NOT NULL WITHIN
next 30 days` from the `subscriptions` table."
Weak answer: "Churn."

**A6. What decisions is the model explicitly
*not* making?**
Good answer: "Not deciding the discount value; not
handling Free or Enterprise plans; not personalising
the outreach content."
Weak answer: [Silence.]

**A7. What non-ML baseline ships first, and what
does the ML version have to beat it by?**
Good answer: "A rules-based list based on
usage_frequency and support_contacts; ML has to
improve Recall@800 by 5% absolute to justify the
launch."
Weak answer: "We're going straight to XGBoost."

**A8. What is the cold-start plan?**
Good answer: "Not applicable — the product has years
of history." *or* "First quarter is rules-based;
LLM zero-shot on labels for the second; fine-tuned
model in Q3."
Weak answer: "We'll see what data we have."

## Section B — Architecture (chapter 02)

**B1. Which of the nine components exists, and
where do they live?**
Good answer: All nine named with owners and
technologies. "Feature store: the warehouse plus a
DynamoDB online cache; owned by data eng."
Weak answer: A diagram with only "model" and
"database."

**B2. Batch, streaming, or online? Why?**
Good answer: "Batch. Retention team runs weekly;
nightly refresh is more than fresh enough; saves
~$40k/yr vs online."
Weak answer: "Online, because latency."

**B3. What is "the model" — the full contract?**
Good answer: "Artifact, featurizer, reference
profile, calibration cutoff, and registry version
label. All five are versioned together."
Weak answer: "The `.joblib` file."

**B4. What owns the predictions table?**
Good answer: "The warehouse; one row per prediction
keyed by `prediction_id`; retention 90 days;
lifecycle owned by data eng."
Weak answer: [Uncertain glance at data eng.]

**B5. What orchestrator schedules the pipelines, and
what monitors it?**
Good answer: "Airflow; DAG-run failure alerts to the
ML on-call channel; freshness SLO on the training
job."
Weak answer: "Cron on a jump box."

## Section C — Training-serving skew (chapter 03)

**C1. Is the feature transform code written once, or
in two places?**
Good answer: "Once, in a shared `features` package;
imported by the training pipeline and the serving
service."
Weak answer: "Reimplemented in Python for training
and Go for serving; we cross-check them
occasionally."

**C2. Are fitted preprocessors serialised with the
model artifact?**
Good answer: "sklearn Pipeline; the fitted
StandardScaler and OneHotEncoder are part of the
artifact; loaded together with the model."
Weak answer: "We refit at serving time."

**C3. What is the point-in-time semantics of every
time-window feature?**
Good answer: "Feature store's point-in-time-correct
offline join for training; online store's latest
value at serving; freshness SLO of 15 minutes."
Weak answer: "We take the last 30 days."

**C4. Is there a canonical fixture in CI that asserts
training-time and serving-time feature vectors
match byte-for-byte on the same input?**
Good answer: "Yes; 50 rows covering nulls and edge
cases; test blocks promotion on any drift."
Weak answer: "We have integration tests."

**C5. Does the rollout plan include a shadow-mode
step before real traffic?**
Good answer: "Three days of shadow-mode; per-request
feature parity checked on a sample."
Weak answer: "Ramp from 0 to 50% in a day."

**C6. Is the feature-anomaly counter alerting in
production?**
Good answer: "Yes; per-feature per-version; alert
threshold calibrated from a two-week historical
backfill."
Weak answer: "We monitor it on a dashboard."

## Section D — Feedback loops (chapter 04)

**D1. Does the model's action determine whether the
label is measurable?**
Good answer: "Yes for the outreach cohort; no for
the untreated hold-out cohort. We keep a 10%
hold-out as the training-label gold set."
Weak answer: [Blank stare.]

**D2. Where do training labels come from —
observation, randomised, or external?**
Good answer: "Passive observation on the treated
cohort; randomised control on the 10% hold-out;
retention-team review on flagged high-value
accounts."
Weak answer: "The warehouse."

**D3. What fraction of serving traffic is
exploration?**
Good answer for ranking / recommender: "5% of top-K
positions are Thompson-sampled from a wider
candidate set." Good answer for scoring systems:
"1-10% untreated cohort; no exploration in the
serving path itself."
Weak answer: "None."

**D4. What diversity metrics are on the dashboard?**
Good answer for ranking: "Catalogue coverage,
entropy of the recommendation distribution, and
new-item ramp time — same dashboard as CTR."
Weak answer: "Accuracy@K."

**D5. What is the periodic full-population retrain
cadence?**
Good answer: "Every four quarterly retrains, one
retrain uses the untreated cohort plus a
randomised sample from the last four quarters."
Weak answer: "We retrain every week."

**D6. For high-stakes decisions, is there a
human-in-the-loop review?**
Good answer: "10% of top-scored users are reviewed
before outreach by the retention team; review
outcomes are the label-gold set."
Weak answer: "Not applicable; the ML team owns
this."

## Section E — A/B testing (chapter 05)

**E1. What is the unit of randomisation, and does
interference threaten it?**
Good answer: "User (stable subscriber_id); no
interference because outreach is 1:1 with the
subscriber."
Weak answer: "We'll figure it out."

**E2. What is the primary metric, and does it move
within the test's duration?**
Good answer: "30-day-forward NRR on the treated
cohort; the horizon fits inside the 3-week test."
Weak answer: "AUC."

**E3. What are the guardrails, and what are their
stopping rules?**
Good answer: "p95 latency < 500ms — stop if
breached for 10 minutes; unsubscribe rate must not
regress by more than 5% relative — stop if
regressed on any daily read; per-slice regression
on any n>5000 cell — stop and investigate."
Weak answer: "We'll watch latency."

**E4. What is the power analysis?**
Good answer: "Baseline 92%, MDE 0.3pp absolute,
alpha 0.05, power 0.80, per-variant N = 120k with
CUPED; ~3 weeks at 50/50 given weekly eligible
volume."
Weak answer: "We'll run it for a couple weeks and
see."

**E5. What is the ramp plan?**
Good answer: "Shadow → 1% canary → 50/50 test →
read at 3 weeks → launch or roll back."
Weak answer: "50/50 from day one."

**E6. What is the pre-registered decision rule?**
Good answer: "Launch if primary metric improves by
≥0.3pp with p<0.05, no guardrail regression, no
slice regression per (E3). Ambiguous means no."
Weak answer: "We'll decide at read-out."

**E7. Is there an SRM check?**
Good answer: "Automated; part of the
experimentation platform; alert on the read-out
dashboard."
Weak answer: "What is SRM?"

## Section F — Build vs buy and delegation (chapter 06)

**F1. For every component in the reference
architecture, is the build-vs-buy question
answered?**
Good answer: Table with each component and a
recommendation; specific rows have "vendor X + a
thin adapter" or "in-house v0."
Weak answer: "We're building all of it."

**F2. Is this task in the shape of a specialist
peer track (LLM App, RAG, Fine-Tuning,
Generative Eval)?**
Good answer: "Task is tabular scoring; no
delegation." *or* "Task is customer-support
question-answering; delegated to the RAG team
with a joint eval harness."
Weak answer: "We considered LLMs but decided to
train our own text classifier."

**F3. For every buy, what is the total cost of
ownership, the exit cost, and the boundary
team's readiness?**
Good answer: Named in the comparison table; each
under review by the neighbouring team lead.
Weak answer: "The vendor is cheap."

**F4. For every build, what is the maintenance
plan?**
Good answer: "0.2x ML engineer FTE for ongoing
maintenance; documented handoff; runbook."
Weak answer: "We'll figure it out post-launch."

## Section G — Not-goals (revisited)

**G1. What is explicitly out of scope, and why?**
Good answer: A list. "This model does not
personalise the outreach content — handled
manually by retention. This model does not
handle Free or Enterprise plans — separate
economics, separate models. This model does not
address B2B accounts — feature availability is
too different."
Weak answer: [Silence.]

**G2. What would you build if the scope were
doubled?**
Good answer: "Add the B2B track; add outreach
content personalisation via the LLM app team;
add real-time bandit selection on offer
variants." (Interviewers ask this to test that
non-goals are actually decisions, not oversights.)

## The five-minute version

The full checklist is 40 questions. In practice, when
you have five minutes to sanity-check someone's
design, ask these five:

1. **What business metric does this move, and by
   how much?**
2. **What decision does the model *make* — and what
   is the untreated control that lets you know if
   you're right?**
3. **How do you know the features at serving time
   match the features at training time?**
4. **How will you A/B test this — unit of
   randomisation, primary metric, guardrails,
   stopping rule?**
5. **What's out of scope?**

The five questions land on the five hardest failure
modes: an unfocused problem (chapter 01), a
degenerate feedback loop (chapter 04), training-
serving skew (chapter 03), an unmeasured launch
(chapter 05), and scope creep (chapter 01/06).
Everything else in the checklist is a refinement
of one of the five.

## What the checklist is not

A checklist is a way to *avoid* being surprised. It
is not a substitute for judgment. Two failure modes
of checklists in general:

- **Ticking the box without earning the answer.**
  "Yes, we have a feature store." A vendor logo on
  a slide is not the same as a working
  point-in-time-correct offline join. The
  reviewer's job is to press until the answer is a
  claim about *behaviour*, not a claim about
  *tooling*.
- **Reading the checklist as the design.** A design
  that answers every question and adds nothing is
  a document, not a plan. The value of the design
  is the specific business context and the
  architectural choices that follow from it. The
  checklist is the floor, not the ceiling.

## The design review as a habit

The two moves that turn one-off reviews into a
practice:

- **Every ML system in production has a design
  document in the repo.** New joiners read it;
  quarterly reviews update it; incidents append to
  it. The doc is a living artifact.
- **Every promotion (chapter 03) or A/B launch
  (chapter 05) references the design doc.** If
  something is being built that the design doc
  doesn't cover, the design doc gets an update
  before the change ships. Not a bureaucracy — a
  memory.

Teams that do this find their oncall load drops. The
things the checklist would have caught don't become
incidents.

## Common pitfalls

1. **Using the checklist as an interview script.**
   Read it as a starting set of questions, not a
   script. Some designs sail through half of it;
   spend the time on the parts that matter.
2. **Treating unanswered questions as a fail.**
   Some questions genuinely have no answer at the
   design-review moment. The action is "we owe
   this by launch," not "reject the design."
3. **A design review that never gets written up.**
   The room decides the design is fine and nobody
   documents what changed. Six months later the
   arguments are lost.
4. **The reviewer has never worked on the
   underlying business.** Bringing a design review
   from the systems-design toolkit doesn't
   substitute for asking "is this the right
   problem to solve?" — that requires domain
   context.
5. **Delegation of the review to a committee.** A
   committee cannot ask sharp questions. A single
   experienced reviewer who has read the design
   doc can. The room can join.
6. **Reviewing the model but not the loop.**
   Chapter 04's loop analysis is easy to skip.
   Skip it and the model looks fine on paper;
   the loop erodes it in production.
7. **A checklist that lives in a wiki.** The
   design doc lives in the repo; the checklist
   lives with the design doc. Wikis rot; repos
   don't.

## Summary

- A design review is an audit of specific claims.
  This chapter's checklist is the list of claims.
- The list has 40+ questions across seven
  sections: business framing, architecture,
  skew, loops, A/B testing, build-vs-buy, and
  non-goals.
- The five-minute version is five questions.
  Memorise them.
- The checklist is a floor. A design that only
  passes the checklist is a document; a design
  that adds the business context and the specific
  architectural choices is a plan.
- Design reviews become habit-forming when every
  system has a design doc in the repo and every
  promotion or A/B launch references it.

## Where this goes next

The module ends here. Concretely, by the end of
mod-110 you can:

- Take a business goal to a written blueprint
  (chapter 01).
- Draw the reference architecture and defend the
  component boundaries (chapter 02).
- Prevent, detect, and debug training-serving skew
  (chapter 03).
- Diagnose and mitigate feedback loops (chapter
  04).
- Design and run a defensible A/B test on a live
  ML system (chapter 05).
- Make (and defend) build-vs-buy decisions and
  delegate to peer specialist tracks (chapter 06).
- Run a design review — as author or reviewer —
  from the checklist above (chapter 07).

That is the full arc of "ML systems design" as a
practice. The four exercises in this module — a
design-interview walkthrough, a skew investigation,
an A/B design, and a build-vs-buy exercise — are
where each of these skills becomes muscle memory.
The pointer to the paired *ml-engineer-learning*
solutions repo is in the module README.

The next role-appropriate step in the ecosystem is
sideways rather than upward: pick a specialist
track that matches your interest — LLM Application,
RAG, Fine-Tuning, or Evaluation — and pair with the
peers who own it. The systems-design vocabulary
from this module carries over directly.
