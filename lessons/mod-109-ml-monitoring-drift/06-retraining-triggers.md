# 06 — Retraining triggers

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 01–05.

## Motivation

You have drift scores (chapter 03), a prediction-shift
signal and possibly a performance estimate (chapter 04),
and — with a delay — a real accuracy metric from the
labels table (chapter 05). None of them individually says
"retrain now." Chapter 06 is the decision rule that
takes all of them and outputs one of:

- **Do nothing.** The signal is noise or the drift is
  benign.
- **Investigate.** The signal is real; a human should
  look at *why* before touching the model.
- **Retrain and re-evaluate.** The signal is real and
  the fix is (probably) a fresh model.
- **Roll back to a previous version.** The current
  model is worse than the previous one; retraining
  cannot happen faster than the rollback.

The failure mode this chapter prevents is **retraining
as a reflex**. A drift alert fires; a pipeline kicks off;
a model deploys; nobody looked at the data. Half the time
the new model is worse. The trigger design is what stops
that outcome.

## The trigger has three parts

Every retraining trigger — cron, event, or human-in-the-
loop — is three components:

1. **Signal.** A measurable thing that says "the model's
   world has changed." Drift score, prediction shift,
   estimated performance, measured performance, calendar
   date, or an ad-hoc human decision.
2. **Decision.** A rule that turns the signal into
   `{do-nothing, investigate, retrain, rollback}`.
   Includes debounce, hysteresis, and priority when
   multiple signals fire at once.
3. **Action.** The retraining pipeline: refresh the
   training data, retrain, evaluate against a stable
   holdout, promote or reject the candidate. Owned by
   `mod-106` (registry) and this chapter jointly.

Getting the signal right is chapter 03–05. Getting the
decision right is this chapter. The action is a pipeline
you inherit from `mod-106`; the trigger's job is to hand
it a well-scoped input.

## Trigger flavours

Three flavours cover almost every real system. Pick the
one that matches the model's cadence; combine when the
model warrants both.

### Scheduled (calendar-driven)

Retrain every N days regardless of signal. Simple,
predictable, and hard to argue with as a floor. Well
suited to:

- Models where features and labels come in cleanly on
  a cadence (weekly cohorts of business outcomes).
- Domains where the world moves slowly and consistently
  (macroeconomic factors, seasonality-heavy problems).
- The "does anyone actually notice this model exists"
  systems where scheduled retraining is a cheap
  insurance policy.

Downside: retraining when nothing needs it wastes
compute; retraining too rarely misses fast drift.

### Event-driven

A signal (drift, prediction shift, performance drop) fires
the trigger. Well suited to:

- Fast-moving domains (fraud, ad serving, content
  ranking).
- Cases where the labels arrive quickly enough that a
  performance-drop trigger is not a lagging indicator.
- Systems with enough monitoring infrastructure that a
  false-positive trigger is not too expensive.

Downside: chases noise if the decision rule is loose;
easy to build a self-reinforcing loop where retraining
on a drifted window makes future drift alerts more
likely.

### Human-in-the-loop

A signal fires; a human ML engineer reviews it; the
human triggers the retrain. Well suited to:

- Systems where each retrain is expensive to evaluate
  (large models, heavy A/B tests, regulated domains).
- Novel drift the automated rules have not seen before.
- Any system where the cost of a bad retrain (revenue,
  user trust, regulatory) exceeds the cost of a
  reviewer's time.

Downside: doesn't scale; blocks on the reviewer's
availability; the reviewer becomes a bottleneck.

Most mature systems layer the three: **scheduled** as a
floor, **event-driven** for fast changes, **human-in-
the-loop** for signals the rules aren't confident about.

## The decision rule — the actual logic

A decision rule is a small piece of code (or a
configuration file) that reads the current monitoring
state and outputs an action. A sketch:

```python
def decide_action(state: MonitoringState) -> Action:
    # Rollback wins over everything.
    if state.rolling_accuracy_delta < -0.05 \
       and state.previous_version_accuracy > state.current_accuracy:
        return Action.ROLLBACK

    # Fast, high-severity drift → investigate now.
    if state.multivariate_drift > 0.35 \
       or state.n_feature_alerts >= 3:
        return Action.INVESTIGATE

    # Sustained moderate drift over N windows → retrain.
    if state.consecutive_drift_windows >= 3 \
       and state.multivariate_drift > 0.20:
        return Action.RETRAIN

    # Measured accuracy drop below the reference band.
    if state.rolling_accuracy < state.reference_accuracy \
                              - 2 * state.reference_accuracy_std:
        return Action.RETRAIN

    # Scheduled floor.
    if state.days_since_last_retrain >= 30:
        return Action.RETRAIN

    return Action.DO_NOTHING
```

Real rules are longer and per-model. What matters is that
the rule is:

- **Written down.** Not carried in someone's head.
- **Reviewable.** In code, in the same repo as the
  monitoring job. Changes go through review.
- **Testable.** Unit tests for representative
  `MonitoringState` inputs. When the rule changes, the
  tests catch unintended regressions.
- **Auditable.** Every decision writes a row to a
  `retraining_decisions` table with the state that
  produced it. When someone asks "why did the model
  retrain last Tuesday?", the answer is a query.

## Debounce, hysteresis, cooldown

Three anti-noise patterns keep the trigger honest:

- **Debounce.** Require the signal to be present in N
  consecutive windows before firing. A single hour of
  drift is not a retrain; three consecutive days
  probably is. Chapter 03's window sizing sets what "N"
  means in wall-clock time.
- **Hysteresis.** The threshold to enter the alerting
  state is higher than the threshold to exit it. Prevents
  the trigger from flapping when the signal wobbles
  around a single threshold.
- **Cooldown.** After a retrain, no new retrain for M
  days (or until the next labelled cohort matures). Stops
  the trigger from firing on the *new* model's cold-start
  noise.

Without these, a busy monitoring system can retrain
every day and never converge.

## Guardrails — the "did the retrain make things
better?" gate

A retrain that produces a worse model must not deploy.
Every retrain that is a candidate for promotion goes
through the same evaluation gate as `mod-107` (evaluation)
and `mod-106` (registry):

1. **Backtest on a stable holdout.** A frozen, versioned
   holdout that predates the drifted window. Compare the
   candidate to the incumbent on it. Regression on the
   holdout blocks promotion.
2. **Backtest on the recent-window cohort.** Fairness to
   the drift: the whole point of retraining is to do
   better on the new distribution. Both models on the
   drifted window; the candidate should win.
3. **Slice checks.** Chapter 04's slice list. The
   candidate cannot regress any protected or high-value
   slice below a threshold, even if aggregate looks
   better.
4. **Shadow / canary.** Deploy the candidate at low
   traffic; compare its live prediction distribution
   and (delayed) accuracy against the incumbent's before
   promoting.
5. **Kill switch.** A rollback path that the platform
   team's serving pipeline supports. `mod-108` chapter
   11's model-version-in-response makes this
   attributable.

The gate is not optional. A retrain that fails any of
these steps is *not* promoted; it is a labelled artifact
in the registry, the trigger records `retrain_result =
rejected`, and a human investigates.

## The specific signals — how each turns into a decision

A short catalogue of the most common signals and what to
do with each. Each row assumes the debounce, hysteresis,
and cooldown machinery above.

### Univariate feature drift (chapter 03)

- **A single feature above threshold, sustained.** Almost
  never a retrain. Usually an upstream data change; page
  the data engineering owner. Chapter 07 walks a
  concrete case.
- **Multiple features above threshold simultaneously.**
  Investigate; if root cause is a real distributional
  shift (not a data-quality bug), consider retraining.

### Multivariate drift (chapter 03)

- **Above threshold with no univariate offenders.** A
  correlation structure changed. Uncommon but important
  — usually reflects a real shift the model isn't
  tuned for. Investigate; likely retrain.
- **Above threshold with univariate offenders too.**
  Redundant confirmation of the univariate story;
  respond as for univariate.

### Prediction-distribution shift (chapter 04)

- **Small shift, features stable, accuracy stable.**
  Noise; ignore.
- **Large shift, features stable, accuracy stable.**
  A new segment of traffic is legitimate but different.
  Do nothing to the model; consider a segmented deploy
  (a per-segment model) as a longer-term action.
- **Large shift, features stable, accuracy dropping.**
  Concept drift. Retrain, with sampling that includes
  the new segment.
- **Large shift, features drifting, accuracy unknown.**
  Data drift causing prediction drift; may or may not
  be a real problem. Wait for labels before retraining;
  the shift alone is not enough.

### Measured accuracy regression (chapter 05)

- **Aggregate accuracy drops below the calibrated band.**
  Retrain, provided the guardrails clear.
- **Slice accuracy drops but aggregate holds.** The
  specific slice matters. Retrain if the slice is a
  business priority; consider segmented modelling if
  the slice is small but valuable.
- **Sudden discontinuity in accuracy on a specific
  date.** Something changed on that date. Investigate
  before retraining — the fix might be a data pipeline
  rollback, not a new model.

### Time since last retrain (scheduled)

- **N days since last retrain.** Retrain as scheduled.
  The candidate goes through the same guardrail gate as
  event-driven retrains; a scheduled retrain that fails
  the gate is still not promoted.

### Upstream data event

- **New feature added to the feature store.** Not a
  retrain trigger by itself; a *training-run* trigger
  once the feature has been backfilled to the training
  history.
- **Deprecated feature removed.** Retrain becomes
  necessary before the feature is fully removed.
  Coordinate with `mod-103` on the timeline.

## Sampling for the retrain — the trap door

The retrain data itself has to be picked. This is the
step where "retrain on the drifted window" makes things
worse. Two rules:

1. **Retrain data spans a window long enough to include
   pre-drift and post-drift.** A retrain that trains
   only on the last 7 days of a drifting distribution
   forgets everything the previous 6 months knew.
2. **Stratify by cohort.** The training set should still
   look like the production distribution. Over-sampling
   the newly-drifted segment is a modelling choice,
   not a default.

For fast-moving domains a longer window with a
recency-weighted loss is often better than a shorter
window. `sample_weight` in scikit-learn / LightGBM /
XGBoost is enough to implement it.

## The retraining pipeline — what the trigger hands off

The trigger's output is not a model; it is a well-scoped
input to a training run. Every trigger should populate:

- **`training_data_window`** — start and end dates.
- **`reference_holdout_id`** — the frozen holdout to
  compare against.
- **`incumbent_model_version`** — what to compare the
  candidate to.
- **`trigger_reason`** — human-readable + machine-
  readable ("drift.multivariate=0.31", "scheduled",
  "accuracy_drop.slice=enterprise").
- **`decision_rule_version`** — the git SHA of the
  rule that produced this trigger.
- **`request_id`** — for tracing this specific retrain.

The training pipeline (`mod-106`) consumes these,
produces a candidate, runs the evaluation gate, and
either promotes or rejects. Every step is logged; every
model has a lineage back to the trigger.

## The retraining loop can go wrong

Three failure modes worth naming out loud, because they
show up when the automation is *working* and the
diagnosis is subtle.

### The self-reinforcing bad-retrain loop

The model retrains on a drifted window; the new model's
predictions are worse; the labels reflect that; the
next drift check fires again; the pipeline retrains again
on the drifted-plus-new-worse-predictions window;
repeat. The evaluation gate is what stops this — but only
if the gate compares the candidate to a stable, pre-drift
baseline. If the baseline is also a recent model, both
are drifting; the gate becomes useless.

### The "silent" retraining regression

Every retrain slightly regresses on a small slice. Over
many retrains the slice's accuracy has degraded 20%.
The aggregate metric is fine. Nobody noticed. Fix: slice
metrics in the evaluation gate, not just aggregate.

### The "we retrained but nothing happened" case

The trigger fires; the model retrains; the new model is
promoted; the drift alert is still firing a week later.
This usually means the drift was in a feature the model
uses only weakly, or the drift is real but the labels for
the drifted segment are also drifted (labeler drift, per
chapter 05). Investigate — the retrain wasn't the
right fix.

## What the retraining decision *isn't*

Two things that are not the trigger's job:

- **Deciding whether to change the model architecture.**
  If the model is fundamentally the wrong shape (linear
  when non-linear is needed, single-model when
  segmented is better), that is a modelling project,
  not a retraining trigger. The trigger's job is
  incremental refresh.
- **Deciding to add a new feature.** Adding a feature
  changes the training-set schema, the serving-time
  schema, and the reference profile. It is a
  well-scoped project (`mod-103` + `mod-106` +
  `mod-108`), not a trigger event.

Keep the trigger's action narrow. Anything broader gets
scoped out to a normal project.

## Common pitfalls

1. **Retraining on drift alone.** Retraining without
   corroborating accuracy drop or slice signal can make
   things worse.
2. **No cooldown.** Trigger fires every window; every
   retrain slightly changes the prediction distribution;
   next window triggers again. Circular.
3. **Evaluation gate uses the recent window as the
   baseline.** Both incumbent and candidate look
   equally bad; gate passes; regressions ship. Baseline
   is a stable pre-drift holdout.
4. **No slice metrics in the gate.** Aggregate wins;
   slices lose; nobody notices for weeks.
5. **The trigger fires; nobody knows why.** Every
   decision writes an audit row (state + reason +
   version). Debugging a bad retrain has to be a
   query, not archaeology.
6. **Trigger changes without tests.** A one-line
   threshold change silently doubles the retrain rate;
   compute costs go up; nobody notices until the bill.
7. **A retrain that doesn't go through the registry.**
   `mod-106`'s registry is the system of record. A
   trigger that side-loads a `.pkl` into the serving
   directory is a rollback nightmare.
8. **No rollback plan.** A bad candidate ships; the
   trigger has no `ROLLBACK` action; the fix is a
   panic release. Rollback is a first-class action.
9. **Human-in-the-loop trigger that never gets
   reviewed.** The queue fills; alerts age; the trigger
   becomes silent. Either the review has an SLA or the
   trigger is not "human-in-the-loop".
10. **Assuming the world will keep drifting the way it
    just drifted.** A retrain optimised for last week
    is not automatically the right answer for next
    week. Continue monitoring the *new* model as
    aggressively as the old one.

## Summary

- A trigger is (signal, decision, action). Getting the
  signal right is chapters 03–05; getting the decision
  right is a small, tested, auditable rule.
- Three flavours — scheduled, event-driven, human-in-
  the-loop — layer well. Scheduled as a floor;
  event-driven for fast signals; human review for
  ambiguous cases.
- Debounce, hysteresis, and cooldown are non-negotiable
  anti-noise machinery.
- Every retrain passes an evaluation gate: stable
  holdout, recent-window cohort, slice checks,
  shadow/canary. A regression on any of them blocks
  promotion.
- Sampling for the retrain matters at least as much as
  the trigger itself. Long windows, stratified,
  recency-weighted when appropriate.
- Rollback is a first-class action. The trigger
  supports it explicitly.

## Where this goes next

Chapter 07 is the end-to-end debugging walkthrough:
alert fires → what do you look at first → how do you
attribute cause → what mitigation ships → what the
post-mortem looks like. Everything in chapters 01–06
becomes a step in that walkthrough.
