# 07 — Debugging a production regression end-to-end

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01–06.

## Motivation

Chapters 02–06 built the instrumentation, the detectors,
the labels loop, and the retraining trigger. This chapter
is the *use* of all of it: **an alert fires; what does the
next hour look like?**

The chapter walks a canonical production regression from
alert to post-mortem. The specific numbers and the exact
root cause are illustrative; the *procedure* is what to
carry forward. A drift alert can mean a dozen different
things, and the sequence of questions that separate them
is the difference between an hour of clean work and a
week of finger-pointing.

## The oncall mindset

Before the runbook, three habits that separate an
effective ML on-call from a stressful one:

1. **Assume nothing about the model until you can name
   an evidence trail.** "The model regressed" is a
   claim, not a diagnosis. What signal, over what
   window, compared to what baseline?
2. **Follow the data upstream first, the model second.**
   Most regressions start with a data change, not a
   model change. Chapters 03 and 05 both make this
   the default hypothesis.
3. **Every action taken during incident response
   generates a log line.** The oncall's memory is
   unreliable at 03:14. The audit trail is not.

The chapter's walkthrough embodies all three.

## The alert — what fires and what it means

At 09:12 on a Tuesday, the drift job (chapter 03) posts
this alert:

```
[SEV-3] churn-model :: multivariate drift score 0.34
(threshold 0.20) — sustained over 3 windows
Feature drift alerts: plan_tier (PSI 0.28),
  months_active (PSI 0.19), country (PSI 0.14)
Prediction distribution shift (JSD): 0.11 (threshold 0.05)
Reference profile: rp_2026_05_10  Version: churn-1.4.2
Runbook: <link to this chapter's checklist>
```

Two things this alert tells you *before you touch a
dashboard*:

- **The drift is real** (three consecutive windows;
  chapter 06's debounce), and
- **It correlates with prediction-distribution shift.**
  Chapters 03 + 04 firing together is stronger evidence
  than either alone.

Two things it does *not* tell you:

- **What broke.** Drift is a symptom.
- **Whether the model is actually wrong yet.** No
  accuracy alert has fired; labels for this window will
  not mature for weeks.

The first ten minutes are about turning the alert into a
short list of specific hypotheses.

## Step 1 — Confirm the signal is real

Before triaging root cause, rule out infrastructure
issues that could produce a false drift alert:

- **Is the drift job's clock aligned?** A stale
  reference profile ID, a window boundary that shifted,
  a job that ran twice on the same window.
- **Is the prediction table complete for the alerted
  window?** A missing partition can distort the
  histogram.
- **Are all model versions accounted for?** A rolling
  deploy in progress might have half the window served
  by v1.4.2 and half by v1.4.3 — the mixed distribution
  will drift trivially against a v1.4.2-only reference.
- **Is the reference profile correct?** Sanity-check
  the mean and quantiles against your memory of the
  training set; a corrupted or overwritten reference
  produces phantom drift.

If the answer to any of these is "no", the alert is a
tooling bug; fix it and re-evaluate. If everything checks
out, proceed.

Time budget for this step: 5–10 minutes. If it takes
longer, your monitoring metadata is not clear enough —
that's a chapter 02 follow-up, not an incident action.

## Step 2 — Localise the drift

The alert named three drifting features. Look at each in
order of PSI, from highest to lowest.

For **`plan_tier`** (PSI 0.28):

1. Open the feature's histogram over the last two
   weeks. Draw the reference distribution as a horizontal
   band or overlay.
2. What changed? A new category appearing (`enterprise-
   plus` was 0% and is now 12%)? An existing category
   growing (`pro` doubled)? The whole distribution
   shifted?
3. When did it change? A sharp discontinuity on a
   specific day is a code change or a data-pipeline
   event. A gradual slope over two weeks is a real
   shift in traffic.

The pattern here matters. In the walkthrough, `plan_tier`
shows a sharp discontinuity on 2026-05-28 — a new value
`enterprise-plus` appears where nothing was before.
That's a schema change upstream.

Repeat for the other flagged features. In the walkthrough:

- `months_active` (PSI 0.19) — the whole distribution
  shifted slightly toward higher values, gradual over
  two weeks. Real.
- `country` (PSI 0.14) — a specific country's share
  doubled. Real (or a bot-traffic surge; step 3 will
  disambiguate).

Now you have a shape: one hard schema change, two
softer distributional changes.

## Step 3 — Follow the data upstream

The hard schema change is the strongest lead. Follow it:

1. **What upstream table populates `plan_tier`?** In
   most systems, the feature store's lineage
   metadata answers this in one query. If you don't
   have a feature store, `mod-103`'s data catalog
   or a grep of the feature-transform code does.
2. **Did that table change?** Compare the upstream
   table's schema and value distributions across the
   drift boundary. In the walkthrough, the
   `subscriptions` table added an `enterprise-plus`
   plan on 2026-05-28 as part of a product launch. The
   feature-transform code mapped unknown categories to
   `unknown_category` (chapter 02's counter fired
   quietly), the model treated them as `plan_tier=unknown`,
   and the predicted-churn probability collapsed to
   the model's `unknown_category` mean.
3. **Was anyone told?** If the launch was on a change
   log, checking it will save the hour it would have
   taken to work this out from the data. Chapter 06's
   "add a feature is a project" flag is what prevents
   this in the future.

For the softer shifts (`months_active`, `country`):

1. **Was there a marketing campaign, a new region
   launch, or a partner integration?** These show up
   as gradual distributional shifts, and they're
   *real* — the model just hasn't been retrained to
   handle them well.
2. **Does the feature-anomaly counter (chapter 02)
   have anything to say?** A rising null rate on the
   country feature would suggest a data-collection
   bug, not real traffic.

At this point, one hour into the incident, you have a
diagnosis: **the `enterprise-plus` value is being served
as `unknown_category`, artificially depressing predicted
churn for those users; two softer drifts are legitimate
distribution changes that the model will need retraining
for eventually.**

## Step 4 — Decide the action

Chapter 06's decision rule is your reference. Apply it:

- **`plan_tier` schema-change issue.** Not a retrain
  target *yet* — retraining without adding
  `enterprise-plus` to the training data would embed
  the bug. The immediate action is a **data-pipeline
  fix**: add `enterprise-plus` to the feature schema,
  backfill the last N days of predictions with the
  correct mapping (best-effort), and page the
  data-engineering owner. The model gets retrained
  once the feature is correctly available and the
  training set has been backfilled.
- **`months_active` and `country` shifts.** These are
  real distributional changes; the model will
  eventually need retraining for them, but no
  measured accuracy regression has fired. The
  decision rule's cool-off (chapter 06) keeps this on
  the *next scheduled retrain*, not an emergency
  retrain. Log it in the retraining triggers table
  with `trigger_reason = "drift.plan_tier + drift.
  months_active + drift.country — deferred to
  scheduled"`.

Two decisions, two audit rows. No emergency retrain.

## Step 5 — Mitigate quickly

Between "root cause identified" and "root cause fixed"
can be days. In the meantime, mitigate the user-facing
impact:

- **Add a serving-time guardrail.** For any request
  where `plan_tier == "enterprise-plus"`, return the
  reference-average prediction for the `pro` cohort
  (the closest existing category) with a warning
  header. Better than the collapsed `unknown` prior.
- **Alert the platform team of the mitigation.** They
  should know the service now has a non-model
  fallback path so it doesn't get attributed to a
  serving bug later.
- **Instrument the mitigation.** Emit a counter for
  "requests served by the guardrail" and monitor it
  until the underlying fix ships.

Mitigation is temporary tech debt; open the ticket to
remove it in the same commit.

## Step 6 — Communicate

Two audiences:

- **The oncall channel** — a short structured update:
  what fired, root cause, mitigation in place, next
  steps, ETA. Update on progress.
- **Downstream consumers** — the product team using
  the model. "We know the model is predicting low
  churn for `enterprise-plus` users; we've deployed a
  fallback; retraining is scheduled for next week."
  Short, honest, dated.

The single most common failure mode of an incident is
silence between step 3 and the post-mortem. A weekly
model dashboard is not communication; a message in the
channel is.

## Step 7 — Follow the fix through

The root cause was upstream. The data-engineering owner
adds `enterprise-plus` to the feature store, backfills
the transform. That is not the end.

- **Verify the feature-anomaly counter drops to zero.**
  Chapter 02's `unknown_category` counter for
  `plan_tier` should return to baseline.
- **Verify the drift score drops below threshold.**
  Chapter 03's next window should show `plan_tier` PSI
  under threshold.
- **Verify predicted churn distribution recovers.**
  Chapter 04's prediction histogram should recover the
  pre-drift shape (minus the legitimate distribution
  changes from `months_active` / `country`).
- **Only then remove the serving guardrail.** In its
  own commit. The commit message references the
  incident ticket.

If any of these don't happen, the fix wasn't the fix.
Investigate again.

## Step 8 — The post-mortem

A short template. The point is not exhaustive prose; the
point is a durable artifact the next oncall can learn
from.

```
# Post-mortem — churn-model prediction regression (2026-05-28)

## TL;DR
An upstream schema change (`subscriptions.plan` added
`enterprise-plus`) was not propagated to the feature
transform, so the churn model saw those users as
`plan_tier=unknown_category`. Predicted churn collapsed
for that cohort. Detected by the multivariate drift
alert on 2026-06-08 (11 days after the introduction).
Mitigated with a serving-time fallback on 2026-06-08;
schema fix shipped on 2026-06-10; guardrail removed on
2026-06-11.

## Timeline (UTC)
2026-05-28 14:03  Upstream launch enables `enterprise-plus`.
2026-05-28 14:07  First requests with the new value; served as unknown.
                  Feature-anomaly counter starts ticking; not yet alerting.
2026-06-08 09:12  Multivariate drift alert fires (3 windows sustained).
2026-06-08 09:22  Oncall confirms alert is real (step 1 checklist).
2026-06-08 09:45  Root cause identified via feature drilldown.
2026-06-08 10:15  Serving-time guardrail deployed.
2026-06-08 10:30  Data-eng owner paged; joint work begins.
2026-06-10 16:00  Schema fix + backfill shipped.
2026-06-11 09:00  Metrics recovered; guardrail removed.

## What went well
- Multivariate drift alert fired reliably; debounce
  prevented false alerts in the previous week.
- Root cause was identified in ~30 min via chapter 07's
  playbook.
- Serving guardrail limited user impact.

## What went poorly
- 11-day gap between schema change and detection.
  Feature-anomaly counter was ticking the whole time
  but was not on any alerting rule.
- No cross-team change notification for the upstream
  launch. Data eng, product, and ML did not have a
  shared changelog.

## Action items
1. Add an alert on `ml_feature_anomaly_total` rate per
   feature per version. Owner: <you>. Due: 1 week.
2. Add a lineage check to the drift job that fails
   loudly if the upstream schema has changed since the
   reference profile was written. Owner: data eng.
   Due: 2 weeks.
3. Add feature-schema changes to the launch-checklist
   for product releases. Owner: PM. Due: next launch.
4. Retraining ticket: retrain `churn-model` on the
   drift-adjusted dataset once `enterprise-plus` has
   30 days of history. Owner: <you>. Due: 2026-07-10.

## Lessons for future oncalls
- The feature-anomaly counter is a *leading* indicator
  and should be on the primary alerting surface, not
  just a dashboard widget.
- "Retrain now" is rarely the right response to a
  schema-change regression. Fix the schema first;
  retrain second.
```

The action items are the *only* mandatory outcome. Prose
without a list of dated, owned tasks is not a
post-mortem.

## A gallery of common regressions and the shortest
path to root cause

The walkthrough above is a specific case. Six other
common shapes, and where to look first:

1. **Upstream schema change (this chapter's walkthrough).**
   Feature-anomaly counter + univariate drift on one
   feature. Fix data pipeline.
2. **A new segment of users.** Univariate drift on
   `country` / `cohort` / `plan_tier` without any
   anomaly counters. Traffic is real; retrain if the
   segment is important, or add a segmented model.
3. **A model rollout regression.** Prediction shift
   correlated with `model_version` label; no data
   drift. Compare v_new vs v_old on the same requests
   (chapter 06's shadow / canary). Roll back.
4. **A latent bug in a preprocessing transform.**
   Prediction distribution collapses to a specific
   value ("everyone is 0.5"). Feature histograms look
   fine because the bug is in the transform. Look at
   the model's raw input vector, not the raw feature
   value.
5. **Delayed labels arriving from a different source.**
   Sudden step change in the accuracy line. Not the
   model — the labels table has a new upstream. Check
   `label_source` distribution over time.
6. **A retrain that made things worse.** Accuracy
   metric drops after a promotion; drift metrics are
   silent; only the `model_version` label separates
   the good period from the bad. Chapter 06's rollback
   action.
7. **Labeler drift.** Accuracy drops for human-labelled
   data but implicit-feedback labels look stable.
   Chapter 05's gold-set replay.

Each is a shape you'll recognise on second exposure. The
chapter 07 procedure is the same for all of them.

## The debugging query catalogue

A running list of jq / SQL queries worth having in your
head or your snippets file. Small investment; large
payoff on the day of the incident.

```sql
-- Feature-anomaly rate over the last 24h, per feature and kind
SELECT feature_name, kind,
       COUNT(*) / 24.0 AS per_hour_rate
FROM ml_events
WHERE event = 'feature_anomaly'
  AND ts >= NOW() - INTERVAL '24 hours'
GROUP BY feature_name, kind
ORDER BY per_hour_rate DESC;

-- Prediction distribution by model version over the last 7 days
SELECT model_version,
       DATE_TRUNC('hour', ts) AS hour,
       PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY prediction) AS p50,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY prediction) AS p95
FROM predictions
WHERE ts >= NOW() - INTERVAL '7 days'
GROUP BY model_version, hour
ORDER BY hour;

-- Slice accuracy for the last matured cohort
SELECT slice_metadata->>'country' AS country,
       AVG((predicted_class = label)::int) AS accuracy,
       COUNT(*) AS n
FROM predictions_labeled
WHERE label_at >= NOW() - INTERVAL '14 days'
  AND predicted_at <= NOW() - INTERVAL '30 days'  -- maturity
GROUP BY country
HAVING COUNT(*) > 100
ORDER BY accuracy;

-- Recent retraining decisions with reasons
SELECT decided_at, action, trigger_reason,
       (state->>'multivariate_drift')::float AS mv_drift,
       (state->>'rolling_accuracy')::float   AS acc
FROM retraining_decisions
ORDER BY decided_at DESC
LIMIT 20;
```

```bash
# One prediction, everything the service said about it:
kubectl logs deploy/churn -n serving \
  | jq 'select(.prediction_id == "pred_01HGX...")'

# Feature-anomaly log lines, last hour, top offenders:
kubectl logs deploy/churn -n serving --since=1h \
  | jq -c 'select(.event == "feature_anomaly") | .feature_name' \
  | sort | uniq -c | sort -rn | head
```

None of these are specific to any monitoring library.
All of them work against the shape chapters 02–05
already built.

## What you owe the *next* oncall

The debugging story is durable only if you leave the
system easier to debug than you found it. Concretely:

- **A runbook link on every alert.** The alert message
  points at a specific playbook page; the playbook page
  points at this chapter's procedure with the specific
  service's queries filled in.
- **A dashboard the alert deep-links to.** One click
  from the alert to the "everything you need for this
  service" board.
- **A `NOTES.md` in the model's repo** with lessons
  from previous incidents. Not a wiki. Not a Slack
  thread. In the repo, next to the code, so the next
  ML engineer inherits it.
- **A quarterly game-day.** Simulate a regression (a
  feature-column swap, a stale reference profile); walk
  the on-call through it. Cheap; catches everything
  that atrophied since the last real incident.

## Common pitfalls

1. **Retraining as the first response.** Every incident
   in this chapter has a data or schema fix upstream of
   a retrain. Retraining on a broken feature embeds
   the break.
2. **Reasoning about the model in isolation.** The
   service, the feature pipeline, the upstream tables,
   and the labels loop are one system. Drill upstream.
3. **Not distinguishing "drift is real" from "the
   drift is a bug".** Both fire the same alert;
   different remedies.
4. **Skipping step 1** (confirm the signal). Chasing a
   tooling artifact for two hours is a common way to
   waste an oncall shift.
5. **No mitigation while the fix is in flight.** Users
   see the regression for days. The serving-time
   guardrail is temporary; it also matters.
6. **No post-mortem action items.** The incident
   repeats in six months because nobody added the
   alert that would have caught it.
7. **Prose post-mortem with no owner or due date.**
   Actions don't happen.
8. **Blaming the model when the labels are
   drifting.** Chapter 05's labeler-drift check saves
   this from being an unbounded investigation.
9. **Silence.** Everyone in the org hears about the
   incident from Slack rumours instead of from you.
10. **No rehearsal.** The first time you run through
    the procedure is not during a real incident.

## Summary

- Alerts are hypotheses. The eight-step procedure —
  confirm, localise, trace upstream, decide, mitigate,
  communicate, follow through, post-mortem — turns a
  hypothesis into a fix.
- Most regressions are upstream data changes, not
  model degradation. The feature-anomaly counter and
  the univariate drilldown are your first stops.
- Retraining is not the first response. Fix the data
  first; retrain second; measure the recovery.
- Post-mortems are lists of dated, owned action items.
  Everything else in them is context.
- The system gets more debuggable one incident at a
  time. What you learn goes into the runbook, the
  dashboard, and the repo — for the next oncall.

## Where this goes next

`mod-110` (ML systems design) is the whole picture: the
service from `mod-108`, the monitoring surface from this
module, the feature store, the training pipeline, the
labels loop, the model registry, and the rollout
tooling, composed into a single system you can draw on a
whiteboard and defend at a design review. This module
ends here; the next one puts every piece of it back
into the larger picture.
