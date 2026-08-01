# 04 — Concept drift and prediction-distribution shift

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 01, 02, 03.

## Motivation

Chapter 03's drift job is quiet. Every feature's PSI is
under threshold. The multivariate score is stable. The
model still gets worse week over week.

This happens when the inputs look the same but the
*relationship* between inputs and outputs has changed —
**concept drift**. Fraud rings adapt to a detector;
users' preferences shift after a product change; a
regulatory change moves what "risky" means. `P(X)` is
unchanged, `P(y|X)` has moved, and none of chapter 03's
detectors can see it because they never look at `y`.

Concept drift is the hard case. Confirming it requires
ground truth (chapter 05); ground truth is often delayed
by days or weeks. This chapter is about **what to do in
the meantime**: read prediction-distribution shift as a
proxy, run performance-estimation methods that operate
without labels, and set up the monitoring so that when
labels finally arrive, the picture snaps into focus
instead of prompting a second investigation.

## Three axes of shift, one more time

Because they are easy to conflate, and because chapter 07
depends on you being able to name them mid-incident:

- **Data drift** (`P(X)`) — chapter 03. Measurable
  without labels. Common. First to fire.
- **Prediction-distribution shift** (`P(ŷ)`) — this
  chapter. Measurable without labels. Downstream of
  data drift, but also downstream of concept drift you
  cannot yet confirm.
- **Concept drift** (`P(y|X)`) — this chapter. Requires
  labels. The "expensive" signal. Slowest to fire.

The trick is that they usually **do not fire together**.
A model can regress in accuracy with no univariate feature
drift; a model can look drifted on every feature and still
be perfectly accurate (the shift is inside the model's
robustness envelope). The three signals are correlated,
not equivalent. Alert on each; make decisions on all
three.

## Prediction-distribution shift — the free proxy signal

Chapter 02 emitted `ml_prediction_score` as a histogram.
That histogram is a probability mass function over the
prediction range. Comparing it to the reference profile's
prediction distribution using the same tests as chapter
03 (PSI, JSD, KS) gives you a scalar drift score on
`P(ŷ)`.

Why it is worth watching independently:

- **It responds to both data drift and concept drift.**
  Either can move `P(ŷ)`. That's a feature: you get a
  single downstream signal to alert on.
- **It's the only ML-native quality signal that is
  never delayed.** You have it in real time; you never
  wait on a label.
- **It's an interpretable business signal.** "The model
  is predicting churn 2× as often this week" is a
  sentence a product manager understands. `PSI = 0.24`
  is not.

Two useful angles on the same histogram:

1. **Distribution distance** — PSI or JSD between the
   live histogram and the reference. Fast, symmetric,
   comparable across model versions.
2. **Predicted class ratio** — for classifiers, the
   fraction of predictions above a decision threshold.
   Concise; often the single number product cares about.

Both should be on the dashboard. Both should have
calibrated thresholds (same procedure as chapter 03: back
fill on historical windows; set the threshold above
"normal" wobble).

### Prediction shift is a symptom, not a cause

The failure mode: you alert on prediction shift, retrain
on the drifted window, and make the model *worse* because
the shift was real. Prediction shift tells you
*something* changed; it does not tell you what to do.
Pair it with:

- Per-feature drift (chapter 03) — to attribute the
  shift to specific upstream changes.
- A ground-truth-joined accuracy metric (chapter 05) —
  to know whether the model is still right.
- Business KPIs — the model's ultimate customer.

Retraining on prediction shift alone is chapter 06's
first pitfall.

## Concept drift — the shape of the problem

Concept drift comes in flavours that matter for
detection:

- **Sudden / abrupt.** A cutover event: a regulatory
  change, a new fraud vector, a product change. `P(y|X)`
  moves in a day. Detectable within a few days of
  labels arriving.
- **Gradual.** `P(y|X)` slowly evolves. User preferences,
  slowly shifting seasonality. Weeks to detect; visible
  as a downward trend on rolling accuracy.
- **Incremental.** Slow, monotonic degradation as the
  environment moves. Similar to gradual in practice.
- **Recurring / seasonal.** `P(y|X)` cycles. Holiday
  shopping, tax season, weekday vs. weekend. Handled by
  training-set design more than by monitoring.

Detecting concept drift is fundamentally a question about
`y` and `ŷ`. Once you have labels (chapter 5), the
detection reduces to:

- Rolling model performance (accuracy, AUC, log loss,
  MAE, MAPE) over a sliding window vs. a stable baseline
  — a CUSUM- or Page-Hinkley-style change detector on
  the metric works.
- Per-slice performance (top segments, protected
  groups, top-value customers) — often the aggregate is
  fine while a specific slice regresses.

Concept drift detection in the online-learning literature
(ADWIN, DDM, EDDM, KSWIN, HDDM) tends to assume labels
arrive with each new sample. In production ML the more
common shape is batched, delayed labels — a nightly join,
a weekly cohort. The change-detection algebra is the
same; the cadence is different.

<!-- needs-research: verify that these algorithm names (ADWIN, DDM, EDDM, KSWIN, HDDM) still map to widely referenced implementations (e.g., River / scikit-multiflow) and cite the survey paper (Gama et al., 2014 "A survey on concept drift adaptation") for grounding. -->

## Performance estimation without labels

When labels are delayed by days or weeks, you still want
*some* estimate of "how is the model doing now?". A small
family of techniques estimates model performance directly
from the live prediction stream, without waiting for
`y`:

- **Confidence-Based Performance Estimation (CBPE).**
  For classifiers whose predicted probabilities are
  well-calibrated on the reference data, the expected
  accuracy on a new window is a function of the
  predicted probabilities themselves. Well suited to
  well-calibrated tree ensembles and logistic-regression
  variants. Sensitive to prediction drift, of course —
  its role is to translate `P(ŷ)` shift into an
  estimated accuracy delta.
- **Direct Loss Estimation (DLE).** For regression,
  a secondary model predicts the loss from the primary
  model's features and prediction; the loss estimator
  is trained on the labeled reference set. Applied to
  the live stream, it gives a running estimate of loss.
- **Reverse Classification Accuracy / density-ratio
  methods.** Weight the reference-set losses by the
  ratio of live-to-reference density; produces an
  importance-weighted performance estimate. Reference:
  research literature on importance-weighted risk
  estimation; NannyML's `PerformanceEstimator` is a
  practical implementation.

<!-- needs-research: cite the NannyML documentation directly for CBPE / DLE and, if possible, the underlying papers. Also flag whether these methods are still considered SOTA as of the current curriculum revision. -->

The catch with all of them: they only work while the
assumption they rest on holds. CBPE assumes calibration
stays valid; DLE assumes the loss-estimator generalises.
Both are invalidated by strong concept drift (which is
often when you most need them). Treat their output as an
educated guess, not measured accuracy, and always
recompute measured accuracy the moment labels arrive.

## Segment monitoring — the aggregate hides regressions

The single most common way concept drift gets missed:
overall accuracy is stable because most requests are
easy; a small, valuable slice regressed hard; nobody
notices.

Two habits fix this:

1. **Slice the monitoring dashboard.** Prediction
   distribution and (when labels arrive) accuracy,
   broken out by:
   - Cohort — new users vs. existing.
   - Geography — country, region.
   - Traffic source — organic vs. paid, iOS vs.
     Android vs. web.
   - Volume tier — top-value customers vs. long tail.
   - Time of day / day of week.
2. **Alert on slice regressions, not just aggregate.**
   A 5% aggregate accuracy drop and a 20% drop on the
   enterprise-customer slice are two different
   conversations.

The slices should not be dozens; pick the three or four
that matter to the business. A per-country line chart
with 40 lines is nobody's dashboard.

## The dashboards that make concept drift visible

Extend chapter 02's Grafana boards:

- **Prediction rate by class over time, with the
  reference class ratio as a horizontal line.** The
  moment the live ratio pulls away from the reference,
  the eye catches it.
- **Predicted-score histogram as a heatmap over time.**
  Same as chapter 02; concept drift shows as slow drift
  of the modes, sometimes with a second mode appearing.
- **Rolling accuracy per slice.** One line per slice;
  bold line for aggregate; reference-window accuracy as
  a horizontal band.
- **Estimated-vs-measured accuracy.** Two lines: the
  performance-estimation output (real time) and the
  measured accuracy (lagged). Divergence between them
  is itself a signal that CBPE/DLE assumptions have
  broken.

## Change detection on the metric

Once accuracy (or loss, or estimated accuracy) is a time
series, alerting on it becomes a standard change-detection
problem. Three approaches, roughly in order of
sophistication:

1. **Threshold on a rolling mean.** "7-day accuracy < X".
   Blunt but effective; the threshold is a business
   number.
2. **Statistical process control (SPC).** Track the
   metric with a control chart (mean ± k·σ); alert on
   points outside the band. Handles heteroscedasticity
   badly but is simple.
3. **CUSUM / Page-Hinkley.** Accumulate deviations from
   a baseline; alert when the accumulation exceeds a
   threshold. Well-suited to gradual drift because it
   integrates over time.

Whichever you pick, calibrate the threshold on
back-tested historical windows. As with drift detectors,
alerting on statistical significance in a large-volume
production stream will page every day.

## Where prediction shift and concept drift diverge

Two scenarios sharpen the distinction:

**Scenario A — prediction shift with no concept drift.**
A new geography launches; users there have different
feature values; the model predicts more "high risk"
because the region legitimately has more high-risk users.
Data drift is real; concept drift is not; the model is
right. Retraining on the drifted window without careful
sampling **can hurt** — you'd overweight the new region
and unlearn what worked in the old ones.

**Scenario B — concept drift with no prediction shift.**
Fraud patterns evolve. Fraudsters now use the same
feature values as legitimate users. Overall `P(X)` and
`P(ŷ)` are stable; the model's accuracy on the
fraud-positive subset collapses. Only ground truth
reveals it. This is the case chapter 05 exists for.

The reason both matter: the retraining decision
(chapter 06) is different in each. Chapter 07 walks a
regression where distinguishing these two is the whole
diagnosis.

## The label lag — planning around it

A concept-drift monitor's speed is set by how fast labels
arrive. Some rough categories:

- **Immediate feedback** (seconds/minutes): search
  clicks, autocomplete accepts, chat response ratings.
  Concept-drift monitoring is close to real time.
- **Short-cycle feedback** (hours/days): recommendation
  conversions, ad click-throughs, fraud
  chargebacks (partial).
- **Medium-cycle feedback** (days/weeks): fraud
  chargebacks (full), refund windows, customer support
  outcomes, subscription renewal (partial).
- **Long-cycle feedback** (weeks/months+): churn,
  medical outcomes, credit default, long-term retention.

For long-cycle problems, the monitoring plan has to lean
harder on data drift, prediction-distribution shift, and
performance estimation. The concept-drift alarm might
only fire a month after the change; the earlier signals
are what you actually respond to.

Chapter 05 makes the label plumbing explicit; chapter 06
turns the decision rule around lag length into a policy.

## Common pitfalls

1. **Assuming data drift ⇒ concept drift.** A shift in
   `P(X)` does not by itself mean the model is wrong.
   Waiting for corroborating signal prevents
   unnecessary retrains.
2. **Assuming no data drift ⇒ no concept drift.** The
   inputs can look identical while `P(y|X)` moves.
   Chapter 03's silence is not chapter 04's silence.
3. **Alerting on aggregate accuracy only.** A slice
   regression can wipe out a customer segment while
   the aggregate line barely wobbles.
4. **Treating a performance estimate as measured
   accuracy.** CBPE / DLE are educated guesses; they
   break exactly where measured accuracy would surprise
   you.
5. **No baseline for the metric.** "Accuracy is 0.83"
   is meaningless without "the baseline is 0.87."
   The reference profile should carry the reference-
   window metric, per slice.
6. **Rolling window too short.** Metric bounces around
   noise; alert fatigue. Weekly or bi-weekly windows
   for most delayed-label problems.
7. **No debouncing on the change detector.** A single
   noisy week pages the on-call. Require N consecutive
   windows.
8. **No plan for the seasonal case.** Recurring drift
   (holiday, weekend) tricks change detectors. Either
   detrend the metric or fold the seasonal comparison
   into the baseline ("this week vs. same week last
   year").

## Summary

- Concept drift is `P(y|X)` moving; detecting it
  requires ground truth; ground truth is usually
  delayed.
- Prediction-distribution shift (`P(ŷ)`) is the free
  proxy: measurable in real time, correlated with both
  data drift and concept drift, useful as a leading
  indicator and dangerous as a retraining trigger by
  itself.
- Performance-estimation methods (CBPE, DLE, importance-
  weighted variants) provide *estimated* accuracy in
  the label-lag window. Treat them as guesses; verify
  when labels arrive.
- Segment the monitoring — cohort, geography, traffic
  source, value tier. Aggregate metrics hide slice
  regressions.
- Change-detection on the accuracy time series is
  standard: thresholds, SPC, or CUSUM/Page-Hinkley.
  Debounce and calibrate.
- Data drift, prediction shift, and concept drift are
  correlated but not equivalent. Alert on each; decide
  on all three.

## Where this goes next

Chapter 05 is the plumbing this chapter assumed:
**where do the labels come from, how do they get joined
to predictions, and what does the join look like once
they do?** Delayed-label pipelines, feedback loops from
the product, active-learning queues, human labeling
services, and the warehouse table that becomes the
system-of-record for "how the model actually did."
