# mod-109-ml-monitoring-drift: ML Monitoring & Drift Detection

**Estimated effort:** 12 hours.

## Learning objectives

- Instrument the ML-specific signals that generic
  observability stacks do not surface.
- Detect data drift, concept drift, and
  prediction-distribution shift.
- Design retraining triggers and ground-truth-collection
  loops.
- Debug a production model regression end-to-end.

## Chapters

1. [Why ML monitoring is its own discipline](01-why-ml-monitoring-is-its-own-discipline.md)
2. [Instrumenting ML-specific signals](02-instrumenting-ml-signals.md)
3. [Data drift detection](03-data-drift-detection.md)
4. [Concept drift and prediction-distribution shift](04-concept-and-prediction-drift.md)
5. [Ground-truth collection loops](05-ground-truth-collection.md)
6. [Retraining triggers](06-retraining-triggers.md)
7. [Debugging a production regression end-to-end](07-debugging-a-production-regression.md)

## Exercises

- [exercise-01 — ML metrics instrumentation](exercises/exercise-01-ml-metrics-instrumentation.md)
- [exercise-02 — Data drift detection](exercises/exercise-02-data-drift-detection.md)
- [exercise-03 — Retraining trigger design](exercises/exercise-03-retraining-trigger-design.md)
- [exercise-04 — Production regression post-mortem](exercises/exercise-04-production-regression-postmortem.md)

## Structure

- `01-…md` … `07-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.

## Where this sits in the track

- **Input.** A live, containerised model service from
  `mod-108` (deployment) and a paired experiment /
  registry story from `mod-106`. This module assumes
  you have something to monitor.
- **Output.** An ML-specific observability surface
  (prediction and feature distributions, per-version
  latency, feature-anomaly counters), a scheduled
  drift-detection job with calibrated thresholds, a
  ground-truth join into a `predictions_labeled`
  warehouse table, an auditable retraining decision
  rule, and a rehearsed incident-response playbook.

## Non-goals

- **Kubernetes-level observability.** The service mesh,
  pod-level CPU / memory, ingress error rates — owned
  by the AI Infrastructure Engineer / ML Platform
  tracks. This module *consumes* those signals during
  chapter 07's debugging; it does not build them.
- **Feature store internals and schema evolution.**
  `mod-103` (data engineering) owns them. This module
  consumes the feature schema; it does not implement
  it.
- **LLM- and generative-specific evaluation** (toxicity,
  hallucination scoring, embedding-similarity drift on
  prompts). A distinct discipline; separate module.
- **Fairness and bias auditing.** Overlaps with
  slice-based drift monitoring; covered as a topic in
  `mod-107` and in specialised fairness tracks.

## What ships when this module is done

Concretely, by the end of the module and exercises the
learner has produced:

- A `metrics.py` module that exposes the six ML-native
  metrics from chapter 02.
- A `reference_profile.json` frozen with the model
  artifact, per chapter 02's schema.
- A scheduled drift job that emits a per-feature and
  multivariate drift score, with thresholds calibrated
  from a historical backfill (chapter 03).
- A `predictions_labeled` warehouse table joined via
  `prediction_id` (chapter 05).
- A tested, auditable retraining decision rule
  (chapter 06).
- A written incident-response playbook and a
  post-mortem template (chapter 07).

## Where to go next

`mod-110` (ML systems design) composes this module with
`mod-108` (serving), `mod-106` (tracking / registry),
and `mod-103` (data engineering) into an end-to-end ML
system you can defend at a design review.
