# mod-110-ml-systems-design: ML Systems Design

**Estimated effort:** 14 hours.

The capstone module of the ML Engineer track. Every prior
module built a *component* of an ML system; this one is
about the *whole system* — from a stakeholder's ambiguous
business ask to a defensible one-page blueprint, a
reference architecture with named owners, a plan for the
two failure modes that break more ML systems than anything
else (training-serving skew, feedback loops), an A/B test
that measures the launch honestly, a build-vs-buy posture
on every component, and a design-review checklist that
turns the whole thing into a habit.

## Learning objectives

- Walk a problem from business goal to ML system
  blueprint.
- Identify training/serving skew, data flywheels, and
  feedback loops.
- Design an A/B test with appropriate guardrails.
- Decide build-vs-buy and when to delegate to peer
  specialist tracks (LLM application, RAG, fine-tuning,
  evaluation).

## Chapters

1. [From business goal to ML system blueprint](01-business-goal-to-ml-blueprint.md)
2. [Reference architecture of an ML system](02-reference-architecture-of-an-ml-system.md)
3. [Training-serving skew](03-training-serving-skew.md)
4. [Feedback loops and data flywheels](04-feedback-loops-and-data-flywheels.md)
5. [A/B testing ML systems](05-ab-testing-ml-systems.md)
6. [Build vs buy, and delegating to peer specialist tracks](06-build-vs-buy-and-delegation.md)
7. [The ML systems design review checklist](07-design-review-checklist.md)

## Exercises

- [exercise-01 — System design interview walkthrough](exercises/exercise-01-system-design-interview-walkthrough.md)
- [exercise-02 — Training-serving skew investigation](exercises/exercise-02-training-serving-skew-investigation.md)
- [exercise-03 — A/B test design for an ML rollout](exercises/exercise-03-ab-test-design.md)
- [exercise-04 — Build vs buy, and delegating to peer specialist tracks](exercises/exercise-04-build-vs-buy-and-delegate.md)

## Structure

- `01-…md` … `07-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.

## Where this sits in the track

- **Input.** Every prior module. `mod-103` (data
  engineering) is where the feature pipeline and the
  feature store come from; `mod-104` / `mod-105`
  (classical ML / deep learning) provide the model shapes
  that land in the blueprint's *ML formulation* section;
  `mod-106` (experiment tracking) is the registry the
  blueprint's *system shape* references; `mod-107`
  (evaluation) is where the blueprint's baseline and
  metrics come from; `mod-108` (deployment / serving) is
  the *system shape → serving* handoff; `mod-109`
  (monitoring / drift) is the observability surface every
  blueprint plugs into.
- **Output.** The artifacts a mature ML organisation
  expects for every production ML system:
  - A one-page **blueprint** (chapter 01 template) —
    context, formulation, data, system shape, baseline
    and phasing, risks, non-goals.
  - A **reference-architecture diagram** (chapter 02)
    with owners and technology per component.
  - A **row-parity harness** in CI (chapter 03) that
    blocks promotion on any training-vs-serving feature
    mismatch.
  - A **feedback-loop mitigation plan** (chapter 04) —
    untreated hold-out, exploration budget, diversity
    metrics, periodic full-population retrains.
  - A pre-registered **A/B design doc** (chapter 05) —
    unit, primary metric, guardrails with stopping
    rules, power analysis, ramp, decision rule — and
    the read-out template a launch review will read.
  - A component-by-component **build-vs-buy table**
    (chapter 06) with peer-track delegation memos
    where the problem shape says LLM App / RAG /
    Fine-Tuning / Generative Eval.
  - A repo-resident **design doc** for every production
    ML system, walked against the chapter 07 checklist
    at every promotion or A/B launch.

## Non-goals

- **LLM-specific and generative-specific system design.**
  Chapter 06 names when to delegate; this module does
  not teach the LLM Application, RAG, Fine-Tuning, or
  Generative Evaluation practices themselves. Those are
  separate tracks in the ecosystem.
- **Deep dives on the underlying components.** Feature
  stores, model registries, drift detection, incident
  response — each is covered end-to-end in a prior
  module. This module composes them; it does not
  re-teach them.
- **Vendor selection at the SKU level.** Chapter 06 is
  about the *shape* of a build-vs-buy decision, not
  about picking between two vector-store vendors as
  they exist next quarter. The chapter names concrete
  examples so the framework has teeth; the framework
  outlives the examples.
- **Organisational design.** The peer-track delegation
  map (chapter 06 Section 2) is deliberately about
  roles and handoffs, not about team structures,
  reporting lines, or people managing.

## What ships when this module is done

Concretely, by the end of the module and exercises the
learner has produced:

- A **defended blueprint + reference-architecture
  diagram** for a realistic mid-market SaaS product ask
  (exercise-01), walked against the chapter 07
  checklist.
- A **skew-investigation packet** — reproducer,
  diagnosis, fix, row-parity test in CI, artifact
  manifest, shadow-mode plan (exercise-02).
- An **A/B design packet** — pre-registered design doc,
  reproducible power analysis, risk register, read-out
  template with two hypothetical results, dry-run
  reviewer script (exercise-03).
- A **build-vs-buy packet** — nine-component decision
  table, contested-component deep dive, peer-track
  delegation memo, pilot proposal, cross-problem
  comparison (exercise-04).

## Where to go next

The module ends the ML Engineer track. The next
role-appropriate step in the ecosystem is *sideways*
rather than upward — pick a specialist track that matches
your interest and pair with the peers who own it:

- **LLM Application Engineer** — production systems
  whose primary model is a hosted LLM.
- **RAG / Retrieval Engineer** — retrieval-augmented
  systems over a specific corpus.
- **Fine-Tuning / Model-Training Specialist** —
  adapting foundation models to specific tasks.
- **Generative Evaluation Specialist** — the discipline
  of evaluating non-deterministic outputs.

The systems-design vocabulary from this module —
blueprint, reference architecture, skew, feedback loop,
A/B, build-vs-buy, checklist — carries over directly to
each. Chapter 06 Section 2 is the map.
