# exercise-01 — System Design Interview Walkthrough

> **Estimated effort:** 3 hours.
> **Companion chapters:** [01 — From business goal to
> ML system blueprint](../01-business-goal-to-ml-blueprint.md),
> [02 — Reference architecture of an ML system](../02-reference-architecture-of-an-ml-system.md),
> [07 — The ML systems design review checklist](../07-design-review-checklist.md).

## Objective

Walk a realistic, ambiguous business goal end-to-end into
a defensible ML system design. Produce two artifacts:

- A **one-page blueprint** (chapter 01's template) that
  makes every design decision explicit.
- A **reference-architecture diagram** (chapter 02's
  nine components) with owners and technology choices.

By the end you should be able to defend the design in a
15-minute conversation against the questions in
chapter 07's checklist — not because you memorised the
checklist, but because the design already answered them.

The exercise is not "pick a model." It is "make every
non-model decision explicit before the model
conversation starts."

## Prerequisites

- Chapters 01, 02, and 07 read.
- Rough familiarity with the reference architecture from
  the prior modules (`mod-103` data engineering,
  `mod-107` evaluation, `mod-108` serving, `mod-109`
  monitoring) so you know what "batch," "feature store,"
  and "model registry" mean concretely.
- A whiteboard, a drawing tool (Excalidraw, tldraw,
  Miro), or plain ASCII in a markdown file — whatever
  produces a diagram you can defend.

No code is written in this exercise. It is design work.

## Problem statement

You are the ML engineer on a small ML team at a fictional
mid-size SaaS company **StreamlineHR** — an HR-software
vendor serving small and mid-market businesses (10-500
employees). The Head of Customer Success shows up at your
team's planning meeting with the following ask:

> "We're losing too many customers in year two of their
> contract. Renewal rate is dropping. We think ML could
> help — like, predict which customers are at risk so
> our CS team can reach out before they cancel. Can your
> team build that?"

Pick **exactly one** of the following three variants for
your walkthrough. All three are realistic; the design
implications differ.

### Variant A — The renewal-risk model (default)

The Head of CS's ask, taken at face value. Predict
per-customer risk of non-renewal at the end of the
current contract. The CS team acts on the prediction.

### Variant B — The feature-adoption recommender

A related but different framing: instead of predicting
churn, predict which product feature the customer is
most likely to adopt next, so the CS team can proactively
onboard them onto it (adoption breadth is correlated
with retention).

### Variant C — The support-ticket priority scorer

A third framing at a similar altitude: score inbound
support tickets by urgency so the support team can
triage. Not a churn model directly; a customer-experience
model whose second-order effect is retention.

You will walk one to completion. Pick before starting;
switching mid-exercise is not the same drill.

## Requirements

### Part A — The blueprint (≈ 90 min)

Produce a `blueprint.md` file in the format from
chapter 01's Section 6 worked example. The document
must include all six sections:

1. **Business context and success criteria.**
   Name the business decision the model informs, the
   business metric it targets, the smallest change
   worth building for (MDE), and the person
   accountable for the model in production. You do
   not have access to StreamlineHR's real financials;
   invent plausible-but-defensible numbers and label
   them as assumptions in a *Assumptions* subsection.

2. **The ML formulation.** Task type, unit of
   prediction, horizon, label definition (in plain
   English a CS-team lead could understand), and what
   the model is explicitly *not* deciding.

3. **Data.** Where the ground truth lives, the label
   latency, what features are available at prediction
   time, and the cold-start plan.

4. **System shape.** Which of the four shapes (batch,
   streaming, online, embedded) fits, defended in
   one paragraph.

5. **Baseline and phasing.** The non-ML baseline you
   ship first (v0), the v1 model, and a v2 sketch.

6. **Risks and non-goals.** At least four risks (with
   owners) and at least three explicit non-goals.

Length target: two pages. If it goes to three, tighten;
if it fits on one, you are probably skipping something.

### Part B — The reference-architecture diagram
(≈ 60 min)

Draw the nine-component reference architecture from
chapter 02 as it applies to your chosen variant. Every
component must be:

- **Named** with a specific technology (or "none — the
  system does not have this component and here is why
  that is acceptable").
- **Owned** by a specific team (yourself, data eng,
  platform eng, the CS team, a vendor, etc.).
- **Sized.** For each component you own, a one-line
  estimate of the ongoing work — "1 DAG in Airflow,
  0.1 FTE ongoing" or "3 warehouse tables, no
  ongoing headcount."

Export or commit the diagram as `architecture.png`
(or `architecture.svg` or an ASCII `architecture.txt` —
whatever your tooling produces). Alongside it, write
`architecture.md`: one paragraph per component
justifying the naming, ownership, and sizing choices.

### Part C — The self-review pass (≈ 30 min)

Walk chapter 07's checklist against your blueprint and
diagram. In `self_review.md`, for **each** of the seven
sections of the checklist (A through G), record:

- **Questions the design answers well** — one line
  each with the question number and where in the
  blueprint the answer lives.
- **Questions the design does not answer** — one line
  each with the question number and either the plan
  to answer it before launch or an argument for why
  the question is not applicable to this design.

The point is not to have 40 green ticks. It is to have
a written accounting of what is decided and what is
deferred, so a reviewer can see it.

## Starter guidance

- **Pick the variant first, then don't switch.** The
  variants have different loops, different label
  latencies, and different peer-track handoffs;
  switching halfway wastes the setup.
- **Batch is your default unless you can defend
  online in one paragraph.** Chapter 02 argues for
  this at length; the design review will ask.
- **The blueprint's *Assumptions* subsection is
  where you get to invent numbers.** "Assume
  annual contract value averages $18,000" is fine
  as long as it is labelled. Assumptions dressed
  up as facts are not.
- **Draw the whole diagram, even the trivial
  components.** The `predictions_table` box is one
  line in SQL; drawing it forces the design to
  name where predictions land. The `labels_loop`
  box is a SQL join; drawing it forces the design
  to name label latency and label source.
- **The non-goals section is often the most
  informative.** Try to write four. If you can
  only write one, you have not yet made the
  scoping decisions.
- **The self-review is not scoring the design; it
  is scoping the follow-ups.** A design that has
  a lot of "deferred until phase 2" is fine if
  the deferrals are named.

## Acceptance criteria

- `blueprint.md` exists and contains all six of
  chapter 01's sections, with an `Assumptions`
  subsection under Section 1 that labels every
  invented number.
- The blueprint's `Business context` section names
  a specific business metric that already exists at
  a company like StreamlineHR and a numeric MDE.
- The blueprint's `ML formulation` section defines
  the unit, horizon, and label in one paragraph
  each.
- The blueprint's `Non-goals` section lists at
  least three explicit non-goals.
- `architecture.md` (or equivalent) exists and
  covers all nine components with named
  technology, named owner, and one-line sizing.
- `architecture.png` (or `.svg` or `.txt`) exists
  and can be shown to a reviewer.
- `self_review.md` exists and walks all seven
  sections of chapter 07's checklist, marking each
  question as answered (with a pointer) or
  deferred (with a plan or a "not applicable"
  argument).
- **A dry-run review.** Pair with a peer (or run
  it against yourself with a 15-minute timer) and
  walk them through the design. Any question you
  cannot answer in one sentence goes into
  `self_review.md` as a follow-up.

## Stretch goals

- **Cross-variant compare.** Do a second variant
  (10-20 minutes each, blueprint only) and write a
  short `compare.md` noting where the designs
  diverge — different loops, different label
  latencies, different peer handoffs. The point is
  to feel how the framing changes the shape.
- **A design-review script.** Write a
  `dry_run_script.md` — the specific 8-10
  questions a reviewer should ask about your
  design, in order. This is chapter 07's
  five-minute version made specific to your
  system.
- **A one-page "why this metric" defense.** The
  primary business metric you chose is one of
  many possible. Write half a page defending it
  against the two most plausible alternatives.
- **A "what changes if we're wrong" section.** For
  each of the top three risks in the blueprint,
  describe the observable signal that would tell
  you the risk was materialising and the specific
  action to take.

## What to hand in

```
exercise-01/
├── blueprint.md              # the one-page blueprint (chapter 01 template)
├── architecture.md           # per-component justification (chapter 02 shape)
├── architecture.png          # (or .svg / .txt) the nine-component diagram
├── self_review.md            # chapter 07 checklist walk
├── compare.md                # OPTIONAL — second variant compare
├── dry_run_script.md         # OPTIONAL — reviewer script
└── README.md                 # 1 paragraph: which variant, what's in the folder
```

Solutions live in the paired `ml-engineer-solutions`
repo — do not peek until you have submitted yours. A
"solution" here is an example design; there is no
uniquely correct answer. The point of the exercise is
that any answer is defensible in the checklist's terms.
