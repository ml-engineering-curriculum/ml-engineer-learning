# exercise-04 — Build vs buy, and delegating to peer specialist tracks

> **Estimated effort:** 3 hours.
> **Companion chapter:** [06 — Build vs buy, and
> delegating to peer specialist tracks](../06-build-vs-buy-and-delegation.md).
> Chapter 02 (reference architecture) and chapter 07
> (design-review checklist) are useful background.

## Objective

For a realistic ML product ask, produce the paperwork
that a serious build-vs-buy decision — and any peer-team
handoff it implies — actually requires:

- A **component-by-component build-vs-buy table** for the
  chapter 02 reference architecture, with a
  recommendation per row and a defense in one sentence.
- A **deep-dive comparison** for the one contested
  component where the recommendation is not obvious, in
  the one-page format from chapter 06 Section 3.
- A **delegation memo** for any part of the problem that
  belongs to one of the four peer specialist tracks
  (LLM Application, RAG, Fine-Tuning, Generative
  Evaluation), naming the handoff contract.
- A **time-boxed pilot proposal** for any component you
  cannot decide today.

The exercise is not "pick the vendor" or "argue for
build." It is *"produce the artifacts a design review
will attach to when the decision is challenged, so the
argument is on rows in a table rather than in
free-form debate."*

## Prerequisites

- Chapter 06 read; chapters 02 and 07 skimmed.
- Familiarity with the vendor landscape you have already
  seen in the track (`mod-106` — MLflow / W&B;
  `mod-108` — SageMaker / Vertex / Modal / KServe;
  `mod-109` — Prometheus / Grafana / Evidently /
  Arize / WhyLabs).
- No code is written in this exercise. It is design and
  writing.

## Problem statement

You are the ML engineer at a fictional mid-market SaaS
company **AcornSupport** — a customer-support-platform
vendor whose customers are B2B teams of 20-500 seats. The
head of product shows up at your team's planning meeting
with a two-part ask:

> "Our customers have thousands of internal FAQ / KB
> documents each, and their support agents spend way too
> long searching them when a customer ticket comes in. We
> want to (a) auto-triage inbound tickets — flag
> priority, route to the right queue — and (b) give
> agents a *ask-your-KB* box that answers questions with
> citations from the customer's KB. Can your team
> build both?"

The ask spans two adjacent-but-different problems. One is
a tabular / classification problem. The other is a
retrieval + generation problem. How you split the work
between "in-house build," "vendor buy," and "delegate to
a peer track" is the exercise.

Pick **exactly one** of the two problems to walk in
detail (Requirements Parts A–D), then do a lightweight
comparison of both in Part E.

### Problem 1 — Ticket triage (default)

- Predict per inbound ticket: priority (P1/P2/P3/P4) and
  the correct destination queue (billing, tech-support,
  onboarding, cancellations).
- Input: ticket subject + body (text), customer
  metadata (plan, tenure, past-ticket counts), and the
  time of day.
- Output: two labels per ticket. Latency budget: the
  agent's queue tool refreshes every few seconds; a
  100-500ms per-ticket score is fine.
- Volume: ~180k tickets/week aggregate across all
  AcornSupport customers.

### Problem 2 — Agent-facing KB Q&A

- Input: an agent-typed question in a chat sidebar,
  scoped to the current customer's KB.
- Output: an answer citing the specific KB documents
  it drew from. Agent can click through.
- Latency budget: first token in ≤ 2s; complete answer
  in ≤ 8s.
- Corpus: per-tenant, average 400-3,000 docs, updated
  weekly by the customer. Some tenants: 50k+ docs.

## Requirements

### Part A — The reference-architecture decision
table (≈ 45 min)

Produce `decisions/bvb_table.md`. Walk the nine
components of the chapter 02 reference architecture
(source systems, feature pipeline, feature store,
training pipeline, experiment tracker / registry,
serving, predictions table, observability, labels
loop — plus orchestrator) and, for each, produce a
row with:

| # | Column | Content |
|---|--------|---------|
| 1 | Component | Chapter 02 name. |
| 2 | Chosen problem's need | One line: what this problem needs from the component (or "n/a — this problem does not use one"). |
| 3 | Build v0 alternative | The minimum in-house version. "Two warehouse tables plus a Python client" is a valid answer. |
| 4 | Buy option(s) | Named vendors or OSS projects, one to three of them. |
| 5 | Recommendation | Build / Buy-vendor-X / Buy-OSS-Y / Defer / N/A. |
| 6 | One-sentence why | The rationale that survives a design review. |

Format the table so it fits on a single page in your
editor's preview; no free-form prose between rows.

You do not need to include per-row TCO estimates in
this table — the deep-dive in Part B is where the
numbers land. This table's job is coverage.

### Part B — The contested-component deep dive
(≈ 45 min)

Pick **one** row from Part A's table where the
recommendation is genuinely non-obvious (i.e., you
found yourself wanting to write "it depends" in
column 6). Write `decisions/deepdive_<component>.md`
in the chapter 06 Section 3 one-page comparison
format:

- **Criterion** column: the seven questions from
  chapter 06 Section 1's checklist, plus any
  scenario-specific criteria (data-residency,
  procurement lead time, integration cost with
  existing systems).
- **In-house build** column: your best-effort
  answer per criterion.
- **Vendor / OSS option** column: same. If there are
  two credible vendors, add a third column.
- **Recommendation** row at the bottom: the specific
  choice with the specific caveat ("Vendor X plus a
  thin adapter to cover the 15% they don't do").

For the TCO row specifically:

- Include a three-year total with explicit
  assumptions (headcount cost, infra cost, vendor
  list price, one migration event if you switch).
  Label every number as an assumption; you are not
  expected to invent AcornSupport's real financials.
- Note the *year-one* vs *year-three* crossover.
  Vendor markup usually loses year one and wins
  year three; if your comparison shows the opposite,
  that is worth flagging.

Keep the document to one page. If it needs two,
you are copying arguments across cells that could
be one row.

### Part C — The delegation memo (≈ 30 min)

Regardless of which problem you picked, write
`decisions/DELEGATION.md`. Structure:

1. **Which of the four peer tracks (if any) does the
   problem shape suggest?** Chapter 06 Section 2
   lists LLM Application, RAG, Fine-Tuning, and
   Generative Evaluation. Name the one(s) and, in
   one paragraph each, argue why the shape says
   "delegate."
   - For Problem 1 (ticket triage): the honest
     answer might be "none — this is tabular +
     short-text classification, generalist ML
     Engineer owns it." Say so, explicitly, if
     that is what you conclude.
   - For Problem 2 (KB Q&A): the honest answer is
     almost certainly "RAG, plus LLM Application,
     plus Generative Evaluation." Say so, and
     name the handoff for each.
2. **The handoff contract per delegated peer team.**
   Use chapter 06 Section 2's "handoff shape"
   template. For each delegated track, write:
   - What the ML Engineer produces (the business
     problem, the eval set, the labels or the
     labels-pipeline, the production integration
     surface).
   - What the peer team owns (the model / prompt /
     retriever / eval harness / deployment path).
   - Who owns the A/B test (chapter 05).
   - Who owns the on-call rotation for the
     resulting system.
3. **What stays with the ML Engineer even if
   delegated.** The blueprint (chapter 01), the
   business metric, the reference architecture
   context, and the design review are yours
   regardless of who builds the model.

If your chosen problem does not delegate to any
peer track, `DELEGATION.md` is still required —
it should say so in one paragraph, name the two
or three alternative framings you considered, and
explain why none of them fit the four-track menu.
"We considered delegating the KB Q&A to LLM App +
RAG but decided to keep it in-house because our
team has the retrieval expertise" is a defensible
answer *if* it is written down.

### Part D — The pilot proposal (≈ 30 min)

If Part B left you with an "it depends" answer
(and it usually will), produce `decisions/PILOT.md`
— a time-boxed pilot proposal in the chapter 06
Section 3 shape:

- **The question the pilot answers.** One
  sentence. "Does Vendor X's per-tenant vector
  store handle a 50k-doc tenant within our
  latency budget and cost envelope?"
- **The non-critical use case.** Which internal or
  small customer will host the pilot; why it is
  safe to run there; what the rollback looks like
  if the pilot fails.
- **The metrics that decide.** Three to five
  specific, measurable outcomes with pass/fail
  thresholds. "p95 latency < 2s on a 50k-doc
  tenant; monthly cost < $X per tenant;
  answer-groundedness rate ≥ 80% on the pilot
  eval set."
- **The duration.** Three months is the default;
  shorter is fine if the metrics stabilise;
  longer is a smell.
- **The stop-and-decide gate.** What the review at
  the end of the pilot looks like — who
  attends, what the artifacts are (the same
  ones this exercise produces), what the
  possible outcomes are (adopt / drop /
  extend / build v0).
- **The exit plan if the vendor is dropped.**
  What data has to be exported; what
  integrations have to be unwound; what the
  rough migration cost is.

If everything in Part B was decisively decidable,
`PILOT.md` should say so and name the *one*
future component you would put on a pilot next
year if the system grows.

### Part E — Cross-problem compare (≈ 20 min)

Write `decisions/COMPARE.md` — a short document
that runs Part A's table (headers only, one line
of "our recommendation" per component) for the
**other** problem you did not pick, alongside your
picked one. Two columns. The point is to see how
the same nine-component template produces two
very different-looking systems:

- The triage system is mostly warehouse tables,
  a scheduled batch job, and a tabular model on
  a boring buy-list of trackers and observability.
- The KB Q&A system is a vector store + an LLM
  API + a reranker + a very different observability
  surface, delegated across two or three peer
  tracks.

You do not have to fill Part A's table in full for
the other problem — headers and the recommendation
column are enough.

## Starter guidance

- **Do the "buy" defense first, even if you'll
  end up building.** Chapter 06's argument is
  that "build" is the default bias; steel-man the
  buy so the build recommendation, if you get
  there, is arrived at rather than assumed.
- **Every "buy" row names the neighbouring team
  that has to operate it.** If data engineering
  is not staffed to operate a vendor feature
  store, the buy is a short-lived experiment.
  Chapter 06 Q6 is doing real work here.
- **Every "build" row names the maintenance
  headcount.** 0.1-0.2 FTE per year *per component*
  is the honest ongoing number. If the sum across
  all your "build" recommendations exceeds your
  team's ongoing capacity, the plan is not
  fundable.
- **The delegation memo is not optional even
  when the answer is "none."** Writing "we
  considered X and Y and Z and none apply" is a
  design-review-quality answer; leaving it out
  is a shrug.
- **The pilot proposal is your escape valve when
  the decision is not decidable today.** A pilot
  is a real answer. "We'll figure it out later"
  is not.
- **Numbers are labelled assumptions, not
  claims.** "Assume 3 engineers × $180k
  fully-loaded × 6 months = $270k to build v0."
  is a fine cell; "$270k to build v0" as a
  bare number is not.
- **The vendor-cost spreadsheet is not the
  artifact.** The one-page comparison in Part B
  is. A spreadsheet backs it; the review reads
  the page.

## Acceptance criteria

- `decisions/bvb_table.md` exists and covers all
  nine reference-architecture components (plus the
  orchestrator), one recommendation and one
  sentence per row.
- `decisions/deepdive_<component>.md` exists,
  covers all seven chapter 06 Section 1 questions
  plus scenario-specific criteria, and includes a
  three-year TCO with labelled assumptions and a
  year-one-vs-year-three crossover note.
- `decisions/DELEGATION.md` names which of the four
  peer tracks (if any) the problem delegates to and
  the handoff contract per delegated track; if
  none, it names the alternatives considered.
- `decisions/PILOT.md` exists — either a real
  pilot proposal for a contested component or a
  one-paragraph note that no pilot is needed
  today and one candidate for next year.
- `decisions/COMPARE.md` exists and runs the
  headers-only comparison across the picked and
  not-picked problems.
- Every "buy" row in Part A names the
  neighbouring team that operates the vendor;
  every "build" row names the ongoing FTE.
- The whole packet is readable in 15 minutes by
  a colleague who has read chapter 06 but does
  not know AcornSupport.

## Stretch goals

- **A "vendor conversation" prep sheet.** For
  the deep-dive vendor from Part B, write the
  ten questions you would ask the vendor in a
  30-minute sales call. Not the "how does your
  product work" questions — the ones that stress
  the assumptions in your comparison table.
  Data-portability contract, list-price
  transparency, migration story if they get
  acquired, roadmap for the specific integration
  your system needs.
- **A design-review script.** Write
  `decisions/DRY_RUN.md` — the specific 8-10
  questions a senior reviewer would ask about
  your recommendation, with your one-sentence
  answers. This is chapter 07 Section F applied
  to this specific choice.
- **A "what does it cost to change our mind?"
  section.** For your top recommendation, walk
  what a reversal one year in would cost —
  engineering time, data migration, contract
  breaks, retraining. This is the exit-cost
  question with real numbers.
- **A second deep dive.** Pick a second
  contested component from Part A and produce a
  second `deepdive_*.md`. Diminishing returns
  after two.
- **Cross-tenant / multi-tenant considerations.**
  For Problem 2, add a section on how the
  build-vs-buy decision changes when the corpus
  is per-tenant and some tenants are 100x the
  size of others. Vector-store vendors price very
  differently per tenant, per storage GB, and
  per query.

## What to hand in

```
exercise-04/
├── decisions/
│   ├── bvb_table.md            # NEW — nine-component table
│   ├── deepdive_<component>.md # NEW — one-page comparison
│   ├── DELEGATION.md           # NEW — peer-track handoff memo
│   ├── PILOT.md                # NEW — pilot proposal (or "none" note)
│   ├── COMPARE.md              # NEW — headers-only cross-problem
│   └── DRY_RUN.md              # OPTIONAL — reviewer questions
└── README.md                   # 1 paragraph: which problem, what's here
```

Solutions live in the paired `ml-engineer-solutions`
repo — do not peek until you have submitted yours. As
with exercise-01, "the solution" here is one plausible
answer; the point of the exercise is that any answer
is defensible in chapter 06's terms.
