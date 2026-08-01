# 06 — Build vs buy, and delegating to peer specialist tracks

> **Reading time:** ~20 minutes.
> **Prerequisite chapters:** 01–05.

## Motivation

The best decision on an ML project is often the one that
takes the ML engineer *out* of the critical path. A
managed service that costs $2k/month and does 90% of
what a custom system would do is almost always cheaper —
on people, on time, on future maintenance — than the
custom system, even after the vendor's markup. A peer
team that specialises in the specific problem shape (LLM
apps, RAG, fine-tuning, evaluation) is usually going to
build a better result than a generalist ML engineer
picking up the domain from scratch.

Yet the default in most ML teams is to build. The
reasons are cultural (building is more interesting than
integrating), organisational (a team's headcount is
justified by what it builds), and technical (integrations
have their own tax). The build-first bias is the
single largest source of avoidable engineering debt in
production ML systems.

This chapter is the framework that keeps the bias in
check. Section 1 is the build-vs-buy decision framework
itself; Section 2 is the specific delegation map — the
four peer specialist tracks and when the ML engineer
should hand off to them instead of learning the domain
alone; Section 3 is how to say "no" or "buy" in a
design review without losing credibility.

## Section 1 — The build-vs-buy framework

The question is not "should we build or should we
buy?" in isolation. It is: **for each component of the
reference architecture in chapter 02, is the marginal
value of building it in-house worth the total cost of
ownership?**

Rewrite the diagram from chapter 02 as a decision
table:

| Component | Common build | Common buy |
|-----------|-------------|-----------|
| Source systems | Owned | Owned |
| Feature pipeline | Owned | Rare (dbt / SQLMesh templates) |
| Feature store | Rare in-house | Feast / Tecton / Databricks FS |
| Training pipeline | Owned | Rare (Vertex AI, SageMaker) |
| Experiment tracker / registry | In-house | MLflow (OSS), W&B, Comet |
| Serving | Owned | SageMaker, Vertex, Modal, KServe |
| Predictions table | Owned (in warehouse) | n/a |
| Observability | Grafana, Prometheus | Arize, WhyLabs, Fiddler |
| Labels loop | Owned (SQL) | Sometimes Snorkel, Scale |
| Orchestrator | Airflow, Dagster (OSS) | Astronomer, Prefect Cloud |

The default for a small team is "build the boring
plumbing, buy the specialised components," but the
default is not the answer. Walk the checklist below
per component.

### Checklist: the seven build-vs-buy questions

1. **Is this our core competitive advantage?**
   If yes, building is a serious option — the
   competitive edge is the reason to own it. If no,
   buying is the default. The recommender that ranks
   the product catalogue is core to an e-commerce
   business; the feature store the recommender uses
   is not.

2. **What is the total cost of ownership over three
   years?** Include: engineering time to build (1x
   headcount for how many months?), ongoing
   maintenance (0.2x headcount per year, minimum),
   infrastructure cost, migration cost if you
   change your mind. Compare to the vendor's
   three-year total. The vendor's markup usually
   loses on year one and wins on year three.

3. **How mature is the market?** A component that
   ten vendors sell (experiment trackers,
   observability) has commodity pricing and
   commodity behaviour. A component that two
   vendors sell (feature stores, some MLOps
   platforms) is riskier — pricing power is on
   their side and lock-in matters more.

4. **How much of the vendor's product do we
   actually need?** If the vendor's product does
   50 things and you need 5 of them, you are paying
   for the 45 you don't. Sometimes worth it (the
   integration surface is worth the overpay);
   sometimes a strong signal to look at the
   open-source alternative.

5. **What is the exit cost if this vendor
   disappears?** Cloud provider ML platforms don't
   disappear; startup ML platforms sometimes do.
   Data portability, model-artifact portability,
   and dependency lock-in bound the risk. A vendor
   whose format is a proprietary binary is a
   different risk than a vendor whose format is a
   MLflow package.

6. **Is the boundary team ready to consume it?**
   Buying a feature store means data engineering
   has to operate it. Buying a model-observability
   product means SRE has to route its alerts. If
   the neighbouring team is not staffed for the
   ongoing relationship, the vendor is a
   short-lived experiment.

7. **What is the "build v0" alternative?** Before
   the vendor conversation, sketch the minimum
   internal version. A "feature store" that is
   two warehouse tables and one Python client is
   sometimes exactly enough for the first year.
   The comparison is not "buy vs a full custom
   platform"; it is "buy vs the minimum viable
   in-house version."

### Where the buy usually wins

- **Experiment tracking / model registry.** MLflow,
  Weights & Biases, Comet, and Neptune are cheap
  ($0-$50/user/month at small scale), mature, and
  have well-known integration surfaces. Building
  your own is a solved problem with a solved
  vendor.
- **ML observability at scale.** Arize, WhyLabs,
  Fiddler. The offering is meaningfully more than
  a Grafana board — drift detection, slice
  monitoring, root-cause tooling. If your team is
  running one or two models, `mod-109`'s Prometheus
  approach is enough; at ten-plus models, the
  vendor pays for itself.
- **LLM providers.** OpenAI, Anthropic, Google. The
  build alternative — training a foundation model —
  is not a fair comparison; even fine-tuning against
  a proprietary base is a different question. See
  section 2 for the delegation story.
- **Managed training platforms.** Vertex AI,
  SageMaker, and Databricks handle the training-
  pipeline plumbing (job orchestration,
  distributed training, artifact storage) with a
  known contract. For teams without a dedicated
  ML platform team, the buy is usually right.
- **Annotation and labelling tools.** Scale AI,
  Labelbox, Snorkel. The build alternative is
  a labelling UI, a workflow tool, and a QA
  process. That is a real product, and vendors
  build them.

### Where the build usually wins

- **The feature pipeline logic itself.** The
  transforms are domain-specific and shape the
  model. This is code you write; a vendor cannot
  write it for you.
- **The model architecture and training code.**
  Buy the platform; build the model. AutoML has a
  role for baselines and specific narrow problems;
  it does not replace the modelling work.
- **The A/B experimentation platform for large,
  latency-sensitive systems.** Netflix, LinkedIn,
  Airbnb, Uber, and Microsoft all built theirs
  in-house because the specific requirements of
  live traffic exposure and metric computation at
  their scale are not commoditised. At smaller
  scale, buying works fine.
- **The core business logic that surrounds the
  model.** The intervention selector, the outreach
  scheduler, the pricing engine — code that the
  business's competitive differentiation lives in.
- **The predictions table and the labels loop.** In
  most systems these are warehouse tables and SQL
  queries. Building is a week of work; buying is
  overhead.

## Section 2 — Delegation to peer specialist tracks

The ML Engineer role exists inside an ecosystem of
adjacent specialists. Four of them own problems that
overlap with what a generalist ML engineer might be
tempted to solve alone. In each case, the right
answer is often to *collaborate with or delegate to*
that peer, not to grow the ML Engineer's scope.

The four tracks:

### Track 1: LLM Application Engineer

**What they own:** production systems whose primary
model is a large language model — usually a hosted
one (OpenAI, Anthropic, Google) via API. Prompt
engineering, function calling / tool use, agents,
LLM-specific evaluation, latency/cost tuning for
token streams, prompt caching, safety filters,
guardrails on outputs.

**When to delegate to them:**

- The task is text-shaped: classification, extraction,
  summarisation, chat, question-answering. An LLM
  zero-shot or few-shot is often good enough for the
  first year, and building a custom text model is
  meaningfully more expensive than paying the API.
- The system is agentic — chains of LLM calls, tool
  use, external actions. This is its own discipline
  and requires the LLM app team's operational
  patterns (tracing, prompt versioning, evaluation
  harnesses).
- The build alternative is training a text model
  from scratch. In 2026 this is almost never the
  right answer at typical team scale.

**When *not* to delegate:**

- The task is tabular or structured (numeric or
  categorical features → prediction). LLMs are
  strong at text; they are worse and slower and
  more expensive than a boosted-tree model for
  tabular prediction.
- Latency is under 100ms with high QPS and a strict
  cost budget. LLM inference doesn't fit.
- The output is a fine-grained numerical prediction
  where calibration and interpretability matter.
  LLMs are the wrong tool.

**The handoff shape:** the ML Engineer defines the
business problem (chapter 01) and the evaluation
metric. The LLM app team owns the prompt, the model
choice, the tool wiring, and the LLM-specific
observability. Both teams share responsibility for
the A/B test (chapter 05). Deployment sits with
whoever is closer to the serving path.

### Track 2: RAG / Search / Retrieval Engineer

**What they own:** retrieval-augmented generation
systems and the specialised retrieval infrastructure
underneath them — embedding models, vector databases
(Pinecone, Weaviate, pgvector, Milvus, LanceDB),
hybrid retrieval (dense + BM25), reranking pipelines,
context-window management, chunking strategies,
citation and grounding, retrieval-specific
evaluation (Recall@K, groundedness, faithfulness).

**When to delegate to them:**

- The task requires grounding on a specific corpus
  (documentation search, internal knowledge base,
  customer-support KB, legal-document Q&A).
- The system needs source citations and users need
  to click through to the underlying document.
- Retrieval quality is the primary bottleneck —
  the LLM is only as good as what it retrieves.
- The corpus is large, updated frequently, or
  multi-tenant.

**When *not* to delegate:**

- The task is a straightforward classification on
  short text. Retrieval is overkill.
- The corpus is small enough (a few hundred
  documents, a few million tokens) to fit in a
  long-context prompt. Buy the prompt; skip
  retrieval.
- The users don't need grounding; the LLM's
  general knowledge is enough.

**The handoff shape:** the ML Engineer defines the
business problem and the eval set (real queries with
graded relevance labels). The RAG team owns the
retrieval architecture, the vector store choice,
the chunking strategy, the reranker, and the
LLM/retrieval integration. Evaluation is joint.

### Track 3: Fine-Tuning / Model-Training Specialist

**What they own:** adapting foundation models to
specific tasks or domains — supervised fine-tuning,
instruction tuning, preference optimisation (RLHF,
DPO), adapter-based methods (LoRA, QLoRA),
domain adaptation, distillation. Also: training and
serving the resulting weights, including quantisation,
GPU inference optimisation, and multi-tenant serving.

**When to delegate to them:**

- Prompt engineering on a hosted LLM has hit a
  ceiling and the task requires the model to
  internalise a specific style, format, or domain
  vocabulary.
- Cost or latency requirements make hosted-API
  inference infeasible and a smaller fine-tuned
  model would meet both.
- There is a proprietary data asset that a
  fine-tuned model can leverage in a way a
  general-purpose model cannot.
- Compliance requires the model to run in a
  specific environment (on-prem, air-gapped).

**When *not* to delegate:**

- The baseline hosted LLM is good enough. Most
  projects that consider fine-tuning discover
  that the prompt was the problem.
- The available training data is small (< a few
  thousand high-quality examples). Fine-tuning
  usually underperforms good in-context examples
  at small scale.
- The team cannot afford the ongoing cost of
  hosting the fine-tuned model.

**The handoff shape:** the ML Engineer produces the
training dataset (or the labels loop that will
produce it), the evaluation harness, and the
production integration. The fine-tuning team owns
the training run, the hyperparameter choices, the
model-quality tradeoffs, and the deployment of the
fine-tuned artifact.

### Track 4: Evaluation / QA Specialist for Generative
   Systems

**What they own:** the discipline of evaluating
generative systems whose outputs are not
deterministic and cannot be scored against a fixed
ground-truth vector. Rubric-based evaluation,
LLM-as-judge evaluation, adversarial testing,
red-teaming, factuality and grounding checks,
regression suites for prompt / model changes,
safety and bias evaluation for generative outputs.

**When to delegate to them:**

- The system involves an LLM whose outputs are
  free-form text.
- Small changes to the prompt or model version
  produce hard-to-quantify differences in output
  quality that need a defensible measurement
  process.
- The system has safety or compliance requirements
  (medical, legal, financial) where "the output
  looked fine" is not a passing bar.
- Multi-turn agents where the evaluation is on the
  trajectory, not the single response.

**When *not* to delegate:**

- The system's output is a classification or a
  score with a real ground truth. `mod-107` metrics
  apply; the generalist ML engineer owns it.

**The handoff shape:** the ML Engineer or LLM App
Engineer produces the system; the Evaluation
Specialist owns the eval harness, the offline
regression suite, and the read-out from any
prompt/model change. A/B testing (chapter 05) is
still on the ML Engineer.

### The composition — how to draw the org chart

For a mid-size team, a typical composition:

- **1 ML Engineer** (generalist) owns the tabular
  systems, the reference architecture, and the
  A/B experimentation practice.
- **1 LLM Application Engineer** owns any customer-
  facing LLM feature.
- **A shared Data Engineer** owns the source
  systems, the feature pipeline plumbing, and the
  warehouse.
- **A shared Platform Engineer** owns the
  serving infrastructure below the container.

At smaller scale (one or two people total), the ML
Engineer wears more hats but *knows the boundaries*
— which is what this chapter is about. When the
team grows, the boundaries harden into roles.

At larger scale, add:

- **RAG / Retrieval** as its own team once you have
  two or three retrieval-based products.
- **Fine-Tuning** as its own team once you have
  either the data volume or the cost pressure to
  justify running your own models.
- **Generative Eval** as its own team or embedded
  practice once any customer-facing feature
  produces free-form output at scale.

## Section 3 — Making the argument in a design review

The build-vs-buy conversation is rarely a decision
between "correct" and "incorrect" answers; it is a
decision between two defensible choices whose
downsides play out differently over time. Which
means the presentation matters as much as the
analysis.

Two useful moves:

### The written comparison

A one-page table with three columns — the two
options and the criteria — is the artifact. Every
row is a criterion from Section 1's checklist plus
any project-specific ones. The last row is a
recommendation. The design-review discussion attaches
to specific rows; the meeting is 15 minutes instead
of an hour.

```
Criterion                       In-house build   Vendor X
Core competitive advantage      No               No
Three-year TCO                  $340k            $180k
Market maturity                 —                8+ vendors
Coverage of our needs           100%             85%
Exit cost                       —                Data export → warehouse; medium
Neighbouring team ready         Yes (us)         Yes (data-eng owns the runtime)
Build v0 alternative            Yes ($60k, 3 mo) —
Recommendation                                    Vendor X + a thin adapter for the 15%
```

The vendor's 15% gap is called out. The adapter
is scoped and estimated. The recommendation is
specific.

### The pilot

If a component is genuinely uncertain, the
right answer in the design review is a
time-boxed pilot: pay for the vendor for three
months on one non-critical use case; measure
the specific metrics that would sway the
decision; make the permanent decision at the
end of the pilot.

Pilots are how good teams stay honest about
build-vs-buy without losing the year to
speculation.

## Common pitfalls

1. **The "we'll build it later" trap.** The team
   buys short-term with a plan to replace next
   year. Next year never comes; the vendor
   relationship deepens; migration is now
   impossible. Sometimes fine (the vendor is
   permanent); sometimes an unwelcome surprise.
2. **The "not invented here" trap.** The team
   refuses the vendor because building is more
   interesting. The custom system consumes an
   engineer-year, works as advertised for two
   years, and then everyone who understood it has
   left. The debt outlasts the incumbents.
3. **Buying the platform, building the model
   anyway.** Common on Vertex AI / SageMaker —
   the team pays for the platform and then
   ignores its training-pipeline primitives to
   run their own Kubernetes jobs. Pick a lane.
4. **Delegating without a handoff document.** The
   ML Engineer hands the LLM app to the LLM app
   team without defining the metric, the eval
   harness, or the ownership boundary. Six
   weeks later, both teams are surprised by the
   other's expectations.
5. **Reinventing the LLM evaluation wheel for a
   text classification task.** `mod-107`'s
   classification metrics apply; the eval team
   isn't needed.
6. **A "build" decision with no maintenance
   headcount.** The system ships; the person who
   built it moves on; the system rots. The build
   was the wrong choice by the end of year two.
7. **A "buy" decision that never inspects the
   vendor's cost curve.** Usage grows; the
   monthly bill triples; the migration case is
   now the "buy" defense. Watch the curve;
   renegotiate before it bites.
8. **Delegating strategy to the vendor.** The
   vendor's roadmap becomes your roadmap. When
   they deprecate the feature you rely on, you
   are on their timeline, not yours.
9. **Not naming the peer track when the problem
   shape says "LLM app" or "RAG."** The ML
   Engineer spends six weeks reinventing
   retrieval-augmented generation, ships a
   worse version of what the LLM app team
   already deployed, and duplicates the eval
   surface.
10. **Framing the decision as "we can't afford
    to buy" when the labour math says
    otherwise.** Engineer-months are the
    dominant cost; vendor markup is usually
    smaller than a headcount.

## Summary

- The default in ML teams is to build; the default
  is often wrong. Every component of the reference
  architecture deserves a build-vs-buy question.
- The seven-question checklist is the working
  framework: competitive advantage, three-year
  TCO, market maturity, coverage, exit cost,
  boundary team, and build-v0 alternative.
- The buy usually wins on trackers, observability,
  LLM providers, managed training platforms, and
  annotation tools; the build usually wins on
  domain-specific model code, business logic, and
  small warehouse-backed plumbing.
- Four peer specialist tracks — LLM App, RAG,
  Fine-Tuning, Generative Evaluation — own
  problems the generalist ML Engineer should
  delegate to rather than solve alone. Each has
  a specific handoff shape.
- The design-review artifact is the one-page
  comparison table; the honesty check is the
  time-boxed pilot.

## Where this goes next

Chapter 07 is the synthesis: the design-review
checklist. Everything from chapters 01–06 turns
into a set of questions you take into every ML
system design conversation — what's the business
metric, what's the loop, what's the guardrail,
what's bought vs built, what's not-goal. The
checklist is what makes the module habit-forming
rather than one-shot.
