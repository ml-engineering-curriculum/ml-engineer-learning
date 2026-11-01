# Job Requirements — Machine Learning Engineer

**Role level:** 20 (build altitude, mid-career)
**Track:** `ml-engineer-learning`
**Research window:** 2026-08-03 → 2026-11-01 (last 90 days)
**Today:** 2026-11-01

This file documents the requirements catalog for the Machine Learning Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the current curriculum plan lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json); the proposed additive delta for this cycle lives in [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json).

## Status — 38-posting refresh, zero curriculum additions (third consecutive cycle)

The 2026-11 cycle sampled **38 URL-verified live postings** (window 2026-08-03 → 2026-11-01). All 32 postings from the 2026-10 cycle were re-verified via WebFetch on 2026-11-01 — **22 are still live, 10 are no longer in-window** (7 expired, 3 redirected to parent career pages with the requisition removed). One of the 22 live-but-stale postings (`post-022` Exadel) now reads as "Senior Machine Learning Engineer" and was dropped under the Senior-exclusion filter. That leaves **21 postings carried forward**. **17 new postings** were added from Greenhouse / SmartRecruiters searches targeting new 2026-10 / 2026-11 requisitions, bringing the total to 38 — comfortably above the ≥25 minimum.

Applying the continuity-bias rule strictly:

- Every existing module carries ≥0.21 posting-frequency support; the top five sit at 0.55–0.95.
- No emerging theme clears BOTH the ≥3-postings AND ≥0.30-frequency gates under the methodology consistent with prior cycles.
- **Curriculum delta this cycle: zero additions.** See [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) — `modules`, `exercises`, and `projects` are all empty arrays.

The one signal that materially moved is **"agentic AI" as a generic qualification** — if counted loosely (any mention of "agent systems", "tool use", "agentic AI", or "agentic frameworks" beyond specific named frameworks), it reaches **0.37** across 14 postings. Under the prior-cycle methodology that counted only specific named frameworks (LangChain / LangGraph / AutoGen / MCP), the strict count is **0.11** — down slightly from 0.13 in 2026-10. The appropriate response under the WIDEN-not-add rule is to update the existing [`mod-105` lesson 10 `llm-vocab-and-delegation-map`](lessons/mod-105-deep-learning-fundamentals/10-transformer-llms-and-when-to-delegate.md) and [`mod-110` lesson 06 `build-vs-buy-and-delegation`](lessons/mod-110-ml-systems-design/06-build-vs-buy-and-delegation.md) lessons to reference contemporary agent systems (MCP, tool use, agent harnesses) by name **next cycle** — not to add a module now. This is flagged as the primary watchlist item for 2026-02.

## Methodology

1. Two parallel research agents fanned out on 2026-11-01: one re-verified every URL from the 2026-10 cycle, the other searched Greenhouse (`job-boards.greenhouse.io`) and SmartRecruiters (`jobs.smartrecruiters.com`) for new 2026-10/2026-11 requisitions matching the strict title filter (`Machine Learning Engineer`, `ML Engineer`, `Deep Learning Engineer`, `Model Training Engineer`, plus clear-match variants like `AI/ML Engineer`). Senior / Staff / Principal / Lead / Director / Head titles and specialist-track variants (ML Platform, MLOps, Applied Scientist, Research Scientist, pure Data Scientist, LLM Application Engineer, NLP Engineer, Fine-Tuning Engineer, RAG Engineer, Agent Engineer) were excluded. Three fresh-find candidates (Figma "Software Engineer — Machine Learning", Bot Auto "Algorithm Engineer, Deep Learning & Vision", NewsBreak "Machine Learning Engineer, LLM Post-Training" — the latter being a fine-tuning specialist variant by body) were dropped under the title/specialist filter.
2. Each posting was verified live via WebFetch (HTTP 200 + posting body present) before inclusion. Candidates that returned 404/410, filled-position redirects, "job has expired" banners, or bot-blocked bodies were dropped — not fabricated.
3. Frequencies were computed against the 38-posting evidence base using explicit textual mentions in the required or preferred sections. The 'agentic AI generics' bucket was tracked separately from the strict 'named agentic framework' count to make the methodology transparent.
4. Each requirement was re-checked against the frequency table and the continuity-bias gate. Only requirements below the level-20 threshold were left in `covered-at-higher-level` or `covered-at-peer`.
5. Authoritative references from prior cycles (AWS/GCP/Azure ML certifications, *Designing ML Systems*, *ML Engineering* by Burkov, Google *Rules of ML*, MLOps community references, PyTorch/scikit-learn/MLflow docs) are preserved as depth citations.

Lever (`jobs.lever.co`) and Ashby (`jobs.ashbyhq.com`) remain unavailable to WebFetch for the **third consecutive cycle** — Lever returns HTTP 404 on tested org slugs, Ashby returns header-only stubs with no qualifications body. The 38-posting evidence base cleared the posting minimum without them. See `research_status.needs_research_note` in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) for the retry plan (next cycle should attempt a browser-capable fetcher).

## Requirement themes → curriculum ownership

`Freq` is the fraction of the 38 in-window postings that cite the requirement in required or preferred sections. Continuity-bias threshold is 0.30.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Math, statistics, optimization foundations | 0.21 | `ml-engineer` (this) | [`mod-101-ml-foundations`](lessons/mod-101-ml-foundations) |
| 2 | Python + ML toolchain (NumPy, Pandas, scikit-learn, Jupyter) | 0.95 | `ml-engineer` | [`mod-102-python-ml-toolchain`](lessons/mod-102-python-ml-toolchain) |
| 3 | Data engineering for ML (cleaning, feature engineering, SQL, validation) | 0.58 | `ml-engineer` | [`mod-103-data-engineering-for-ml`](lessons/mod-103-data-engineering-for-ml) |
| 4 | Classical ML modeling (regression, trees, ensembles, model selection) | 0.29 | `ml-engineer` | [`mod-104-classical-ml-modeling`](lessons/mod-104-classical-ml-modeling) |
| 5 | Deep learning fundamentals (PyTorch, CNN, RNN, Transformer encoder) | 0.76 | `ml-engineer` | [`mod-105-deep-learning-fundamentals`](lessons/mod-105-deep-learning-fundamentals) |
| 6 | Experiment tracking + reproducibility (MLflow / W&B) | 0.32 | `ml-engineer` | [`mod-106-experiment-tracking-reproducibility`](lessons/mod-106-experiment-tracking-reproducibility) |
| 7 | Model evaluation & validation (metrics, k-fold, baselines, error analysis) | 0.34 | `ml-engineer` | [`mod-107-model-evaluation-validation`](lessons/mod-107-model-evaluation-validation) |
| 8 | Model packaging + deployment (FastAPI, Docker, batch/online inference) | 0.68 | `ml-engineer` | [`mod-108-model-deployment-serving`](lessons/mod-108-model-deployment-serving) |
| 9 | ML-specific monitoring (data/concept drift, retraining triggers) | 0.29 | `ml-engineer` | [`mod-109-ml-monitoring-drift`](lessons/mod-109-ml-monitoring-drift) |
| 10 | ML systems design (training/serving skew, A/B testing, retraining cadence) | 0.55 | `ml-engineer` | [`mod-110-ml-systems-design`](lessons/mod-110-ml-systems-design) |
| 11 | LLM / transformer awareness (light touch — vocabulary, when to delegate) | 0.50 | `ml-engineer` (delegation only) → peer specialists for depth | [`mod-105` lesson 10](lessons/mod-105-deep-learning-fundamentals/10-transformer-llms-and-when-to-delegate.md) + [`mod-110` lesson 06](lessons/mod-110-ml-systems-design/06-build-vs-buy-and-delegation.md); depth at `llm-application-developer`, `rag-engineer`, `nlp-engineer`, `fine-tuning-engineer` |
| 12 | Git / Linux / project hygiene | 0.29 | `ai-infra-junior-engineer-learning` (level 10) | Prerequisite; listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 13 | Cloud fundamentals + Kubernetes deployment | 0.37 | `ai-infra-engineer-learning` (level 20, peer) | Link-out to that curriculum; this track touches cloud only where required to deploy a model |
| 14 | Distributed training (DDP/FSDP/DeepSpeed/Ray, multi-node GPU clusters) | 0.18 | `training-pipeline-engineer-learning` (level 35) | Out of scope at level 20. Below threshold. Linked out. See emerging-themes note below — the DDP/FSDP/DeepSpeed vocabulary is beginning to appear in mid-level ML Engineer reqs (post-034, post-036). |
| 15 | Self-service ML platform engineering (feature stores, multi-tenant compute) | 0.05 | `ai-infra-ml-platform-learning` (level 30) | Below threshold. ML Engineer is a customer of these platforms, not their builder. Linked out. |
| 16 | ML/AI security depth (data poisoning, model extraction, adversarial inputs, agent security) | 0.18 | `ai-infra-security-learning` (level 35) | Below threshold. Up slightly from 0.13 — reflects more public-sector / defense postings in the fresh sample (Scale AI Public Sector, CoVar, Rackner). Surfaced as awareness in [`mod-110`](lessons/mod-110-ml-systems-design). |

### Emerging themes (below the continuity-bias gate — no action this cycle)

| # | Theme | Freq | Prior cycle | Disposition |
|---|---|---|---|---|
| E1 | Agentic AI / agent systems / tool-use generics (loose — any mention beyond specific named frameworks) | 0.37 | n/a (newly tracked) | **Primary watchlist item — strongest emerging signal.** Crosses 0.30 under loose definition. Fourteen postings reference agents at some level, from passing mention (post-038 Reddit's RAG/agentic nice-to-have) to meaningful requirement (post-040 Scale AI's "GenAI, Agentic AI" required bullet, post-039 Glean's "agent systems" in the main qualifications). Level-20 obligation remains conversant-level vocabulary and knowing when to delegate — depth belongs to [`llm-application-developer-learning`](https://github.com/ai-infra-curriculum/llm-application-developer-learning) (level 25). If this persists at ≥0.35 next cycle, UPDATE the existing [`mod-105` lesson 10 `llm-vocab-and-delegation-map`](lessons/mod-105-deep-learning-fundamentals/10-transformer-llms-and-when-to-delegate.md) and [`mod-110` lesson 06 `build-vs-buy-and-delegation`](lessons/mod-110-ml-systems-design/06-build-vs-buy-and-delegation.md) to reference contemporary agent systems (MCP, tool use, agent harnesses) by name — NOT a new module. |
| E2 | Agentic / orchestration frameworks (LangChain, LangGraph, AutoGen, MCP — strict named count) | 0.11 | 0.13 | Watchlist (down slightly). Strict count is below gate across all three cycles observed. If this crosses 0.30 in a future cycle (not just the loose-definition version at E1), widen the delegation-map exercise in [`mod-105` lesson 10](lessons/mod-105-deep-learning-fundamentals/10-transformer-llms-and-when-to-delegate.md) to reference the named frameworks explicitly. Peer-owned by `llm-application-developer-learning`. |
| E3 | AI-assisted developer tools (Claude Code, Cursor, Codex, Copilot) | 0.05 | 0.09 | Watchlist (down). The DoorDash posting that was the prior cycle's only REQUIRED mention expired. Affirm UPGRADED Claude Code / Cursor from preferred to REQUIRED this cycle — a notable qualitative shift inside a declining raw count. If REQUIRED mentions reach ≥3 distinct employers next cycle, add as a single exercise in [`mod-102-python-ml-toolchain`](lessons/mod-102-python-ml-toolchain), not a new module — it's a working-style skill layered on Python fluency. |
| E4 | Sub-second / hardware-aware inference (TensorRT / Triton / vLLM / ONNX / edge GPU) | 0.16 | 0.19 | Watchlist (down slightly — the Maven x2 Jetson/edge-GPU mentions expired this cycle; replaced only by Intuitive's TensorRT requirement and Scale AI's edge/on-prem/air-gapped deployment preference). Already incrementally covered by [`mod-108` lesson 10 `load-testing-and-tuning`](lessons/mod-108-model-deployment-serving/10-load-testing-and-tuning.md). If this crosses 0.30 in a future cycle, the right increment is an exercise added to [`mod-108`](lessons/mod-108-model-deployment-serving) — NOT a new module. |
| E5 | Robotics ML — reinforcement / imitation learning + sim-to-real (Isaac Sim, MuJoCo, Gazebo) | 0.11 | 0.13 | Specialist vertical (stable at ~0.11-0.13 for two consecutive cycles). Four postings this cycle: Torc Robotics RL Planning (carryover), Path Robotics (carryover), Scale AI Public Sector (new — deep RL as required), Skild AI (new — Isaac Sim/Gazebo). If this signal persists at ≥0.11 for a third cycle, flag for curriculum ops to decide whether a `robotics-ml-engineer` track is warranted. |
| E6 | LLM pretraining / SFT / RLHF / RLAIF / LoRA / reward modeling | 0.08 | 0.06 | [`fine-tuning-engineer-learning`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning) territory (stable across all three observed cycles). Not level-20. No change. Note: the "NewsBreak Machine Learning Engineer, LLM Post-Training" posting we found this cycle was EXCLUDED under the fine-tuning-specialist filter — if specialist-titled 'ML Engineer, LLM Post-Training' roles become common, confirm fine-tuning-engineer coverage rather than extending base ML Engineer scope. |
| E7 | Distributed training vocabulary (DDP / FSDP / DeepSpeed / Ray) in mid-level ML Engineer reqs | 0.18 | 0.19 | New observation this cycle (not previously tracked separately). Prior cycles counted this under the general distributed-training theme (level-35 peer); fresh signal is that named frameworks (DDP/FSDP/DeepSpeed by name) are starting to appear in mid-level ML Engineer II reqs (post-034 May Mobility Training Infra, post-036 Lila Sciences, post-040 Scale AI). Below gate. If this crosses 0.30, the right increment is a mod-108 (or mod-105) exercise referencing the vocabulary and the hand-off to the peer `training-pipeline-engineer-learning` track — NOT a new module. |

## Posting evidence

The 38 postings sampled this cycle. Full quotes and per-requirement `evidence_post_ids` live in [`.aicg/job-requirements.json`](.aicg/job-requirements.json).

| ID | Source | Employer | Title | Location | URL |
|---|---|---|---|---|---|
| post-002 | carryover | Torc Robotics | Machine Learning Engineer, II - App Engine | Ann Arbor, MI; Fort Worth, TX | [job-boards.greenhouse.io/torcrobotics/jobs/8525554002](https://job-boards.greenhouse.io/torcrobotics/jobs/8525554002) |
| post-003 | carryover | Torc Robotics | Machine Learning Engineer II — Learned Planning (RL) | Remote US / Ann Arbor hybrid | [job-boards.greenhouse.io/torcrobotics/jobs/8852888002](https://job-boards.greenhouse.io/torcrobotics/jobs/8852888002) |
| post-006 | carryover | TeleTracking Technologies | Machine Learning Engineer III | Pittsburgh, PA | [job-boards.greenhouse.io/teletrackingtechnologiesinc/jobs/5207632008](https://job-boards.greenhouse.io/teletrackingtechnologiesinc/jobs/5207632008) |
| post-007 | carryover | RZR Global | Machine Learning Engineer - L3 | Bengaluru | [job-boards.greenhouse.io/rzr/jobs/4241380009](https://job-boards.greenhouse.io/rzr/jobs/4241380009) |
| post-008 | carryover | Affirm | Machine Learning Engineer II (Underwriting ML) | Remote US | [job-boards.greenhouse.io/affirm/jobs/7822387003](https://job-boards.greenhouse.io/affirm/jobs/7822387003) |
| post-009 | carryover | Path Robotics | Machine Learning Engineer, Robot Learning | Columbus, OH | [job-boards.greenhouse.io/pathrobotics/jobs/8116143002](https://job-boards.greenhouse.io/pathrobotics/jobs/8116143002) |
| post-012 | carryover | CLEAR | Machine Learning Engineer II | New York, NY | [job-boards.greenhouse.io/clear/jobs/8070543](https://job-boards.greenhouse.io/clear/jobs/8070543) |
| post-014 | carryover | 10a Labs | Machine Learning Engineer | Remote | [job-boards.greenhouse.io/10alabs/jobs/4273194009](https://job-boards.greenhouse.io/10alabs/jobs/4273194009) |
| post-015 | carryover | CaseGuard | Machine Learning Engineer - Computer Vision | Arlington, VA | [job-boards.greenhouse.io/caseguard/jobs/5120675008](https://job-boards.greenhouse.io/caseguard/jobs/5120675008) |
| post-016 | carryover | Interwell Health | Machine Learning Engineer | Remote US | [job-boards.greenhouse.io/interwellhealth/jobs/5136653008](https://job-boards.greenhouse.io/interwellhealth/jobs/5136653008) |
| post-017 | carryover | May Mobility | Machine Learning Engineer II — Perf Evaluation | Remote US | [job-boards.greenhouse.io/maymobility/jobs/8187068002](https://job-boards.greenhouse.io/maymobility/jobs/8187068002) |
| post-018 | carryover | Pinterest | Machine Learning Engineer II, Responsible AI | San Francisco, CA | [job-boards.greenhouse.io/pinterest/jobs/8162046](https://job-boards.greenhouse.io/pinterest/jobs/8162046) |
| post-019 | carryover | Point72 | Machine Learning Engineer, GenAI Technology | New York, NY | [job-boards.greenhouse.io/point72/jobs/8651402002](https://job-boards.greenhouse.io/point72/jobs/8651402002) |
| post-020 | carryover | Avride | Machine Learning Engineer | Austin, TX | [job-boards.greenhouse.io/avride/jobs/4013230009](https://job-boards.greenhouse.io/avride/jobs/4013230009) |
| post-023 | carryover | NCS | Machine Learning Engineer | Singapore | [jobs.smartrecruiters.com/NCS3/6000000001170210](https://jobs.smartrecruiters.com/NCS3/6000000001170210--eg-machine-learning-engineer) |
| post-024 | carryover | ProSidian Consulting | Machine Learning Engineer | Alexandria, VA | [jobs.smartrecruiters.com/prosidianconsulting/744000119029477](https://jobs.smartrecruiters.com/prosidianconsulting/744000119029477-machine-learning-engineer) |
| post-027 | carryover | HeartVista | Machine Learning Engineer | Palo Alto, CA | [jobs.smartrecruiters.com/HeartVista/743999717950865](https://jobs.smartrecruiters.com/HeartVista/743999717950865-machine-learning-engineer) |
| post-028 | carryover | CapTech Consulting | Machine Learning / Data Science Engineer | Philadelphia, PA | [jobs.smartrecruiters.com/CapTechConsulting/744000115233237](https://jobs.smartrecruiters.com/CapTechConsulting/744000115233237-machine-learning-data-science-engineer-) |
| post-029 | carryover | Vosker (Defendec / Reconeyez) | Machine Learning Engineer | Tallinn, Estonia | [jobs.smartrecruiters.com/Vosker/744000132358110](https://jobs.smartrecruiters.com/Vosker/744000132358110-machine-learning-engineer-defendec-reconeyez) |
| post-031 | carryover | Bosch Group | AI/ML Engineer | Cluj, Romania | [jobs.smartrecruiters.com/BoschGroup/744000140212639](https://jobs.smartrecruiters.com/BoschGroup/744000140212639-ai-ml-engineer) |
| post-032 | carryover | ASOS | Machine Learning Engineer | London, UK | [jobs.smartrecruiters.com/ASOS/744000138330460](https://jobs.smartrecruiters.com/ASOS/744000138330460-machine-learning-engineer) |
| post-033 | new | StackAdapt | Machine Learning Engineer | Remote CA/US | [job-boards.greenhouse.io/stackadapt/jobs/4018063009](https://job-boards.greenhouse.io/stackadapt/jobs/4018063009) |
| post-034 | new | May Mobility | Machine Learning Engineer II — Training Infrastructure | Remote US | [job-boards.greenhouse.io/maymobility/jobs/8580411002](https://job-boards.greenhouse.io/maymobility/jobs/8580411002) |
| post-035 | new | True Anomaly | Machine Learning Engineer (II-III), Space Agentic AI | Denver, CO / Long Beach, CA | [job-boards.greenhouse.io/trueanomalyinc/jobs/5138856007](https://job-boards.greenhouse.io/trueanomalyinc/jobs/5138856007) |
| post-036 | new | Lila Sciences | Machine Learning Engineer, Physical Sciences AI | Cambridge, MA / San Francisco | [job-boards.greenhouse.io/lilasciences/jobs/4116513009](https://job-boards.greenhouse.io/lilasciences/jobs/4116513009) |
| post-037 | new | Intuitive Surgical | Machine Learning Engineer | Sunnyvale, CA | [jobs.smartrecruiters.com/Intuitive/744000118228852](https://jobs.smartrecruiters.com/Intuitive/744000118228852-machine-learning-engineer) |
| post-038 | new | Reddit | Machine Learning Engineer, Ads | Remote Ontario | [job-boards.greenhouse.io/reddit/jobs/7131934](https://job-boards.greenhouse.io/reddit/jobs/7131934) |
| post-039 | new | Glean | Machine Learning Engineer, Assistant Quality | Mountain View, CA | [job-boards.greenhouse.io/gleanwork/jobs/4711484005](https://job-boards.greenhouse.io/gleanwork/jobs/4711484005) |
| post-040 | new | Scale AI | Machine Learning Engineer, Public Sector | Denver / Honolulu / DC | [job-boards.greenhouse.io/scaleai/jobs/4631848005](https://job-boards.greenhouse.io/scaleai/jobs/4631848005) |
| post-041 | new | PatternAI | Machine Learning Engineer | San Mateo, CA | [jobs.smartrecruiters.com/patternai/743999808217169](https://jobs.smartrecruiters.com/patternai/743999808217169-machine-learning-engineer) |
| post-042 | new | CoVar | Machine Learning Engineer | Durham, NC | [job-boards.greenhouse.io/covar/jobs/5097883007](https://job-boards.greenhouse.io/covar/jobs/5097883007) |
| post-043 | new | Loka, Inc. | Machine Learning Engineer | LatAm / N. Macedonia / Portugal / Serbia | [job-boards.greenhouse.io/lokainc/jobs/4097610007](https://job-boards.greenhouse.io/lokainc/jobs/4097610007) |
| post-044 | new | AI Squared | Machine Learning Engineer | Washington, DC | [job-boards.greenhouse.io/aisquared/jobs/4604010006](https://job-boards.greenhouse.io/aisquared/jobs/4604010006) |
| post-045 | new | Rackner | AI/ML Engineer — GenAI Mission Systems | Remote US (onsite Laurel, MD occasional) | [job-boards.greenhouse.io/rackner/jobs/4717776005](https://job-boards.greenhouse.io/rackner/jobs/4717776005) |
| post-046 | new | Bosch Group (Robert Bosch LLC) | Deep Learning Engineer II (Multiple) | Pittsburgh, PA | [jobs.smartrecruiters.com/BoschGroup/744000126551708](https://jobs.smartrecruiters.com/BoschGroup/744000126551708-deep-learning-engineer-ii-multiple-positions-ref286216i-) |
| post-047 | new | ServiceNow / Moveworks | ML Engineer, GAI Search Platform — Moveworks | Mountain View, CA | [jobs.smartrecruiters.com/ServiceNow/744000125181170](https://jobs.smartrecruiters.com/ServiceNow/744000125181170-machine-learning-engineer-gai-search-platform-moveworks) |
| post-048 | new | Skild AI | Computer Vision AI & ML Engineer | San Mateo, CA | [job-boards.greenhouse.io/skildai-careers/jobs/5239930008](https://job-boards.greenhouse.io/skildai-careers/jobs/5239930008) |
| post-049 | new | EVB | Machine Learning Engineer | Remote (contractor) | [jobs.smartrecruiters.com/EVB/744000146744404](https://jobs.smartrecruiters.com/EVB/744000146744404-machine-learning-engineer) |

### Prior-cycle postings dropped this cycle

Full reasons live in `diff_vs_prior_cycle.dropped_postings` in [`.aicg/job-requirements.json`](.aicg/job-requirements.json):

- `post-001` DoorDash — Greenhouse requisition removed
- `post-004` PitchBook Data — Greenhouse requisition removed
- `post-005` Abnormal AI — redirect to parent, requisition absent
- `post-010` Maven Robotics (Manipulation) — redirect to parent, requisition absent
- `post-011` Maven Robotics (Perception) — redirect to parent, requisition absent
- `post-013` Glean (Search Quality) — Greenhouse requisition removed (replaced by `post-039` Glean Assistant Quality, a distinct new req)
- `post-021` STR — Greenhouse requisition removed
- `post-022` Exadel — title now reads "Senior Machine Learning Engineer"; fails Senior-exclusion filter
- `post-025` Nielsen — SmartRecruiters expired banner
- `post-026` Bosch Braga — SmartRecruiters expired banner
- `post-030` Coface — SmartRecruiters expired banner

## Ownership map — quick reference

- **ML Engineer (this track, level 20)** owns the build-altitude foundation: data → feature engineering → model → evaluation → deploy → monitor → iterate.
- **Senior / Staff / Principal ML Engineer** (levels 30 / 40 / 48) inherit this curriculum and add depth, architecture, and leadership framing.
- **AI Infrastructure Engineer** (level 20, peer) owns Docker, Kubernetes, cloud, CI/CD plumbing.
- **AI Infrastructure MLOps Engineer** (`ai-infra-mlops-learning`) owns deep MLOps automation (Kubeflow / Argo / Airflow orchestration engineering beyond "use it correctly").
- **AI Infrastructure ML Platform Engineer** (level 30) owns self-service platforms — feature stores, managed-service depth (SageMaker / Vertex AI / Azure ML).
- **Training Pipeline Engineer** (level 35) owns distributed training infrastructure (DDP / FSDP / DeepSpeed / Ray).
- **Model Evaluation Engineer / AI Evaluation Engineer** own evaluation engineering depth (LLM-as-judge, DeepEval-style production eval tooling).
- **NLP / RAG / LLM Application / Fine-Tuning / Applied AI Engineers** own their respective specializations, including agentic frameworks (LangChain / LangGraph / AutoGen / MCP).
- **AI Infrastructure Security Engineer** (level 35) owns ML/AI security depth.
- **AI Governance Analyst** owns governance and compliance depth.

## Diff vs 2026-10 cycle

| Signal | 2026-10 (32 posts) | 2026-11 (38 posts) | Direction |
|---|---|---|---|
| Python + ML toolchain | 0.91 | 0.95 | ↑ (near-universal) |
| Deep learning fundamentals | 0.78 | 0.76 | stable |
| Model deployment | 0.69 | 0.68 | stable |
| Data engineering | 0.63 | 0.58 | stable (small ↓) |
| ML systems design | 0.47 | 0.55 | ↑ (more end-to-end-framed postings in the new sample) |
| LLM / transformer awareness (broad) | 0.44 | 0.50 | ↑ (partly driven by agentic-AI generics) |
| Model evaluation | 0.34 | 0.34 | stable |
| Experiment tracking | 0.31 | 0.32 | stable |
| ML monitoring | 0.31 | 0.29 | stable |
| Math/stats foundations | 0.25 | 0.21 | stable (small ↓) |
| Classical ML modeling | 0.31 | 0.29 | stable |
| Cloud + K8s | 0.31 | 0.37 | ↑ (more consultancy / public-sector postings) |
| **Emerging — agentic AI generics (loose)** | n/a | 0.37 | **NEW — crosses loose gate; strongest emerging signal** |
| **Emerging — agentic frameworks (strict, named)** | 0.13 | 0.11 | ↓ |
| **Emerging — AI coding tools** | 0.09 | 0.05 | ↓ (DoorDash expired; Affirm promoted to REQUIRED) |
| **Emerging — low-latency inference** | 0.19 | 0.16 | ↓ |
| **Emerging — robotics RL / sim-to-real** | 0.13 | 0.11 | stable |
| **Emerging — LLM post-training** | 0.06 | 0.08 | stable |
| **Emerging — distributed training vocabulary (DDP/FSDP/DeepSpeed by name)** | 0.19 | 0.18 | stable (new sub-observation that named-framework mentions are appearing in level-20 reqs) |

Most frequencies are stable within normal sample noise. The one new signal worth watching is the broad "agentic AI generics" bucket at 0.37, which under a loose definition crosses the 0.30 gate — but the WIDEN-not-add rule means the appropriate response (if it persists next cycle) is to update the existing delegation-map lesson in mod-105 + mod-110 to name current agent systems, not to add a module. Agentic-framework (strict named count) continues to drop. AI coding tools dropped in raw count but Affirm's promotion of Claude Code / Cursor from preferred to REQUIRED is a notable qualitative signal to watch.

## Conclusion

The 38-posting evidence base validates the existing 10-module / 3-project curriculum without gaps. Third consecutive zero-additions cycle. No emerging theme clears both the ≥3-postings and ≥0.30-frequency gates under the methodology consistent with prior cycles. The one loose-definition crossover (agentic AI generics at 0.37) is correctly handled by WIDENING the existing delegation-map coverage path next cycle — not by adding a module at level 20, where the obligation is conversant-level vocabulary and knowing when to delegate to peer specialists.

<!-- needs-research: next cycle, (1) retry Lever (jobs.lever.co) and Ashby (jobs.ashbyhq.com) with a browser-capable fetcher — three consecutive cycles blocked is enough that the harness gap is affecting coverage breadth, (2) revisit the agentic-AI signal with the strict-vs-loose split made transparent above; if named-framework count reaches 0.30 OR the loose-definition count stays ≥0.35, widen mod-105 lesson 10 llm-vocab-and-delegation-map and mod-110 lesson 06 build-vs-buy-and-delegation to reference LangChain / LangGraph / MCP / AutoGen by name, (3) re-check the newly-flagged 'distributed training (DDP / FSDP / DeepSpeed) in mid-level ML Engineer reqs' trend observed in post-034 May Mobility Training Infra and post-036 Lila Sciences — currently 0.18, peer-owned at level 35, but if it crosses 0.30 at level 20 the right increment is a cross-link exercise in mod-108 or mod-105, not a new module, (4) confirm whether 'AI coding tools' as a REQUIRED qualification spreads to ≥3 distinct employers (Affirm now REQUIRES Claude Code / Cursor, upgraded from preferred — DoorDash's prior REQUIRED mention expired). -->
