# Job Requirements — Machine Learning Engineer

**Role level:** 20 (build altitude, mid-career)
**Track:** `ml-engineer-learning`
**Research window:** 2026-07-03 → 2026-10-01 (last 90 days)
**Today:** 2026-10-01

This file documents the requirements catalog for the Machine Learning Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the current curriculum plan lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json); the proposed additive delta for this cycle lives in [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json).

## Status — 32-posting refresh, zero curriculum additions

The 2026-10 cycle sampled **32 URL-verified live postings** (2026-07-03 → 2026-10-01) spanning US / EU / APAC and covering adtech, fintech, healthtech, autonomous vehicles / robotics, retail, defense, enterprise SaaS, hardware, and public-sector tech. All postings were fetched with WebFetch (HTTP 200, body-contains-job verified) at observation. The 32-posting evidence base exceeds the ≥25-posting minimum.

Applying the continuity-bias rule strictly:

- Every existing module carries ≥0.25 posting-frequency support; the top five sit at 0.44–0.91.
- No emerging theme clears BOTH the ≥3-postings AND ≥0.30-frequency gates.
- **Curriculum delta this cycle: zero additions.** See [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) — `modules`, `exercises`, and `projects` are all empty arrays. No modules or exercises are removed (removals are out of scope per the packet contract).

The 2026-08 cycle's zero-additions decision is reaffirmed by this larger sample. The only emerging signal materially *climbing* vs the previous cycle is low-latency / hardware-aware inference (0.14 → 0.19) — still below the 0.30 gate and already incrementally covered by [`mod-108` lesson 10 `load-testing-and-tuning`](lessons/mod-108-model-deployment-serving/10-load-testing-and-tuning.md). Agentic frameworks moved in the opposite direction (0.24 → 0.13), softening the previous cycle's "close enough to watch" flag.

## Methodology

1. Two parallel research agents fanned out across Greenhouse and SmartRecruiters (plus server-rendered direct careers pages) filtering strictly on the four allowed titles (`Machine Learning Engineer`, `ML Engineer`, `Deep Learning Engineer`, `Model Training Engineer`) and excluding Senior/Staff/Principal/Lead/Director/Head, ML Platform, MLOps, Applied Scientist, Research Scientist, Data Scientist, and pure LLM-application / NLP / RAG / fine-tuning / agent-engineer specialist variants.
2. Each posting was verified live via WebFetch (HTTP 200 + posting body present) before inclusion. Candidates that returned 404/410, filled-position redirects, or bot-blocked bodies were dropped — not fabricated.
3. Frequencies were computed against the 32-posting evidence base using explicit textual mentions in the required or preferred sections. Implied-only signals were not counted.
4. Each requirement was re-checked against the frequency table and the continuity-bias gate. Only requirements below the level-20 threshold were left in `covered-at-higher-level` or `covered-at-peer`.
5. Authoritative references from prior cycles (AWS/GCP/Azure ML certifications, *Designing ML Systems*, *ML Engineering* by Burkov, Google *Rules of ML*, MLOps community references, PyTorch/scikit-learn/MLflow docs) are preserved as depth citations that survive the ephemerality of individual postings.

Lever (`jobs.lever.co`) and Ashby (`jobs.ashbyhq.com`) remain unavailable to WebFetch (403 / client-side render) for the second consecutive cycle. This cycle still cleared the posting minimum without them. See `research_status.needs_research_note` in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) for the retry plan.

## Requirement themes → curriculum ownership

`Freq` is the fraction of the 32 in-window postings that cite the requirement in required or preferred sections. Continuity-bias threshold is 0.30.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Math, statistics, optimization foundations | 0.25 | `ml-engineer` (this) | [`mod-101-ml-foundations`](lessons/mod-101-ml-foundations) |
| 2 | Python + ML toolchain (NumPy, Pandas, scikit-learn, Jupyter) | 0.91 | `ml-engineer` | [`mod-102-python-ml-toolchain`](lessons/mod-102-python-ml-toolchain) |
| 3 | Data engineering for ML (cleaning, feature engineering, SQL, validation) | 0.63 | `ml-engineer` | [`mod-103-data-engineering-for-ml`](lessons/mod-103-data-engineering-for-ml) |
| 4 | Classical ML modeling (regression, trees, ensembles, model selection) | 0.31 | `ml-engineer` | [`mod-104-classical-ml-modeling`](lessons/mod-104-classical-ml-modeling) |
| 5 | Deep learning fundamentals (PyTorch, CNN, RNN, Transformer encoder) | 0.78 | `ml-engineer` | [`mod-105-deep-learning-fundamentals`](lessons/mod-105-deep-learning-fundamentals) |
| 6 | Experiment tracking + reproducibility (MLflow / W&B) | 0.31 | `ml-engineer` | [`mod-106-experiment-tracking-reproducibility`](lessons/mod-106-experiment-tracking-reproducibility) |
| 7 | Model evaluation & validation (metrics, k-fold, baselines, error analysis) | 0.34 | `ml-engineer` | [`mod-107-model-evaluation-validation`](lessons/mod-107-model-evaluation-validation) |
| 8 | Model packaging + deployment (FastAPI, Docker, batch/online inference) | 0.69 | `ml-engineer` | [`mod-108-model-deployment-serving`](lessons/mod-108-model-deployment-serving) |
| 9 | ML-specific monitoring (data/concept drift, retraining triggers) | 0.31 | `ml-engineer` | [`mod-109-ml-monitoring-drift`](lessons/mod-109-ml-monitoring-drift) |
| 10 | ML systems design (training/serving skew, A/B testing, retraining cadence) | 0.47 | `ml-engineer` | [`mod-110-ml-systems-design`](lessons/mod-110-ml-systems-design) |
| 11 | LLM / transformer awareness (light touch — vocabulary, when to delegate) | 0.44 | `ml-engineer` (delegation only) → peer specialists for depth | [`mod-105` lesson 10](lessons/mod-105-deep-learning-fundamentals/10-transformer-llms-and-when-to-delegate.md) + [`mod-110` lesson 06](lessons/mod-110-ml-systems-design/06-build-vs-buy-and-delegation.md); depth at `llm-application-developer`, `rag-engineer`, `nlp-engineer`, `fine-tuning-engineer` |
| 12 | Git / Linux / project hygiene | 0.31 | `ai-infra-junior-engineer-learning` (level 10) | Prerequisite; listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 13 | Cloud fundamentals + Kubernetes deployment | 0.31 | `ai-infra-engineer-learning` (level 20, peer) | Link-out to that curriculum; this track touches cloud only where required to deploy a model |
| 14 | Distributed training (DDP/FSDP/Ray, multi-node GPU clusters) | 0.19 | `training-pipeline-engineer-learning` (level 35) | Out of scope at level 20. Below threshold. Linked out. |
| 15 | Self-service ML platform engineering (feature stores, multi-tenant compute) | 0.09 | `ai-infra-ml-platform-learning` (level 30) | Below threshold. ML Engineer is a customer of these platforms, not their builder. Linked out. |
| 16 | ML/AI security depth (data poisoning, model extraction, adversarial inputs, agent security) | 0.13 | `ai-infra-security-learning` (level 35) | Below threshold. Surfaced as awareness in [`mod-110`](lessons/mod-110-ml-systems-design). |

### Emerging themes (below the continuity-bias gate — no action this cycle)

| # | Theme | Freq | Prior cycle | Disposition |
|---|---|---|---|---|
| E1 | AI-assisted developer tools (Claude Code, Cursor, Codex) | 0.09 | 0.14 | Watchlist (slight decline). Only DoorDash lists this as REQUIRED; Affirm and Pinterest list as PREFERRED. If this crosses 0.30 AND spreads beyond one REQUIRED-mention employer, add as a single exercise in [`mod-102-python-ml-toolchain`](lessons/mod-102-python-ml-toolchain), not a new module — it's a working-style skill layered on Python fluency. |
| E2 | Agentic / orchestration frameworks (LangChain, LangGraph, AutoGen, MCP) | 0.13 | 0.24 | Watchlist (signal softened — opposite direction from 2026-08 flag). Peer-specialist territory ([`llm-application-developer-learning`](https://github.com/ai-infra-curriculum/llm-application-developer-learning)). If this crosses 0.30 next cycle, widen the delegation-map exercise in [`mod-105` lesson 10](lessons/mod-105-deep-learning-fundamentals/10-transformer-llms-and-when-to-delegate.md) to reference frameworks by name; do not add a module here. |
| E3 | Sub-second / hardware-aware inference (TensorRT / Triton / vLLM / ONNX / edge GPU) | 0.19 | 0.14 | Watchlist — closest-to-threshold emerging signal. Already incrementally covered by [`mod-108` lesson 10 `load-testing-and-tuning`](lessons/mod-108-model-deployment-serving/10-load-testing-and-tuning.md). Deeper hardware-aware optimization belongs to the peer inference-platform track. If this crosses 0.30 next cycle, the right increment is an exercise added to [`mod-108`](lessons/mod-108-model-deployment-serving) — NOT a new module. |
| E4 | Robotics ML — reinforcement / imitation learning + sim-to-real (Isaac Sim, MuJoCo, Gazebo) | 0.13 | n/a (not surfaced in 2026-08) | Specialist vertical. Postings that require this (Torc Robotics ×2, Path Robotics, Maven Robotics ×2) hire on the base ML Engineer skillset plus domain learning. No ownership here; if the signal persists, flag for curriculum ops to decide whether a `robotics-ml-engineer` track is warranted. |
| E5 | LLM pretraining / SFT / RLHF / RLAIF / LoRA / reward modeling | 0.06 | 0.05 | [`fine-tuning-engineer-learning`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning) territory. Not level-20. No change. |

## Posting evidence

The 32 postings sampled this cycle. Full quotes and per-requirement `evidence_post_ids` live in [`.aicg/job-requirements.json`](.aicg/job-requirements.json).

| ID | Employer | Title | Location | URL |
|---|---|---|---|---|
| post-001 | DoorDash | Machine Learning Engineer, Marketplace Optimization | San Francisco / Sunnyvale | [job-boards.greenhouse.io/doordashusa/jobs/7580407](https://job-boards.greenhouse.io/doordashusa/jobs/7580407) |
| post-002 | Torc Robotics | Machine Learning Engineer, II - App Engine | Ann Arbor, MI; Fort Worth, TX | [job-boards.greenhouse.io/torcrobotics/jobs/8525554002](https://job-boards.greenhouse.io/torcrobotics/jobs/8525554002) |
| post-003 | Torc Robotics | Machine Learning Engineer II — Learned Planning (RL) | Remote US / Ann Arbor hybrid | [job-boards.greenhouse.io/torcrobotics/jobs/8852888002](https://job-boards.greenhouse.io/torcrobotics/jobs/8852888002) |
| post-004 | PitchBook Data (Morningstar) | Machine Learning Engineer | Seattle, WA | [job-boards.greenhouse.io/pitchbookdata/jobs/4687505006](https://job-boards.greenhouse.io/pitchbookdata/jobs/4687505006) |
| post-005 | Abnormal AI | Machine Learning Engineer I - Message Security | Remote Singapore | [job-boards.greenhouse.io/abnormalsecurity/jobs/7816935003](https://job-boards.greenhouse.io/abnormalsecurity/jobs/7816935003) |
| post-006 | TeleTracking Technologies | Machine Learning Engineer III | Pittsburgh, PA | [job-boards.greenhouse.io/teletrackingtechnologiesinc/jobs/5207632008](https://job-boards.greenhouse.io/teletrackingtechnologiesinc/jobs/5207632008) |
| post-007 | RZR Global | Machine Learning Engineer - L3 | Bengaluru | [job-boards.greenhouse.io/rzr/jobs/4241380009](https://job-boards.greenhouse.io/rzr/jobs/4241380009) |
| post-008 | Affirm | Machine Learning Engineer II (Underwriting ML) | Remote US | [job-boards.greenhouse.io/affirm/jobs/7822387003](https://job-boards.greenhouse.io/affirm/jobs/7822387003) |
| post-009 | Path Robotics | Machine Learning Engineer, Robot Learning | Columbus, OH | [job-boards.greenhouse.io/pathrobotics/jobs/8116143002](https://job-boards.greenhouse.io/pathrobotics/jobs/8116143002) |
| post-010 | Maven Robotics | Machine Learning Engineer — Robot Manipulation | SF Bay Area | [job-boards.greenhouse.io/mavenrobotics/jobs/4139600008](https://job-boards.greenhouse.io/mavenrobotics/jobs/4139600008) |
| post-011 | Maven Robotics | Machine Learning Engineer — Robot Perception | SF Bay Area | [job-boards.greenhouse.io/mavenrobotics/jobs/4525064008](https://job-boards.greenhouse.io/mavenrobotics/jobs/4525064008) |
| post-012 | CLEAR | Machine Learning Engineer II | New York, NY | [job-boards.greenhouse.io/clear/jobs/8070543](https://job-boards.greenhouse.io/clear/jobs/8070543) |
| post-013 | Glean | Machine Learning Engineer, Search Quality | Mountain View, CA | [job-boards.greenhouse.io/gleanwork/jobs/4006735005](https://job-boards.greenhouse.io/gleanwork/jobs/4006735005) |
| post-014 | 10a Labs | Machine Learning Engineer | Remote | [job-boards.greenhouse.io/10alabs/jobs/4273194009](https://job-boards.greenhouse.io/10alabs/jobs/4273194009) |
| post-015 | CaseGuard | Machine Learning Engineer - Computer Vision | Arlington, VA | [job-boards.greenhouse.io/caseguard/jobs/5120675008](https://job-boards.greenhouse.io/caseguard/jobs/5120675008) |
| post-016 | Interwell Health | Machine Learning Engineer | Remote US | [job-boards.greenhouse.io/interwellhealth/jobs/5136653008](https://job-boards.greenhouse.io/interwellhealth/jobs/5136653008) |
| post-017 | May Mobility | Machine Learning Engineer II — Perf Evaluation | Remote US | [job-boards.greenhouse.io/maymobility/jobs/8187068002](https://job-boards.greenhouse.io/maymobility/jobs/8187068002) |
| post-018 | Pinterest | Machine Learning Engineer II, Responsible AI | San Francisco, CA | [job-boards.greenhouse.io/pinterest/jobs/8162046](https://job-boards.greenhouse.io/pinterest/jobs/8162046) |
| post-019 | Point72 | Machine Learning Engineer, GenAI Technology | New York, NY | [job-boards.greenhouse.io/point72/jobs/8651402002](https://job-boards.greenhouse.io/point72/jobs/8651402002) |
| post-020 | Avride | Machine Learning Engineer | Austin, TX | [job-boards.greenhouse.io/avride/jobs/4013230009](https://job-boards.greenhouse.io/avride/jobs/4013230009) |
| post-021 | STR | Machine Learning Engineer | Woburn, MA / Arlington, VA | [job-boards.greenhouse.io/systemstechnologyresearch/jobs/4700584006](https://job-boards.greenhouse.io/systemstechnologyresearch/jobs/4700584006) |
| post-022 | Exadel | Middle Machine Learning Engineer | Bulgaria / Hungary / Poland | [job-boards.greenhouse.io/externaljobboards/jobs/6209673004](https://job-boards.greenhouse.io/externaljobboards/jobs/6209673004) |
| post-023 | NCS | Machine Learning Engineer | Singapore | [jobs.smartrecruiters.com/NCS3/6000000001170210](https://jobs.smartrecruiters.com/NCS3/6000000001170210--eg-machine-learning-engineer) |
| post-024 | ProSidian Consulting | Machine Learning Engineer | Alexandria, VA | [jobs.smartrecruiters.com/prosidianconsulting/744000119029477](https://jobs.smartrecruiters.com/prosidianconsulting/744000119029477-machine-learning-engineer) |
| post-025 | The Nielsen Company | AI / ML Engineer II / III | Bengaluru | [jobs.smartrecruiters.com/TheNielsenCompany/3743990014309006](https://jobs.smartrecruiters.com/TheNielsenCompany/3743990014309006-ai-ml-engineer-ii-iii) |
| post-026 | Bosch Group (Car Multimedia) | Deep Learning Engineer – AI Safety | Braga, Portugal | [jobs.smartrecruiters.com/BoschGroup/744000127913891](https://jobs.smartrecruiters.com/BoschGroup/744000127913891-deep-learning-engineer-ai-safety-f-m-div-) |
| post-027 | HeartVista | Machine Learning Engineer | Palo Alto, CA | [jobs.smartrecruiters.com/HeartVista/743999717950865](https://jobs.smartrecruiters.com/HeartVista/743999717950865-machine-learning-engineer) |
| post-028 | CapTech Consulting | Machine Learning / Data Science Engineer | Philadelphia, PA | [jobs.smartrecruiters.com/CapTechConsulting/744000115233237](https://jobs.smartrecruiters.com/CapTechConsulting/744000115233237-machine-learning-data-science-engineer-) |
| post-029 | Vosker (Defendec / Reconeyez) | Machine Learning Engineer | Tallinn, Estonia | [jobs.smartrecruiters.com/Vosker/744000132358110](https://jobs.smartrecruiters.com/Vosker/744000132358110-machine-learning-engineer-defendec-reconeyez) |
| post-030 | Coface | Data Scientists / ML Engineers | Bois-Colombes, France | [jobs.smartrecruiters.com/Coface/744000141953005](https://jobs.smartrecruiters.com/Coface/744000141953005-data-scientists-ml-engineers-) |
| post-031 | Bosch Group | AI/ML Engineer | Cluj, Romania | [jobs.smartrecruiters.com/BoschGroup/744000140212639](https://jobs.smartrecruiters.com/BoschGroup/744000140212639-ai-ml-engineer) |
| post-032 | ASOS | Machine Learning Engineer | London, UK | [jobs.smartrecruiters.com/ASOS/744000138330460](https://jobs.smartrecruiters.com/ASOS/744000138330460-machine-learning-engineer) |

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

## Diff vs 2026-08 cycle

| Signal | 2026-08 (21 posts) | 2026-10 (32 posts) | Direction |
|---|---|---|---|
| Python + ML toolchain | 0.76 | 0.91 | ↑ (near-universal) |
| Deep learning fundamentals | 0.67 | 0.78 | ↑ |
| Model deployment | 0.76 | 0.69 | ↓ (stable, within sample noise) |
| Data engineering | 0.67 | 0.63 | ↓ (stable) |
| ML systems design | 0.62 | 0.47 | ↓ (fewer end-to-end-framed postings this cycle) |
| LLM / transformer awareness | 0.52 | 0.44 | ↓ (still above light-touch threshold) |
| Model evaluation | 0.52 | 0.34 | ↓ (still above gate) |
| Experiment tracking | 0.43 | 0.31 | ↓ |
| ML monitoring | 0.43 | 0.31 | ↓ |
| Math/stats foundations | 0.38 | 0.25 | ↓ |
| Classical ML modeling | 0.33 | 0.31 | stable |
| Cloud + K8s | 0.38 | 0.31 | stable |
| **Emerging — AI coding tools** | 0.14 | 0.09 | ↓ |
| **Emerging — agentic frameworks** | 0.24 | 0.13 | ↓ (signal softened) |
| **Emerging — low-latency inference** | 0.14 | 0.19 | ↑ (closest to gate) |
| **Emerging — robotics RL / sim-to-real** | n/a | 0.13 | newly surfaced |
| **Emerging — LLM post-training (SFT/RLHF)** | 0.05 | 0.06 | stable |

Several existing-module frequencies declined this cycle (math/stats 0.38 → 0.25, model evaluation 0.52 → 0.34, ML systems design 0.62 → 0.47). None of these drops cross the "no support" threshold that would warrant demoting a module, and all remain at or above the ≥0.25 floor. The declines are explained by a sample that leaned slightly more vertical (robotics / autonomous-vehicle / CV domain postings) than the 2026-08 cycle, where several postings framed the whole lifecycle explicitly. Removals remain out of scope for this packet.

## Conclusion

The 32-posting evidence base validates the existing 10-module / 3-project curriculum without gaps. No emerging theme clears both the ≥3-postings and ≥0.30-frequency gates. The one climbing signal (low-latency / hardware-aware inference, 0.14 → 0.19) is already incrementally covered by [`mod-108` lesson 10](lessons/mod-108-model-deployment-serving/10-load-testing-and-tuning.md); if it crosses 0.30 next cycle, the right increment is a single mod-108 exercise, not a new module. Agentic-framework signal softened (0.24 → 0.13) in the opposite direction of the previous cycle's "close enough to watch" flag.

<!-- needs-research: next cycle, (1) retry Lever (browser-capable fetcher) and Ashby (headless render) now that two consecutive cycles have been blocked, (2) re-check whether low-latency inference crosses 0.30 — it is the closest-to-threshold emerging signal, (3) confirm whether 'AI coding tools' as a REQUIRED (not merely preferred) qualification spreads beyond DoorDash, (4) if robotics RL / sim-to-real signal persists at ≥0.13 for another cycle, flag for curriculum ops to decide whether a `robotics-ml-engineer` track is warranted. -->
