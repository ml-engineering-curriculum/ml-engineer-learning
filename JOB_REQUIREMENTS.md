# Job Requirements — Machine Learning Engineer

**Role level:** 20 (build altitude, mid-career)
**Track:** `ml-engineer-learning`
**Research window:** 2026-05-03 → 2026-08-01 (last 90 days)
**Today:** 2026-08-01

This file documents the requirements catalog for the Machine Learning Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the current curriculum plan lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json); the proposed additive delta for this cycle lives in [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json).

## Status — live postings backfilled, zero curriculum additions

The bootstrap cycle's `postings` array (empty) has been replaced with **21 URL-verified live postings** (2026-05-03 → 2026-08-01) spanning US / EU / APAC and covering adtech, fintech, healthtech, autonomous vehicles, research infrastructure, retail, and defense. Every posting was fetched with WebFetch (HTTP 200, body-contains-job verified) at time of observation.

The stated target was ≥25. Actual: 21. **Reason for shortfall (transparent):** Lever (`jobs.lever.co`) returned HTTP 403 to WebFetch across the roles queried; Ashby (`jobs.ashbyhq.com`) renders posting bodies client-side, so WebFetch returned empty bodies for the specific requisitions. Additional sponsor employers (Anthropic, OpenAI, Pinterest, Airbnb, Robinhood, Reddit, Netflix, Uber, Meta) had either only Senior/Staff/Principal/Applied-Scientist requisitions open (excluded by the title filter) or expired base-title requisitions that redirected to a jobs-index page. Per the no-invention rule, no postings were fabricated to close the gap. The 21-posting evidence base still clears the ≥3-posts evidence gate for every requirement below.

Applying the continuity-bias rule strictly:

- Every existing module carries ≥0.33 posting-frequency support (see table below).
- No emerging theme clears BOTH the ≥3-postings AND ≥0.30-frequency gates.
- **Curriculum delta this cycle: zero additions.** See [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) — `modules`, `exercises`, and `projects` are all empty arrays. No modules or exercises are removed (removals are out of scope per the packet contract).

## Methodology

1. Two parallel research agents fanned out across Greenhouse, Lever, Ashby, SmartRecruiters, and direct company careers pages, filtering strictly on the four allowed titles (`Machine Learning Engineer`, `ML Engineer`, `Deep Learning Engineer`, `Model Training Engineer`) and excluding Senior/Staff/Principal/Lead, ML Platform, MLOps, Applied Scientist, Research Scientist, Data Scientist, and pure LLM-application / NLP / RAG / fine-tuning variants.
2. Each posting was verified live via WebFetch (HTTP 200 + posting body present) before inclusion.
3. Frequencies were computed against the 21-posting evidence base using explicit textual mentions in the required/preferred sections. Implied-only signals were not counted.
4. Every requirement was re-checked against the frequency table and the continuity-bias gate. Only requirements below the level-20 threshold were left in `out_of_scope` or `covered_at_higher_level`.
5. Authoritative references from the bootstrap cycle were preserved (AWS/GCP/Azure ML certifications, *Designing ML Systems*, *ML Engineering* by Burkov, Google *Rules of ML*, MLOps community references, PyTorch/scikit-learn/MLflow docs) as depth citations that survive the ephemerality of individual postings.

## Requirement themes → curriculum ownership

`Freq` is the fraction of the 21 in-window postings that cite the requirement in required or preferred sections. Continuity-bias threshold is 0.30.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Math, statistics, optimization foundations | 0.38 | `ml-engineer` (this) | [`mod-101-ml-foundations`](lessons/mod-101-ml-foundations) |
| 2 | Python + ML toolchain (NumPy, Pandas, scikit-learn, Jupyter) | 0.76 | `ml-engineer` | [`mod-102-python-ml-toolchain`](lessons/mod-102-python-ml-toolchain) |
| 3 | Data engineering for ML (cleaning, feature engineering, validation) | 0.67 | `ml-engineer` | [`mod-103-data-engineering-for-ml`](lessons/mod-103-data-engineering-for-ml) |
| 4 | Classical ML modeling (regression, trees, ensembles, model selection) | 0.33 | `ml-engineer` | [`mod-104-classical-ml-modeling`](lessons/mod-104-classical-ml-modeling) |
| 5 | Deep learning fundamentals (PyTorch, CNN, RNN, Transformer encoder) | 0.67 | `ml-engineer` | [`mod-105-deep-learning-fundamentals`](lessons/mod-105-deep-learning-fundamentals) |
| 6 | Experiment tracking + reproducibility (MLflow / W&B) | 0.43 | `ml-engineer` | [`mod-106-experiment-tracking-reproducibility`](lessons/mod-106-experiment-tracking-reproducibility) |
| 7 | Model evaluation & validation (metrics, k-fold, baselines, error analysis) | 0.52 | `ml-engineer` | [`mod-107-model-evaluation-validation`](lessons/mod-107-model-evaluation-validation) |
| 8 | Model packaging + deployment (FastAPI, Docker, batch/online inference) | 0.76 | `ml-engineer` | [`mod-108-model-deployment-serving`](lessons/mod-108-model-deployment-serving) |
| 9 | ML-specific monitoring (data/concept drift, retraining triggers) | 0.43 | `ml-engineer` | [`mod-109-ml-monitoring-drift`](lessons/mod-109-ml-monitoring-drift) |
| 10 | ML systems design (training/serving skew, A/B testing, retraining cadence) | 0.62 | `ml-engineer` | [`mod-110-ml-systems-design`](lessons/mod-110-ml-systems-design) |
| 11 | LLM / transformer awareness (light touch — vocabulary, when to delegate) | 0.52 | `ml-engineer` (delegation only) → peer specialists for depth | [`mod-105`](lessons/mod-105-deep-learning-fundamentals) + [`mod-110`](lessons/mod-110-ml-systems-design); depth at `llm-application-developer`, `rag-engineer`, `nlp-engineer`, `fine-tuning-engineer` |
| 12 | Git / Linux / project hygiene | 0.10 | `ai-infra-junior-engineer-learning` (level 10) | Prerequisite; listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 13 | Cloud fundamentals + Kubernetes deployment | 0.38 | `ai-infra-engineer-learning` (level 20, peer) | Link-out to that curriculum; ML Engineer track touches cloud only where required to deploy a model |
| 14 | Distributed training (DDP/FSDP/Ray, multi-node GPU clusters) | 0.24 | `training-pipeline-engineer` (level 35) | Out of scope at level 20. Below threshold. Linked out. |
| 15 | Self-service ML platform engineering (feature stores, multi-tenant compute) | 0.14 | `ai-infra-ml-platform-learning` (level 30) | Below threshold. ML Engineer is a customer of these platforms, not their builder. Linked out. |
| 16 | ML/AI security depth (data poisoning, model extraction, adversarial inputs) | 0.19 | `ai-infra-security-learning` (level 35) | Below threshold. Most in-window mentions are US-clearance / DoD-domain, not ML-specific security. Surfaced as awareness in mod-110. |

### Emerging themes (below the continuity-bias gate — no action this cycle)

| # | Theme | Freq | Disposition |
|---|---|---|---|
| E1 | AI-assisted developer tools (Claude Code, Cursor, Codex) | 0.14 | Watchlist. Below 0.30 gate. If it crosses, add as an exercise in [`mod-102-python-ml-toolchain`](lessons/mod-102-python-ml-toolchain), not a new module — it's a working-style skill layered on Python fluency. |
| E2 | Agentic / orchestration frameworks (LangChain, LangGraph, AutoGen, CrewAI) | 0.24 | Watchlist. Peer-specialist territory ([`llm-application-developer-learning`](https://github.com/ai-infra-curriculum/llm-application-developer-learning)). If it crosses 0.30 next cycle, widen the delegation-map exercise in mod-105 to reference frameworks by name; do not add a module here. |
| E3 | Sub-second inference (TensorRT / vLLM / ONNX, batching, latency SLOs) | 0.14 | Already covered incrementally by [`mod-108`](lessons/mod-108-model-deployment-serving) exercise `load-test-and-tune`. Deeper hardware-aware optimization belongs to the peer inference-platform track. |
| E4 | LLM pretraining / SFT / RLHF / post-training alignment | 0.05 | Fine-tuning-engineer track — [`fine-tuning-engineer-learning`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning). Not level-20. |

## Posting evidence

The 21 postings sampled this cycle. Full quotes and per-requirement `evidence_post_ids` live in [`.aicg/job-requirements.json`](.aicg/job-requirements.json).

| ID | Employer | Title | Location | URL |
|---|---|---|---|---|
| post-001 | DoorDash | Machine Learning Engineer, Marketplace Optimization | San Francisco / Sunnyvale | [job-boards.greenhouse.io/doordashusa/jobs/7580407](https://job-boards.greenhouse.io/doordashusa/jobs/7580407) |
| post-002 | StackAdapt | Machine Learning Engineer | UK / IE / DE (remote) | [job-boards.greenhouse.io/stackadapt/jobs/4131244009](https://job-boards.greenhouse.io/stackadapt/jobs/4131244009) |
| post-003 | 10a Labs | Machine Learning Engineer | Washington D.C. | [job-boards.greenhouse.io/10alabs/jobs/4273194009](https://job-boards.greenhouse.io/10alabs/jobs/4273194009) |
| post-004 | CERN | Machine Learning Engineer (TE-DPS-AIM) | Geneva, Switzerland | [jobs.smartrecruiters.com/CERN/744000117611022](https://jobs.smartrecruiters.com/CERN/744000117611022-machine-learning-engineer-te-dps-aim-2026-94-grae-) |
| post-005 | CERN | Machine Learning Engineer (IT-CD-PI) | Geneva, Switzerland | [jobs.smartrecruiters.com/CERN/744000119112428](https://jobs.smartrecruiters.com/CERN/744000119112428-machine-learning-engineer-it-cd-pi-2026-98-grae-) |
| post-006 | JobTeaser | Machine Learning Engineer (W/M/D) | Paris, France | [jobs.smartrecruiters.com/JobTeaser/744000135987829](https://jobs.smartrecruiters.com/JobTeaser/744000135987829-machine-learning-engineer-w-m-d-) |
| post-007 | DRW | Machine Learning Engineer | Tel Aviv | [job-boards.greenhouse.io/drweng/jobs/6242898](https://job-boards.greenhouse.io/drweng/jobs/6242898) |
| post-008 | CoVar | Machine Learning Engineer | Durham, NC | [job-boards.greenhouse.io/covar/jobs/5097883007](https://job-boards.greenhouse.io/covar/jobs/5097883007) |
| post-009 | Intuitive | Machine Learning Engineer | Sunnyvale, CA | [jobs.smartrecruiters.com/Intuitive/744000118228852](https://jobs.smartrecruiters.com/Intuitive/744000118228852-machine-learning-engineer) |
| post-010 | Visa | Machine Learning Engineer | Foster City, CA (hybrid) | [jobs.smartrecruiters.com/Visa/744000109087615](https://jobs.smartrecruiters.com/Visa/744000109087615-machine-learning-engineer) |
| post-011 | Adyen | Machine Learning Engineer | Bengaluru | [job-boards.greenhouse.io/adyen/jobs/7349167](https://job-boards.greenhouse.io/adyen/jobs/7349167) |
| post-012 | Vannevar Labs | Machine Learning Engineer | Remote (US) | [job-boards.greenhouse.io/vannevarlabs/jobs/5045652007](https://job-boards.greenhouse.io/vannevarlabs/jobs/5045652007) |
| post-013 | Tapestry (Alphabet) | Machine Learning Engineer, Tapestry | Mountain View, CA | [job-boards.greenhouse.io/tapestryenergy/jobs/8633983002](https://job-boards.greenhouse.io/tapestryenergy/jobs/8633983002) |
| post-014 | CLEAR | Machine Learning Engineer II | New York, NY | [job-boards.greenhouse.io/clear/jobs/8070543](https://job-boards.greenhouse.io/clear/jobs/8070543) |
| post-015 | Stripe | Machine Learning Engineer | Toronto | [stripe.com/careers/listing/machine-learning-engineer/8014859](https://stripe.com/careers/listing/machine-learning-engineer/8014859) |
| post-016 | Affirm | Machine Learning Engineer II (Servicing ML) | Remote Canada | [job-boards.greenhouse.io/affirm/jobs/7719651003](https://job-boards.greenhouse.io/affirm/jobs/7719651003) |
| post-017 | Affirm | Machine Learning Engineer II (Servicing ML) | Remote US | [job-boards.greenhouse.io/affirm/jobs/7719653003](https://job-boards.greenhouse.io/affirm/jobs/7719653003) |
| post-018 | May Mobility | Machine Learning Engineer II — Training Infrastructure | Remote USA | [job-boards.greenhouse.io/maymobility/jobs/8580411002](https://job-boards.greenhouse.io/maymobility/jobs/8580411002) |
| post-019 | May Mobility | Machine Learning Engineer II — Performance Evaluation | Remote USA | [job-boards.greenhouse.io/maymobility/jobs/8187068002](https://job-boards.greenhouse.io/maymobility/jobs/8187068002) |
| post-020 | Twilio | Machine Learning Engineer | Remote Spain | [job-boards.greenhouse.io/twilio/jobs/8007455](https://job-boards.greenhouse.io/twilio/jobs/8007455) |
| post-021 | Applied Materials | Machine Learning Engineer (Generative AI) | Santa Clara, CA | [jobs.appliedmaterials.com/…/91326559808](https://jobs.appliedmaterials.com/job/santa-clara/machine-learning-engineer-generative-ai/95/91326559808) |

## Ownership map — quick reference

- **ML Engineer (this track, level 20)** owns the build-altitude foundation: data → feature engineering → model → evaluation → deploy → monitor → iterate.
- **Senior / Staff / Principal ML Engineer** (levels 30 / 40 / 48) inherit this curriculum and add depth, architecture, and leadership framing.
- **AI Infrastructure Engineer** (level 20, peer) owns Docker, Kubernetes, cloud, CI/CD plumbing.
- **AI Infrastructure MLOps Engineer** (`ai-infra-mlops-learning`) owns deep MLOps automation (Kubeflow / Argo / Airflow orchestration engineering beyond "use it correctly").
- **AI Infrastructure ML Platform Engineer** (level 30) owns self-service platforms — feature stores, managed-service depth (SageMaker / Vertex AI / Azure ML).
- **Training Pipeline Engineer** (level 35) owns distributed training infrastructure (DDP / FSDP / DeepSpeed / Ray).
- **Model Evaluation Engineer / AI Evaluation Engineer** own evaluation engineering depth (LLM-as-judge, DeepEval-style production eval tooling).
- **NLP / RAG / LLM Application / Fine-Tuning / Applied AI Engineers** own their respective specializations, including agentic frameworks (LangChain / LangGraph / AutoGen / CrewAI).
- **AI Infrastructure Security Engineer** (level 35) owns ML/AI security depth.
- **AI Governance Analyst** owns governance and compliance depth.

## Conclusion

The 21-posting evidence base validates the existing 10-module / 3-project curriculum without gaps. Frequencies for the four out-of-scope requirements (distributed training, ML platform engineering, deep security, git/Linux prerequisites) all fell below the level-20 threshold as expected. Three emerging themes (AI coding tools, agentic frameworks, low-latency inference) are on the watchlist but below the additive gate; two are peer-owned and one is already incrementally covered.

<!-- needs-research: next cycle, retry Lever (browser-capable fetcher) and Ashby (headless render) to close the 21→25 gap; re-check emerging theme E2 (agentic frameworks) — 0.24 is close enough to 0.30 that one more posting cycle of growth would trigger a peer-track handoff proposal. -->
