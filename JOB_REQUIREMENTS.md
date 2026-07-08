# Job Requirements — Machine Learning Engineer

**Role level:** 20 (build altitude, mid-career)
**Track:** `ml-engineer-learning`
**Research window:** 2026-03-27 → 2026-06-25 (last 90 days)
**Today:** 2026-06-25

This file documents the requirements catalog used to seed the Machine Learning Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json).

## Status — bootstrap session, postings deferred

<!-- needs-research: collect ≥25 distinct in-window postings titled "Machine Learning Engineer" (NOT senior/staff/principal/platform/MLOps) and re-validate every requirement against the live evidence. -->

This packet was authored in a bootstrap session **without WebSearch / WebFetch permissions**. Per the project rules ("Do not invent facts, incidents, or salary figures. Cite sources."), the `postings` array in `.aicg/job-requirements.json` is intentionally empty: the curriculum cannot honestly claim to have analysed 25 live postings when none were fetched.

The autonomous research loop is expected to fill that gap on its next cycle. To make that loop deterministic, this document instead grounds the requirements catalog in **authoritative public references** that publish what the role is hired against (certification exam guides, canonical books, official documentation). Every requirement below cites at least one such reference, and every requirement is shaped so that posting-frequency evidence can be added underneath it without restructure.

## Methodology

1. Sourced the canonical task domains for a mid-level Machine Learning Engineer from public references — see `authoritative_references` in `.aicg/job-requirements.json`:
   - AWS Certified Machine Learning Engineer – Associate exam guide
   - Google Cloud Professional Machine Learning Engineer exam guide
   - Microsoft Azure DP-100 exam guide
   - *Designing Machine Learning Systems* — Chip Huyen (O'Reilly, 2022)
   - *Machine Learning Engineering* — Andriy Burkov (2020)
   - Google Developers, *Rules of Machine Learning*
   - Kreuzberger et al., *Machine Learning Operations (MLOps)* (arXiv:2205.02302)
   - DeepLearning.AI / Coursera *ML Engineering for Production* specialization
   - Official PyTorch, scikit-learn, MLflow documentation
2. Mapped each task domain to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that covers it.
3. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill, with higher levels linking back rather than duplicating fundamentals. Peer specialist tracks (NLP / RAG / Fine-tuning / LLM Application / Evaluation / Training Pipeline / MLOps / ML Platform) own depth in their domain.
4. Flagged everything that has not yet been validated against in-window postings so the next cycle can demote any requirement whose evidence stays empty.

## Requirement themes → curriculum ownership

The table below lists each requirement theme, its planned owner per the level hierarchy, and the curriculum coverage path. **Freq** is intentionally blank for this cycle — it will be backfilled by the next research pass.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Math, statistics, optimization foundations | <!-- needs-research --> | `ml-engineer` (this) | [`mod-101-ml-foundations`](lessons/mod-101-ml-foundations) |
| 2 | Python + ML toolchain (NumPy, Pandas, scikit-learn, Jupyter) | <!-- needs-research --> | `ml-engineer` | [`mod-102-python-ml-toolchain`](lessons/mod-102-python-ml-toolchain) |
| 3 | Data engineering for ML (cleaning, feature engineering, validation) | <!-- needs-research --> | `ml-engineer` | [`mod-103-data-engineering-for-ml`](lessons/mod-103-data-engineering-for-ml) |
| 4 | Classical ML modeling (regression, trees, ensembles, model selection) | <!-- needs-research --> | `ml-engineer` | [`mod-104-classical-ml-modeling`](lessons/mod-104-classical-ml-modeling) |
| 5 | Deep learning fundamentals (PyTorch, CNN, RNN, Transformer encoder) | <!-- needs-research --> | `ml-engineer` | [`mod-105-deep-learning-fundamentals`](lessons/mod-105-deep-learning-fundamentals) |
| 6 | Experiment tracking + reproducibility (MLflow / W&B) | <!-- needs-research --> | `ml-engineer` | [`mod-106-experiment-tracking-reproducibility`](lessons/mod-106-experiment-tracking-reproducibility) |
| 7 | Model evaluation & validation (metrics, k-fold, baselines, error analysis) | <!-- needs-research --> | `ml-engineer` | [`mod-107-model-evaluation-validation`](lessons/mod-107-model-evaluation-validation) |
| 8 | Model packaging + deployment (FastAPI, Docker, batch/online inference) | <!-- needs-research --> | `ml-engineer` | [`mod-108-model-deployment-serving`](lessons/mod-108-model-deployment-serving) |
| 9 | ML-specific monitoring (data/concept drift, retraining triggers) | <!-- needs-research --> | `ml-engineer` | [`mod-109-ml-monitoring-drift`](lessons/mod-109-ml-monitoring-drift) |
| 10 | ML systems design (training/serving skew, A/B testing, retraining cadence) | <!-- needs-research --> | `ml-engineer` | [`mod-110-ml-systems-design`](lessons/mod-110-ml-systems-design) |
| 11 | LLM / transformer awareness (light touch — vocabulary, when to delegate) | <!-- needs-research --> | `ml-engineer` (light) → peer specialists for depth | [`mod-105`](lessons/mod-105-deep-learning-fundamentals) + [`mod-110`](lessons/mod-110-ml-systems-design); depth at `llm-application-developer`, `rag-engineer`, `nlp-engineer`, `fine-tuning-engineer` |
| 12 | Git / Linux / project hygiene | n/a — prerequisite | `ai-infra-junior-engineer-learning` (level 10) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 13 | Cloud fundamentals + Kubernetes deployment | n/a — peer track | `ai-infra-engineer-learning` (level 20, peer) | Link-out to that curriculum; the ML Engineer track touches cloud only where strictly required to deploy a model |
| 14 | Distributed training (DDP/FSDP/Ray, multi-node GPU clusters) | n/a — out of scope | `training-pipeline-engineer` | Out of scope at level 20. Linked out. |
| 15 | Self-service ML platform engineering (feature stores, multi-tenant compute) | n/a — peer track | `ai-infra-ml-platform-learning` (level 30) | The ML Engineer is a customer of these platforms, not their builder. Linked out. |
| 16 | Deep MLOps / CI-CD-for-ML platform engineering | n/a — peer track | `ai-infra-mlops-learning` | Module 106 covers MLOps foundations; deep automation/platforms linked out. |
| 17 | ML/AI security depth (data poisoning, model extraction, adversarial inputs) | n/a — higher level | `ai-infra-security-learning` (level 35) | Surfaced as awareness in mod-110; depth owned upstream. |
| 18 | Responsible AI / governance reviews | n/a — peer track | `ai-governance-analyst` | Surfaced as awareness in mod-110; depth owned upstream. |

## Posting evidence

<!-- needs-research: populate with the ≥25 in-window postings sampled next cycle. Use the same table shape as ai-infra-agentic-ai-engineer-learning/JOB_REQUIREMENTS.md. -->

No postings were sampled this cycle. See the **Status** section above for the reason. The next autonomous research cycle should fan out across `job-boards.greenhouse.io`, `jobs.lever.co`, `jobs.ashbyhq.com`, and major employer ATS roots (Stripe, OpenAI, Anthropic, Databricks, Scale AI, Pinterest, Airbnb, DoorDash, Instacart, Robinhood, SoFi, Mercury, Plaid, Brex, Etsy, Niantic, Riot, Spotify, Spotify Research, plus healthtech/biotech: Recursion, Tempus, Flatiron, Hims, Ro, Oscar; plus retail/adtech: Wayfair, Wayfair Research, Walmart Global Tech, Teads, The Trade Desk).

For each posting, capture employer, exact title, URL, date_observed, date_posted (or `estimated:2026-MM`), location, 5–10 verbatim requirement bullets, 2–6 preferred-qualification bullets, salary range when published, and one short representative quote. Filter out Senior / Staff / Principal / Platform / MLOps / Research Scientist / Data Scientist / pure NLP / pure LLM-application titles — those belong to other tracks.

## Ownership map — quick reference for next cycle

When backfilling postings, use this ownership decision to keep the curriculum from drifting into peer territory:

- **ML Engineer (this track, level 20)** owns the build-altitude foundation: data → feature engineering → model → evaluation → deploy → monitor → iterate.
- **Senior / Staff / Principal ML Engineer** (levels 30 / 40 / 50) inherit this curriculum and add depth, architecture, and leadership framing.
- **AI Infrastructure Engineer** (level 20, peer) owns Docker, Kubernetes, cloud, CI/CD plumbing.
- **AI Infrastructure MLOps Engineer** (`ai-infra-mlops-learning`) owns deep MLOps automation.
- **AI Infrastructure ML Platform Engineer** (level 30) owns self-service platforms.
- **Training Pipeline Engineer** owns distributed training infrastructure.
- **Model Evaluation Engineer / AI Evaluation Engineer** own evaluation engineering depth.
- **NLP / RAG / LLM Application / Fine-Tuning / Applied AI Engineers** own their respective specializations.
- **AI Infrastructure Security Engineer** (level 35) owns ML/AI security depth.
- **AI Governance Analyst** owns governance and compliance depth.

## Conclusion

<!-- needs-research: re-run on the next autonomous cycle with web tools enabled, populate `postings` in .aicg/job-requirements.json, backfill the Freq column above, and demote any requirement whose evidence_post_ids stays empty. -->

The curriculum plan in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json) is structured so that requirement frequencies can be added underneath each module without restructure. The themes themselves are grounded in cited public references, not invented; they are the standard task domains hired against in industry training programs and certification guides.
