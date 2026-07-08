# Machine Learning Engineer Curriculum

**Role level:** 20 (build-altitude, mid-career foundation)
**Status:** planned — modules and projects below are the planned scope authored from [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json). Lessons and projects will be drafted by subsequent autonomous content cycles.

## Overview

This track teaches the end-to-end practitioner workflow of a mid-career Machine Learning Engineer: frame a problem → engineer data → train and tune a model → evaluate it honestly → deploy it → monitor it → iterate. It is the **foundation track** that the senior, staff, and principal Machine Learning Engineer tracks (and several peer specialist tracks) inherit and extend.

Total planned commitment: **136 hours** across 10 modules + **125 hours** across 3 projects = **~261 hours**.

## Ownership rule

Following the project-wide ownership rule, this curriculum:

- **Owns** the build-altitude ML practitioner workflow end-to-end.
- **Defers up** to `senior-ml-engineer-learning` / `staff-ml-engineer-learning` / `principal-ml-engineer-learning` for architectural and leadership depth.
- **Defers sideways** to peer specialist tracks for depth in their domain — `nlp-engineer`, `rag-engineer`, `llm-application-developer`, `fine-tuning-engineer`, `model-evaluation-engineer`, `ai-eval-engineer`, `training-pipeline-engineer`, `applied-ai-engineer`, `ai-infra-mlops-learning`, `ai-infra-ml-platform-learning`.
- **Defers down** to `ai-infra-junior-engineer-learning` for the engineering-craft prerequisites (Git, Linux, Python project hygiene).
- **Links out** to `ai-infra-engineer-learning` (peer at level 20) for Docker/Kubernetes/cloud depth, to `ai-infra-security-learning` for ML/AI security depth, and to `ai-governance-analyst` for governance depth.

See [`JOB_REQUIREMENTS.md`](JOB_REQUIREMENTS.md) for the requirements-to-coverage map and the cited public references the catalog is grounded in.

## Module Plan

| Module | Title | Hours | Status |
|---|---|---|---|
| mod-101-ml-foundations | ML Foundations: Math, Statistics, and Problem Framing | 14 | planned |
| mod-102-python-ml-toolchain | Python ML Toolchain: NumPy, Pandas, scikit-learn, Jupyter | 12 | planned |
| mod-103-data-engineering-for-ml | Data Engineering for ML | 14 | planned |
| mod-104-classical-ml-modeling | Classical ML Modeling: Regression, Trees, Ensembles | 16 | planned |
| mod-105-deep-learning-fundamentals | Deep Learning Fundamentals with PyTorch | 18 | planned |
| mod-106-experiment-tracking-reproducibility | Experiment Tracking & Reproducibility | 10 | planned |
| mod-107-model-evaluation-validation | Model Evaluation & Validation | 12 | planned |
| mod-108-model-deployment-serving | Model Packaging & Deployment | 14 | planned |
| mod-109-ml-monitoring-drift | ML Monitoring & Drift Detection | 12 | planned |
| mod-110-ml-systems-design | ML Systems Design | 14 | planned |

## Project Plan

| Project | Title | Hours | Status |
|---|---|---|---|
| project-101-tabular-ml-end-to-end | End-to-End Tabular ML Project | 40 | planned |
| project-102-deep-learning-vision-project | Deep Learning Vision Project | 35 | planned |
| project-103-ml-system-capstone | ML System Capstone: Monitored Production Service | 50 | planned |

## Module summaries

### mod-101 — ML Foundations
Linear algebra, probability/statistics, optimization intuition, and the framing skill that decides whether to model a problem as regression, classification, or ranking.

### mod-102 — Python ML Toolchain
NumPy / Pandas / scikit-learn fluency, reproducible Python environments, and unit-testing for ML code.

### mod-103 — Data Engineering for ML
Ingestion from SQL/object stores, data audits, feature engineering for tabular / text / image data, schema and distribution validation.

### mod-104 — Classical ML Modeling
Regression with regularization, decision trees, random forests, gradient-boosted trees (XGBoost / LightGBM), k-fold cross-validation, hyperparameter search (Optuna), SHAP interpretability.

### mod-105 — Deep Learning Fundamentals
A from-scratch PyTorch training loop, MLP / CNN / small Transformer encoder, regularization, mixed-precision training, and the vocabulary needed to know when to delegate to the LLM / NLP / RAG / fine-tuning specialist tracks.

### mod-106 — Experiment Tracking & Reproducibility
MLflow (with a W&B side-by-side), seed control, environment pinning, dataset + model versioning, a small experiment sweep and reproducibility report.

### mod-107 — Model Evaluation & Validation
Metric selection by task, leakage-resistant validation (time/group/nested), informative baselines, error analysis, and lightweight fairness/robustness slicing.

### mod-108 — Model Packaging & Deployment
FastAPI + Docker, health checks, structured logging, batch vs. online inference, load-test-and-tune; hand-off interface to the AI Infrastructure Engineer track for Kubernetes deployment.

### mod-109 — ML Monitoring & Drift Detection
ML-specific signals beyond generic observability, data/concept/prediction drift, retraining triggers, ground-truth-collection loops, and a worked production-regression postmortem.

### mod-110 — ML Systems Design
Business goal → ML system blueprint, training/serving skew, data flywheel design, A/B testing with guardrails, build-vs-buy decisions, and the delegation map to peer specialist tracks.

## Assessment

Each module ships **1 quiz** plus the exercises and a lab listed in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json). Each project ships a portfolio-grade README and an explicit assessment rubric covering functionality, code quality, evaluation, deployment, monitoring, and documentation.

## Where to go after this curriculum

- **`senior-ml-engineer-learning`** — next-up role on the ML engineering ladder.
- **`ai-infra-engineer-learning`** — peer track that owns the Docker / Kubernetes / cloud / CI-CD plumbing this curriculum hands off to.
- **`ai-infra-mlops-learning`** — peer track for deep MLOps automation.
- **`ai-infra-ml-platform-learning`** — peer track at level 30 for self-service ML platform engineering.
- **`nlp-engineer-learning` / `rag-engineer-learning` / `llm-application-developer-learning` / `fine-tuning-engineer-learning`** — peer specialist tracks for LLM / NLP / RAG / fine-tuning depth.
- **`model-evaluation-engineer-learning` / `ai-eval-engineer-learning`** — peer specialist tracks for evaluation engineering depth.
- **`training-pipeline-engineer-learning`** — peer specialist track for distributed training depth.
- **`ai-infra-security-learning`** — for ML/AI security depth.

<!-- needs-research: backfill industry-frequency evidence into JOB_REQUIREMENTS.md once the autonomous research loop runs with WebSearch / WebFetch permissions; demote any module or exercise whose underlying requirement does not show up in ≥3 in-window postings. -->
