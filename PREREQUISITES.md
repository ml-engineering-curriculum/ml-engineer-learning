# Prerequisites for the Machine Learning Engineer track

This curriculum starts at **level 20 (build-altitude, mid-career foundation)**. Before working through it, you should already be comfortable with the items below. The level-10 [`ai-infra-junior-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-learning) track is the canonical place to acquire them.

## Required

- **Python (intermediate)** — functions, classes, modules, virtual environments, packaging with `pip`/`uv`/`poetry`, basic type hints, structured logging, basic unit testing.
- **Linux command line** — file system navigation, processes, permissions, shell scripting basics.
- **Git** — clone, branch, commit, push, pull, resolve a merge conflict, write a clean PR description.
- **SQL fundamentals** — `SELECT` / `JOIN` / `GROUP BY` / window functions, reading an `EXPLAIN`.
- **Basic statistics** — mean, median, variance, distributions, hypothesis testing intuition. This curriculum extends but does not re-teach the high-school / first-year-university level basics.
- **Basic linear algebra** — vectors, matrices, dot products, matrix multiplication.
- **REST APIs** — understand request/response, JSON, status codes, calling an HTTP API from Python.
- **HTTP, TCP/IP, DNS** at a working level.

## Recommended

- One prior end-to-end machine learning project (Kaggle competition, course capstone, hobby model). You can complete the curriculum without one, but `mod-104` and the projects will be easier to internalise if you already have the experience of going from CSV to "a model that does something."
- Exposure to a major cloud provider (AWS / GCP / Azure) at the level of "I have spun up an EC2 / Compute Engine instance and SSH'd into it."
- Docker basics — pull, build, run. Module 108 covers this within the ML packaging context, but a head start helps.

## Out of scope as prerequisites

You do **not** need to know any of the following before starting; they are either taught in the curriculum or owned by a peer/upstream track:

- PyTorch / scikit-learn / MLflow — taught here.
- Kubernetes, deep cloud, CI/CD plumbing — owned by [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) (peer track at level 20). The ML Engineer track links out to it.
- Deep MLOps automation — owned by [`ai-infra-mlops-learning`](https://github.com/ai-infra-curriculum/ai-infra-mlops-learning).
- LLMs / RAG / fine-tuning / NLP depth — owned by [`llm-application-developer-learning`](https://github.com/ml-engineering-curriculum/llm-application-developer-learning), [`rag-engineer-learning`](https://github.com/ml-engineering-curriculum/rag-engineer-learning), [`nlp-engineer-learning`](https://github.com/ml-engineering-curriculum/nlp-engineer-learning), [`fine-tuning-engineer-learning`](https://github.com/ml-engineering-curriculum/fine-tuning-engineer-learning).
- Distributed training infrastructure — owned by [`training-pipeline-engineer-learning`](https://github.com/ml-engineering-curriculum/training-pipeline-engineer-learning).
- Deep evaluation engineering — owned by [`model-evaluation-engineer-learning`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) and [`ai-eval-engineer-learning`](https://github.com/ml-engineering-curriculum/ai-eval-engineer-learning).
- ML/AI security and governance depth — owned by [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning) and [`ai-governance-analyst-learning`](https://github.com/ml-engineering-curriculum/ai-governance-analyst-learning).

## Hardware

A laptop with 16 GB RAM is enough for Modules 101–104 and 106–110. Module 105 (Deep Learning Fundamentals) and Project 102 (Deep Learning Vision Project) benefit from a GPU — a free-tier Colab / Kaggle notebook is sufficient if you do not have a local one. No multi-GPU or distributed setup is required at this level.

<!-- needs-research: confirm the prerequisite list against the live posting evidence on the next autonomous research cycle. -->
