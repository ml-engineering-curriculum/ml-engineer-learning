# mod-108-model-deployment-serving: Model Packaging & Deployment

**Estimated effort:** 14 hours.

## Learning objectives

- Package a trained model behind a FastAPI service with health
  checks and structured logs.
- Containerise the service with Docker and run it locally and
  in CI.
- Implement batch and online inference paths and reason about
  latency vs. throughput.
- Hand off to the AI Infrastructure Engineer / ML Platform
  tracks for Kubernetes deployment.

## Chapters

1. [Why serving is its own discipline](01-why-serving-is-its-own-discipline.md)
2. [Packaging the model artifact](02-packaging-the-model-artifact.md)
3. [Anatomy of a FastAPI prediction service](03-fastapi-prediction-service.md)
4. [Health checks and the service lifecycle](04-health-checks-and-lifecycle.md)
5. [Structured logging and request observability](05-structured-logging-and-observability.md)
6. [Containerising the service with Docker](06-containerising-with-docker.md)
7. [Running the container locally and in CI](07-running-in-ci.md)
8. [Batch vs online inference](08-batch-vs-online-inference.md)
9. [Latency, throughput, and concurrency](09-latency-throughput-and-concurrency.md)
10. [Load testing and tuning](10-load-testing-and-tuning.md)
11. [Handoff to the platform team](11-handoff-to-platform.md)

## Exercises

- [exercise-01 — FastAPI prediction service](exercises/exercise-01-fastapi-prediction-service.md)
- [exercise-02 — Dockerise the service](exercises/exercise-02-dockerise-the-service.md)
- [exercise-03 — Batch vs online inference](exercises/exercise-03-batch-vs-online-inference.md)
- [exercise-04 — Load test and tune](exercises/exercise-04-load-test-and-tune.md)

## Structure

- `01-…md` … `11-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.

## Where this sits in the track

- **Input.** A trained, validated model from `mod-104` /
  `mod-105`, an experiment run from `mod-106`, and defensible
  evaluation numbers from `mod-107`. You have an artifact; this
  module makes it callable.
- **Output.** A containerised HTTP service with a health check,
  structured logs, and a defensible latency/throughput budget.
  `mod-109` (monitoring) instruments what happens after it goes
  live; `mod-110` (systems design) covers how it composes with
  the rest of the ML platform.

## Non-goals

Kubernetes, service meshes, autoscaling, canary rollouts, and
GPU inference infrastructure belong to the **AI Infrastructure
Engineer** and **ML Platform Engineer** tracks. Chapter 11
draws the boundary explicitly.
