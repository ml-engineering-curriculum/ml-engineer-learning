# 11 — Handoff to the platform team

> **Reading time:** ~20 minutes.
> **Prerequisite chapters:** 01–10.

## Motivation

You have a containerised FastAPI service with health checks,
structured logs, a load-tested latency budget, and a batch
scoring job. It runs on your workstation and in CI. **Now
what?**

In a real organisation, the answer is: **hand it off to a
platform team**. That team runs Kubernetes (or ECS, or
Vertex AI, or SageMaker); they own the ingress, the
autoscaling, the secrets management, the observability
stack, and the deploy pipeline. Your service goes into
their platform. The handoff is a contract.

The AI Infrastructure Engineer and ML Platform Engineer
tracks in this curriculum are the *other side* of the
handoff — the people who consume the artifact you produced
in this module. This chapter is the deliverables list, the
contract, and the discussion the two sides should have.

Get the handoff right and the platform team can deploy your
service in an afternoon. Get it wrong and the service sits
in a queue for weeks while the two teams negotiate
"can it be autoscaled? does it drain cleanly? what does
readyz check?" one email at a time.

## The one-page handoff document

Every deployable service should ship with a short
`RUNBOOK.md` in the repo. The platform team reads this
before they touch the service. The template:

```markdown
# churn-service — RUNBOOK

## Service summary
One paragraph: what does this service do, who calls it, what
happens when it is down?

## Image
Registry: ghcr.io/example/churn-service
Tag policy: git SHA on each merge to main
Base image: python:3.11-slim
Runs as: non-root UID 1000
Port: 8080

## Endpoints
- POST /predict         — single prediction
- POST /predict:batch   — batch (max 1000 rows)
- GET  /healthz         — liveness (trivial)
- GET  /readyz          — readiness (model loaded + warm)
- GET  /metrics         — Prometheus
- GET  /model           — model metadata (name, version, git SHA)
- GET  /docs            — OpenAPI UI (disable in prod?)

## Startup
- Cold start: ~15 s (model load + warm-up)
- Recommended startupProbe: initialDelay=10, periodSeconds=5,
  failureThreshold=12 (60 s budget)

## Shutdown
- SIGTERM triggers graceful drain
- readyz flips to 503 immediately
- Configure preStop: sleep 15 to let the LB de-register
- gunicorn --graceful-timeout 30

## Resources
- CPU request: 500m, limit: 2
- Memory request: 1Gi, limit: 2Gi
- No GPU

## Capacity
- Load test: 400 QPS at p95 ≤ 100 ms with 8 workers on
  4 vCPU / 4 GiB
- Recommended max per pod: 300 QPS
- Recommended HPA metric: CPU 70%

## Configuration (env vars)
- ARTIFACT_DIR     — path to model dir            default /artifact
- LOG_LEVEL        — INFO / DEBUG                default INFO
- WORKERS          — gunicorn workers            default 4
- GRACEFUL_TIMEOUT_S — shutdown drain window     default 30

## Secrets
- None. The service does not need any secrets to start.
  (If model loads from S3, add: AWS_ACCESS_KEY_ID etc.)

## Model artifact
- Baked into the image at /artifact
- Metadata at /artifact/metadata.json
- Rebuild the image to change model versions

## Observability
- Logs: JSON on stdout; fields: ts, level, event, request_id,
  model_version, git_sha, latency_ms
- Metrics: predictions_total (counter),
  prediction_latency_seconds (histogram)
- Alerts owned by the ML team on:
    * availability < 99.5% over 5m
    * p95 latency > 200 ms over 5m

## SLOs
- Availability: 99.5% (30-day)
- Latency: p95 ≤ 100 ms at ≤ 300 QPS

## Runbook: common failures
| Symptom              | Likely cause              | Action                              |
|----------------------|---------------------------|-------------------------------------|
| Pod CrashLoopBackOff | Model load fails          | Check logs; verify /artifact mounted |
| readyz never 200     | Model warm-up failing     | Check logs; verify input schema      |
| p95 spike            | Feature store slow        | Check downstream; page mod-103 owner |
| OOMKilled            | Batch endpoint OOM        | Reduce max batch or increase memory  |

## Contacts
- Owner: ml-eng-team@example.com
- On-call: PagerDuty rotation ml-services
- Escalation: engineering-manager@example.com
```

This is the minimum. Nothing in it should be a surprise to
the platform team; every field maps onto something they
already have to know.

## What the ML engineer owns

You (this module) own **the service inside the container**:

- The model artifact and its metadata (chapter 02).
- The FastAPI app and its request/response schema
  (chapter 03).
- The health check semantics (chapter 04).
- Structured logs and the metrics surface (chapter 05).
- The Dockerfile and image contents (chapter 06).
- CI: unit tests, integration tests, image build
  (chapter 07).
- The batch scoring job (chapter 08).
- A defensible latency-throughput budget (chapters 09–10).
- The RUNBOOK.md that describes all of the above.
- Model-quality alerts (drift, staleness) that require
  ML judgment.

Nothing above requires Kubernetes knowledge. It is all
Python + Docker + a service contract.

## What the platform team owns

They own **everything outside the container** on the way to
production:

- Kubernetes manifests (Deployment, Service, Ingress,
  HPA, PDB).
- Namespace, RBAC, service accounts.
- Secrets management (Vault, SealedSecrets, cloud KMS).
- Autoscaling — the HPA that matches your capacity story.
- The service mesh, mTLS, network policies.
- Ingress routing, TLS termination, WAF.
- The deploy pipeline — canary, blue/green, rollback.
- Cluster-level observability — Prometheus, Loki, Grafana,
  tracing.
- Cluster capacity planning and node lifecycle.
- Cluster-level alerts (node down, network partition,
  OOMKill counts, control plane health).

You know these exist. You produce a service that plays
nicely with them. You do not (usually) implement them.

## What sits at the seam

Some responsibilities are joint. They are the most common
source of arguments; agree explicitly:

- **The image registry.** Who pushes, who pulls, who
  cleans up old tags? Usually the platform team runs the
  registry and provides push credentials to your CI.
- **The tag policy.** Immutable tags (git SHA per merge)
  vs. mutable tags (`latest`, `prod`). Immutable-only is
  strongly recommended; a rollback should point at a
  specific SHA.
- **Base image maintenance.** Someone has to update
  `python:3.11-slim` when a CVE lands. Often the platform
  team maintains a curated base image; your Dockerfile
  `FROM`s theirs.
- **Where the model artifact lives.** Baked in vs. object
  storage (chapter 06). If object storage, the platform
  team provisions the bucket and grants read access to
  the service account.
- **Configuration drift.** Env vars vs. a ConfigMap; the
  ConfigMap can be edited independently of a deploy. Who
  can edit which?
- **Alert ownership.** Model-quality alerts (drift,
  accuracy degradation) belong to ML; infra-level alerts
  (CPU, pod restarts) belong to platform. Draw the line
  in the runbook.

## Things the platform team will do — and what you owe them

### Autoscaling

Kubernetes' Horizontal Pod Autoscaler scales the number of
replicas based on a metric (CPU utilisation, custom
metrics, or QPS-like signals). Your side of the contract:

- **A capacity story.** Chapter 10's latency-throughput
  curve. "Each pod sustains X QPS at p95 ≤ Y ms."
- **A scaling metric that makes sense.** CPU is the default
  because it's already collected; QPS-per-pod (custom
  metric) is often more direct. If prediction is very
  cache-friendly (chapter 08's pre-compute pattern), CPU
  underestimates load.
- **A fast, cheap `/readyz`.** Autoscaling adds pods; those
  pods need to be ready before they can absorb traffic.
  Slow readyz = slow scale-out = SLO burn during traffic
  spikes.

### Rolling deploys

The platform team ships a new image by starting new pods,
waiting for readyz, then draining old pods. Your side:

- **Correct readyz semantics.** Chapter 04. Do not accept
  traffic before warm-up.
- **Graceful shutdown.** Chapter 04. In-flight requests
  finish before exit.
- **Backwards-compatible schema changes.** During a
  rolling deploy, both old and new pods serve at once.
  A response schema change that removes a field breaks
  callers pinned to the old shape. Versioned routes
  (`/v1/predict`, `/v2/predict`) or additive-only changes.

### Canary and blue/green

Some platforms roll new versions gradually — 1% of traffic,
then 5%, then 25%. Your side:

- **`/model` endpoint** so the platform can verify version
  splits.
- **`model_version` on every response** so downstream
  consumers can distinguish canary from stable.
- **Metrics labelled by `model_version`** so the platform
  can compare canary vs. stable p95 automatically.

### Multi-tenancy

Many services on shared nodes. Your side:

- **Resource requests / limits.** Chapter 06 mentioned
  them; the RUNBOOK states them. Under-request and the
  pod gets scheduled poorly; over-request and the platform
  wastes cluster capacity.
- **CPU and memory profile that matches the request.** If
  you asked for 500m and use 2 cores, you are stealing
  from other services.

### Kubernetes-shaped observability

The platform team runs the Prometheus and log aggregators.
Your side is the shape:

- **Metric names, labels, and buckets** stable across
  deploys. Renaming `predictions_total` to
  `prediction_count_total` breaks every dashboard and
  alert.
- **Log fields** stable across deploys. Downstream ELK/
  Loki queries pin against field names.
- **Cardinality discipline.** Do not label metrics with
  `user_id` or `request_id`. Prometheus buckets explode;
  the platform team pages you.

## Things you own that the platform will not

Some things are ML-native and the platform team cannot do
for you:

- **Model-quality monitoring.** `mod-109` is the whole
  module. Data drift, prediction drift, concept drift,
  ground-truth-delayed accuracy — all of these are ML
  problems the platform's Prometheus does not know how to
  measure.
- **Retraining pipelines and model promotion.** MLflow
  registry, DVC datasets, retraining triggers.
- **Feature-store schema evolution.** New feature added
  to the feature store; retrain the model; new feature
  appears in the schema; the service must handle both
  during rollout.
- **Prediction quality investigations.** "Users are
  complaining the recommendations are worse this week."
  Platform observability does not see this.

Do not push these onto the platform team. Own them.

## Kubernetes 101 — what to skim before the handoff meeting

You do not need to know Kubernetes deeply. You should
recognise the vocabulary the platform team uses:

- **Pod** — a set of one or more containers running
  together. Your service is (usually) one container per
  pod.
- **Deployment** — a controller that keeps N replicas of a
  pod running. What the platform team creates from your
  image.
- **Service** — an internal DNS name and IP that load-
  balances across pods.
- **Ingress** — how external HTTP traffic reaches your
  Service. Provides TLS, hostname routing.
- **HPA (HorizontalPodAutoscaler)** — the autoscaler.
  Consumes your latency-throughput budget.
- **Probes** — liveness, readiness, startup. Your `/healthz`,
  `/readyz`, `/startupz` endpoints from chapter 04.
- **ConfigMap / Secret** — mounted config or env vars.
  Platform-managed; your service consumes.
- **PDB (PodDisruptionBudget)** — "at most K replicas can
  be down at once". Platform sets it.
- **preStop hook** — command run before SIGTERM. The
  `sleep 15` from chapter 04 lives here.

For depth, the AI Infrastructure Engineer track and the ML
Platform Engineer track go deep. This module gives you
enough to be a good customer, not a Kubernetes operator.

## The onboarding conversation — a checklist

The first time you hand a service to a new platform team:

1. **Walk through the RUNBOOK together.** Get them to nod
   at every section. Wherever they push back is where the
   contract needs alignment.
2. **Show the load-test results.** They translate directly
   to the HPA configuration.
3. **Discuss the tag policy.** Immutable per-SHA is the
   default; agree on the tag format.
4. **Agree on the registry.** Push credentials from CI.
5. **Discuss secrets.** Which secrets, where they come
   from, which service account can read them.
6. **Discuss rollout strategy.** Rolling? Canary? What
   thresholds abort the rollout?
7. **Agree on alerts.** Who owns which alert; what pager
   goes where.
8. **Agree on an incident runbook link.** Where the
   on-call engineer starts.

None of these is a technical decision at the code level.
All of them save weeks of "one email at a time".

## Where the two tracks meet

This module ends where the AI Infrastructure Engineer and
ML Platform Engineer tracks pick up:

- **AI Infrastructure Engineer** — Kubernetes,
  autoscaling, service mesh, observability platform,
  GPU inference infrastructure. The people who run the
  cluster your service lives in.
- **ML Platform Engineer** — the ML-specific platform:
  model registry, feature store, ML orchestration,
  inference server (Triton, KServe, Ray Serve), training
  infrastructure. The people who make it easier to hand
  off a service than to run one yourself.

Between the three tracks (yours, theirs, theirs), an
end-to-end ML system emerges. Nobody has to be all three.

## Common pitfalls

1. **"Just give the platform team access to my repo."**
   They shouldn't have to read your code to deploy it.
   The RUNBOOK is the contract.
2. **No capacity story.** "How many pods should we run?"
   is not a question the platform team can answer without
   your load-test numbers.
3. **Same probe for liveness and readiness.** Turns a slow
   start into a CrashLoop. Chapter 04.
4. **No `graceful-timeout`.** Deploys drop in-flight
   requests; the platform team sees a spike in 502s
   during every deploy and files a bug against you.
5. **Secrets baked into the image.** The platform team's
   security review rejects the deploy. Env vars, always.
6. **Mutable image tag (`latest`).** Rollback is
   ambiguous. Immutable per-SHA.
7. **High-cardinality metric labels.** Prometheus explodes;
   platform team pages you. Never `user_id`, never
   `request_id`, in metric labels.
8. **Silent breaking schema changes.** V1 clients break in
   the middle of a rolling deploy. Additive changes or
   version the route.
9. **No `/model` endpoint.** Impossible for the platform
   team to verify canary version splits.
10. **Assuming the platform team will "figure it out".**
    They will figure it out by pushing the ambiguity back
    onto you. Write the RUNBOOK.

## Summary

- The end of this module is the *start* of the platform
  team's work. The handoff is a contract encoded in a
  RUNBOOK.md.
- You own the service inside the container; they own the
  cluster around it. Some responsibilities sit at the
  seam and require an explicit conversation.
- Kubernetes, autoscaling, service mesh, canary rollouts,
  and specialist inference servers are their side.
  Chapters 01–10 built everything on your side; this
  chapter is the label on the box.
- The RUNBOOK is short, prescriptive, and answers every
  question the platform team is going to ask on day one.
- The AI Infrastructure Engineer and ML Platform Engineer
  tracks pick up from here.

## Where this goes next

`mod-109` (monitoring) is what happens *after* the service
is deployed. Data drift, prediction drift, delayed
ground-truth accuracy — the model-quality signals the
platform's Prometheus cannot see. `mod-110` (systems
design) is the whole picture: this module plus `mod-109`
plus data engineering plus platform, composed into an ML
system you can reason about end-to-end.
