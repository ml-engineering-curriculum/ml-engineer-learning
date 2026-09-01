# Resources for mod-108 — Model Packaging & Deployment

A curated short list. The goal is depth-on-demand for the
chapter you got stuck on, not a comprehensive bibliography.
Primary documentation is listed first; papers, standards,
and adjacent tools follow.

## Cross-cutting: the four projects the module rests on

Every chapter in this module leans on at least one of
these. Bookmark them once; the rest becomes lookup.

- **FastAPI — official documentation.**
  <https://fastapi.tiangolo.com/>
  The framework used from chapter 03 onwards. The
  "Tutorial - User Guide" is the shortest path to the
  patterns the module uses; the "Advanced User Guide"
  covers `lifespan`, dependency injection, background
  tasks, and middleware in depth.
- **Starlette — official documentation.**
  <https://www.starlette.io/>
  FastAPI is a Starlette application; the middleware,
  request/response, and testing primitives chapter 03
  and chapter 05 call live here. Read `Middleware` and
  `TestClient` at least once.
- **Pydantic v2 — official documentation.**
  <https://docs.pydantic.dev/latest/>
  Every request/response model in the module is a
  Pydantic v2 model. The "Concepts → Models" and
  "Concepts → Validators" pages are the ones the
  chapters implicitly assume.
- **Docker — official documentation.**
  <https://docs.docker.com/>
  Chapter 06 and 07 rest on the Dockerfile reference,
  the BuildKit user guide, and the CLI reference.
  Bookmark <https://docs.docker.com/reference/dockerfile/>
  and <https://docs.docker.com/build/buildkit/> first.

## Chapter 01 — Why serving is its own discipline

- **Google SRE Book — "Service Level Objectives".**
  <https://sre.google/sre-book/service-level-objectives/>
  The vocabulary chapter 01 quietly uses when it
  distinguishes "the model is accurate" from "the
  service is available". Free HTML.
- **Beyer, Betsy, et al. *Site Reliability Engineering*
  (O'Reilly, 2016).**
  <https://sre.google/sre-book/table-of-contents/>
  The full book, free HTML. Chapters 3 (Embracing
  Risk), 4 (SLOs), and 5 (Eliminating Toil) are the
  operational-mindset companion.
- **Sculley, D., et al. "Hidden Technical Debt in
  Machine Learning Systems" (NeurIPS 2015).**
  <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  The paper behind chapter 01's premise that "the
  model is the small part". "CACE" — changing
  anything changes everything — is the reason chapter
  02's metadata contract exists.

## Chapter 02 — Packaging the model artifact

- **`joblib` — official documentation.**
  <https://joblib.readthedocs.io/en/stable/>
  The persistence primitive chapter 02 recommends
  for scikit-learn pipelines. `joblib.dump` and
  `joblib.load` are the two functions to know.
- **Python `pickle` module — official reference.**
  <https://docs.python.org/3/library/pickle.html>
  The security warning at the top of the page is the
  reason chapter 02 refuses to load artifacts from
  untrusted sources.
- **scikit-learn — Model persistence guide.**
  <https://scikit-learn.org/stable/model_persistence.html>
  The scikit-learn maintainers' own advice on
  saving, loading, and version-compatibility of
  fitted estimators. The "Security & maintainability
  limitations" section is required reading.
- **ONNX — official specification and format.**
  <https://onnx.ai/>
  <https://github.com/onnx/onnx/blob/main/docs/IR.md>
  The framework-agnostic serialisation format
  chapter 02 lists as the stretch option.
  Interoperability between scikit-learn, PyTorch,
  and inference servers (ONNX Runtime, Triton).
- **ONNX Runtime — official documentation.**
  <https://onnxruntime.ai/docs/>
  The runtime that consumes ONNX graphs. The
  "Performance" section is what chapter 10's tuning
  lever #6 assumes.
- **`skl2onnx` — scikit-learn to ONNX converter.**
  <https://onnx.ai/sklearn-onnx/>
  The library that turns a scikit-learn `Pipeline`
  into an ONNX graph. Chapter 02's "convert once,
  serve anywhere" path.
- **MLflow — Model format reference.**
  <https://mlflow.org/docs/latest/models.html>
  The `MLmodel` YAML schema is the reference
  point when chapter 02 talks about "the metadata
  the artifact carries". Compatible with the
  registry work in `mod-106`.
- **Hugging Face — Safetensors.**
  <https://huggingface.co/docs/safetensors/index>
  The safer alternative to `pickle` / `torch.save`
  for tensor weights. Chapter 02's mention of "no
  arbitrary code execution on load" is exactly the
  property `safetensors` was built to provide.

## Chapter 03 — Anatomy of a FastAPI prediction service

- **FastAPI — Tutorial.**
  <https://fastapi.tiangolo.com/tutorial/>
  The path-parameter, body, and response-model
  chapters are the primary sources for chapter 03's
  `/predict` shape.
- **FastAPI — Dependencies.**
  <https://fastapi.tiangolo.com/tutorial/dependencies/>
  The `Depends` mechanism chapter 03 uses to inject
  the loaded `ModelBundle` into every route.
- **FastAPI — Testing.**
  <https://fastapi.tiangolo.com/tutorial/testing/>
  The `TestClient` pattern the exercise-01 test
  suite is built around.
- **FastAPI — `Response` and custom response classes.**
  <https://fastapi.tiangolo.com/advanced/custom-response/>
  `ORJSONResponse`, `UJSONResponse`, and streaming.
  Chapter 10's stretch tuning lever.
- **Uvicorn — official documentation.**
  <https://www.uvicorn.org/>
  The ASGI server FastAPI runs on. The
  "Deployment" section covers workers, gunicorn
  integration, and process management.
- **Gunicorn — official documentation.**
  <https://docs.gunicorn.org/en/stable/>
  The pre-fork worker manager the module uses in
  production (with `UvicornWorker`). The
  "Configuration" and "Settings" pages are the
  reference for `--workers`, `--threads`,
  `--graceful-timeout`, and the timeout family.
- **Uvicorn workers reference in Gunicorn.**
  <https://docs.gunicorn.org/en/stable/settings.html#worker-class>
  `-k uvicorn.workers.UvicornWorker`. The exact
  string chapter 03 and chapter 06 both use.

## Chapter 04 — Health checks and lifecycle

- **FastAPI — Lifespan events.**
  <https://fastapi.tiangolo.com/advanced/events/>
  The `lifespan` context manager pattern chapter 04
  builds startup and shutdown around.
- **Kubernetes — Configure liveness, readiness, and
  startup probes.**
  <https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/>
  The concrete definitions of liveness / readiness /
  startupProbe chapter 04 aligns `/healthz` and
  `/readyz` to. Read at least the "Types of probe"
  and "Configure probes" sections.
- **Docker — HEALTHCHECK reference.**
  <https://docs.docker.com/reference/dockerfile/#healthcheck>
  The Dockerfile instruction chapter 06 wires to
  `/readyz`. The interval / timeout / start-period /
  retries fields are what the exercise tunes.
- **12-Factor App — VI. Processes.**
  <https://12factor.net/processes>
  **12-Factor App — IX. Disposability.**
  <https://12factor.net/disposability>
  The design principles behind chapter 04's
  "graceful shutdown is the primitive". Terse; worth
  reading in full.

## Chapter 05 — Structured logging and observability

- **`python-json-logger` — README.**
  <https://github.com/madzak/python-json-logger>
  The stdlib-friendly JSON logger chapter 05 uses
  as the default. `pip install python-json-logger`.
- **`structlog` — official documentation.**
  <https://www.structlog.org/en/stable/>
  The alternative chapter 05 lists for teams that
  want first-class context binding. The
  "Getting Started" and "Bound Loggers" sections are
  the shortest path.
- **Python `logging` module — official reference.**
  <https://docs.python.org/3/library/logging.html>
  The stdlib primitives everything above wraps.
  Chapter 05's `dictConfig` recipe lives in
  `logging.config`.
  <https://docs.python.org/3/library/logging.config.html>
- **RFC 5424 — The Syslog Protocol.**
  <https://www.rfc-editor.org/rfc/rfc5424>
  The reference for structured log levels
  (`emerg`/`alert`/…/`debug`) and the shape of a
  timestamped log record. Chapter 05's field
  choices trace back to this.
- **OpenTelemetry — Semantic Conventions for HTTP.**
  <https://opentelemetry.io/docs/specs/semconv/http/>
  The industry-standard names for `http.method`,
  `http.route`, `http.status_code`, and
  `http.request.duration`. If your logs align with
  these, downstream tooling parses them for free.
- **OpenTelemetry — Semantic Conventions for General
  Attributes.**
  <https://opentelemetry.io/docs/specs/semconv/general/attributes/>
  Where `service.name`, `service.version`,
  `deployment.environment`, and `request.id` live.
- **W3C Trace Context.**
  <https://www.w3.org/TR/trace-context/>
  The `traceparent` header standard. When a
  downstream service wants to correlate its trace
  with yours, this is the contract.
- **RFC 7807 — Problem Details for HTTP APIs.**
  <https://www.rfc-editor.org/rfc/rfc7807>
  The standard for structured error responses.
  Chapter 05's error-body shape is a RFC 7807
  minimal `application/problem+json`.

## Chapter 06 — Containerising the service with Docker

- **Docker — Dockerfile reference.**
  <https://docs.docker.com/reference/dockerfile/>
  The canonical reference for every instruction the
  chapter uses (`FROM`, `COPY`, `RUN`, `USER`,
  `HEALTHCHECK`, `ENTRYPOINT`, `CMD`).
- **Docker — Multi-stage builds.**
  <https://docs.docker.com/build/building/multi-stage/>
  The primary source for chapter 06's multi-stage
  pattern.
- **Docker — Build cache and layer semantics.**
  <https://docs.docker.com/build/cache/>
  Why the "copy dependency files, install deps,
  then copy source" order matters.
- **Docker — BuildKit.**
  <https://docs.docker.com/build/buildkit/>
  The default builder as of Docker 23. Cache
  mounts, secret mounts, and SBOM generation live
  here.
- **`.dockerignore` reference.**
  <https://docs.docker.com/build/concepts/context/#dockerignore-files>
  The exclusion file exercise-02 asks you to write.
- **Docker Official Python image.**
  <https://hub.docker.com/_/python>
  The base image chapter 06 recommends. Read the
  "Image variants" section for `-slim`, `-alpine`,
  and `-bookworm` trade-offs.
- **Google — `distroless` base images.**
  <https://github.com/GoogleContainerTools/distroless>
  The minimal-runtime alternative chapter 06's
  stretch goal explores. No shell, no package
  manager, just the language runtime.
- **`uv` — official documentation.**
  <https://docs.astral.sh/uv/>
  The Rust-based Python installer / resolver
  chapter 06's Dockerfile uses for fast, reproducible
  builds. `uv sync --frozen` is the reproducibility
  primitive.
- **`pip` — official user guide.**
  <https://pip.pypa.io/en/stable/user_guide/>
  The fallback when `uv` is not available. Chapter
  06 shows both forms.
- **PEP 508 — Dependency specification.**
  <https://peps.python.org/pep-0508/>
  The syntax of a `pyproject.toml` `dependencies`
  entry — useful when the container build fails
  because the specifier is subtly wrong.
- **CIS Docker Benchmark.**
  <https://www.cisecurity.org/benchmark/docker>
  The industry checklist for hardened Docker
  configurations. Chapter 06's "non-root user, no
  secrets in image, pinned base" advice is a
  subset. Free after registration.
  <!-- needs-research: the CIS benchmark PDFs are
  behind a registration wall and the version
  numbers roll forward; verify the current
  benchmark version if quoted directly. -->

## Chapter 07 — Running in CI

- **GitHub Actions — official documentation.**
  <https://docs.github.com/en/actions>
  The primary reference. The "Reusing workflows",
  "Contexts", and "Environment variables" pages are
  the ones chapter 07 quietly assumes.
- **`docker/setup-buildx-action`.**
  <https://github.com/docker/setup-buildx-action>
  The action that provisions BuildKit on a GitHub
  runner. Required for the `cache-from: type=gha`
  pattern.
- **`docker/build-push-action`.**
  <https://github.com/docker/build-push-action>
  The higher-level action for build (and optionally
  push) with a GHA cache. Chapter 07's workflow
  uses this.
- **`docker/login-action`.**
  <https://github.com/docker/login-action>
  Authenticates to a container registry (Docker
  Hub, GHCR, ECR, GAR). Needed for the "push on
  main" stretch goal in exercise-02.
- **GitHub Actions — cache action.**
  <https://github.com/actions/cache>
  The stdlib cache action. Relevant even when
  BuildKit's `type=gha` cache is what handles the
  image layers.
- **`actions/upload-artifact`.**
  <https://github.com/actions/upload-artifact>
  For persisting the junit and k6 result JSON
  between jobs / runs.
- **Trivy — vulnerability scanner.**
  <https://aquasecurity.github.io/trivy/latest/>
  Chapter 06 / exercise-02 stretch goal. The
  `trivy image` CLI is the "does my base have
  critical CVEs" step.
- **Grype — Anchore's alternative scanner.**
  <https://github.com/anchore/grype>
  The other widely used image scanner.

## Chapter 08 — Batch vs online inference

- **Apache Airflow — official documentation.**
  <https://airflow.apache.org/docs/apache-airflow/stable/>
  The most widely deployed batch orchestrator. The
  "Tutorial" and "DAGs" sections are the shortest
  path to wrapping the chapter's batch job in a
  scheduled DAG.
- **Prefect — official documentation.**
  <https://docs.prefect.io/>
  The Python-first alternative to Airflow. Simpler
  local development story.
- **Dagster — official documentation.**
  <https://docs.dagster.io/>
  The asset-oriented alternative. Chapter 08
  mentions all three; the choice is platform-
  driven, not ML-driven.
- **Apache Spark — MLlib inference pipelines.**
  <https://spark.apache.org/docs/latest/ml-pipeline.html>
  The scale-out batch option when pandas /
  DataFrame no longer fits. Chapter 08 lists Spark
  as the "very large data" escalation.
- **Ray — Ray Serve.**
  <https://docs.ray.io/en/latest/serve/index.html>
  Chapter 08 mentions Ray Serve for the hybrid
  patterns (batch feature computation, online
  serving). The "Getting Started" and
  "Deployment" pages are the entry points.
- **Ray — Ray Data batch inference guide.**
  <https://docs.ray.io/en/latest/data/batch_inference.html>
  The chapter-08 "batch is a data pipeline that
  calls a model" pattern, at scale.
- **Feast — Feature store documentation.**
  <https://docs.feast.dev/>
  The open-source feature store chapter 08's
  hybrid pattern 2 references. `mod-103`
  (data engineering) covers this in depth.
- **NVIDIA Triton Inference Server.**
  <https://developer.nvidia.com/triton-inference-server>
  <https://docs.nvidia.com/deeplearning/triton-inference-server/>
  The reference production inference server for
  GPU-served models. Implements the micro-batching
  chapter 08 warns you not to build in-house.
- **vLLM — high-throughput LLM inference.**
  <https://docs.vllm.ai/en/latest/>
  The chapter-08 mention for LLM-specific serving.
  Continuous batching + PagedAttention are its
  distinguishing features.
- **KServe — Kubernetes model inference platform.**
  <https://kserve.github.io/website/>
  The Kubernetes-native serving stack chapter 11
  hands off to. Owned by the AI Platform track,
  not this module.
- **TorchServe — official documentation.**
  <https://pytorch.org/serve/>
  PyTorch's reference inference server. Alternative
  to Triton for CPU or single-GPU PyTorch models.

## Chapter 09 — Latency, throughput, and concurrency

- **Dean, Jeffrey, and Luiz André Barroso. "The Tail
  at Scale" (*Communications of the ACM*, 2013).**
  <https://cacm.acm.org/research/the-tail-at-scale/>
  The paper behind chapter 09's tail-amplification
  argument — "your p99 is someone else's p50"
  originates here. Required reading if you touch
  fan-out systems.
- **Little, John D. C. "A Proof for the Queueing
  Formula: L = λW" (*Operations Research*, 1961).**
  <https://doi.org/10.1287/opre.9.3.383>
  The original paper. Chapter 09's use of Little's
  Law is worth grounding in the source.
- **Little, John D. C., and Stephen C. Graves.
  "Little's Law" (in *Building Intuition*, Springer,
  2008).**
  <https://doi.org/10.1007/978-0-387-73699-0_5>
  The modern tutorial version, with worked
  examples and an accessible proof.
- **PEP 703 — Making the Global Interpreter Lock
  Optional in CPython.**
  <https://peps.python.org/pep-0703/>
  The PEP that motivates chapter 09's discussion
  of the GIL and its effect on threaded ML
  serving. Currently experimental in CPython 3.13.
- **`asyncio` — Python standard library reference.**
  <https://docs.python.org/3/library/asyncio.html>
  The concurrency primitive behind chapter 09's
  async section. `asyncio.to_thread` and
  Starlette's `run_in_threadpool` are the bridges
  chapter 03 uses.
- **`anyio` — official documentation.**
  <https://anyio.readthedocs.io/en/stable/>
  Starlette's underlying async library. The
  `to_thread.run_sync` helper is the mechanism
  behind `run_in_threadpool`.
- **Kleppmann, Martin. *Designing Data-Intensive
  Applications* (O'Reilly, 2017).**
  <https://dataintensive.net/>
  Chapter 1 ("Reliable, Scalable, and Maintainable
  Applications") is the definitive treatment of
  latency percentiles, tail latency, and the
  "average is the wrong statistic" argument.
- **Tene, Gil. "How NOT to Measure Latency" (talk).**
  <https://www.youtube.com/watch?v=lJ8ydIuPFeU>
  <https://highscalability.com/how-not-to-measure-latency/>
  The 45-minute talk that named "coordinated
  omission" and reset the industry's understanding
  of load-test measurement. Required viewing
  before running chapter 10's load tests.

## Chapter 10 — Load testing and tuning

- **k6 — official documentation.**
  <https://grafana.com/docs/k6/latest/>
  The module's default load generator. The
  "Test types" and "Executors" pages are the
  reference for `ramping-arrival-rate` vs
  `ramping-vus` — chapter 10 pitfall #4.
- **k6 — Thresholds.**
  <https://grafana.com/docs/k6/latest/using-k6/thresholds/>
  The primary reference for turning a load test
  into a pass/fail CI gate. Exercise-04 Part E
  uses this.
- **k6 — Output formats.**
  <https://grafana.com/docs/k6/latest/results-output/>
  How to write JSON for post-processing (the
  `--out json=<file>` pattern the exercise
  requires).
- **wrk — GitHub repository.**
  <https://github.com/wg/wrk>
  The C-based HTTP benchmark chapter 10 lists as
  the fast, minimal alternative. Read
  `SCRIPTING.md` for the Lua hooks.
- **Locust — official documentation.**
  <https://docs.locust.io/en/stable/>
  The Python-first alternative. The
  "Writing a locustfile" tutorial covers the
  scenario style chapter 10 shows.
- **Gunicorn — worker types.**
  <https://docs.gunicorn.org/en/stable/design.html>
  The reference for why chapter 09's
  `-k uvicorn.workers.UvicornWorker` choice
  matters — sync vs async worker semantics.
- **Gunicorn — settings reference.**
  <https://docs.gunicorn.org/en/stable/settings.html>
  Every knob chapter 10 tunes:
  `workers`, `threads`, `worker-connections`,
  `graceful-timeout`, `timeout`, `keep-alive`.
- **HdrHistogram — reference implementation and
  paper.**
  <https://github.com/HdrHistogram/HdrHistogram>
  <https://hdrhistogram.github.io/HdrHistogram/>
  The histogram data structure both wrk and k6
  use internally to compute percentiles without
  data loss at the tail. Worth knowing the name
  when reading load-test output.

## Chapter 11 — Handoff to platform

- **Kubernetes — Pod lifecycle.**
  <https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/>
  The reference for pod phases, container states,
  and probe interactions. Chapter 11's RUNBOOK
  fields map onto sections of this page.
- **Kubernetes — Container probes.**
  <https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#container-probes>
  Duplicate of chapter 04's primary reference;
  chapter 11 rereads it from the platform's
  perspective.
- **Kubernetes — Assign CPU / memory resources.**
  <https://kubernetes.io/docs/tasks/configure-pod-container/assign-cpu-resource/>
  <https://kubernetes.io/docs/tasks/configure-pod-container/assign-memory-resource/>
  The reference for request / limit semantics,
  which chapter 11 (and the RUNBOOK) captures.
- **Kubernetes — Horizontal Pod Autoscaler.**
  <https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/>
  The autoscaler the load-test numbers from
  chapter 10 feed. Kept as reference; belongs to
  the platform track.
- **12-Factor App — full site.**
  <https://12factor.net/>
  All twelve factors. Chapter 11's RUNBOOK template
  hits ~half of them explicitly (config, backing
  services, port binding, concurrency, disposability,
  logs).
- **OpenAPI Specification 3.1.**
  <https://spec.openapis.org/oas/latest.html>
  FastAPI generates OpenAPI 3.1 automatically. The
  RUNBOOK's `/docs` bullet is a pointer to this
  contract.
- **Prometheus — client libraries: Python.**
  <https://prometheus.github.io/client_python/>
  If chapter 11's `/metrics` endpoint is
  Prometheus-scraped (the common case), this is
  the library. Not required by chapter 11; useful
  for the exercise-02 stretch and for `mod-109`.
- **OpenTelemetry — Python SDK.**
  <https://opentelemetry.io/docs/languages/python/>
  The vendor-neutral alternative for traces,
  metrics, and structured logs. `mod-109`
  (monitoring) uses this in earnest.

## Adjacent standards and specifications

- **HTTP/1.1 — RFC 9112.**
  <https://www.rfc-editor.org/rfc/rfc9112>
  The transport under everything the module does.
  Worth skimming section 6 (Message Body) and 9
  (Connection Management) at least once.
- **HTTP semantics — RFC 9110.**
  <https://www.rfc-editor.org/rfc/rfc9110>
  The status code and method semantics chapter 03
  uses (200 vs 400 vs 422 vs 500).
- **JSON — RFC 8259.**
  <https://www.rfc-editor.org/rfc/rfc8259>
  The interchange format the module never
  questions. Read it once for edge cases
  (duplicate keys, number precision).
- **CloudEvents — event format specification.**
  <https://cloudevents.io/>
  Relevant when the batch job's output is
  consumed as events rather than files. Chapter
  08's "the consumer is asynchronous" case has an
  event-driven cousin.
- **OCI Image Format Specification.**
  <https://github.com/opencontainers/image-spec>
  The vendor-neutral standard for the container
  images Docker produces. Useful when platform
  tooling (Kaniko, buildah, podman) is not Docker.

## Datasets used in the exercises

The exercises deliberately do not prescribe a dataset;
they reuse whatever is in play from `mod-104`. Two
options work well:

- **UCI Adult (Census Income).**
  <https://archive.ics.uci.edu/dataset/2/adult>
  Loadable via
  `sklearn.datasets.fetch_openml("adult", version=2)`.
  Small, tabular, non-trivial preprocessing pipeline
  — exercises the "column order matters" property
  chapter 02 hammers on.
- **NYC Taxi trip data.**
  <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
  Monthly Parquet snapshots. Enough rows to
  exercise the batch job's chunking; realistic
  schema drift between months.

<!-- needs-research: on each autonomous research
cycle, verify these URLs still resolve. In
particular: (a) FastAPI / Starlette / Pydantic
docs (versioning can shift URL structure);
(b) Gunicorn / Uvicorn versioned pages (`stable`
vs numbered releases); (c) k6 docs (moved from
k6.io/docs to grafana.com/docs/k6 in 2023 —
check that the current canonical is still the
Grafana host); (d) Kubernetes docs (page names
occasionally rename between minor releases);
(e) Ray, Feast, and Triton pages (heavier
release cadence, versioned URLs); (f) SRE book
URL (`sre.google` has redirected before). Swap
in current pages when a doc URL 404s. -->
