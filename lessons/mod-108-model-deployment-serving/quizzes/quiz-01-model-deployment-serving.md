# quiz-01 — Model Packaging & Deployment

> **Format:** 11 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–11 of this module (one question per chapter).
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter the point
> comes from — go re-read that section if you missed the question.

---

## Questions

### 1. (Chapter 01) A teammate hands you a `model.pkl` and says "the hard part is done, just put a server in front of it." Chapter 01 insists this is wrong because the artifact on its own leaves two silent gaps, and because a "serving contract" is specifically four things. Which option most faithfully reproduces both claims?

A. The only real gap is the lack of an HTTP server; once a Flask route wraps
   `pickle.load`, the artifact is production-ready. A "contract" means
   whatever the caller asks for on the day they integrate.
B. The two silent gaps are **the implicit input contract** (the training
   pipeline's assumptions about column order, dtypes, imputation defaults,
   and one-hot categories, none of which are visible from a JSON payload)
   and **the implicit environment** (the exact Python and library versions
   the pickle expects, which the caller does not have). A **serving
   contract** has four parts: (1) a URL and a request schema (validated on
   entry), (2) a response schema (versioned, with the model version in the
   body), (3) a **health check** (an HTTP endpoint that returns `200` when
   the service can serve traffic and non-`200` when it cannot, split into
   liveness and readiness — chapter 04), and (4) a **latency and throughput
   budget** (a p95 the service commits to at a stated request rate — without
   it, autoscaling cannot scale and nobody can tell a regression from
   noise). Miss any of the four and downstream teams either paper over the
   gap badly or refuse to depend on the service.
C. The two gaps are `requirements.txt` being incomplete and the port number
   not being documented. The contract is just the URL.
D. There are no gaps; `pickle.load` is deterministic across Python versions,
   and HTTP is self-documenting.

### 2. (Chapter 02) A colleague ships a scikit-learn pipeline by committing `model.pkl` to git and writing the preprocessing scaler by hand in the serving code ("I know the mean and std from the notebook"). Chapter 02 names three questions the artifact has to answer and the specific pitfall this approach walks into. Which option captures both?

A. Hand-coding the scaler is fine as long as the numbers match the notebook;
   the three questions are "what is the file extension, what is the git SHA,
   and who trained it?"
B. An artifact has to answer three questions: **(1) how to turn the request
   payload into model input** (the preprocessing pipeline — scalers,
   imputers, one-hot encoders, column order, dtypes — must travel *with*
   the model, not be re-implemented in the serving code), **(2) how to
   compute a prediction from that input** (the estimator itself), and
   **(3) what was trained and how to identify it later** (metadata:
   `model_name`, `model_version`, `git_sha`, framework version, Python
   version, training dataset hash, input/output schema, eval metrics,
   MLflow run ID — the fields every incident ticket and every log line will
   ask for). Re-implementing the scaler by hand in the service is **chapter
   02 pitfall #2**: the service is now guaranteed to drift from training
   the moment the scaler changes, silently, without an error. The fix is to
   serialise the scaler *inside* the same `Pipeline` as the estimator (joblib
   captures the whole object graph) and let the service call
   `pipeline.predict` — there is no "scaler code" in the service at all.
   Related pitfalls from the chapter: pickling a lambda (lambdas don't
   survive a pickle round-trip in a fresh process; use a module-level
   function), loading the model per request (reload cost defeats every
   latency goal — load once in `lifespan` and hold the reference), and
   unpickling a file from an unverified source (pickle can execute arbitrary
   code at load time; model files are code, treat them that way).
C. The three questions are "is it big? is it fast? is it accurate?" Hand-
   coding the scaler is the recommended pattern because it makes the
   service "framework-independent".
D. Pickling always works across sklearn minor versions, and metadata is a
   formatting preference, not a reproducibility hook.

### 3. (Chapter 03) An engineer writes `async def predict(...)` in FastAPI, inside which they call `model.predict_proba(features)` directly on a scikit-learn pipeline (no `await`, no thread-pool offload). Under 50 concurrent callers the service degenerates to serving requests one after another. Chapter 03 names the rule that was broken and the two canonical fixes. Which option names both?

A. The issue is that `async def` is always slower than `def`; the fix is to
   rename every handler to `def`.
B. The rule: **`async def` runs on the main asyncio event loop and yields
   control at `await` points**; a CPU-bound call with no `await` blocks the
   loop for its entire duration, so **no other connection in that worker
   can make progress**. For scikit-learn / XGBoost / small CPU PyTorch
   inference, the two valid shapes are: **(a) write the handler as `def`
   (sync)**, which FastAPI runs on its threadpool so concurrent requests
   actually overlap; or **(b) keep `async def` but wrap the CPU call in
   `await run_in_threadpool(...)`** (from `starlette.concurrency`) so the
   heavy work runs off the event loop while the loop stays flowing. The
   chapter's recommendation is explicit: **sync `def` for CPU-bound
   prediction; `async def` only for handlers that genuinely `await` I/O**
   (a feature-store call via an async client, an `aioboto3` download, a
   downstream HTTP hop). Mixing them badly — the pattern above — is listed
   as chapter 03 pitfall #3 and is the single most common FastAPI-serving
   performance regression. Chapter 09 formalises this under the three
   concurrency axes (processes / threads / async) and explains why workers
   provide CPU parallelism, threads help I/O and GIL-releasing C
   extensions, and async multiplexes I/O.
C. The issue is pydantic validation being slow; disable `response_model`
   and the throughput recovers.
D. Add `asyncio.sleep(0)` before each prediction and the event loop will
   yield correctly on its own.

### 4. (Chapter 04) A platform engineer reports: "every time the model takes more than 30 seconds to load, Kubernetes puts the pod into CrashLoopBackOff." The ML engineer wired the same endpoint to both the `livenessProbe` and the `readinessProbe`. Chapter 04 names the precise distinction between the two and the lifecycle discipline the service should follow. Which option captures both the diagnosis and the correct shape?

A. CrashLoopBackOff is normal Kubernetes behaviour during model loading;
   add `initialDelaySeconds=120` to the liveness probe and move on.
B. **Liveness and readiness are different signals**, and conflating them is
   the bug. **Liveness** answers "is this process making forward progress —
   should I kill it?"; if non-`200`, the orchestrator **restarts** the pod.
   Liveness must be **trivial, fast, local** — no model reference, no
   downstream dependency: `@app.get("/healthz") -> {"status": "ok"}` is the
   whole implementation. **Readiness** answers "can this instance serve
   traffic right now?"; if non-`200`, the orchestrator **removes it from
   the load balancer** but does not kill it. Readiness checks whether the
   `lifespan` finished, `_bundle is not None`, warm-up succeeded, and
   critical dependencies are reachable. With a 30-second model load and
   **separate probes**, liveness returns `200` from t=0 (process is alive)
   while readiness returns `503` until t=30; the pod is never restarted.
   With the **same probe** wired to both, Kubernetes' default
   (`periodSeconds=10, failureThreshold=3`) kills the pod after ~30 s of
   failed probes — right as the model finishes loading — and the cycle
   repeats. The full lifecycle is **startup → warm-up (a dummy prediction
   on synthetic input so the first live request doesn't pay JIT / page-
   fault cost) → serving → drain (flip a `_shutting_down` flag so readyz
   returns `503` *before* the process declines connections, so the LB stops
   sending traffic; gunicorn's `--graceful-timeout` lets in-flight requests
   finish) → exit**. Use `startupProbe` for services with long loads so the
   startup budget is decoupled from steady-state probe frequency. Keep
   `/healthz` and `/readyz` unauthenticated and PII-free; the orchestrator
   doesn't send tokens.
C. Rename `/healthz` to `/livez` — the name is what Kubernetes uses to
   decide whether to restart.
D. Have `/healthz` run a real prediction every poll so you know the model
   still works — any failure should restart the pod.

### 5. (Chapter 05) An oncall engineer tails the service logs during an incident and sees `INFO: request came in`, `INFO: predicted 0.87`, `ERROR: something went wrong`. Chapter 05 names the three principles the logging setup should have followed, the one piece of code that does most of the work, and a specific list of things that must **not** go into logs. Which option captures all three?

A. Free-text logs are fine; the fix is to switch log level from INFO to
   DEBUG so more lines appear.
B. The three principles: **(1) structured over prose** — every log line is
   a JSON object with typed fields, not a prose sentence; `jq` becomes the
   query engine. **(2) a `request_id` stitches events together** — one
   identifier per request, read from an `X-Request-ID` header or generated
   on entry, propagated to every log line the request produces and echoed
   back in the response header. **(3) model metadata on every prediction
   line** — `model_name`, `model_version`, `git_sha` from chapter 02's
   metadata, so rolling-deploy incidents can be attributed to the version
   (not the timestamp) and `kubectl logs | jq 'select(.model_version ==
   "1.4.2")'` just works. The load-bearing piece of code is a single
   `RequestLogMiddleware` that reads/generates `request_id`, times the
   request with `time.perf_counter()`, and logs one `request_end` line
   with `method`, `path`, `status_code`, `duration_ms`, and the id —
   correctly wrapping the handler in `try/finally` so exceptions are still
   logged. Log to **stdout** (not a file — containers stream stdout to the
   platform's log collector; files require rotation and are invisible),
   with **UTC ISO-8601 timestamps** (local timezones make cross-service
   timeline reconstruction impossible). What must **not** go into logs:
   **request bodies verbatim**, **PII** (name, email, phone, SSN, home
   address, fine-grained location), **PHI** (diagnoses, medications, lab
   results), **PCI** (PAN, CVV, full IBAN), and **authentication material**
   (bearer tokens, API keys, session cookies, passwords). Log the
   *prediction* (a scalar / small vector) not the input; hash user IDs at
   the boundary; whitelist fields; bake redaction into the logging config
   as a `logging.Filter` so a sleepy developer's `extra={"email": ...}`
   gets redacted automatically.
C. Logs should contain the full request body so support can reproduce
   issues — PII compliance is somebody else's problem.
D. Random `print()` calls are acceptable because the container will pick
   them up anyway.

### 6. (Chapter 06) A reviewer looks at a Dockerfile that starts `FROM python:latest`, does `COPY . .` immediately followed by `RUN pip install -r requirements.txt`, runs as root, uses the Alpine base variant, and bakes `DATABASE_PASSWORD=hunter2` into an `ENV` line. Chapter 06 names a specific pitfall for each of those choices. Which option lists the fixes correctly?

A. All five are fine; chapter 06 only cares about the final image size, not
   the specific steps.
B. Five chapter-06 pitfalls, each with its fix: **(1) `FROM python:latest`
   is forbidden** — pin the minor version *and* the base digest
   (`python:3.11-slim@sha256:...`) so a rebuild months later cannot silently
   pull a different base. **(2) `COPY . . ; RUN pip install`** invalidates
   the expensive dependency layer on every source change; the discipline
   is `COPY pyproject.toml uv.lock ./ ; RUN uv sync --frozen ; COPY app/ .`
   so a code change only invalidates the cheap application layer.
   Combined with a `.dockerignore` that excludes `.git`, `data/`, `.env`,
   `.venv/`, tests, and the Dockerfile itself — otherwise `COPY . .` ships
   your training data and git history into production. **(3) running as
   root** is rejected by platform admission controllers (`runAsNonRoot`)
   and widens the blast radius on container escape; `groupadd app && useradd
   --uid 1000 app && USER app` is the fix. **(4) Alpine for ML** uses musl
   not glibc, so most ML wheels (numpy, pandas, scipy, torch) fall back to
   compiling from source — a 20-minute fragile build for no real benefit;
   `python:3.11-slim` (Debian, glibc, ~50 MB) is the pragmatic default.
   **(5) secrets in the image** — bake nothing in; secrets come via
   environment variables at run time, from the platform's secret store.
   Two more chapter-06 bonuses: set `ENV PYTHONUNBUFFERED=1` so logs don't
   disappear on crash; use a multi-stage build (`AS builder` / `AS
   runtime`) so compilers never leave the build stage; add a
   `HEALTHCHECK` with `--start-period` matching model load time; and if
   you build on Apple Silicon for an x86 target, set `--platform
   linux/amd64` or the image runs under emulation.
C. Pin to `latest` so you always get security updates; run as root for
   simplicity; ship secrets in the image so the container can "self-
   configure".
D. Alpine is always smaller so always correct; `.dockerignore` only
   matters in Node projects.

### 7. (Chapter 07) A reviewer looks at a CI job whose step is `docker build -t churn:ci . && echo "image built OK"` and asks "is this actually testing the service?". Chapter 07 says no, and spells out what a minimum smoke / integration layer needs. Which option captures both what the smoke test must do and the CI-specific discipline?

A. "The image built" is proof the service works; no further tests are
   needed.
B. "Image built" is not a test; the container can build and crash on
   start. The minimum smoke test (`scripts/smoke.sh`) must: **poll
   `/readyz`** for up to ~60 s (model load can take longer than one probe
   interval) — if it never returns `200`, fail loudly; **hit `/healthz`**;
   **POST a valid body to `/predict`** and assert the response is a JSON
   object containing a `model_version` and a probability in `[0, 1]`.
   Assert on **shape, not value** — the model can legitimately return any
   probability; coupling to a specific number breaks every time the model
   is retrained. Two layers of test above the smoke script: **unit tests**
   use FastAPI's `TestClient` inside a `with` block (so `lifespan` runs)
   and swap in a fake bundle via `app.dependency_overrides[get_bundle]` —
   fast, no container needed. **Integration tests** hit a running container
   via `docker compose`. Mark them `-m integration` so the fast tier runs
   on every push and the slow tier runs on PR. CI-specific discipline:
   build with BuildKit's GHA cache (`cache-from: type=gha, cache-to:
   type=gha,mode=max`) so dependency layers are reused between runs;
   `load: true, push: false` so subsequent `docker run` steps can see the
   image (don't push on every commit — that's a separate job on merge to
   main); **print `docker logs` with `if: always()`** before teardown, so
   when the smoke test fails you see the traceback instead of "curl
   returned 000"; and `docker rm -f` in a cleanup step so hung containers
   don't leave the runner wedged. For the model artifact, commit a small
   fixture under `tests/fixtures/artifact/` — real production services
   pull version-pinned from object storage, but a fixture is the level-
   appropriate choice. Never use `--reload` in production (filesystem
   watcher, single worker, degrades under sustained edits); it is a
   local-development-only mode.
C. Push on every commit so the registry always has the latest; it's
   simpler than tag-on-merge.
D. Run `--reload` in CI so the container picks up test files — it's
   faster that way.

### 8. (Chapter 08) A product manager asks: "for the homepage recommendations we show every logged-in user, should we serve from a model API or precompute nightly? And can we 'just' build a micro-batcher inside our FastAPI service to speed up the GPU inference we're adding?" Chapter 08 provides a decision matrix and names three hybrid patterns. Which option best handles both questions?

A. Always use online — batch is legacy; and definitely build the micro-
   batcher yourself, it's "only a queue".
B. The decision is a **product question, not a technology question**:
   *when does the caller need the prediction?* Pick **online** if **any**
   of: a human is waiting (interactive UI), the prediction is part of a
   synchronous decision (the fraud check must approve or decline before
   the transaction commits), or features are only known at request time
   (the search query just typed). Pick **batch** if **all** of: the
   consumer is asynchronous (a nightly email campaign, a downstream
   table), features are known ahead of time (daily-cadence attributes),
   and stale-by-hours predictions are acceptable. For homepage
   recommendations where the set of users is bounded and knowable in
   advance and scores change slowly, **hybrid pattern 1 — pre-compute,
   serve from cache** is the right shape: a nightly batch job writes
   `(user_id → score)` to a KV store (Redis / DynamoDB); the online path
   is a 1–5 ms lookup, not a 10–100 ms model call. New users with no
   cache entry need an explicit policy (default value, fallback to online
   inference, or reject) — don't leave it implicit. The other two
   hybrids: **pattern 2** — batch pre-computes expensive stable features
   (30-day aggregates, embeddings) into a feature store; online joins
   those with request-time features at prediction time (this is the
   industry-standard feature-store pattern — Feast, Tecton, SageMaker FS,
   Vertex FS). **Pattern 3 — micro-batching on the request path** — a
   coordinator collects incoming requests in a short window (e.g., 10 ms)
   and runs a single batched model call, slicing the output back to
   callers. The trade-off is a small latency floor for much higher
   throughput; worth it on GPU-served models whose per-example cost drops
   sharply with batch size, rarely worth it on CPU models with ms
   inference. **Do not build micro-batching in-house the first time** —
   Triton, Ray Serve, TorchServe, KServe, and vLLM all ship well-tested
   implementations; DIY buys adversarial latency spikes and tail-latency
   coupling. Pragmatic operational rules for the batch path: idempotent
   versioned output (`s3://scores/dt=2026-06-14/run=<uuid>/`, not
   overwrite `latest/`), a `_MANIFEST.json` with model version / row
   count / start / end so downstream can tell "did it run today?", a
   schema check that fails the job loudly on drift, bounded-memory
   chunking from the start (10 000–100 000 rows is typical), and
   **sharing the scoring code path between batch and online** — the
   #1 chapter-08 pitfall is batch and online computing subtly different
   features for the same user.
C. Pick batch for everything — it's cheaper. The caller can wait a day.
D. Batch and online are mutually exclusive; no system uses both.

### 9. (Chapter 09) A reviewer asks "how fast is the service?" and the engineer replies "the average model call is 18 ms." Chapter 09 would send this back with specific objections, plus a formula that ties concurrency, latency, and throughput together. Which option both names the reporting discipline and the formula?

A. Average latency is the right number; percentiles are academic.
B. **Latency is a distribution, not a number**, and the mean is essentially
   useless — one 10-second outlier moves the mean of a thousand 20 ms
   requests by 10 ms while hiding the specific user who waited 10 seconds.
   Report **p50, p95, p99, and max**; alert on p95 and p99 crossing
   thresholds. Also name **which latency** you measured — the four common
   measurement points produce four different numbers: **in the client**
   (includes network both ways — what the user experiences), **at the
   load balancer** (server-side, excludes client network), **in the
   handler** (excludes queueing in front of the worker), and **the model
   call alone** (the smallest of the four; the number ML engineers
   usually quote by default). Reporting 18 ms for the model call while a
   downstream team measures 400 ms from the client is the canonical
   mismatch. **Throughput is meaningless without a quality-of-service
   clause**: "this service sustains 500 QPS at p95 ≤ 100 ms" is a
   defensible sentence; "500 QPS" alone is not. The **saturation point**
   is the QPS above which latency degrades non-linearly (queueing starts
   dominating). The formula is **Little's Law: L = λ × W**, where L is
   the average number of requests in the system, λ is the average
   arrival rate, and W is the average time in the system. Rearrangements
   that matter: **λ = L / W** (8 workers × (1 / 0.050 s) caps throughput
   at 160 QPS regardless of traffic) and **W = L / λ** (under target
   throughput, average time-in-system is set by concurrency — the rest
   is queueing). Tail latency **amplifies through fan-out**: a parent
   request calling N downstream services in parallel sees the *max* of
   their latencies, so if each has p99 = 100 ms, a 10-way fan-out has
   ~10% chance of hitting at least one tail call — your p99 becomes a
   downstream caller's p50. Three concurrency axes in Python:
   **processes** (`gunicorn -w N` — independent interpreters, separate
   GILs, true parallelism for CPU-bound work; memory multiplies by N;
   start with N = cores or cores − 1), **threads** (shared memory + GIL;
   help for I/O waits and for C extensions that release the GIL like
   numpy / torch / sklearn; useless for pure-Python CPU work), and
   **async** (one event loop multiplexes many I/O-bound connections;
   poison for CPU-bound work with no `await` — see chapter 03). For
   CPU-bound ML serving the default is `-w cores`, `--threads 1`, sync
   handlers; mix in threads when the handler also waits on a feature
   store; use `async + run_in_threadpool` when the handler genuinely
   awaits I/O. Memory is the silent constraint — model memory ×
   workers — OOMKills are what shows up at 3 a.m.
C. Report only the model's forward-pass time; network, queueing, and
   feature retrieval are someone else's problem.
D. Little's Law says throughput equals latency; adding workers always
   doubles throughput.

### 10. (Chapter 10) An engineer runs a load test by opening Locust from their laptop over home Wi-Fi, ramps to "100 virtual users", reports the mean latency as 42 ms, and merges the PR. Chapter 10 says at least five things are wrong with that run. Which option names them and the reproducible shape of a real test?

A. The test is fine; mean latency over Wi-Fi is the industry standard.
B. Five chapter-10 pitfalls, all violated: **(1) testing from a laptop
   over Wi-Fi** — the latency and jitter are the home network's, not the
   service's; run the generator **adjacent to the service** (same host for
   local dev, same region / AZ / VPC for a real deployment). **(2) `vus` /
   `ramping-vus`** — VU-based load self-throttles (each virtual user waits
   for the previous response before sending the next), so saturation is
   **invisible**: as the service slows down the VUs slow down too, and
   the latency-throughput curve never shows its knee. Use
   **`ramping-arrival-rate`** so the generator keeps injecting N requests
   per second regardless of service state — that is how you *see*
   saturation. **(3) reporting the mean** — meaningless (see chapter 09);
   report **p50, p95, p99, max**, and **compare http_req_waiting vs
   http_req_duration** to tell whether the service or the client is the
   bottleneck. **(4) no warm-up** — the first-request JIT / page-fault
   cost contaminates the percentiles; send a short warm-up at low QPS
   *before* the measurement window. **(5) one measurement, no
   reproducibility** — a load test is only useful if numbers move as
   expected when something changes: same generator machine, same service
   host, same network path, same load profile (arrival rate, ramp shape,
   duration, payload distribution), fresh-restart service with warm
   workers, excluded ramp-up from reported percentiles, and the artifact
   under test (version + git SHA + env vars) recorded alongside the
   results. Plus a sixth that the setup likely hit: **the generator is
   the bottleneck** — if Locust / wrk / k6 is pegged at 100% CPU, the
   latency includes queueing *inside the generator*; `htop` the generator
   during the run and prefer `wrk` / `k6` (both in C/Go, faster per-
   request than Locust's Python). What a real test produces: the
   **latency-throughput curve** — p95 (y) vs target QPS (x), flat below
   saturation and hockey-sticking upward at the **knee**. Report a
   sustained QPS a **25–50% margin below the knee** so bursts don't push
   the service into the steep zone. Tune in ranked order: **workers**
   (start at cores, measure N, N-1, N+2 — more workers past CPU
   saturation adds context switching with no throughput gain), **batch
   size** (amortise per-request overhead on `/predict:batch`),
   **model complexity**, **feature computation** (if `feature_ms`
   dominates `total_ms`, optimising the model call is wasted work),
   **serialisation** (orjson for large embedding responses), **framework
   choice** (onnxruntime on CPU, `torch==2.x+cpu` wheel, `n_jobs=1` at
   request time so workers provide the parallelism), **allocator** (last
   — the gains are single-digit percent). CI load testing is **regression
   detection**, not absolute validation: a short "p95 stays under X at Y
   QPS" k6 threshold assertion on every PR; a longer nightly curve on
   main. The absolute number depends on the runner's CPU; the delta
   between runs on the same runner is meaningful. **Change one variable
   at a time** — baseline vs tuned must differ on exactly one lever or
   the result is uninterpretable.
C. VUs are the right executor because they look like users; arrival-rate
   is academic.
D. Chasing 100% CPU on the load generator is the whole point of a load
   test.

### 11. (Chapter 11) A platform engineer tells the ML team: "we're happy to deploy your churn-service, just send us the container." The ML engineer replies "great, here it is — figure out the Kubernetes bits." Chapter 11 says this handoff is a contract, not a delivery, and names the specific deliverable plus the split of responsibilities. Which option captures both?

A. "Here is the image" is sufficient; the platform team is paid to figure
   out the rest.
B. The handoff is a **contract encoded in a `RUNBOOK.md`** that ships in
   the repo, not a container tossed over the wall. The runbook answers,
   in a format the platform team can nod through section by section:
   **image** (registry, tag policy — immutable per-SHA, not `latest`;
   base image; non-root UID; port), **endpoints** (`/predict`,
   `/predict:batch`, `/healthz`, `/readyz`, `/metrics`, `/model`, and
   whether `/docs` should be exposed in prod), **startup** (cold-start
   time + recommended `startupProbe` budget), **shutdown** (SIGTERM
   behaviour, that readyz flips to 503 first, recommended `preStop:
   sleep 15`, `--graceful-timeout 30`), **resources** (CPU / memory
   requests and limits, GPU y/n), **capacity** (the defended sentence
   from chapter 09: "sustains N QPS at p95 ≤ M ms with WORKERS=W on
   {cores} vCPU / {ram} GiB" — the number the HPA will consume),
   **configuration** (every env var, its default, what it does),
   **secrets** (names and what they're used for; never baked into the
   image), **model artifact** (baked in vs. object storage; metadata
   location), **observability** (log fields, metric names and labels,
   which alerts the ML team owns), **SLOs**, a **common-failure runbook
   table**, and **contacts / on-call rotation**. The responsibility split:
   **the ML engineer owns the service inside the container** — the
   artifact and its metadata (ch 02), the FastAPI app and schema (ch 03),
   health-check semantics (ch 04), structured logs and metrics surface
   (ch 05), the Dockerfile and image contents (ch 06), CI (ch 07), the
   batch scoring job (ch 08), the defensible latency-throughput budget
   (chs 09–10), the runbook, and **model-quality alerts** that require
   ML judgment (drift, staleness — `mod-109`). **The platform team owns
   everything outside the container**: Kubernetes manifests (Deployment,
   Service, Ingress, HPA, PDB), namespace / RBAC / service accounts,
   secrets management, autoscaling policy, service mesh and mTLS,
   ingress / TLS / WAF, deploy pipeline (rolling / canary / blue-green /
   rollback), cluster-level observability (Prometheus / Loki / Grafana /
   tracing), capacity planning, and cluster-level alerts. Some items
   **sit at the seam** and require explicit conversation: who maintains
   the base image, who provisions the model-artifact bucket, env var vs
   ConfigMap for which settings, mutable vs immutable image tags (always
   immutable per-SHA for defensible rollback), and alert ownership —
   model-quality alerts (drift, accuracy degradation) belong to ML,
   infra-level alerts (CPU, pod restarts, OOMKill) belong to platform.
   Things the ML engineer owns that the platform **cannot** do:
   model-quality monitoring, retraining pipelines and model promotion,
   feature-store schema evolution, prediction-quality investigations
   ("users say the recommendations are worse this week"). Kubernetes
   vocabulary the ML engineer should recognise — pod / deployment /
   service / ingress / HPA / probes / ConfigMap / Secret / PDB / preStop
   — without being a Kubernetes operator; depth belongs to the **AI
   Infrastructure Engineer** and **ML Platform Engineer** tracks, which
   are specifically out of scope for this module. Chapter 11 pitfalls,
   all common: no capacity story, same probe for liveness and readiness,
   no `graceful-timeout`, secrets baked into the image, mutable `latest`
   tag, high-cardinality metric labels (`user_id` / `request_id` in a
   Prometheus label blows up cardinality and pages the platform team),
   silent breaking schema changes during rolling deploys, no `/model`
   endpoint (the platform can't verify canary version splits), and
   "the platform team will figure it out" (they will — by pushing the
   ambiguity back onto you).
C. The RUNBOOK is paperwork for regulated industries only; a chat message
   is faster.
D. Kubernetes manifests and autoscaling policy belong to the ML engineer;
   the platform team only runs the registry.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **B.** Chapter 01 ("The gap between 'trains' and 'serves'" names the
   two gaps verbatim — implicit input contract, implicit environment —
   and "What a serving contract looks like" names the four contract
   parts: URL/request schema, response schema, health check, latency/
   throughput budget). The "miss any of the four and downstream teams
   will paper over the gap themselves" line is in the same section.

2. **B.** Chapter 02 ("The three questions the artifact has to answer"
   lists preprocessing, estimator, and metadata explicitly; "Common
   pitfalls" names re-implementing the scaler by hand as pitfall #2,
   pickling lambdas as #1, loading per request as #3, and unpickling
   untrusted files as #6). "Pickle is code" / "never unpickle a file
   from a source you do not trust" is in the chapter summary.

3. **B.** Chapter 03 ("Sync vs async route handlers" states the rule
   verbatim — sync for CPU-bound, async only when there is real awaited
   I/O, or wrap CPU work in `run_in_threadpool`). The `async def` with
   blocking prediction pattern is chapter 03 pitfall #3. Chapter 09's
   three-axes concurrency section ("Concurrency in Python") is the
   formalisation.

4. **B.** Chapter 04 ("Liveness vs. readiness — the precise distinction"
   is the chapter's opening framing; "Why the distinction matters in
   practice" contains the exact CrashLoopBackOff walkthrough with the
   Kubernetes default `periodSeconds=10, failureThreshold=3`; "The full
   lifecycle" is the startup → warm-up → serving → drain → exit diagram;
   "`/healthz`, `/livez`, `/readyz` — pick a convention" and the
   `startupProbe` recommendation are the rest).

5. **B.** Chapter 05 ("The logging model" states the three principles
   verbatim — structured over prose, `request_id` stitches events, model
   metadata on every prediction; "The request middleware" is the exact
   `RequestLogMiddleware` pattern; "What to log" and "What NOT to log"
   are explicit shortlists; "A structured logging setup" states stdout +
   UTC ISO-8601). Pitfall #6 is "Random `print()` calls bypass the
   formatter"; pitfall #4 is "File-based logs in a container"; pitfall
   #5 is "Local timezone in timestamps".

6. **B.** Chapter 06 ("Base image choice" names `python:3.11-slim` with
   a pinned digest; "Layer caching: `pip install` before `COPY .`" is
   the dependency-before-source rule; "`.dockerignore` — do not ship the
   dataset" is the exclude list; "Run as non-root" states the Kubernetes
   `runAsNonRoot` rejection reason; "`python:3.11-alpine` — the 'small
   image' temptation" explains the musl / manylinux wheel problem;
   "Passing configuration" is env-vars-not-baked-in). The ten "Common
   pitfalls" list enumerates all five breakages plus `PYTHONUNBUFFERED`,
   platform mismatches on ARM, CUDA torch on CPU nodes, and missing
   `HEALTHCHECK`.

7. **B.** Chapter 07 ("A smoke-test script" is the `scripts/smoke.sh`
   pattern — poll readyz, hit healthz and /predict, assert on shape not
   value; "Automated tests" is the unit-vs-integration split with
   `dependency_overrides` and `TestClient` in a `with` block;
   "GitHub Actions: the reference workflow" is the full `ci.yml` with
   `cache-from: type=gha`, `load: true, push: false`, `docker logs` with
   `if: always()`, and `docker rm -f` cleanup). Pitfall #1 is "No CI test
   at all" and pitfall #2 is "`--reload` in production".

8. **B.** Chapter 08 ("The two shapes at a glance" is the online-vs-batch
   comparison table; "The decision matrix" is the "any online condition
   true → online; all batch conditions true → batch" rule; "Hybrid
   pattern 1 — pre-compute, serve from cache" is the user-recommendation
   shape; "Hybrid pattern 2 — online inference with pre-computed
   features" is the feature-store pattern; "Hybrid pattern 3 — micro-
   batching on the request path" explains the latency/throughput trade
   and names Triton / Ray Serve / TorchServe / KServe / vLLM; "Batch
   scoring: the pragmatic patterns" lists idempotent output, manifest,
   schema check, recovery, and backfills). Pitfall #7 ("Micro-batching
   invented in-house") and pitfall #8 ("Feature drift between batch and
   online — share the code path") are both explicit.

9. **B.** Chapter 09 ("Latency, precisely" names the four measurement
   points; "Percentiles, not averages" states the p50/p95/p99/max rule
   and "the mean is essentially useless"; "Throughput, precisely"
   requires the quality-of-service clause; "Little's Law" states
   L = λ × W verbatim with both rearrangements; "The tail-latency
   amplifier" derives the `1 − (1-p99)^N` fan-out probability;
   "Concurrency in Python — the three axes" is processes / threads /
   async with the "Pattern matrix" table; "Memory — the silent
   constraint" is the model × workers OOM story).

10. **B.** Chapter 10 ("Where to run the load" rules out laptop-over-
    internet and warns about the generator as bottleneck; "Producing the
    latency-throughput curve" says use `ramping-arrival-rate`, not
    `ramping-vus`; "Reading the output" explains p50/p95/p99/max and the
    `http_req_waiting` vs `http_req_duration` comparison; "The
    reproducible load-test recipe" is the five-point list; "Tuning
    knobs, ranked by impact" is workers → batch size → model complexity
    → feature computation → serialisation → framework → allocator;
    "Load testing in CI" is the short-regression-gate pattern).
    Pitfall #1 is testing from a laptop, #2 is reporting the mean, #3
    is skipping warm-up, #4 is VUs instead of arrival-rate, #5 is the
    generator as bottleneck, #8 is changing multiple variables.

11. **B.** Chapter 11 ("The one-page handoff document" is the exact
    `RUNBOOK.md` template; "What the ML engineer owns" is the inside-
    the-container list; "What the platform team owns" is the outside-
    the-container list; "What sits at the seam" names the joint
    responsibilities; "Things the platform team will do — and what you
    owe them" covers autoscaling, rolling deploys, canary, multi-
    tenancy, and Kubernetes-shaped observability; "Things you own that
    the platform will not" names model-quality monitoring, retraining,
    feature-store schema evolution, prediction-quality investigations;
    "Kubernetes 101 — what to skim before the handoff meeting" is the
    vocabulary list; "Where the two tracks meet" explicitly hands off to
    the AI Infrastructure Engineer and ML Platform Engineer tracks).
    The ten "Common pitfalls" enumerate every failure in the question,
    including the cardinality-label and silent-schema-change ones.

</details>
