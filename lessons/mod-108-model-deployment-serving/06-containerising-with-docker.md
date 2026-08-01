# 06 — Containerising the service with Docker

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01–05.
> **Prerequisite installs:** Docker Desktop (or `docker` +
> `containerd`) on your workstation. This chapter's examples
> use BuildKit, the default on modern Docker.

## Motivation

The service works on your laptop. On someone else's laptop,
it does not — different Python version, different `numpy`
minor, different libc. In production, none of the "someone
else's laptop" excuses apply: the service either runs or it
does not.

A container image is the artifact that solves the "runs on
my machine" problem. It contains an OS-level filesystem, the
right Python interpreter, your dependencies pinned to exact
versions, your service code, and a well-defined entry
command. Anywhere a container runtime exists — a laptop, a
CI runner, an ECS task, a Kubernetes pod — the image runs
identically.

This chapter builds the image. The next one runs it locally
and in CI. The two together give you a service that a
platform team can deploy without touching your Python
environment.

## The one-file version, then the refinements

The smallest Dockerfile that would ship the service from
chapter 03:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
 && uv sync --frozen --no-dev

COPY app/ ./app/
COPY artifact/ /artifact/

EXPOSE 8080
CMD ["uv", "run", "gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", "-b", "0.0.0.0:8080"]
```

Build: `docker build -t churn-service:0.1.0 .`
Run: `docker run --rm -p 8080:8080 churn-service:0.1.0`
Test: `curl -s -X POST localhost:8080/predict -H 'content-type:
application/json' -d '{"age":42,"plan_tier":"pro","months_since_signup":18}'`

This works. It also has some rough edges the rest of the
chapter fixes: it does not cache dependency installation
between builds, it runs as root, the image is larger than it
needs to be, and there is no `HEALTHCHECK`.

## Base image choice

The `FROM` line is the single most consequential decision.

### `python:3.11-slim` — the pragmatic default

The `-slim` variants are Debian-based, stripped of most
non-essential Debian packages. They are ~50 MB unpacked,
have `pip` and a functional glibc, and include enough of a
compiler toolchain for many pip-only dependencies to install.

**When it's right:** most Python ML services. Every ML
dependency wheels-clean on this platform. This is what to
use unless you have a specific reason otherwise.

### `python:3.11` (full) — the "just make it work" option

Includes build essentials, git, and more. Larger (~300–400
MB unpacked) but sometimes needed if a dependency requires
building from source. Ship the slim variant to production
after debugging.

### `python:3.11-alpine` — the "small image" temptation

Alpine uses musl instead of glibc. Many Python ML wheels
(`numpy`, `pandas`, `scipy`, `torch`) publish binaries only
for `manylinux` (glibc). On Alpine, `pip install numpy`
falls back to compiling from source — slow, fragile, and
often broken.

**When it's right:** rarely, for ML. Do not use Alpine as
the ML serving base image unless you have measured that it
saves more than it costs; the tradeoff usually goes the
other way.

<!-- needs-research: as of 2026, confirm which of the major
Python ML libraries (numpy, pandas, scikit-learn, torch,
lightgbm, xgboost) publish musl wheels. -->

### Framework-specific bases

`pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime`,
`tensorflow/tensorflow:2.x-gpu`, `nvcr.io/nvidia/tensorrt:...`
— framework-provided images with GPU support baked in. Use
them for GPU inference; a stock `python:3.11-slim` cannot
serve CUDA workloads.

### Distroless

Google's `gcr.io/distroless/python3-debian12` images strip
the shell and package manager, minimising attack surface.
Difficult to debug (no shell) but excellent security posture
for production. Consider once your service is stable.

**Pin the base image to a digest** — `python:3.11-slim@sha256:
b1f9a...` — so a rebuild months later cannot change the base
underneath you. `docker inspect --format '{{index .RepoDigests
0}}' python:3.11-slim` gets the digest.

## Layer caching: `pip install` before `COPY .`

Docker builds are layered; each instruction produces one
layer; layers are cached by input. The single most important
Dockerfile discipline is to **install dependencies before
copying source code**, so a code change does not invalidate
the (expensive) dependency install layer.

Wrong:

```dockerfile
COPY . .
RUN pip install -r requirements.txt      # re-runs on every source change
```

Right:

```dockerfile
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt   # cached
COPY app/ ./app/                                     # cheap layer
```

The `--no-cache-dir` flag on `pip install` tells pip not to
save its own cache in the layer — the wheel cache is useful
during install but only bloats the image afterwards.

If you use `uv`:

```dockerfile
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
COPY app/ ./app/
RUN uv sync --frozen --no-dev --no-editable
```

`--mount=type=cache` is BuildKit's cache mount — persistent
across builds but not shipped in the image. On repeat builds
the uv cache is reused.

## Multi-stage builds

Half the reason to reach for multi-stage: keep build-time
tooling (compilers, dev headers, test frameworks) out of the
runtime image. The runtime image ships only what production
needs.

```dockerfile
# --- build stage ---------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
 && uv sync --frozen --no-dev

COPY app/ ./app/

# --- runtime stage -------------------------------------------
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 app \
 && useradd  --uid 1000 --gid 1000 --create-home app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app   /app/app
COPY artifact/ /artifact/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl --fail http://localhost:8080/readyz || exit 1

CMD ["gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", "-b", "0.0.0.0:8080", \
     "--timeout", "30", "--graceful-timeout", "30"]
```

What each block does:

- **`AS builder`** — a full build environment; installs
  compilers if a dependency needs to build. Discarded from
  the final image.
- **`AS runtime`** — the shipped image. Copies only the
  virtualenv and the source from the builder.
- **`useradd app`** — a non-root user (see below).
- **`ENV PYTHONUNBUFFERED=1`** — Python writes stdout
  synchronously. Otherwise logs buffer and disappear on
  crash.
- **`ENV PYTHONDONTWRITEBYTECODE=1`** — no `.pyc` files;
  a tiny image-size win.
- **`HEALTHCHECK`** — Docker's own health check, distinct
  from Kubernetes probes. Useful for local `docker ps`.
  `--start-period` is the model-load window during which
  failures don't count.
- **`--graceful-timeout 30`** — gunicorn's window for
  in-flight requests to finish on SIGTERM (chapter 04).

## Run as non-root

Two reasons to run the container as a non-root user:

- **Blast radius.** A container escape from a non-root
  process cannot easily escalate on the host.
- **Kubernetes `runAsNonRoot` policies.** Many platform
  teams reject images that run as UID 0. If your image
  ships as root, the platform team's admission controller
  rejects it and you have to redo the base.

The Dockerfile snippet above creates `app` with UID 1000,
chowns the source, and `USER app`s. All subsequent commands
run as that user.

If your service needs to write to a directory (a model
download cache, a log file), `chown -R app:app /path`
during the build so the process has write access.

## `.dockerignore` — do not ship the dataset

Every file in the build context (`docker build .`) is a
candidate for `COPY`. Without a `.dockerignore`, `COPY . .`
copies:

- Your `.git` directory (100s of MB, contains git history).
- Your test dataset (`data/`, potentially many GB).
- The `.venv` from your local development (fat).
- Local `.env` files (secrets).

The right `.dockerignore`:

```gitignore
# .dockerignore
**/__pycache__
**/*.pyc
**/*.pyo
**/.pytest_cache
**/.ruff_cache
**/.mypy_cache

.git
.gitignore
.github/

.venv/
env/
node_modules/

data/
notebooks/

.env
.env.*
*.env
secrets/
credentials.json

tests/
docs/
Dockerfile
docker-compose.yml
.dockerignore
README.md
```

Note **`data/`** and **`.env`** — the two most common causes
of embarrassing image contents. Also `Dockerfile` itself,
which the build does not need to see.

## Passing configuration: environment variables

The service should not have production configuration baked
in. Anything that varies per environment is an environment
variable read at startup:

```python
# app/config.py
import os
from pydantic import BaseModel


class Settings(BaseModel):
    artifact_dir: str = os.getenv("ARTIFACT_DIR", "/artifact")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    workers: int = int(os.getenv("WORKERS", "4"))
    graceful_timeout_s: int = int(os.getenv("GRACEFUL_TIMEOUT_S", "30"))


settings = Settings()
```

Then `docker run -e LOG_LEVEL=DEBUG -e WORKERS=8 ...`. The
platform team sets these in their Kubernetes ConfigMap /
Secret; you promise the service will honour them.

**Never bake secrets into the image.** Not database
passwords, not API keys, not model-registry credentials.
Pass them at run time via environment variables (from
Secrets), a mounted file, or a metadata service.

## Model artifact: build-in vs. mount-at-startup

Two viable patterns:

### Baked into the image

```dockerfile
COPY artifact/ /artifact/
```

- **Pros.** One image → one model version. Immutable
  artifact. Simple rollout: deploy the new image.
- **Cons.** Image is model-sized (hundreds of MB for many
  models, gigabytes for LLMs). Every model change is an
  image build. Cannot A/B two versions of the model from
  the same image.

Use this for small models with infrequent updates. It is the
default for this module's exercises.

### Downloaded at startup

```dockerfile
ENV MODEL_URI=s3://models/churn/latest/
CMD ["python", "-m", "app.download_and_serve"]
```

Where `download_and_serve` fetches the artifact from
`MODEL_URI` at startup, then runs gunicorn.

- **Pros.** Small image, decoupled from model version.
- **Cons.** Startup takes longer (readyz stays `503` during
  download). Requires network access to the artifact store
  at boot.

Use this when the same image serves many models, or the
model is large enough that image size becomes painful.
`mod-106` chapter 07 (model registry) is the natural source.

## The image-size mindset

An ML serving image often lands between 500 MB and 3 GB —
`torch` alone is ~600 MB, `numpy + pandas + scikit-learn`
around 200 MB, add a model and you are already at 1 GB. That
is normal; do not chase 100-MB Alpine dreams for their own
sake. The wins that matter:

- **Multi-stage** cuts hundreds of MB by dropping build
  tooling. Do this always.
- **`--no-cache-dir` on pip** saves ~100 MB.
- **Remove `apt` lists** (`rm -rf /var/lib/apt/lists/*`)
  after `apt-get install`.
- **Pick the right torch build.** `torch==2.x+cpu` is ~200
  MB; `torch==2.x` (with CUDA) is ~2 GB. If you serve on
  CPU, install the `+cpu` build explicitly — pytorch.org's
  install matrix has the exact wheel URL.
- **BuildKit cache mounts** for `pip` / `uv`. Speeds
  rebuilds, does not touch shipped size.

Focus on layer *hit rate* over absolute size. A 2 GB image
where the top layer is 1 KB of application code will deploy
in seconds because only that layer transfers. A 500 MB
image where every layer changes on every commit will deploy
slower.

## Reproducible builds

Two flags matter:

- **Pin the base image.** `FROM python:3.11-slim@sha256:...`.
  Without the digest, a rebuild months later can silently
  pull a different `python:3.11-slim` (base OS updated).
- **Pin every dependency.** `uv.lock` / `poetry.lock` /
  `pip-tools`' `requirements.txt`. `mod-102` chapter 07
  covered this; the container is where the discipline pays
  off.

A `docker build` in six months should produce the same image
digest as today. If it does not, someone is going to have a
bad afternoon debugging why.

## Local smoke test

Before pushing, prove the image works:

```bash
docker build -t churn-service:0.1.0 .
docker run --rm -d --name churn -p 8080:8080 churn-service:0.1.0

curl -s http://localhost:8080/healthz
curl -s http://localhost:8080/readyz
curl -s -X POST http://localhost:8080/predict \
     -H 'content-type: application/json' \
     -d '{"age":42,"plan_tier":"pro","months_since_signup":18}'

docker logs churn | tail -20
docker rm -f churn
```

Watch for four things:

- Both endpoints return `200` (chapter 04).
- `/predict` returns a JSON body with `model_version`
  (chapter 03).
- Logs on stdout are JSON (chapter 05).
- Image size (`docker images churn-service:0.1.0`) is what
  you expect.

Chapter 07 automates this in CI.

## Common pitfalls

1. **`COPY . .` without `.dockerignore`.** Ships secrets and
   data. Image balloons; leaked credentials in registries
   are common enough that GitHub scans for them.
2. **`pip install` before `COPY pyproject.toml`.** Every
   source change invalidates the dependency layer; builds
   take 10× as long as they need to.
3. **Root user.** Rejected by platform admission
   controllers; larger blast radius on escape.
4. **`PYTHONUNBUFFERED=0` (default).** Logs disappear on
   crash. Set to `1`.
5. **Alpine base for ML.** `pip install numpy` compiles
   from source; the image build takes 20 minutes and is
   fragile. Use `python:3.11-slim`.
6. **`FROM python:latest`.** Never. Pin the minor version;
   pin the base digest.
7. **Secrets in the image.** Baked-in database URLs, API
   keys. Passed via env vars at run time, not build time.
8. **CUDA torch installed on CPU-only nodes.** Doubles or
   triples the image size for no benefit. Use the CPU-only
   torch wheel.
9. **No `HEALTHCHECK`.** `docker ps` reports "up" for a
   process that is crashing internally. Add one; use
   `--start-period` to match the model load time.
10. **Building on ARM Mac, deploying to x86 Linux.** The
    image runs slowly under emulation or fails to start.
    Build with `--platform linux/amd64` (or set up a
    multi-arch build with `docker buildx`).

## Summary

- Docker turns "runs on my machine" into "runs the same
  everywhere". A Dockerfile is a short recipe for the
  image; the discipline is in the details.
- Base image: `python:3.11-slim`, pinned to a digest.
  Multi-stage build to keep runtime lean.
- Install dependencies before copying source so the
  dependency layer caches. Use BuildKit cache mounts for
  the pip / uv cache.
- Run as non-root, stream logs to stdout, use environment
  variables for config, keep secrets out of the image.
- `.dockerignore` aggressively — no `.git`, no `data/`, no
  `.env`. Model artifact either baked in (small model) or
  downloaded at startup (large model, many models per
  image).
- Smoke-test the image locally before pushing.

## Where this goes next

Chapter 07 wires the image into CI: an integration test that
starts the container, hits `/predict`, and asserts the
response. The same commands work in a GitHub Actions
workflow; the same commands work in `docker-compose` for
local development.
