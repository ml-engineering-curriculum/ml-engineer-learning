# exercise-02 — Dockerise the service

> **Estimated effort:** 3–4 hours.
> **Companion chapters:** [06 — Containerising the service with Docker](../06-containerising-with-docker.md),
> [07 — Running the container locally and in CI](../07-running-in-ci.md).
> **Prerequisite:** exercise-01 (the FastAPI service exists).

## Objective

Take the FastAPI service from exercise-01 and produce a
production-shaped container image, wired into local
development via `docker-compose` and into CI via GitHub
Actions. By the end you should have:

- A **multi-stage Dockerfile** that produces a small,
  non-root, health-checked image.
- A **`.dockerignore`** that keeps secrets and data out of
  the build context.
- A **`compose.yaml`** for one-command local runs.
- A **smoke-test script** (`scripts/smoke.sh`) that
  exercises `/healthz`, `/readyz`, and `/predict`.
- A **GitHub Actions workflow** that builds the image,
  runs the container, runs the smoke test, and (on
  failure) surfaces the container logs.

The success criterion is that a teammate could clone the
repo, type `docker compose up`, get a working service, and
that a broken change cannot merge to `main` without CI
noticing.

## Prerequisites

- Exercise-01 complete.
- Docker Desktop or equivalent (`docker` + `containerd`)
  with BuildKit enabled.
- A GitHub repository (fork this exercise into your own to
  run Actions).

## Problem statement

### Part A — The Dockerfile (≈ 60 min)

Write a multi-stage `Dockerfile` following chapter 06:

- **`builder` stage** based on `python:3.11-slim`
  (pinned to a digest). Installs `uv` (or `pip-tools`, or
  Poetry), syncs deps into a virtualenv, copies the `app/`
  source.
- **`runtime` stage** based on the same `python:3.11-slim`
  digest. Creates a non-root `app` user (UID 1000). Copies
  the virtualenv and source from the builder. Copies the
  `artifact/` directory to `/artifact`.
- Sets `ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1`.
- Runs as `USER app`.
- Includes a `HEALTHCHECK` that curls `/readyz`, with
  `--start-period` matching the model load time.
- `CMD` runs `gunicorn` with `UvicornWorker`, workers
  configurable via `WORKERS` env var, `--graceful-timeout
  30`.
- Every configurable value is an env var; nothing is
  hard-coded that varies between environments.

Also write **`.dockerignore`** that excludes at least:
`__pycache__`, `.git`, `.venv`, `.env*`, `data/`, `tests/`,
`notebooks/`, `Dockerfile`, `.dockerignore`, `README.md`.

**Report the final image size.** Include it in the
`NOTES.md`. Also report the size before you applied
multi-stage — a simple single-stage image built from your
service. The delta is the win you paid for.

### Part B — Compose and smoke test (≈ 45 min)

Write `compose.yaml`:

- One service: `churn-service` (or whatever your service is).
- Builds from `.`.
- Binds `./artifact` into `/artifact` as read-only.
- Environment: `LOG_LEVEL=DEBUG`, `WORKERS=1`.
- Exposes port `8080`.
- Docker `healthcheck` block matching the Dockerfile
  `HEALTHCHECK`.

Write `scripts/smoke.sh` (chapter 07):

- Reads `BASE` from env (default `http://localhost:8080`).
- Polls `/readyz` for up to 60 s; fails loudly if never
  ready.
- Curls `/healthz`; expects 200.
- Curls `POST /predict` with a valid body; asserts response
  has a `model_version` and a probability in `[0, 1]`.
- Exits non-zero on any failure with a useful message.

Verify manually:

```
docker compose up --build -d
./scripts/smoke.sh
docker compose logs churn-service | tail -20
docker compose down
```

All four steps should succeed on a clean checkout.

### Part C — GitHub Actions (≈ 60 min)

Add `.github/workflows/ci.yml` with two jobs (chapter 07):

- **`unit`** — checkout, install with `uv`, run `ruff check`
  and `pytest -m "not integration"`. Uploads the junit
  report as an artifact.
- **`build`** — depends on `unit`. Sets up BuildKit
  (`docker/setup-buildx-action`). Builds the image with
  GHA cache (`cache-from/cache-to: type=gha`) and
  `load: true`. Starts the container with
  `WORKERS=1`. Polls `/readyz`. Runs `scripts/smoke.sh`.
  Prints `docker logs` on failure. Tears down cleanly in
  an `if: always()` step.

The workflow triggers on `pull_request` and on `push` to
`main`. Add a status badge to `README.md`.

### Part D — Break it deliberately (≈ 30 min)

To prove CI actually works, do the following on a branch:

- Introduce a syntax error in `app/main.py`. Push. Confirm
  the `unit` job fails.
- Revert. Now introduce a **runtime** bug: pass
  `WORKERS=-1` (invalid gunicorn arg) in the workflow.
  Push. Confirm the `build` job fails and the printed
  `docker logs` shows gunicorn's error.
- Revert. Now remove `/readyz` from the service. Push.
  Confirm the `build` job's readiness poll times out and
  fails.
- Revert. Push a passing branch.

Include screenshots or the run URLs in `NOTES.md` for
each of the three failure modes. The point is that you
have *seen* CI catch things, not that you have shipped a
green workflow.

## Starter guidance

- **Use `python:3.11-slim` as the base.** Do not reach for
  Alpine; chapter 06 explains why. If a dependency fails
  to build in `-slim`, install its `apt` prerequisites
  (`libgomp1`, `libgcc-s1`, etc.) in the builder stage
  only.
- **`docker buildx` for cross-arch.** If you are on Apple
  Silicon and the CI runner is x86-64, add
  `--platform linux/amd64` to your local build, or use
  `docker buildx build` for a multi-arch image.
- **Chapter 07's workflow works as a starting point** —
  adapt it, don't clone it. The exercise is testing that
  you can reason about the pieces.
- **`docker logs` is your friend in CI.** When the smoke
  test fails, log output is your only signal. Do not
  swallow it.
- **`.dockerignore` first.** A `COPY . .` before you write
  the ignore file will ship your `.git/` directory and
  every checkpoint in `data/`. Fix ignore, then rebuild
  and check `docker image ls` for the smaller size.

## Acceptance criteria

- `docker build -t <image>:local .` produces an image
  smaller than the equivalent single-stage build. Report
  both sizes in `NOTES.md`.
- The image runs as UID 1000, not root
  (`docker run --rm --entrypoint id <image>:local` shows
  `uid=1000`).
- `docker compose up --build` starts the service; the
  smoke script passes against `localhost:8080`.
- `docker inspect <container>` shows a healthy status
  once the model has loaded (`HEALTHCHECK` is functional).
- The GitHub Actions workflow succeeds on the passing
  branch and fails on each of the three deliberate
  breakage branches from Part D.
- No secrets in the image (`docker history <image>` shows
  no environment values, no baked-in credentials).

## Stretch goals (optional)

- **Distroless runtime.** Swap the runtime base to
  `gcr.io/distroless/python3-debian12`. Measure size and
  cold-start impact. Note: no shell, no `curl` — the
  Docker `HEALTHCHECK` needs to use `python -c` instead.
- **Model in object storage.** Rework the image so the
  artifact is not baked in; the container downloads it
  from an S3 / MinIO path at startup. Measure the
  additional cold-start cost. Keep readiness `503` until
  download completes.
- **Multi-arch build.** `docker buildx` for both
  `linux/amd64` and `linux/arm64`. Push to a registry.
  Verify a pull on both architectures.
- **Push to GHCR on main.** Add a third CI job, gated on
  `github.ref == 'refs/heads/main'`, that authenticates
  to GHCR and pushes the image tagged with the commit
  SHA.
- **Base image scan.** Add a `trivy image` (or `grype`,
  or `docker scout`) step to CI that fails the build on
  HIGH/CRITICAL vulnerabilities in the base image.
- **Reproducible build check.** Run `docker build` twice
  and diff `docker image ls` sizes / manifest. Bonus if
  the layers hash identically (harder — requires
  `SOURCE_DATE_EPOCH` and build determinism).

## What to hand in

```
exercise-02/
├── Dockerfile
├── .dockerignore
├── compose.yaml
├── scripts/
│   └── smoke.sh
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/                      # from exercise-01
├── tests/                    # from exercise-01
├── training/                 # from exercise-01
├── artifact/                 # from exercise-01
├── pyproject.toml
├── README.md                 # includes CI badge, sizes, run instructions
└── NOTES.md                  # image-size delta, three failure-mode screenshots/URLs
```

Solutions live in the paired `ml-engineer-solutions` repo.
