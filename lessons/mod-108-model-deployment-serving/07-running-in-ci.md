# 07 — Running the container locally and in CI

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 01–06.

## Motivation

An image that only ever runs on your workstation is little
better than a script in your home directory. Two milestones
lift it out:

1. **Runs locally with a single command.** A teammate clones
   the repo, types one command, and gets the service on
   `localhost:8080`. No "first, install Python 3.11 and
   these seventeen dependencies".
2. **Runs in CI on every commit.** Every pull request builds
   the image, starts it in a container, and asserts that
   `/healthz`, `/readyz`, and `/predict` all behave. A PR
   that breaks the service cannot merge.

Chapter 06 built the image. This chapter runs it — first
under `docker-compose` for local iteration, then under
GitHub Actions for CI. The code is small; the payoff is
being able to catch a broken service *before* it lands on
the platform team's plate.

## Local runs with `docker-compose`

`docker-compose` (or the newer `docker compose` — same tool,
different invocation) declares a stack of containers in
`compose.yaml`. For a single-service ML deployment this
looks like overkill; it is not, because it hides a dozen
`docker run` flags behind one command and gives you a
canonical place to add the next dependency (a Postgres for
integration tests, a Redis for caching, a mock feature
store).

```yaml
# compose.yaml
services:
  churn-service:
    build:
      context: .
      dockerfile: Dockerfile
    image: churn-service:local
    ports:
      - "8080:8080"
    environment:
      LOG_LEVEL: DEBUG
      WORKERS: "1"                     # single worker for easier debugging
      ARTIFACT_DIR: /artifact
    volumes:
      - ./artifact:/artifact:ro        # bind-mount for fast iteration
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/readyz"]
      interval: 10s
      timeout: 3s
      retries: 5
      start_period: 30s
```

Two conveniences worth noticing:

- **`volumes: ./artifact:/artifact:ro`** — the model is
  bind-mounted from the workstation. Retraining locally
  overwrites the file; the next request uses the new model
  without rebuilding the image. Set to `:ro` (read-only) to
  catch any accidental writes.
- **`WORKERS: 1`** — one process is easier to attach a
  debugger to; multi-worker gunicorn spreads requests across
  processes and confounds breakpoints.

Run the stack:

```
docker compose up --build       # builds image, starts container, streams logs
docker compose down             # tears down
```

For iterating on service code without rebuilding, mount the
source too:

```yaml
volumes:
  - ./app:/app/app:ro
  - ./artifact:/artifact:ro
```

Combined with `uvicorn --reload` in a development entrypoint,
edits to `app/main.py` are picked up in seconds.

**Do not run `--reload` in production.** It watches the
filesystem, is single-worker, and has known
memory-consumption issues under sustained edits. Use it
locally; use `gunicorn -w N` in the shipped image.

## A smoke-test script

The single script that tells you "the image is not
completely broken":

```bash
#!/usr/bin/env bash
# scripts/smoke.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8080}"

echo "→ healthz"
curl -sf "${BASE}/healthz" >/dev/null

echo "→ readyz"
# poll for up to 60s while the model loads
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/readyz")
  if [[ "${code}" == "200" ]]; then break; fi
  sleep 1
done
if [[ "${code}" != "200" ]]; then
  echo "readyz never returned 200"; exit 1
fi

echo "→ predict"
body='{"age":42,"plan_tier":"pro","months_since_signup":18}'
resp=$(curl -sf -X POST "${BASE}/predict" \
              -H "content-type: application/json" \
              -d "${body}")
echo "${resp}" | python -c \
  "import json,sys; b=json.loads(sys.stdin.read()); \
   assert 0.0 <= b['churn_probability'] <= 1.0; \
   assert b['model_version']"

echo "OK"
```

Three properties:

- **Poll for readyz.** The model load may take longer than
  the first probe. Retry with a timeout.
- **Assert on the response shape, not the value.** The
  service can legitimately return any probability; asserting
  `0.87` couples the test to the specific weights.
- **Assert on `model_version` being present.** Catches the
  regression where someone removes it from the response.

Run it locally: `docker compose up -d && ./scripts/smoke.sh`.
Same script runs in CI; the abstraction is thin on purpose.

## Automated tests

The smoke test is the low bar. Above it: real tests that
exercise the service without needing a container.

**Unit tests** — pytest against the FastAPI app directly,
using its test client. No HTTP, no container.

```python
# tests/test_predict.py
from fastapi.testclient import TestClient

from app.main import app, get_bundle
from serving.load import ModelBundle


class FakeModel:
    def predict_proba(self, X):
        return [[0.1, 0.9]] * len(X)


def _fake_bundle():
    return ModelBundle(model=FakeModel(),
                       metadata={"model_name": "test",
                                 "model_version": "0.0.0",
                                 "git_sha": "abc123"})


def test_predict_happy_path():
    app.dependency_overrides[get_bundle] = _fake_bundle
    with TestClient(app) as client:
        r = client.post("/predict",
                        json={"age": 42, "plan_tier": "pro",
                              "months_since_signup": 12})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["model_version"] == "0.0.0"
    app.dependency_overrides.clear()


def test_predict_bad_input_rejects():
    app.dependency_overrides[get_bundle] = _fake_bundle
    with TestClient(app) as client:
        r = client.post("/predict",
                        json={"age": -1, "plan_tier": "pro",
                              "months_since_signup": 12})
        assert r.status_code == 422
    app.dependency_overrides.clear()
```

Two things earn their keep:

- **`app.dependency_overrides`** — the entire reason
  chapter 03 used `Depends(get_bundle)`. Swap in a fake
  bundle; no real model required.
- **`TestClient(app)` in a `with` block** — this triggers
  the `lifespan`, so any lifecycle logic runs.

**Integration tests** — pytest against the running
container. Slower; catch things unit tests can't (the
Dockerfile is wrong, the artifact isn't in the image, the
gunicorn worker crashes on startup). Use `docker compose`
from the tests, or run the smoke script from a pytest
wrapper.

```python
# tests/test_container.py
import subprocess
import time

import httpx
import pytest


@pytest.fixture(scope="module")
def container():
    subprocess.check_call(["docker", "compose", "up", "-d", "--build"])
    for _ in range(60):
        try:
            r = httpx.get("http://localhost:8080/readyz", timeout=1)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    else:
        pytest.fail("service never became ready")
    yield "http://localhost:8080"
    subprocess.check_call(["docker", "compose", "down"])


def test_container_predict(container):
    r = httpx.post(f"{container}/predict",
                   json={"age": 42, "plan_tier": "pro",
                         "months_since_signup": 12})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
```

Mark integration tests with `-m integration` and run them
separately from unit tests so CI can run the fast tests on
every push and the slow ones on PR.

## GitHub Actions: the reference workflow

```yaml
# .github/workflows/ci.yml
name: ci

on:
  pull_request:
  push:
    branches: [main]

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          version: latest
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run pytest -m "not integration" --junitxml=junit.xml
      - if: always()
        uses: actions/upload-artifact@v4
        with:
          name: junit
          path: junit.xml

  build:
    needs: unit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build image
        uses: docker/build-push-action@v6
        with:
          context: .
          push: false
          load: true
          tags: churn-service:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Start container
        run: |
          docker run -d --rm --name churn \
            -p 8080:8080 -e WORKERS=1 \
            churn-service:ci

      - name: Wait for readiness
        run: |
          for i in {1..60}; do
            code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/readyz)
            [[ "$code" == "200" ]] && exit 0
            sleep 1
          done
          docker logs churn
          exit 1

      - name: Smoke test
        run: ./scripts/smoke.sh

      - name: Print logs
        if: always()
        run: docker logs churn

      - name: Tear down
        if: always()
        run: docker rm -f churn
```

Notable choices:

- **Two jobs, unit tests then build.** Fast tests run
  first; slow build is skipped if a unit test fails.
- **`cache-from: type=gha`** — BuildKit's GitHub Actions
  cache backend. Dependency install layers are cached
  between runs; the first build after a dependency change
  is slow, subsequent builds are fast.
- **`load: true, push: false`** — build the image into the
  local Docker daemon so subsequent `docker run` steps can
  use it. Do not push to a registry from CI on every
  commit; that is a separate job on tag or merge to main.
- **`docker logs churn` on failure.** When the smoke test
  fails, the container logs are the only thing that will
  explain why. `if: always()` guarantees they appear even
  if a previous step failed.
- **`docker rm -f churn`** in a cleanup step so a hung
  container does not leave the runner in a weird state.

For pushing to a registry on merge to `main`, add a job
gated on `github.ref == 'refs/heads/main'` that
authenticates to the registry (GitHub Container Registry,
ECR, GAR) and calls `docker push`. Chapter 11 mentions
platform-team registries; the specific `push` step is
platform-specific enough that the exercises leave it as an
extension.

## Managing the model artifact in CI

The container tests need an artifact. Three options:

1. **Commit a small fixture artifact.** For a scikit-learn
   toy model, the file is a few hundred kilobytes; commit
   it under `tests/fixtures/artifact/` and mount it in the
   test. Simple.
2. **Train a fixture in a CI setup step.** For a slightly
   larger model, run the training script in a pre-step and
   let the image pick up the resulting artifact.
3. **Pull from object storage.** Real production artifact
   handling. The CI job assumes read credentials to the
   artifact bucket. More setup; more realistic.

For the level of this module, (1) is the recommendation.
Real production services pick (3), often with the artifact
version pinned by an env var so a CI build tests against a
specific model version.

## Debugging a failing CI

Three commands do 90% of the work:

- **`docker logs <container>`.** The logs from chapter 05
  are structured JSON; pipe through `jq` locally.
- **`docker exec -it <container> sh`.** Poke inside the
  running container. `cat /artifact/metadata.json`, `ls
  /app/app/`, `curl localhost:8080/readyz`.
- **`docker inspect <container>`.** State, health, ports,
  environment. Answers "did my `-e` flag actually reach
  the container?".

In CI you can't attach interactively, but you *can* add a
step that runs `docker exec churn cat /artifact/metadata.
json` when the smoke test fails — often enough to identify
the cause without a re-run.

## Local dev, without Docker

Not every debugging session needs the container. Two
useful modes:

- **`uvicorn app.main:app --reload`** — Python-in-your-
  venv development. Fastest iteration; misses any bug that
  is caused by the container's environment.
- **`docker compose up`** — the container, without CI. The
  compose file above uses bind mounts to keep iteration
  fast.

Rule of thumb: `uvicorn --reload` when tweaking route logic;
`docker compose up` when tweaking anything the Dockerfile
touches (dependencies, environment variables, gunicorn
flags).

## Common pitfalls

1. **No CI test at all.** Every "the image builds" is not
   a test; the image can build and immediately crash on
   start. Actually hit `/readyz` and `/predict`.
2. **`--reload` in production.** Watches filesystem, single
   worker, silently degrades. Development only.
3. **`docker compose up` without `--build`.** Uses a cached
   image; your Dockerfile change is ignored. `--build`
   forces a rebuild.
4. **No `docker logs` on failure in CI.** The failure
   message is "curl returned 000"; the actual traceback is
   in the container that got torn down. Print logs before
   teardown.
5. **Integration tests that share port 8080 with other
   tests.** Race conditions between test runs. Use a
   per-test port or a docker network.
6. **Bind-mounting the whole repo.** The container sees
   your `.venv/`, `.git/`, `data/` — slow, and can hide
   Dockerfile bugs (the container works because your local
   files are patched in). Mount `app/` and `artifact/`
   specifically.
7. **Fixture artifact drift.** The CI artifact was trained
   against an old feature schema; the current code expects
   a new one. Regenerate the fixture on training-code
   changes, or check it in with metadata that lets the
   test detect a mismatch.
8. **Ignoring `docker buildx` for cross-arch.** Build on
   ARM Mac, deploy to x86; the image works locally and
   errors on the runner. Set `--platform linux/amd64` in
   the build step, or use `buildx` for multi-arch.

## Summary

- `docker-compose` gives you a one-command local stack:
  build, run, mount, tear down.
- A smoke-test script that hits `/healthz`, `/readyz`, and
  `/predict` is the low-bar contract every image must
  satisfy. It runs locally and in CI unchanged.
- Two layers of tests: unit (pytest + `TestClient`) that
  runs fast on every push, integration (pytest + docker)
  that runs on PR.
- GitHub Actions: a unit job, then a build+smoke job.
  Cache the Docker layers with `type=gha`. Print container
  logs on failure.
- Ship a fixture artifact with the tests for reproducible
  CI. Real production services pull from object storage,
  version-pinned.

## Where this goes next

Chapter 08 pivots to the *other* inference path. Everything
so far has been online (a request in, a prediction out).
Batch inference — scoring millions of rows on a schedule —
has entirely different constraints and a different
architecture. Chapter 08 covers both paradigms and the
hybrid pattern where they meet.
