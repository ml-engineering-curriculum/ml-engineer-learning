# 05 — Pinning environments: lockfiles, Docker, hardware

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** `mod-102` (Python toolchain,
> `uv` / Poetry basics), chapter 04 of this module.

## Motivation

Chapter 04 pinned the random seed. But if the *code* around
the seed is a different version of PyTorch, a different
cuDNN, or a slightly different version of a preprocessing
library, the seed is meaningless.

A run is a function of:

- The code you wrote (git SHA).
- The data you fed it (chapter 06).
- The seed (chapter 04).
- **The environment the code ran in.** This chapter.

The environment includes:

- The Python interpreter version.
- The exact versions of every direct and transitive
  dependency.
- Native libraries linked into wheels (BLAS, CUDA, cuDNN).
- The operating system and CPU/GPU architecture.

We control the first three with **lockfiles** and pin the
last three with **container images** identified by digest.
For the pieces we cannot pin (hardware), we **log them** in
the run so a reader can tell whether a discrepancy is a bug
or a hardware difference.

## The lockfile: not the same as `requirements.txt`

A `requirements.txt` with pinned versions is *almost* a
lockfile — but not quite. Two gaps:

- **Transitive dependencies drift.** You pin `pandas==2.2.0`;
  `pandas` requires `numpy>=1.23`. Any `numpy` in that range
  can install. Tomorrow's `numpy` release changes a numeric
  edge case. Your `requirements.txt` is unchanged; your
  install is not.
- **Wheel hashes are unverified.** Without `--hash` lines,
  `pip install` will happily accept a re-uploaded (or
  compromised) wheel that has the same version but different
  bytes.

A real lockfile pins every direct *and* transitive
dependency to an exact version and, ideally, to a specific
wheel hash. Modern Python tooling produces one automatically:

- **`uv lock`** produces `uv.lock` — TOML, hash-pinned, one
  entry per resolved package. Reproducible from `uv sync`.
- **`poetry lock`** produces `poetry.lock` — same idea, older
  ergonomics. Reproducible from `poetry install`.
- **`pip-tools`** — `pip-compile requirements.in →
  requirements.txt` with `--generate-hashes`. Older but
  works.

`mod-102` chapter 07 covers the toolchain choice; this
chapter assumes you have one. **Whichever you use, commit
the lockfile.** A repo without a lockfile is a repo without
reproducibility.

### The three commands you actually run

For `uv`:

```bash
# Lock the resolution based on pyproject.toml
uv lock

# Install exactly what the lockfile says (fails if there's
# any drift from the lock)
uv sync --frozen

# Add a new dependency (updates the lock)
uv add "mlflow>=2.10"
```

For Poetry, replace with `poetry lock`, `poetry install --sync`,
and `poetry add`.

The `--frozen` (uv) / `--sync` (Poetry) flag is the one that
matters for reproducibility: it means "install what the lock
says, exactly; do not try to resolve anything new". CI should
always use it.

## Pinning PyTorch specifically

PyTorch is the awkward dependency in most ML lockfiles
because it ships as different wheels for different CUDA
versions. `pip install torch` on a machine without an
NVIDIA driver gets you the CPU wheel; on one with a driver,
it can get a CUDA wheel with the driver's minor version
number baked into the wheel filename.

The reliable way to pin PyTorch is to use the official index
matching your target:

```toml
# pyproject.toml (uv style)
[project]
dependencies = [
    "torch==2.4.0",
    "torchvision==0.19.0",
]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
```

Two things this buys you:

- The wheel comes from the PyTorch index — no ambiguity
  about which CUDA build.
- The lockfile records the exact wheel URL and hash.
  `uv sync --frozen` on a fresh machine installs the same
  bytes.

For a CPU-only run, swap the index to `whl/cpu`. Read the
current PyTorch install page (`resources.md`) for the exact
URL — the matrix changes with each release.

<!-- needs-research: verify the current PyTorch install-page
URL and the recommended `uv sources` syntax on the latest
stable release; historically the guidance has shifted. -->

## Recording the environment inside a run

Even with a lockfile, you should log the *actual resolved*
environment inside the run. It confirms that the lockfile
was honoured, and it survives a broken lockfile.

The one-liner:

```python
import subprocess, mlflow

freeze = subprocess.check_output(["pip", "freeze"]).decode()
mlflow.log_text(freeze, "env/pip_freeze.txt")
```

More thoroughly:

```python
import json, platform, sys, subprocess, mlflow

def env_manifest() -> dict:
    return {
        "python":         sys.version,
        "platform":       platform.platform(),
        "processor":      platform.processor(),
        "machine":        platform.machine(),
        "pip_freeze":     subprocess.check_output(["pip", "freeze"]).decode(),
        "git_sha":        subprocess.check_output(
                              ["git", "rev-parse", "HEAD"]).decode().strip(),
        "git_diff":       subprocess.check_output(
                              ["git", "diff", "HEAD"]).decode(),
    }

mlflow.log_dict(env_manifest(), "env/manifest.json")
```

Add GPU details when you have a GPU:

```python
import torch, mlflow

if torch.cuda.is_available():
    gpu = {
        "cuda":          torch.version.cuda,
        "cudnn":         torch.backends.cudnn.version(),
        "device_name":   torch.cuda.get_device_name(0),
        "device_count":  torch.cuda.device_count(),
        "driver":        subprocess.check_output(
                             ["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"]).decode().strip(),
    }
    mlflow.log_dict(gpu, "env/gpu.json")
```

Chapter 09's report template refers back to these files —
the reader who wants to rerun should be able to pull the
manifest and recreate the environment from it.

## Containerising the environment

A lockfile pins Python packages. It does not pin the OS
libraries the wheels are dynamically linked against, the
system `glibc`, or the CUDA runtime. For that, use a
container.

The minimum useful Dockerfile for a training image:

```dockerfile
# Dockerfile
# Pin the base image by SHA digest, not by tag.
# Tags like "python:3.11-slim" move. Digests don't.
FROM python:3.11-slim@sha256:{{FILL_IN_DIGEST}}

# Deterministic apt install of any native libs your wheels need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy just the resolver files first for cache-friendly builds.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv==0.4.0 \
    && uv sync --frozen --no-dev

COPY . .

ENTRYPOINT ["uv", "run", "python", "-m", "src.train"]
```

Two habits that make this reproducible:

- **Pin the base image by digest**
  (`python:3.11-slim@sha256:...`). Tags are mutable.
  `docker pull python:3.11-slim` today and next month can
  return different bytes; digest references never do.
- **Copy the lockfile first, then the code.** Docker cache
  reuses the dependency install layer for every code-only
  change — fast rebuilds, same environment.

For CUDA training, start from an NVIDIA base image
(`nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:...`) rather
than `python:*` — you inherit the CUDA runtime libraries at
the exact version you built PyTorch against.

Reference the image by digest in the tracker too:

```python
import subprocess, mlflow

image_id = subprocess.check_output(
    ["docker", "inspect", "--format={{.Id}}", "training:latest"]
).decode().strip()
mlflow.log_param("docker_image_id", image_id)
```

Now the run record answers "what container did this ship
in" — without which "same code" is a lie.

## Hardware — what to log, what to hope for

You cannot make a T4 return the same floating-point answer
as an H100. The best you can do is:

- **Log the hardware fingerprint** on every run: CPU model,
  RAM, GPU model and count, driver version.
- **Group comparisons by hardware.** In MLflow, tag runs
  with `hardware=A100-80GB`; filter to that hardware when
  reading variance numbers.
- **State the tolerance.** Chapter 04's tier-2
  ("tolerance-bounded on GPU") is the honest claim. Say what
  the tolerance is: "top-line accuracy within 0.1% across
  A100 and H100" is a real, testable statement.

Example hardware tag block, on top of the earlier env
manifest:

```python
mlflow.set_tags({
    "hardware.cpu":      platform.processor() or "unknown",
    "hardware.machine":  platform.machine(),
    "hardware.gpu":      torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu",
    "hardware.n_gpu":    str(torch.cuda.device_count()),
})
```

## The reproducibility triangle, revisited

Chapter 01 introduced code + data + config + env. Chapter 04
covered the seed (part of config); this chapter covers the
env. That leaves code (git, straightforward) and data
(chapter 06, DVC).

A concrete checklist of what a "reproducible" run's tracker
record contains after this chapter:

- [x] `mlflow.source.git.commit` — from MLflow autolog.
- [x] `code/uncommitted.diff` — if the working tree was
      dirty.
- [x] Explicit `mlflow.log_params` for every hyperparameter,
      including `seed`.
- [x] `env/pip_freeze.txt` — the resolved dependency list at
      run time.
- [x] `env/manifest.json` — Python, platform, git SHA, git
      diff.
- [x] `env/gpu.json` — CUDA / cuDNN / driver / device name
      when applicable.
- [x] `hardware.*` tags for filtering.
- [x] `docker_image_id` param — for containerised runs.
- [ ] `data/*` — dataset fingerprint; wired up in chapter 06.
- [ ] `model/*` — model artefact with signature; done in
      chapter 02 via `mlflow.<flavour>.log_model`.

Everything in this list is a couple of lines of code once
and a call in every training script. Chapter 09's
reproducibility report is essentially "here is where each of
these lives for this run".

## Two-tier CI, from the environment perspective

For a repo with proper environment pinning, CI runs two
distinct checks:

- **Fast unit check.** `uv sync --frozen`; run tests
  including chapter 04's byte-identical CPU determinism
  test. This should pass on every PR in a couple of minutes.
- **Slow environment check.** Build the training container
  from the pinned lockfile + base-image digest. Run a
  one-epoch smoke train inside it. Publish the container to
  a registry with the git SHA as the tag. This can run on
  merges to `main` only.

The point of the second check is not to train a model — it
is to prove the container is buildable and the training
script starts inside it. When a lockfile bit-rots (a wheel
disappears from PyPI, a base image is retired), the second
check catches it early and loudly.

## Common pitfalls

1. **No lockfile.** `requirements.txt` with `pandas` (no
   version) is not a pin. Adopt `uv` or Poetry; commit the
   lock.
2. **Lockfile committed but CI uses `pip install`.** Then
   the lock is decorative. CI must use the tool's install
   verb with the frozen flag (`uv sync --frozen`).
3. **Base image referenced by tag.** `FROM
   python:3.11-slim` is not reproducible. Pin the digest.
4. **PyTorch installed from the wrong index.** `pip install
   torch` without the PyTorch index gets a wheel whose CUDA
   version is a lottery. Configure the index explicitly.
5. **`pip freeze` inside the container missing from the run
   record.** If the lock ever drifts from the container,
   only the freeze reveals it. Log it every time.
6. **Environment logged but not used in queries.** Adding
   `hardware.*` tags is only useful if downstream comparison
   filters on them. Chapter 09's report template does.
7. **Notebook-only reproducibility.** Notebooks encourage
   in-place mutation of state. The reproducible thing is a
   script the notebook exports to (`jupytext` /
   `nbconvert --to script`). `mod-102` chapter 04's advice
   applies here: notebooks explore, scripts train.

## Summary

- Lockfiles pin Python; **commit** `uv.lock` /
  `poetry.lock`; use `--frozen` in CI.
- Containers pin the OS layer; reference the base image by
  **digest**, not by tag.
- PyTorch specifically requires pointing your resolver at
  the official PyTorch index for a deterministic CUDA build.
- Log the resolved environment inside every run —
  `pip freeze`, Python version, platform, git SHA, dirty
  diff, GPU model, CUDA / cuDNN / driver. MLflow makes each
  of these a one-line call.
- Hardware differences are the one thing you cannot pin. Log
  the hardware fingerprint, group comparisons by hardware,
  and state the tolerance in the reproducibility report
  (chapter 09).

## Where this goes next

Chapter 06 completes the provenance triangle by versioning
the *data* — the vertex not covered by any of code, config,
or environment. The tool is **DVC**.
