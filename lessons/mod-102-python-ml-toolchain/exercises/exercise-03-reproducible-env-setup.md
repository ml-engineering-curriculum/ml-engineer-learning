# exercise-03 — Reproducible environment setup

> **Estimated effort:** 2 hours.
> **Companion chapter:** [07 — Reproducible Python environments](../07-reproducible-python-environments.md).

## Objective

Stand up a real Python ML project the way it will exist for the
rest of the curriculum: `pyproject.toml`, a lockfile, a pinned
interpreter, an `src/` package layout, a working test command, and
a CI job that refuses to install a drifted environment. Prove the
environment is reproducible by rebuilding it from scratch — twice.

By the end you should be able to bootstrap a new ML project in
under fifteen minutes and never again field a "works on my
machine" bug report.

## Prerequisites

- A working Python 3.11 or 3.12 installation on your machine (or
  the ability to install one via `uv python install`).
- Either **[`uv`](https://docs.astral.sh/uv/)** (recommended) or
  **[`poetry`](https://python-poetry.org/)**. Pick one and use it
  for the whole exercise — do not mix.
- `git` and a free GitHub / GitLab account (or a local remote, if
  you prefer to stay offline).
- Chapter 07 of this module.

## Problem statement

You will build a small project called `ml_toolkit` that exposes one
trivial function (`standardise(X)` from chapter 02 works fine) and
proves-out the reproducibility machinery around it. The *code* is
almost incidental — the point is the surrounding envelope.

### Part A — bootstrap the project (≈ 30 min)

Create a fresh directory `ml_toolkit/` and:

1. **Initialise the project** with your chosen tool
   (`uv init ml_toolkit` or `poetry new ml_toolkit`).
2. **Pin the Python interpreter** to a specific minor version — not
   a range — via `.python-version` (uv) or `pyproject.toml`
   (poetry). Choose 3.11 or 3.12; document which and why in
   `REPORT.md`.
3. **Add the following direct dependencies**:
   - runtime: `numpy>=1.26`, `pandas>=2.1`, `scikit-learn>=1.4`
   - dev: `pytest>=8`, `ruff>=0.4`, `mypy>=1.9` (or `pyright`)

   Use the tool's `add` command; do not hand-edit `pyproject.toml`.
4. **Adopt the `src/` layout**: put your package at
   `src/ml_toolkit/__init__.py`, not at the repo root. `uv init
   --lib` scaffolds this; with poetry you may need to move the
   directory manually.

Confirm you have all of the following files, and only these files,
tracked in git:

- `pyproject.toml`
- `uv.lock` (or `poetry.lock`)
- `.python-version`
- `.gitignore` — must exclude `.venv/`, `__pycache__/`,
  `.mypy_cache/`, `.ruff_cache/`, `dist/`, `*.egg-info/`
- `README.md`
- `src/ml_toolkit/__init__.py`
- `src/ml_toolkit/features.py` — home to `standardise(X)`
- `tests/test_features.py`

### Part B — write and run one real test (≈ 20 min)

In `src/ml_toolkit/features.py`, implement:

```python
import numpy as np
import numpy.typing as npt

def standardise(X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Zero-mean, unit-variance each column of X."""
    return (X - X.mean(axis=0)) / X.std(axis=0)
```

In `tests/test_features.py`, write at least two tests: one asserting
per-column mean ≈ 0 after transformation, one asserting per-column
std ≈ 1. Use `np.random.default_rng(0)` and `np.allclose`.

Confirm all of these work from a clean shell:

```bash
uv run pytest -q                # (or `poetry run pytest -q`)
uv run ruff check .
uv run mypy src tests
```

### Part C — prove reproducibility twice (≈ 30 min)

Now the meaningful part: rebuild the environment from scratch and
confirm you get bit-for-bit the same package versions.

1. **Delete and rebuild locally.**
   ```bash
   rm -rf .venv
   uv sync --frozen            # or `poetry install --no-update`
   uv run pytest -q
   ```
   All tests must pass. `--frozen` / `--no-update` refuses to
   modify the lockfile — if the lockfile is out of date relative
   to `pyproject.toml`, this fails on purpose.

2. **Rebuild on a second machine (or in a Docker container).**
   Write a `Dockerfile` (~20 lines is plenty) that:
   - Starts from a slim `python:3.11-slim` (or 3.12) base image.
   - Installs `uv` (or `poetry`).
   - Copies `pyproject.toml`, the lockfile, and `.python-version`.
   - Runs `uv sync --frozen`.
   - Copies the rest of the source.
   - Runs `pytest`.

   Build and run it:
   ```bash
   docker build -t ml_toolkit:test .
   docker run --rm ml_toolkit:test uv run pytest -q
   ```

3. **Diff the installed versions.** Run `uv pip freeze` (or
   `poetry export --without-hashes -f requirements.txt`) both
   locally and in the container; the two lists must be identical.
   Include the (short) diff or a "no differences" note in
   `REPORT.md`.

If you do not have Docker available, use a fresh venv on a
different Python installation via `pyenv` / a second machine / a
GitHub Codespace and demonstrate the same equivalence.

### Part D — CI enforcement (≈ 30 min)

Write a `.github/workflows/ci.yml` (or the equivalent for your
platform) that runs on every push and pull request. It must:

1. Install `uv` (`astral-sh/setup-uv@v3`) or `poetry`.
2. Run `uv sync --frozen` (or `poetry install --no-update`). This
   is the enforcement — the build fails if the lockfile is out of
   date.
3. Run `uv run ruff check .`
4. Run `uv run mypy src tests`
5. Run `uv run pytest -q`

Push the branch and confirm CI passes.

Then **deliberately break it** to see the enforcement work:

- Edit `pyproject.toml` by hand to bump one dependency version
  without regenerating the lockfile. Push.
- Confirm CI fails at the `sync --frozen` step, with a message that
  points at the drifted lockfile.
- Revert.

Include a link (or a screenshot) of the failing run in `REPORT.md`.

## Starter guidance

- `uv init --lib` gives you the `src/` layout for free; without
  `--lib` you get a script layout.
- `.python-version` is a one-line file — just the version string
  like `3.11`. Do not put quotes around it.
- If you get "no lockfile found" errors, you probably ran `pip
  install` at some point. Delete `.venv`, delete any stray
  `requirements.txt`, and run `uv sync` (which regenerates the
  lockfile) or `uv lock` (which regenerates only the lockfile).
- On CI, the `setup-uv` action handles caching for you. On
  poetry, add a manual `actions/cache@v4` step for `~/.cache/pypoetry`
  and `.venv/` to avoid rebuilding the world on every push.
- `uv sync --frozen` is the CI-safe form; `uv sync` on its own
  will silently update the lockfile, which is not what you want in
  CI.

## Acceptance criteria

- `pyproject.toml`, the lockfile, `.python-version`, and
  `.gitignore` are all committed. `.venv/` and cache directories
  are *not* committed.
- `uv sync --frozen && uv run pytest -q` (or the poetry equivalent)
  is green on a fresh clone.
- The Docker build (or second-machine rebuild) produces an
  installed-package listing identical to the local one.
- CI passes on the happy path, and *fails* when the lockfile drifts
  (Part D deliberate-break step). The failure message is captured
  in `REPORT.md`.
- No `pip install <something>` calls appear in your history or
  scripts — every dependency change went through `uv add` or
  `poetry add`.

## Stretch goals (optional)

- Add [`pre-commit`](https://pre-commit.com/) with hooks for `ruff
  check`, `ruff format`, `mypy`, and `nbstripout` (if you have any
  notebooks). Test locally that `git commit` runs them.
- Publish the package to TestPyPI (`uv publish --index testpypi
  ...` or `poetry publish -r testpypi`) and install it in a fresh
  venv on your machine. Confirm you can `import ml_toolkit`
  without cloning the repo.
- Add a matrix build in CI that runs against both Python 3.11
  and 3.12. Note which (if any) transitive dependency changes
  version between the two.
- Wire in a **lockfile-drift bot** — a scheduled CI job that runs
  `uv lock --upgrade --dry-run` weekly and opens an issue if
  upstream has security fixes available.

## What to hand in

```
ml_toolkit/
├── pyproject.toml
├── uv.lock                    # (or poetry.lock)
├── .python-version
├── .gitignore
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── ml_toolkit/
│       ├── __init__.py
│       └── features.py
├── tests/
│   └── test_features.py
├── README.md
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo — do not
peek until you have submitted yours.
