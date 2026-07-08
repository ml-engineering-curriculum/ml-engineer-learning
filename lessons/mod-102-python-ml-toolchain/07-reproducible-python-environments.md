# 07 — Reproducible Python environments

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 01.

## Motivation

"It worked on Monday" is the canonical bug report of unstable ML
projects. The cause is almost never the model code. The cause is
that the *environment* changed: a transitive dependency was
upgraded, the Python minor version drifted, or a wheel was rebuilt
against a different CUDA toolkit.

An ML engineer who cannot pin and reproduce a Python environment
cannot ship reliable models. This chapter teaches the two modern
tools the field has settled on (`uv` and `poetry`), the concepts
that make them work (lockfiles, virtual environments, version
specifiers), and the workflow rules that prevent the "works on my
machine" trap.

## What you are actually pinning

When you say "Python environment", three things need to be
controlled:

1. **The Python interpreter version.** `3.11.7` vs. `3.12.1` is not
   a rounding error — the bytecode and several stdlib APIs are
   different.
2. **The direct dependencies** — the packages your code imports.
   `pandas`, `scikit-learn`, etc.
3. **The transitive dependencies** — the packages your direct
   dependencies import. These are responsible for the most surprising
   "broke overnight" bugs.

A `requirements.txt` with `pandas` (no version) pins *none* of these.
A `requirements.txt` with `pandas==2.2.3` pins one. A proper
**lockfile** pins all three, including the exact version of every
transitive dependency, often along with a content hash so you can
detect tampering.

The two modern tools that produce real lockfiles for Python are
**`uv`** and **`poetry`**. Either is fine. The classical
`pip install -r requirements.txt` workflow is **not** sufficient —
even with `pip freeze`, you do not get the deterministic
resolution behaviour a lockfile gives you.

## Virtual environments, briefly

A **virtual environment** is a self-contained directory with its own
`python` binary and `site-packages`. It isolates one project's
dependencies from every other project on the machine.

You should never `pip install` into the system Python. Every ML
project gets its own venv. The tools below create and manage that
venv for you; the underlying mechanism is just the standard library
`venv` module.

## Path A — `uv` (recommended for new projects)

[`uv`](https://docs.astral.sh/uv/) is a fast, Rust-implemented
Python project and environment manager from Astral (the makers of
`ruff`). It replaces `pip`, `pip-tools`, `virtualenv`, `pyenv`, and
parts of `poetry` with one tool. As of 2026 it is the fastest
mainstream choice and is becoming the default in new projects.

### Bootstrapping a project

```bash
# Initialise a new project (creates pyproject.toml, .python-version)
uv init my-ml-project
cd my-ml-project

# Pin Python to a specific minor version
uv python pin 3.11

# Add direct dependencies
uv add numpy pandas scikit-learn matplotlib

# Add development-only dependencies (tests, linters, type-checker)
uv add --dev pytest ruff mypy

# Run a command inside the project environment
uv run python -c "import sklearn; print(sklearn.__version__)"
uv run pytest
```

After this you have:

- **`pyproject.toml`** — the canonical project manifest, including
  declared dependencies and version constraints.
- **`uv.lock`** — the lockfile: every transitive dependency pinned
  to an exact version with a content hash.
- **`.venv/`** — the virtual environment uv manages for you.

### Reproducing the environment elsewhere

On a teammate's machine, CI, or a fresh container:

```bash
uv sync   # installs exactly what's in uv.lock
```

That single command guarantees the same set of versions everyone
else has. No more "but it works in my env."

### What goes in version control

Commit:

- `pyproject.toml`
- `uv.lock`
- `.python-version` (so `uv python install` picks the right
  interpreter on a fresh machine)

Ignore:

- `.venv/`
- `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`

A `.gitignore` for these can be copied from `uv init`'s scaffold or
from any modern Python project template.

## Path B — `poetry`

[`poetry`](https://python-poetry.org/) is the older incumbent in
this space and remains in wide use. The semantics are similar to
`uv`'s; the syntax differs.

```bash
poetry init
poetry add numpy pandas scikit-learn
poetry add --group dev pytest ruff mypy
poetry install        # like `uv sync`
poetry run pytest
```

Poetry produces:

- `pyproject.toml` (its own dialect, similar to `uv`'s)
- `poetry.lock`

Pick **one** of `uv` or `poetry` per project. Mixing creates two
sources of truth for the dependency graph and they will drift.

## Version specifiers, briefly

When you say `pandas>=2.2`, you are writing a
[PEP 440 version specifier](https://peps.python.org/pep-0440/). The
common forms:

| Specifier | Meaning |
|---|---|
| `pandas` | any version. Almost never what you want. |
| `pandas==2.2.3` | exactly this version. |
| `pandas>=2.2,<3` | the 2.x series, no major bumps. |
| `pandas~=2.2.0` | "compatible release" — equivalent to `>=2.2.0,<2.3`. |
| `pandas^2.2.0` | Poetry-specific: `>=2.2.0,<3`. |

The recommendation: keep your **direct** dependencies on
relatively-loose ranges (`>=X.Y,<X+1`) in `pyproject.toml`, and let
the **lockfile** pin everything exactly. That way `uv lock`/`poetry
lock` can pick up patch-level security fixes when you bump
intentionally, without weekly drift from upstream pre-releases.

## Pinning Python itself

Pinning the interpreter version is as important as pinning the
packages. Two complementary mechanisms:

- **`.python-version`** — read by `uv`, `pyenv`, and several IDEs. A
  one-line file that names the interpreter version.
- **`requires-python = ">=3.11,<3.13"`** in `pyproject.toml` — the
  range your code declares support for. Used by package resolvers
  to refuse to install your project on incompatible interpreters.

For ML projects, prefer pinning a single minor version
(`.python-version: 3.11`) rather than a range. Subtle bugs in C
extensions across minor versions are a common cause of
"reproduces locally but crashes in CI on a different Python."

## A typical ML project layout

```text
my-ml-project/
├── pyproject.toml          # declared dependencies, Python range, tool config
├── uv.lock                 # pinned environment (or poetry.lock)
├── .python-version         # 3.11
├── .gitignore              # ignores .venv/, caches
├── src/
│   └── my_ml_project/
│       ├── __init__.py
│       ├── features.py
│       └── model.py
├── tests/
│   ├── test_features.py
│   └── test_model.py
├── notebooks/              # paired .py with jupytext (see chapter 04)
└── README.md
```

The `src/<package>/` layout is the modern Python convention: it
prevents accidental imports of in-progress code that has not been
installed into the venv. Tools like `uv`, `poetry`, and `pip
install -e .` understand it natively.

## Pinning *what scikit-learn etc.* depend on

ML libraries are tightly coupled to their numerical backends.
`scikit-learn` is built against a specific range of `numpy` and
`scipy`. A `torch` wheel is built against a specific CUDA version.
The lockfile takes care of the Python side; CUDA is its own concern.

Two practical rules:

- For CPU-only environments (most of `mod-101`–`mod-104` and `mod-106`–
  `mod-110`), trust the lockfile. The default `pip`/`uv` package
  index serves CPU-only wheels.
- For GPU environments (`mod-105`, Project 102), follow the official
  PyTorch installation matrix at
  <https://pytorch.org/get-started/locally/> — you will install
  PyTorch from a specific CUDA-tagged index URL, not the default
  PyPI. Pin that URL in your project (`uv` supports per-index
  resolution; Poetry supports `source` blocks).

## CI: enforce reproducibility

A reproducibility policy that is not enforced by CI quietly rots.
Two CI checks worth adding to every ML repo:

1. **Lockfile-deterministic install.** `uv sync --frozen` (or
   `poetry install --no-update`) refuses to update the lockfile and
   installs exactly what's pinned. If the lockfile is out of date
   relative to `pyproject.toml`, the build fails — forcing the
   author to run `uv lock` (or `poetry lock`) intentionally.
2. **Lockfile drift on PRs.** A separate CI job that runs `uv lock
   --upgrade --dry-run` and posts a comment if upstream has security
   fixes available. Optional but pays for itself.

CI is also where you discover that your "works locally" environment
relies on a side-loaded package or a hand-edited file. Run your
tests in a fresh `uv sync`'d container in CI from day one.

## Pitfalls collected

1. **`pip install` outside a venv.** You will eventually break your
   system Python this way. Use a venv. Always.
2. **`pip freeze > requirements.txt`** as a substitute for a real
   lockfile. It captures *what is installed* but not the *resolution
   graph*. It also leaks platform-specific wheels.
3. **Editing `pyproject.toml` by hand without regenerating the
   lockfile.** Now the two disagree. Always go through
   `uv add` / `poetry add`.
4. **Forgetting to commit the lockfile.** No lockfile in the repo →
   no reproducibility. CI is a partial defence; commit it.
5. **Trusting "latest"**. A library upgrading from `2.x` to `3.0`
   between your training run and your serving deployment can change
   numerical behaviour silently. Lock everything. Bump intentionally.

## Summary

- A reproducible environment pins the Python interpreter, the
  direct dependencies, and every transitive dependency via a
  lockfile.
- `uv` and `poetry` are the two modern, lockfile-based managers.
  Pick one per project.
- Commit `pyproject.toml`, the lockfile, and `.python-version`. Do
  not commit the `.venv`.
- Enforce reproducibility in CI with `uv sync --frozen` or
  `poetry install --no-update`.
- For GPU work, follow the official PyTorch install matrix — the
  CUDA stack is its own dimension.

## Where this goes next

Chapter 08 closes the module with testing — unit tests for
transformers, golden tests for small models, lint, and type-check.
A reproducible environment + a passing test suite + a clean
`Pipeline` is the threshold at which "I trained a model" becomes "I
shipped a model".
