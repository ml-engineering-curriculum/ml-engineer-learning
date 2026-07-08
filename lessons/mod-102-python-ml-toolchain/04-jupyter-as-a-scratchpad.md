# 04 — Jupyter: useful scratchpad, dangerous deliverable

> **Reading time:** ~20 minutes.
> **Prerequisite chapters:** 01.

## Motivation

Jupyter notebooks are the dominant interactive Python environment in
data and ML work. JupyterLab, classic Notebook, VS Code's notebook
UI, and Google Colab all run the same `.ipynb` format and the same
IPython kernel underneath. You will spend a large fraction of your
exploratory time in one.

Notebooks are wonderful for **exploration**. They are catastrophic as
**deliverables**. A working ML engineer learns to use them for what
they are good at and to graduate code out of them once it becomes
production-shaped. This chapter is about the discipline that keeps
notebooks useful.

## Why notebooks are good

The notebook's value comes from the **read-evaluate-print loop on
state**:

- You can inspect intermediate variables without restructuring your
  code.
- You can display rich output inline: tables, plots, model summaries,
  HTML widgets.
- You can interleave prose and code, which is the right format for an
  EDA report (exploratory data analysis) where the conclusion is as
  important as the calculation.

This is exactly the workflow you want when you are **investigating**
something — a new dataset, an unfamiliar model's behaviour, a failing
training run.

## Why notebooks are dangerous

Three failure modes recur in every team that lives in notebooks:

1. **Hidden state.** Cell execution is order-independent — you can
   run cell 5, then 3, then 7 — but the kernel's variable state
   tracks *what you ran*, not *what's on screen*. Notebooks that
   "work" when you walk away can be irreproducible the moment the
   kernel restarts.
2. **Untested logic.** The cell-by-cell layout is unfriendly to unit
   tests. Logic that lives only in notebooks rarely gets tested,
   rarely gets reviewed, and rarely gets reused — it accretes copy-
   paste duplicates in five other notebooks.
3. **Poor diffability.** `.ipynb` is JSON with embedded output. Plain
   `git diff` of a notebook is unreadable; reviewers cannot tell what
   changed in the *code* versus what changed because someone rerun
   a cell with a different random seed.

The fix is not to abandon notebooks — it is to use them for the
phase of work they fit (exploration), and to graduate code into
plain Python modules the moment it becomes a building block someone
else might call.

## The "promote out" discipline

A practical workflow that survives team scale:

1. **Explore in a notebook.** Inspect, plot, eyeball, prototype the
   transformation or model. Optimise for thinking speed, not code
   quality.
2. **The moment you copy-paste a cell**, that logic earns the right
   to live in a `.py` module. Move it; the notebook now imports it.
3. **The moment a colleague asks "can I reuse what you did?"**, the
   logic earns a unit test (see chapter 08) and a real function
   signature.
4. **Notebooks that survive in the repo** are reports or runbooks,
   not libraries. Their cells are short, top-to-bottom, and call
   tested functions from `.py` modules.

The mental model: a notebook is a *consumer* of your library code,
not the *source* of it. It is the Python equivalent of a Jupyter
notebook calling into the boring, tested `src/` directory.

```text
src/
  features.py        # tested transformers
  model.py           # tested training/eval functions
notebooks/
  01_eda.ipynb       # imports from src/, displays plots and tables
  02_baseline.ipynb  # imports from src/, runs the baseline experiment
```

To make this workflow ergonomic, put the package on Python's path
once at the top of every notebook:

```python
%load_ext autoreload
%autoreload 2

from src import features, model
```

`autoreload` re-imports edited `.py` files automatically, so edits to
`src/features.py` show up immediately in the notebook without a
kernel restart.

## Repeatable, top-to-bottom notebooks

A notebook should run cleanly under **Kernel → Restart and Run All**.
If it doesn't, treat that as a bug, not a quirk:

- Fix it now, while you remember which cell does what.
- Add the failing case (a missing import, a referenced-but-deleted
  variable, an out-of-order cell) to your "smell list" so you catch
  it next time.
- For long-running cells, gate them behind an explicit
  `if RUN_HEAVY:` flag; do not just "skip cell 17 when running top-
  to-bottom".

Two complementary tools make this enforceable:

- **`nbconvert --execute`** runs a notebook headlessly and fails if
  any cell errors. Wire it into CI for notebooks that ship as
  reports.
- **[`papermill`](https://papermill.readthedocs.io/)** parameterises
  a notebook (you can pass in a different date range, dataset, model
  hyperparameter) and executes it deterministically. Useful for
  recurring reports and parameterised reruns.

## Notebook hygiene for source control

The default `.ipynb` JSON includes the entire output stream — base64-
encoded PNGs and all. That makes `git diff` painful and pollutes
your repo with binary churn.

Two complementary fixes:

- **Strip output before committing.** The
  [`nbstripout`](https://github.com/kynan/nbstripout) tool installs a
  git pre-commit filter that strips `outputs` and `execution_count`
  on commit. Configure once; forget about it.
- **Pair every notebook with a stable `.py` mirror** using
  [`jupytext`](https://jupytext.readthedocs.io/). Jupytext maintains
  a paired `.py` "percent format" file that *is* diffable in a normal
  code review. The notebook stays in your editor; reviewers read the
  `.py`.

A team that adopts either of these is one that can review notebook
changes in a PR. A team that adopts neither is one that says "I'll
review the next commit" and never does.

## What about Colab?

Google Colab is a hosted notebook environment with free GPU access.
It is excellent for `mod-105` and Project 102 (vision), where the
GPU matters and you do not want to set up CUDA locally.

The discipline is the same: Colab is a scratchpad. Anything that
matters lives in a real repo, not a one-off notebook URL. Mount your
Google Drive or clone your repo at the start of every Colab session;
do not let your code live "in the cloud" untracked.

## The `IPython` magics worth knowing

Magics are the `%` and `%%` commands you can run inside Jupyter cells.
A short list that earns its keep:

- `%load_ext autoreload` / `%autoreload 2` — re-import edited modules
  on the fly.
- `%timeit expr` / `%%timeit` — micro-benchmark a single expression
  or whole cell.
- `%%capture` — silence chatty output (warnings, prints) for one
  cell.
- `%env VAR=value` — set environment variables for the kernel.
- `%matplotlib inline` (Jupyter) / `%matplotlib widget` (interactive
  plots) — pick which one your environment supports and stick with
  it.
- `%debug` — drop into a post-mortem `pdb` after an exception in the
  previous cell. Worth its weight in gold when you can't reproduce a
  bug.

## Summary

- Notebooks excel at exploration and reporting; they fail as
  source-of-truth libraries.
- Hidden cell state and the JSON format are the two biggest
  liabilities; both are manageable with discipline and tooling.
- "Promote out" of notebooks early: copy-paste means the logic
  belongs in a `.py` module.
- Notebooks must run cleanly under Restart and Run All. Enforce it
  with `nbconvert --execute` if you can.
- Strip outputs (or use `jupytext`) before committing; otherwise PR
  diffs become unreadable.

## Where this goes next

Chapter 05 leaves the notebook and starts on the scikit-learn API,
which is where most of the "logic that should be promoted out of a
notebook" actually lives.
