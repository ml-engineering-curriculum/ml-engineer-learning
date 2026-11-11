# quiz-01 — Experiment Tracking & Reproducibility

> **Format:** 12 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–09 of this module.
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter the point
> comes from — go re-read that section if you missed the question.

---

## Questions

### 1. (Chapter 01) A teammate says "the churn model we trained in July was reproducible — I re-implemented it from the design doc and got similar numbers." Which option most accurately matches how chapter 01 names the three bars of reproducibility, and which bar the teammate actually demonstrated?

A. The teammate demonstrated **repeatable** — the strictest bar — because they
   got similar numbers from a design doc.
B. Chapter 01 uses three names: **repeatable** (same code, data, env, hardware,
   seed → same result), **reproducible** (same code, data, env; different
   hardware → within a stated tolerance), and **replicable** (same *idea*,
   re-implemented from a description → qualitatively similar result). What the
   teammate described is **replicable** — a research-community bar, not the
   bar this module aims at. The module targets repeatable on CPU and
   reproducible on GPU.
C. The three bars are **repeatable / reproducible / replicable** and they are
   synonyms; the teammate cleared all three.
D. The teammate demonstrated **reproducible**, and this module's aim is
   **replicable** on both CPU and GPU.

### 2. (Chapter 02) Which single statement best captures MLflow's object model as chapter 02 teaches it, and the distinction between the four things that attach to a run?

A. MLflow stores everything under a single flat `Run` table — params, metrics,
   tags, and artifacts are all rows of the same type and can be freely
   overwritten.
B. The hierarchy is **Experiment → Run**, and a run carries four kinds of thing:
   **params** (immutable config known at start; logging the same param twice
   is an error), **metrics** (a time series — `log_metric(k, v, step=i)`
   appends `(step, value)` points), **tags** (free-form key/value labels,
   overwritable, used for filtering), and **artifacts** (any file — model,
   plot, `requirements.txt`). High-cardinality tags slow the UI; if it looks
   like a metric, log it as a metric.
C. Params and metrics are the same object; the only difference is that
   metrics have a timestamp.
D. Experiments and Runs are siblings in the hierarchy; params attach to
   experiments, metrics attach to runs, and tags attach to the server.

### 3. (Chapter 02) A training script calls `mlflow.start_run(...)` at the top, logs a few params, and then raises an exception part-way through training. The MLflow UI shows this run stuck in `RUNNING` forever. Chapter 02's advice would prevent this; which option both names the pattern and explains what it does?

A. Add a `try: ... except: mlflow.delete_run(run_id)` block — on any exception
   the run should be deleted.
B. Call `mlflow.end_run(status="FAILED")` manually at the start of every
   script so the run is already closed before training begins.
C. Use the **context manager**: `with mlflow.start_run(run_name=...) as run: ...`.
   On an exception the run is marked `FAILED` rather than left `RUNNING`,
   and nested runs (`nested=True`) compose cleanly — chapter 02 calls this
   out as the safe default and shows the manual
   `start_run` / `finally: end_run` pattern only to contrast with it.
D. `mlflow.start_run` is deprecated — chapter 02 recommends replacing it
   with `mlflow.init_run`.

### 4. (Chapter 03) A regulated-industry team cannot send training data outside its VPC, wants a UI that one engineer can host on a cheap VM, and expects to run the tracker offline on a plane during customer visits. Which tracker does chapter 03's decision guide point at, and why?

A. **Weights & Biases**, because its UI is more polished — the deployment
   question doesn't actually matter.
B. **MLflow**, because: it is open-source and self-hostable end-to-end
   (`pip install mlflow` + a SQL backend + an S3/MinIO artefact root), a
   zero-outbound-network deployment is supported, and local
   `file:./mlruns` mode runs offline out of the box. Chapter 03's rule:
   pick **MLflow** when you need self-hosting for compliance, cost, or
   offline reasons; pick **W&B** when you want the best comparison UI
   with zero ops burden and your data can leave your VPC.
C. **TensorBoard**, because it is the only tracker that works offline.
D. Run both MLflow and W&B concurrently "for insurance" — chapter 03
   recommends this as the default for compliance-sensitive teams.

### 5. (Chapter 04) An engineer adds `torch.manual_seed(0)` at the top of a PyTorch training script that uses a scikit-learn `train_test_split` and a `DataLoader(num_workers=4, shuffle=True)`. Two back-to-back runs still produce different validation accuracies. Which option most faithfully explains chapter 04's diagnosis and the recommended fix?

A. `torch.manual_seed(0)` is enough on its own; the issue must be a bug
   in `train_test_split`.
B. The problem is that `torch.manual_seed` only seeds the CPU RNG. Also
   call `torch.cuda.manual_seed_all(0)` and the runs will be identical.
C. **Three separate RNGs are in play.** `torch.manual_seed` does **not**
   seed Python `random` or NumPy — `train_test_split` goes through NumPy;
   DataLoader workers each get a fresh Python process with their own
   unseeded `random`/`numpy` state; the shuffle order in the main process
   reads from a `torch.Generator` you did not pass in. Chapter 04's
   `seed_everything` seeds Python `random`, NumPy (legacy), PyTorch CPU +
   all CUDA devices, and `PYTHONHASHSEED`; the `DataLoader` must also be
   given `worker_init_fn=seed_worker` **and** `generator=g` where `g` is
   a `torch.Generator().manual_seed(seed)`. Miss the `worker_init_fn`
   argument and the DataLoader is the silent source of
   non-reproducibility.
D. The fix is to call `torch.manual_seed(0)` *inside* the training loop at
   every epoch — seeding once at the top is never enough.

### 6. (Chapter 04) Chapter 04 draws a sharp distinction between **seeding every RNG** and **forcing deterministic algorithms**. Which option captures both the distinction and the honest trade-off the chapter asks you to accept?

A. The two are the same thing; `torch.use_deterministic_algorithms(True)`
   already seeds every RNG in the process.
B. Seeding gives you a reproducible **sequence of pseudo-random numbers**;
   framework determinism
   (`torch.use_deterministic_algorithms(True, warn_only=True)`,
   `cudnn.deterministic=True`, `cudnn.benchmark=False`, plus
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` for some cuBLAS ops) makes the same
   **floating-point answers** come out of kernels that would otherwise
   pick algorithms at run time or accumulate in non-deterministic order.
   The cost is real on GPU — cuDNN's auto-tuner is disabled and reductions
   run slower (chapter 04 says 10–40% slower on convnets is not unusual);
   on CPU the cost is negligible. Rule of thumb: turn it **on** for
   development and baselines; a final production run may turn it off once
   the model is chosen, provided the run logs that choice.
C. Chapter 04 says `deterministic=True` is always free of charge and
   should be the production default on GPU.
D. Deterministic mode is incompatible with any `torch.backends.cudnn`
   setting — you must choose one.

### 7. (Chapter 05) Chapter 05 says that a `requirements.txt` with pinned versions is **"almost a lockfile — but not quite"**, and that a container image pinned by tag is not reproducible. Which option both names the two gaps in `requirements.txt` and explains the digest rule for base images?

A. The gaps are that `requirements.txt` is written in TOML instead of YAML,
   and that container tags are case-sensitive. Pin by `:latest` to fix both.
B. **(i)** `requirements.txt` without `--hash` lines lets transitive
   dependencies drift (`pandas==2.2.0` requires `numpy>=1.23`; tomorrow's
   `numpy` can install) and leaves wheel hashes unverified — a re-uploaded
   or compromised wheel of the same version can slip through. A real
   lockfile (`uv.lock`, `poetry.lock`, or `pip-compile --generate-hashes`)
   pins every direct **and** transitive dependency to an exact version and
   to a wheel hash. CI should install with `--frozen` / `--sync` so the
   lockfile is enforced, not just decorative. **(ii)** Container base
   images referenced by **tag** (`FROM python:3.11-slim`) are not
   reproducible — tags are mutable and the same tag can return different
   bytes a month later. Pin by **digest**
   (`FROM python:3.11-slim@sha256:...`); digest references never move.
C. The only real gap is case sensitivity; either lockfiles or
   `requirements.txt` are fine as long as CI uses `pip install -r`.
D. `requirements.txt` with pinned versions is a lockfile under all
   conditions; chapter 05's recommendation is to delete `uv.lock` and
   `poetry.lock` from any repo that has them.

### 8. (Chapter 06) A colleague opens a repo, runs `git checkout abc123` (a commit from three months ago), and finds that `data/features/train.parquet` is missing even though an `.dvc` file is present. Chapter 06 explains the design that makes this expected and the one command that fixes it. Which option most accurately describes both?

A. The repo is broken — DVC should have committed the parquet file to git
   itself; file a bug.
B. By design, DVC keeps **three** pieces of data separate: a small `.dvc`
   pointer file in the repo (a few hundred bytes of YAML with the content
   `md5`, `size`, and `path` — this is what git tracks), a **local cache**
   `.dvc/cache/` that holds the actual bytes content-addressed by hash
   (not in git), and a **remote** (S3, GCS, Azure, SSH, HTTP, or a shared
   filesystem) that holds the team-shared copy. `dvc add` also updates
   `.gitignore` so the real file bytes never land in git. On a fresh
   checkout you have the pointer; `dvc pull` resolves the hash against
   the remote and materialises the bytes locally. The full past-experiment
   workflow is `git checkout <SHA> && dvc pull && uv sync --frozen`,
   which chapter 06 presents as the killer use case.
C. DVC stores every version of every file inline in git LFS; a `git lfs
   pull` is the command.
D. `dvc pull` is only for pushing new data; to retrieve historical files,
   use `dvc fetch --historical` (made-up flag).

### 9. (Chapter 07) A team wants a deployment pipeline that can promote a new model without changing a CI config and can roll back in one step. Chapter 07 contrasts MLflow Model Registry **aliases** against the legacy **stages** API. Which option best matches chapter 07's guidance?

A. Stages (`Staging` / `Production` / `Archived`) are the modern pattern;
   avoid aliases — they are unsupported.
B. Aliases and stages are identical in behaviour; pick whichever is
   alphabetically first.
C. Aliases are the modern primitive: human-readable pointers (e.g.
   `@champion`, `@staging`, `@challenger`) that name a specific version
   and move as new versions get promoted. A deploy pipeline loads
   `models:/<name>@champion` and never has to know which numeric version
   is current. Promotion is one API call
   (`client.set_registered_model_alias(name, alias, version)`); rollback
   is the same call with the previous version. Stages (`Staging` /
   `Production` / `Archived`) still work but are legacy in current MLflow;
   aliases are richer, not tied to a fixed vocabulary, and chapter 07
   recommends them for new work. Chapter 07 also stresses: always pass a
   `signature=` when registering (consumers read the schema from the
   `MLmodel` manifest) and tag every version with `git_sha`, `data.md5`,
   and the validation metric so the registry is an audit trail.
D. Aliases are a feature of W&B's registry only; MLflow's registry has
   only stages.

### 10. (Chapter 08) A teammate reports: "I ran the same Optuna sweep script twice and got different winning hyperparameters — Optuna is non-deterministic." Chapter 08 lists the exact conditions under which a sweep is reproducible. Which option both diagnoses the probable cause and lists the five conditions the chapter requires?

A. Optuna is intrinsically non-deterministic; chapter 08 says the only
   fix is to replace TPE with grid search.
B. A sweep is reproducible when **all** of these hold: **(i)** the
   sampler was constructed with a fixed seed
   (`TPESampler(seed=0)` — without it the sampler picks different trials
   each run); **(ii)** the storage is durable (`sqlite:///optuna.db` or a
   real DB, not in-memory, so the study persists across restarts and the
   sampler's history informs future trials); **(iii)** the `objective`
   itself is deterministic (chapter 04's `seed_everything` runs inside it);
   **(iv)** the data version is pinned (chapter 06, DVC); **(v)** the
   environment is pinned (chapter 05, lockfile + container digest).
   The most likely missing piece for the teammate is the **sampler
   seed**. Note the caveat: parallel workers reach the same sampler state
   but trial *completion order* varies, so exact reproduction of a
   parallel sweep requires serialising it — for most purposes the seeded
   sampler plus shared storage is enough to make results comparable
   across reruns.
C. Only in-memory storage is reproducible; chapter 08 forbids SQLite.
D. The sampler seed is irrelevant; the only thing that matters is
   `n_trials`.

### 11. (Chapter 08) Optuna's sweep returns a winning configuration. Chapter 08 insists you run one more check before declaring a winner. Which option best describes the check and when it changes the story?

A. Rerun the full sweep from a different random seed and compare. If both
   sweeps pick the same winner, you're done.
B. **The seed sweep.** Retrain the winning configuration under **3–5
   different seeds** (chapter 04's `seed_everything`) and report
   `mean ± std` of the validation metric on the winner. If the seed
   standard deviation is **larger than the gap between the best and the
   second-best Optuna trial**, the "winner" is within noise of the
   runner-up — the sweep found a seed+config combination that looked
   lucky. Chapter 09's report template has an explicit slot for the seed
   sweep's `mean ± std` for this reason. Chapter 08 also recommends
   picking the **median-seed** training (not the max) as the one to
   register — the max is lucky, the median is representative.
C. Retrain the winner with the exact same seed the sweep used. If it
   reproduces, you're done.
D. Skip the check; Optuna's sampler is Bayesian-optimal and its winner
   is by definition the best configuration.

### 12. (Chapter 09) Chapter 09 calls the reproducibility report **"the deliverable the module exists to produce"** and gives a template with nine sections. Which option best describes what makes a report in chapter 09's shape credible, and what makes it rot?

A. The report should be as long as possible; brevity kills credibility.
B. **Credible when:** the report has a copy-pasteable *reproduce block*
   (git SHA + `uv sync --frozen` + `dvc pull` + the training command — a
   stranger with repo access can run the block and land within the stated
   tolerance), a **named determinism tier** (tier 1 CPU byte-identical vs
   tier 2 GPU tolerance-bounded, from chapter 04) with an explicit
   tolerance like "`val_accuracy` within ±0.001 across A100/H100", a
   **data section** with the DVC `md5`, an **environment section** with
   container digest and lockfile SHA, a **known-variance section** that
   honestly names what is not eliminated (augmentation library RNG,
   tokenizer parallelism, different GPU model shifts), and a
   **troubleshooting section**. **Rots when:** the report uses relative
   dates ("last week", "the current model"), ships a best-result number
   without a tolerance, replaces commands with intent ("set up the
   environment and train"), or is committed before the run's artefacts
   land. The report's own smoke test is the **rerun test**: either a CI
   job that follows the reproduce block verbatim on a fresh runner, or
   the next teammate who follows it as their first task — any question
   they ask is a missing sentence.
C. Any `REPORT.md` with a title and a metric is enough.
D. The report should list only prose; concrete commands should be kept in
   a separate internal wiki page the reader must request access to.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **B.** Chapter 01 ("What 'reproducible' actually means") introduces the
   three bars by name. **Repeatable** is bit-identical under same hardware
   and seed; **reproducible** is the paper-style bar (same code/data/env,
   different hardware, within stated tolerance); **replicable** is the
   research-community bar where only the *idea* carries across (a re-
   implementation from a description). The chapter explicitly says the
   module targets repeatable on CPU and reproducible on GPU, and that
   replicable "is not our target". The teammate's "re-implemented from a
   design doc and got similar numbers" is textbook *replicable*.

2. **B.** Chapter 02 ("The object model") gives the table of the four
   attachments and their write/read APIs. The key distinctions are stated
   directly: params are immutable (re-logging the same key raises);
   metrics are a time series indexed by `step`; tags are mutable and
   intended for filtering; artifacts are files. The "high-cardinality
   tags slow the UI; if it looks like a metric, log it as a metric"
   warning is pitfall #5 in the same chapter.

3. **C.** Chapter 02 ("`mlflow.start_run` and the context manager pattern")
   says the context-manager form is the safe way because "on exception,
   the run is marked `FAILED`" — the alternative leaves it `RUNNING`
   forever. Pitfall #1 of chapter 02 ("Forgetting the context manager")
   restates it, and the chapter shows nested runs via `nested=True`.

4. **B.** Chapter 03 ("Deployment model" and "When to pick which") says
   MLflow is "open source. Self-hostable end to end", that "zero-outbound-
   network mode is a common, supported deployment", and that
   local `file:./mlruns` works offline. The "When to pick which" section
   lists compliance, cost, and offline needs as the MLflow signals. The
   "Common pitfalls" section separately warns against running two
   trackers "for insurance", ruling out D.

5. **C.** Chapter 04 ("The RNG layers you actually touch", "DataLoader
   determinism — the piece everyone misses") lists the four+
   independent RNG layers and explicitly notes "Setting
   `torch.manual_seed` does not seed NumPy or Python `random`" and
   "DataLoader workers get fresh RNGs. Without a `worker_init_fn` they
   will each produce a different order on every run, even with the same
   top-level seed." The chapter's `seed_everything` harness and the two
   required DataLoader arguments (`worker_init_fn=seed_worker`,
   `generator=g`) are the canonical fix.

6. **B.** Chapter 04 ("Framework-level determinism (the expensive part)"
   and "What seeding does *not* control"). Seeding controls the pseudo-
   random sequence; `torch.use_deterministic_algorithms` and the cuDNN
   flags force identical floating-point output from kernels that would
   otherwise vary. The chapter states "10–40% slower is not unusual on
   convnets" and gives the "turn determinism ON for development /
   baselines; may turn OFF for final production training runs" rule, and
   names the `CUBLAS_WORKSPACE_CONFIG=:4096:8` requirement.

7. **B.** Chapter 05 ("The lockfile: not the same as `requirements.txt`")
   names the two gaps (transitive drift and unverified hashes) and
   recommends `uv.lock` / `poetry.lock` / `pip-compile --generate-hashes`
   with `--frozen` / `--sync` in CI. "Containerising the environment"
   says: "Pin the base image by **digest**
   (`python:3.11-slim@sha256:...`). Tags are mutable. `docker pull
   python:3.11-slim` today and next month can return different bytes;
   digest references never do." The "Common pitfalls" section restates
   both (pitfall #1 and #3).

8. **B.** Chapter 06 ("The mental model" and "Reproducing a past
   experiment's data") describes the pointer / local cache / remote
   split, the `.dvc` file's anatomy (`md5`, `size`, `hash`, `path`), the
   automatic `.gitignore` update, and the canonical past-experiment
   workflow `git checkout <SHA>` + `dvc pull` + `uv sync --frozen`. The
   chapter calls this "the killer use case for DVC in this module".

9. **C.** Chapter 07 ("Aliases: promotion without lifecycle hell") spells
   out the alias API (`set_registered_model_alias`), shows the deploy
   pipeline loading `models:/<name>@champion` without knowing the numeric
   version, and shows the rollback case. The chapter notes "Historically
   MLflow also had **stages** (`Staging` / `Production` / `Archived`).
   Aliases are the modern replacement — richer, more flexible, and not
   tied to a fixed vocabulary." Pitfall #2 says "Using stages when
   aliases exist. Stages are legacy in current MLflow. New workflows
   should use aliases." The "The model signature" section also stresses
   "Always pass a signature when you register a model", and chapter 07's
   tag block (git_sha, data.md5, val_metric) is the audit trail.

10. **B.** Chapter 08 ("Reproducing a sweep") lists exactly five
    conditions in that order: seeded sampler, durable storage,
    deterministic `objective`, pinned data (chapter 06), pinned
    environment (chapter 05). Pitfall #2 in the same chapter says
    "Unseeded sampler. Two runs of the 'same' sweep suggest different
    trials; you cannot reproduce a result. Always `TPESampler(seed=...)`."
    The chapter also discusses the parallel-workers caveat ("the seed
    makes the sampler deterministic but the ordering of trial completion
    is not"), which rules out claiming exact reproduction under parallel
    workers.

11. **B.** Chapter 08 ("Seed sweeps — the sanity check on your winner")
    says: "When Optuna hands you a winner, retrain it under 3–5 different
    seeds and record mean + std of the validation metric. If the seed
    variance is larger than the gap between the top two Optuna
    configurations, the 'winner' is within noise of the runner-up."
    Chapter 09's report template has a seed-sweep row for exactly this
    reason, and chapter 08 (in the chapter-09-facing summary) recommends
    taking the median-seed retrain — not the max — as the version to
    register, because "the max is lucky; the median is honest."

12. **B.** Chapter 09 ("The template", "Writing prose that survives",
    "The rerun test — the report's own smoke test", "Common pitfalls").
    The template's § 2 is the reproduce block; § 7 names the tier
    (chapter 04's tier-1 CPU vs tier-2 GPU); § 8 is the "known
    variance sources not eliminated" section; § 9 is the troubleshooting
    section. The "Writing prose that survives" section names "absolute
    values, not relatives" and "show the command, not the intent" as the
    anti-rot habits. The common-pitfalls list spells out best-result-
    without-tolerance, relative dates, prose-without-commands, and
    missing determinism claims. The "rerun test" — either a CI job or
    the next teammate — is the report's own smoke test.

</details>
