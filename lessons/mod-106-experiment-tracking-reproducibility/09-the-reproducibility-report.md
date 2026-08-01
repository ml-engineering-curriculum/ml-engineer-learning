# 09 — The reproducibility report

> **Reading time:** ~20 minutes.
> **Prerequisite chapters:** all of 01–08.

## Motivation

Everything in this module — MLflow, DVC, seeds, lockfiles,
sweeps — exists so that a specific document can be short and
credible: **the reproducibility report**.

Its job is one job: *a teammate reads it, follows the
commands, and gets the same number back.* Not "roughly the
same" — the same, to the tolerance you stated. If the report
lets them do that, the module has delivered its outcome. If
it doesn't, the tracker, the sweep, and the model registry
are of academic interest only.

This chapter is a template, a checklist, and a short guide to
writing prose that ages well. Every exercise in this module
ends in a `REPORT.md` following this shape.

## The template

Copy this into `REPORT.md` at the top of every project;
fill in as you go.

```markdown
# Reproducibility Report — {{project}}

**Author.** {{name}} ({{contact}})
**Date.** {{yyyy-mm-dd}}
**Best result.** `{{metric_name}} = {{value}}` (±{{std}}, seeds {{seeds}})
**MLflow run(s).** {{run_id_or_url}}
**Registered model.** `models:/{{name}}/{{version}}` (alias `{{alias}}`)

## 1. Task

One paragraph. What is the model predicting, on what data,
for what purpose. Why this metric and not another.

## 2. How to reproduce

Copy-pasteable commands, in order. A stranger with repo
access should be able to run this block and get the number
in "Best result".

```bash
git clone {{repo}}
cd {{repo}}
git checkout {{git_sha}}

# Environment
uv sync --frozen
# (or) docker pull {{image_ref_with_digest}}
#      docker run --rm {{image_ref_with_digest}}

# Data
dvc pull

# Run
python -m src.train --config configs/{{config}}.yaml --seed {{seed}}

# Expected: val_accuracy ≈ {{value}} (±{{tolerance}})
```

## 3. Data

- **Dataset name and source.** {{name}} — {{URL_or_internal_pointer}}
- **DVC pointer.** `{{path_to_.dvc}}` @ `md5: {{data_md5}}`
- **Split.** {{train_size}} / {{val_size}} / {{test_size}}
- **Preprocessing.** Where the code lives ({{path_or_stage}}).
- **Known caveats.** {{any relevant filtering, label noise, drift risks}}

## 4. Model

- **Architecture / algorithm.** {{name}} ({{parameter_count_if_neural}})
- **Framework and version.** {{e.g. pytorch==2.4.0, sklearn==1.5.0}}
- **Signature.** `{{path_to_MLmodel_or_registry_URI}}`
- **Selected hyperparameters.**
  ```yaml
  {{full hyperparameter block, straight out of params.yaml}}
  ```

## 5. Training procedure

- **Sweep.** Optuna study `{{study_name}}` — {{n_trials}} trials,
  sampler `{{sampler}}(seed={{seed}})`, pruner
  `{{pruner}}({{pruner_params}})`. Search space:

  ```
  learning_rate:    log-uniform [1e-3, 1.0]
  max_depth:        int [3, 12]
  ...
  ```

- **Winner.** Trial {{n}} — {{best_params}}.
- **Seed sweep.** Retrained the winner under seeds {{seeds}};
  result was `{{metric}} = {{mean}} ± {{std}}`.
- **Hardware.** {{cpu_or_gpu}}; {{wallclock}} per full train.

## 6. Environment

- **Python.** {{version}}
- **OS / platform.** {{platform}}
- **Container.** `{{image_ref}}@sha256:{{digest}}`
- **Lockfile.** `uv.lock` at `{{git_sha}}`
- **GPU (if any).** {{model}}; CUDA {{version}}; cuDNN {{version}};
  driver {{version}}
- **`pip freeze`.** In run artefact `env/pip_freeze.txt`
  ({{mlflow_run_id}})

## 7. Determinism claim

- **Tier.** {{`repeatable-CPU` | `reproducible-GPU`}}
- **Seeded RNGs.** Python, NumPy (legacy + new API),
  PyTorch CPU, PyTorch CUDA (`manual_seed_all`),
  DataLoader (`worker_init_fn` + `generator`).
- **Deterministic flags.** `torch.use_deterministic_algorithms(True)`,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`.
- **Tolerance.** {{e.g. "val_accuracy within ±0.001 across
  A100/H100; byte-identical on CPU"}}
- **Verified by.** {{test_name}} in `tests/`.

## 8. Known variance sources not eliminated

Any honest gap in the reproducibility story goes here. E.g.:

- Different GPU model shifts `val_accuracy` by up to
  {{value}}.
- Data augmentation library `X` uses its own global RNG;
  we do not seed it and accept ±{{value}} variance from
  that.
- HuggingFace `tokenizers` parallelism is enabled for speed;
  disable with `TOKENIZERS_PARALLELISM=false` for
  byte-identical CPU runs.

## 9. Where to look if reproduction fails

Diagnostic order:

1. **Metric wildly different?** Check `dvc pull` succeeded
   and `data.md5` matches. Data drift is the most common
   cause.
2. **Metric slightly different?** Check GPU model matches;
   fall back to CPU to confirm byte-identical training.
3. **Import fails?** Check `uv sync --frozen` succeeded;
   confirm the container digest matches.
4. **Nothing runs at all?** Check `git checkout {{git_sha}}`
   was clean (no uncommitted changes).
5. **Contact.** {{author}} for questions the above doesn't
   answer.
```

Fields in `{{double_braces}}` are the mandatory fill-ins.
Everything else is prose you write.

## Why every section is there

Each section prevents a specific failure mode of past
reports:

- **§ 1 (Task).** Prevents "I ran your model, what does the
  number mean?" A stranger needs one paragraph to know what
  they are looking at.
- **§ 2 (How to reproduce).** The lead paragraph most readers
  skip to. Anything else in the report is context for this
  block.
- **§ 3 (Data).** Prevents "which CSV?"
- **§ 4 (Model).** Prevents "which model file? Which
  version?" — with a registry URI, this is one line.
- **§ 5 (Training).** Prevents "how many trials? Which
  ones?" and separates *the sweep* from *the final training
  run*.
- **§ 6 (Environment).** Prevents "it doesn't work on my
  machine". The container digest and the lockfile SHA are
  the load-bearing lines.
- **§ 7 (Determinism claim).** Names the tolerance
  explicitly. The reader knows which bar you cleared
  (chapter 04's tier-1 vs tier-2).
- **§ 8 (Known variance).** Honesty section. Every project
  has gaps; saying so up front prevents "we discovered a
  reproducibility bug" reports later.
- **§ 9 (Where to look).** Debugging playbook. Saves an
  hour every time someone gets a slightly wrong number.

If a section is missing, the failure mode returns.

## Writing prose that survives

Reports rot when the author moves on and the reader
inherits them. Three habits keep the report alive for
longer:

- **Absolute values, not relatives.** "2026-08-01" beats
  "last week". "commit `abc123`" beats "the current head".
  A relative reference is a landmine to future you.
- **URIs over names.** `models:/adult-income-classifier@champion`
  is a URI a machine can dereference. "the champion model"
  is prose that could mean anything a month from now.
- **Show the command, not the intent.** "Run the training
  script" is a wish. `python -m src.train --config
  configs/hgb.yaml --seed 0` is a fact. If a reader has to
  guess how to translate intent into a command, half of them
  will guess wrong.

The report is a snapshot. Snapshots decay. The mitigations
above slow the decay so the snapshot is still useful long
enough to matter.

## Automating the pre-fill

Most of the template can be generated from the tracker.
Here is a script that grabs the fields for a specific
MLflow run and prints the pre-filled skeleton:

```python
# scripts/report_prefill.py
import sys
from datetime import date
import mlflow

def prefill(run_id: str) -> str:
    run = mlflow.get_run(run_id)
    d = run.data
    p, m, t = d.params, d.metrics, d.tags
    return f"""# Reproducibility Report — {t.get("mlflow.runName", run.info.run_name)}

**Author.** {t.get("mlflow.user", "unknown")}
**Date.** {date.today().isoformat()}
**Best result.** `val_accuracy = {m.get("val_accuracy", "?"):.4f}`
**MLflow run.** {run.info.run_id}
**Registered model.** {p.get("registered_model_uri", "TODO")}

## 2. How to reproduce
```bash
git checkout {t.get("mlflow.source.git.commit", "TODO")}
uv sync --frozen
dvc pull
python -m src.train --seed {p.get("seed", "0")}
```

## 3. Data
- **DVC pointer.** md5: `{p.get("data.md5", "TODO")}`

## 6. Environment
- **Python:** {t.get("mlflow.python.version", "?")}
- **Container:** {p.get("docker_image_id", "TODO")}
"""

if __name__ == "__main__":
    print(prefill(sys.argv[1]))
```

Run:

```bash
python scripts/report_prefill.py <run_id> > REPORT.md
# then edit the TODOs and the prose sections
```

The point is not that the tool writes the report — it is
that the *facts* are auto-filled from the tracker so the
author's attention goes to the sections that require
judgment (§ 1, § 7 tolerance, § 8 known gaps, § 9
troubleshooting).

## The reproducibility checklist

Before you ship the report, walk this list:

- [ ] `git status` is clean.
- [ ] All params, metrics, and artefacts logged to MLflow.
- [ ] `mlflow.log_dict(env_manifest, "env/manifest.json")`
      present.
- [ ] `mlflow.log_text(pip_freeze, "env/pip_freeze.txt")`
      present.
- [ ] GPU details logged if a GPU was used.
- [ ] `docker_image_id` param recorded if containerised.
- [ ] Data referenced via DVC hash; hash logged as
      `data.md5` param.
- [ ] Model logged via `mlflow.<flavour>.log_model` with a
      signature.
- [ ] Best model registered (`registered_model_name=` on
      log_model, or `create_model_version` in CI).
- [ ] Registered version tagged with `git_sha`, `data.md5`,
      and validation metric.
- [ ] Alias set (`@champion` or your convention).
- [ ] Optuna study persisted (SQLite or Postgres URL).
- [ ] Sweep plots (`plot_optimization_history` etc.) logged
      as artefacts.
- [ ] Seed sweep (3–5 seeds) run on the winner; mean ± std
      in report.
- [ ] Determinism test in `tests/` passing.
- [ ] `REPORT.md` filled in and committed.

Every item on this list corresponds to something in an
earlier chapter. If any is missing, go back to the
corresponding chapter and add the instrumentation.

## The rerun test — the report's own smoke test

The best proof the report works is that *someone else* runs
it. Two lightweight versions of this in practice:

- **CI reproduction job.** A weekly (or on-demand) CI job
  that follows § 2's commands verbatim on a fresh runner
  and asserts the metric matches within the stated
  tolerance. Fails loudly if the container digest bit-rots,
  the DVC remote loses data, or the metric drifts.
- **New-teammate test.** The next person who joins runs the
  report as their first task. Any question they have to ask
  is a missing sentence in the report. Update it.

Both catch different failure modes: CI catches infrastructure
rot; the teammate catches prose rot.

## Common pitfalls

1. **Prose without commands.** "Set up the environment and
   train the model" is not a reproduction step. Show the
   commands.
2. **Best-result number without a tolerance.** "0.87
   accuracy" invites bug reports for 0.869. State the
   tolerance.
3. **Relative dates and vague references.** "last week", "the
   current model", "our latest data". Every relative
   reference decays.
4. **Missing determinism claim.** If tier isn't stated, the
   reader guesses — usually wrong. Say tier-1-CPU or
   tier-2-GPU explicitly.
5. **No plan for when reproduction fails.** § 9 exists so
   readers do not create tickets for problems the report
   can address in a paragraph.
6. **Committing REPORT.md before the run finishes.** A report
   for a broken run is worse than no report. Write it after
   the run's artefacts land.
7. **One giant report, never split.** As a project grows,
   split by model or by milestone. A report per registered
   model version is a good rule.

## Summary

- The reproducibility report is the deliverable the module
  exists to produce. Everything before this chapter is
  instrumentation; this chapter is the interface between
  the run and the reader.
- Follow the template. Every section maps to a specific past
  failure mode.
- Auto-fill the facts from the tracker; write prose only
  where judgment is needed.
- Ship with a checklist walk-through, a tolerance, and a
  troubleshooting plan. Verify with a rerun.

## Where this goes next

You are done with `mod-106`. The next step in the ML Engineer
track is `mod-107` (model evaluation and validation) — how to
tell whether the model you just made reproducible is actually
correct. `mod-108` (deployment) picks up your registered
model version and puts it behind a serving stack. `mod-109`
(monitoring & drift) tells you when the input distribution
has shifted enough that the report's guarantees no longer
hold — which is the moment the whole cycle in this module
starts over.
