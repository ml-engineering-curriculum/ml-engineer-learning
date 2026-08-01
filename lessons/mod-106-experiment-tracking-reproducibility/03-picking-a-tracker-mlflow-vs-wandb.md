# 03 — Picking a tracker: MLflow vs Weights & Biases

> **Reading time:** ~20 minutes.
> **Prerequisite chapter:** 02 (MLflow basics).

## Motivation

Every ML team eventually adopts a tracker. The two you are
most likely to see in the wild are **MLflow** (open source,
self-hosted or on Databricks) and **Weights & Biases**
(hosted SaaS, generous free tier). Aim, Neptune, Comet,
ClearML, and TensorBoard exist too. This chapter is not a
league table — it is a decision aid.

The good news: the mental model from chapter 02 — experiment
→ run → params + metrics + artefacts — carries across every
tracker. What differs is not the concepts but the deployment
model, the polish of the UI, the integration surface, and
who pays the bill.

## The two objects to compare

Read the same run in both tools; the vocabulary lines up
one-to-one:

| Concept | MLflow | Weights & Biases |
|---|---|---|
| Group of related runs | `Experiment` | `Project` |
| One training execution | `Run` (`run_id`) | `Run` (`id`) |
| Immutable settings | `params` | `config` |
| Time series numbers | `metrics` (per `step`) | `log()` (per `step`) |
| Free-form labels | `tags` | `tags` (list of strings) |
| Attached files | `artifacts` | `Artifacts` (with lineage) |
| Model with signature | `mlflow.<flavour>.log_model` | `wandb.Model` / registry |
| Sweep / HPO | Nested runs + Optuna | `wandb.sweep` (native) |
| Model registry | `Model Registry` | `Model Registry` |

Everything in chapter 02 has a W&B analogue. Switching from
one to the other is mostly rewriting one file of
instrumentation code.

## The same instrumented run, in W&B

Here is the scikit-learn example from chapter 02, rewritten
against W&B. Note how little changes:

```python
# train.py (W&B version)
import wandb
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


def main(seed=0, learning_rate=0.1, max_depth=6, max_iter=200):
    wandb.init(
        project="adult-income-baseline",
        name="hgb-baseline",
        config={
            "seed":          seed,
            "learning_rate": learning_rate,
            "max_depth":     max_depth,
            "max_iter":      max_iter,
            "model":         "HistGradientBoostingClassifier",
        },
        tags=["baseline"],
    )

    ds = fetch_openml("adult", version=2, as_frame=True)
    X, y = ds.data, (ds.target == ">50K").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)

    proba_te = clf.predict_proba(X_te)
    preds_te = proba_te.argmax(axis=1)

    wandb.log({
        "val_accuracy": float(accuracy_score(y_te, preds_te)),
        "val_log_loss": float(log_loss(y_te, proba_te)),
    })

    # Artifact with lineage
    import joblib, pathlib
    pathlib.Path("outputs").mkdir(exist_ok=True)
    joblib.dump(clf, "outputs/model.joblib")

    art = wandb.Artifact("hgb-baseline-model", type="model")
    art.add_file("outputs/model.joblib")
    wandb.log_artifact(art)

    wandb.finish()


if __name__ == "__main__":
    main()
```

Three W&B-specific patterns worth spotlighting:

- **`wandb.init(config=...)`** — the `config` dict is the W&B
  equivalent of `mlflow.log_params`, and it is also mutable
  during the run (you can add to it later).
- **`wandb.log({...})`** — one call, many metrics, at the
  same implicit step. Pass `step=` for explicit control.
- **`wandb.Artifact`** — first-class object with a name and
  a type. Later runs can `use_artifact("name:latest")`, and
  W&B builds a lineage graph automatically.

## Where the two genuinely differ

Not decorative differences; ones that decide the choice.

### Deployment model

- **MLflow.** Self-hostable end to end. The Databricks
  hosted version exists and is popular; the open-source
  server is a `pip install mlflow`, a SQL database, and an
  S3 bucket away. Zero-outbound-network mode is a common,
  supported deployment.
- **W&B.** Hosted SaaS is the primary product. An on-prem
  option (`W&B Server`) exists for enterprise; it has a
  license cost.

If your data cannot leave your VPC without a compliance
review, MLflow is the default. If you want a UI you don't
have to run, W&B is the default.

### UI polish and comparison ergonomics

- **W&B's UI is materially better** — smoother comparison
  plots, easier parallel-coordinate views, richer media
  logging (images, audio, video, tables, HTML, 3D
  point-clouds), reports as first-class shareable
  documents.
- **MLflow's UI is functional** — enough for the daily job,
  clunkier for exploratory analysis. It has improved
  substantially in recent releases.

If comparing dozens of runs on many axes is a daily
activity, the UI difference is real. If most of your
analysis happens in a notebook via `search_runs()`, it
matters less.

### Sweeps / HPO

- **W&B** has native sweeps (`wandb.sweep`) with grid,
  random, and Bayesian sampling. The sweep is a first-class
  object in the UI.
- **MLflow** does not have a native sweep; the standard
  pairing is **Optuna** (chapter 08) writing each trial as
  an MLflow run.

Both approaches work. Chapter 08 uses Optuna + MLflow
because Optuna's samplers are excellent, its API is stable,
and the pattern portable to any tracker.

### Artefacts and lineage

- **W&B Artifacts** track lineage natively — "run B used
  artifact `train-data:v3` produced by run A". The DAG
  view is built in.
- **MLflow** tracks artefacts and (increasingly) inputs, but
  lineage in the same sense usually comes from pairing
  MLflow with **DVC** (chapter 06) for datasets and the
  **Model Registry** for models.

If you like end-to-end lineage as a built-in feature, W&B is
ahead. If you like each concern as a separate,
composable tool (data in DVC, model in registry, code in
git), MLflow's design fits.

### Model Registry

Both have one. Both let you assign a stage or alias
(`@production`, `@staging`) to a specific model version and
promote across stages. Both integrate with a webhook / CI
handoff for deployment.

- MLflow's registry is covered end to end in **chapter 07**.
- W&B's Model Registry parallels it; if you use W&B, the
  chapter's concepts apply directly.

### Offline / air-gapped

- **MLflow** runs offline out of the box. Every artefact
  can live on local disk.
- **W&B** has an `offline` mode (`WANDB_MODE=offline`) that
  queues logs locally and syncs later with `wandb sync`.
  It works well; verify with your security team that the
  eventual sync is acceptable.

### Cost

- **MLflow.** Open source. Self-hosted infrastructure only.
- **W&B.** Free tier for individuals and open-source
  projects; per-seat pricing for teams; enterprise pricing
  for on-prem.

<!-- needs-research: W&B pricing tiers and any changes to the
free-for-academics / free-for-open-source policies. Confirm
before recommending in a real engagement. -->

## When to pick which

A rough decision guide, aimed at a real team choice:

- **Pick MLflow** when any of these applies:
  - You need self-hosting for compliance, cost, or offline
    reasons.
  - You already run Databricks (their hosted MLflow is the
    default).
  - You want the tracker composable with independent tools
    (DVC for data, git for code, Model Registry for models)
    and not bundled into one vendor's product.
  - Cost matters and headcount is small — nothing to pay
    per seat.
- **Pick W&B** when any of these applies:
  - You want the best comparison UI on the market with zero
    ops burden.
  - Rich media logging (images, video, tables) is a daily
    need — computer vision, generative modelling.
  - You want native sweeps and lineage without wiring extra
    tools together.
  - Your team is comfortable with a SaaS dependency and your
    data can leave your VPC.

If you have no strong reason either way, learn MLflow first
(this module) because the self-hosted skill is more
transferable across employers.

## Other trackers worth knowing

You will encounter these; they are not covered in depth.

- **TensorBoard.** Comes with TensorFlow, works with PyTorch
  via `torch.utils.tensorboard.SummaryWriter`. Excellent for
  local exploratory metrics; not designed to compare
  hundreds of runs across a team. Both MLflow and W&B can
  ingest TensorBoard event files if you have legacy runs.
- **Aim.** Open source, self-hosted, focused on fast
  querying and comparison. A newer alternative to MLflow;
  the mental model is the same.
- **Neptune.** Hosted SaaS + on-prem. Popular in Kaggle-adjacent
  communities. Feature set overlaps W&B.
- **Comet.** Hosted SaaS. Similar surface to W&B; historically
  strong in comparison plots.
- **ClearML.** Open source with a hosted option. Includes an
  orchestrator (jobs, queues, agents) that goes beyond pure
  tracking.
- **DVC Studio.** Adds a tracking-ish UI on top of DVC
  experiments (chapter 06). Interesting if your data
  versioning is DVC-native already.

Every one of them speaks the same three nouns:
*experiment → run → attachments.* Fluency in one plus this
chapter's translation table is enough to onboard onto any of
them in a day.

## Migration and lock-in

Both MLflow and W&B store run data in structured backends
you can query. Migrating between them is mostly:

1. Export runs from the source. MLflow: `search_runs()` +
   `download_artifacts`. W&B: `wandb.Api()` iteration.
2. For each run in the source, `mlflow.start_run()` /
   `wandb.init()` on the target and re-log its params,
   metrics (respecting step), tags, and artefacts.

The friction is not the API; it is that historical model
artefacts and their loading dependencies may be pinned to
the old tracker's helpers. `mlflow.pytorch.log_model` writes
a specific manifest; loading it via W&B requires bypassing
that path.

**The practical rule:** switch trackers early if you are
going to. After a year of runs, the accumulated tooling
around one tracker (dashboards, reports, cron jobs, CI
webhooks) is what will resist migration, not the runs
themselves.

## Common pitfalls

1. **Choosing on the demo, not the deployment.** W&B's demo
   is stunning. If you cannot deploy SaaS, that beauty is
   irrelevant. Choose against your constraints, not the
   marketing.
2. **Running two trackers at once "for insurance".** Doubles
   the instrumentation burden; halves the confidence that
   either is complete. Pick one and commit.
3. **Assuming tracker == pipeline.** A tracker records runs.
   Neither MLflow nor W&B replaces an orchestrator (Airflow,
   Prefect, Dagster) or a data-versioning tool (DVC). Chapters
   06 and 09 come back to this.
4. **Leaking credentials.** W&B's API key belongs in a
   secret store; commit it to git once and every open-source
   scanner will find it. `WANDB_API_KEY` env var, never a
   hard-coded string.

## Summary

- MLflow and W&B share the same mental model. Learning one
  makes the other a translation exercise.
- The real axis of choice is deployment (self-host vs SaaS),
  UI polish, and cost — not the API.
- MLflow wins on self-hosting, offline / air-gapped
  workflows, and composition with independent tools.
- W&B wins on UI, rich-media logging, native sweeps, and
  built-in lineage.
- Other trackers exist (Aim, Neptune, Comet, ClearML,
  TensorBoard) and speak the same nouns. Pick one; migrate
  only if a constraint forces it.

## Where this goes next

Chapter 04 leaves the tracker behind and turns to the second
half of the module: **reproducibility**. The first
prerequisite for a reproducible run is a controlled random
seed at every layer that has one.
