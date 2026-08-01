# 07 — Model registry and promotion

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 02 (MLflow basics), 06 (data
> versioning) — you should understand how a run produces a
> model artefact and where the data comes from.

## Motivation

You trained a model. It is one file among many under
`mlruns/*/artifacts/model/`. Someone asks: which model is
the current production one? Which one should we deploy next
week? What version was serving traffic when yesterday's bug
report came in?

The **Model Registry** is the small piece of MLflow that
answers those questions. It gives every model a name, every
retraining a version number, and every promotion decision
a place to live that isn't Slack.

The registry is not the deploy target. It sits *between*
the training run and the deployment: training produces a
new version, someone (or CI) decides that version is ready,
the registry tags it, the deploy pipeline pulls the tagged
version. Chapter 07's job is the middle step.

## The mental model

Two nouns, one relationship:

- **Registered model.** A named entity — e.g.
  `"adult-income-classifier"`. Has metadata (description,
  tags), and a list of versions.
- **Model version.** A specific version of that model —
  `version 1`, `version 2`, …, one per registration event.
  Each version points at a specific run's `model/` artefact.

Between them:

- **Aliases** — human-readable pointers that name a version
  (`@champion`, `@staging`, `@shadow`). An alias moves as
  new versions get promoted. Deployers query
  `models:/adult-income-classifier@champion` and always
  get the current champion.
- **Tags** — free-form key/value labels on a version (e.g.
  `validated=true`, `test_suite_sha=abc123`). Used for
  filtering and audit trail.

Historically MLflow also had **stages** (`Staging` /
`Production` / `Archived`). Aliases are the modern
replacement — richer, more flexible, and not tied to a fixed
vocabulary. Both are still supported in current MLflow;
prefer aliases for new work.

<!-- needs-research: verify the current MLflow guidance on
stages vs aliases; the API is evolving and stages have been
labelled legacy in some releases. -->

## Registering a model from a training run

Two ways. The one-liner during training:

```python
import mlflow, mlflow.sklearn

with mlflow.start_run() as run:
    ...
    mlflow.sklearn.log_model(
        clf,
        artifact_path="model",
        registered_model_name="adult-income-classifier",
    )
```

Passing `registered_model_name` tells MLflow: "log this
model to the run *and* register it as a new version of the
named registered model, creating the registered model if it
doesn't exist yet." The version number is auto-assigned.

The two-step version, for when you want a run to exist first
and only some runs to be registered:

```python
from mlflow import MlflowClient

client = MlflowClient()
mv = client.create_model_version(
    name="adult-income-classifier",
    source=f"runs:/{run_id}/model",
    run_id=run_id,
    tags={"validated": "true"},
)
print(f"registered as version {mv.version}")
```

Use the two-step version in CI: train + eval + register only
if eval passes.

## Loading a registered model

Every registered version has a stable URI. Load by name +
version:

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/adult-income-classifier/3")
predictions = model.predict(X_new)
```

Or by alias (recommended for deployers):

```python
model = mlflow.pyfunc.load_model("models:/adult-income-classifier@champion")
```

`mlflow.pyfunc.load_model` returns a generic PyFunc wrapper —
it handles sklearn, PyTorch, XGBoost, TF, all flavours,
because the run's `MLmodel` manifest (chapter 02) declares
which flavour to load with.

## Aliases: promotion without lifecycle hell

Aliases are the primitive that makes safe promotion easy.

```python
client = MlflowClient()

# Promote version 3 to champion
client.set_registered_model_alias(
    name="adult-income-classifier",
    alias="champion",
    version=3,
)

# Meanwhile, put version 4 up as a challenger
client.set_registered_model_alias(
    name="adult-income-classifier",
    alias="challenger",
    version=4,
)
```

A deploy pipeline pulls `models:/adult-income-classifier@champion`
and never has to know which numeric version that maps to.
Promotion becomes one API call — no CI file to redeploy, no
config to bump.

Roll back:

```python
client.set_registered_model_alias(
    name="adult-income-classifier",
    alias="champion",
    version=2,        # last known good
)
```

Aliases are cheap; use them liberally. Common conventions:

- `@production` / `@staging` / `@shadow` — the three
  environments.
- `@champion` / `@challenger` — A/B test framing.
- `@last-known-good` — for one-command rollback.

## Tags: the audit trail

Every model version can carry tags. Use them to record
*what was true* about the version at registration time:

```python
client.set_model_version_tag(
    name="adult-income-classifier",
    version=3,
    key="git_sha",
    value=os.environ["GIT_SHA"],
)
client.set_model_version_tag(
    name="adult-income-classifier",
    version=3,
    key="data_md5",
    value=data_md5,      # from the DVC pointer
)
client.set_model_version_tag(
    name="adult-income-classifier",
    version=3,
    key="val_accuracy",
    value=str(val_acc),
)
```

Six months later, tag lookups answer "what code, data, and
metric was version 3 registered with?" without opening
individual runs.

## A minimal promotion workflow

Here is the workflow most small-to-medium teams settle on.
Every stage is a distinct CI job or a distinct human
decision:

```
train.py ──► run in MLflow ──► [pass eval?] ──► register as v(N+1)
                                                          │
                                                          ▼
                                        set alias @challenger
                                                          │
                                                          ▼
                                    shadow deploy compares
                                    @champion vs @challenger
                                                          │
                                                          ▼
                                    metric-based auto-promote
                                    or human `set_alias @champion`
```

The registry gives you an auditable object at every arrow.
When someone asks "what changed between vN and vN+1", the
answer is `git diff` on the tagged git SHAs + `dvc diff` on
the tagged data hashes.

## The registry vs. the artefact store — what's the difference

Two things that sound alike:

- **The artefact store** (chapter 02) holds the *file* the
  run produced. `runs:/<run_id>/model` is a file URI.
- **The registry** holds *pointers to those files*, plus
  names, versions, aliases, tags, and lineage.

You cannot delete an artefact and expect the registered
version pointing at it to still work — the registry's
version is a reference, not a copy. Keep this in mind if
you have a garbage-collection policy on `mlruns/`.

## The model signature — why it matters at registry time

The run's `MLmodel` manifest includes an *input signature*
if you passed `signature=` to `log_model` (chapter 02). The
signature is a schema:

```yaml
signature:
  inputs: '[{"name": "age", "type": "long"},
            {"name": "workclass", "type": "string"},
            ...]'
  outputs: '[{"type": "long"}]'
```

Consumers of a registered model — a serving container, an
inference SDK, a UI — read the signature to know what to
pass. Without it, every consumer has to reverse-engineer the
expected columns from the training code, which defeats the
purpose of a registry.

**Always pass a signature** when you register a model.
`infer_signature(X_train, model.predict(X_train))` is enough
for 90% of tabular cases. For PyTorch, pass an input example
tensor and let MLflow infer the shape and dtype.

## Programmatic search in the registry

Everything in the UI has a Python equivalent:

```python
from mlflow import MlflowClient

client = MlflowClient()

# List all registered models
for rm in client.search_registered_models():
    print(rm.name, [v.version for v in rm.latest_versions])

# List all versions of one model
for v in client.search_model_versions(
    "name='adult-income-classifier'"
):
    print(v.version, v.creation_timestamp, v.aliases)

# Find versions tagged as validated
for v in client.search_model_versions(
    "name='adult-income-classifier' and tags.validated = 'true'"
):
    print(v.version, v.tags)
```

This is how CI jobs make promotion decisions — query the
registry, apply a rule, call `set_registered_model_alias`.
No YAML pipeline modelling; no hand-edited config.

## Webhooks and the deploy handoff

The registry is not a deploy target. It is the *trigger*.
Two patterns for the handoff:

- **Pull.** Deploy service periodically queries the registry
  for the `@production` alias; if the version changed since
  last pull, redeploy. Simple; adds latency equal to the
  poll interval.
- **Push.** MLflow webhooks (or a wrapper) fire on
  `set_registered_model_alias` events. The webhook target is
  a CI job that redeploys. Faster, more moving parts.

Either way, the registry's job ends at the alias. Deployment
is `mod-108`.

<!-- needs-research: MLflow's built-in webhook support has
historically varied by version and by managed vs OSS. Verify
what ships in the current stable release; document any
fallback (e.g., poll-based agents). -->

## Governance features worth knowing

Real teams need more than a name and a version. The registry
offers:

- **Descriptions.** Human text on the registered model and
  on each version. "This is the credit-risk model owned by
  the underwriting team."
- **User attribution.** Every action (`create_version`,
  `set_alias`) records the user who did it. Useful for
  audit; requires a real (SQL-backed) tracking server, not
  the local file store.
- **Prevented deletes.** Configure the server to require an
  explicit `--archive` step before delete, so a model
  version's artefacts survive an accidental UI click.
- **Access control.** Managed MLflow (Databricks) has RBAC;
  self-hosted OSS MLflow has proxying + auth via a reverse
  proxy (nginx, oauth2-proxy). The registry inherits
  whatever the tracking server has.

The point of the registry is that it becomes the *system of
record* for "which model went where". That requires the
above to work end to end; do not skimp on the deployment.

## Common pitfalls

1. **Registering a model without a signature.** Downstream
   consumers cannot introspect the schema. Always pass
   `signature=`.
2. **Using stages when aliases exist.** Stages are legacy in
   current MLflow. New workflows should use aliases.
3. **Manual promotion via UI clicks only.** No audit, no
   code. Every promotion should be a scripted call that lives
   in CI or in a runbook — even if a human triggers it.
4. **Deleting a run's artefact but keeping the registered
   version.** The pointer dangles; loads fail. If you retain
   registered versions, retain the underlying artefacts.
5. **Registering every training run.** Registry hygiene
   matters. Register only versions that pass eval. Let
   experimentation live as unregistered runs.
6. **No tag for git SHA or data hash.** The version is opaque
   without them. The two lines of `set_model_version_tag` are
   the cheapest audit trail you will ever build.
7. **Different environments pointing at the same registered
   model.** Fine, as long as each has a distinct alias
   (`@staging`, `@production`). Bad, if two environments
   both read `@production` and you promote across only one.

## Summary

- The Model Registry names, versions, and aliases model
  artefacts so that "which model?" is a lookup instead of a
  question.
- Register from a training run by passing
  `registered_model_name=` to `log_model`, or by calling
  `client.create_model_version` after an eval gate in CI.
- Load by name+version or by alias. Deployers should load by
  alias; promotion is one API call.
- Tag every version with git SHA, data hash, and validation
  metric. Six months later, the tags are the audit trail.
- Aliases beat stages; script every promotion; keep unregistered
  runs for experimentation and register only what passes.
- The registry is the trigger, not the deploy target. `mod-108`
  is the deploy.

## Where this goes next

Chapter 08 turns to the fourth objective — running a small
experiment sweep. **Optuna** is the tool; MLflow captures
each trial as a nested run, and the winning configuration
gets registered by the pattern in this chapter.
