# 02 — Packaging the model artifact

> **Reading time:** ~30 minutes.
> **Prerequisite chapter:** 01. `mod-106` chapter 07 (model
> registry) is a useful companion — this chapter is what
> "artifact" means in registry vocabulary.

## Motivation

Before the service, the artifact. An artifact is *whatever
you have to load into a fresh process to reproduce the
prediction*. That is at least the model weights, almost always
the preprocessing pipeline, and — if you want six-month-old-
you to make sense of it — a small amount of metadata.

Get the packaging wrong and every other chapter of this
module is fighting the same fire: the service starts, the
model loads, and the first live request produces the wrong
number because a scaler was fit on a different set of columns,
or an sklearn version cannot unpickle the object, or the
pipeline was saved with a lambda that no longer exists.

This chapter is the short list of packaging formats you
should know, the trade-offs between them, and — most
importantly — the metadata that has to travel with the file.

## The three questions the artifact has to answer

Before picking a format, be explicit about what "the artifact"
needs to contain. A serving process, on a machine that has
never seen the training code, will need to answer three
questions:

1. **How do I turn the request payload into model input?**
   Column order, dtypes, imputation defaults, one-hot
   categories, normalisation statistics. This is the
   *preprocessing pipeline* — the `sklearn.pipeline.Pipeline`
   or equivalent transformer chain. It has to travel with the
   model, not be re-implemented in the serving code.
2. **How do I compute a prediction from that input?** The
   estimator — an `sklearn` model, a `torch.nn.Module`, a
   boosted-tree booster, an ONNX graph. This is the "model"
   in the informal sense.
3. **What did I train, and how do I identify it later?**
   Version, git SHA, training dataset hash, framework
   versions, metrics on the eval set, the MLflow run ID. This
   is *metadata*. It is the reason chapter 05 (structured
   logging) can put "model_version": "1.4.2" in every log
   line.

An artifact that only answers question 2 is not enough. Every
"model looks fine but production is broken" story is a story
about the first or third question.

## Formats — the shortlist

### Pickle / joblib

The scikit-learn default. `joblib.dump(model, "model.pkl")`
serialises the entire Python object — including preprocessing
pipelines, custom transformers, and any state hanging off
them.

```python
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier()),
]).fit(X_train, y_train)

joblib.dump(pipe, "artifact/model.joblib")
```

Loading is symmetric — `joblib.load("artifact/model.joblib")`
returns the pipeline, and `pipe.predict(...)` works
immediately.

**Strengths.**

- Preserves the entire object graph, including custom code.
- One line to save, one line to load.
- Handles NumPy arrays efficiently (`joblib` uses `pickle`
  protocol 5 with out-of-band buffers).

**Weaknesses.**

- **Not safe to load untrusted files.** `pickle.load` can
  execute arbitrary code embedded in the file. The Python
  documentation is explicit: never unpickle a file from a
  source you do not trust. Treat model files like source
  code — sign them, put them behind a registry, and never
  load one supplied by a user.
- **Framework-version-sensitive.** A pipeline pickled with
  `scikit-learn 1.4` may raise, warn, or silently misbehave
  when loaded with `scikit-learn 1.6`. The scikit-learn
  changelog documents version-sensitive attributes; the
  official guidance is that pickled models are compatible
  only within the same version. In practice, minor-version
  changes often work — but you cannot rely on it.
- **Custom transformers must be importable.** If your
  pipeline includes `MyFeatureBuilder`, the serving process
  must be able to `from mymodule import MyFeatureBuilder` or
  the load will raise. Lambda functions never survive a
  pickle round-trip in a fresh process.

Use joblib when the training and serving environments share a
`pyproject.toml` and you control both. That is the common
case for the level of this module.

### ONNX

The **Open Neural Network Exchange** format is a framework-
neutral graph representation. You export a trained model from
its native framework (PyTorch, scikit-learn, XGBoost, LightGBM)
to `.onnx`, and any ONNX runtime can execute it — including
Python, C++, C#, Java, and JavaScript runtimes.

```python
# scikit-learn -> ONNX via skl2onnx
from skl2onnx import to_onnx
onnx_model = to_onnx(pipe, X_train[:1].astype("float32"))
with open("artifact/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Strengths.**

- **Framework-independent.** A caller written in Go can load
  the same model your Python trainer produced.
- **Fast inference.** ONNX Runtime is optimised and often
  faster than the native framework for small-batch inference.
- **Serialised as a protobuf graph.** No arbitrary code
  execution risk on load.

**Weaknesses.**

- **Not every operator is supported.** Custom scikit-learn
  transformers and PyTorch layers may not have an ONNX
  equivalent; conversion can silently drop a step or fail.
  Verify the exported model produces the same predictions as
  the source model on a held-out sample *before* you ship it.
- **Preprocessing is fragile.** `skl2onnx` exports many
  scikit-learn preprocessors, but not all of them cleanly.
  Reason about `object`-dtype columns, `OneHotEncoder(handle_
  unknown="ignore")`, and string features very carefully.

Reach for ONNX when you need cross-language serving, when
inference latency matters more than the packaging convenience,
or when the platform team standardises on an inference server
(Triton, ONNX Runtime Server) that speaks ONNX natively.

### TorchScript / `torch.export`

For PyTorch models specifically. `torch.jit.script` or
`torch.jit.trace` produces a `.pt` file that a C++ runtime
(`libtorch`) can load without a Python interpreter.

```python
import torch
scripted = torch.jit.script(model.eval())
scripted.save("artifact/model.pt")
```

`torch.export` (introduced in PyTorch 2.x) is the newer
successor and produces an `ExportedProgram` — read the PyTorch
2.x export documentation for the current recommendation.

**Strengths.** Ships a PyTorch model to any environment that
has `libtorch`; strips the Python dependency; enables
graph-level optimisations.

**Weaknesses.** Python-side control flow (Python `if`, list
comprehensions) is not always faithfully captured; `trace`
loses control flow entirely, `script` requires a subset of
Python. Debugging conversion errors is its own skill.

<!-- needs-research: as of 2026, confirm the current PyTorch
recommendation between `torch.jit.script`, `torch.jit.trace`,
and `torch.export` for production serving. Link the current
PyTorch export docs. -->

### MLflow `pyfunc`

MLflow's `pyfunc` flavour wraps *any* model — scikit-learn,
PyTorch, XGBoost, a raw Python function — behind a uniform
`predict(model_input)` interface. The artifact is a directory
containing the model, its Python dependencies, and enough
metadata for MLflow to reconstruct the environment.

```python
import mlflow

mlflow.sklearn.log_model(
    pipe,
    name="model",
    input_example=X_train.head(5),
    signature=mlflow.models.infer_signature(X_train, pipe.predict(X_train)),
)
```

The saved directory contains `MLmodel` (metadata), `python_
env.yaml` (dependencies), `input_example.json`, `signature`
(input/output schema), and the model bytes themselves.

**Strengths.**

- **Uniform load interface.** `mlflow.pyfunc.load_model(uri)`
  returns something with `.predict(pandas_df) -> ...`
  regardless of the underlying framework.
- **Signature enforcement.** The saved signature is a schema.
  MLflow can validate incoming payloads against it, which is
  a useful defence against schema drift.
- **Environment metadata.** The `python_env.yaml` /
  `requirements.txt` files let the serving image reconstruct
  the training dependencies deterministically.

**Weaknesses.** Ties you to MLflow. The `pyfunc` wrapping
adds a small amount of overhead per prediction (usually
negligible; measure if you care).

For the level of this module, **MLflow `pyfunc` is the
default recommendation** when `mod-106` is already in the
picture — you already logged the model there, and MLflow's
signature is the input/output contract chapter 03 will lift
into pydantic.

### The framework-native format

Every framework has its own serialisation: XGBoost's
`booster.save_model("model.json")`, LightGBM's
`booster.save_model("model.txt")`, Hugging Face's
`AutoModel.save_pretrained(path)`, etc.

Use these when:

- The framework's format is more compact or faster to load
  than pickling the wrapper object,
- You want to load the model in a runtime that speaks the
  framework's format natively (e.g., XGBoost's C++ API,
  Hugging Face's TGI),
- The framework is a first-class citizen in a platform-
  provided inference server.

Combine with a separate serialised preprocessing pipeline —
usually joblib — if the framework only handles the model
itself.

## Metadata: the file next to the file

Whichever format you choose, save an `MLmodel.json` (or
`metadata.json`) alongside it. The contract:

```json
{
  "model_name": "customer-churn",
  "model_version": "1.4.2",
  "git_sha": "9a3c5f1",
  "trained_at": "2026-05-14T10:12:03Z",
  "framework": {
    "name": "scikit-learn",
    "version": "1.4.2"
  },
  "python_version": "3.11.9",
  "training_dataset": {
    "uri": "s3://data/customers/2026-05-13/",
    "hash": "sha256:c9f2..."
  },
  "input_schema": [
    {"name": "age", "dtype": "int64", "nullable": false},
    {"name": "plan_tier", "dtype": "category",
     "categories": ["free", "pro", "enterprise"]}
  ],
  "output_schema": [
    {"name": "churn_probability", "dtype": "float32"}
  ],
  "eval_metrics": {"roc_auc": 0.881, "pr_auc": 0.612},
  "mlflow_run_id": "a1b2c3d4"
}
```

Each field earns its place:

- **`model_version` and `git_sha`** — the two things every
  incident ticket will ask for. Chapter 05 puts them in every
  log line.
- **`framework.version` and `python_version`** — the two
  things every "cannot load model" error will need. Chapter
  06's Dockerfile pins to these.
- **`training_dataset.hash`** — the reproducibility hook.
  `mod-106` chapter 06 (DVC) is how you get this hash;
  chapter 09 there tied it to the MLflow run.
- **`input_schema` and `output_schema`** — the schema chapter
  03 lifts into pydantic. MLflow's `mlflow.models.Model.
  signature` produces exactly this shape; use it if you have
  it.

Writing this file by hand is the fallback. MLflow's
`log_model(..., signature=..., input_example=...)` produces
most of it automatically — a strong reason to route packaging
through MLflow if you can.

## Loading, once, at startup

Whichever format you use, **load the model exactly once, at
service startup, and hold the reference for the process
lifetime.** Reloading per request adds tens to hundreds of
milliseconds and defeats every latency goal in chapter 09.

```python
# app/model.py
from functools import lru_cache
import joblib

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(_artifact_path())
```

The `lru_cache` idiom is a lightweight singleton. FastAPI's
`lifespan` context manager (chapter 03) is the more explicit
version, and gives you a place to warm the model with a dummy
prediction — often necessary because the *first* call after
load is materially slower than subsequent ones (JIT
compilation, page faulting the weights into RAM).

## Where the artifact lives

An artifact on your laptop is not deployable. Somewhere
between "trained" and "served" it needs a home the serving
process can reach:

- **Local filesystem inside the image.** Simplest: build the
  model directly into the Docker image (chapter 06). One
  image, one model version. Redeploy to update.
- **Object storage** (S3, GCS, Azure Blob). The service
  reads the artifact at startup from a URI. Decouples image
  from model; the same image can serve different model
  versions.
- **Model registry.** MLflow, SageMaker, Vertex, or an in-
  house registry. The service loads by *name* and *version*
  (`models:/customer-churn/production`) and the registry
  resolves to storage. This is where `mod-106` chapter 07
  ends and this module begins.

For this module, either "baked into the image" or "loaded
from object storage at startup" is fine. Chapter 04
(lifecycle) covers when *readiness* has to wait for that
load to finish.

## A worked round-trip

```python
# training/save_artifact.py
import json
import subprocess
from pathlib import Path

import joblib

def save_artifact(pipeline, out_dir: Path, *, name: str, version: str,
                  framework: str, framework_version: str, python_version: str,
                  dataset_uri: str, dataset_hash: str,
                  input_schema: list, output_schema: list,
                  eval_metrics: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.joblib")

    metadata = {
        "model_name": name,
        "model_version": version,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip(),
        "framework": {"name": framework, "version": framework_version},
        "python_version": python_version,
        "training_dataset": {"uri": dataset_uri, "hash": dataset_hash},
        "input_schema": input_schema,
        "output_schema": output_schema,
        "eval_metrics": eval_metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
```

```python
# serving/load.py
import json
from pathlib import Path

import joblib


class ModelBundle:
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata

    @property
    def version(self) -> str:
        return self.metadata["model_version"]


def load_bundle(artifact_dir: Path) -> ModelBundle:
    model = joblib.load(artifact_dir / "model.joblib")
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    return ModelBundle(model=model, metadata=metadata)
```

That `ModelBundle` is what the rest of the module carries
around: `bundle.model.predict(...)` at request time, and
`bundle.version` in every log line, response header, and
metric label.

## Common pitfalls

1. **Pickling a `lambda`.** Any lambda embedded in a
   pipeline (e.g., `FunctionTransformer(lambda x: x + 1)`)
   will fail to load in a fresh process. Use a module-level
   function.
2. **Fitting the scaler in the notebook, then re-implementing
   `x - mean / std` by hand in the service.** Your service is
   now guaranteed to drift from training. Serialise the
   scaler.
3. **Loading the model at request time.** "Simplest thing
   that works" — until the first p99 latency measurement.
4. **Pinning `scikit-learn>=1.4` in the training environment
   and `scikit-learn` (unversioned) in the serving image.**
   The next `pip install` will pull a newer version and
   unpickle behaviour becomes a lottery. Pin exactly.
5. **No `git_sha` in metadata.** When an incident ticket asks
   "what model was serving at 03:14?" the answer is "we
   don't know".
6. **Unpickling a file from an unverified source.** The
   security warning is not theoretical — pickle can execute
   arbitrary Python. Model files are code; treat them that
   way.
7. **Exporting to ONNX and skipping the equivalence check.**
   Silent operator substitutions will produce a model that
   loads, runs, and returns subtly different numbers.

## Summary

- An artifact must answer three questions: how to preprocess
  the input, how to compute the prediction, and how to
  identify what was trained.
- Format choices trade off portability, speed, and
  convenience. **Joblib** for scikit-learn when you control
  both ends; **ONNX** for cross-language / cross-runtime;
  **TorchScript / `torch.export`** for PyTorch in a
  Python-free runtime; **MLflow `pyfunc`** as the uniform
  default when `mod-106` is already in the picture.
- Metadata travels with the artifact: version, git SHA,
  framework version, dataset hash, input/output schema. Every
  incident and every log line uses it.
- Load once, at startup. Never per request.
- Pickle is code. Never unpickle a file from a source you
  do not trust.

## Where this goes next

Chapter 03 wraps the `ModelBundle` from this chapter in a
FastAPI service: request validation with pydantic (using the
`input_schema` from metadata), the `/predict` route, the
lifespan hook that loads the model exactly once, and the
first cut of a request handler that returns the model
version alongside the prediction.
