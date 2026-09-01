# Resources for mod-106 — Experiment Tracking & Reproducibility

A curated short list. The goal is depth-on-demand for the
chapter you got stuck on, not a comprehensive bibliography.
Primary documentation is listed first; papers, standards, and
tooling references follow.

## Cross-cutting: MLflow itself

- **MLflow — official documentation.**
  <https://mlflow.org/docs/latest/index.html>
  The main reference. Every API named in chapters 02, 07, and
  08 (`mlflow.start_run`, `log_param`, `log_metric`,
  `log_artifact`, `log_model`, `search_runs`,
  `MlflowClient`, Model Registry) is documented here with the
  full signature and examples.
- **MLflow Tracking — user guide.**
  <https://mlflow.org/docs/latest/tracking.html>
  The chapter-02 material in book form. Covers experiments,
  runs, params/metrics/tags/artefacts, autologging, and the
  tracking URI backends.
- **MLflow Models — packaging and flavours.**
  <https://mlflow.org/docs/latest/models.html>
  The `MLmodel` manifest, the flavour helpers
  (`mlflow.sklearn`, `mlflow.pytorch`, `mlflow.xgboost`, …),
  signatures, and input examples cited by chapter 02.
- **MLflow Model Registry — user guide.**
  <https://mlflow.org/docs/latest/model-registry.html>
  Registered models, versions, aliases, tags, and the
  promotion workflow chapter 07 walks through.
- **MLflow — REST API reference.**
  <https://mlflow.org/docs/latest/rest-api.html>
  Useful when you need to script the tracking server from a
  language other than Python (CI job, Go service, notebook
  kernel proxy).
- **MLflow — GitHub repository and release notes.**
  <https://github.com/mlflow/mlflow>
  Worth skimming when you upgrade a minor version — the
  Model Registry and autologging surfaces evolve.
- **Chen, Andrew, et al. "Developments in MLflow: A System to
  Accelerate the Machine Learning Lifecycle" (DEEM 2020).**
  <https://dl.acm.org/doi/10.1145/3399579.3399867>
  The paper behind the tool. Section 3 (design) is the
  clearest single account of the tracking-server /
  artefact-store split chapter 02 describes.
- **Zaharia, Matei, et al. "Accelerating the Machine Learning
  Lifecycle with MLflow" (IEEE Data Eng. Bull., 2018).**
  <http://sites.computer.org/debull/A18dec/p39.pdf>
  The original MLflow announcement paper. Historical context
  for how the tracker-plus-registry-plus-projects trio was
  originally scoped.

## Cross-cutting: Weights & Biases

- **Weights & Biases — official documentation.**
  <https://docs.wandb.ai/>
  The main reference. The chapter-03 W&B example maps onto
  the "Track experiments" and "Log data with wandb.log"
  guides.
- **W&B — Artifacts guide.**
  <https://docs.wandb.ai/guides/artifacts/>
  The dataset / model versioning story on the W&B side.
  Directly comparable to DVC (chapter 06) and MLflow's Model
  Registry (chapter 07).
- **W&B — Sweeps guide.**
  <https://docs.wandb.ai/guides/sweeps/>
  W&B's native hyperparameter-search product; the alternative
  to Optuna + MLflow (chapter 08) if you are already on the
  W&B stack.
- **W&B — Model Registry guide.**
  <https://docs.wandb.ai/guides/registry/>
  The W&B analogue of MLflow's Model Registry chapter 07
  covers.

## Cross-cutting: broader tracking landscape

- **Neptune.ai — documentation.**
  <https://docs.neptune.ai/>
  The tracker chapter 03 mentions as the "cleanest metadata
  model" alternative. Same experiment → run mental model.
- **Aim — documentation.**
  <https://aimstack.readthedocs.io/>
  Open-source, self-hosted, very fast UI. Worth knowing as an
  MLflow alternative when the MLflow UI feels heavyweight.
- **Comet — documentation.**
  <https://www.comet.com/docs/>
  Long-established hosted tracker; competes with W&B.
- **ClearML — documentation.**
  <https://clear.ml/docs/latest/docs/>
  Tracker + pipeline orchestrator + agent; broader scope
  than MLflow.
- **TensorBoard — documentation.**
  <https://www.tensorflow.org/tensorboard/>
  The oldest of the "look at loss curves" tools. Works with
  PyTorch via `torch.utils.tensorboard.SummaryWriter`. Not a
  full tracker (no cross-run registry) but frequently
  embedded inside MLflow / W&B artefact viewers.

## Cross-cutting: reproducibility as a practice

- **Pineau, Joelle, et al. "Improving Reproducibility in
  Machine Learning Research (a report from the NeurIPS 2019
  Reproducibility Program)" (JMLR, 2021).**
  <https://www.jmlr.org/papers/v22/20-303.html>
  The paper behind the ML Reproducibility Checklist. Section
  4 (the checklist) is the standard chapters 01 and 09 quietly
  aim at.
- **ML Reproducibility Checklist.**
  <https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf>
  Two pages. Print it out; use it as the acceptance criterion
  for `REPORT.md` in exercise-04.
- **Sculley, D., et al. "Hidden Technical Debt in Machine
  Learning Systems" (NeurIPS 2015).**
  <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  The paper behind chapter 01's "silent failure mode"
  framing. The "changes anything, changes everything" (CACE)
  principle is why tracking exists.
- **Amershi, Saleema, et al. "Software Engineering for Machine
  Learning: A Case Study" (ICSE-SEIP 2019).**
  <https://ieeexplore.ieee.org/document/8804457>
  Microsoft's field study of the ML lifecycle inside a large
  organisation. Sections 4 and 5 (workflow and challenges)
  are the observed version of what this module tries to
  enforce.
- **Papers With Code — Reproducibility Reports.**
  <https://paperswithcode.com/rc2022>
  A repository of community reproductions of published ML
  results. Worth reading a couple of reports for the tone
  and shape of the write-up chapter 09 targets.
- **FAIR Principles (Findable, Accessible, Interoperable,
  Reusable).**
  <https://www.go-fair.org/fair-principles/>
  The data-management principles behind DVC-style content
  addressing and dataset provenance.

## Chapter 01 — Why tracking and reproducibility

- **Peng, Roger D. "Reproducible Research in Computational
  Science" (Science, 2011).**
  <https://www.science.org/doi/10.1126/science.1213847>
  The paper that popularised the "repeatable / reproducible
  / replicable" distinction chapter 01 uses.
- **National Academies of Sciences, Engineering, and Medicine.
  *Reproducibility and Replicability in Science* (2019).**
  <https://nap.nationalacademies.org/catalog/25303/reproducibility-and-replicability-in-science>
  The book-length treatment. Chapter 3's definitions of
  reproducibility vs. replicability are the ones chapter 01
  compresses into a table.
- **The Turing Way — *A Handbook for Reproducible, Ethical
  and Collaborative Data Science*.**
  <https://the-turing-way.netlify.app/>
  A community-maintained textbook. The "Reproducible
  Research" and "Version Control" chapters are the friendlier
  long-form pass over mod-106's first half.
- **ACM — Artifact Review and Badging (Version 1.1).**
  <https://www.acm.org/publications/policies/artifact-review-and-badging-current>
  The badging vocabulary academic venues use ("Artifacts
  Evaluated — Reproduced", "Results Reproduced"). Useful
  reference for the phrases chapter 01 defines.

## Chapter 02 — MLflow: runs, metrics, and artifacts

- **MLflow Tracking — Python API.**
  <https://mlflow.org/docs/latest/python_api/mlflow.html>
  Every function chapter 02 uses (`log_param`, `log_metric`,
  `log_metrics`, `log_artifact`, `log_artifacts`, `set_tag`,
  `log_dict`, `log_text`, `log_figure`, `start_run`,
  `set_experiment`, `set_tracking_uri`, `search_runs`).
- **MLflow — autologging documentation.**
  <https://mlflow.org/docs/latest/tracking/autolog.html>
  The per-framework list of what `mlflow.<framework>.autolog()`
  captures and any known caveats.
- **MLflow — dataset tracking (`mlflow.data`).**
  <https://mlflow.org/docs/latest/tracking/data-api.html>
  The `mlflow.data.from_pandas`, `from_numpy`, `from_spark`,
  and `log_input` surface cited at the end of chapter 02.
- **MLflow — system tags reference.**
  <https://mlflow.org/docs/latest/tracking/tracking-api.html#system-tags>
  The `mlflow.source.git.commit`, `mlflow.source.name`,
  `mlflow.user`, and other auto-attached tags described in
  chapter 02.
- **MLflow — tracking backends and artefact stores.**
  <https://mlflow.org/docs/latest/tracking/backend-stores.html>
  <https://mlflow.org/docs/latest/tracking/artifacts-stores.html>
  The SQL-backend + object-store deployment discussed at the
  end of chapter 02.

## Chapter 03 — Picking a tracker: MLflow vs Weights & Biases

- **W&B — `wandb.init` and `wandb.log` API reference.**
  <https://docs.wandb.ai/ref/python/init>
  <https://docs.wandb.ai/ref/python/log>
  The two functions the W&B version of chapter 02's example
  is built on.
- **W&B — Experiments comparison guide.**
  <https://docs.wandb.ai/guides/track/compare/>
  The "compare runs" UI that mirrors MLflow's Compare view.
- **W&B — Reports (write-ups from runs).**
  <https://docs.wandb.ai/guides/reports/>
  W&B's built-in "reproducibility report"-adjacent surface;
  a hosted alternative to writing `REPORT.md` by hand.
- **DVC — DVCLive.**
  <https://dvc.org/doc/dvclive>
  A lightweight logger that writes plain files to disk and
  ships to MLflow / W&B / Studio. Worth knowing as the
  minimal-buy-in option chapter 03 alludes to.

## Chapter 04 — Controlling randomness: seeds and determinism

- **PyTorch — reproducibility notes.**
  <https://pytorch.org/docs/stable/notes/randomness.html>
  The canonical reference. What `torch.manual_seed`,
  `torch.use_deterministic_algorithms`, cuDNN benchmark
  mode, and the `CUBLAS_WORKSPACE_CONFIG` env var actually
  guarantee.
- **PyTorch — `torch.utils.data.DataLoader` worker seeding.**
  <https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading>
  The `worker_init_fn` + generator pattern chapter 04 uses
  to give each DataLoader worker a deterministic RNG.
- **PyTorch — `torch.use_deterministic_algorithms`.**
  <https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>
  The single switch that flips PyTorch into "raise on
  nondeterministic op" mode. Read the notes for the list of
  ops that have no deterministic implementation on CUDA.
- **NVIDIA — cuDNN reproducibility guidance.**
  <https://docs.nvidia.com/deeplearning/cudnn/latest/reference/reproducibility.html>
  The vendor side of the "cuDNN benchmark mode + non-
  deterministic algorithms" story. Worth reading once so you
  know which ops are the culprits.
- **NumPy — random sampling (`numpy.random`).**
  <https://numpy.org/doc/stable/reference/random/index.html>
  The `Generator` / `default_rng` API. Chapter 04's harness
  seeds the legacy global RNG as well as pointing users at
  `default_rng`.
- **NEP 19 — Random Number Generator Policy.**
  <https://numpy.org/neps/nep-0019-rng-policy.html>
  Why `numpy.random.seed` is legacy and why new code should
  prefer `numpy.random.default_rng(seed)`.
- **Python — `random` module documentation.**
  <https://docs.python.org/3/library/random.html>
  `random.seed`, and the note that this RNG is not
  cryptographically secure (use `secrets` for tokens, not
  RNG state).
- **CPython — `PYTHONHASHSEED` documentation.**
  <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>
  The environment variable chapter 04's harness sets to
  stabilise hash-based iteration order (`set`, `dict`).
- **TensorFlow — `tf.keras.utils.set_random_seed`.**
  <https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed>
  The TF-side equivalent of chapter 04's harness. Include if
  your team also runs TF code.
- **scikit-learn — Glossary: `random_state`.**
  <https://scikit-learn.org/stable/glossary.html#term-random_state>
  Explains what `random_state=` does in every sklearn
  estimator, splitter, and cross-validator — the "seed
  every entry point" advice chapter 04 gives, sklearn-side.

## Chapter 05 — Pinning environments: lockfiles, Docker, hardware

- **uv — official documentation.**
  <https://docs.astral.sh/uv/>
  The modern default. Chapter 05's `uv lock` / `uv sync
  --frozen` workflow is documented under "Projects" and
  "Locking and syncing".
- **Poetry — official documentation.**
  <https://python-poetry.org/docs/>
  The other widely-used lockfile toolchain. `poetry lock` /
  `poetry install --no-update` is the equivalent workflow.
- **pip-tools.**
  <https://github.com/jazzband/pip-tools>
  `pip-compile --generate-hashes` — the older, still-supported
  approach chapter 05 mentions.
- **conda-lock.**
  <https://github.com/conda/conda-lock>
  Lockfiles for `conda` environments. Relevant if your team
  is on the conda-forge stack rather than PyPI.
- **PyPA — Python Packaging User Guide.**
  <https://packaging.python.org/>
  The umbrella reference for `pyproject.toml`, wheels, and
  the sdist/wheel distinction chapter 05 leans on.
- **PEP 517 / PEP 518 — build system spec.**
  <https://peps.python.org/pep-0517/>
  <https://peps.python.org/pep-0518/>
  Why `pyproject.toml` looks the way it does. Skim once so
  the `[build-system]` block stops being magic.
- **PEP 621 — project metadata in `pyproject.toml`.**
  <https://peps.python.org/pep-0621/>
  The `[project]` table (name, version, dependencies) every
  modern tool consumes.
- **PEP 665 — a lockfile format for Python (withdrawn).**
  <https://peps.python.org/pep-0665/>
  Historical: the aborted attempt to standardise a
  cross-tool lockfile format. Useful context for why every
  tool still has its own.
- **Docker — Dockerfile reference.**
  <https://docs.docker.com/engine/reference/builder/>
  Every directive (`FROM`, `RUN`, `COPY`, `ENV`,
  `WORKDIR`, `USER`, `ENTRYPOINT`) chapter 05's image uses.
- **Docker — pulling by digest.**
  <https://docs.docker.com/engine/reference/commandline/pull/#pull-an-image-by-digest-immutable-identifier>
  The `image@sha256:…` form chapter 05 requires for a
  reproducible `docker pull`.
- **Docker BuildKit — reproducible builds.**
  <https://docs.docker.com/build/building/best-practices/#reproducibility>
  Best-practice guide for byte-comparable image builds
  (SOURCE_DATE_EPOCH, deterministic layer order).
- **Reproducible Builds project.**
  <https://reproducible-builds.org/>
  The broader software-supply-chain movement. Worth knowing
  the name; the techniques (fixed timestamps, sorted file
  order) generalise to ML container images.
- **NVIDIA — CUDA compatibility documentation.**
  <https://docs.nvidia.com/deploy/cuda-compatibility/>
  The driver / toolkit / cuDNN compatibility matrix chapter
  05 flags as the "silent breakage" surface.
- **Development Containers (`devcontainer.json`).**
  <https://containers.dev/>
  The VS Code / GitHub Codespaces standard for
  reproducible development environments. A lighter-weight
  path than shipping a full production image for local dev.
- **`repo2docker`.**
  <https://repo2docker.readthedocs.io/>
  A tool that reads a repo's config files and builds an
  image automatically. The technology behind Binder; worth
  knowing when the goal is "click and run" reproducibility
  for a notebook.

## Chapter 06 — Versioning data with DVC

- **DVC — official documentation.**
  <https://dvc.org/doc>
  The main reference. Chapter 06's six-command intro maps
  onto the "Get Started" guide.
- **DVC — User Guide.**
  <https://dvc.org/doc/user-guide>
  The long-form explanation of pointer files, the cache, and
  remotes.
- **DVC — Pipelines (`dvc.yaml`) reference.**
  <https://dvc.org/doc/user-guide/pipelines/defining-pipelines>
  <https://dvc.org/doc/user-guide/project-structure/dvcyaml-files>
  The features → train → evaluate pipeline chapter 06 walks
  through.
- **DVC — remotes.**
  <https://dvc.org/doc/user-guide/data-management/remote-storage>
  S3, GCS, Azure Blob, SSH, HTTP, WebHDFS, and shared
  filesystem remotes.
- **DVC — command reference.**
  <https://dvc.org/doc/command-reference>
  Every `dvc <verb>` (add, push, pull, repro, run, gc, get,
  import, plots) with flags.
- **git-LFS.**
  <https://git-lfs.com/>
  The narrower predecessor for large-file versioning.
  Chapter 06 contrasts DVC's scope against LFS; useful
  reference if a team is already on LFS.
- **LakeFS — official documentation.**
  <https://docs.lakefs.io/>
  Git-like versioning on top of an object store. An
  alternative to DVC when the dataset is a table on S3, not
  a file on disk.
- **Delta Lake — documentation.**
  <https://docs.delta.io/>
  Table-format versioning (Time Travel) for Parquet on
  object stores. Adjacent to DVC when the data is
  columnar-warehouse-shaped.
- **Apache Iceberg — documentation.**
  <https://iceberg.apache.org/docs/latest/>
  The other widely-adopted table format with snapshot-based
  versioning.
- **Pachyderm — documentation.**
  <https://docs.pachyderm.com/>
  Data-versioning + pipeline orchestrator; the Kubernetes-
  native comparator to DVC.
- **Gebru, Timnit, et al. "Datasheets for Datasets"
  (Communications of the ACM, 2021).**
  <https://arxiv.org/abs/1803.09010>
  The dataset-documentation template chapter 06 alludes to
  when it says "name the dataset, don't just point at bytes".

## Chapter 07 — Model registry and promotion

- **MLflow — Model Registry Python API.**
  <https://mlflow.org/docs/latest/python_api/mlflow.client.html>
  `create_registered_model`, `create_model_version`,
  `set_registered_model_alias`, `set_model_version_tag`, and
  the search APIs chapter 07 uses.
- **MLflow — Model Registry workflows guide.**
  <https://mlflow.org/docs/latest/model-registry.html#workflows>
  The recommended promotion workflows: aliases (modern) and
  stages (legacy).
- **MLflow — `models:/` URI scheme.**
  <https://mlflow.org/docs/latest/models.html#model-uri>
  The `models:/name/version` and `models:/name@alias`
  URI forms deployers query in chapter 07.
- **MLflow — model serving (built-in).**
  <https://mlflow.org/docs/latest/deployment/index.html>
  `mlflow models serve` and the built-in deployment plugins.
  The bridge from chapter 07 (registration) to `mod-108`
  (serving).
- **Mitchell, Margaret, et al. "Model Cards for Model
  Reporting" (FAT* 2019).**
  <https://arxiv.org/abs/1810.03677>
  The model-documentation template. Aliases + tags in the
  Model Registry are the machine-readable half; model cards
  are the human-readable half.
- **Google — Model Card Toolkit.**
  <https://github.com/tensorflow/model-card-toolkit>
  Reference implementation of the model-cards paper.
- **Hugging Face — Model Cards.**
  <https://huggingface.co/docs/hub/model-cards>
  The hosted-hub side. Every model on the Hub has one; a
  useful reference for what a well-populated model card
  looks like.
- **ONNX — official documentation.**
  <https://onnx.ai/>
  <https://onnx.ai/onnx/intro/>
  The interchange format worth knowing when a model needs to
  cross framework boundaries between training (PyTorch) and
  serving (something else). Chapter 07 mentions it as an
  export path.

## Chapter 08 — Hyperparameter sweeps with Optuna

- **Optuna — official documentation.**
  <https://optuna.readthedocs.io/en/stable/>
  The main reference. Every `suggest_*`, sampler, pruner,
  storage backend, and callback chapter 08 uses.
- **Optuna — tutorial.**
  <https://optuna.readthedocs.io/en/stable/tutorial/index.html>
  The step-by-step walkthrough. The "First Optimization"
  and "Pythonic Search Space" sections are the natural pair
  with chapter 08's "hello, Optuna".
- **Optuna — samplers reference.**
  <https://optuna.readthedocs.io/en/stable/reference/samplers/index.html>
  `TPESampler`, `RandomSampler`, `CmaEsSampler`,
  `GPSampler`, `NSGAIISampler`. What each is good at and
  when to switch.
- **Optuna — pruners reference.**
  <https://optuna.readthedocs.io/en/stable/reference/pruners.html>
  `MedianPruner`, `HyperbandPruner`,
  `SuccessiveHalvingPruner`, `PatientPruner`, and the
  reporting protocol.
- **Optuna — `optuna-integration` (MLflow, W&B, Lightning, …).**
  <https://optuna-integration.readthedocs.io/en/latest/>
  The `MLflowCallback` and `WeightsAndBiasesCallback`
  chapter 08 uses to wire each trial into the tracker as a
  nested run.
- **Optuna Dashboard.**
  <https://optuna-dashboard.readthedocs.io/>
  The UI companion. Not required, but the parallel-
  coordinates and slice plots are the fastest way to see
  which hyperparameters matter.
- **Akiba, Takuya, et al. "Optuna: A Next-generation
  Hyperparameter Optimization Framework" (KDD 2019).**
  <https://arxiv.org/abs/1907.10902>
  The paper. Section 3 (design) and section 4 (algorithms)
  are the sourced version of chapter 08's TPE + pruner
  discussion.
- **Bergstra, James, and Yoshua Bengio. "Random Search for
  Hyper-Parameter Optimization" (JMLR 2012).**
  <https://www.jmlr.org/papers/v13/bergstra12a.html>
  The paper behind "random search beats grid search when
  only some hyperparameters matter" — cited in chapter 08's
  intro.
- **Bergstra, James, et al. "Algorithms for Hyper-Parameter
  Optimization" (NeurIPS 2011).**
  <https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html>
  The TPE paper. The default sampler in Optuna.
- **Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams.
  "Practical Bayesian Optimization of Machine Learning
  Algorithms" (NeurIPS 2012).**
  <https://arxiv.org/abs/1206.2944>
  The Gaussian-process Bayesian-optimisation paper. The
  other classical branch of sequential model-based
  optimisation.
- **Li, Lisha, et al. "Hyperband: A Novel Bandit-Based
  Approach to Hyperparameter Optimization" (JMLR 2018).**
  <https://arxiv.org/abs/1603.06560>
  The Hyperband paper. The algorithm behind Optuna's
  `HyperbandPruner` and `SuccessiveHalvingPruner`.
- **Falkner, Stefan, Aaron Klein, and Frank Hutter. "BOHB:
  Robust and Efficient Hyperparameter Optimization at
  Scale" (ICML 2018).**
  <https://arxiv.org/abs/1807.01774>
  BOHB — TPE + Hyperband. Worth knowing as the "successor"
  to TPE that some Optuna alternatives use.
- **Ray Tune — documentation.**
  <https://docs.ray.io/en/latest/tune/index.html>
  The distributed-sweep scheduler chapter 08 flags as the
  "when Optuna's parallelism stops scaling" next step.
- **Kubeflow Katib — documentation.**
  <https://www.kubeflow.org/docs/components/katib/>
  Kubernetes-native HPO. The other production-scale
  alternative when the workload lives on K8s.

## Chapter 09 — The reproducibility report

- **Papers With Code — Reproducibility Reports (2019–2022).**
  <https://paperswithcode.com/rc2022>
  The community archive of reproduction reports. Reading two
  or three is the fastest way to internalise the template
  chapter 09 sketches.
- **Joint Committee for Guides in Metrology. *Evaluation of
  Measurement Data — Guide to the Expression of Uncertainty
  in Measurement* (GUM, 2008).**
  <https://www.bipm.org/en/publications/guides/gum.html>
  The metrology reference for stating a measurement with an
  uncertainty (mean ± std). Chapter 09's "best result:
  value ±std, seeds …" line is the ML dialect of this.
- **Bouthillier, Xavier, et al. "Accounting for Variance in
  Machine Learning Benchmarks" (MLSys 2021).**
  <https://arxiv.org/abs/2103.03098>
  The paper behind chapter 09's "one seed is not a
  result — quote a distribution over seeds" advice.
- **Henderson, Peter, et al. "Deep Reinforcement Learning that
  Matters" (AAAI 2018).**
  <https://arxiv.org/abs/1709.06560>
  The classic RL-reproducibility paper. Section 5 (variance
  in reported results) is the concrete case study for the
  "quote seed variance" advice.
- **Hydra — official documentation.**
  <https://hydra.cc/>
  The config-management framework worth knowing when the
  `--config` argument in chapter 09's reproduce block starts
  needing composition, overrides, or sweeps.
- **OmegaConf.**
  <https://omegaconf.readthedocs.io/>
  Hydra's underlying config library. Directly usable as a
  lightweight alternative to Hydra.

## Standards, adjacent references

- **SLSA — Supply-chain Levels for Software Artifacts.**
  <https://slsa.dev/>
  The software-supply-chain framework. Relevant to chapter
  05's "pin by digest, not by tag" and chapter 07's
  "who signed off on this model version".
- **Sigstore — cosign.**
  <https://docs.sigstore.dev/cosign/overview/>
  Container-image signing. The mechanism SLSA-conformant
  ML pipelines use to make "this image is what the CI
  produced" verifiable.
- **CycloneDX — SBOM specification.**
  <https://cyclonedx.org/specification/overview/>
  Software Bill of Materials for a Docker image or a Python
  environment. The machine-readable version of "here are
  the exact package versions in this run".
- **NIST — AI Risk Management Framework (AI RMF 1.0).**
  <https://www.nist.gov/itl/ai-risk-management-framework>
  Where "reproducibility" sits in the broader responsible-AI
  vocabulary. Useful cross-reference for teams whose
  compliance function asks for it.

## Datasets used in the exercises

The exercises deliberately do not prescribe a dataset — they
reuse whatever is in play from `mod-103`, `mod-104`, or
`mod-105`. Two options that appear in the chapter examples:

- **UCI Adult (Census Income).**
  Kohavi, Ron. "Scaling Up the Accuracy of Naive-Bayes
  Classifiers: A Decision-Tree Hybrid" (KDD 1996).
  <https://archive.ics.uci.edu/dataset/2/adult>
  Loadable via `sklearn.datasets.fetch_openml("adult",
  version=2)`. Used in chapter 02 and the sweep example in
  chapter 08.
- **CIFAR-10 / CIFAR-100.**
  Krizhevsky, Alex. "Learning Multiple Layers of Features
  from Tiny Images" (Tech report, University of Toronto,
  2009).
  <https://www.cs.toronto.edu/~kriz/cifar.html>
  Loadable via `torchvision.datasets.CIFAR10` /
  `CIFAR100`. A good default when the training script comes
  from `mod-105`.

<!-- needs-research: on each autonomous research cycle, verify these URLs still resolve. In particular: (a) MLflow docs pages (they reorganised the URL scheme once; another rename is plausible on a 3.x cut); (b) DVC docs (they periodically shuffle the User Guide vs Command Reference split); (c) Optuna sampler/pruner references (samplers list expands most releases); (d) W&B docs (frequently move under /guides/ subpaths). Swap in current pages when a doc URL 404s. -->
