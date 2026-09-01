# Resources for mod-109 — ML Monitoring & Drift Detection

A curated short list. The goal is depth-on-demand for the
chapter you got stuck on, not a comprehensive bibliography.
Primary documentation is listed first; papers, standards,
and tools follow.

## Cross-cutting: the stacks the module leans on

Every chapter reads from at least one of these. Bookmark
them once; the rest becomes lookup.

- **Prometheus — official documentation.**
  <https://prometheus.io/docs/introduction/overview/>
  The metrics format and query language the module
  emits into. The "Concepts → Metric types" and
  "Best practices → Metric and label naming" pages
  are the primary sources chapter 02 quietly follows.
- **Prometheus Python client library.**
  <https://prometheus.github.io/client_python/>
  The library chapter 02's `Counter` / `Histogram` /
  `Gauge` snippets use. Read the "Instrumenting" and
  "Exposing" pages first; the "Multiprocess Mode"
  page is what exercise-01's stretch goal exercises.
- **OpenTelemetry — Python SDK.**
  <https://opentelemetry.io/docs/languages/python/>
  The vendor-neutral alternative for traces, metrics,
  and structured logs. If your platform standardises
  on OTel rather than raw Prometheus, this is where
  chapter 02's surface lands.
- **OpenTelemetry — Semantic Conventions.**
  <https://opentelemetry.io/docs/specs/semconv/>
  The industry-standard names for service, HTTP, and
  resource attributes. If your logs and metrics carry
  these names, downstream tooling parses them for free.
- **Grafana — official documentation.**
  <https://grafana.com/docs/grafana/latest/>
  The dashboard surface chapters 02, 04, and 07 all
  assume. The "Panels and visualizations" and
  "Alerting" sections are the ones the module leans on.

## Drift-detection libraries — the four the module cites

Chapter 03's shortlist. All are actively maintained and
open-source; pick one for the first drift job, then roll
your own only where the library falls short.

- **Evidently AI — official documentation.**
  <https://docs.evidentlyai.com/>
  Batteries-included Python library that computes a
  catalogue of univariate and multivariate drift
  tests, renders HTML/JSON reports, and integrates
  with the common orchestrators. The "Metrics" and
  "Presets" pages are the shortest path to a first
  drift report.
- **whylogs (WhyLabs) — official documentation.**
  <https://whylogs.readthedocs.io/>
  Profile-based data logging: you emit a per-window
  profile, and drift is measured *between* profiles.
  Handles very high-cardinality features well and
  scales to streaming.
- **NannyML — official documentation.**
  <https://nannyml.readthedocs.io/>
  Focused on **performance estimation without
  labels** (CBPE for classification, DLE for
  regression) and univariate/multivariate drift.
  Chapter 04's label-lag section is exactly where
  NannyML earns its keep.
- **Alibi Detect (Seldon) — official documentation.**
  <https://docs.seldon.io/projects/alibi-detect/>
  Broader drift + outlier + adversarial detection
  toolkit with kernel-based methods (MMD),
  classifier two-sample tests, and drift on
  embeddings. Best fit when the features are
  embeddings or the data is high-dimensional.

## Chapter 01 — Why ML monitoring is its own discipline

- **Sculley, D., et al. "Hidden Technical Debt in
  Machine Learning Systems" (NeurIPS 2015).**
  <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  The paper behind chapter 01's premise that the
  model is the small part of an ML system. The
  "Data Dependencies Cost More than Code
  Dependencies" and "Monitoring and Testing"
  sections are directly relevant.
- **Breck, E., et al. "The ML Test Score: A Rubric
  for ML Production Readiness and Technical Debt
  Reduction" (IEEE Big Data 2017).**
  <https://research.google/pubs/pub46555/>
  Google's 28-item production-readiness checklist.
  Section IV ("Tests for Monitoring") maps almost
  one-to-one onto this module's chapter list.
- **Zinkevich, Martin. "Rules of Machine Learning:
  Best Practices for ML Engineering" (Google
  Developers).**
  <https://developers.google.com/machine-learning/guides/rules-of-ml>
  Rules 8, 14, 27, 28, and 32 are the ones this
  module keeps rediscovering. Free HTML; ~30 min
  read.
- **Google SRE Book — "Monitoring Distributed
  Systems".**
  <https://sre.google/sre-book/monitoring-distributed-systems/>
  The vocabulary chapter 01 uses when it separates
  "the service is broken" from "the model is
  wrong". Free HTML.
- **Chip Huyen — *Designing Machine Learning
  Systems* (O'Reilly, 2022).**
  <https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/>
  Chapter 8 ("Data Distribution Shifts and
  Monitoring") is the module's closest full-length
  companion.

## Chapter 02 — Instrumenting ML-specific signals

- **Prometheus — Metric types.**
  <https://prometheus.io/docs/concepts/metric_types/>
  The reference for `Counter`, `Gauge`, `Histogram`,
  and `Summary`. Chapter 02's six-metric surface
  picks from these.
- **Prometheus — Best practices: Metric and label
  naming.**
  <https://prometheus.io/docs/practices/naming/>
  The `ml_*` prefixes, the `_total` / `_seconds`
  suffixes, and the "no `user_id` labels" advice all
  trace back to this page.
- **Prometheus — Instrumentation best practices.**
  <https://prometheus.io/docs/practices/instrumentation/>
  "The three types" and "How much should I
  instrument?" are the shortest path to chapter 02's
  cardinality discipline.
- **Prometheus — Histogram and summary.**
  <https://prometheus.io/docs/practices/histograms/>
  Why the bucket choice cannot be changed after
  emission, and why you pick buckets from the
  reference distribution instead of guessing.
- **python-json-logger — README.**
  <https://github.com/madzak/python-json-logger>
  The JSON logger `mod-108` chapter 05 introduces
  and this module extends. `pip install
  python-json-logger`.
- **structlog — official documentation.**
  <https://www.structlog.org/en/stable/>
  The alternative logger with first-class context
  binding — useful when the prediction event needs
  many correlated fields.
- **MLflow — Model Signature and metadata.**
  <https://mlflow.org/docs/latest/model/signatures.html>
  The reference for the schema half of chapter 02's
  reference profile. Compatible with the registry
  work in `mod-106`.

## Chapter 03 — Data drift detection

- **SciPy — `scipy.stats.ks_2samp`.**
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html>
  The two-sample Kolmogorov–Smirnov test chapter 03
  calls directly. Read the "Notes" section on why
  the p-value collapses at production sample sizes.
- **SciPy — `scipy.stats.chi2_contingency`.**
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>
  The chi-squared test for categoricals.
- **SciPy — `scipy.stats.wasserstein_distance`.**
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html>
  The Wasserstein-1 (Earth Mover's) distance
  chapter 03 recommends when unit-preserving drift
  matters.
- **SciPy — `scipy.spatial.distance.jensenshannon`.**
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html>
  JSD for probability-mass comparisons. Note that
  SciPy returns the square root of the divergence;
  square it for the JSD scalar.
- **Karakoulas, Grigoris. "Empirical Validation of
  Retail Credit-Scoring Models" (RMA Journal, 2004).**
  <!-- needs-research: this paper is often cited as
  the origin of the PSI cutoffs (< 0.1, 0.1–0.2, ≥
  0.2) used across credit scoring; the actual
  citation is behind the RMA paywall and difficult
  to verify. Confirm the reference (author, year,
  journal issue) and swap in a stable URL, or drop
  the specific attribution in favour of "widely
  used industry heuristics." -->
- **Lopez-Paz, David, and Maxime Oquab. "Revisiting
  Classifier Two-Sample Tests" (ICLR 2017).**
  <https://arxiv.org/abs/1610.06545>
  The paper behind chapter 03's domain-classifier
  drift detector. Short, readable, and the feature-
  importance interpretation trick is exactly what
  the exercise-02 attribution asks for.
- **Gretton, Arthur, et al. "A Kernel Two-Sample
  Test" (JMLR, 2012).**
  <https://jmlr.org/papers/v13/gretton12a.html>
  The MMD paper. Long but foundational; skim
  sections 1–2 for the intuition before reaching
  for Alibi Detect's `MMDDrift`.
- **Rabanser, Stephan, Stephan Günnemann, and
  Zachary C. Lipton. "Failing Loudly: An Empirical
  Study of Methods for Detecting Dataset Shift"
  (NeurIPS 2019).**
  <https://arxiv.org/abs/1810.11953>
  A benchmark of univariate and multivariate drift
  detectors across shift types. Read section 5 for
  the "when does each detector actually catch the
  shift?" table.
- **Great Expectations — official documentation.**
  <https://docs.greatexpectations.io/>
  The data-quality alternative chapter 03 explicitly
  separates from drift. Owns schema and value-range
  validation; complementary to (not a replacement
  for) statistical drift.
- **TensorFlow Data Validation — user guide.**
  <https://www.tensorflow.org/tfx/data_validation/get_started>
  The TFX-native alternative for schema and
  distribution validation. Same story as Great
  Expectations for the drift discussion.
- **Kullback, S., and R. A. Leibler. "On Information
  and Sufficiency" (Annals of Mathematical
  Statistics, 1951).**
  <https://doi.org/10.1214/aoms/1177729694>
  The original KL divergence paper. JSD is the
  bounded, symmetric variant chapter 03 prefers;
  understanding KL first makes JSD's edges
  intelligible.

## Chapter 04 — Concept drift and prediction-distribution shift

- **Gama, João, et al. "A Survey on Concept Drift
  Adaptation" (ACM Computing Surveys, 2014).**
  <https://dl.acm.org/doi/10.1145/2523813>
  The definitive survey. Section 3 (types of
  drift), section 4 (detection methods), and
  section 5 (adaptation strategies) are the ones
  chapter 04 quietly leans on. Cite this before any
  concept-drift-algorithm name.
- **Bifet, Albert, and Ricard Gavaldà. "Learning
  from Time-Changing Data with Adaptive Windowing"
  (SDM 2007).**
  <https://doi.org/10.1137/1.9781611972771.42>
  The ADWIN algorithm paper. Foundational for
  online change-detection on the accuracy time
  series.
- **Gama, João, et al. "Learning with Drift
  Detection" (SBIA 2004).**
  <https://doi.org/10.1007/978-3-540-28645-5_29>
  The DDM (Drift Detection Method) paper the online-
  learning tools reference.
- **Page, E. S. "Continuous Inspection Schemes"
  (Biometrika, 1954).**
  <https://doi.org/10.1093/biomet/41.1-2.100>
  The original CUSUM paper. Chapter 04's change-
  detection section leans on this whenever the
  metric is a time series.
- **River — official documentation.**
  <https://riverml.xyz/>
  The maintained descendant of `scikit-multiflow`.
  Ships production implementations of ADWIN, DDM,
  EDDM, KSWIN, HDDM_A/W, and Page–Hinkley. The
  `river.drift` module is the shortest path to the
  online-learning algorithms chapter 04 names.
- **NannyML — Performance Estimation for
  Classification (CBPE).**
  <https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/binary_performance_estimation/standard_metric_estimation.html>
  The reference implementation of the CBPE method
  chapter 04 lists. The "How it works" page has the
  math.
- **NannyML — Direct Loss Estimation (DLE).**
  <https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/regression_performance_estimation.html>
  The regression counterpart. Read alongside CBPE
  for the label-free performance-estimation story.
- **Ren, Jie, et al. "A Simple Fix to Mahalanobis
  Distance for Improving Near-OOD Detection"
  (2021).**
  <https://arxiv.org/abs/2106.09022>
  Adjacent to concept-drift monitoring: the
  Mahalanobis-distance trick is one of the cheaper
  ways to flag "this input looks unlike anything
  the model saw during training". Useful when the
  model returns confidence scores.

## Chapter 05 — Ground-truth collection loops

- **Ekaterina Feklistov, & the Google Blog. "Data
  Cascades in High-Stakes AI" (CHI 2021).**
  <!-- needs-research: the actual paper is Sambasivan,
  Nithya, et al., "'Everyone wants to do the model
  work, not the data work': Data Cascades in
  High-Stakes AI" (CHI 2021). Confirm the author
  ordering and swap the citation to the ACM DL URL
  <https://dl.acm.org/doi/10.1145/3411764.3445518>. -->
  Chapter 05's "labels are hard and delayed and
  often biased" argument condenses this paper.
  Required reading.
- **Northcutt, Curtis G., Anish Athalye, and
  Jonas Mueller. "Pervasive Label Errors in Test
  Sets Destabilize Machine Learning Benchmarks"
  (NeurIPS 2021).**
  <https://arxiv.org/abs/2103.14749>
  Empirical measurement of label noise across the
  ten most-used ML benchmarks. Sobering evidence
  that "the labels are ground truth" is a
  simplification.
- **Bottou, Léon, et al. "Counterfactual Reasoning
  and Learning Systems" (JMLR, 2013).**
  <https://jmlr.org/papers/v14/bottou13a.html>
  The paper behind chapter 05's warning about
  logged-bandit / exposure bias in implicit
  feedback. Section 4 (Logged Bandit Feedback) is
  the target read.
- **Little, Roderick J. A., and Donald B. Rubin.
  *Statistical Analysis with Missing Data* (3rd ed.,
  Wiley, 2019).**
  <https://www.wiley.com/en-us/Statistical+Analysis+with+Missing+Data%2C+3rd+Edition-p-9780470526798>
  Foundational reference for the missing-negative
  problem and inverse-propensity weighting.
- **Amazon SageMaker — Ground Truth (documentation).**
  <https://docs.aws.amazon.com/sagemaker/latest/dg/sms.html>
  A vendor human-labelling service, useful as a
  concrete reference for chapter 05's "human
  labelling as a pipeline" pattern (queue, gold
  tasks, adjudication).
- **Scale AI — documentation.**
  <https://scale.com/docs>
  The other widely used human-labelling vendor.
  Same shape; different SLA and pricing model.
- **Label Studio — official documentation.**
  <https://labelstud.io/guide/>
  The open-source alternative for in-house
  labelling. `pip install label-studio`.
- **Snorkel — programmatic labelling (docs).**
  <https://www.snorkel.org/>
  <https://snorkel.readthedocs.io/en/latest/>
  When human labels are unaffordable, weak
  supervision fills the gap. Chapter 05 does not
  cover it, but the labels-table shape it produces
  is the same.
- **Settles, Burr. "Active Learning Literature
  Survey" (University of Wisconsin–Madison, 2010).**
  <https://minds.wisconsin.edu/handle/1793/60660>
  The reference for the active-learning strategies
  chapter 05 gestures at (uncertainty sampling,
  query-by-committee, expected-model-change).

## Chapter 06 — Retraining triggers

- **Google Cloud — "MLOps: Continuous delivery and
  automation pipelines in machine learning".**
  <https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
  The paper-length white paper defining CT
  (continuous training) alongside CI/CD. Chapter
  06's trigger flavours map onto its "level 1" and
  "level 2" MLOps maturity.
- **Sculley, D., et al. "Hidden Technical Debt in
  Machine Learning Systems" (NeurIPS 2015).**
  <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  Cited already for chapter 01. The
  "Feedback Loops" section is the reason chapter 06
  makes the evaluation gate non-optional.
- **Andrew Ng, et al. "A Chat with Andrew on MLOps:
  From Model-centric to Data-centric AI" (DeepLearning.AI,
  2021).**
  <https://www.deeplearning.ai/the-batch/issue-84/>
  The framing behind chapter 06's "the retrain data
  matters at least as much as the trigger" rule.
- **Netflix Tech Blog — Machine Learning Platform.**
  <https://netflixtechblog.com/tagged/machine-learning>
  Case studies of production retraining loops at
  scale. Not a specification; useful for calibrating
  what "mature MLOps" looks like in a large org.
- **Uber Engineering — Michelangelo posts.**
  <https://www.uber.com/blog/michelangelo-machine-learning-platform/>
  The originating post on Uber's ML platform.
  Chapter 06's "trigger writes an event that a
  training pipeline consumes" pattern is roughly
  Michelangelo's shape.
- **Google — TFX Guide.**
  <https://www.tensorflow.org/tfx/guide>
  Chapter 06's evaluation-gate machinery
  (`Evaluator`, `Pusher`, `ModelValidator`) has a
  concrete implementation here. Even if you do not
  use TFX, its component naming is the reference
  vocabulary.

## Chapter 07 — Debugging a production regression

- **Google SRE Workbook — "Incident Response".**
  <https://sre.google/workbook/incident-response/>
  The IMAG (Incident Management at Google) protocol
  underlying chapter 07's incident structure. Free
  HTML.
- **Google SRE Book — "Postmortem Culture:
  Learning from Failure".**
  <https://sre.google/sre-book/postmortem-culture/>
  The template chapter 07's post-mortem section
  extends. The blameless framing is table-stakes.
- **PagerDuty — Incident Response Documentation.**
  <https://response.pagerduty.com/>
  An open, opinionated incident-response playbook
  covering severities, roles, and communication
  cadence. Complements the SRE workbook.
- **Beyer, Betsy, et al. *The Site Reliability
  Workbook* (O'Reilly, 2018).**
  <https://sre.google/workbook/table-of-contents/>
  The full workbook, free HTML. Chapter 9 (Incident
  Response) and 10 (Postmortem Culture: Learning
  from Failure) are the ones this chapter mirrors
  for ML incidents.
- **Allspaw, John. "Blameless PostMortems and a Just
  Culture" (Etsy Code as Craft, 2012).**
  <https://www.etsy.com/codeascraft/blameless-postmortems/>
  The short essay that popularised blameless
  post-mortems as a cultural norm.
- **Rein, Andreas, et al. "Underspecification
  Presents Challenges for Credibility in Modern
  Machine Learning" (JMLR, 2022).**
  <https://jmlr.org/papers/v23/20-1335.html>
  D'Amour et al.'s paper on underspecification.
  Explains why two models with equivalent training-
  time performance can behave very differently under
  distribution shift — the case chapter 07 walks a
  regression through.
  <!-- needs-research: verify the author list here;
  the paper is often cited as "D'Amour et al." with
  ~40 authors and I want to make sure the primary
  attribution is not mis-stated. -->

## Adjacent standards and specifications

- **Prometheus — Text-based exposition format.**
  <https://prometheus.io/docs/instrumenting/exposition_formats/>
  The format `/metrics` serves. Useful when writing
  the exercise-01 metric-shape tests.
- **OpenMetrics — specification.**
  <https://openmetrics.io/>
  The IETF-track spec that Prometheus's format
  standardises to. Read once to understand
  `_total` conventions and label semantics.
- **RFC 5424 — The Syslog Protocol.**
  <https://www.rfc-editor.org/rfc/rfc5424>
  The severity levels and structured-data field
  chapter 02's prediction-log records inherit.
- **RFC 7807 — Problem Details for HTTP APIs.**
  <https://www.rfc-editor.org/rfc/rfc7807>
  Referenced in `mod-108` chapter 05; relevant here
  because the mitigation guardrail (chapter 07) is
  usually a fallback response with a warning header
  in this format.
- **W3C Trace Context.**
  <https://www.w3.org/TR/trace-context/>
  The `traceparent` header that links a prediction
  log line to a request trace across services.

## Datasets used in the exercises

The exercises deliberately do not prescribe a dataset;
they reuse whatever is in play from `mod-104` /
`mod-108`. Two options work well for the "inject
synthetic drift" step:

- **UCI Adult (Census Income).**
  <https://archive.ics.uci.edu/dataset/2/adult>
  Loadable via
  `sklearn.datasets.fetch_openml("adult", version=2)`.
  Small, tabular, mixed dtypes — good for
  exercising univariate PSI on both categorical
  (`education`, `workclass`) and continuous
  (`age`, `capital_gain`) features.
- **NYC Taxi trip data.**
  <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
  Monthly Parquet snapshots. Real, month-to-month
  distribution shifts (fare structure changes,
  Covid impact, seasonality) provide a natural
  drift signal without synthesis.
- **Give Me Some Credit (Kaggle).**
  <https://www.kaggle.com/c/GiveMeSomeCredit>
  The classic credit-scoring dataset the industry
  uses for PSI examples. Suitable when exercise-02
  wants a domain-realistic PSI walkthrough.

<!-- needs-research: on each autonomous research
cycle, verify these URLs still resolve. In
particular: (a) Evidently / whylogs / NannyML /
Alibi Detect docs (all four ship on aggressive
release cadences and page structures shift);
(b) SciPy documentation (URLs stable across minor
versions but occasionally reorganised);
(c) Google Cloud MLOps whitepaper URL (Google
docs URLs have redirected in the past);
(d) Prometheus docs and OpenTelemetry docs (both
occasionally shuffle when they cut major
versions); (e) River docs (successor to
scikit-multiflow, still stabilising); (f) SRE
book / workbook (`sre.google` has redirected
before); (g) NYC TLC page (URL has moved
across nyc.gov reorganisations). Swap in current
pages when a doc URL 404s. -->
