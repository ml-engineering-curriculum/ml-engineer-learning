# Resources for mod-110 — ML Systems Design

A curated short list. The goal is depth-on-demand for the
chapter you got stuck on, not a comprehensive bibliography.
Primary documentation and canonical papers are listed
first per section; secondary reading follows.

Every prior module has its own `resources.md`; this file
does *not* re-list the tools already cited there (Prometheus,
Grafana, MLflow, scikit-learn, FastAPI, Evidently,
whylogs, etc.). Pull those files up alongside this one
when you need the plumbing detail; use this list for the
*systems-design* practice on top of them.

## Cross-cutting: the long-form companions

Two references you can read cover-to-cover in a weekend
and that reappear in almost every chapter of this module.
Bookmark once.

- **Huyen, Chip. *Designing Machine Learning Systems*
  (O'Reilly, 2022).**
  <https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/>
  The full-length companion to this module. Chapter 2
  (business framing) pairs with our chapter 01; chapter 7
  (model deployment and prediction service) pairs with
  our chapter 02; chapter 8 (data distribution shifts /
  monitoring) pairs with our chapter 03 and with
  `mod-109`; chapter 9 (continual learning) pairs with
  our chapter 04. If you read one book alongside this
  track, read this one.
- **Ameisen, Emmanuel. *Building Machine Learning
  Powered Applications* (O'Reilly, 2020).**
  <https://www.oreilly.com/library/view/building-machine-learning/9781492045106/>
  The other end-to-end reference. Lighter on the
  systems-scale specifics; stronger on the
  business-goal-to-first-model workflow that our
  chapter 01 walks.
- **Zinkevich, Martin. "Rules of Machine Learning:
  Best Practices for ML Engineering" (Google Developers).**
  <https://developers.google.com/machine-learning/guides/rules-of-ml>
  Cited already in `mod-109` — worth re-reading through
  this module's lens. Rules 1-3 (start without ML) map
  to chapter 01's baseline discussion; Rules 4-15 (the
  first pipeline) map to chapter 02; Rules 29-31
  (serving-training skew) map to chapter 03; Rules
  32-39 (slowed growth and the training-serving loop)
  map to chapters 04 and 05. Free HTML, ~30 min.
- **Sculley, D., et al. "Hidden Technical Debt in
  Machine Learning Systems" (NeurIPS 2015).**
  <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  The paper the whole module quietly rests on. The
  "Data Dependencies," "Feedback Loops," and
  "Configuration Debt" sections are the ones you will
  keep returning to.
- **Breck, E., et al. "The ML Test Score: A Rubric for
  ML Production Readiness and Technical Debt Reduction"
  (IEEE Big Data 2017).**
  <https://research.google/pubs/pub46555/>
  Google's 28-item production-readiness rubric. Read
  once, then read chapter 07 — our checklist is a
  close relative shaped for the design-review
  conversation rather than the pre-launch audit.

## Chapter 01 — From business goal to ML system blueprint

The chapter's frame ("business decision → ML formulation
→ system shape → baseline and phasing") is not novel.
The best short-form references:

- **Zinkevich, Rules 1-3.**
  <https://developers.google.com/machine-learning/guides/rules-of-ml>
  "Don't be afraid to launch a product without machine
  learning." The pillar of chapter 01 Section 5.
- **Huyen, *DMLS*, Chapter 2 ("Introduction to Machine
  Learning Systems Design").**
  The framing of business objectives vs ML objectives
  vs mismatch cost. Direct companion.
- **Amershi, Saleema, et al. "Software Engineering for
  Machine Learning: A Case Study" (ICSE 2019).**
  <https://www.microsoft.com/en-us/research/publication/software-engineering-for-machine-learning-a-case-study/>
  Microsoft's nine-stage ML workflow diagram. Chapter
  01's blueprint sits on top of this shape.
- **Ng, Andrew. "Machine Learning Yearning" (self-
  published, 2018).**
  <https://www.deeplearning.ai/machine-learning-yearning/>
  Free PDF. Especially Chapters 5-12 (single-number
  evaluation metric, dev/test split, and how to pick
  a first target) — the muscle memory chapter 01's
  Section 2 assumes.
- **Kim, Eugene, and Julie Chapman. "The Rise of the
  Machine Learning Engineer" (Chip Huyen's blog,
  various years).**
  <https://huyenchip.com/2020/12/27/real-world-mlops-questions-answered.html>
  Practitioner writing that walks the same
  business-first framing chapter 01 uses; useful for
  variety of voice.

## Chapter 02 — Reference architecture of an ML system

The nine-component reference architecture in the chapter
lines up with several published references. Read any two
in cross-section:

- **Google Cloud — "MLOps: Continuous delivery and
  automation pipelines in machine learning."**
  <https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
  Paper-length whitepaper. Its "level 0 / level 1 /
  level 2" architecture diagrams are the closest
  vendor-published match to our chapter 02's diagram.
  Free HTML.
- **Google — TFX Guide.**
  <https://www.tensorflow.org/tfx/guide>
  Reference vocabulary for the training-pipeline half
  of the diagram. Even if you never use TFX, the
  `ExampleGen → StatisticsGen → SchemaGen → Transform →
  Trainer → Evaluator → Pusher` component graph is a
  shared shorthand.
- **Uber Engineering — "Meet Michelangelo: Uber's
  Machine Learning Platform."**
  <https://www.uber.com/blog/michelangelo-machine-learning-platform/>
  The originating public post on an internal ML
  platform. Our diagram is roughly Michelangelo minus
  Uber's naming.
- **Airbnb Engineering — "Bighead: Airbnb's End-to-End
  Machine Learning Platform."**
  <https://medium.com/airbnb-engineering/bighead-airbnbs-end-to-end-machine-learning-platform-51e1e7e006a9>
  Same architecture, different vocabulary. Cross-read
  with Michelangelo and it becomes obvious that the
  chapter's nine components are what falls out of the
  problem regardless of the org.
- **Feast — official documentation.**
  <https://docs.feast.dev/>
  The OSS feature store the chapter names. The
  "Concepts → Feature views" and "Concepts → Point-in-
  time correctness" pages are what chapter 02's
  section-3 discussion assumes.
- **Tecton — Feature Store 101.**
  <https://www.tecton.ai/blog/what-is-a-feature-store/>
  Vendor-published but reasonable overview. Useful
  companion to Feast's docs for the "why do we have a
  feature store at all" argument.
- **Chip Huyen. "Real-time machine learning: challenges
  and solutions" (2022).**
  <https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html>
  Long-form essay on the online/streaming/batch
  distinction our chapter 02 leans on. Especially
  useful for the "prefer batch until proven otherwise"
  argument.
- **Data Council / Databricks — Feature Store papers
  and case studies.**
  <https://www.databricks.com/glossary/feature-store>
  <!-- needs-research: several of the widely cited
  DDIA-style ML platform talks (from Netflix, LinkedIn,
  Instacart) don't have single canonical URLs and are
  distributed as conference video only. Verify one or
  two are worth adding here as case-study references
  and swap in stable URLs. -->

## Chapter 03 — Training-serving skew

The best sources on skew are Google's — because Google
was the first team large enough to publish about it.

- **Zinkevich, Rules 29-31 ("Serving-Training Skew").**
  <https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew>
  The 3-rule mini-chapter that chapter 03 quietly
  expands.
- **Baylor, Denis, et al. "TFX: A TensorFlow-Based
  Production-Scale Machine Learning Platform" (KDD
  2017).**
  <https://research.google/pubs/pub46484/>
  Section 5 ("Data Analysis, Transformation, and
  Validation") is the origin of the "shared transform
  library" pattern chapter 03 recommends as the fix
  for cause #1.
- **TensorFlow Extended — "TFT: TensorFlow Transform".**
  <https://www.tensorflow.org/tfx/guide/tft>
  The concrete "transform is code that runs in both
  places" solution. Reads even if you're on scikit-
  learn — the pattern generalises.
- **Polyzotis, Neoklis, et al. "Data Management
  Challenges in Production Machine Learning" (SIGMOD
  2017).**
  <https://research.google/pubs/pub46178/>
  Broader-scope paper that names training-serving skew,
  data validation, and feature-store correctness as
  distinct disciplines. Chapter 03 splits skew from
  what `mod-103` covers in data validation; this paper
  is where the split is articulated.
- **Feast — "Point-in-time correctness".**
  <https://docs.feast.dev/getting-started/concepts/point-in-time-joins>
  The concrete implementation of the pattern chapter 03
  Section 2 argues for. Skim once even if you're not
  using Feast; the semantics are the point.
- **scikit-learn — Pipeline reference.**
  <https://scikit-learn.org/stable/modules/compose.html>
  The `Pipeline` object exercise-02 asks you to
  serialise as a single artifact. The "Combining
  estimators" page has the persistence discussion.
- **joblib — Persistence documentation.**
  <https://joblib.readthedocs.io/en/latest/persistence.html>
  The artifact format the exercise uses. Read the
  "Warnings" section (Python version pinning, class
  paths) — real pre-production skew often shows up as
  a deserialisation issue rather than a compute
  divergence.
- **Kubeflow — "Model Serving with KFServing / KServe".**
  <https://kserve.github.io/website/>
  When the serving path is a separate service (not the
  same process as training), the KServe transformer
  pattern is one concrete way to keep the transform
  colocated with the model artifact.

## Chapter 04 — Feedback loops and data flywheels

The feedback-loop literature is scattered across
counterfactual inference, contextual bandits, and
recommender-fairness.

- **Bottou, Léon, et al. "Counterfactual Reasoning and
  Learning Systems" (JMLR 2013).**
  <https://jmlr.org/papers/v14/bottou13a.html>
  The reference paper on why "the data your model
  produces is not the counterfactual." Section 4
  (Logged Bandit Feedback) is chapter 04 Section 1's
  underpinning; section 5 (real ad system) is a case
  study of the exploration-budget argument in
  practice.
- **Chaney, Allison, Brandon Stewart, and Barbara
  Engelhardt. "How Algorithmic Confounding in
  Recommendation Systems Increases Homogeneity and
  Decreases Utility" (RecSys 2018).**
  <https://dl.acm.org/doi/10.1145/3240323.3240370>
  The specific paper behind chapter 04's "the loop
  narrows the catalogue" claim. Small, simulation-
  based, clear.
- **Ensign, Danielle, et al. "Runaway Feedback Loops
  in Predictive Policing" (FAT* 2018).**
  <https://arxiv.org/abs/1706.09847>
  The formal treatment of the degenerate predictive-
  policing loop chapter 04 uses as its canonical bad
  example. Read once for the model; the intuition
  transfers to any high-stakes prediction that
  determines its own labels.
- **Pariser, Eli. *The Filter Bubble: What the
  Internet Is Hiding from You* (Penguin, 2011).**
  <https://en.wikipedia.org/wiki/The_Filter_Bubble>
  The origin of the term. Not a technical text; the
  vocabulary and one worked example is what chapter 04
  Section 2's "filter bubble" callout leans on.
- **Li, Lihong, et al. "A Contextual-Bandit Approach
  to Personalized News Article Recommendation" (WWW
  2010).**
  <https://arxiv.org/abs/1003.0146>
  The paper behind chapter 04's exploration-budget
  recommendation. LinUCB is one specific instantiation;
  the framing applies to any contextual-bandit setup.
- **Chapelle, Olivier, and Lihong Li. "An Empirical
  Evaluation of Thompson Sampling" (NeurIPS 2011).**
  <https://papers.nips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html>
  Thompson sampling as the practical exploration
  algorithm chapter 04 cites; short and readable.
- **Joachims, Thorsten, Adith Swaminathan, and Tobias
  Schnabel. "Unbiased Learning-to-Rank with Biased
  Feedback" (WSDM 2017).**
  <https://arxiv.org/abs/1608.04468>
  The reference paper for position bias and inverse-
  propensity weighting in ranking — chapter 04
  Section 3's "position and selection debiasing"
  mitigation.
- **Netflix Tech Blog — "Learning a Personalized
  Homepage."**
  <https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a>
  Case study of an exploration-aware production
  recommender. Practical companion to the paper trail
  above.

## Chapter 05 — A/B testing ML systems

A/B testing on the web has its own literature,
substantial and well-organised. Read at least the
Microsoft group's book once; the papers are the
follow-up.

- **Kohavi, Ron, Diane Tang, and Ya Xu. *Trustworthy
  Online Controlled Experiments: A Practical Guide to
  A/B Testing* (Cambridge, 2020).**
  <https://experimentguide.com/>
  The standard reference. Chapters 3 (statistical
  concepts), 17 (SRM), 18 (variance reduction /
  CUPED), and 22 (novelty effects) are the ones
  chapter 05 aligns with. If you read one book for the
  A/B half of this module, read this one.
- **Kohavi, Ron, et al. "Online controlled
  experiments at large scale" (KDD 2013).**
  <https://ai.stanford.edu/~ronnyk/2013 controlledExperimentsAtScale.pdf>
  <!-- needs-research: this PDF URL has moved before.
  Prefer the KDD 2013 canonical URL
  <https://dl.acm.org/doi/10.1145/2487575.2488217>
  if the Stanford mirror stops resolving. -->
  Bing's early public writeup of the scale required to
  detect small effects. Chapter 05's power-analysis
  section quietly cites this.
- **Deng, Alex, et al. "Improving the Sensitivity of
  Online Controlled Experiments by Utilizing Pre-
  Experiment Data" (WSDM 2013).**
  <https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf>
  <!-- needs-research: same URL sensitivity as the
  KDD 2013 paper above; verify the primary link
  resolves and fall back to the ACM DL landing page
  if it doesn't. -->
  The CUPED paper. Chapter 05 Section 4 recommends
  CUPED; this is the mechanism.
- **Fabijan, Aleksander, et al. "Diagnosing Sample
  Ratio Mismatch in Online Controlled Experiments"
  (KDD 2019).**
  <https://dl.acm.org/doi/10.1145/3292500.3330722>
  The reference paper on SRM diagnosis. Chapter 05
  Section 6's SRM subsection compresses this.
- **Hohnhold, Henning, Deirdre O'Brien, and Diane
  Tang. "Focusing on the Long-term: It's Good for
  Users and Business" (KDD 2015).**
  <https://research.google/pubs/pub43887/>
  Google's paper on the short-term vs long-term
  metric mismatch — closely related to chapter 05's
  novelty-effect discussion.
- **Xie, Huizhi, and Juliette Aurisset. "Improving
  the Sensitivity of Online Controlled Experiments:
  Case Studies at Netflix" (KDD 2016).**
  <https://www.researchgate.net/publication/307014104>
  Netflix's public write-up of variance-reduction
  techniques in practice. Useful counterpart to the
  Microsoft/Bing perspective.
- **Bakshy, Eytan, Dean Eckles, and Michael S.
  Bernstein. "Designing and Deploying Online Field
  Experiments" (WWW 2014).**
  <https://arxiv.org/abs/1409.3174>
  The paper introducing Facebook's PlanOut. Chapter
  05's ramp-plan section leans on the ideas here even
  when the tooling is not PlanOut.
- **statsmodels — `power` module.**
  <https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations>
  The library exercise-03's power-analysis script
  builds on. The `NormalIndPower`,
  `TTestIndPower`, and proportions-based classes are
  the ones you will use most.
- **SciPy — `scipy.stats.ttest_ind`,
  `scipy.stats.proportions_ztest`.**
  <https://docs.scipy.org/doc/scipy/reference/stats.html>
  The primitive hypothesis-test functions. Read the
  "Notes" sections carefully — the "welch's t-test
  vs student's" choice matters more than it looks.
- **Bojinov, Iavor, and Neil Shephard. "Time Series
  Experiments and Causal Estimands: Exact
  Randomization Tests and Trading" (JASA 2019).**
  <https://arxiv.org/abs/1706.07840>
  The reference for the switchback-design mention in
  chapter 05 Section 1. Advanced.
- **Howard, Steven R., et al. "Time-uniform,
  nonparametric, nonasymptotic confidence sequences"
  (Annals of Statistics 2021).**
  <https://arxiv.org/abs/1810.08240>
  The always-valid confidence-sequence paper cited in
  chapter 05's stretch discussion. The math is heavy;
  the "why do we need this at all" motivation is on
  the first two pages.

## Chapter 06 — Build vs buy, and delegating to peer specialist tracks

The build-vs-buy topic has more industry blogging than
formal literature. Read a few practitioner writeups and
one canonical reference on total cost of ownership.

- **Christensen, Clayton M. *The Innovator's Solution*
  (Harvard Business Review Press, 2003).**
  <https://hbr.org/product/the-innovators-solution-creating-and-sustaining-successful-growth/1953-HBK-ENG>
  The "modularity vs integration" chapters (2-3) are
  the strategic frame chapter 06 Section 1 uses when
  it distinguishes "build the competitive edge, buy
  the plumbing." Business-strategy source, not ML,
  but the framing is directly applicable.
- **Amazon Web Services — "Build vs Buy: A Framework
  for Cloud Decisions" (AWS Executive Insights).**
  <https://aws.amazon.com/executive-insights/content/build-vs-buy/>
  Vendor-published but reasonable primer on the TCO
  comparison chapter 06 Section 1 walks. Useful when
  you need the vocabulary in a design review with
  non-ML stakeholders.
- **Chip Huyen. "MLOps Tooling Landscape" (2021).**
  <https://huyenchip.com/2021/11/07/mlops-tooling-landscape.html>
  Long-form landscape review that names the categories
  of vendor chapter 06 Section 1's table lists (feature
  stores, trackers, observability). Best cross-check
  when the vendor list has moved.
- **The MAD (ML, AI, Data) Landscape — Matt Turck.**
  <https://mattturck.com/mad2024/>
  <!-- needs-research: the MAD landscape is republished
  annually and the URL changes each year (mad2024 →
  mad2025 → …). Update on each autonomous cycle to the
  current year's edition. -->
  The industry map of "who sells what." Not a
  recommendation source — but essential for
  understanding whether a category is 3-vendor thin or
  20-vendor commoditised.
- **State of AI Report — annual.**
  <https://www.stateof.ai/>
  Long-form annual report with a "tooling" section
  that covers the same landscape as the MAD map from
  a research-and-adoption angle.
- **MLflow — official documentation.**
  <https://mlflow.org/docs/latest/>
  The OSS incumbent for experiment tracking and the
  model registry. Cited in the chapter 06 decision
  table as the default "buy" for that component.
- **Weights & Biases — Reports gallery.**
  <https://wandb.ai/site/reports>
  The vendor's own writeups of what a mature
  experiment-tracking workflow looks like. Useful for
  the "what does the vendor actually give you" side of
  a build-vs-buy comparison.
- **Feast vs Tecton — comparison writeups.**
  <https://docs.feast.dev/reference/alpha-vs-batch-materialization-engine>
  <https://www.tecton.ai/blog/tecton-vs-feast/>
  Vendor-slanted (Tecton's post) and OSS-slanted
  (Feast's docs) versions of the same trade-off.
  Reading both is the honest way to calibrate.
- **Arize AI — documentation.**
  <https://docs.arize.com/arize>
- **WhyLabs — documentation.**
  <https://docs.whylabs.ai/docs/>
- **Fiddler — documentation.**
  <https://docs.fiddler.ai/>
  The three widely used ML-observability vendors named
  in chapter 06's "where buy usually wins" list. Skim
  once to know the shape.
- **KServe — official documentation.**
  <https://kserve.github.io/website/>
- **BentoML — official documentation.**
  <https://docs.bentoml.com/en/latest/>
- **Modal — official documentation.**
  <https://modal.com/docs>
- **Amazon SageMaker — Developer Guide.**
  <https://docs.aws.amazon.com/sagemaker/>
- **Google Vertex AI — documentation.**
  <https://cloud.google.com/vertex-ai/docs>
- **Databricks — Machine Learning documentation.**
  <https://docs.databricks.com/en/machine-learning/index.html>
  The serving-and-platform vendors chapter 06 names.
  You are not expected to be current on all of them;
  bookmark for the specific buy-vs-buy comparison
  when the time comes.
- **Ray Serve — official documentation.**
  <https://docs.ray.io/en/latest/serve/index.html>
  OSS serving alternative useful when the "build v0"
  in chapter 06 Q7 leans on Ray for orchestration.

### Peer-track pointers (chapter 06 Section 2)

You are not expected to learn these tracks in this
module — chapter 06 is specifically about *when to
delegate to them*. Bookmark the entry points so you can
have a productive first conversation with the peer team.

- **LLM Application Engineer track — entry points.**
  - **Anthropic — Claude API documentation.**
    <https://docs.anthropic.com/>
  - **OpenAI — API reference.**
    <https://platform.openai.com/docs/>
  - **Google — Gemini API reference.**
    <https://ai.google.dev/gemini-api/docs>
- **RAG / Retrieval track — entry points.**
  - **Lewis, Patrick, et al. "Retrieval-Augmented
    Generation for Knowledge-Intensive NLP Tasks"
    (NeurIPS 2020).**
    <https://arxiv.org/abs/2005.11401>
    The originating RAG paper. Read once for the
    frame; the tooling has moved a lot since.
  - **Pinecone — Learn: Retrieval-Augmented
    Generation.**
    <https://www.pinecone.io/learn/retrieval-augmented-generation/>
    Vendor-published but the clearest short primer on
    the retrieval half of RAG.
  - **pgvector — README.**
    <https://github.com/pgvector/pgvector>
    The Postgres extension for vector search — often
    the honest "build v0" the RAG team compares
    against Pinecone / Weaviate / Milvus.
  - **LlamaIndex, LangChain — documentation.**
    <https://docs.llamaindex.ai/>
    <https://python.langchain.com/docs/>
    The two most common OSS orchestration libraries
    the RAG team will already know.
- **Fine-Tuning track — entry points.**
  - **Hugging Face — PEFT (LoRA) documentation.**
    <https://huggingface.co/docs/peft>
    The reference implementation of the
    parameter-efficient fine-tuning approach chapter
    06 mentions.
  - **Hu, Edward J., et al. "LoRA: Low-Rank
    Adaptation of Large Language Models" (ICLR 2022).**
    <https://arxiv.org/abs/2106.09685>
    The LoRA paper.
  - **Dettmers, Tim, et al. "QLoRA: Efficient
    Finetuning of Quantized LLMs" (NeurIPS 2023).**
    <https://arxiv.org/abs/2305.14314>
    The follow-up that made single-GPU fine-tuning
    accessible.
  - **OpenAI — Fine-tuning guide.**
    <https://platform.openai.com/docs/guides/fine-tuning>
  - **Anthropic — Fine-tuning documentation.**
    <https://docs.anthropic.com/>
    The hosted-API fine-tuning entry points, when the
    "buy" answer is "fine-tune the hosted model."
- **Generative Evaluation track — entry points.**
  - **Zheng, Lianmin, et al. "Judging LLM-as-a-Judge
    with MT-Bench and Chatbot Arena" (NeurIPS 2023).**
    <https://arxiv.org/abs/2306.05685>
    The reference paper for LLM-as-a-judge
    evaluation, its biases, and its calibration.
  - **Anthropic — "Building evaluations".**
    <https://docs.anthropic.com/en/docs/build-with-claude/develop-tests>
    Vendor-published primer on the eval loop for
    hosted-LLM applications.
  - **HELM — Holistic Evaluation of Language Models
    (Stanford CRFM).**
    <https://crfm.stanford.edu/helm/latest/>
    The academic reference for multi-dimensional LLM
    evaluation. Not a production tool; useful for
    understanding the scoring dimensions.
  - **Ragas — RAG evaluation framework.**
    <https://docs.ragas.io/>
    The OSS entry point for retrieval-and-generation
    evaluation metrics (context precision, context
    recall, faithfulness, answer relevancy) — the
    ones the peer eval team will use if the delegated
    problem is a RAG system.

## Chapter 07 — The ML systems design review checklist

Design-review culture as a practice is drawn from software
engineering broadly, not from ML specifically.

- **Google — "How to have a good design review" (Site
  Reliability Engineering book, chapter 32).**
  <https://sre.google/sre-book/evolving-sre-engagement-model/>
  <!-- needs-research: the exact chapter title has
  changed across editions; verify the chapter number
  and title against the free HTML version, and prefer
  a chapter-specific URL if one exists. -->
  The origin of much of the design-review vocabulary
  the module uses.
- **Google SRE Workbook — "Non-Abstract Large System
  Design."**
  <https://sre.google/workbook/non-abstract-design/>
  A more prescriptive design-review method. Chapter
  07's "audit specific claims, not free-form debate"
  framing owes a lot to this.
- **Kleppmann, Martin. *Designing Data-Intensive
  Applications* (O'Reilly, 2017).**
  <https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/>
  The systems-design reference the whole track sits
  on. Chapter 12 (The Future of Data Systems) is the
  closest analog to what chapter 07 is doing for ML.
  If DDIA is not on your bookshelf, it is a bigger
  gap than any single URL in this list.
- **Breck, E., et al. "The ML Test Score" (cited
  above).**
  Chapter 07's checklist is a design-review descendant
  of this pre-launch rubric.
- **Amershi, Saleema, et al. "Software Engineering for
  Machine Learning" (cited above).**
  The nine-stage workflow this paper describes is the
  scaffold chapter 07's seven sections cover in the
  same order.

## Adjacent standards and specifications

Pointers into other roles' vocabularies. You are not
expected to be current on these; a design review with a
Data Engineer or a Privacy Engineer will use them.

- **NIST AI Risk Management Framework 1.0.**
  <https://www.nist.gov/itl/ai-risk-management-framework>
  US federal reference for the "how do we govern ML
  systems" question. Chapter 06's delegation memo and
  chapter 07's non-goals section both benefit from
  familiarity with the NIST vocabulary in regulated
  domains.
- **EU AI Act — official text and Commission page.**
  <https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai>
  Legislative text that will shape ML system design in
  Europe from 2025 onward. Not a spec to implement
  against directly, but the categories (high-risk,
  limited-risk, prohibited) are worth knowing when a
  design review touches customer-facing decisions.
- **ISO/IEC 42001 — AI Management System standard.**
  <https://www.iso.org/standard/81230.html>
  The management-system standard for AI, released 2023.
  Analogous in shape to ISO 27001 for security.
  Useful vocabulary for the "who owns this in a
  regulated org" side of chapter 07.
- **Model Cards for Model Reporting (Mitchell, Margaret,
  et al., FAT* 2019).**
  <https://arxiv.org/abs/1810.03993>
  The originating paper on the model-card artifact
  that pairs with the design doc chapter 07 argues
  should live in the repo.
- **Datasheets for Datasets (Gebru, Timnit, et al.,
  Communications of the ACM 2021).**
  <https://arxiv.org/abs/1803.09010>
  The dataset counterpart. When chapter 01's data
  section names the training set, a datasheet is the
  document format the org expects.

## Case studies worth reading end-to-end

Not for the tooling; for the *decisions*. Read one when
the abstract feels similar to the system you're
designing.

- **Netflix — "Machine Learning at Netflix" (tag).**
  <https://netflixtechblog.com/tagged/machine-learning>
  Broad archive; the posts on recommendations,
  personalisation, and A/B testing are especially
  chapter-05-adjacent.
- **Airbnb — "Machine Learning at Airbnb" (tag).**
  <https://medium.com/airbnb-engineering/tagged/machine-learning>
  Search ranking, price recommendation, and the
  Bighead platform.
- **Uber — Engineering blog, "AI" tag.**
  <https://www.uber.com/blog/engineering/ai/>
  Michelangelo, forecasting, fraud, marketplace
  matching. Chapter 02's architecture is often what
  falls out of these posts.
- **LinkedIn Engineering — "Data" and "AI" tags.**
  <https://engineering.linkedin.com/blog>
  Feed ranking, People-You-May-Know, and the
  experimentation platform (chapter 05-adjacent).
- **Instacart — Engineering blog.**
  <https://tech.instacart.com/>
  Search, ranking, and forecasting posts with
  system-shape detail.
- **Spotify — Engineering blog.**
  <https://engineering.atspotify.com/>
  Recommendation, search, and the internal ML
  platform.
- **Meta / Facebook — Engineering blog.**
  <https://engineering.fb.com/category/ml-applications/>
  Feed ranking and the PyTorch-adjacent tooling
  posts.

<!-- needs-research: on each autonomous research cycle,
verify the URLs still resolve. Particular attention:
(a) The KDD paper mirrors on personal / lab websites —
some 404 as researchers move institutions; prefer the
ACM DL landing pages when the mirrors go stale.
(b) Vendor product docs — Databricks, Vertex, SageMaker,
Modal, Feast, Tecton, Arize, WhyLabs, Fiddler — all ship
new IA on a quarterly cadence and the deep-link URLs
shift. (c) The MAD landscape by Matt Turck is
republished annually as `mad<year>`; bump the year each
cycle. (d) The State of AI report is published each
autumn; verify the current-year URL. (e) Peer-track
entry points (LLM API docs, PEFT, LlamaIndex,
LangChain, Ragas) evolve very fast; sanity-check the
"getting started" or "concepts" pages against the top
of the docs tree each cycle. (f) Regulatory URLs (NIST
AI RMF, EU AI Act, ISO 42001) are sometimes reorganised
as versions are published; check for a newer edition
before the URL breaks. -->
