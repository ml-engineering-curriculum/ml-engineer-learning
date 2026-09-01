# Resources for mod-107 — Model Evaluation & Validation

A curated short list. The goal is depth-on-demand for the
chapter you got stuck on, not a comprehensive bibliography.
Primary documentation is listed first; papers, standards,
and adjacent tools follow.

## Cross-cutting: scikit-learn's evaluation surface

Every chapter in this module references at least one
scikit-learn API. Bookmark the top-level metrics and
model-selection references once; the rest of the module
becomes lookup.

- **scikit-learn — `sklearn.metrics` API reference.**
  <https://scikit-learn.org/stable/api/sklearn.metrics.html>
  The canonical list of every scoring function used in
  chapters 02–05, plus the `make_scorer` factory used to
  plug custom metrics into `cross_val_score` /
  `GridSearchCV`.
- **scikit-learn — Model evaluation user guide.**
  <https://scikit-learn.org/stable/modules/model_evaluation.html>
  The narrative version of the metrics API. Sections on
  classification metrics, regression metrics, and multi-
  class averaging are the chapter-02 / chapter-03
  companions.
- **scikit-learn — Cross-validation user guide.**
  <https://scikit-learn.org/stable/modules/cross_validation.html>
  The chapter-06 / chapter-08 companion. The "Cross-
  validation iterators" section enumerates every splitter
  we discuss (`KFold`, `StratifiedKFold`, `GroupKFold`,
  `StratifiedGroupKFold`, `TimeSeriesSplit`, `LeaveOneGroupOut`,
  `RepeatedKFold`).
- **scikit-learn — Common pitfalls and recommended practices.**
  <https://scikit-learn.org/stable/common_pitfalls.html>
  Sections on data leakage, inconsistent preprocessing,
  and randomness. Chapter 06's leakage taxonomy is a
  superset of the list here.
- **scikit-learn — Glossary: `random_state`.**
  <https://scikit-learn.org/stable/glossary.html#term-random_state>
  What `random_state` means on estimators, splitters, and
  cross-validators. Read once so the chapters can
  assume it.
- **scikit-learn — Dummy estimators.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html>
  The trivial-baseline primitives chapter 09 uses.

## Cross-cutting: reference textbooks

- **Hastie, Tibshirani, Friedman. *The Elements of
  Statistical Learning* (2nd ed., Springer, 2009).**
  <https://hastie.su.domains/ElemStatLearn/>
  Free PDF. Chapter 7 ("Model Assessment and Selection")
  is the classical companion to chapters 02, 06, and 09
  here — the source of the "in-sample vs out-of-sample
  error" vocabulary the module quietly uses.
- **James, Witten, Hastie, Tibshirani. *An Introduction to
  Statistical Learning* (2nd ed., Springer, 2021).**
  <https://www.statlearning.com/>
  Free PDF. Chapter 5 ("Resampling Methods") is the
  friendlier walk through cross-validation and the
  bootstrap that chapter 06 assumes.
- **Manning, Raghavan, Schütze. *Introduction to
  Information Retrieval* (Cambridge, 2008).**
  <https://nlp.stanford.edu/IR-book/>
  Free HTML. Chapter 8 ("Evaluation in information
  retrieval") is the source for the precision@k /
  recall@k / MAP / MRR / nDCG vocabulary in chapter 05.
- **Barocas, Hardt, Narayanan. *Fairness and Machine
  Learning: Limitations and Opportunities* (MIT Press,
  2023).**
  <https://fairmlbook.org/>
  Free HTML. The reference for the fairness definitions
  and the impossibility results chapter 11 cites.
  Chapters 2 and 3 (Classification, Relative Notions
  of Fairness) are the direct companions.

## Chapter 01 — Why evaluation and validation

- **scikit-learn — Model evaluation user guide.**
  <https://scikit-learn.org/stable/modules/model_evaluation.html>
  The "Overview" and "The `scoring` parameter" sections
  are the mechanical version of chapter 01's "make the
  metric explicit" argument.
- **Kohavi, Ron. "A Study of Cross-Validation and
  Bootstrap for Accuracy Estimation and Model Selection"
  (IJCAI 1995).**
  <https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf>
  The classical study behind "k-fold CV is a reasonable
  default" — the paper chapter 01 (and chapter 06)
  quietly rests on.

## Chapter 02 — Metrics for regression

- **scikit-learn — Regression metrics.**
  <https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics>
  Every function chapter 02 names
  (`mean_absolute_error`, `mean_squared_error`,
  `root_mean_squared_error`, `r2_score`,
  `mean_absolute_percentage_error`,
  `mean_pinball_loss`).
- **scikit-learn — `HuberRegressor`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html>
  The estimator behind the Huber-loss discussion in
  chapter 02.
- **Hyndman, Rob J., and Anne B. Koehler. "Another look
  at measures of forecast accuracy" (*International
  Journal of Forecasting*, 2006).**
  <https://doi.org/10.1016/j.ijforecast.2006.03.001>
  The paper behind MASE and the case against MAPE for
  intermittent-demand and near-zero data.
- **Hyndman, Rob J., and George Athanasopoulos.
  *Forecasting: Principles and Practice* (3rd ed.,
  OTexts).**
  <https://otexts.com/fpp3/accuracy.html>
  Free HTML. Section 5.8 ("Evaluating forecast accuracy")
  is the working forecaster's companion to chapter 02.

## Chapter 03 — Metrics for classification

- **scikit-learn — Classification metrics.**
  <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics>
  Every function chapter 03 names
  (`accuracy_score`, `precision_score`, `recall_score`,
  `f1_score`, `fbeta_score`, `roc_auc_score`,
  `average_precision_score`, `log_loss`,
  `classification_report`).
- **scikit-learn — `precision_recall_curve` and
  `roc_curve`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html>
  The primitives chapter 03's threshold-tuning
  discussion relies on.
- **Davis, Jesse, and Mark Goadrich. "The Relationship
  Between Precision-Recall and ROC Curves" (ICML 2006).**
  <https://doi.org/10.1145/1143844.1143874>
  The paper behind chapter 03's rule: use PR curves, not
  ROC curves, when the positive class is rare.
- **Saito, Takaya, and Marc Rehmsmeier. "The Precision-
  Recall Plot Is More Informative than the ROC Plot When
  Evaluating Binary Classifiers on Imbalanced Datasets"
  (*PLOS ONE*, 2015).**
  <https://doi.org/10.1371/journal.pone.0118432>
  Later confirmation with worked examples.
- **Fawcett, Tom. "An introduction to ROC analysis"
  (*Pattern Recognition Letters*, 2006).**
  <https://doi.org/10.1016/j.patrec.2005.10.010>
  The tutorial that established the vocabulary chapter 03
  uses for ROC space, iso-performance lines, and convex
  hulls.
- **Provost, Foster, and Tom Fawcett. "Robust
  Classification for Imprecise Environments" (*Machine
  Learning*, 2001).**
  <https://doi.org/10.1023/A:1007601015854>
  The cost-sensitive / class-imbalanced companion. The
  paper that made "the confusion matrix is not enough"
  the professional default.

## Chapter 04 — Calibration

- **scikit-learn — Probability calibration user guide.**
  <https://scikit-learn.org/stable/modules/calibration.html>
  The chapter-04 companion; documents
  `CalibratedClassifierCV`, `calibration_curve`, and the
  Platt / isotonic wrappers.
- **scikit-learn — `CalibratedClassifierCV`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html>
  The estimator wrapper chapter 04 recommends for
  post-hoc recalibration.
- **Platt, John. "Probabilistic Outputs for Support
  Vector Machines and Comparisons to Regularized
  Likelihood Methods" (1999).**
  <https://www.researchgate.net/publication/2594015>
  The sigmoid-fitting recalibrator. The "Platt scaling"
  in `CalibratedClassifierCV(method="sigmoid")`.
- **Zadrozny, Bianca, and Charles Elkan. "Transforming
  Classifier Scores into Accurate Multiclass Probability
  Estimates" (KDD 2002).**
  <https://doi.org/10.1145/775047.775151>
  The isotonic-regression recalibrator. The "isotonic"
  in `CalibratedClassifierCV(method="isotonic")`.
- **Guo, Chuan, et al. "On Calibration of Modern Neural
  Networks" (ICML 2017).**
  <https://arxiv.org/abs/1706.04599>
  The paper chapter 04 cites for "modern deep classifiers
  are systematically over-confident" and the temperature-
  scaling recipe.
- **Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos
  Hauskrecht. "Obtaining Well Calibrated Probabilities
  Using Bayesian Binning" (AAAI 2015).**
  <https://ojs.aaai.org/index.php/AAAI/article/view/9602>
  The paper behind Expected Calibration Error (ECE) and
  the binning scheme chapter 04 uses.
- **Brier, Glenn W. "Verification of Forecasts Expressed
  in Terms of Probability" (*Monthly Weather Review*,
  1950).**
  <https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2>
  The paper that gave the Brier score its name. The
  original context is weather forecasting; the metric
  transferred verbatim.
- **Kumar, Ananya, Percy Liang, and Tengyu Ma. "Verified
  Uncertainty Calibration" (NeurIPS 2019).**
  <https://arxiv.org/abs/1909.10155>
  The paper that documents the estimator bias in ECE and
  proposes debiased alternatives. Worth reading before
  quoting ECE as the sole calibration metric.
- **`netcal` — calibration library.**
  <https://fabiankueppers.github.io/calibration-framework/>
  Third-party library covering temperature scaling, beta
  calibration, and more sophisticated recalibrators than
  scikit-learn ships. Chapter 04 references it as the
  next step past `CalibratedClassifierCV`.
  <!-- needs-research: verify the current maintained
  location of the `netcal` docs — the framework has
  moved hosts before. -->

## Chapter 05 — Metrics for ranking

- **scikit-learn — Ranking metrics.**
  <https://scikit-learn.org/stable/modules/model_evaluation.html#ranking-metrics>
  `ndcg_score`, `dcg_score`, `label_ranking_average_precision_score`,
  `label_ranking_loss` — the ranking-metric subset
  chapter 05 uses from scikit-learn.
- **scikit-learn — `ndcg_score`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html>
  The nDCG implementation with the `k` parameter used
  throughout chapter 05.
- **Manning, Raghavan, Schütze. *Introduction to
  Information Retrieval*, chapter 8.**
  <https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html>
  The precision@k / recall@k / MAP / MRR /
  interpolated-precision walk-through.
- **Järvelin, Kalervo, and Jaana Kekäläinen. "Cumulated
  Gain-Based Evaluation of IR Techniques" (*ACM
  Transactions on Information Systems*, 2002).**
  <https://doi.org/10.1145/582415.582418>
  The original nDCG paper; the source for the
  logarithmic discount chapter 05 shows.
- **Voorhees, Ellen M., and Donna K. Harman. *TREC:
  Experiment and Evaluation in Information Retrieval*
  (MIT Press, 2005).**
  The reference for pooled-relevance judgements and
  the evaluation methodology behind the TREC family of
  benchmarks. Cited as the source for chapter 05's
  "you never have all the judgements you would like"
  point.
  <!-- needs-research: no free canonical URL for the
  book; link to the MIT Press page if verified. -->
- **Craswell, Nick. "Mean Reciprocal Rank" (in
  *Encyclopedia of Database Systems*, Springer,
  2009).**
  <https://doi.org/10.1007/978-0-387-39940-9_488>
  The compact definition of MRR chapter 05 cites.
- **Robertson, Stephen. "A New Interpretation of Average
  Precision" (SIGIR 2008).**
  <https://doi.org/10.1145/1390334.1390453>
  The paper that unpacks what MAP actually rewards —
  useful for reviewers who ask "why not just accuracy@k".

## Chapter 06 — Splits, leakage, and cross-validation

- **scikit-learn — Cross-validation user guide.**
  <https://scikit-learn.org/stable/modules/cross_validation.html>
  The chapter-06 primary reference.
- **scikit-learn — `KFold`, `StratifiedKFold`,
  `RepeatedKFold`, `ShuffleSplit`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html>
  The four splitters chapter 06 walks through.
- **scikit-learn — `Pipeline` and `ColumnTransformer`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>
  The primitives that make "fit preprocessing inside the
  CV loop" the default, not an extra step.
- **scikit-learn — Common pitfalls: data leakage.**
  <https://scikit-learn.org/stable/common_pitfalls.html#data-leakage>
  The subset of the leakage taxonomy that scikit-learn's
  own docs single out.
- **Kaufman, Shachar, Saharon Rosset, and Claudia
  Perlich. "Leakage in Data Mining: Formulations,
  Detection, and Avoidance" (KDD 2011).**
  <https://doi.org/10.1145/2020408.2020496>
  The classical taxonomy behind chapter 06's list of
  leakage kinds. Section 3 is worth reading verbatim.
- **Nadeau, Claude, and Yoshua Bengio. "Inference for the
  Generalization Error" (*Machine Learning*, 2003).**
  <https://doi.org/10.1023/A:1024068626366>
  The paper behind the corrected-resampled *t*-test
  chapter 06 cites for "is the difference between two
  CV means statistically meaningful?"
- **Varma, Sudhir, and Richard Simon. "Bias in error
  estimation when using cross-validation for model
  selection" (*BMC Bioinformatics*, 2006).**
  <https://doi.org/10.1186/1471-2105-7-91>
  The paper behind chapter 08's rule that
  hyperparameter selection and error estimation must
  use disjoint CV loops (nested CV).
- **Cawley, Gavin C., and Nicola L. C. Talbot. "On
  Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation" (*JMLR*,
  2010).**
  <https://www.jmlr.org/papers/v11/cawley10a.html>
  The other classical citation for "the CV score you
  tuned on is optimistically biased". Chapter 06's
  argument for a held-out test set boils down to this
  paper.

## Chapter 07 — Time-based validation

- **scikit-learn — `TimeSeriesSplit`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
  The primitive chapter 07 wraps. The `gap` and
  `max_train_size` parameters are the load-bearing ones.
- **scikit-learn — `TimeSeriesSplit` user-guide section.**
  <https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split>
  The narrative version — why row-position-based
  splitting requires sorted data.
- **sktime — splitters.**
  <https://www.sktime.net/en/stable/api_reference/split.html>
  `SlidingWindowSplitter`, `ExpandingWindowSplitter`,
  `CutoffSplitter`, `TemporalTrainTestSplitter`. Chapter
  07's alternatives when scikit-learn's single splitter
  is not enough.
- **Bergmeir, Christoph, and José M. Benítez. "On the use
  of cross-validation for time series predictor
  evaluation" (*Information Sciences*, 2012).**
  <https://doi.org/10.1016/j.ins.2011.12.028>
  The paper behind chapter 07's argument that random-CV
  is not merely optimistic but *systematically* biased
  for time series.
- **Hyndman, Rob J., and George Athanasopoulos.
  *Forecasting: Principles and Practice*, section 5.10
  ("Time series cross-validation").**
  <https://otexts.com/fpp3/tscv.html>
  The forecasting-textbook version of walk-forward and
  rolling-origin evaluation.
- **Roberts, David R., et al. "Cross-validation
  strategies for data with temporal, spatial,
  hierarchical, or phylogenetic structure" (*Ecography*,
  2017).**
  <https://doi.org/10.1111/ecog.02881>
  The paper that generalises "you need a structure-aware
  split" beyond time. Directly relevant to chapter 08
  as well.

## Chapter 08 — Group-based and nested cross-validation

- **scikit-learn — `GroupKFold` and
  `StratifiedGroupKFold`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html>
  The two group-aware splitters chapter 08 uses.
- **scikit-learn — `LeaveOneGroupOut` and
  `LeavePGroupsOut`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html>
  When the number of groups is small enough that
  leave-one-out is affordable.
- **scikit-learn — Nested cross-validation example.**
  <https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html>
  The canonical worked example — a `GridSearchCV`
  wrapped in `cross_val_score`. Chapter 08's group
  version is a structural extension of this.
- **scikit-learn — `GridSearchCV`,
  `RandomizedSearchCV`, `HalvingGridSearchCV`,
  `HalvingRandomSearchCV`.**
  <https://scikit-learn.org/stable/modules/grid_search.html>
  The tuning primitives chapter 08 references. Halving
  variants require the `enable_halving_search_cv`
  experimental flag.
- **Varma, Sudhir, and Richard Simon. "Bias in error
  estimation when using cross-validation for model
  selection" (BMC Bioinformatics, 2006).**
  <https://doi.org/10.1186/1471-2105-7-91>
  Duplicate reference; the primary source for chapter
  08's nested-CV argument.
- **Cawley, Gavin C., and Nicola L. C. Talbot. "On
  Over-fitting in Model Selection…" (JMLR, 2010).**
  <https://www.jmlr.org/papers/v11/cawley10a.html>
  Duplicate reference; the other primary source.
- **sktime — `GroupTimeSeriesSplit`.**
  <https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.split.SingleWindowSplitter.html>
  <!-- needs-research: verify sktime's exact export name
  for a group-aware time-series splitter, and update
  the URL when the API stabilises. -->

## Chapter 09 — Baselines that tell you something

- **scikit-learn — `DummyClassifier` / `DummyRegressor`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html>
  The primary primitives.
- **Hyndman, Rob J., and George Athanasopoulos.
  *Forecasting: Principles and Practice*, section 5.2
  ("Some simple forecasting methods").**
  <https://otexts.com/fpp3/simple-methods.html>
  The persistence / seasonal-naive / drift baselines
  chapter 09 lists for time-series tasks, with the
  reasoning for each.
- **Grinsztajn, Léo, Edouard Oyallon, and Gaël
  Varoquaux. "Why do tree-based models still outperform
  deep learning on tabular data?" (NeurIPS 2022).**
  <https://arxiv.org/abs/2207.08815>
  The paper chapter 09 cites for "GBDT is the tabular
  strong baseline; deep tabular models should be judged
  against it".
- **Rendle, Steffen, et al. "Neural Collaborative
  Filtering vs. Matrix Factorization Revisited" (RecSys
  2020).**
  <https://arxiv.org/abs/2005.09683>
  The paper behind chapter 09's recommendation-baseline
  advice: a well-tuned matrix-factoriser is often the
  right strong baseline before any deep model.
- **Landis, J. Richard, and Gary G. Koch. "The
  Measurement of Observer Agreement for Categorical
  Data" (*Biometrics*, 1977).**
  <https://doi.org/10.2307/2529310>
  The paper behind Cohen's-κ interpretation ranges
  ("slight", "fair", "moderate", "substantial",
  "almost perfect") that chapter 09 uses to talk about
  human-agreement ceilings.

## Chapter 10 — Error analysis

- **scikit-learn — `confusion_matrix`,
  `ConfusionMatrixDisplay`, `classification_report`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html>
  The primitives behind chapter 10's move 1.
- **Ribeiro, Marco Tulio, Sameer Singh, and Carlos
  Guestrin. "'Why Should I Trust You?': Explaining the
  Predictions of Any Classifier" (KDD 2016).**
  <https://arxiv.org/abs/1602.04938>
  The LIME paper. The "look at individual errors"
  discipline chapter 10 formalises has its roots here.
- **Ribeiro, Marco Tulio, Tongshuang Wu, Carlos
  Guestrin, and Sameer Singh. "Beyond Accuracy:
  Behavioral Testing of NLP Models with CheckList"
  (ACL 2020).**
  <https://arxiv.org/abs/2005.04118>
  The paper that formalised structured error slicing
  into unit-test-shaped behavioural tests. Directly
  relevant to chapter 10's "turn a slice into a
  hypothesis" step.
- **Northcutt, Curtis G., Anish Athalye, and Jonas
  Mueller. "Pervasive Label Errors in Test Sets
  Destabilize Machine Learning Benchmarks" (NeurIPS
  2021 Datasets & Benchmarks).**
  <https://arxiv.org/abs/2103.14749>
  The paper behind chapter 10's warning that a
  meaningful fraction of top-loss rows are often label
  errors, not model errors. Companion tool: `cleanlab`
  (below).
- **`cleanlab`.**
  <https://cleanlab.ai/>
  <https://github.com/cleanlab/cleanlab>
  Open-source library for label-error detection.
  Useful when the top-loss rows in chapter 10's move
  2 keep turning out to be annotation mistakes.
- **`Snorkel` / `Snorkel Flow` — behavioural error
  slicing.**
  <https://snorkel.ai/>
  Adjacent tool for slice-based data programming.
  Mentioned here as the industrial descendant of
  CheckList's slice discipline.
- **Efron, Bradley, and Robert Tibshirani. *An
  Introduction to the Bootstrap* (Chapman & Hall,
  1993).**
  The reference for bootstrap confidence intervals on
  metrics and per-slice metrics. Chapter 10's stretch
  goal ("bootstrap the slice metric") is the ML
  application of this book's chapters 6 and 12.

## Chapter 11 — Fairness and robustness slices

### Fairness

- **Fairlearn — official documentation.**
  <https://fairlearn.org/main/user_guide/>
  The chapter-11 companion for the level-20 subgroup
  report. `MetricFrame`, `demographic_parity_difference`,
  `equalized_odds_difference`, and the
  `selection_rate` primitive are the ones the exercise
  uses.
- **Fairlearn — `fairlearn.metrics` API reference.**
  <https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.html>
  The full list of fairness metrics with signatures.
- **AIF360 — official documentation.**
  <https://aif360.readthedocs.io/en/stable/>
  IBM's counterpart. Broader metric set, broader
  mitigation-algorithm set, older. Widely cited in
  academic fairness papers.
- **Barocas, Hardt, Narayanan. *Fairness and Machine
  Learning* (MIT Press, 2023).**
  <https://fairmlbook.org/>
  Primary reference (also under "Cross-cutting"). The
  fairness definitions and impossibility results
  chapter 11 cites are in chapters 2–4.
- **Hardt, Moritz, Eric Price, and Nathan Srebro.
  "Equality of Opportunity in Supervised Learning"
  (NeurIPS 2016).**
  <https://arxiv.org/abs/1610.02413>
  The paper that named equalised-odds / equal-
  opportunity as fairness criteria.
- **Chouldechova, Alexandra. "Fair Prediction with
  Disparate Impact: A Study of Bias in Recidivism
  Prediction Instruments" (*Big Data*, 2017).**
  <https://arxiv.org/abs/1610.07524>
  The paper behind one branch of the fairness-
  impossibility result: predictive parity and
  equalised odds cannot both hold under unequal base
  rates.
- **Kleinberg, Jon, Sendhil Mullainathan, and Manish
  Raghavan. "Inherent Trade-Offs in the Fair
  Determination of Risk Scores" (ITCS 2017).**
  <https://arxiv.org/abs/1609.05807>
  The other branch of the same impossibility result.
- **Buolamwini, Joy, and Timnit Gebru. "Gender Shades:
  Intersectional Accuracy Disparities in Commercial
  Gender Classification" (FAT* 2018).**
  <https://proceedings.mlr.press/v81/buolamwini18a.html>
  The paper that demonstrated intersectional subgroup
  gaps in shipping systems. The canonical case study
  behind chapter 11's "look at intersections" advice.
- **Mitchell, Margaret, et al. "Model Cards for Model
  Reporting" (FAT* 2019).**
  <https://arxiv.org/abs/1810.03677>
  The documentation template that a completed
  slice-report artefact naturally slots into.

### Robustness

- **Hendrycks, Dan, and Thomas Dietterich. "Benchmarking
  Neural Network Robustness to Common Corruptions and
  Perturbations" (ICLR 2019).**
  <https://arxiv.org/abs/1903.12261>
  The paper that formalised the "naturally-occurring
  corruption" evaluation regime chapter 11's Part B
  operationalises for image data.
- **Recht, Benjamin, et al. "Do ImageNet Classifiers
  Generalize to ImageNet?" (ICML 2019).**
  <https://arxiv.org/abs/1902.10811>
  The paper that made "distribution-shift as a routine
  probe" a mainstream evaluation practice.
- **Koh, Pang Wei, et al. "WILDS: A Benchmark of in-the-
  Wild Distribution Shifts" (ICML 2021).**
  <https://arxiv.org/abs/2012.07421>
  A benchmark of realistic distribution shifts across
  ten domains. Useful reference for the *kinds* of
  shift chapter 11 asks you to construct.
- **`albumentations` — image-perturbation library.**
  <https://albumentations.ai/docs/>
  The library chapter 11 recommends for image
  perturbations in Part B.
- **`nlpaug` — text-perturbation library.**
  <https://nlpaug.readthedocs.io/en/latest/>
  Character / word / sentence perturbations for NLP
  robustness slices.
- **`TextAttack` — NLP robustness framework.**
  <https://textattack.readthedocs.io/en/latest/>
  Both natural and adversarial NLP perturbations.
  Chapter 11 restricts the level-20 report to the
  natural subset.

### Distribution shift diagnostics

- **`evidently` — data / model monitoring library.**
  <https://docs.evidentlyai.com/>
  The library that ships PSI, KS, Wasserstein, and
  Jensen-Shannon distance out of the box. Chapter 11
  uses these as pre-launch diagnostics; `mod-109` uses
  the same primitives as production drift signals.
- **`alibi-detect` — drift and outlier detection.**
  <https://docs.seldon.io/projects/alibi-detect/en/latest/>
  The other production-drift toolkit. Same primitives,
  different API surface.
- **Rabanser, Stephan, Stephan Günnemann, and Zachary
  C. Lipton. "Failing Loudly: An Empirical Study of
  Methods for Detecting Dataset Shift" (NeurIPS 2019).**
  <https://arxiv.org/abs/1810.11953>
  The comparative study of shift-detection methods.
  Useful context for "which diagnostic actually
  catches which kind of shift".
- **Sculley, D., et al. "Hidden Technical Debt in
  Machine Learning Systems" (NeurIPS 2015).**
  <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  The paper behind chapter 11's "the deployed model
  will encounter distributions the training data did
  not" framing.

### Population Stability Index (PSI)

- **Siddiqi, Naeem. *Credit Risk Scorecards:
  Developing and Implementing Intelligent Credit
  Scoring* (Wiley, 2005).**
  The credit-scoring textbook where the PSI ranges
  chapter 11 quotes (< 0.1 stable; 0.1–0.25 small
  shift; > 0.25 large shift) originate.
  <!-- needs-research: the "> 0.25 is a large shift"
  cutoff is widely reproduced but the original
  citation is diffuse across the credit-modelling
  literature; verify against Siddiqi's chapter on
  characteristic analysis before pinning the number
  to a single source. -->
- **Yurdakul, Bilal. "Statistical Properties of
  Population Stability Index" (Ph.D. dissertation,
  Western Michigan University, 2018).**
  <https://scholarworks.wmich.edu/dissertations/3208/>
  The reference chapter 11 cites for PSI's known
  small-sample bias and the effect of the binning
  scheme.

## Adjacent standards and references

- **NIST — AI Risk Management Framework (AI RMF 1.0).**
  <https://www.nist.gov/itl/ai-risk-management-framework>
  The umbrella framework in which "measure and manage"
  ML risk sits. The Measure function's sub-categories
  (2.5 Evaluation of AI System Performance, 2.11 Fairness
  and Bias, 2.6 Safety, 2.7 Security and Resilience) are
  the vocabulary chapter 11's slice report answers to.
- **ISO/IEC 5259 — Data quality for analytics and
  machine learning.**
  <https://www.iso.org/standard/81088.html>
  The ISO umbrella standard for ML data quality.
  Chapter 06's leakage taxonomy and chapter 11's
  distribution-shift probes are the evaluation-side
  operationalisation of parts of this standard.
  <!-- needs-research: ISO 5259 is a multi-part
  standard whose parts publish on different
  schedules; verify which parts are current before
  citing them individually. -->
- **EU AI Act (Regulation (EU) 2024/1689).**
  <https://eur-lex.europa.eu/eli/reg/2024/1689/oj>
  The regulation that turns "evaluation is not
  optional" into a legal requirement for high-risk
  AI systems. Articles 9 (risk management), 10 (data
  governance), and 15 (accuracy, robustness, and
  cybersecurity) are the direct references for the
  discipline this module teaches.
- **UK ICO — Guidance on AI and data protection.**
  <https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/>
  The UK-side regulator's take. Sections on fairness,
  accuracy, and statistical accuracy overlap with
  chapters 03 and 11 directly.

## Datasets used in the exercises

The exercises deliberately do not prescribe a dataset —
they reuse whatever is in play from `mod-103`, `mod-104`,
`mod-105`, or `mod-106`. Two options appear in the
chapter examples and the exercise prompts:

- **UCI Adult (Census Income).**
  Kohavi, Ron. "Scaling Up the Accuracy of Naive-Bayes
  Classifiers: A Decision-Tree Hybrid" (KDD 1996).
  <https://archive.ics.uci.edu/dataset/2/adult>
  Loadable via
  `sklearn.datasets.fetch_openml("adult", version=2)`.
  Ships with `sex`, `race`, and `age` — a working
  example for exercise-04's subgroup report.
- **NYC Taxi trip data.**
  <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
  Monthly Parquet snapshots. Pickup borough, hour-of-
  day, and weekend/weekday give operational-subgroup
  slices; holding out the latest week gives a
  distribution-shift probe.

<!-- needs-research: on each autonomous research
cycle, verify these URLs still resolve. In
particular: (a) scikit-learn docs pages (they
periodically reorganise between /modules/, /api/,
and /modules/generated/); (b) sktime API pages
(the splitter API has renamed exports across
minor versions); (c) fairlearn docs (moved from
fairlearn.github.io to fairlearn.org); (d) the
NYC TLC data landing page (they have restructured
the download layout at least twice); (e) EU AI Act
consolidated version (numbered articles may shift
between the OJ text and later corrigenda). Swap in
current pages when a doc URL 404s. -->
