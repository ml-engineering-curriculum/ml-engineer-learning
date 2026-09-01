# Resources for mod-104 — Classical ML Modeling: Regression, Trees, Ensembles

A curated short list. The goal is depth-on-demand for the chapter
you got stuck on, not a comprehensive bibliography. Primary
documentation is listed first; textbooks, papers, and tooling
references follow.

## Linear and logistic regression (chapter 02)

- **scikit-learn — linear models (User Guide §1.1).**
  <https://scikit-learn.org/stable/modules/linear_model.html>
  The main-line reference for every estimator in chapters 02–03:
  `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`,
  `ElasticNet`, `SGDClassifier`, and their `*CV` variants.
- **scikit-learn — `LogisticRegression`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
  The `solver`, `penalty`, `C`, and `class_weight` arguments
  chapter 02 uses. Read the "solver" table — it decides which
  penalties are actually available.
- **scikit-learn — solver choice for logistic regression.**
  <https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>
  The table that answers "why is `saga` the right default when I
  need L1 or elastic-net?"
- **Hastie, Trevor, Robert Tibshirani, and Jerome Friedman.
  *The Elements of Statistical Learning* (2nd ed., Springer, 2009).**
  Free PDF at <https://hastie.su.domains/ElemStatLearn/>. Chapter 3
  (linear regression) and chapter 4 (linear classification) are the
  canonical academic treatment behind chapter 02.
- **James, Gareth, et al. *An Introduction to Statistical Learning*
  (2nd ed., Springer, 2021).** Free PDF at
  <https://www.statlearning.com/>. Gentler pass over the same
  material as *ESL*, with worked R examples that translate directly
  to scikit-learn.

## Regularisation: Ridge, Lasso, elastic-net (chapter 03)

- **scikit-learn — `Ridge`, `RidgeCV`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html>
- **scikit-learn — `Lasso`, `LassoCV`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html>
- **scikit-learn — `ElasticNet`, `ElasticNetCV`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html>
- **scikit-learn — regularisation-path plots (examples).**
  <https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html>
  The `lasso_path` / `enet_path` examples exercise-01 asks you to
  reproduce.
- **Tibshirani, Robert. "Regression Shrinkage and Selection via the
  Lasso" (JRSS-B, 1996).**
  <https://www.jstor.org/stable/2346178>
  The original Lasso paper. Short and readable; the coordinate-
  descent solver came later but the theory is unchanged.
- **Hoerl, Arthur E., and Robert W. Kennard. "Ridge Regression:
  Biased Estimation for Nonorthogonal Problems" (Technometrics,
  1970).**
  <https://doi.org/10.1080/00401706.1970.10488634>
  The Ridge paper. The bias–variance argument chapter 03 uses is
  descended directly from section 3.
- **Zou, Hui, and Trevor Hastie. "Regularization and Variable
  Selection via the Elastic Net" (JRSS-B, 2005).**
  <https://hastie.su.domains/Papers/elasticnet.pdf>
  The elastic-net paper. The "grouping effect" argument for keeping
  correlated features together is here.
- **Friedman, Jerome, Trevor Hastie, and Rob Tibshirani.
  "Regularization Paths for Generalized Linear Models via Coordinate
  Descent" (JStatSoft, 2010).**
  <https://www.jstatsoft.org/article/view/v033i01>
  The paper behind the `glmnet` algorithm that scikit-learn's
  coordinate-descent solvers implement.

## Decision trees (chapter 04)

- **scikit-learn — decision trees (User Guide §1.10).**
  <https://scikit-learn.org/stable/modules/tree.html>
  The reference for `DecisionTreeClassifier` / `DecisionTreeRegressor`:
  the CART splitting criterion, the pruning parameters (`max_depth`,
  `min_samples_leaf`, `ccp_alpha`), and the visualisation utilities.
- **scikit-learn — `DecisionTreeClassifier`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>
- **scikit-learn — `plot_tree` and `export_graphviz`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html>
  The two ways to render a fitted tree — one for notebooks, one for
  publication.
- **Breiman, Leo, Jerome Friedman, Richard Olshen, and Charles
  Stone. *Classification and Regression Trees* (CART) (Wadsworth,
  1984).**
  The book that introduced the CART algorithm scikit-learn
  implements. Long out of print but widely available; the algorithm
  chapters (2–4) still read cleanly.
- **Quinlan, J. Ross. "Induction of Decision Trees" (Machine
  Learning, 1986).**
  <https://link.springer.com/article/10.1007/BF00116251>
  The ID3 paper. Historical context — ID3 → C4.5 → C5.0 is the
  alternative tree lineage to CART.

## Random forests and bagging (chapter 05)

- **scikit-learn — ensemble methods (User Guide §1.11).**
  <https://scikit-learn.org/stable/modules/ensemble.html>
  Random forests, extra-trees, bagging, and the boosting families
  all live here.
- **scikit-learn — `RandomForestClassifier` /
  `RandomForestRegressor`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>
  The `n_estimators`, `max_features`, `oob_score`, and `n_jobs`
  arguments chapter 05 relies on.
- **scikit-learn — `BaggingClassifier`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html>
  Generic bagging wrapper — useful when you want to bag a
  non-tree base learner.
- **scikit-learn — `ExtraTreesClassifier`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html>
- **Breiman, Leo. "Random Forests" (Machine Learning, 2001).**
  <https://link.springer.com/article/10.1023/A:1010933404324>
  The original paper. The out-of-bag error argument chapter 05
  uses is from section 3.1.
- **Breiman, Leo. "Bagging Predictors" (Machine Learning, 1996).**
  <https://link.springer.com/article/10.1007/BF00058655>
  The bagging paper. Predates random forests and gives the
  variance-reduction argument on its own.

## Gradient-boosted trees (chapter 06)

- **scikit-learn — `HistGradientBoostingClassifier` /
  `HistGradientBoostingRegressor`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html>
  Chapter 06's "zero-dependency" default. The
  `categorical_features="from_dtype"` and `early_stopping="auto"`
  arguments are the ones that changed most recently.
- **XGBoost — documentation.**
  <https://xgboost.readthedocs.io/en/stable/>
  The `Python API`, the `Parameters` page, and the `Tree Booster`
  overview are the three tabs you keep open while tuning.
- **XGBoost — parameters reference.**
  <https://xgboost.readthedocs.io/en/stable/parameter.html>
  The canonical list. `learning_rate`, `max_depth`, `subsample`,
  `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma` — all in
  one page.
- **LightGBM — documentation.**
  <https://lightgbm.readthedocs.io/en/stable/>
  The scikit-learn API page and the `Parameters Tuning` guide are
  chapter 06's LightGBM references.
- **LightGBM — parameter tuning guide.**
  <https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html>
  The official "start here, then tune this, then that" playbook
  from the LightGBM authors.
- **CatBoost — documentation.**
  <https://catboost.ai/en/docs/>
  The `Python package` and `Overfitting detector` pages are the
  ones exercise-03 might send you to if you pick CatBoost.
- **Chen, Tianqi, and Carlos Guestrin. "XGBoost: A Scalable Tree
  Boosting System" (KDD 2016).**
  <https://arxiv.org/abs/1603.02754>
  The XGBoost paper. Section 2 is the regularised-objective
  argument chapter 06 quotes.
- **Ke, Guolin, et al. "LightGBM: A Highly Efficient Gradient
  Boosting Decision Tree" (NeurIPS 2017).**
  <https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>
  The LightGBM paper. Introduces GOSS (gradient-based one-side
  sampling) and EFB (exclusive feature bundling).
- **Prokhorenkova, Liudmila, et al. "CatBoost: unbiased boosting
  with categorical features" (NeurIPS 2018).**
  <https://arxiv.org/abs/1706.09516>
  The CatBoost paper. The ordered-boosting and target-statistics
  arguments for why CatBoost handles categoricals differently.
- **Friedman, Jerome H. "Greedy Function Approximation: A Gradient
  Boosting Machine" (Annals of Statistics, 2001).**
  <https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full>
  The paper that named "gradient boosting". Reads as the theory
  behind everything the modern libraries implement.
- **Shwartz-Ziv, Ravid, and Amitai Armon. "Tabular Data: Deep
  Learning is Not All You Need" (Information Fusion, 2022).**
  <https://arxiv.org/abs/2106.03253>
  Empirical case that gradient-boosted trees remain competitive
  with tabular deep learning. The paper chapter 01 leans on.
- **Grinsztajn, Léo, Edouard Oyallon, and Gaël Varoquaux. "Why do
  tree-based models still outperform deep learning on tabular
  data?" (NeurIPS 2022).**
  <https://arxiv.org/abs/2207.08815>
  Companion argument to Shwartz-Ziv & Armon, with more
  benchmarks and a discussion of the inductive biases that make
  trees a good fit for tabular data.

## Cross-validation strategies (chapter 07)

- **scikit-learn — cross-validation (User Guide §3.1).**
  <https://scikit-learn.org/stable/modules/cross_validation.html>
  The full catalogue of splitters — `KFold`, `StratifiedKFold`,
  `GroupKFold`, `StratifiedGroupKFold`, `TimeSeriesSplit`,
  `RepeatedKFold` — and the semantics of `cross_val_score` /
  `cross_validate`.
- **scikit-learn — `TimeSeriesSplit`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
- **scikit-learn — `GroupKFold` and `StratifiedGroupKFold`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html>
- **scikit-learn — common pitfalls and recommended practices.**
  <https://scikit-learn.org/stable/common_pitfalls.html>
  Data leakage in preprocessing, the "fit-then-transform on the
  whole dataset" trap, and the correct `Pipeline` shape. Read
  alongside chapter 07.
- **scikit-learn — nested cross-validation example.**
  <https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html>
  The canonical worked example of "select hyperparameters in an
  inner loop, evaluate in an outer loop" that chapter 07 (and
  chapter 08) refer to.
- **Cawley, Gavin C., and Nicola L. C. Talbot. "On Over-fitting
  in Model Selection and Subsequent Selection Bias in Performance
  Evaluation" (JMLR, 2010).**
  <https://jmlr.org/papers/v11/cawley10a.html>
  The paper behind chapter 07's warnings about the winner's-curse
  gap between best CV score and held-out test score.
- **Bergmeir, Christoph, Rob Hyndman, and Bonsoo Koo. "A note on
  the validity of cross-validation for evaluating autoregressive
  time series prediction" (Computational Statistics & Data
  Analysis, 2018).**
  <https://doi.org/10.1016/j.csda.2017.11.003>
  When plain k-fold is (sometimes) fine for time series, and when
  it is not — the nuance behind chapter 07's `TimeSeriesSplit`
  discipline.

## Hyperparameter search (chapter 08)

- **scikit-learn — hyperparameter tuning (User Guide §3.2).**
  <https://scikit-learn.org/stable/modules/grid_search.html>
  The reference for `GridSearchCV`, `RandomizedSearchCV`, and
  the halving variants.
- **scikit-learn — `RandomizedSearchCV`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>
- **scikit-learn — `HalvingGridSearchCV` and
  `HalvingRandomSearchCV`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html>
  The successive-halving family chapter 08 recommends for
  expensive estimators. Still marked "experimental" in the
  scikit-learn source; check the import path.
- **Optuna — documentation.**
  <https://optuna.readthedocs.io/en/stable/>
  The library exercise-03 uses. The "Tutorial" and "Key Features"
  sections are the fastest way in.
- **Optuna — samplers reference.**
  <https://optuna.readthedocs.io/en/stable/reference/samplers/index.html>
  `TPESampler`, `CmaEsSampler`, `NSGAIISampler` — the search
  algorithms behind `create_study(sampler=...)`.
- **Optuna — pruners reference.**
  <https://optuna.readthedocs.io/en/stable/reference/pruners.html>
  `MedianPruner`, `HyperbandPruner`, `SuccessiveHalvingPruner` —
  the early-stopping-of-trials mechanism.
- **Optuna — visualization.**
  <https://optuna.readthedocs.io/en/stable/reference/visualization/index.html>
  The `plot_optimization_history`, `plot_param_importances`,
  `plot_slice`, and `plot_parallel_coordinate` calls exercise-03
  asks you to produce.
- **Bergstra, James, and Yoshua Bengio. "Random Search for
  Hyper-Parameter Optimization" (JMLR, 2012).**
  <https://www.jmlr.org/papers/v13/bergstra12a.html>
  The paper behind chapter 08's argument for random over grid
  search when the effective dimensionality of the search space is
  low.
- **Bergstra, James, et al. "Algorithms for Hyper-Parameter
  Optimization" (NeurIPS 2011).**
  <https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html>
  The paper introducing TPE (Tree-structured Parzen Estimator),
  the default sampler Optuna uses.
- **Akiba, Takuya, et al. "Optuna: A Next-generation
  Hyperparameter Optimization Framework" (KDD 2019).**
  <https://arxiv.org/abs/1907.10902>
  The Optuna paper. Section 3 (define-by-run API) is the
  argument for the `def objective(trial)` shape exercise-03 uses.
- **Li, Lisha, et al. "Hyperband: A Novel Bandit-Based Approach
  to Hyperparameter Optimization" (JMLR, 2018).**
  <https://www.jmlr.org/papers/v18/16-558.html>
  The Hyperband paper. Successive halving with adaptive budgets;
  the theoretical grounding for `HalvingRandomSearchCV`.

## Feature importance and selection (chapter 09)

- **scikit-learn — feature selection (User Guide §1.13).**
  <https://scikit-learn.org/stable/modules/feature_selection.html>
  `SelectKBest`, `SelectFromModel`, `RFE`, and `RFECV`.
- **scikit-learn — permutation importance.**
  <https://scikit-learn.org/stable/modules/permutation_importance.html>
  The user-guide page and the `permutation_importance` API. Read
  the "Misleading values on strongly correlated features" example.
- **scikit-learn — impurity-based importance warning.**
  <https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation>
  The paragraphs chapter 09 quotes verbatim on why
  `feature_importances_` is a smoke test, not a deliverable.
- **Strobl, Carolin, Anne-Laure Boulesteix, Achim Zeileis, and
  Torsten Hothorn. "Bias in random forest variable importance
  measures: illustrations, sources and a solution" (BMC
  Bioinformatics, 2007).**
  <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25>
  The paper that documented the cardinality bias in impurity-based
  importance and proposed permutation importance as a partial fix.
- **Altmann, André, et al. "Permutation importance: a corrected
  feature importance measure" (Bioinformatics, 2010).**
  <https://academic.oup.com/bioinformatics/article/26/10/1340/193348>
  The Altmann correction that produces a p-value alongside the
  raw permutation-importance number. Worth knowing exists.
- **Meinshausen, Nicolai, and Peter Bühlmann. "Stability
  Selection" (JRSS-B, 2010).**
  <https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00740.x>
  The paper behind the bootstrap-and-count pattern chapter 09
  (and exercise-01) uses for defensible feature selection.

## SHAP (chapter 10)

- **`shap` — documentation.**
  <https://shap.readthedocs.io/en/latest/>
  The library reference. The "API examples" and "Explainers"
  sections are the two you re-read most.
- **`shap` — `TreeExplainer`.**
  <https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html>
  The `feature_perturbation="tree_path_dependent"` vs.
  `"interventional"` argument, the `data` parameter, and the
  `model_output` values ("raw" vs. "probability").
- **`shap` — plots API.**
  <https://shap.readthedocs.io/en/latest/api.html#plots>
  Reference for `shap.plots.bar`, `beeswarm`, `waterfall`,
  `scatter`, `force`, and their arguments.
- **Lundberg, Scott M., and Su-In Lee. "A Unified Approach to
  Interpreting Model Predictions" (NeurIPS 2017).**
  <https://arxiv.org/abs/1705.07874>
  The paper that introduced SHAP. Sections 2–3 are the axiomatic
  argument chapter 10 summarises.
- **Lundberg, Scott M., et al. "From local explanations to global
  understanding with explainable AI for trees" (Nature Machine
  Intelligence, 2020).**
  <https://www.nature.com/articles/s42256-019-0138-9>
  The TreeSHAP paper — the polynomial-time exact algorithm for
  Shapley values on tree ensembles that makes production SHAP
  practical.
- **Shapley, Lloyd S. "A value for n-person games" (in
  *Contributions to the Theory of Games II*, 1953).**
  The original cooperative-game-theory paper. Cited in chapter 10
  for the four axioms (efficiency, symmetry, dummy, additivity)
  that make Shapley values unique.
- **Molnar, Christoph. *Interpretable Machine Learning* (2nd ed.,
  2022).** Free HTML at <https://christophm.github.io/interpretable-ml-book/>.
  Chapter 9 (SHAP) is the most complete practitioner-oriented
  treatment of everything chapters 09–10 cover. Chapter 8 (LIME)
  and chapter 5 (feature effects: PDP, ALE, ICE) round out the
  interpretability toolbox.
- **Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "'Why
  Should I Trust You?': Explaining the Predictions of Any
  Classifier" (KDD 2016).**
  <https://arxiv.org/abs/1602.04938>
  The LIME paper. Predecessor to SHAP; the local-linear-
  approximation idea is worth understanding for context.

## Cross-cutting: books and long reads

- **Hastie, Trevor, Robert Tibshirani, and Jerome Friedman.
  *The Elements of Statistical Learning* (2nd ed., Springer, 2009).**
  Free PDF at <https://hastie.su.domains/ElemStatLearn/>. Chapters
  3–4 (linear models), 9 (trees), 10 (boosting), and 15 (random
  forests) map almost one-to-one onto chapters 02–06 of this
  module.
- **James, Gareth, et al. *An Introduction to Statistical Learning*
  (2nd ed., Springer, 2021).** Free PDF at <https://www.statlearning.com/>.
  Gentler pass over the same material. Chapter 6 (linear model
  selection / regularisation) and chapter 8 (trees) are the
  ones to cross-read with 03 and 04–06.
- **Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn,
  Keras, and TensorFlow* (3rd ed., O'Reilly, 2022).** Chapter 4
  (training models), chapter 6 (decision trees), and chapter 7
  (ensemble learning) are the practitioner-oriented companions to
  chapters 02–06.
- **Molnar, Christoph. *Interpretable Machine Learning* (2nd ed.,
  2022).** Free HTML at <https://christophm.github.io/interpretable-ml-book/>.
  The natural companion to chapters 09–10.
- **Huyen, Chip. *Designing Machine Learning Systems* (O'Reilly,
  2022).** Chapter 6 ("Model Development and Offline Evaluation")
  is the systems-level view of the same territory. Pairs with
  `mod-107` more than with this module, but the section on
  ensembling in production is directly relevant to chapter 06.
- **Kaggle — "State of Machine Learning and Data Science" survey.**
  <https://www.kaggle.com/kaggle-survey-2022>
  <!-- needs-research: swap in the most recent year's survey when a
  new one is published; Kaggle typically releases annually. -->
  Empirical evidence for the "gradient boosting is still the
  practitioner default on tabular data" claim in chapter 01.

## Datasets used in the exercises

- **California Housing.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>
  Regression, 20 640 rows, 8 numeric features. The default target
  for exercise-01.
- **Ames Housing.**
  <https://www.openml.org/d/42165>
  Loadable via `sklearn.datasets.fetch_openml("house_prices",
  as_frame=True)`. De Cock, Dean. "Ames, Iowa: Alternative to the
  Boston Housing Data as an End of Semester Regression Project"
  (*JSE*, 2011). <https://jse.amstat.org/v19n3/decock.pdf>
- **Diabetes.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>
- **UCI Adult (Census Income).**
  <https://archive.ics.uci.edu/dataset/2/adult>
  Also loadable via `sklearn.datasets.fetch_openml("adult",
  version=2, as_frame=True)`. The default classification target
  for exercises 02, 03, and 04.
- **Breast Cancer Wisconsin.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>
  Small binary-classification dataset, used in exercise-01's
  stretch goals.
- **HIGGS.**
  <https://archive.ics.uci.edu/dataset/280/higgs>
  Larger classification dataset (11M rows) — exercise-03 optional
  scale target.

<!-- needs-research: on each autonomous research cycle, verify these URLs still resolve and swap in the current stable-version page when a project promotes a new major (scikit-learn 1.x, XGBoost 2.x → 3.x, LightGBM 4.x → 5.x, Optuna 3.x → 4.x, shap 0.x → 1.x, etc.). -->
