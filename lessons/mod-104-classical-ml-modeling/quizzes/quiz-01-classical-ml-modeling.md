# quiz-01 — Classical ML Modeling: Regression, Trees, Ensembles

> **Format:** 12 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–10 of this module.
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter the point
> comes from — go re-read that section if you missed the question.

---

## Questions

### 1. (Chapter 01) A teammate argues that "classical ML is just high-bias, low-variance stuff; deep learning dominates everywhere now." Which statement most faithfully matches how chapter 01 frames the four model families this module teaches?

A. All of linear/tree/forest/boosting have high bias and low variance; the
   module exists only as a historical footnote.
B. A linear model is high-bias / low-variance; a deep unpruned tree is
   low-bias / high-variance; a random forest averages many high-variance
   trees so the variance drops without changing bias much; a
   gradient-boosted ensemble controls both terms through learning rate,
   tree depth, and the number of boosting rounds. The four families each
   move a different term, which is why the module teaches them in that
   order.
C. Random forests reduce bias; gradient boosting reduces variance.
D. Regularisation on a linear model *lowers* bias because the penalty
   adds information.

### 2. (Chapter 02) You need to fit a `LogisticRegression` with an L1 penalty on a 500k-row sparse one-hot matrix (~2000 columns). Chapter 02 says which configuration is appropriate, and why?

A. `LogisticRegression(penalty="l1", solver="lbfgs")` — the default is
   always fine.
B. `LogisticRegression(penalty="l1", solver="newton-cholesky")` — fastest
   for `n_samples ≫ n_features`.
C. `LogisticRegression(penalty="l1", solver="saga")` with the numeric
   branch of the pipeline standardised via `StandardScaler(with_mean=False)`
   so sparsity is preserved. The chapter's solver table shows that
   `lbfgs` and `newton-cholesky` do not handle L1 at all, `liblinear`
   handles L1 but is the "small, sparse" solver, and `saga` is the
   L1/elastic-net-capable solver that scales to this size; and the
   chapter explicitly warns that subtracting the mean on sparse one-hot
   features destroys sparsity.
D. `LogisticRegression(penalty="l1", solver="liblinear")` with no
   scaling — penalised linear models are scale-invariant.

### 3. (Chapter 03) A reviewer's PR tightens two regularisers in one commit: a `LogisticRegression` changes from `C=10` to `C=0.1`, and a `Ridge` regressor changes from `alpha=0.1` to `alpha=10`. Chapter 03 calls out the convention flip between these two classes. Which of the two changes made the regulariser **stronger**?

A. Only the `LogisticRegression` change. `Ridge(alpha=10)` is *weaker*
   than `Ridge(alpha=0.1)` because `alpha=0` recovers OLS.
B. Only the `Ridge` change. For `LogisticRegression`, larger `C` means
   stronger regularisation.
C. Both. The chapter notes that `LogisticRegression` uses `C = 1 / α`,
   so dropping `C` from `10` to `0.1` is a 100× increase in regularisation
   strength; `Ridge` uses `alpha` directly, so moving from `0.1` to `10`
   is also a 100× increase. The classes use opposite numeric conventions
   — "a common source of confusion" per the chapter.
D. Neither. Regularisation strength is only decided by the choice of
   `penalty=`, not by the numeric value.

### 4. (Chapter 03) A colleague fits `LassoCV` once on a feature set with several correlated columns and writes in the report: "the important features are X, Y, Z — the Lasso selected them." Which critique and fix does chapter 03 recommend?

A. The critique is that Lasso can only select at most `n_samples`
   features. Switch to Ridge.
B. The critique is that Lasso's coefficients are reported on the raw
   feature scale. Multiply the features by 1000 before interpreting.
C. The critique is that on correlated features Lasso picks one member of
   a correlated pair and zeros the other, essentially at random, and
   *which* member survives changes across resamples. The fix is a
   stability analysis: bootstrap the fit (e.g. ~200 resamples), record
   whether each coefficient is non-zero on each bootstrap, and report
   the **selection frequency** per feature. Features selected in roughly
   fewer than half the bootstraps are coin-flip stable and should not be
   called "selected". Elastic-net (with the L2 term sharing credit
   between correlated features) is the usual remedy when sparsity is
   still desired.
D. Nothing is wrong; `LassoCV`'s built-in `.alpha_` already corrects for
   feature correlation via cross-validation.

### 5. (Chapter 04) Chapter 04 recommends setting `max_depth` (or `min_samples_leaf`) on a stand-alone decision tree, but chapter 05 says `max_depth=None` is the *right* default inside a `RandomForestClassifier`. Which explanation reconciles the two pieces of advice?

A. The two chapters contradict each other; follow chapter 05.
B. A single tree grown to `max_depth=None` keeps splitting until every
   leaf is nearly pure — at the limit, one training row per leaf, 100%
   training accuracy, and terrible generalisation. Inside a random
   forest, each of the ~500 trees is grown deep on a *different* bootstrap
   sample; the trees memorise different rows, and when their
   probabilities are averaged, the disagreements cancel. The ensemble
   average is what makes the uncapped depth safe, not any property of
   the individual tree.
C. `max_depth=None` is required for CART's recursion to terminate.
D. `DecisionTreeClassifier` and `RandomForestClassifier` use different
   split criteria, which is why the right default differs.

### 6. (Chapter 05) Chapter 05 explicitly flags one `RandomForest*` scikit-learn default as **not** matching Breiman's original recommendation and quietly reducing the model to plain bagging. Which is it?

A. `RandomForestClassifier.n_estimators=100` is too few — Breiman
   recommended 500.
B. `RandomForestClassifier.max_features="sqrt"` — should be `"log2"`.
C. `RandomForestRegressor.max_features=1.0` — i.e. "all features
   considered at every split". This turns off the random-subspace
   decorrelation that makes a random forest a random forest; set it
   explicitly to `"sqrt"` or `0.33` for regression.
D. `RandomForestRegressor.bootstrap=True` — should be `False` to match
   Breiman's paper.

### 7. (Chapter 06) You fit `LGBMClassifier(n_estimators=5000, learning_rate=0.05, ...)` with `fit(X_train, y_train, eval_set=[(X_val, y_val)])` and no `early_stopping` callback. What actually happens, per chapter 06?

A. LightGBM infers early stopping from the presence of `eval_set` and
   stops at the best iteration automatically.
B. Training errors immediately — `eval_set` without an early-stopping
   callback is an invalid configuration.
C. Training runs all 5000 iterations; the `eval_set` is only used to
   *report* validation scores round-by-round and does nothing to shorten
   training. Chapter 06 pitfall #5 lists this exactly: both an
   `eval_set` **and** an explicit `early_stopping(stopping_rounds=...)`
   callback (or `early_stopping_rounds=` constructor argument) are
   needed.
D. LightGBM chooses `n_estimators // 10 = 500` as the default when
   `eval_set` is passed.

### 8. (Chapter 07) Each of 5000 users contributes ~10 sessions. Your model will score users at inference that it has **never seen** during training. A teammate's PR uses `KFold(n_splits=5, shuffle=True, random_state=0)` as the CV, scoring ROC-AUC. Chapter 07 says what is wrong and what to use instead?

A. Nothing — `shuffle=True` erases grouping; the CV estimate is
   honest.
B. The wrong split is being *stratified*. Switch to `StratifiedKFold` —
   group leakage is just a class-imbalance issue.
C. Rows from the same user are spread across train and validation folds
   in every iteration. The model learns per-user baselines (typical
   session length, baseline spend) and CV scores it on those same users
   — so the CV estimate memorises each user instead of measuring
   generalisation to *new* users. Use `GroupKFold(n_splits=5)` with
   `groups=X["user_id"]`. The chapter warns the mismatch can inflate the
   reported score by tens of percentage points relative to the deployed
   model's real-world performance.
D. Switch to `TimeSeriesSplit` — every multi-row-per-entity dataset is
   time-ordered.

### 9. (Chapter 07) A PR runs `X_imputed = SimpleImputer().fit_transform(X)` once and then scores the model with `cross_val_score(clf, X_imputed, y, cv=5)`. The CV score looks great; the reviewer says it's optimistic. Chapter 07 agrees — why, and what is the idiomatic fix?

A. `SimpleImputer` is stateless, so nothing leaks; the score is
   trustworthy.
B. The imputer's medians were learned from every row — including the
   rows that will land in each fold's validation slice. Each fold's
   training set then benefits from statistics computed with those
   validation rows participating. The leak is small on a wide dataset
   and large on a narrow one. The fix is to put the imputer inside a
   `Pipeline` so `cross_val_score` refits it inside every training
   fold, with no validation-fold rows contributing to the fitted
   medians.
C. `cross_val_score` already refits every transformer it sees, so no
   fix is needed.
D. Replace `SimpleImputer` with `KNNImputer`; only the KNN variant is
   leakage-safe inside CV.

### 10. (Chapter 08) You will tune a gradient booster over `learning_rate ∈ [1e-3, 0.3]` and `num_leaves ∈ [15, 255]`. A colleague proposes `GridSearchCV` with `learning_rate ∈ np.linspace(0.001, 0.3, 5)`. Chapter 08 flags two issues. Which option names both correctly?

A. Nothing is wrong; `GridSearchCV` is the right tool and `linspace`
   does the right thing.
B. Issue 1: `learning_rate` spans orders of magnitude, so a
   linearly-spaced grid clusters samples at the high end and misses the
   low-end regime where many boosters actually converge best. Sample
   from `scipy.stats.loguniform(1e-3, 0.3)` instead. Issue 2: with two
   or more continuous hyperparameters that genuinely span orders of
   magnitude, `RandomizedSearchCV` (Bergstra & Bengio's result) finds
   better configurations than `GridSearchCV` on the same total number
   of trials, because grid wastes trials varying the hyperparameters
   that don't matter — use random search as the first-pass method and
   reserve grid for 2-D orthogonal refinement.
C. The learning rate should be in a linear grid; the issue is
   `num_leaves` — use `linspace(15, 255, 5)` instead.
D. Tune `n_estimators` in the same grid; without it, the search is
   under-specified.

### 11. (Chapter 09) You fit a `RandomForestClassifier` on a dataset mixing binary flags and high-cardinality continuous features, print `.feature_importances_`, and are about to paste the top-10 into `REPORT.md`. Chapter 09 lists three specific biases of this quantity. Which option names them correctly and identifies the right cross-check?

A. `.feature_importances_` is unbiased and defensible; no cross-check
   needed.
B. It is biased only toward one-hot columns; pre-encode as ordinals and
   the bias disappears.
C. Impurity-based importance is biased toward (i) high-cardinality
   features (more candidate split points = more chances to find a lucky
   split, even on noise), (ii) features that happen to win the top
   split of each tree (that split's impurity reduction is amplified
   because it is credited on the whole dataset), and (iii) any feature
   that overfits — impurity is measured on the *training* set, so a
   feature that memorises noise looks important. The chapter's
   recommended cross-check is **permutation importance on a held-out
   set** (or SHAP). Treat `.feature_importances_` as a smoke test, not
   as a reportable deliverable.
D. The bias only applies to regression random forests; classifiers are
   unaffected.

### 12. (Chapter 10) Which statement about SHAP as chapter 10 presents it is correct?

A. SHAP values prove causality: if `age` has the largest SHAP
   contribution, age *causes* the outcome and dropping it would move
   the real-world metric.
B. SHAP satisfies the **efficiency** axiom —
   `base_value + Σ_j φ_j(x) = model_output(x)` — so each prediction is
   exactly decomposed into a base value (the model's expected
   prediction) plus a per-feature contribution. For tree ensembles,
   **TreeSHAP** computes these contributions exactly in polynomial
   time, which is what makes exact SHAP practical in production; the
   general-purpose Shapley calculation is exponential in the number of
   features and would otherwise be intractable. SHAP faithfully
   explains the model as trained — including a leaky feature that
   dominates the summary, which is itself a useful leakage diagnostic.
C. SHAP solves the correlated-features problem completely: two
   perfectly correlated features each receive the full contribution.
D. TreeSHAP requires a background sample of at least 10 000 rows to be
   numerically stable.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **B.** Chapter 01 ("The bias–variance frame you will keep coming
   back to") walks through each family exactly in these terms: linear =
   high bias / low variance; single deep tree = low bias / enormous
   variance; random forest = averaged high-variance trees (variance
   drops, bias roughly unchanged); gradient boosting = both terms
   controlled via learning rate, tree depth, and `n_estimators`. The
   chapter states that every algorithmic choice in the module is "at
   some level, a choice about the bias–variance trade-off".

2. **C.** Chapter 02's solver table shows `lbfgs` (default) and
   `newton-cholesky` do not support L1 at all; `liblinear` supports L1
   but is listed as the "small, sparse" solver; `saga` is the solver
   row that supports L1 *and* elastic-net *and* scales to large
   `n_samples`. The chapter's rule-of-thumb — "Use `saga` when you
   want L1 or elastic-net and your dataset is bigger than `liblinear`
   can handle" — fits this case exactly. Chapter 02 also notes that
   `StandardScaler(with_mean=False)` is what you use on sparse
   one-hots so centering does not destroy the sparsity; `saga`
   additionally needs standardised features to converge quickly.

3. **C.** Chapter 03 ("The penalised objective") spells out the
   convention flip: "`LogisticRegression` uses `C = 1 / α`. Small `C`
   means strong regularisation, large `C` means weak. This is a common
   source of confusion — Ridge and Lasso classes use `alpha`;
   `LogisticRegression` uses `C`. They mean opposite things." Pitfall
   #2 of the same chapter restates it. Both changes in the PR increase
   regularisation strength by 100×.

4. **C.** Chapter 03 ("Where Lasso hurts", "Coefficient stability:
   bootstrap and refit", and pitfall #3) names this exact failure mode:
   with correlated features, Lasso picks one and zeros the other
   "essentially at random", and the chosen column changes across
   bootstrap samples. The chapter's recommended remedy is a bootstrap
   stability analysis reporting *selection frequency* per feature, with
   the threshold "A feature selected in <50% of bootstraps is a coin
   flip; do not report it as 'selected'". Elastic-net with `l1_ratio`
   around 0.3–0.7 is also named as the standard structural fix when
   you want sparsity but the correlated group should share credit.

5. **B.** Chapter 04 pitfall #1 says: "Growing an unlimited tree 'to
   see how well it fits'. It will fit training data perfectly and
   generalise terribly. Set `max_depth` (or `min_samples_leaf`) up
   front." Chapter 05 ("The scikit-learn interface") then says: "The
   reason `max_depth=None` is safe here — and not in a single tree —
   is that the ensemble average protects you. A single depth-`None`
   tree memorises. Five hundred depth-`None` trees trained on different
   bootstraps memorise *different* rows, and their disagreements cancel
   on average."

6. **C.** Chapter 05 pitfall #2: "Setting `max_features=1.0` on a
   regression `RandomForestRegressor` and being disappointed by the
   accuracy. The regression default is `1.0` (all features), which
   turns off decorrelation and reduces the ensemble to plain bagging.
   Set `max_features="sqrt"` or `max_features=0.33`." The chapter
   contrasts this with the classification default, which is `"sqrt"`.

7. **C.** Chapter 06 pitfall #5: "`fit(..., eval_set=...)` without
   `early_stopping_rounds`. The library trains all `n_estimators`
   rounds; the eval set is only used for reporting. Both have to be
   set." The "Early stopping: how you actually tune `n_estimators`"
   section shows the correct pattern with `callbacks=[early_stopping(50)]`
   for LightGBM.

8. **C.** Chapter 07 ("`GroupKFold`: for grouped rows") describes this
   exact scenario: "If the dataset has multiple rows per entity
   (multiple sessions per user, multiple images per patient, multiple
   invoices per company) and at prediction time the model will see new
   entities, random splitting **cheats**. Rows from the same user
   appear in both train and validation folds; the model memorises
   per-user information; CV overstates its generalisation to new
   users." The chapter quantifies the mismatch: "a random split can
   inflate an accuracy metric by tens of percentage points relative to
   the group-aware split."

9. **B.** Chapter 07 ("Where the pipeline goes") gives this exact
   example and conclusion: "If you `SimpleImputer().fit_transform(X)`
   before `cross_val_score`, the imputer has seen every row's
   mean/median — including the validation rows — and CV overstates
   generalisation. This mistake is almost too obvious to mention until
   you have made it three times." The fix is to put the imputer inside
   a `Pipeline`, which `cross_val_score` refits per fold.

10. **B.** Chapter 08 pitfall #2: "Linear grid over a log-scale
    hyperparameter. `[0.001, 0.01, 0.1, 1, 10]` linearly-spaced misses
    everything in the low end; `loguniform` finds it." And the
    "Random search: the default first pass" section cites
    Bergstra & Bengio's result that random search over the same total
    trial count finds better configurations than grid search because
    most hyperparameters have a long tail of importance — grid wastes
    trials varying the unimportant ones; random search does not.

11. **C.** Chapter 09 ("Method 2: impurity-based importance") lists the
    three biases in these terms: "Bias toward high-cardinality
    features" (more candidate splits, more chances of a lucky split
    even on noise), "Bias toward the 'top of the tree'" (the root-split
    winner is credited on the full dataset), and "Computed on the
    training set" (overfitting features look important). The chapter
    quotes `sklearn`'s user guide to the same effect and recommends
    permutation importance on a held-out set as the cross-check: "Use
    it as a smoke test, not as the deliverable."

12. **B.** Chapter 10 ("Shapley values, in one page") states the
    efficiency axiom explicitly: "The Shapley values sum to
    `v({1, ..., n})`", which for ML is the equation
    `model_prediction(x) = expected_prediction + Σ_j φ_j(x)` in the
    "What SHAP gives you" section. The computational note — "`2^n`
    coalitions per prediction. For any non-trivial `n`, that is
    impossible. TreeSHAP is the algorithm that makes it tractable for
    tree ensembles by exploiting the tree structure to compute exact
    Shapley values in polynomial time" — is why TreeSHAP is the
    95%-of-cases explainer. The "pitfalls that SHAP does not solve"
    section rules out options A (SHAP is not causal), C (correlated
    features still share credit), and the TreeSHAP section rules out
    D (TreeSHAP with the default `tree_path_dependent` perturbation
    does not require a background sample at all; `interventional`
    needs one, but 10 000 rows is not the required size — the
    chapter's rule of thumb for background sets elsewhere is 50–500
    rows).

</details>
