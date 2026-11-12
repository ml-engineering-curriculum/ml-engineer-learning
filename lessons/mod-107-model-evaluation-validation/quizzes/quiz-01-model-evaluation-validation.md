# quiz-01 — Model Evaluation & Validation

> **Format:** 11 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–11 of this module (one question per chapter).
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter the point
> comes from — go re-read that section if you missed the question.

---

## Questions

### 1. (Chapter 01) A teammate says "we hit 0.83 F1 on the held-out set — ship it." Chapter 01 names the four things that must be right for a reported number to be trustworthy, and insists that the "test set" and the "validation set" are *not* the same thing. Which option most faithfully reproduces both claims?

A. The four requirements are the hardware, the random seed, the batch size,
   and the number of epochs; "test" and "validation" are synonyms and the
   choice between them is purely stylistic.
B. The four requirements are **the split** (genuinely unseen, no leakage
   through time / groups / preprocessing), **the metric** (captures the
   decision the model drives), **the slice breakdown** (worst slice reported
   alongside the aggregate), and **the baseline comparison** (strong enough
   that the result is meaningful). A **test set** is touched *once*, at the
   end, to report the number; a **validation set** (or the folds inside CV)
   is touched *repeatedly* during development for hyperparameter choice /
   early stopping / threshold tuning. By the time you have tuned against a
   validation set fifty times it is no longer unseen, which is why the
   module distinguishes the two throughout.
C. "Test" and "validation" are the same thing; one number is enough; miss
   any of the four and you can still defend the result because `model.score`
   is fine by default.
D. The four requirements are accuracy, precision, recall, and F1; a "test
   set" is just whatever you pass to `.score()` last.

### 2. (Chapter 02) A product manager wants a stakeholder-readable error for a house-price regressor. A junior engineer reports R² = 0.78 and nothing else. Which option best captures chapter 02's advice — which number to lead with, which baseline to publish alongside it, and which pitfall the R²-only report walks into?

A. R² alone is fine; it is unitless so no companion metric is needed.
B. Report **RMSE** or **MAE** in the units of the target (dollars) as the
   primary number — stakeholders cannot act on "0.78 of the variance". MAE
   is median-optimal and robust to outliers; RMSE is mean-optimal and
   sensitive to large errors. Always report a companion metric from a
   different family, and compare against the **MAE-optimal constant**
   (median of `y_train`) or **MSE-optimal constant** (mean of `y_train`) —
   "predict the median" is the dumbest baseline for MAE, "predict the mean"
   for RMSE. The pitfall the R²-only report walks into: R² on test is
   computed against *test-set* variance, so a test set with narrower spread
   than training will produce a lower R² even when RMSE is fine.
C. Switch to MAPE; it is always safe regardless of the target distribution.
D. Report training R²; it always increases with more features so it is a
   more reliable comparison.

### 3. (Chapter 03) A fraud model on a 0.1% positive-class dataset reports accuracy = 0.999 and ROC-AUC = 0.98. The deployment review asks what the model actually catches. Which option best matches chapter 03's diagnosis — why accuracy and ROC-AUC alone are misleading here, and what to report instead?

A. Accuracy of 0.999 proves the model is excellent; no further metrics are
   needed.
B. On a 99.9%-negative dataset the "always predict negative" baseline hits
   99.9% accuracy — so the headline number is uninformative. ROC-AUC is
   also unreliable on heavy imbalance because its false-positive-rate
   denominator is the entire negative class, which is huge; a large
   absolute increase in false positives barely moves the curve, so ROC-AUC
   can be 0.98 while precision is 0.05. The imbalance-appropriate
   threshold-independent metric is **PR-AUC** (average precision), whose
   baseline is approximately the positive-class base rate (~0.001 here) —
   a meaningful reference. At an operating threshold, report **precision**,
   **recall**, and **F₁** (or F_β with β > 1 when missing fraud costs more
   than flagging an innocent transaction), and state the threshold
   explicitly; the threshold must be chosen on validation, never on test.
C. Switch to micro-averaged F1 across both classes; that is always
   imbalance-robust.
D. The problem is only the metric name — rename "accuracy" to "quality
   score" and ship.

### 4. (Chapter 04) Two classifiers both reach ROC-AUC 0.90 on the same task. Model A's scores cluster between 0.2 and 0.6; Model B's score of 0.7 is right 70% of the time. The scores will be aggregated into an expected-loss calculation. Which option most faithfully explains chapter 04's distinction and its advice about *when* to calibrate?

A. ROC-AUC 0.90 guarantees the scores are probabilities; either model is
   fine.
B. **Ranking / discrimination** (ROC-AUC, PR-AUC) and **calibration** ("when
   the model says 0.7, it is right 70% of the time") are distinct. A well-
   calibrated model must rank well, but a well-ranking model can be
   miscalibrated. Diagnose with a **reliability diagram** (sort predictions
   into bins and plot empirical fraction positive vs mean predicted
   probability — the diagonal is perfect). Summarise with **ECE** (weighted
   average bin gap) and **Brier** (mean squared error between the predicted
   probability and the outcome — a proper scoring rule that decomposes into
   reliability + resolution + uncertainty). *Calibrate* when the score
   feeds a downstream expected-value calculation, is displayed to a user
   as a probability, is aggregated across a population, or when training
   used class resampling / reweighting; *skip* calibration when the score
   is only used for ranking or when the threshold is tuned on validation
   (which absorbs smooth monotone miscalibration). Fit the recalibrator on
   a **held-out calibration set** the base model did not see — scikit-learn's
   `CalibratedClassifierCV` handles the train/calib split by internal CV.
C. Report ROC-AUC alone and move on; calibration is only an academic
   concern.
D. Train on log loss once; that eliminates all possible calibration issues
   regardless of model family.

### 5. (Chapter 05) A search team is evaluating a reranker. Only the top-10 results are shown; relevance labels are graded (0–4); multiple items per query are relevant. A junior engineer proposes reporting ROC-AUC on a flattened (query, item) binary classification. Which option most accurately reflects chapter 05's framing and the correct metric choice?

A. ROC-AUC is perfect here; ranking metrics are redundant.
B. The task's output is a **ranked list per query**, so ranking metrics are
   correct and classification metrics answer the wrong question (the test:
   is the model producing a list per query, or a score per row?). With
   **graded** relevance and position-sensitive user behaviour, **nDCG@k**
   is the workhorse: `DCG@k = Σ rel_i / log₂(i+1)` (or the `2^{rel_i} − 1`
   numerator convention, which is scikit-learn's default), normalised by
   the ideal DCG. Position 1 has weight 1; position 5 has weight ≈ 2.58;
   position 10 has weight ≈ 3.46 — top-of-list dominates without a hard
   cut. Alternatives for context: **MAP** assumes binary relevance and
   binarises graded signal (lossy here); **MRR** is for "find one answer
   and stop" tasks (wrong here, multiple items relevant); **precision@k /
   recall@k / hit@k** are count-based and ignore order *within* the top-k.
   Always name `k` when reporting; hold out **queries** (not items) for
   evaluation — a query that appeared in training is group-leakage, which
   is chapter 08's subject.
C. MAP is always better than nDCG because it has a simpler formula.
D. Report precision@1 only; since the UI shows ten results, only the first
   matters.

### 6. (Chapter 06) A colleague runs `StandardScaler().fit_transform(X)` on the full dataset, then does a 5-fold `KFold(shuffle=True)` inside `cross_val_score` and reports the mean. Chapter 06 names five leakage families and a canonical fix. Which option both identifies the leakage kind and applies chapter 06's remedy?

A. Nothing is wrong; `cross_val_score` always handles preprocessing
   correctly.
B. The error is **preprocessing leakage** — the scaler was fit on the full
   dataset *before* splitting, so each validation fold's mean and variance
   contributed to the scaling parameters, and the model saw a transformer
   that already "knew" the test set. The universal fix is to wrap every
   preprocessing step with learned parameters (`StandardScaler`, `PCA`,
   `SelectKBest`, `SimpleImputer`, target encoder, feature selector) inside
   a scikit-learn `Pipeline` or `ColumnTransformer`; `cross_val_score` on
   a `Pipeline` refits the transformer on each training fold and applies
   the fitted transformer to the validation fold, which is the paved road.
   This is one of five leakage families in chapter 06 (the others are
   **target leakage** — a feature causally downstream of the label;
   **train/test contamination** — duplicates or trivially derivable rows on
   both sides; **group leakage** — correlated rows per entity across folds,
   which is chapter 08's territory; and **temporal leakage** — training on
   rows from after the validation window, direct or via
   future-contaminated features, which is chapter 07's territory). Standard
   k-fold is a variance-reduction tool, not a leakage cure — it catches
   none of these on its own.
C. Switch to `shuffle=False`; that fixes it.
D. The error is the metric choice, not the preprocessing.

### 7. (Chapter 07) A forecasting team predicts whether a customer will churn in the next 30 days, using daily observations. They run `TimeSeriesSplit(n_splits=5)` with default arguments on a time-sorted frame and are pleased with the metric. Chapter 07 names the specific leak this configuration still allows, and the two scikit-learn knobs that close it. Which option captures both?

A. `TimeSeriesSplit` with defaults is always correct; no further knobs are
   needed.
B. The leak is **target-lag leakage**: the label at time `t` is defined by
   observing the customer through `t + 30`, so a training row at `t` is
   implicitly using data from the future that the validation fold is trying
   to predict. The fix is `TimeSeriesSplit(gap=30)` — scikit-learn's **gap**
   argument skips that many rows between the end of the training set and
   the start of the test set, matching the maximum label horizon. The
   second knob is **`max_train_size`**: `None` is the default (expanding
   window — each fold's training set grows); an integer caps it (rolling
   window). Pick the one that matches deployment. Three other guards in
   chapter 07 pair with the gap: **the DataFrame must be sorted by time**
   before splitting (`TimeSeriesSplit` uses row order, not timestamp —
   unsorted data produces meaningless folds silently); **features must be
   time-safe** (rolling means, target encodings, and lagged features must
   be computed using only data available at each row's observation
   timestamp, not fit on the whole series); and **each fold should span at
   least one full seasonal cycle** so no fold's validation window is a
   seasonally biased slice.
C. The fix is to shuffle the time-series first with
   `KFold(shuffle=True)` — that is faster.
D. Set `gap=1` always; the label horizon is irrelevant.

### 8. (Chapter 08) A medical team splits an imaging dataset with `KFold(shuffle=True)` and gets F1 = 0.93. They repeat with `GroupKFold(groups=patient_id)` and F1 drops to 0.74. They also want to tune hyperparameters with a grid search inside `cross_val_score` and report a single number. Which option best explains both phenomena from chapter 08?

A. The lower score is a bug in `GroupKFold`; report the higher number.
B. The gap is **group leakage**: with random k-fold, multiple scans of the
   same patient land on both sides of the split, so the model memorises the
   patient's anatomy rather than learning the disease; the metric reports
   "how well can we memorise a patient?" and the deployment reports a much
   worse number on a new patient. `GroupKFold` guarantees no group appears
   in two folds (fold sizes become approximate because groups are variable-
   sized; it is not stratified by default — use **`StratifiedGroupKFold`**
   when class balance also matters). Pick the group to match the
   **deployment's generalisation target** (patient if deployment sees new
   patients; batch if correlation is within a manufacturing batch; use
   `LeaveOneGroupOut` for small-K leave-one-entity-out evaluation).
   Separately, "tune with CV and report with CV on the same loop" is the
   canonical **selection bias** failure — the outer loop's validation sets
   are consumed by the inner search, so the reported number includes
   selection noise. The fix is **nested CV**: an inner `GridSearchCV` for
   hyperparameters and an outer `cross_val_score` for reporting, with
   independent random states; the outer number is the honest expected score
   of the *tuning procedure*, not of a fixed hyperparameter setting. Cost
   is `outer × inner × grid` fits. Combine with group splits by passing
   `groups` to the outer loop *and* threading it to the inner search (via
   `fit_params={"groups": groups}` in current scikit-learn, or a wrapper) —
   passing `groups` only to the outer leaves the inner search leaking.
C. Nested CV exists only for academic papers and should never be used in
   industry.
D. Use `LeaveOneOut` on patient rows; it is equivalent to `GroupKFold`.

### 9. (Chapter 09) A deep learning team submits a model with F1 = 0.72 on a 5%-positive-class dataset and asks for merge approval. Chapter 09 would send the PR back with specific questions. Which option best matches chapter 09's reporting discipline and the two independent purposes baselines serve?

A. F1 = 0.72 is enough; baselines are optional polish for academic papers.
B. Baselines serve two purposes: a **sanity floor** (a trivial baseline the
   model must beat — otherwise the model is worse than doing nothing; this
   catches broken pipelines, reversed classes, and target-encoding bugs)
   and a **fair comparison** (a strong-classical baseline the model is
   proposed to *improve on* — a well-tuned logistic regression, a well-
   tuned GBDT such as `HistGradientBoostingClassifier` / LightGBM / XGBoost
   with a modest sweep on the same features). On tabular data the strong
   baseline is specifically a tuned GBDT; on non-tabular (image/text) the
   strong baseline is a **pre-trained model with a linear head**. The
   module's canonical report publishes **four rows**: trivial (majority /
   stratified / `DummyClassifier`), strong-classical (tuned logreg and
   GBDT), the proposed model, and — where applicable — a **human / oracle
   upper bound** from inter-annotator agreement (Cohen's κ / Fleiss's κ).
   The gap between trivial and strong-classical is "what does a competent
   classical solution give us?"; the gap between strong-classical and the
   proposed model is "what does the increment buy?". For time-series,
   the **persistence baseline** (`ŷ[t] = y[t-1]`) is mandatory; for
   retrieval / recommendation, the **popularity baseline** is mandatory.
   `DummyClassifier`'s `strategies` are `"most_frequent"`, `"prior"`,
   `"stratified"`, `"uniform"`, and `"constant"`. Log baselines *in the
   same MLflow run* as the model so a run whose `model_f1 < baseline_f1`
   is visibly broken.
C. Run `DummyClassifier(strategy="uniform")` only; that is the only
   baseline worth reporting.
D. Baselines belong in a separate document never shown to reviewers.

### 10. (Chapter 10) A classifier reports F1 = 0.78 on validation and a reviewer asks "where does it fail?". The engineer replies with "everywhere, 22% of predictions are wrong". Chapter 10 names five moves that turn an aggregate metric into actionable claims. Which option both names the five moves and captures the discipline chapter 10 asks you to apply?

A. Report a lower aggregate metric; that is the only honest response.
B. The five moves, in order of effort: **(1) confusion matrix** — read the
   off-diagonal for concentrated-vs-diffuse confusion and asymmetric errors
   (A→B more often than B→A points at class imbalance or asymmetric
   features; `classification_report` gives per-class precision / recall /
   F1); **(2) sort by per-row loss and look at the top 20** — for
   classification compute per-row negative log likelihood of the true
   class, for regression the absolute residual, for ranking the inverse
   rank of the first relevant item (this is the highest-leverage read —
   the top 20 are "the model's autobiography" and surface label errors,
   systematic feature gaps, genuinely hard examples, and recognisable
   patterns); **(3) per-slice metrics** across categorical features,
   bucketed numerical features, time, and two-feature interactions, with
   an explicit `min_n` guard whose exclusions you *state*; **(4) cluster
   the errors** (KMeans / HDBSCAN on the error rows' features, or UMAP on
   a learned embedding for text / image data — clusters are hypotheses,
   not causes; verify by reading examples); **(5) compare error sets
   across models** — the `only_A` / `only_B` / `both` partition exposes
   regressions the aggregate smooths away. The output is a ranked list of
   hypotheses, each with a slice, the current metric, the population
   metric, and an estimate of **volume × gap** (small noisy slices are not
   where the metric moves — a 3000-row slice at 10% error moves aggregate
   more than a 30-row slice at 40% error). Sample from each of three
   prediction groups — correct-with-high-confidence, wrong-with-high-
   confidence, and uncertain near the threshold — in proportion to
   operational impact; analyse *both* training and validation; save the
   analysis as an MLflow artefact so the next model can diff its error set
   against yours.
C. Cherry-pick ten failure examples and present them as "the problems".
D. Cluster labels are ground truth; never verify by reading individual
   examples.

### 11. (Chapter 11) A product review asks: "aggregate ROC-AUC is 0.867 — is the model fit to deploy?" Chapter 11 describes the level-20 report that answers this question, and spells out what the report is *not*. Which option both captures the three-dimensional slice report and the honest scope limits?

A. The aggregate is enough; subgroup analysis only matters for regulated
   industries.
B. Report **three dimensions**, each as a slice (chapter 10 mechanics):
   **(1) subgroup performance** — the primary metric per subgroup for
   sensitive attributes (gender, race, age — check the deployment
   jurisdiction's legal regime), proxy attributes (postcode, browser agent),
   and operational subgroups (new vs returning, mobile vs desktop), with a
   stated `min_n` and explicit exclusions; **(2) perturbation robustness**
   — the metric under small, task-preserving perturbations (tabular:
   Gaussian noise at σ = 0.05 / 0.10 / 0.20 × feature std, categorical
   flips at 1–10%; text: synonym / typo / casing — `nlpaug`, `TextAttack`;
   image: rotations / brightness-contrast / JPEG / occlusion —
   `albumentations`, `imgaug`; report clean and perturbed separately, the
   drop, and the exact perturbation seed and intensity so a reader can
   reproduce); **(3) distribution shift** — evaluate on data from a
   plausibly different distribution the deployment will actually see
   (time shift: hold out the latest quarter; geographic shift; version /
   device shift; rare-event shift via up-sampled positives), with one
   diagnostic (**PSI** — > 0.25 is the credit-modelling "large shift"
   heuristic — or the **KS statistic**) that quantifies the shift in the
   *data*, not the model. Format the result as a **slice table**: aggregate
   at the top, then slices ordered by `|Δ|` descending so the worst slice
   is in the first two rows; publish the aggregate *and* the worst slice —
   the gap is the deployment risk. Pick between fairness definitions
   (**demographic parity** — equal positive-prediction rate across groups;
   **equalised odds** — equal TPR and FPR across groups) deliberately and
   document which — Barocas, Hardt & Narayanan show they cannot in general
   be satisfied simultaneously, so you choose per the deployment's ethical
   / legal duty. What this is *not*: a full fairness audit (needs domain
   ethicists and legal review); adversarial robustness (FGSM / PGD / certified
   defences — a different research area; chapter 11's perturbations are
   naturally occurring, not adversarial); a stress test against shifts you
   cannot construct (the honest response for those is a `mod-109` monitoring
   plan); and not a technical go / no-go — the report triggers a stakeholder
   conversation about whether the surfaced trade-off is acceptable.
C. The report is a go / no-go decision made by the ML engineer alone.
D. Adversarial FGSM / PGD perturbations are the level-20 default.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **B.** Chapter 01 ("The evaluation loop" and "The single distinction to
   internalise early") lists the four conditions — split, metric, slice
   breakdown, baseline — and spells out that miss any of them and "the
   number is theatre". The same chapter explicitly defines the test set as
   "touched *once*, at the end" and the validation set / CV folds as
   "touched *repeatedly* during model development", and says "every result
   reported in this module distinguishes" the two.

2. **B.** Chapter 02 ("MAE — the honest default", "MSE and RMSE — the
   outlier-sensitive pair", "R² — the unitless 'variance explained'", and
   the "which one do I report?" decision tree). The chapter names MAE as
   median-optimal and RMSE as mean-optimal, says "never report R² alone",
   and flags the test-set-variance pitfall under "When it misleads". The
   "Report two. Always." rule is in the decision tree.

3. **B.** Chapter 03 ("Accuracy — and why it fails", "ROC-AUC — threshold-
   independent, imbalance-sensitive", "PR-AUC — the imbalanced-classes
   companion", and the "which metric do I report?" decision tree). The
   dumb-baseline accuracy point is in "Accuracy — and why it fails"; the
   ROC-AUC-vs-imbalance warning is in "When ROC-AUC misleads"; the PR-AUC
   base-rate reference is the "baseline PR-AUC ≈ base rate" line; the
   threshold-on-validation rule is "Do not pick the threshold on the test
   set".

4. **B.** Chapter 04 ("What 'calibrated' means", "The reliability diagram",
   "Expected Calibration Error", "Brier score", "Why models are
   miscalibrated by default", "Recalibration methods", "When calibration
   is (and isn't) worth doing"). The two-dimensions framing ("Ranking /
   discrimination" vs "Calibration") is stated verbatim. The "held-out
   calibration set" rule and the `CalibratedClassifierCV` CV-internal
   split are in "Recalibration methods". "Skip calibration when the score
   is only used for ranking" is a bullet in "When calibration is (and
   isn't) worth doing".

5. **B.** Chapter 05 ("The setup: queries, items, and relevance", "nDCG"
   and its log-discount weights, "When to prefer classification metrics
   over ranking metrics", and "Common pitfalls"). The "is the model
   producing a list per query, or a score per row?" test is the chapter's
   named decision. The nDCG weights at positions 1 / 5 / 10 are stated
   directly. "Evaluating on the training queries" is pitfall #6, which
   links to chapter 08 as the group-leakage chapter.

6. **B.** Chapter 06 ("The leakage taxonomy" — all five families — and
   "Preprocessing leakage — the pipeline saw the test set"). The chapter's
   remedy is the universal `Pipeline` / `ColumnTransformer` wrap, with the
   explicit statement that `cross_val_score` on a `Pipeline` does the right
   thing automatically. "Standard k-fold is a variance-reduction tool, not
   a leakage cure" is in the chapter's motivation.

7. **B.** Chapter 07 ("Scikit-learn's `TimeSeriesSplit`", "Gap periods
   and target-lag leakage"). The chapter states "insert a gap between
   training and testing equal to the maximum label horizon" and documents
   both `max_train_size` (expanding vs rolling) and `gap`. The "sort by
   time before splitting" rule is in "`TimeSeriesSplit`"; the time-safe-
   features rule and seasonality guard are in their own sections.

8. **B.** Chapter 08 ("Group-based splitting", "`GroupKFold` — the fix",
   "Nested cross-validation: the problem", "The nested pattern", and
   "Combining nested CV with group / time splits"). The patient-anatomy
   memorisation example is the chapter's opening illustration. The
   `StratifiedGroupKFold` recommendation for classification with groups
   is in the same section. The nested-CV selection-bias explanation is
   the "WRONG — inner CV picks hyperparameters" code block; the
   `fit_params={"groups": groups}` thread-through is in "Combining nested
   CV with group / time splits".

9. **B.** Chapter 09 ("The trivial baselines", "The strong baselines",
   "When to build a task-specific baseline", "Reporting baselines
   alongside the model", "Human performance as an oracle", "Baselines
   within the training loop"). The two-purpose framing (sanity floor vs
   fair comparison) is the chapter's opening distinction. The four-row
   table is the "template for the report". The persistence / popularity
   baselines are mandatory for time-series / retrieval respectively per
   "The trivial baselines". `DummyClassifier`'s strategies list is in
   "Classification". The MLflow logging pattern is in "Baselines within
   the training loop".

10. **B.** Chapter 10 ("The five moves" and the sub-sections for each, "A
    worked pattern", "From analysis to iteration", "Common pitfalls"). The
    "top 20 errors are the model's autobiography" line is chapter 10's
    direct phrasing. The `volume × gap` ranking rule is step 4 of "From
    analysis to iteration" and pitfall #2 of "Common pitfalls". The
    three prediction groups — correct-high-confidence, wrong-high-
    confidence, uncertain-near-threshold — are "The one-slide summary".
    The "analyse both training and validation" rule is in "A worked
    pattern". The "save the analysis as an artefact" rule is pitfall #5.

11. **B.** Chapter 11 ("The three dimensions", "Subgroup performance",
    "Perturbation robustness", "Distribution shift", "When one number is
    not enough — reporting a slice table", "What this chapter is not",
    "Fairness metrics with a caution", "Common pitfalls"). The three-
    dimension structure, the perturbation library suggestions, the PSI
    threshold rule-of-thumb, and the "worst slice at the top of the table"
    ordering are all stated in the chapter. The Barocas-Hardt-Narayanan
    incompatibility citation is the "both definitions have their critics"
    paragraph. "What this chapter is not" lists the full-audit / adversarial /
    monitoring exclusions verbatim. Pitfall #7 ("Ignoring the worst slice
    because the aggregate is fine") is the one-line form of the deployment-
    risk point.

</details>
