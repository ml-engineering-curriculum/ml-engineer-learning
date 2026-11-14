# quiz-01 — ML Monitoring & Drift Detection

> **Format:** 10 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–07 of this module (every chapter; chapters 02, 03,
> and 06 get two questions because they carry the bulk of the mechanics).
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter and the
> section the point comes from — a missed question is a pointer back to the
> section that teaches it.

---

## Questions

### 1. (Chapter 01) A platform engineer says "the churn service is healthy — `/healthz` returns 200, CPU sits at 30%, p95 latency is 78 ms, zero 5xx." Meanwhile the business metric has dropped 4% because the model has been quietly predicting the majority class on a new corner of feature space. Chapter 01 names (a) the specific gap this leaves, (b) the three axes along which the model's world can shift, and (c) why this cannot be delegated to the platform team. Which option captures all three?

A. There is no gap; a 200 from `/healthz` plus green latency and error-rate
   panels is the whole observability story. Model quality is implicit in those
   signals, and the platform team already owns them.
B. Generic observability (APM, logs, metrics, traces) catches **"the service
   is broken"** — process liveness, request rate, latency, 5xx, pod health —
   but does **not** catch **"the service is fine, but the model is wrong."**
   That gap is **ML monitoring**: measuring what the model believes about
   the world, and noticing when the world stopped agreeing. The three axes
   to distinguish (and *never* conflate in an incident conversation) are
   **data drift** — `P(X)` moves, measurable **without any labels**, covered
   in chapter 03; **concept drift** — `P(y|X)` moves (the inputs look
   identical but the right answer is different, e.g., fraud rings adapting),
   requires ground truth, covered in chapter 04 for detection and chapter 05
   for the label plumbing; and **prediction-distribution shift** — `P(ŷ)`
   moves even though no code changed, the free proxy you watch in real
   time because it is correlated with both of the others but needs no
   labels. "Label shift" (`P(y)` moves, `P(X|y)` stable) is a special case
   that surfaces the same way through a shifted predicted-class ratio.
   Two structural reasons this is **uniquely the ML engineer's job**: only
   you can name the invariants (that `income` was log-normal around $52k;
   that predicted-churn had mean 0.12 — the platform team cannot see those
   from traffic volume), and only you can **act on the signal** (retrain /
   roll back / downweight the new segment / wait and see — an ML judgment
   call, not an SRE decision).
C. "Drifting" is a single thing. One drift score across all features is
   enough; distinguishing data, concept, and prediction shift is pedantic.
D. Ground truth will always arrive quickly enough that measuring accuracy in
   production is the whole monitoring plan — the three axes are academic.

### 2. (Chapter 02) An engineer instruments the service with `ml_predictions_total{user_id="...", request_id="..."}` plus a `prediction` log line that carries the raw request body and an email field. They skip the reference profile ("we'll compute it from the last 7 days of traffic when we need it"). Chapter 02 names specific failures in each of those three choices. Which option captures all three, plus the four surfaces the chapter wants you to think in?

A. All three choices are fine; Prometheus labels are free, logs can contain
   anything, and a rolling 7-day reference is more "current" than the training
   distribution.
B. Three chapter-02 pitfalls, plus the frame: **(1) High-cardinality metric
   labels (`user_id`, `request_id`, `session_id`, `image_url`) blow up series
   count** — the platform team sees it in `prometheus_tsdb_head_series` and
   politely asks you to fix it. Label values must be small enumerable sets:
   `model_version` (dozens), `route`, `class`, `feature_name` — fine;
   per-request IDs — never. Logs handle high cardinality; the metrics system
   does not. **(2) Logging raw feature values / PII by default** is the
   compliance trap — log the *prediction*, not the input; hash feature
   vectors with `features_hash` if you need a replay handle; keep raw
   values in a separate access-controlled offline table if you need them.
   **(3) A reference computed from recent production traffic is not a
   reference** — drift measured against a drifting reference reports zero
   drift right up until the model is unusable (also chapter 03 pitfall #3).
   The reference is the **training-time distribution**, snapshotted once at
   training time, **frozen with the model version**, and **never mutated**
   for a deployed model. It has to include not just summary statistics but
   a **random sample** of ≤ 10 k rows — the sample is what powers the
   multivariate drift checks (domain classifier, MMD) in chapter 03;
   summary stats alone cover univariate drift only. The chapter organises
   the whole surface around **four places**: (i) **Prometheus counters /
   histograms** (low cardinality, aggregate, alertable — the six metrics:
   prediction rate per version, prediction-score histogram, inference
   latency per version, bounded feature histograms for the top ~10 drivers,
   feature-anomaly counters for null / out-of-range / unknown-category,
   and model-loaded / load-time gauges), (ii) **structured log events**
   (per-request, higher cardinality, carrying a stable `prediction_id`,
   `features_hash`, and `reference_profile_id` — the join handle for
   chapter 05), (iii) a **prediction table** in the warehouse (one row per
   prediction, or a sampled subset, that chapter 05's labels table joins
   to), and (iv) the **reference profile** artifact itself. Every metric
   also carries `model_version` so a rolling deploy or A/B split doesn't
   silently muddy the numbers.
C. Compute the reference nightly from the latest 7 days of production; label
   everything with `user_id` so you can per-user debug; log the raw payload
   because support might need it.
D. Prometheus is enough on its own; neither logs nor a warehouse table are
   needed for monitoring.

### 3. (Chapter 03) An engineer wires an alert on `scipy.stats.ks_2samp(...).pvalue < 0.05` across every feature. Within a week the oncall channel has 400 pages and everyone ignores it. They also recompute the reference histogram nightly from the last 7 days of production. Chapter 03 names three specific mistakes here and the correct discipline. Which option captures them?

A. More sensitivity is always better; a p-value under 0.05 is the gold
   standard; sliding the reference gives you a fresher baseline.
B. Three chapter-03 pitfalls, with the fixes: **(1) Alerting on p-values at
   scale** — with production traffic volumes, *every* two-sample test
   becomes "statistically significant" because the sample size pushes any
   nonzero effect past any p-threshold. Use the **effect size** (PSI, the
   KS *statistic* D, chi-squared statistic, Wasserstein, JSD) against a
   **calibrated threshold**, not a p-value. **(2) Uncalibrated thresholds
   copied from a blog post** — the `PSI < 0.1 / 0.1–0.2 / ≥ 0.2` cutoffs
   are credit-scoring folk wisdom, a starting point, not a law. A feature
   that is naturally volatile (a category with 3% mass ± 2%) sits above 0.1
   every day without anything being wrong. The calibration procedure:
   backfill the current-window test on 90 days of historical traffic;
   look at the distribution of the per-feature score; set the threshold
   to catch clear anomalies, not the 99th percentile of normal (a
   reasonable heuristic is `max(1.5 × baseline_p99, absolute_floor)`);
   re-calibrate when the model version changes and again as the world
   moves. **(3) Sliding the reference** is the silent killer — drift
   measured against a drifting reference trends monotonically to zero;
   you find out the model is broken from an accuracy alert weeks late.
   **The reference is training-time and immutable.** Only the current
   window moves. Two more chapter-03 disciplines belong in the fix:
   **bin edges come from the reference** (apply the same edges to both
   reference and current; different edges returns silent nonsense), and
   **add an epsilon** (`1e-6` is standard) to zero proportions before the
   log in PSI (`ln(0)` and `ln(q / 0)` are undefined). Operate two
   windows in parallel — a **fast window** (1 hour / 10 k requests) for
   sudden shifts, a **slow window** (1 day / 7 days) for gradual drift —
   and treat **new categories** as a separate signal (chapter 02's
   `unknown_category` counter) rather than folding them into the same
   test, because chi-squared and PSI are structurally bad at scoring
   categories with reference count zero.
C. Alert on the aggregate drift score across all features — attribution is
   never needed.
D. Chi-squared is safe for continuous features and KS is safe for
   categoricals; the test choice doesn't matter.

### 4. (Chapter 03) A model takes 200 features. Every univariate drift score is under threshold. Overall business metrics are quietly worse. A teammate wants to turn off multivariate drift detection because "the per-feature numbers are quiet so there's nothing more to check." Chapter 03 names the specific case univariate tests miss and a cheap, interpretable detector that catches it. Which option captures both?

A. If every univariate test is quiet, there is no drift. Multivariate
   detectors are redundant.
B. Univariate tests inspect **one feature at a time** and miss the case where
   every marginal looks identical to the reference but the **joint**
   distribution has moved — two features used to be correlated and now are
   not; both marginals are still `N(0, 1)`. For that case use a
   **multivariate detector**. Chapter 03's first choice is the **domain
   classifier** (a.k.a. Classifier Two-Sample Test): train a small
   classifier (LightGBM, `HistGradientBoostingClassifier`, logistic
   regression on standardised features) to distinguish reference samples
   from current samples; report `2 × |AUC − 0.5|` ∈ `[0, 1]` as the drift
   score. If the classifier succeeds (AUC ≫ 0.5), the two distributions are
   distinguishable — drift. If it fails (AUC ≈ 0.5), they are not. The
   best part: the classifier's **feature importances name the features
   that carried the drift**, so you get attribution for free. Two
   complementary tools in the shortlist: **MMD** (Maximum Mean Discrepancy,
   a kernel-based test in a reproducing kernel Hilbert space — well suited
   to embeddings or high-dimensional data where the classifier overfits;
   `alibi-detect`'s `MMDDrift` ships it) and **PCA reconstruction error**
   (project the current window onto the reference PCA basis and measure
   reconstruction error — rising error means the current data lies in a
   subspace the training data did not span; cheap, no extra library, a
   good sanity check). **Do not run all three and average** — pick one
   as primary, calibrate it, and keep the others as confirmatory checks
   in an incident. Pair the multivariate score with per-feature scores so
   attribution survives: **drift on the prediction distribution alone is a
   downstream symptom**; pair it with per-feature drift so you can name
   the upstream cause.
C. Averaging every univariate test gives you the multivariate score for
   free.
D. If the domain classifier shows no drift, no retraining is ever needed
   regardless of accuracy.

### 5. (Chapter 04) A churn model's per-feature PSI is under threshold on every feature, the multivariate drift score is stable, but week over week the model is quietly getting worse. An engineer proposes retraining on the latest 7 days of data because "prediction-distribution shift is moving." Chapter 04 names the shape of the problem, the free signal that catches *something*, and the specific trap the proposed retrain would spring. Which option captures all three?

A. If per-feature drift is quiet, nothing is wrong; retraining weekly is
   always safe.
B. The shape is **concept drift** — `P(y|X)` has moved while `P(X)` is
   unchanged — fraud rings adapting to the detector, user preferences
   shifting after a product change, a regulatory change moving what
   "risky" means. Chapter 03's detectors cannot see it because they never
   look at `y`. Confirming concept drift requires **ground truth**, which
   is usually delayed by days or weeks (chapter 05). The free proxy that
   fires in the meantime is **prediction-distribution shift** on `P(ŷ)`:
   the live `ml_prediction_score` histogram versus the reference prediction
   distribution, compared with PSI / JSD / KS. Why it is worth watching
   independently: it **responds to both data drift and concept drift**,
   so it is the single downstream signal; it is the **only ML-native
   quality signal that is never delayed** (real time, no label wait); and
   it is **business-readable** ("the model is predicting churn 2× as often
   this week" — a sentence product understands). The **trap**: **prediction
   shift is a symptom, not a cause.** Retraining on the drifted window
   alone (chapter 06 pitfall #1, chapter 04's "Prediction shift is a
   symptom, not a cause") can make the model **worse** — e.g., Scenario A
   from chapter 04, where a new geography launches, its users legitimately
   have more "high risk" feature values, data drift is real, concept drift
   is not, the model is right, and retraining on the drifted window
   over-weights the new region and unlearns what worked in the old ones.
   Pair prediction shift with per-feature drift (to attribute cause), with
   a measured accuracy metric once labels mature, and with business KPIs.
   In the label-lag window, use **performance estimation without labels**:
   **CBPE** (Confidence-Based Performance Estimation — expected accuracy
   as a function of the predicted probabilities themselves, for classifiers
   whose probabilities are well-calibrated on the reference set), **DLE**
   (Direct Loss Estimation for regression — a secondary model trained on
   the labelled reference to predict the loss from the primary model's
   features and prediction), and importance-weighted variants. All three
   rest on an assumption (calibration still valid, loss estimator still
   generalises) that **breaks exactly where strong concept drift is
   present** — treat their output as an educated guess, verify when labels
   arrive, and plot **estimated vs measured accuracy** so a divergence
   between the two is itself a signal. Finally, concept drift gets missed
   most often because **aggregate accuracy hides a slice regression**:
   slice the dashboard by cohort, geography, traffic source, and
   value tier; alert on slice regressions, not just aggregate.
C. Prediction shift is always benign; ignore it.
D. CBPE and DLE produce *measured* accuracy and you can treat them as
   equivalent to the real thing.

### 6. (Chapter 05) A new model regression requires joining a prediction row to a label row three weeks later, but the prediction log never carried a stable ID and `label_source` was never written. A teammate says "join on `(user_id, minute)` with a 5-minute tolerance — close enough." Chapter 05 names the join-key discipline, the three sources labels come from (each with its own failure mode), and the specific failures that follow when the discipline is skipped. Which option captures them?

A. `(user_id, minute)` is the correct canonical join; a stable prediction ID
   is nice-to-have. All labels come from one source.
B. The whole loop hinges on one piece of plumbing: **generate a `prediction_id`
   server-side at prediction time** (UUIDv4 / ULID — anything with negligible
   collision probability), **return it in the response** (so the caller logs
   it too), **store it on every downstream business event** (the chargeback,
   the click, the moderation decision — each carries the `prediction_id` of
   the prediction that was shown), and **do not reuse `request_id` or
   `user_id`**: a user makes many requests, a batch request produces many
   predictions, only `prediction_id` is 1:1 with a prediction. When it is
   missing, the fallback is a best-effort join on `(user_id, timestamp)` with
   a tolerance window — it works, it is lossy, and it is a tech-debt entry
   against the next release. Labels come from three places, each with a
   distinct failure mode: **(1) Implicit user feedback** (the user's next
   action — click, dismissal, purchase, return) is cheap and fast
   (seconds–hours) but **biased by exposure** (you only see feedback on
   items you showed — the classic logged-bandit problem) and has **missing
   negatives** ("not clicked" is not "not interested" — the user may not
   have seen it; impressions must be logged alongside clicks, so the label
   is `clicked | impressed`); **(2) Explicit business outcomes** (chargeback,
   default, cancellation, refund, medical outcome) are **slow but
   authoritative** (days–months), well-defined, owned by another team, and
   prone to **label leakage** when a feature that is easy to compute
   *post-hoc* from the outcome is pulled into retraining; **(3) Human
   labelling** (radiologists, moderation, crowdworkers, domain experts) is
   slow, expensive, has a budget, and requires inter-annotator agreement
   plus gold-standard tasks — and the labellers themselves drift when
   guidelines change (version the guidelines; store the guideline version
   on each label; replay a small gold-set against new labellers to detect
   drift). Real systems combine all three and inherit all three failure
   modes. Beyond the obvious ones, name five more by name because they are
   the most common silent killers of accuracy metrics: **selection bias**
   (labelling only the subset the model flagged — only labels above
   threshold → you never observe false negatives; fix by labelling a
   random sample or weighting by propensity), **delay bias** (reporting
   current-week accuracy before the maturity window closes — the negative
   class is underrepresented because not all "no chargeback" events have
   arrived yet; the fix is cohort methodology — only report a cohort's
   accuracy once its maturity window has closed), **labeler drift** itself
   (the model looks regressed because the definition of the right answer
   moved), **ghost predictions** (a prediction was made but never shown
   — the fallback path took over, an experiment excluded the user, the
   request timed out; log an `outcome` ∈ {shown, fallback, error} and
   filter), and **post-hoc features in the labels table** (any field not
   available at prediction time leaks into retraining; chapter 06 makes
   this a guardrail). The output is the **`predictions_labeled` warehouse
   table** — primary key `prediction_id`, nullable `label` for immature
   cohorts, enumerable `label_source`, always-populated `weight` (inverse
   sampling probability so stratified samples give unbiased estimates;
   default 1.0), **UPSERTs** for late-arriving labels (not INSERTs), and
   **same warehouse as the business data** (cross-system joins are
   nightmares). **Stratified sampling** makes a small labelling budget
   buy the most information: over-sample the tails of the prediction
   distribution (confident predictions where a labeller disagreeing is
   high-signal), the uncertain middle (where the model is closest to
   wrong), and slices that chapter 04 flagged — under-sample the
   confidently-normal bulk; random sampling on a heavily imbalanced
   label wastes the budget on easy majority-class rows.
C. Any join key works as long as you add enough tolerance; stratified
   sampling is for later.
D. Labels always arrive immediately; the labels table never needs a
   cohort dimension.

### 7. (Chapter 06) A drift alert fires. A teammate wires the trigger to invoke the training pipeline on any multivariate-drift reading above the threshold, with no debounce, no cooldown, and the evaluation gate's baseline set to "the recent week." Chapter 06 names the anatomy of a trigger, the anti-noise machinery, the gate's load-bearing rule, and the one action that wins over every other signal. Which option captures them?

A. A trigger is one component: "retrain if drift > threshold." Debounce and
   cooldown are decorative; the evaluation baseline doesn't matter so long as
   there is one.
B. Every trigger has **three parts**: (1) a **signal** (drift score,
   prediction shift, estimated performance, measured performance, scheduled
   date, ad-hoc human decision — the output of chapters 03–05), (2) a
   **decision** (a small, tested, auditable pure function that reads
   `MonitoringState` and returns one of `DO_NOTHING | INVESTIGATE | RETRAIN
   | ROLLBACK`, with debounce, hysteresis, cooldown, and priority baked
   in), and (3) an **action** (the training pipeline, owned by `mod-106`,
   which the trigger hands a well-scoped input, not a model). The decision
   rule must be **written down, reviewable, testable, and auditable** —
   every firing writes an audit row to a `retraining_decisions` table
   capturing the state + reason + rule version; a trigger that fires and
   nobody knows why is chapter 06 pitfall #5. Three flavours layer well:
   **scheduled** (calendar-driven, "retrain every N days", a floor and
   cheap insurance), **event-driven** (signal-driven, for fast domains),
   **human-in-the-loop** (reviewer decides, for expensive-to-evaluate
   retrains or novel drift) — mature systems stack all three. The
   **anti-noise machinery is non-negotiable**: **debounce** — require the
   signal to be present in N consecutive windows before firing (one hour
   of drift is not a retrain; three consecutive days probably is);
   **hysteresis** — threshold to *enter* the alerting state is higher than
   the threshold to *exit* (prevents flapping when the signal wobbles
   around a single threshold); **cooldown** — after a retrain, no new
   retrain for M days (or until the next labelled cohort matures), so the
   trigger doesn't fire on the new model's cold-start noise. Without these
   a busy monitoring system retrains every day and never converges — and
   the self-reinforcing bad-retrain loop (retrain on drifted window → new
   model is worse → next drift check fires again → repeat) chews through
   compute while making the model worse. **Rollback is a first-class
   action and wins over every other signal**: if measured accuracy
   collapses below the previous version's accuracy, roll back regardless
   of other signals — retraining cannot happen faster than a rollback.
   The **evaluation gate** — every retrain candidate passes **(i) a stable
   pre-drift holdout** (the critical rule: the baseline must **predate the
   drifted window**; if the baseline is also a recent model, both are
   drifting, the gate becomes useless, regressions ship — chapter 06
   pitfall #3), (ii) a backtest on the recent-window cohort (the new model
   should actually do better on the shift that triggered it), (iii)
   **slice checks** on chapter 04's slice list (a candidate cannot regress
   a protected or high-value slice below threshold even if aggregate
   improves — pitfall #4), (iv) shadow / canary at low traffic, and
   (v) a **kill switch**. Regression on any step blocks promotion; a
   trigger fires on **drift alone** only to `INVESTIGATE`, not `RETRAIN`,
   because retraining on drift without corroborating accuracy drop or
   slice signal can make things worse (chapter 06 pitfall #1).
C. Retraining on any drift event is correct; the evaluation gate is
   paperwork.
D. If the trigger ever fires `ROLLBACK`, something is wrong with the
   trigger — real production systems only retrain.

### 8. (Chapter 06) A candidate model retrained on the drifted window beats the recent-week baseline on aggregate accuracy by 2 points, so the engineer promotes it. Three weeks later, enterprise-slice accuracy is down 18%. Chapter 06 names the specific baseline rule the engineer violated, the specific gate step they skipped, and the retraining-data discipline that would have prevented the regression. Which option captures all three?

A. "Beats the recent baseline on aggregate" is the whole evaluation; slice
   checks are paperwork; training on the most recent 7 days is always the
   right window.
B. Three chapter-06 disciplines, violated: **(1) The evaluation gate's
   baseline must predate the drifted window.** Comparing the candidate
   against the **recent week** is pitfall #3 ("Evaluation gate uses the
   recent window as the baseline"): if both the incumbent and the
   candidate are drifting, the gate's comparison measures nothing useful
   and the regression ships. The baseline is a **stable, frozen, versioned
   pre-drift holdout**. **(2) Slice checks are mandatory, not aggregate
   checks** (pitfall #4 "No slice metrics in the gate" plus the "silent
   retraining regression" failure mode): aggregate improves by a couple of
   points; a small-but-valuable slice regresses hard; nobody notices for
   weeks because the aggregate metric is fine. Every retrain is compared
   on chapter 04's slice list (cohort, geography, traffic source, value
   tier) — the candidate cannot regress a protected or high-value slice
   below a configured threshold even if the aggregate is better. **(3)
   Sampling for the retrain** is the trap door: retraining on only the
   most-recent-7-days of a drifting distribution **forgets everything the
   previous six months knew** — the training set has to span **a window
   long enough to include both pre-drift and post-drift**, and it has to
   be **stratified by cohort** so the training distribution still looks
   like production (over-sampling the newly-drifted segment is a modelling
   choice, not a default). For fast-moving domains a longer window with a
   **recency-weighted loss** (`sample_weight` in sklearn / LightGBM /
   XGBoost) is often better than a shorter window — more data plus
   recency bias beats less data. Also implied here: the trigger passes a
   **well-scoped input** to the training pipeline — `training_data_window`
   (dates), `reference_holdout_id` (the frozen holdout to compare
   against), `incumbent_model_version`, `trigger_reason` (human + machine
   readable), `decision_rule_version` (git SHA of the rule that produced
   it), `request_id`. Related failure modes to recognise by name: the
   **"we retrained but nothing happened"** case — the trigger fires, the
   model retrains, the drift alert is still firing a week later, which
   usually means the drift was in a feature the model uses only weakly or
   the labels on the drifted segment are themselves drifted (chapter 05's
   labeler drift) — the retrain was not the right fix. And: **a retrain
   that doesn't go through the registry** (side-loading a `.pkl` into the
   serving directory) is a rollback nightmare — `mod-106`'s registry is
   the system of record.
C. Aggregate on a short recent window is the correct baseline; slices are
   second-order concerns.
D. The trigger's job includes picking the model architecture and adding
   new features — it is a whole-project kick-off.

### 9. (Chapter 07) A SEV-3 fires at 09:12: multivariate drift 0.34, three univariate features flagged, prediction-distribution shift 0.11, no accuracy alert (labels still maturing). The engineer's first instinct is "kick off a retrain." Chapter 07 names the correct procedure, the default hypothesis, the specific response to the drift-plus-no-accuracy-alert shape, and the artifact that closes the loop. Which option captures them?

A. Kick off the retrain immediately; drift alerts always mean the model is
   wrong. Write the post-mortem later if there is time.
B. **Retraining is almost never the first response.** Chapter 07's **oncall
   mindset** has three habits: **(a) assume nothing about the model until
   you can name an evidence trail** ("the model regressed" is a claim, not
   a diagnosis — what signal, over what window, compared to what
   baseline?); **(b) follow the data upstream first, the model second**
   (most regressions start with a data change, not a model change — the
   feature-anomaly counter and univariate drilldown are the first stops);
   **(c) every incident action generates a log line** (the audit trail is
   not the oncall's 03:14 memory). The **eight-step procedure** turns the
   alert into a fix: (1) **Confirm the signal is real** — rule out drift-
   job clock issues, missing prediction-table partitions, a rolling deploy
   mid-window that mixed `model_version` values against a single-version
   reference, a corrupted reference profile (budget 5–10 minutes; longer
   means your monitoring metadata isn't clear enough — a chapter 02
   follow-up, not an incident action); (2) **Localise the drift** — walk
   the flagged features in PSI order, asking *what* changed (new category,
   existing category growing, whole distribution shifted) and *when* (a
   sharp discontinuity on a specific day is a code change or data-pipeline
   event; a gradual slope is a real distributional shift); (3) **Follow
   the data upstream** — map the flagged feature to its upstream table via
   the feature store's lineage metadata, check whether that table changed
   across the drift boundary, and check whether anyone was told (a launch
   that was on a changelog saves the hour of working this out from the
   data — this is the canonical chapter-07 walkthrough of a product
   launching `enterprise-plus` while the feature transform silently maps
   unknown categories to `unknown_category` and the chapter-02
   `unknown_category` counter fires quietly); (4) **Decide the action**
   using chapter 06's rule — the schema-change case is **not a retrain
   target yet**: retraining on a broken feature would **embed the bug**;
   the immediate action is a **data-pipeline fix** (add the value to the
   schema, backfill the last N days, page the data-engineering owner),
   with the retrain scheduled once the feature is correctly available and
   the training set has been backfilled; softer legitimate distributional
   shifts (a country ramp, a marketing campaign) go on the **next
   scheduled retrain**, not an emergency one, and get an audit row with
   `trigger_reason = "drift.<features> — deferred to scheduled"`; (5)
   **Mitigate quickly** with a **serving-time guardrail** (for the
   schema-change case: return the reference-average prediction for the
   closest existing category with a warning header, instrument a counter
   for "requests served by the guardrail," alert the platform team that a
   non-model fallback is live, open the ticket to remove it in the same
   commit — mitigation is temporary tech debt from the first hour); (6)
   **Communicate** — short structured updates to the oncall channel and
   to downstream consumers, dated and honest (silence between step 3 and
   the post-mortem is the single most common incident failure mode); (7)
   **Follow the fix through** — verify the feature-anomaly counter drops
   to zero, verify drift scores drop below threshold, verify predicted
   churn recovers, and **only then remove the guardrail** in its own
   commit referencing the incident ticket (if any of those don't happen,
   the fix wasn't the fix); (8) the **post-mortem**, in a durable template
   whose action items are the only mandatory outcome — **a prose
   post-mortem with no owned, dated action items is not a post-mortem.**
   Six other common shapes to recognise by their first-order signature: a
   **new segment of users** (univariate drift on cohort / geography with
   no anomaly counters — real traffic, retrain if the segment is
   important), a **model rollout regression** (prediction shift
   correlated with `model_version` and no data drift — compare v_new vs
   v_old via chapter 06's shadow / canary and roll back), a **latent
   preprocessing bug** (prediction distribution collapses to a specific
   value — look at the model's raw input vector, not the feature value),
   **delayed labels from a different source** (step change in the
   accuracy line that's not the model — check `label_source`
   distribution), **a retrain that made things worse** (`mod-106`'s
   rollback), and **labeler drift** (chapter 05's gold-set replay saves
   the day). What you owe the next oncall: a **runbook link on every
   alert**, a **dashboard the alert deep-links to**, a **`NOTES.md` in
   the model's repo** (not a wiki — in the repo, next to the code), and
   a **quarterly game-day** that simulates a regression and walks the
   current oncall through it — the first time you run through the
   procedure is not during a real incident.
C. The data-first discipline is optional; model-first is faster.
D. If no accuracy alert has fired, nothing is wrong and the drift alert
   can be silenced.

### 10. (Cross-chapter: 02 + 03 + 07) A reviewer looks at a drift pipeline and asks "where does the reference come from, how are its bin edges picked, and what happens when the model is re-versioned?" Chapter 02 writes the reference profile; chapter 03 computes drift against it; chapter 07 drills into it during incidents. Which option most faithfully strings those three points together?

A. The reference is "yesterday"; bin edges come from today's data; new model
   versions re-use the old reference.
B. The reference profile is a **chapter 02 deliverable** written **once at
   training time**, **frozen with the model artifact** (same version, same
   hash), and **never mutated** for a deployed model. Its content is
   non-negotiable even if the file format is: per-feature summaries
   (`dtype`, `range`, quantiles, `mean` / `std`, category frequencies),
   the **training prediction distribution** (mean / std / quantiles,
   class balance), and a **sample of ≤ 10 k training rows** (the sample is
   what powers multivariate drift checks — summary stats alone cover
   univariate drift only). Every prediction log line stamps the
   `reference_profile_id` it was measured against, so when a profile
   rotates (a new training run) old alerts don't retroactively change
   meaning. Chapter 03's drift job **reads** this file: **bin edges are
   computed once from the reference** and the same edges are applied to
   both reference and current — different edges return silent nonsense
   (chapter 03 pitfall #4). Zero bins get an epsilon (`1e-6`) before the
   log in PSI (chapter 03 pitfall #5). The **reference does not slide**:
   the current window moves; the reference is immutable (chapter 03
   pitfall #3 — a sliding reference reports zero drift monotonically right
   up until the model is unusable). When the model is **re-versioned**,
   you write a **new** reference, calibrate thresholds afresh (chapter 03
   calibration step #4), and at rollout time compute drift **between**
   profiles — not within a drifted deployed profile. Chapter 07's step 1
   checklist uses this discipline: during an incident, **sanity-check the
   reference profile's mean and quantiles** against your memory of the
   training set; a corrupted or overwritten reference produces phantom
   drift, and the first 5–10 minutes rule it out before any root-causing
   starts. Related chapter-02 disciplines: do not emit a histogram per
   feature per version (with 200 features × 5 concurrent versions × bucket
   cardinality, series count explodes — bounded feature set only, top ~10
   drivers from SHAP in `mod-106`); include a warm-up prediction on startup
   so the first live request doesn't spike the latency histogram; use
   extra buckets at the extremes of the prediction histogram so fine tail
   drift ("0.91 vs 0.99") stays visible — a saturated top bucket hides it.
C. The reference's bin edges come from the current window; the profile
   updates nightly from production; and when you re-version the model you
   keep the old reference so "nothing moves."
D. The reference is a convenience; chapter 03 can compute drift against
   whichever window is cheapest to recompute.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **B.** Chapter 01 ("Motivation" states the "service is broken vs. model
   is wrong" split verbatim; "Three kinds of shift you have to distinguish"
   names `P(X)` / `P(y|X)` / `P(ŷ)` and the label-shift aside; "Why this is
   uniquely the ML engineer's job" is the two structural reasons —
   invariant-naming and action-authority — that block delegation to
   platform). The opening paragraph's example (service green, model
   quietly wrong, business metric down 4%) is the chapter's framing scene.

2. **B.** Chapter 02 ("The four surfaces" lists metrics, logs, prediction
   table, reference profile; "Metrics that pay for themselves" lists the
   six — prediction rate, prediction score histogram, inference latency,
   bounded feature histograms, feature-anomaly counter, model-loaded and
   model-load-time gauges — all labelled by `model_version`; "The reference
   profile — the artifact 'normal' is measured against" states "freeze the
   profile with the model version. Never mutate the profile of a deployed
   model" and "Do not use the last N days of production as the reference";
   "Cardinality — the recurring pitfall" lists `user_id` / `request_id` /
   `session_id` / `image_url` as explicitly disallowed and names
   `prometheus_tsdb_head_series` as the platform team's early warning; "The
   prediction log — what to write per request" specifies the
   `prediction_id` / `features_hash` / `reference_profile_id` extensions
   and "Do not log raw feature values by default. PII lives there").

3. **B.** Chapter 03 ("Thresholds — calibrate, don't guess" states "Alerting
   on statistical significance (a p-value below 0.05) will page you every
   hour once traffic is large. Use effect size … with a calibrated
   threshold, always." The four-step calibration procedure — backfill
   current-window test, look at the distribution, set threshold above the
   99th percentile of normal, re-calibrate per model version — is spelled
   out; the `max(baseline_99p × 1.5, absolute_floor)` heuristic is in the
   same section. "Windows — what 'current' means" states "**Do not** slide
   the reference. The reference is the training distribution, immutable."
   "Univariate tests" covers the KS-statistic-not-p-value warning and the
   bin-edges-from-reference rule; "Population Stability Index (PSI) — the
   workhorse" shows the `np.clip(p, 1e-6, None)` epsilon line in the
   reference implementation. Pitfalls #1 (p-values), #3 (sliding
   reference), #4 (different bin edges), #5 (zero-bin `ln(0)`), and #7
   (uncalibrated thresholds) are each explicit in the "Common pitfalls"
   list).

4. **B.** Chapter 03 ("Multivariate drift — when univariate is not enough"
   opens with the exact "every marginal looks identical but the joint has
   moved" framing; "Domain classifier (a.k.a. 'Classifier Two-Sample
   Test')" gives the LightGBM pseudocode with `drift_score = 2 * abs(AUC
   − 0.5)` and names feature importances as the attribution mechanism;
   the "Which multivariate detector — a shortlist" section ranks domain
   classifier first, MMD second (for embeddings / high-dimensional data),
   PCA reconstruction third as a cheap sanity check, and instructs **Do
   not run all three and average. Pick one and calibrate it**. Pitfall #10
   is "Drift on the prediction distribution alone. Prediction drift is a
   downstream symptom. Pair it with per-feature drift so you can attribute
   cause").

5. **B.** Chapter 04 ("Three axes of shift, one more time" is the vocabulary
   recap; "Prediction-distribution shift — the free proxy signal" names
   both reasons it is worth watching — responds to both drift types, never
   delayed, business-readable — and the exact "Prediction shift is a
   symptom, not a cause" section; "Where prediction shift and concept
   drift diverge" names Scenario A (new geography, legitimate shift,
   retraining hurts) and Scenario B (fraud patterns evolve, features
   stable, accuracy collapses only visible with labels); "Performance
   estimation without labels" explains CBPE for calibrated classifiers,
   DLE for regression, and importance-weighted variants, and warns their
   assumptions break under strong concept drift — "Treat their output as
   an educated guess, not measured accuracy, and always recompute measured
   accuracy the moment labels arrive"; "Segment monitoring — the aggregate
   hides regressions" names cohort / geography / traffic source / value
   tier). Pitfalls #2 ("Assuming no data drift ⇒ no concept drift"), #3
   ("Alerting on aggregate accuracy only"), and #4 ("Treating a
   performance estimate as measured accuracy") are explicit in the
   chapter.

6. **B.** Chapter 05 ("The join key — the whole loop hinges on this"
   states the `prediction_id` discipline verbatim, including "Generate
   the ID at prediction time", "Return it in the response", "Store it on
   the business event", "Do not reuse `request_id` or `user_id`", and the
   `(user_id, timestamp)` fallback as explicit tech debt; "The three
   sources of labels" is implicit feedback / explicit business outcomes /
   human labelling with each failure mode; "The failure modes" names
   selection bias, delay bias, labeler drift, missing negatives, label
   leakage, and ghost predictions explicitly; "The consumer contract" is
   the `predictions_labeled` table schema with `prediction_id` as PK,
   nullable `label`, enumerable `label_source`, always-populated `weight`,
   UPSERTs for late-arriving labels, same warehouse as business data;
   "Stratified sampling — the labeling budget's best friend" is the
   over-sample-tails / over-sample-uncertain-middle / over-sample-flagged-
   slices / under-sample-confidently-normal pattern with the `weight = 1
   / sampling_probability` formula).

7. **B.** Chapter 06 ("The trigger has three parts" is signal / decision /
   action; "Trigger flavours" is scheduled / event-driven / human-in-the-
   loop with the layering note; "The decision rule — the actual logic"
   shows the `decide_action` pseudocode with rollback-wins-over-everything
   as the first branch; "Debounce, hysteresis, cooldown" is the anti-noise
   section verbatim; "Guardrails — the 'did the retrain make things
   better?' gate" is the five-step gate — stable-holdout backtest,
   recent-window backtest, slice checks, shadow/canary, kill switch — and
   the explicit **The gate is not optional**; "The retraining loop can go
   wrong" names the self-reinforcing bad-retrain loop and the "silent
   retraining regression"). Pitfalls #1 ("Retraining on drift alone"),
   #2 ("No cooldown"), #3 ("Evaluation gate uses the recent window as the
   baseline"), #4 ("No slice metrics in the gate"), #5 ("The trigger
   fires; nobody knows why"), and #8 ("No rollback plan") are each
   explicit in the chapter.

8. **B.** Chapter 06 ("The retraining loop can go wrong" names the "silent"
   retraining regression — slice regresses over many retrains while
   aggregate holds; "Guardrails" step 1 is "Backtest on a stable holdout
   … that predates the drifted window" — pitfall #3's "Evaluation gate
   uses the recent window as the baseline" is the explicit violation; step
   3 is "Slice checks … The candidate cannot regress any protected or
   high-value slice below a threshold, even if aggregate looks better" —
   pitfall #4; "Sampling for the retrain — the trap door" states the
   long-window / cohort-stratified / recency-weighted rule and warns that
   a retrain window of only the last 7 days of a drifting distribution
   "forgets everything the previous 6 months knew"; "The retraining
   pipeline — what the trigger hands off" lists the six fields —
   `training_data_window`, `reference_holdout_id`, `incumbent_model_
   version`, `trigger_reason`, `decision_rule_version`, `request_id`).
   Pitfall #7 ("A retrain that doesn't go through the registry") is also
   relevant.

9. **B.** Chapter 07 ("The oncall mindset" lists the three habits —
   evidence trail, data-upstream-first, every-action-logs; "The alert"
   through "Step 8 — The post-mortem" walks the eight-step procedure
   verbatim including the time budgets, the "`enterprise-plus` /
   `unknown_category`" canonical walkthrough, the serving-time guardrail
   mitigation shape, and the dated-action-items post-mortem template; "A
   gallery of common regressions" lists the seven shapes — upstream
   schema change, new segment, model rollout regression, latent
   preprocessing bug, delayed labels from a different source, retrain
   that made things worse, labeler drift; "What you owe the next oncall"
   names the runbook link, dashboard deep-link, `NOTES.md` in the repo,
   and the quarterly game-day). Pitfalls #1 ("Retraining as the first
   response"), #2 ("Reasoning about the model in isolation"), #3 (not
   distinguishing "drift is real" from "drift is a bug"), #4 ("Skipping
   step 1"), #5 ("No mitigation while the fix is in flight"), #6 ("No
   post-mortem action items"), #9 ("Silence"), and #10 ("No rehearsal")
   are each named.

10. **B.** Chapter 02's "The reference profile — the artifact 'normal' is
    measured against" names the four disciplines — include a sample,
    `reference_profile_id`, freeze with the model version, do not use
    production as the reference. Chapter 03's "Univariate tests"
    (bin-edges from the reference), "Common pitfalls" #3 (sliding
    reference), #4 (different bin edges), #5 (zero-bin `ln(0)` with
    epsilon) each carry over the discipline. Chapter 07's "Step 1 —
    Confirm the signal is real" explicitly checks "Is the reference
    profile correct? Sanity-check the mean and quantiles against your
    memory of the training set; a corrupted or overwritten reference
    produces phantom drift." Chapter 03's calibration step #4 ("Re-
    calibrate when the model version changes") and chapter 06's
    "Guardrails" step 1 (shadow/canary + per-version reference) tie off
    the re-versioning story. Chapter 02's "Common pitfalls" #2 (feature
    histogram per feature per version), #6 (saturated tail buckets), #7
    (no warm-up prediction) also back the surrounding disciplines named
    in the answer.

</details>
