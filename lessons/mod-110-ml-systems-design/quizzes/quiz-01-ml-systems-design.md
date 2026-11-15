# quiz-01 — ML Systems Design

> **Format:** 11 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–07 of this module (every chapter; chapters 01, 03,
> 05, and 06 get two questions because they carry the bulk of the mechanics
> and vocabulary).
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter and the
> section the point comes from — a missed question is a pointer back to the
> section that teaches it.

---

## Questions

### 1. (Chapter 01) A stakeholder asks for "an AI to reduce churn." Six weeks later a data scientist demos a model with AUC 0.83 and nobody knows what to do with a prediction, the retention team has no intake path, and the label turned out to be last quarter's `is_active` ETL flag. Chapter 01 names (a) the question that should have been asked first, (b) the three things that must be nailed about the target, and (c) the specific trap the team walked into by letting the warehouse decide the label. Which option captures all three?

A. "AUC 0.83" is the whole story; a retention team not being set up to act on
   predictions is somebody else's problem; the `is_active` flag is as good a
   label as any because it's already in the warehouse.
B. Chapter 01 says the first question is not "what does the model predict?"
   but **"what does the business do differently because of the prediction?"**
   — a churn model nobody acts on has zero value regardless of AUC, and the
   business decision (automated action vs ranking vs human-in-the-loop
   routing vs suggestion vs internal analytics) sets the bar for accuracy,
   latency, explainability, and monitoring. The three things to nail about
   the target are the **unit of prediction** (user? session? (user, item) at
   a timestamp? subscription?), the **prediction horizon** ("churn *when*?"
   — 7, 30, 90 days, end-of-billing-cycle, each a different label and
   different wait-time to know whether the model was right), and the
   **label definition in plain English a non-ML colleague could read**
   ("cancelled subscription" vs "downgraded plan" vs "zero activity for 14
   days" vs "explicit unsubscribe" — each a different target and a
   different pipeline). The specific trap: **letting the label be whatever
   the data warehouse already has.** The `is_active` flag got picked up
   because it was convenient; six months in, somebody notices the flag is
   set by a monthly ETL and the model is predicting churn for people who
   obviously churned two weeks ago. Discipline: define the label to fit
   the business decision, then build or buy the pipeline — never the
   other way around.
C. The first question is always "XGBoost or transformer"; the unit and
   horizon don't matter if the model is accurate enough; the label can be
   whatever the warehouse returns because the ML team doesn't own data
   engineering.
D. "AI to reduce churn" is a precise-enough brief; any one-page blueprint
   is bureaucratic overhead; label definition is a data-engineering
   problem, not an ML-engineering problem.

### 2. (Chapter 01) A proposal for a new personalisation model picks online serving "because it sounds modern," skips the non-ML baseline "because we have XGBoost in mind," calls the cold-start plan "we'll see what data we have," and omits the non-goals section "to stay flexible." Chapter 01 names (a) the default system shape and when to deviate, (b) the three jobs the non-ML baseline is doing even before any model ships, (c) what a missing cold-start plan actually guarantees, and (d) why the non-goals section is load-bearing. Which captures all four?

A. Online serving is always the right default because latency; baselines are
   a courtesy; cold-start plans are optional; non-goals are bureaucracy.
B. Four chapter-01 disciplines. **(a)** The default service shape is
   **batch** — "the model runs on a schedule against a warehouse table and
   writes results back to the warehouse" — and streaming / online /
   embedded each get a paragraph of defense in the blueprint. Most models
   the business asks for do not need online serving; the trap is picking
   online "because it sounds modern" when a 60-minute-delayed batch would
   produce the same business outcome. **(b)** Every ML project ships a
   **non-ML baseline first**, not as a courtesy but as a discipline: it
   **anchors the accuracy conversation** (AUC 0.83 sounds great until a
   one-line heuristic hits 0.79), it **de-risks the pipeline** (data pipes,
   feature transforms, serving path, and monitoring surface get built
   against a working system, not a green-field one), and it **generates
   labels for the model** (the baseline's outcomes are the model's next
   training set — this is the cold-start plan's hinge). **(c)** No
   cold-start plan is not a neutral choice: it means the model will be
   trained on whatever data leaks in by accident, which is usually the
   worst possible split. The four named options are *ship-without-ML-first*
   (rules-based gathers labels for next quarter's model), *bootstrap-from-
   adjacent-data* (US churn transferred to UK launch with a plan to
   retrain), *human-labelers-in-the-loop-from-day-one* (expensive;
   build labelling ops before the model), and *LLM zero-shot or few-shot*
   (cheap and fast for text, not permanent, chapter 06 names the
   delegation). **(d)** Non-goals prevent scope creep *and* give the
   design-review audience an easy way to redirect the discussion when
   someone asks "have you considered adding X?" — "yes, and we explicitly
   chose not to; here is why." A blueprint without non-goals has a scope
   that drifts by the third meeting and a launch date that slips by two
   months.
C. Batch is for toy problems; the baseline is nice if time permits; cold
   start handles itself; non-goals are the design doc's filler section.
D. Any of the four shapes is fine; baselines, cold-start plans, and
   non-goals are all optional optimisations.

### 3. (Chapter 02) A reviewer looks at a design slide that shows only a "model" box and a "database" box. The team says "feature store? we read from the warehouse." "Model registry? the `.joblib` lives in an S3 bucket." "Labels loop? we'll sort it out later." Chapter 02 names (a) the nine components every production ML system has even at laptop scale, (b) the two symmetric halves of the feature pipeline that *must* produce the same output, (c) what "the same model" actually is (the piece almost everyone draws as one box but is actually five artifacts that move together), and (d) the two antipatterns in "`.joblib` in S3" and "we'll figure out the labels loop later." Which captures all four?

A. The nine components are a textbook abstraction; a laptop-scale system
   only needs a model and a database; "the model" is just `model.joblib`;
   missing a labels loop is fine as long as predictions get logged.
B. Four chapter-02 disciplines. **(a)** Every production ML system is a
   variation of the same diagram: **source systems**, **feature pipeline**,
   **feature store** (offline + online halves), **training pipeline**,
   **experiment tracker + model registry**, **serving path** (one of
   batch / streaming / online / embedded from chapter 01), **predictions
   table + prediction log**, **observability surface** (the `mod-109`
   deliverable), **labels loop**, and the **orchestrator** that schedules
   the pipelines. All nine exist even if a component is a 20-line script
   instead of a hosted service; naming them in a design review spots the
   missing one before it becomes an incident. **(b)** The feature
   pipeline's two halves — **batch feature computation** and **online
   feature computation** — must produce *identical* outputs given the
   same inputs; they are the training-serving-skew battlefield (chapter
   03). The pattern that prevents most skew is **writing the transform
   once** in a shared library both callers import (or using a feature
   store built around this idea). **(c)** "The model" in production is
   **not just the `.joblib` file**; it is the file **plus the featureiser
   code that produced its inputs plus the reference profile that defines
   'normal' plus the calibration cutoff that turns a probability into a
   decision plus the version label the registry resolves to**. Change
   any of the five without the others knowing and you have shipped a
   different model without a version bump. The registry entry records
   all five together; the serving path resolves them together; the
   monitoring surface labels every metric with `model_version` so you
   can attribute post-hoc metrics to *the exact combination* that was
   serving. **(d)** Pitfalls #3 and #4: a **model registry that is
   `model.joblib` in an S3 bucket** is fine for a solo project and
   catastrophic for a team — no version, no promotion, no rollback;
   **predictions logged but not persisted** means the log ages out at 7
   days while labels arrive at 30, and you never compute accuracy.
   Build the predictions table first; every other analysis presupposes
   it.
C. The nine components are a nice-to-have; the batch-and-online-feature
   halves can safely diverge "a little"; "the model" is a single file;
   skipping the labels loop is fine.
D. The experiment tracker and the model registry are the same thing and
   either can be skipped; the orchestrator is somebody else's problem;
   the observability surface is optional for small models.

### 4. (Chapter 02) A proposal to build a brand-new "real-time personalisation" service centres on streaming features into an online KV store, re-deriving features at request time in Go "because the training path is in Python," and reading the predictions from the service's logs with a 7-day retention. Chapter 02 names (a) the service shape to default to and the paragraph that justifies deviating from it, (b) the two ways you can "not have a feature store" and which one breaks on time-window features, (c) the predictions-table discipline the proposal is skipping, and (d) the component whose silent failure makes the whole diagram age out of date. Which captures them?

A. "Real-time" is always the right default; feature stores are optional;
   the predictions log is sufficient; the orchestrator doesn't matter.
B. Four chapter-02 disciplines. **(a)** Default to **batch** — "the
   cheapest possible production ML system … fits more use cases than
   teams tend to admit." Streaming / online / embedded each get a
   defended paragraph in the blueprint: streaming is picked only when
   decisions cascade in near-real-time *and* a 60-minute-delayed batch
   would not produce the same business outcome; online is picked only
   when the prediction is *part* of the user request. **(b)** If your
   system doesn't have a feature store, it has one of two implicit
   versions: a **warehouse playing both roles with the online path
   reading through some cache** (fine for batch and low-QPS online
   systems; breaks at high QPS), or a **serving-time feature
   computation where the service re-derives features on every request**
   (fine for *stateless* features; **catastrophic for time-window
   features** — you're going to reimplement the offline aggregation and
   it will diverge). Rewriting feature code in a second language on the
   serving path is chapter 03 cause #1 in the making. **(c)** The
   **predictions table** in the warehouse is **one row per prediction
   keyed by `prediction_id`**, retained per the data policy (30 / 90 /
   365 days). The **prediction log** in the log stream carries the same
   content with different retention and feeds real-time dashboards.
   Without both, you cannot compare `v_new` against `v_old` on the same
   requests, attribute a business outcome to a prediction (the chapter
   05 A/B test), debug "why did the model say that for this user?", or
   compute accuracy once labels mature. A 7-day log that times out
   before labels arrive is pitfall #4 ("predictions logged but not
   persisted"). **(d)** The **orchestrator** (Airflow, Dagster, Prefect)
   is "not glamorous; unusually load-bearing" — when it quietly fails to
   schedule the retraining job the model silently ages out of date;
   when it fails to run the labels loop the accuracy metric silently
   reports the last successful run's numbers forever. Alerts on job
   success/failure and freshness are non-negotiable (pitfall #7).
C. Streaming is always better because it's faster; the serving service
   should always re-derive features in its own language for latency;
   a 7-day log is enough.
D. Feature stores are a vendor sales pitch; the predictions table
   duplicates the log; the orchestrator is automatically monitored by
   the platform team.

### 5. (Chapter 03) The offline eval set says AUC 0.86. Ten days into a 10% ramp, business metrics are flat and the intervention list looks wrong. A feature's distribution is nearly identical for both populations, but a different feature is 0 for 40% of production requests and was 0 for 3% of training rows. A teammate says "this is drift, let's retrain." Chapter 03 names (a) what training-serving skew actually is (and the three things it is NOT), (b) the four canonical causes every skew incident collapses into, (c) why the right first action is not a retrain, and (d) the single highest-value defence the chapter identifies. Which captures them?

A. Skew and drift are the same thing; retraining fixes both; there is only
   one cause and one fix.
B. Four chapter-03 disciplines. **(a)** Training-serving skew is **any
   systematic difference between the feature values a model sees during
   training and the feature values it sees during inference, holding the
   underlying data-generating process fixed**. It is **not data drift**
   (`mod-109` ch 03 — the *underlying distribution of inputs* has shifted;
   drift is a change in the world, skew is a bug in your code), **not
   concept drift** (`P(y|X)` has moved — the right answer changed), and
   **not label leakage** (the training path illegally sees the label; same
   code path in training and serving, but a bug in the training path).
   Skew is a divergence *between* paths; leakage is a bug in one path.
   **(b)** Every skew incident is one of four canonical causes: **(1) two
   implementations of the same transform** (the serving path reimplements
   the training transform in its own language / framework and the two
   *almost* agree), **(2) the time-window feature computed differently**
   (point-in-time incorrectness, backfill vs streaming aggregation,
   feature-freshness staleness), **(3) preprocessing that is fit in
   training and not applied at serving** (the fitted `StandardScaler` /
   `OneHotEncoder` / target encoder / embedding vocab was never
   serialised with the artifact), and **(4) feature source drift** (two
   warehouse tables that are usually the same drift out of sync at the
   day-of-launch of a new plan; or training pulled from a snapshot and
   serving pulls live). **(c)** The right first action is not a retrain
   — a retrain would re-embed the bug: if the serving path is reading
   the feature from the wrong source or computing it with the wrong
   point-in-time semantics, the next training set inherits the divergence
   and the next model is trained on serving-shaped garbage. The fix is
   in the **code**, and until the code is fixed a serving-time guardrail
   (chapter 07 step 5) mitigates. **(d)** The chapter's **single
   highest-value defence** is the **CI row-parity test on a canonical
   fixture**: 20–50 rows of realistic input (including nulls, unusual
   categories, boundary values), run through both the training-time
   transform (loaded from the artifact) and the serving-time transform
   (as it will run in production), with `np.allclose(..., atol=1e-9)`
   or byte-identical outputs required and a failing test blocking
   promotion. "A warning in a log gets ignored." Prevention beats
   detection but detection is the backstop; the pair of the row-parity
   test (promotion-time) and the feature-anomaly counter (continuous in
   production) is the belt and the braces.
C. Skew is a kind of drift; one retrain fixes both; row-parity tests are
   paperwork.
D. If the offline AUC was good, skew is impossible — the online
   underperformance must be concept drift.

### 6. (Chapter 03) A team wires a shadow-mode step that compares aggregate prediction distributions over an hour and passes. After launch, a specific user segment gets markedly worse predictions for four months before a business-metric drilldown surfaces it. The team's fixture-parity test has `atol=1e-3` and uses synthetic rows with no nulls or unusual categories. Chapter 03 names (a) why aggregate-only shadow comparison is weaker than per-request parity, (b) the specific weaknesses in their fixture, (c) how a lazy cached serving-time feature can sit under green metrics, and (d) the operational discipline that prevents "we fixed it this time but future skew is on its own." Which captures them?

A. Aggregate distribution comparison is enough; fixtures don't need edge
   cases; cached features are automatically safe; row-parity tests are a
   one-time investment.
B. Four chapter-03 disciplines. **(a)** A **shadow rollout that only
   compares aggregate distributions** is pitfall #4: two very different
   models can produce similar aggregate distributions. The **stronger
   check** is **per-request feature parity** on a sampled subset: for
   every request, log the feature vector each model actually saw; if the
   two vectors differ for the same request, one of the two paths is
   wrong (usually the candidate; not always). The pair of comparisons —
   aggregate prediction-distribution (KS / JSD) plus per-request feature
   parity — is what shadow mode actually measures. **(b)** Pitfall #5
   ("Fixture-parity tests that use a synthetic fixture with no edge
   cases") plus the row-parity discipline: **the bug is in the edge
   case**. The fixture must cover rows with **nulls, unusual categories,
   boundary values** (day boundaries, timezone edges, missing optional
   fields) — the places skew hides. Loosening `atol` from `1e-9` to
   `1e-3` is pitfall-adjacent (a tempting shortcut that hides the next
   bug); the real fix is to make the two paths byte-identical and keep
   the tolerance strict. **(c)** Pitfall #9 ("Serving-time features
   computed lazily and cached with a wrong TTL. The value is stale in
   unpredictable ways and every request looks fine") plus the "overnight
   refresh that lied" war story: the serving path was reading an 11-day
   stale `activity_last_7d` from an online store whose refresh job had
   been quietly failing, while training used a fresh value. The fix is
   an `updated_at` column both the serving path and the freshness monitor
   read, a warning header on predictions served with stale features, and
   the online-store freshness SLO on the alerting surface. **(d)** The
   **skew audit checklist** — nine items including "for every feature,
   is there exactly one code path that computes it?", "for every fitted
   transformer, is it part of the artifact and loaded with the model?",
   "for every time-window feature, is the point-in-time semantics
   identical between training and serving?", "for every feature, is the
   source table named in the registry entry?", "does CI include a
   row-parity test on a fixture with null / edge / boundary cases?",
   "is a shadow-mode rollout planned before the first real traffic?",
   "is `ml_feature_anomaly_total` alerting in production?" The audit
   runs at every promotion — any "no" or "I'm not sure" is work to do
   before launch. "Skew is patient. It sits under green metrics for
   weeks. Design assuming it exists somewhere and build the tests that
   will find it."
C. Aggregate distribution similarity is a sufficient shadow-mode check;
   fixtures work best with normal data; stale features are a platform
   problem, not an ML problem.
D. Skew only happens in the serving path; training-time transforms are
   automatically correct; the row-parity test is a nice-to-have.

### 7. (Chapter 04) A recommender shipped last year is quietly getting worse. Offline metrics keep improving. CTR holds. Catalogue coverage has dropped from 62% to 18% and new-item ramp time has risen from 2 weeks to 6 weeks. A teammate says "nothing broke, no feature drifted — nothing to fix." Chapter 04 names (a) why feedback loops are inherent to production ML (not an optional failure mode), (b) the five vocabulary terms that distinguish the loop types, (c) the specific trap of optimising a metric the loop inflates, and (d) the two architectural choices that determine the loop's direction before any mitigation. Which captures them?

A. If nothing broke and the per-feature drift is quiet, there is nothing to
   fix; CTR holding is proof the system is working; loops are an academic
   concern.
B. Four chapter-04 disciplines. **(a)** For most business-critical models
   three facts hold: the **model's output is an action** (recommendation,
   approval, ranking, price — the action changes what the user sees or
   does), the **label is measured downstream of the action** (you cannot
   observe "would this user have clicked if we had shown them the other
   thing" — only the click on the thing you showed), and the **next
   training set comes from measured behaviour under the current model's
   actions**. Combine the three and every retrain trains on data the
   previous model shaped — **the loop is the diagram** from chapter 02
   (predictions table → labels loop → training pipeline). You can't get
   rid of the loop; you can design so it runs in the direction you want.
   **(b)** Fix the terms first: **feedback loop** is the neutral general
   phenomenon; a **data flywheel** is a *virtuous* loop (Netflix,
   Spotify, every search engine — more data → better model → better
   product → more use → more data, and incumbents get farther ahead);
   **selection bias** is the training data not being representative
   because the model itself decided who / what got observed; **position
   bias** is the special case for ranking (items in higher positions get
   more clicks regardless of relevance — training on click data
   attributes the clicks to the items instead of the positions); a
   **filter bubble / echo chamber** is the personalisation degenerate
   case (the model reinforces existing preferences; the user's observed
   preferences shrink to what the model shows them); a **runaway loop**
   is the degenerate case where a small bias amplifies — common in
   demand-driven pricing, in content moderation, in credit-risk
   exclusion, and in the predictive-policing case study ("the loop is
   not measuring underlying crime; it is measuring police presence").
   The structural shape to recognise: **a scoring system whose action
   determines whether the label is even measurable is a runaway loop
   by default.** **(c)** Pitfall #2 ("Optimising the metric the loop
   inflates"): CTR on a recommendation that is over-served is not
   evidence the recommendation is good — it is evidence the loop is
   tightening. The symptoms are exactly what the question describes:
   catalogue coverage falling, entropy of the recommendation
   distribution collapsing, new-item ramp time rising, while aggregate
   metrics stay fine. **(d)** Two architectural decisions bake the
   loop's direction in on day one: **how labels are collected**
   (*passive-observational* — loop is on, exploration and debiasing
   mandatory; *randomised-treatment* — loop calibrated by construction;
   *externally-sourced* from human labellers or a downstream
   ground-truth system — loop broken by construction, expensive but
   reliable) and **where the model sits in the decision chain** (*the
   model is the decision* — tightest loop, every mitigation mandatory;
   *the model is a filter into a human review* — looser; *the model is
   a suggestion* — loosest). The tighter the model is in the loop, the
   more mitigation infrastructure is required.
C. Loops are a fringe concern only for Netflix and Google; small teams
   don't have them; CTR is the only metric that matters.
D. The right first action is retraining on the latest N days; if drift
   detectors are quiet, the loop is definitionally not a problem.

### 8. (Chapter 04) A credit-risk model's offline metrics look great. The team does not keep an untreated cohort, has no exploration budget, and reports only aggregate accuracy. Chapter 04 names (a) the four detectors to design in from day one (not after the fact), (b) the four mitigations to weave into the architecture, (c) the specific dashboard crime of "a recommender whose only diversity metric is accuracy@K," and (d) the one-paragraph loop analysis that belongs in every blueprint. Which captures them?

A. Untreated cohorts are unethical by default; exploration is wasteful;
   aggregate accuracy is the whole picture; loop analyses are academic.
B. Four chapter-04 disciplines. **(a)** Loops don't fire alerts — they
   erode metrics slowly and invisibly. Design four detectors in from
   day one: **(1) diversity metrics** for any ranking or recommendation
   system — **catalogue coverage** (fraction of items recommended at
   least once per week), **entropy of the recommendation distribution**
   (falling entropy means the recommender is collapsing onto a small
   set), and **new-item ramp time** (rising ramp time means the loop is
   closing on incumbents); **(2) an untreated control cohort** — for
   any scoring / intervention system permanently hold back 1–10%
   (depending on cost and ethics) that is **not** subject to the
   model's action, so you have the closest thing to ground truth for
   the population as a whole, both as a training-label gold set and as
   the counterfactual the observability surface compares treated vs
   untreated against over time; **(3) slice performance over time**,
   not just at snapshots — a slice whose performance drops consistently
   over quarters while aggregate stays flat is the fingerprint of a
   loop; for personalisation systems add **novelty rate per user** and
   a **serendipity metric** (items outside the user's historical
   cluster that they engaged with — zero serendipity means the model
   has locked the user in); **(4) explicit counterfactual experiments**
   — periodically A/B a *bandit* or a *random* variant against the
   model at 1–5% traffic for weeks (not quarters) — one of the few ways
   to directly detect a runaway loop. **(b)** Four mitigations weave
   into the architecture from chapter 02: **(i) exploration budget**
   (1–10% of serving traffic goes to items the model is *not* confident
   about — Thompson sampling, UCB, or simple uniform-from-top-K —
   explicitly justified in the blueprint as a cost paid for training-set
   diversity); **(ii) position and selection debiasing** (inverse-
   propensity weighting for ranking so higher-ranked items don't get
   credit for their position; selection-adjusted training for scoring /
   approval systems where the untreated cohort is the crucial input);
   **(iii) regular full-population retraining** — once every N training
   cycles retrain on a *broader* set (untreated cohort, historical
   randomised data, externally sourced labels); one training-config
   flag, one extra pipeline run, one extra evaluation gate — "as boring
   as a nightly cron"; **(iv) human-in-the-loop calibration for
   high-stakes decisions** (credit, hiring, health, content moderation)
   — a fraction of decisions reviewed by humans, outcomes becoming gold
   labels, systematic errors surfaced that offline metrics miss. **(c)**
   Pitfall #8 ("A recommender whose only diversity metric is
   accuracy@K"): neither catalogue coverage nor entropy is on the
   dashboard; the collapse is invisible until the business metric
   drops. Catalogue coverage, entropy, and new-item ramp time belong
   on the **same dashboard** as CTR and revenue. **(d)** The **one-page
   loop analysis** — six questions, six lines, in every blueprint:
   what action does the model take and does the action determine
   whether the label is measurable? (yes → runaway risk); where do the
   training labels come from — passive observation, random assignment,
   or external ground truth?; what fraction of serving traffic is
   exploration?; what untreated / control cohort exists to measure the
   counterfactual?; what diversity metrics are on the dashboard?; what
   is the periodic full-population retrain cadence? In a design review
   this paragraph is the first thing an experienced ML reviewer will
   look for.
C. One detector is enough; mitigations are over-engineering; aggregate
   accuracy covers it; loop analyses are PR, not architecture.
D. Exploration always loses money; untreated cohorts are illegal;
   full-population retraining is a waste of compute.

### 9. (Chapter 05) A proposed A/B test on a marketplace ranking change uses user-level randomisation, reads the primary metric (AUC) at 2 days, lists latency as "we'll watch it," and has no SRM check. The design doc says "we'll decide at read-out." Chapter 05 names (a) the unit-of-randomisation traps that bite marketplaces and multi-device users, (b) the four rules for picking a primary metric, (c) the three-part guardrail contract, and (d) the pre-registered decision rule discipline that keeps the read-out from being a vibe check. Which captures them?

A. User-level randomisation is always correct; AUC is a fine primary metric
   for any A/B; latency doesn't need a threshold; "decide at read-out" is
   mature practice.
B. Four chapter-05 disciplines. **(a)** Two silent traps around the unit:
   **interference / network effects** (a treatment user's behaviour
   affects a control user's experience — marketplace pricing, social
   feed ranking, ride-share matching — so user-level randomisation gives
   **biased estimates**; the fix is a **cluster or switchback design**,
   both of which cost sample size — "this is the reason marketplace
   companies publish so much about experimentation methodology"), and
   **contamination** (a user assigned to treatment logs out and comes
   back on a different device and gets assigned to control — their
   behaviour blends the two, bounding how informative the test can be).
   Both traps get a paragraph in the design doc; don't leave them for
   the read-out. **(b)** Four rules for the primary metric: **(1) a
   business or customer metric, not an ML metric** — AUC did not move
   the number the exec team looks at, conversion / retention / session
   length did; **(2) upstream of the model's action at a horizon shorter
   than the labels' maturation** — if the label matures in 30 days and
   the test runs for two weeks, the primary metric cannot be the label;
   pick a shorter-horizon proxy mechanistically linked to the label;
   **(3) sensitive enough to move** under a plausible win at the test's
   sample size (power calculations quantify this); **(4) has an owner**
   — somebody outside the ML team with a well-formed opinion about
   whether the movement is good. If nobody owns the metric, the test
   result is a curiosity. **(c)** Every guardrail has **three parts**:
   a **direction** (must not decrease / must not increase), a
   **threshold** (what regression is acceptable — often `0` for latency
   SLOs, sometimes "no worse than 2% relative" for revenue), and an
   **owner** (the team that reads the metric and can call the test
   off). Guardrail failure is a **stop-the-test action** with its
   stopping rule set in advance in writing. Standard guardrails on any
   ML rollout: serving latency (p50/p95/p99), error rate (HTTP 5xx and
   application-level), downstream revenue on any revenue-adjacent
   surface, fairness / slice metrics per protected or business-critical
   slice, and behavioural side effects (unsubscribe rate, complaint
   rate, refund rate, support-contact rate). **(d)** The **pre-registered
   decision rule** in the design doc — "launch if primary metric
   improves by ≥ MDE with p < α AND no guardrail regression AND no
   slice regression; otherwise do not launch; **ambiguous means no**" —
   is what keeps the read-out from being a vibe check. Deviating from
   the design doc is a **design decision**, not a judgment call. A
   test with no pre-registered decision rule (pitfall #9) ends with
   "someone launches anyway because they built it."
C. User-level randomisation is always fine even on marketplaces; AUC is
   an acceptable primary metric; guardrails without stopping rules are
   standard; "decide at read-out" is fine.
D. Latency isn't a guardrail; SRM is paperwork; the read-out decides
   whether the test was a win.

### 10. (Chapter 05) A proposed A/B test has a power analysis of "run it for 2 days, see what happens," starts at a 50/50 split on day one, and the author plans to "check the primary metric hourly" to call the result early. Chapter 05 names (a) the four inputs to a real power analysis and the sample-size formula on the whiteboard, (b) two variance-reduction techniques that cut the required sample without biasing the estimate, (c) why the ramp is shadow → canary → test → launch and never 50/50 on day one, and (d) the specific statistical crime "peeking" commits. Which captures them?

A. A power analysis is a formality; any ramp works; checking a running test
   hourly is responsive engineering; there is no cost to peeking.
B. Four chapter-05 disciplines. **(a)** A real power analysis names four
   inputs and does the arithmetic. **Baseline metric value** `p₀` (what
   the primary metric averages today), **minimum detectable effect
   (MDE)** `Δ` (the smallest effect you care about in absolute terms —
   a 0.3pp lift on a 12% baseline is a ~2.5% relative lift), **significance
   level** `α` (conventionally 0.05), **statistical power** `1 − β`
   (conventionally 0.80), and **metric variance** (`σ² = p(1−p)` for a
   proportion; measured from historical data for a continuous metric).
   For a two-sample z-test on a proportion the per-variant sample size
   is approximately `n ≈ (z_{α/2} + z_β)² × 2 × σ² / Δ²` — the worked
   example (`7.85 × 2 × 0.12 × 0.88 / 0.003² ≈ 184,000`) is the kind of
   whiteboard calculation that should be memorised. Divide by the
   ramp-fraction-per-variant to get required *eligible* traffic; divide
   that by the eligible-user daily count to get duration; **round up**
   for weekly seasonality. **(b)** Two variance-reduction techniques
   that **cut the sample size without biasing the estimate**: **CUPED**
   (Controlled experiment Using Pre-Existing Data — subtract a
   pre-period covariate from each observation to reduce variance;
   publicly documented by Microsoft's experimentation team; widely
   adopted) and **stratified sampling** (stratify on known-important
   covariates — segment, region, device — and analyse within strata,
   recombining with weights; reduces variance from covariates the
   randomisation didn't balance). If your platform doesn't support them
   they're the first requests to make of the platform team. **(c)** The
   ramp is **shadow → 1% canary → test ramp (50/50 or asymmetric) →
   read-out → launch or roll back**, not 50/50 on day one, because
   shadow mode (chapter 03) catches training-serving skew before any
   user is affected, the 1% canary catches latency / error-rate / sanity
   regressions before wider exposure, and the test ramp is where the
   designed power actually accumulates. Two things the ramp discipline
   prevents: **a launch based on a launch-day spike** (novelty / bugs;
   read the aggregate over the full duration, not the trend) and
   **peeking**. **(d)** The specific statistical crime of **peeking**
   (watching the test's primary metric every hour and stopping as soon
   as it looks significant) is *the* classic way to fool yourself: a
   test read at a random time is meaningfully more likely to be
   false-positive than one read at its planned duration — the
   per-look false-positive rate compounds across the many looks. Some
   platforms support formal sequential-testing methods (SPRT, always-
   valid p-values) that make continuous monitoring safe; most in-house
   implementations do not. Three practical rules of thumb: a test that
   says "run for 2 days" is almost always underpowered or has a novelty
   problem waiting to bite (never read a test in under a week); a test
   that requires more users than the entire eligible population per
   month is a signal the MDE is too small (renegotiate before starting);
   a test whose primary metric is a rare event (fraud, unsubscribes)
   needs longer than you think because variance dominates. Also watch
   for the four canonical read-out traps: **novelty effects** (distrust
   the aggregate if the effect isn't sustained over the last third of
   the duration), **Simpson's paradox** (aggregate moves one way and
   every sub-group moves the other because the mix shifted — slice-level
   metrics must agree with the aggregate), **sample-ratio mismatch
   (SRM)** (the design says 50/50 but the actual ratio is
   statistically off — a bug in the routing layer invalidates the
   test's statistical claims entirely; fix the routing, don't
   interpret the result), and **guardrail regression that gets
   shrugged off** (the pre-registered stopping rule is the arbiter,
   not the vibe of the read-out meeting).
C. Power analyses are optional; ramping is excessive; hourly peeking is
   responsive; SRM is paperwork.
D. "Run for 2 days" is enough; CUPED is nice-to-have; the aggregate is
   always the truth even when slices disagree.

### 11. (Chapter 06) A team proposes to build its own experiment tracker, its own vector database, a from-scratch LLM, and an in-house annotation tool "because buying creates lock-in." For a tabular churn scoring task they also propose "to train a text classifier from scratch" because the LLM team is overloaded. Chapter 06 names (a) the seven-question build-vs-buy checklist, (b) where the buy usually wins vs where the build usually wins, (c) the four peer specialist tracks and the shape of each handoff, and (d) when *not* to delegate to a peer track. Which captures them?

A. Build is always better because it's more interesting; vendors create
   lock-in by definition; delegation is a sign of weakness; LLMs should
   be used for every text-shaped problem.
B. Four chapter-06 disciplines. **(a)** For every component of the
   chapter-02 reference architecture, walk the **seven-question
   checklist**: (1) **is this our core competitive advantage?** — if yes,
   building is a serious option; if no, buying is the default; (2)
   **what is the total cost of ownership over three years?** — including
   engineering time to build (1x headcount for how many months?),
   ongoing maintenance (~0.2x headcount per year *per component*,
   minimum), infrastructure cost, migration cost if you change your
   mind — vendor markup usually loses on year one and wins on year
   three; (3) **how mature is the market?** — ten vendors = commodity;
   two vendors = pricing power and lock-in matter more; (4) **how
   much of the vendor's product do we actually need?** — paying for
   50 features when you need 5 is sometimes worth it, sometimes a
   signal to look at OSS; (5) **what is the exit cost if this vendor
   disappears?** — data portability, model-artifact portability,
   dependency lock-in; a proprietary binary is a different risk than
   MLflow; (6) **is the boundary team ready to consume it?** — a
   vendor feature store means data engineering has to operate it; a
   model-observability product means SRE has to route its alerts; if
   the neighbouring team is not staffed the vendor is a short-lived
   experiment; (7) **what is the "build v0" alternative?** — "a feature
   store that is two warehouse tables and one Python client is
   sometimes exactly enough for the first year"; the comparison is
   **not** "buy vs a full custom platform" but "buy vs the minimum
   viable in-house version". **(b)** The **buy usually wins** on:
   experiment tracking and model registry (MLflow / W&B / Comet /
   Neptune — cheap, mature, known integration surface), **ML
   observability at scale** (Arize / WhyLabs / Fiddler — pays for
   itself at 10+ models; Prometheus is enough for 1–2), **LLM
   providers** (training a foundation model is not a fair build
   comparison), **managed training platforms** (Vertex AI / SageMaker
   / Databricks for teams without a dedicated ML platform team), and
   **annotation and labelling tools** (Scale / Labelbox / Snorkel —
   labelling UI + workflow + QA is a real product). The **build
   usually wins** on: the **feature pipeline logic itself** (transforms
   are domain-specific and shape the model), the **model architecture
   and training code** (buy the platform; build the model), the
   **A/B experimentation platform at large, latency-sensitive scale**
   (Netflix / LinkedIn / Airbnb / Uber / Microsoft all built theirs;
   at smaller scale, buying is fine), the **core business logic
   around the model** (intervention selector, outreach scheduler,
   pricing engine), and the **predictions table and labels loop**
   (warehouse tables and SQL — a week of work; buying is overhead).
   **(c)** The four peer specialist tracks and their handoff shapes:
   **(1) LLM Application Engineer** owns production systems whose
   primary model is a hosted LLM — prompts, function calling / tool
   use, agents, LLM-specific eval, token-cost tuning, prompt caching,
   safety filters, output guardrails — ML Engineer defines the
   business problem and the eval metric; **(2) RAG / Retrieval
   Engineer** owns retrieval-augmented systems — embedding models,
   vector databases, hybrid retrieval, rerankers, chunking, citations,
   retrieval-specific eval — delegate when grounding on a corpus is
   required, retrieval is the bottleneck, or sources + citations
   matter; **(3) Fine-Tuning / Model-Training Specialist** owns
   adapting foundation models — SFT, instruction tuning, RLHF/DPO,
   LoRA/QLoRA, distillation, quantisation, GPU inference optimisation
   — delegate when prompt engineering has hit a ceiling, cost/latency
   needs a smaller fine-tuned model, there's a proprietary data asset,
   or compliance requires on-prem; **(4) Generative Evaluation
   Specialist** owns evaluating non-deterministic outputs — rubric-
   based eval, LLM-as-judge, red-teaming, factuality / grounding
   checks, regression suites for prompt/model changes, safety / bias
   eval for generative outputs. Each has a specific handoff:
   ML Engineer produces the business problem + eval set + production
   integration; peer team owns the model / prompt / retriever / eval
   harness; A/B test ownership is shared or stays with ML Engineer;
   deployment goes with whoever is closer to the serving path. **(d)**
   Each track names when **not** to delegate: **LLM App** — the task
   is tabular / structured (LLMs are worse and slower and more
   expensive than a boosted-tree model for tabular prediction),
   latency is under 100ms with high QPS and a strict cost budget, or
   the output is a fine-grained numerical prediction where calibration
   and interpretability matter; **RAG** — straightforward short-text
   classification, corpus fits in a long-context prompt, or users
   don't need grounding; **Fine-Tuning** — baseline hosted LLM is good
   enough (most projects "discover that the prompt was the problem"),
   training data is small (< a few thousand high-quality examples),
   or the team can't afford the ongoing cost of hosting the fine-tuned
   model; **Generative Eval** — the output is a classification or a
   score with a real ground truth (`mod-107` metrics apply). "A tabular
   churn scoring task plus a text classifier the LLM team is not
   needed for" is exactly the shape where the generalist ML Engineer
   owns the whole thing; reinventing retrieval-augmented generation
   when the problem is a tabular score is pitfall #9.
C. Build everything to avoid lock-in; delegate nothing; all text tasks
   go to the LLM track; TCO is a vendor-sales fiction.
D. Buy everything to avoid maintenance; delegate every text task;
   tabular problems go to the LLM track too.

### 12. (Chapter 07) A reviewer walks in cold to a 60-minute design review for a new personalisation model. The design doc is thorough prose but doesn't name a business metric, the per-feature CI parity test, the untreated cohort, the pre-registered A/B decision rule, or any non-goals. Chapter 07 names (a) the shape of a design review (and what it is *not*), (b) the five-minute version of the checklist that catches the hardest failure modes, (c) the two failure modes of checklists themselves, and (d) the two moves that turn one-off reviews into a practice. Which captures them?

A. A design review is a debate about model architecture; prose beats
   checklists; the five-minute version is a quiz; the checklist is a
   document to file.
B. Four chapter-07 disciplines. **(a)** "A design review is **not a
   debate**; it is **an audit of specific claims**." The claims are
   the answers to the checklist's questions, grouped by chapter:
   Section A — business and framing (chapter 01), Section B —
   architecture (chapter 02), Section C — training-serving skew
   (chapter 03), Section D — feedback loops (chapter 04), Section E
   — A/B testing (chapter 05), Section F — build-vs-buy and
   delegation (chapter 06), Section G — not-goals revisited. The
   review typically takes 45–60 minutes; you don't walk every
   question, only the ones the design doesn't already address —
   unanswered questions are the review's real content. Both the
   **author** (walks it before the review; shrugged questions are
   weak spots to fix) and the **reviewer** (walks it during the
   review; shrugged questions become follow-ups) use the same list.
   **(b)** The **five-minute version** of the checklist is five
   questions targeting the five hardest failure modes: **(1) what
   business metric does this move, and by how much?** (unfocused
   problem — chapter 01); **(2) what decision does the model
   *make* — and what is the untreated control that lets you know
   if you're right?** (degenerate feedback loop — chapter 04);
   **(3) how do you know the features at serving time match the
   features at training time?** (training-serving skew — chapter
   03); **(4) how will you A/B test this — unit of randomisation,
   primary metric, guardrails, stopping rule?** (unmeasured launch
   — chapter 05); **(5) what's out of scope?** (scope creep —
   chapters 01 / 06). Memorise them. **(c)** Two failure modes of
   checklists in general: **ticking the box without earning the
   answer** ("yes, we have a feature store" — a vendor logo on a
   slide is not the same as a working point-in-time-correct offline
   join; the reviewer's job is to **press until the answer is a
   claim about *behaviour*, not a claim about *tooling***), and
   **reading the checklist as the design** (a design that answers
   every question and adds nothing is a document, not a plan; the
   value is the specific business context and the architectural
   choices that follow from it — the checklist is the floor, not
   the ceiling). **(d)** Two moves that turn one-off reviews into
   a practice: **every ML system in production has a design
   document *in the repo*** (new joiners read it, quarterly
   reviews update it, incidents append to it; a living artifact —
   "wikis rot; repos don't"), and **every promotion (chapter 03)
   or A/B launch (chapter 05) references the design doc** (if
   something is being built that the doc doesn't cover, the doc
   gets an update before the change ships — "not a bureaucracy, a
   memory"). Teams that do this find their oncall load drops —
   the things the checklist would have caught don't become
   incidents.
C. The design review is a committee process; the checklist should
   live in a wiki; the five-minute version is a nice-to-have but
   not necessary.
D. A thorough prose design doc is better than any checklist; the
   five-minute version is a cheat sheet; design reviews are a
   one-time event, not a habit.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **B.** Chapter 01 Section 1 Q1 ("What is the business decision this
   model informs?") names the five decision shapes (automated action,
   ranking / filtering, human-in-the-loop routing, suggestion / nudge,
   internal analytics) and states the AUC-vs-no-action point verbatim
   ("A churn model that nobody acts on has zero business value regardless
   of its AUC"). Section 2 "The target" nails the three things to
   document — unit of prediction, prediction horizon, label definition.
   The specific trap — "letting the label be whatever the data warehouse
   already has" — is called out as *the* mistake to avoid under "The
   target," with the `is_active` anecdote.

2. **B.** Chapter 01 Section 4 opens with **"Prefer batch until proven
   otherwise. Most models the business asks for do not need online
   serving"** and lists the four shapes in decreasing order of default.
   Section 5 "The mandatory baseline" lists the three jobs verbatim
   (anchors the accuracy conversation; de-risks the pipeline; generates
   labels for the model). Section 3 "The cold-start plan" names the four
   options and states "Not naming a cold-start plan is not a neutral
   choice. It means the model will be trained on whatever data leaks in
   by accident, which is usually the worst possible split." Section 6
   "Non-goals" states the redirect-the-discussion benefit; pitfalls #5
   ("Assuming online serving because it sounds modern"), #6 ("No
   cold-start plan for green-field problems"), and #7 ("No non-goals")
   each back up a leg.

3. **B.** Chapter 02 "The diagram in one paragraph" and "The nine
   components" name every box including the orchestrator; "2. Feature
   pipeline" states "The two halves … must produce *identical* outputs
   given the same inputs. … The pattern that prevents most skew is to
   write the transform *once*, in a shared library." The subsection
   "What 'the same model' is composed of" names the five artifacts
   verbatim — artifact + featureiser + reference profile + calibration
   cutoff + version label — and states "Change any of the five without
   the others knowing and you have shipped a different model without a
   version bump." Pitfalls #3 (`model.joblib` in S3), #4 (predictions
   logged but not persisted), and #5 ("Monitoring on the service, not
   on the pipelines") each name one of the misshapes.

4. **B.** Chapter 02 "The batch-only system" names it "the cheapest
   possible production ML system, and it fits more use cases than teams
   tend to admit." Section "3. Feature store" lists the two implicit
   versions ("a warehouse playing both roles" and "serving-time feature
   computation") and calls the latter **"catastrophic for time-window
   features"** verbatim. Section "7. Predictions table and prediction
   log" names the four things you cannot do without them (compare
   `v_new`/`v_old`, attribute business outcomes, debug a specific
   prediction, compute accuracy once labels mature) and instructs
   "Build them first. Every other analysis presupposes them." Section
   "10. Orchestrator" is "Not glamorous; unusually load-bearing" and
   names the silent-failure modes (retraining DAG stops, labels loop
   stops, accuracy metric silently reports last-run numbers forever).

5. **B.** Chapter 03's opening paragraph defines skew verbatim; "What
   'skew' actually means" distinguishes it from data drift, concept
   drift, and label leakage. "The four canonical causes" names them:
   two implementations, time-window mismatch, missing fitted
   preprocessor, feature-source drift. The "one-off ETL fix that
   doesn't backfill the training data" pitfall (#8) plus chapter 07's
   "Decide the action … the schema-change case is **not a retrain
   target yet**: retraining on a broken feature would **embed the
   bug**" cover (c). "Signal 1: Row parity on a canonical fixture"
   states **"This is the single highest-value skew defence in the
   entire chapter"** verbatim and specifies `np.allclose(..., atol=1e-9)`,
   blocking promotion on failure, the "belt and the braces" pairing
   with the feature-anomaly counter.

6. **B.** Chapter 03 "Signal 2: Prediction parity on shadow traffic"
   names aggregate distribution plus **per-request feature parity**
   together; pitfall #4 ("A shadow rollout that only compares aggregate
   distributions") says so explicitly. Pitfall #5 ("Fixture-parity
   tests that use a synthetic fixture with no edge cases") plus the
   `atol=1e-9` discipline cover (b). "The overnight refresh that lied"
   war story plus pitfall #9 ("Serving-time features computed lazily
   and cached with a wrong TTL") cover (c). "The skew audit — a
   checklist to run at design time" lists the nine audit items and
   closes with **"If the answer to any is 'no' or 'I'm not sure,' you
   have work to do before the promotion"**; the summary's closing
   line "Skew is patient. … Design assuming it exists somewhere and
   build the tests that will find it."

7. **B.** Chapter 04 "The dynamics — why the loop is inevitable" names
   the three facts and the "the loop is the diagram" line verbatim.
   "Vocabulary" fixes the five terms (feedback loop, data flywheel,
   selection bias, position bias, filter bubble, runaway loop). "Two
   loops that are virtuous, three that are not" names the degenerate
   patterns including the exact predictive-policing framing ("the loop
   is not measuring underlying crime; it is measuring police
   presence") and the structural rule **"a scoring system whose action
   determines whether the label is even measurable is a runaway loop
   by default."** Pitfall #2 ("Optimising the metric the loop
   inflates") and section "Two design choices that determine your
   loop's direction" cover (c) and (d), with the label-collection and
   model-position-in-decision-chain trichotomies named verbatim.

8. **B.** Chapter 04 "Detecting a loop before it does damage" names
   the four detectors in order (diversity metrics with catalogue
   coverage / entropy / new-item ramp time; untreated control cohort
   with 1–10% range; slice performance over time with novelty rate
   and serendipity; explicit counterfactual experiments). "Mitigating
   the loop by design" names the four mitigations (exploration budget
   1–10%; position/selection debiasing via IPS; regular full-population
   retraining "as boring as a nightly cron"; human-in-the-loop
   calibration for high-stakes decisions). Pitfall #8 ("A recommender
   whose only diversity metric is accuracy@K") names (c). "The
   one-page loop analysis" lists the six questions verbatim and states
   "in a design review, this paragraph is the first thing an
   experienced ML reviewer will look for."

9. **B.** Chapter 05 Section 1 "Unit of randomisation" names
   **interference / network effects** (marketplace pricing, social
   feed ranking, ride-share matching → biased estimates → cluster or
   switchback) and **contamination** (user logs out, comes back on
   a different device, blends variants) verbatim as "The two silent
   traps." Section 2 "Choosing the primary metric" lists the four
   rules verbatim including "AUC is almost never the primary." Section
   3 "Guardrail metrics" names direction + threshold + owner and
   closes with "Guardrail failure is a stop-the-test action. Set the
   stopping rule in advance in writing." Section 7's design doc names
   a pre-registered decision rule. Pitfall #9 ("A test with no
   pre-registered decision rule. The result is ambiguous; someone
   launches anyway because they built it").

10. **B.** Chapter 05 Section 4 "Sample size and duration" lays out
    the four inputs, the formula, the worked example (`7.85 × 2 × 0.12
    × 0.88 / 0.003² ≈ 184,000`), and the practical rules of thumb
    about "Never read a test in under a week." CUPED and stratified
    sampling are named as "advanced but standard" and attributed.
    Section 5 "Ramp plan" lists shadow → 1% canary → test ramp →
    read-out → launch/roll-back verbatim and names the two things the
    ramp prevents (peeking and launch-day spikes). Section 6 covers
    the four canonical read-out traps — novelty, Simpson's paradox,
    SRM, guardrail-shrug — each with its specific check. Pitfall #1
    ("Reading the test early. Peeking inflates false-positive rate.
    If your platform doesn't support sequential testing, you read at
    the designed duration, period").

11. **B.** Chapter 06 Section 1 "Checklist: the seven build-vs-buy
    questions" lists all seven verbatim including the "build v0"
    alternative. "Where the buy usually wins" and "Where the build
    usually wins" each list five items with the rationale. Section 2
    "Delegation to peer specialist tracks" lists all four tracks (LLM
    Application, RAG, Fine-Tuning, Generative Evaluation) each with
    "What they own," "When to delegate to them," "When *not* to
    delegate," and "The handoff shape." Pitfall #9 ("Not naming the
    peer track when the problem shape says 'LLM app' or 'RAG'")
    covers the inverse — building a worse version of what a peer
    track would deliver. The "tabular + short-text classification =
    generalist ML Engineer owns it" principle is explicit in Track
    1's "When *not* to delegate" and Track 4's.

12. **B.** Chapter 07 "Motivation" states "a design review is not a
    debate; it is an audit of specific claims" verbatim. "The
    five-minute version" lists the five questions exactly as (b) names
    them and attributes each to its chapter's failure mode. "What the
    checklist is not" names the two checklist failure modes verbatim:
    "Ticking the box without earning the answer" and "Reading the
    checklist as the design." "The design review as a habit" names
    the two moves — repo-resident design document, every promotion
    and A/B launch references it — and closes with "Teams that do
    this find their oncall load drops." Pitfalls #3 ("A design review
    that never gets written up"), #7 ("A checklist that lives in a
    wiki"), and the summary line "A design that only passes the
    checklist is a document; a design that adds the business context
    and the specific architectural choices is a plan."

</details>
