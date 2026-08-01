# 06 — Leakage and label quality

> **Reading time:** ~45 minutes.
> **Prerequisite chapters:** 05, and `mod-102` chapter 06 (Pipelines).

## Motivation

Two failure modes account for a wildly disproportionate share of
"the model looked great in the notebook and doesn't work in
production" outcomes:

1. **Data leakage** — the training data contained information the
   model would not have at prediction time. The CV score is
   optimistic; the online metric collapses on deploy.
2. **Bad labels** — a non-trivial fraction of the training targets
   are wrong, and the model has learned to reproduce the noise.
   Improvements in modelling can no longer help; you have hit a
   label-quality ceiling.

Both are cheaper to prevent than to detect. Prevention is a matter
of discipline about *what happens to the data before it becomes
`(X, y)`*, and that is what this chapter is about.

`mod-102` chapter 06 covered one specific leakage vector — fitting
a preprocessor on the full dataset before splitting — and gave you
the `Pipeline` construction that prevents it. This chapter covers
the rest: **target leakage, temporal leakage, group leakage,
train/test contamination,** and the label-quality issues that no
amount of leakage-hygiene can rescue you from.

## What "leakage" actually means

A working definition: **leakage is any pathway by which information
that would not be available at prediction time reaches the model
during training.**

Note "at prediction time". The model in production sees `X` and has
to produce `y`. Anything the model *learned to depend on* that
production `X` will not carry is leakage.

Concrete examples we'll unpack below:

- The label itself, encoded into a feature (`is_fraud` derived from
  a `chargeback_amount` that only exists post-fact).
- A feature computed with a rolling window that includes the label
  row (`user_avg_spend` that includes the transaction we are
  predicting).
- A feature whose value at training time is a *summary* of the
  future (`user_lifetime_value` on a two-year-old training row).
- A join with a table that was populated *after* the label event
  (a `is_churned` label joined to a `latest_plan` column that was
  updated when the churn happened).
- A CV split that puts different rows from the same user in train
  and test folds, when at deployment the user is entirely new.

## Target leakage: the feature *is* the label in disguise

The most obvious kind, and the one that most often slips through in
finance / risk / medical datasets. A column that seems predictive
turns out to be a downstream consequence of the target, not a
signal for it.

Examples:

- Predicting churn: a `days_since_last_login` column includes the
  churn event itself, so churned users have `NULL` (or a very
  large value) and the model achieves 99% accuracy trivially.
- Predicting fraud: an `investigation_status` column is only
  populated when fraud is flagged. Its presence perfectly predicts
  the label.
- Predicting mortality: a `discharge_disposition` column indicates
  death, discharge, or transfer — after the fact.

The detection recipe:

1. **List every feature and ask, "was this value known *before* the
   target event?"** If not, the feature leaks.
2. **Rank features by importance after a first model fit.** Any
   feature that dominates the top of the list (single-feature
   perfect performance) should be suspicious rather than
   satisfying. Investigate before shipping.
3. **Correlate features with the target on a hold-out.** Perfect
   correlation on a binary label with millions of rows is almost
   always leakage, not a great feature.

The cure is upstream, not in the model: exclude the leaking column
from `X` and, ideally, remove it from the feature-extraction step
so it cannot come back.

## Temporal leakage: the future in the past

If the data has a time dimension — and most business data does —
the training set must not contain any information that would only
be available *after* the prediction time.

The pathological example. You want to predict, at time `t`, whether
a user will churn in the next 30 days. Your feature set includes
`user_avg_session_length`, computed as the average over the entire
user history. On a training row from January, this "average"
includes sessions from February through December. That's leakage:
February's sessions were not knowable in January.

The disciplined fix has two parts.

### 1. Split by time, not by row

Random `train_test_split` on time-series data mixes future rows into
the training set. The train/test split for temporal data should be
strictly ordered:

```python
cutoff = pd.Timestamp("2026-06-01", tz="UTC")
train  = df[df.event_ts <  cutoff]
test   = df[df.event_ts >= cutoff]
```

For cross-validation, use `sklearn.model_selection.TimeSeriesSplit`,
which produces expanding-window folds — each fold's training set
strictly precedes its test set.

### 2. Compute features "as of" the label time (point-in-time joins)

Every feature must be computed using *only* data available at the
label event's time. If a label row has `event_ts = 2026-04-15`, then
`user_avg_session_length` for that row must be the average over
sessions with `session_ts < 2026-04-15`.

In SQL, this is the point-in-time join pattern from chapter 03:

```sql
SELECT
  l.user_id,
  l.event_ts,
  AVG(s.duration) OVER (
    PARTITION BY l.user_id
    ORDER BY s.session_ts
    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
  ) AS avg_session_30d
FROM labels l
JOIN sessions s
  ON s.user_id = l.user_id
  AND s.session_ts < l.event_ts       -- CRITICAL
```

In Pandas, the equivalent is `merge_asof` with `allow_exact_matches=False`
or a manually implemented rolling window that respects the label
timestamp.

Feature stores (Feast, Tecton, Databricks Feature Store) exist in
large part to make point-in-time correctness a first-class primitive
so you cannot accidentally compute a feature with future data.

### The "as-of" mental model

When staring at a training row, ask: **"if I were making this
prediction live at `event_ts`, would this feature value be the same
as what I have in the training set?"**

- If yes — safe.
- If no — leakage. Recompute the feature "as of" `event_ts`.
- If "I'm not sure" — the answer is no. Investigate.

Any feature that was computed after the label event, even if
"correct-looking" in the historical data, is a candidate leak.

## Group leakage: the same user in train and test

Random splitting works when rows are independent. They usually are
not.

- Multiple rows per user, but the deployed model will see brand-new
  users at prediction time.
- Multiple images per patient, but the deployed model will see
  brand-new patients.
- Multiple time-adjacent rows from a slow-changing entity — a
  household, a session, a device.

If rows from the same group appear in both train and test, the
model can memorise per-group information (a user's typical spending,
a patient's baseline heart rate) that will not help at inference
because the *inference user* is new.

Use grouped CV:

```python
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

cv = GroupKFold(n_splits=5)
scores = cross_val_score(pipe, X, y, groups=X["user_id"], cv=cv)
```

`GroupKFold` guarantees no group appears in more than one fold. For
temporal + grouped situations (both time-ordered *and* new-entity),
`sklearn.model_selection.GroupTimeSeriesSplit` (available via
`sktime` or as a small hand-rolled class) is the tool.

The choice of split is not a nice-to-have; it is a modelling
decision. A random split on grouped data can overstate the online
metric by 10–30 percentage points.

## Train/test contamination

Even with a correct temporal + group split, contamination sneaks in
through preprocessing.

- **Fitting the scaler on the whole dataset before splitting.** The
  test fold's mean bleeds into the training fold via the scaler
  parameters.
- **Feature-selecting on the whole dataset before splitting.** The
  test fold's labels influenced which features were kept.
- **Encoding categoricals with target statistics computed on the
  whole dataset.** Target encoding on the combined `y` is
  classical leakage; see chapter 07's rules for how to do it
  safely.
- **Oversampling before splitting.** SMOTE the whole dataset, then
  split, and the synthetic points from the training minority class
  are nearest neighbours of the test minority class — trivial to
  memorise.
- **Hyperparameter tuning on the test set.** `GridSearchCV` inside
  `mod-102`'s `Pipeline` uses the *training* folds' CV. Do not
  peek at the held-out test set until you have exactly one final
  model.

Every one of these is prevented by the same construction: **fit
inside a `Pipeline`, and only ever call `Pipeline.fit` on the
training portion.** If your preprocessing cannot be expressed as a
`Pipeline` step, that is a signal to write a custom transformer for
it (`mod-102` chapter 05), not to escape the pipeline discipline.

## The train/serve loop

`mod-101` and `mod-102` both mentioned this; here we make it a rule.

**The training pipeline and the serving pipeline must apply exactly
the same feature transformations to the raw data.** Every
divergence between them is a bug. Two production-grade patterns
enforce this:

1. **A single artefact.** The fitted `Pipeline` (`joblib.dump(pipe,
   ...)`) contains both the fitted transformers and the fitted
   model. At serving time, you `joblib.load` it and call
   `pipe.predict` — the transformations at serve time are, byte
   for byte, the ones from training.
2. **A feature store.** Both training and serving read features
   from the same materialised store, and the store guarantees
   point-in-time correctness. Overkill for a first model, essential
   for a fleet of them.

`mod-108` covers the deployment side; this chapter's job is to keep
you from building a model whose preprocessing exists only in a
Jupyter cell.

## Label quality: the ceiling on your metrics

Even with perfect features and no leakage, your model can only be
as good as its labels. Two truths:

- **All real-world labels are noisy.** Human labellers disagree
  with themselves and each other. Automated labels (heuristics,
  weak supervision) have systematic errors.
- **Label noise puts a hard ceiling on achievable accuracy.** If
  10% of your labels are wrong, no model can exceed ~90% on the
  same distribution. Chasing the last 3% of accuracy on a dataset
  with 10% label noise is spending effort on the wrong problem.

### Estimating label noise

If you have inter-annotator agreement data (multiple labellers per
example), Cohen's kappa or the equivalent gives you a lower bound
on noise: `noise ≈ 1 - agreement`. For image datasets the ImageNet
authors estimated label error at roughly 3–6%; landmark surveys
have found similar rates in text and audio datasets.<!-- needs-research:
verify with the "Pervasive Label Errors" paper (Northcutt et al.
2021) and cite: https://arxiv.org/abs/2103.14749 → confirm current
citation and update rate estimates. -->

If you don't have multi-annotator data, one accessible test:

1. Fit a strong model.
2. Look at the most-confidently-wrong predictions on a held-out
   set (top-k by predicted probability where the label disagrees).
3. Hand-review 50 of them.
4. Count how many are actually label errors vs. model errors. The
   ratio is a rough estimate of your label noise rate.

`cleanlab` (`pip install cleanlab`) automates this — it uses a
model's out-of-fold predictions to rank likely label errors.

### Living with noisy labels

If the noise is random and mild:

- Accept the ceiling. Report it in your README (`"model achieves X;
  estimated label noise Y%"`).
- Use loss functions and evaluation metrics that are more robust to
  noise — e.g. label smoothing during training; per-class F1 rather
  than raw accuracy for imbalanced problems.

If the noise is systematic (some subgroups mislabelled more than
others):

- Fix it upstream. Systematic label noise causes systematic model
  failures — often the ones that show up as fairness issues in
  production.
- Consider using two annotators per row and disagreement
  arbitration for the disputed subgroup.

If the noise is severe (>20%):

- Do not train on the raw labels. Consider weak-supervision
  aggregation (Snorkel), consensus labelling, or reframing the
  task so the label is more reliably defined.

## Sanity checks: the "leakage smell tests"

A short checklist to run before shipping any model.

1. **Baseline accuracy check.** Compare against a trivial baseline
   (majority class, most-recent-value). If your model is only 1%
   better, you have a problem. If it is 40% better than a strong
   baseline, you probably have leakage.
2. **Per-feature importance.** If one feature explains almost all
   the model's predictions, that feature is either the answer key
   or overwhelmingly powerful. Investigate.
3. **Ablate the top feature.** Drop it and refit. If the model
   loses 20+ points of accuracy, that feature deserves a hard
   look for leakage.
4. **Adversarial validation.** Train a classifier to distinguish
   train rows from test rows. If it can (AUC well above 0.5),
   your split is not IID and the "test set" is measuring the
   wrong thing.
5. **Time-ordered hold-out.** Regardless of your CV setup, take
   the last week (or month) of data as a strict hold-out. If the
   held-out score is meaningfully worse than CV, you have
   temporal leakage.
6. **Domain-expert eyeball.** Show your top features to someone
   who knows the domain. "Is it plausible that these are what
   drives the outcome?" A domain expert spots leakage in seconds
   that ML engineers miss for weeks.

## Pitfalls collected

1. **Believing the leaderboard.** Kaggle-style CV scores are the
   number to beat, not the number to trust. Any submission whose
   CV score is dramatically higher than everyone else's is
   probably a leak someone will announce later.
2. **Post-hoc feature engineering with future data.** "I aggregated
   over the full user history" is fine only if that history was
   truncated to the label time. Otherwise it's temporal leakage.
3. **Using `train_test_split(shuffle=True)` on time-series.**
   Silently, casually catastrophic.
4. **`train_test_split` on grouped data without `stratify=` or
   `groups=`.** Some `Adult`-dataset rows are from the same
   household; some medical rows from the same patient. Group first.
5. **Ignoring the noise floor.** Chasing a model improvement that
   is inside the confidence band of the label noise. You are
   overfitting to the noise.
6. **Trusting the automated labeller.** Weak-supervision labels
   (heuristics, distant supervision) have systematic biases that
   look like features to the model. Audit a sample.

## Summary

- Leakage is any information from the future or the label reaching
  the model during training. It comes in target, temporal, group,
  and preprocessing-contamination flavours. Each has a specific
  antidote.
- Temporal leakage is prevented by *time-ordered splits* and
  *point-in-time joins*. Group leakage is prevented by *grouped
  CV*. Preprocessing contamination is prevented by *fitting
  inside a `Pipeline`*.
- The single training/serving artefact (a fitted `Pipeline`) is
  the strongest defence against train/serve skew.
- Labels are noisy. Estimate the noise rate; report it; do not
  chase model improvements smaller than it. Fix systematic label
  noise upstream.
- Run the leakage smell tests before shipping. Especially the
  time-ordered hold-out, the ablation of your top feature, and
  the domain-expert eyeball.

## Where this goes next

Chapter 07 begins the feature-engineering arc. Everything you build
there must respect the leakage rules from this chapter —
`Pipeline`-based, split-first, and point-in-time correct for
temporal features.
