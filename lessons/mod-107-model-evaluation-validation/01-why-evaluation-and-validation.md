# 01 — Why evaluation and validation

> **Reading time:** ~15 minutes.
> **Prerequisite modules:** `mod-104` (classical ML) or
> `mod-105` (deep learning) — something that can produce a
> prediction.

## Motivation

Everything in a modelling project eventually funnels through
one question: **is this model better than what we had
yesterday?** The answer is a number. The whole point of this
module is to make sure that number is honest.

Two failure modes account for most of the "we shipped a
regression" stories in production ML, and neither is caught
by unit tests:

1. **The metric was wrong for the task.** The model optimised
   accuracy on a dataset where 99% of rows are the negative
   class; the "97% accurate" model never predicts a positive.
   Or accuracy improved and revenue dropped, because the
   real cost of a false positive was ten times the cost of a
   false negative and nothing in the training loop knew that.
2. **The validation split was wrong for the task.** Data from
   next week leaked into training. Two rows for the same
   patient ended up on opposite sides of the split. Cross-
   validation was nested with feature selection in the wrong
   order and every fold saw the test set indirectly. The
   held-out number was optimistic; production was not.

Both failures are silent. Both produce a model that looks
fine on the day it was trained and disappoints the moment
someone acts on it. This module is the discipline for
preventing them:

- **Metrics** (chapters 02–05). Pick a number that reflects
  the decision the model is being used to make — regression
  error, classification threshold behaviour, ranking order,
  calibrated probability. Report a companion metric so you
  can see the trade-off, not just the winner.
- **Validation** (chapters 06–08). Design splits that mimic
  the way the model will be used in production. Guard
  against the ways a naive split leaks — through time,
  through groups of correlated rows, through the CV loop
  itself.
- **Baselines and error analysis** (chapters 09–10). Beat
  something meaningful before you claim you have beaten
  anything. Then break the aggregate open — where does the
  model fail, and does that pattern make sense?
- **Slices** (chapter 11). The aggregate number hides sub-
  populations. Report the metric on the slices that matter
  (subgroup fairness, adversarial perturbations, out-of-
  distribution shifts) so the worst-case behaviour is
  visible before it ships.

## The evaluation loop

Every training script produces at least one prediction. The
evaluation loop is how you turn that prediction into a claim
you can defend:

```
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  Training   │───▶│ Prediction  │───▶│   Metric    │
   └─────────────┘    │ on held-out │    │  aggregate  │
                      │    split    │    └──────┬──────┘
                      └──────┬──────┘           │
                             │                  ▼
                             │           ┌─────────────┐
                             └──────────▶│    Slice    │
                                         │  breakdown  │
                                         └──────┬──────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │   Compare   │
                                         │ vs baseline │
                                         └─────────────┘
```

Four things have to be right for the number at the end to be
trustworthy:

- **The split** produced a held-out set that is genuinely
  unseen — no leakage through time, groups, or preprocessing.
- **The metric** captures the decision that will be taken
  with the model. If ranking matters, use a ranking metric;
  if a probability matters, use a probability metric.
- **The aggregate** is not the whole story. Slice by the
  dimensions you care about (segment, region, time bucket,
  sensitive attribute) and report the worst slice, not just
  the mean.
- **The comparison** is against a baseline strong enough to
  make the result meaningful. Beating a coin flip is not an
  achievement.

Miss any of the four and the number is theatre.

## What the four objectives cover, in one paragraph each

- **Metrics for the task (chapters 02–05).** Regression
  chapter grounds MAE / MSE / RMSE / R² in what each is
  robust to and what each is fooled by. Classification
  chapter walks the confusion-matrix vocabulary and the ROC
  vs PR trade-off under class imbalance. Calibration
  chapter distinguishes "accurate at ranking" from
  "accurate as a probability" — the difference between a
  useful score and a usable probability. Ranking chapter
  covers position-aware metrics (MAP, nDCG, MRR, hit@k) and
  when to prefer them to classification metrics.
- **Validation without leakage (chapters 06–08).** Splits
  chapter enumerates the leakage taxonomy — target leakage,
  train/test contamination, group leakage, temporal leakage,
  preprocessing leakage — and where each one hides. Time
  chapter covers walk-forward / expanding-window / rolling-
  origin splits and why "shuffle + split" is malpractice on
  temporal data. Group/nested chapter covers `GroupKFold`,
  `StratifiedGroupKFold`, and nested CV for honest
  hyperparameter selection.
- **Baselines and error analysis (chapters 09–10).**
  Baselines chapter distinguishes trivial baselines
  (majority class, mean prediction, previous value) from
  strong baselines (a well-tuned logistic regression or
  gradient-boosted tree). Error-analysis chapter is the
  detective work — sort the errors, cluster them, look at
  the extremes, and turn the pattern into a fixable claim.
- **Fairness and robustness slices (chapter 11).** The
  level-20 bar: a small, disciplined set of slice reports
  (subgroup performance gaps, noise/perturbation robustness,
  distribution-shift probes) that make the worst case
  visible without launching into a full fairness audit.

## Where this module sits in the track

- **Input.** A model from `mod-104` (classical) or `mod-105`
  (deep learning) that already trains and produces
  predictions. Runs tracked in MLflow from `mod-106` — the
  metrics you compute here get logged there.
- **Output.** A validated model with a defensible metric on
  a defensible split, and a slice report that says where it
  fails. This is the artefact `mod-108` (deployment) picks
  up when someone wants to serve it, and `mod-109`
  (monitoring) references when checking for online drift.

If evaluation is not honest, deployment ships a broken
model faster and monitoring alerts on the wrong things.
This module is the load-bearing joint between "we trained
something" and "we can defend it".

## The single distinction to internalise early

Two words that get used interchangeably and should not:

- **Test set.** A held-out set touched *once*, at the end,
  to report the number you publish. It exists to prevent you
  from tuning on the number you are about to report.
- **Validation set** (or **CV folds**). Held-out sets used
  *repeatedly* during model development — for
  hyperparameter selection, early stopping, feature
  choices. By the time you have tuned against a validation
  set fifty times, it is no longer unseen.

Every result reported in this module distinguishes
"validation" (used to choose the model) from "test" (used to
report the number). Mix them up and the numbers you publish
are not the numbers a stranger would see.

Scikit-learn's `train_test_split` produces one split; that
is fine for a first pass. Cross-validation produces many
splits; each fold is a validation set. A **held-out test
set** is a *third* partition, cut before any of this, that
you do not touch until the model is frozen. Chapters 06–08
make the partition strategy explicit.

## Common objections, briefly answered

- **"The default metric that comes out of `model.score()`
  is fine."** It is fine if you never have to defend the
  number. `model.score()` in scikit-learn is R² for
  regressors and accuracy for classifiers; both are the
  wrong choice for a large fraction of real tasks (chapters
  02–03 make the case).
- **"Cross-validation handles leakage for me."** It handles
  the sampling variance of the metric. It does not stop
  future information from leaking into the past, correlated
  rows from ending up on both sides, or a `StandardScaler`
  fit on the full dataset before the split. Leakage is a
  pipeline property; CV cannot see it.
- **"We do fairness later, when the model matters."** By
  then the metric structure and the training set are baked
  in and slice gaps are far more expensive to move. A five-
  slice report at level 20 catches most of the "we shipped
  a model that fails for a subgroup" incidents; chapter 11
  covers the minimum viable version.
- **"The aggregate is the number the product manager
  wants."** Publish the aggregate *and* the worst slice.
  The gap between them is the risk everyone is buying.

## Summary

- Evaluation is where a model becomes a claim. This module
  is the discipline for making the claim honest.
- Four things have to be right: the split, the metric, the
  slice breakdown, and the comparison against a baseline.
- The module maps onto four sub-topics: metrics (02–05),
  validation without leakage (06–08), baselines and error
  analysis (09–10), and lightweight fairness/robustness
  slices (11).
- Keep "validation" (used many times to tune) and "test"
  (touched once to report) distinct. A number derived from
  a set you tuned on is not a test number.

## Where this goes next

Chapter 02 opens the metrics half of the module with
regression — MAE, MSE, RMSE, R², MAPE — and the trade-offs
between them. It is the shortest metric chapter but the
one that fixes the vocabulary reused in classification,
calibration, and ranking.
