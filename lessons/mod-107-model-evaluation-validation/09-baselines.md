# 09 — Baselines that tell you something

> **Reading time:** ~20 minutes.
> **Prerequisite chapter:** 02, 03.

## Motivation

"Our model scores 0.83" is not a number. "Our model scores
0.83; the baseline that always predicts the majority class
scores 0.65; a well-tuned logistic regression on the same
features scores 0.79" is a number. Every metric this
module has covered is meaningless without a reference
point.

Baselines serve two independent purposes and it is worth
keeping them straight:

- **Sanity floor.** A trivial baseline you must beat —
  otherwise the model is genuinely worse than doing
  nothing. Catches broken pipelines, target-encoding
  bugs, wrong metric direction, and reversed classes.
- **Fair comparison.** A meaningful baseline your model
  is proposed to *improve on*. Beating a lookup-table
  baseline is not an improvement claim; beating a well-
  tuned linear model on the same features is.

Both should exist for every published number. This
chapter walks the trivial baselines (chapter's second
purpose in scikit-learn's `DummyClassifier` /
`DummyRegressor`), the strong classical baselines, and
the discipline for reporting them alongside the "real"
model.

## The trivial baselines

The zero-code, zero-thought reference points. Every
project has these:

### Classification

- **Majority-class predictor.** Always predicts the most
  common class. Accuracy = base rate of the majority
  class. Perfect for the "am I above chance on this
  imbalanced dataset?" question.
- **Prior-probability predictor.** Predicts each class
  with probability equal to its base rate. Useful as a
  reference for probabilistic metrics (log loss, Brier).
- **Stratified random predictor.** Predicts each class
  with probability equal to its base rate, at random.
  Same expected metric as prior for calibration; gives
  variance for accuracy.
- **Uniform random predictor.** 1/K per class. Only
  useful when the classes are approximately balanced.

`sklearn.dummy.DummyClassifier(strategy=...)` implements
`"most_frequent"`, `"prior"`, `"stratified"`, `"uniform"`,
and `"constant"`. Any classification write-up should show
at least `most_frequent` and `stratified` for reference.

### Regression

- **Mean predictor.** Always predicts $\bar{y}$ on the
  training set. Optimal constant for MSE.
- **Median predictor.** Always predicts the training
  median. Optimal constant for MAE.
- **Quantile predictor.** Always predicts a specific
  quantile — useful when the model is being reported on
  quantile loss.

`sklearn.dummy.DummyRegressor(strategy="mean")` /
`"median"` / `"quantile"` / `"constant"`.

### Time-series

- **Persistence baseline.** Predict $\hat{y}[t] = y[t-1]$
  ("tomorrow will be like today"). This is the single
  most important baseline for any forecasting problem;
  many complicated models fail to beat it on real data.
- **Seasonal naive.** Predict $\hat{y}[t] = y[t - S]$
  where $S$ is the seasonal period ("this Monday will be
  like last Monday"). The right baseline when the series
  has clear seasonality.
- **Moving-average baseline.** Predict the mean of the
  last $w$ observations. Useful reference when the
  series is smoother.

Scikit-learn does not ship these; implement in three
lines, or use `sktime`'s `NaiveForecaster`.

### Retrieval and recommendation

- **Popularity baseline.** Recommend the globally most-
  interacted items to every user. The single most
  important baseline for a recommender — beat this before
  claiming personalisation adds value.
- **Recent-items baseline.** Recommend the most recently
  observed items. Often surprisingly hard to beat in
  news / content settings.
- **Random baseline.** For MRR / nDCG / hit@k reference
  points on the metric scale.

## The strong baselines

Trivial baselines catch broken pipelines. **Strong
baselines** are what the model needs to beat to justify
its complexity. The rule of thumb: at level 20, the
strong baseline is *a well-tuned classical model on the
same features you would give the "real" model*.

Concretely, before shipping a deep learning model, prove:

- **A well-tuned logistic regression** (`LogisticRegression`
  with `LiblinearSaga` or `saga` for large-scale, class
  weights tuned, regularisation swept) does not do the
  same job.
- **A well-tuned gradient-boosted tree**
  (`HistGradientBoostingClassifier` in scikit-learn,
  or LightGBM / XGBoost) with default hyperparameters and
  a modest tuning pass does not do the same job.

On tabular data, this is a real constraint. Modern GBDT
implementations are state-of-the-art on structured
tabular problems for most reasonable dataset sizes, and
"we tried a neural net and it beat XGBoost by 0.2%" is a
famously hard claim to defend.

<!-- needs-research: cite Grinsztajn, Oyallon, Varoquaux
(2022) "Why do tree-based models still outperform deep
learning on tabular data?" for the tree-vs-DL tabular
benchmark. -->

On non-tabular data (images, text, audio), the strong
baseline is a **pre-trained model with a linear head**
rather than a from-scratch specialised architecture. A
frozen ImageNet ResNet with a linear classifier on top
is a very strong baseline for most image tasks; a frozen
BERT or sentence-transformer with a linear head is
similarly strong for many text classification tasks. If
your custom architecture cannot beat this, it is not
justified.

## When to build a task-specific baseline

Some tasks admit a domain-specific baseline that neither
the trivial nor the strong classical baselines capture:

- **Fraud detection**: a rule-based baseline (existing
  rules the model is proposed to replace). If the model
  underperforms the rules, ship the rules.
- **Search**: BM25 or TF-IDF retrieval. Neural retrievers
  must beat BM25 on the task's actual metric to justify
  the serving cost.
- **Physical-system prediction**: a physics-based model
  (Hooke's law, mass balance, mean field). The ML model
  must beat the physics; otherwise the physics is doing
  most of the work and the ML component is theatre.

The pattern: **what is the existing thing (a rule set, a
lookup table, a physical model) that would be used if the
ML model did not exist?** That is the baseline the
project's business case implicitly compares against; make
the comparison explicit.

## Reporting baselines alongside the model

The template for the report — publish four rows, always:

| Row | What | Purpose |
|---|---|---|
| Trivial | Majority-class / mean / persistence | Sanity floor |
| Strong-classical | Well-tuned logreg / GBDT / BM25 | Fair reference |
| **Your model** | The proposed model | The claim |
| Human / oracle (if available) | Human performance or perfect information | Upper bound |

The gap between the trivial and strong-classical is the
"how much does a competent solution give you?" number.
The gap between strong-classical and your model is the
"what does the increment buy?" number. The gap between
your model and the oracle (if you can define one) is the
"how much more is left on the table?" number.

Without at least the first three, no reader can size the
result. A "0.83 F1" number is meaningless without knowing
whether the trivial baseline is 0.80 (marginal
improvement) or 0.30 (huge improvement).

## Human performance as an oracle

Some tasks have a natural upper bound: human labellers
disagree with each other at some rate. If two human
annotators agree on a label 92% of the time (Cohen's
kappa 0.85), then 92% agreement between the model and a
single annotator is roughly the ceiling — the model
cannot exceed inter-annotator agreement without becoming
more consistent than the humans it is being judged
against.

Reporting this ceiling changes the conversation. "Model
accuracy 0.87; inter-annotator agreement 0.92" says the
model is close to the ceiling; "Model accuracy 0.87;
inter-annotator agreement 0.99" says there is 12 points
of headroom. Two very different states of the world; the
same headline number.

Compute inter-annotator agreement on the tasks that have
multiple labels per item; use `cohen_kappa_score` or
`fleiss_kappa` for the standard statistics.

<!-- needs-research: cite Landis & Koch (1977) or a modern
IR/NLP reference for kappa interpretation ranges. -->

## Baselines within the training loop

Baselines are not just a report artefact. Instrument the
training loop to log the trivial baseline metric on the
same validation set as the model. Two benefits:

- **Regression detection.** If the model metric drops
  below the baseline metric during training (learning
  rate too high, gradient explosion, label bug), you see
  it immediately.
- **Comparability across data versions.** The trivial
  baseline metric on validation depends only on the
  data. Its value gives you a fingerprint for the
  dataset; if the baseline moves substantially between
  runs, your data pipeline changed.

The MLflow pattern from `mod-106`:

```python
import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

# Fit and log the baseline in the same run as the model.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_tr, y_tr)
mlflow.log_metric(
    "baseline_majority_f1",
    f1_score(y_val, baseline.predict(X_val), average="macro"),
)
```

Every run's UI now shows the model's F1 next to the
baseline F1; a run whose model_f1 is below baseline_f1
is broken, full stop.

## Common pitfalls

1. **Publishing a number with no baseline.** The most
   common failure. "F1 = 0.72" is not a claim; it is a
   number.
2. **Only using a trivial baseline.** Beating the
   majority classifier by 15 points on an imbalanced
   dataset says nothing about whether a competent
   solution (logreg) would beat you by 10 points.
3. **Weak "strong" baselines.** A logistic regression
   fitted with default hyperparameters on raw features
   is not a well-tuned baseline. Sweep at least a small
   grid; the whole point is to make the comparison fair.
4. **Different features for baseline and model.** The
   baseline uses the raw features; the model uses
   engineered features. The lift you report includes the
   feature engineering, not the model architecture.
   Report both — model on baseline features (isolates
   architecture lift) and both on engineered features
   (isolates feature-engineering lift).
5. **Baselines on the training set.** Baseline metrics
   belong on the held-out set, computed the same way as
   the model's metric.
6. **Persistence baseline not evaluated.** On time series
   this is the single most common shipping-a-worse-
   model-than-nothing case. Compute the persistence
   metric; publish it; if you cannot beat it, the model
   is not shipping.
7. **Comparing across metric families.** "Model log loss
   improved 5%; baseline log loss unchanged; but model
   accuracy also improved." Whether the improvement is
   real depends on which metric matters. State the
   primary metric before comparing.

## A worked baseline table

The template your report should look like, for a
classification task with a 5% positive class:

| Model | ROC-AUC | PR-AUC | F1@0.5 | log loss |
|---|---|---|---|---|
| DummyClassifier(most_frequent) | 0.500 | 0.050 | 0.000 | 0.196 |
| DummyClassifier(stratified)    | 0.500 | 0.050 | 0.050 | 0.680 |
| LogisticRegression (tuned)     | 0.881 | 0.312 | 0.264 | 0.083 |
| HistGradientBoosting (tuned)   | 0.926 | 0.423 | 0.371 | 0.070 |
| **Proposed model**             | **0.937** | **0.464** | **0.398** | **0.067** |

Reading the row order:

- The **trivial** rows give the "am I above chance?"
  reference. ROC-AUC = 0.5 confirms the metric is well-
  calibrated; PR-AUC = 0.05 confirms the base rate.
- The **strong-classical** rows give the "am I better
  than a competent classical solution?" reference. The
  proposed model beats HistGradientBoosting by ~0.011
  ROC-AUC and ~0.041 PR-AUC. That is the *actual claim
  being defended*.
- Without both baseline rows, the raw "0.937 ROC-AUC"
  number reads as impressive; against the strong
  baseline, the increment is smaller than a reader might
  assume.

## Summary

- Every reported metric needs at least a trivial and a
  strong baseline. Without them, the number is
  uninformative.
- Trivial baselines catch broken pipelines
  (`DummyClassifier`, `DummyRegressor`, persistence).
- Strong baselines are the actual comparison point — a
  well-tuned classical model on the same features (linear
  / GBDT / BM25 / pre-trained-with-linear-head). Beat
  this before claiming the increment.
- Report baselines *in the same run* as the model, on
  the same validation set, in the same table.
- Human agreement is the natural upper bound where
  labels are subjective; report it as a ceiling.

## Where this goes next

Chapter 10 is error analysis — the detective work of
turning "the model scores 0.83" into "the model scores
0.83, and here is where it fails, and here is why".
