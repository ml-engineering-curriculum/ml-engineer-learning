# 10 — Error analysis

> **Reading time:** ~25 minutes.
> **Prerequisite chapters:** 03, 09.

## Motivation

The aggregate metric is a summary of the model's
behaviour. Error analysis is the practice of breaking that
summary apart until each remaining "how does this fail?"
question has a concrete answer.

Every meaningful modelling improvement starts as an
observation about a specific failure mode: "the model is
worse on short sentences", "the model over-predicts on
Fridays", "the model misclassifies image X and the ten
neighbours in feature space with it". Aggregate metrics
never surface these; error analysis does. Skip it and you
tune the aggregate blindly, and the next iteration is
as likely to move things in the wrong direction as the
right one.

This chapter walks the small set of moves that turn a
metric into a claim about what the model is doing wrong:
looking at the errors themselves, grouping them, sorting
by loss, cross-referencing against inputs, and turning
patterns into concrete hypotheses that a future model
can be measured against.

## The one-slide summary

Every model has three groups of predictions:

- **Correct with high confidence.** The bulk. Reassuring,
  uninteresting.
- **Wrong with high confidence.** The failures that
  matter most. These are the errors your reader will
  point at first.
- **Uncertain (near the decision threshold).** The
  boundary cases. Interesting for calibration and
  threshold tuning; often the source of the highest-
  leverage improvements.

Error analysis is the discipline of sampling from each of
these groups, in proportion to their operational impact,
and asking what the wrong ones have in common.

## The five moves

Five kinds of look you should do on every model, ordered
by the effort each requires:

### 1. Confusion matrix — the two-minute look

For classification, print the confusion matrix. Read the
off-diagonal. Two questions:

- **Which class is confused with which?** In a 10-class
  problem, is the confusion diffuse (each class confused
  with every other) or concentrated (class 3 and class 8
  are indistinguishable, everything else is fine)?
  Concentrated confusion is a data / feature story;
  diffuse confusion is a model-capacity story.
- **Are the errors symmetric?** If class A is misclassified
  as B twice as often as B is misclassified as A, the
  model has an asymmetric error mode. That is a fixable
  finding; it usually points at class imbalance in the
  training set or an asymmetric feature representation.

```python
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)

cm = confusion_matrix(y_true, y_pred, labels=classes)
print(classification_report(y_true, y_pred, target_names=classes))

ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    include_values=True, xticks_rotation=45
)
```

`classification_report` prints per-class precision,
recall, F1, and support. If any class's F1 is much worse
than others, that is the first slice to open.

### 2. Sort by loss — look at what the model got most wrong

The single highest-leverage move. Compute the per-row
loss for every held-out prediction, sort in decreasing
order, and look at the top 20 rows.

```python
import numpy as np
import pandas as pd

# For classification with proba: per-row log loss
proba = clf.predict_proba(X_val)
log_loss_row = -np.log(np.clip(proba[np.arange(len(y_val)), y_val],
                               1e-6, 1.0))

df = X_val.copy()
df["true"] = y_val
df["pred"] = clf.predict(X_val)
df["p_true"] = proba[np.arange(len(y_val)), y_val]
df["loss"] = log_loss_row

df.sort_values("loss", ascending=False).head(20)
```

What you are looking for in the top 20:

- **Label errors.** A high-loss row whose "true" label
  is wrong (annotation mistake, ambiguous case). Common
  and important — a small fraction of label noise puts a
  floor under any model's loss.
- **Systematic feature gaps.** Rows where a critical
  feature is missing / imputed / out-of-distribution.
  Points at a data-quality issue upstream.
- **Genuine hard examples.** Rows that a domain expert
  would also disagree on. These are the *upper bound* of
  what the model can achieve without additional signal.
- **A recognisable pattern.** Ten of the top twenty errors
  come from the same customer segment / date range /
  document type. That is a slice; open it (move 3).

The mental model: **the top 20 errors are the model's
autobiography.** They tell you what it does not know,
what the data does not know, and what the label does not
know. Read them.

For regression, sort by absolute error rather than log
loss. For ranking, sort by inverse rank of the first
relevant item.

### 3. Slice by feature and metric

Aggregate metrics per-slice for every feature that plausibly
matters. The pattern:

```python
import pandas as pd


def per_slice_metric(df, feature, target, pred, metric_fn, min_n=50):
    rows = []
    for value, group in df.groupby(feature):
        if len(group) < min_n:
            continue
        rows.append({
            feature:    value,
            "n":        len(group),
            "metric":   metric_fn(group[target], group[pred]),
        })
    return pd.DataFrame(rows).sort_values("metric")
```

Slices to check on every project:

- **Categorical features** — per-category metric. Country,
  segment, product line, model version.
- **Numerical features bucketed into deciles** — is the
  error concentrated in the extremes?
- **Time / date bucketed by week, month, hour-of-day** —
  seasonal patterns, regime shifts.
- **Interactions between two features** — often
  surprising. "Errors are fine on `country=US` overall
  and fine on `segment=enterprise` overall, but
  concentrated on `country=US AND segment=enterprise`".

The two `min_n` guards matter: excluding tiny slices
prevents noise from dominating; but *state the exclusion*
so the report doesn't hide small but important
subgroups.

Chapter 11's fairness slices are a more disciplined
version of this move focused on sensitive attributes.
The mechanics are the same.

### 4. Cluster the errors

Once slices show *where* the errors concentrate, cluster
the errors themselves to find latent groupings not
captured by any explicit feature. Two approaches:

- **Cluster errors in the feature space.** Fit a `KMeans`
  or `HDBSCAN` on the error rows' features; inspect each
  cluster for common attributes. Reveals hidden
  subpopulations.
- **Cluster errors in a learned embedding.** For text /
  image data, embed the inputs with a sentence
  transformer / vision encoder; project errors into 2D
  with UMAP or t-SNE; look at the visual clusters.

The output of clustering is a hypothesis: "these 40
errors are all short titles about celebrity news; the
model has not learned that category" — a claim you can
test with a targeted training-set addition.

**Do not over-interpret cluster labels.** A cluster is
suggestive, not causal. Verify by pulling actual examples
from the cluster and reading them.

### 5. Compare error sets between two models

When comparing two models (say, an old baseline and a new
model), do not just compare aggregate metrics. Compare
the *sets of errors*:

- **Errors only Model A makes.** What did the change fix?
- **Errors only Model B makes.** What did the change
  break? These are the regressions that aggregate metrics
  smooth away.
- **Errors both models make.** The ceiling — hard cases
  neither model addresses.

The `venn3` visualisation is instructive: three
overlapping circles for {only A, only B, both}. If the
"only B" region is non-trivial, the new model has
regressions that the aggregate improvement is hiding —
a decision-worthy piece of information the aggregate
metric never reveals.

## A worked pattern

The typical error-analysis session on a classification
task:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss


def analyse(clf, X, y, feature_cols_to_slice, k_examples=20):
    proba = clf.predict_proba(X)
    pred  = clf.predict(X)
    # Per-row negative log likelihood of the true class.
    per_row_loss = -np.log(np.clip(
        proba[np.arange(len(y)), y], 1e-6, 1.0
    ))

    df = X.copy().reset_index(drop=True)
    df["true"] = np.asarray(y)
    df["pred"] = pred
    df["p_true"] = proba[np.arange(len(y)), y]
    df["loss"]   = per_row_loss

    # 1. Confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))

    # 2. Top-K worst
    print(f"\nTop {k_examples} highest-loss rows:")
    print(df.sort_values("loss", ascending=False).head(k_examples))

    # 3. Per-slice metric
    for col in feature_cols_to_slice:
        if col not in df.columns:
            continue
        print(f"\nPer-{col} log loss:")
        by_col = df.groupby(col).agg(
            n=("loss", "size"),
            mean_loss=("loss", "mean"),
        ).query("n >= 50").sort_values("mean_loss", ascending=False)
        print(by_col.head(10))

    # 4. Overall
    print(f"\nMean log loss: {df['loss'].mean():.4f}")

    return df
```

Two disciplines when using this:

- **Do the same analysis on the training set.** If the
  same errors show up on training that show up on
  validation, they are a signal the model cannot express
  (capacity or feature limit) rather than an overfitting
  story.
- **Actually read the top-K rows.** The temptation is to
  look at aggregate summaries only. The top-K row-level
  read is where hypotheses come from.

## From analysis to iteration

Error analysis pays off only if it feeds the next
iteration. The pattern:

1. **Look at the errors, in the moves above.**
2. **Write hypotheses.** "The model is 15% worse on
   short-title queries because short titles have less
   context." "The model over-predicts on Fridays because
   the training data was collected mostly on Mon–Thu."
3. **Turn each hypothesis into a measurement.** For each
   hypothesis, name the slice, the current metric on that
   slice, and the target. "Short-title F1 is 0.62; the
   population F1 is 0.78; goal is to close the gap to
   within 0.05."
4. **Address the highest-leverage hypothesis first.**
   Not the easiest, not the most novel — the one whose
   slice has the largest volume × gap. Volume matters
   because a small slice with a large gap moves the
   aggregate very little.
5. **Re-run analysis after each change.** Confirm the
   targeted slice improved *and* that no other slice
   regressed.

The output of every iteration should be: (a) the
aggregate metric, (b) the metric on the addressed slice,
(c) the top-3 regressed slices (if any), and (d) the
next hypothesis.

## Error analysis for regression, ranking, retrieval

The move names are the same; the primitives differ.

- **Regression.** Sort by `abs(y - y_hat)` for the top-K
  worst residuals. Plot residuals vs each feature (a
  residual-vs-x scatter is the classical
  diagnostic-plot for regression). Look for structure
  (heteroscedasticity, systematic bias in a subpopulation)
  in the residual pattern.
- **Ranking.** Sort by the rank of the first relevant
  item per query. Look at the queries where the model
  put the first relevant item at rank > 10. Same slice /
  cluster analysis on the queries themselves.
- **Retrieval.** Look at queries with low PR-AUC.
  Categorise the missing-from-top-k items ("we missed
  because the query has no keyword overlap with the
  correct doc"; "we missed because the correct doc is
  short and BM25 hates it"). Each category is a
  hypothesis.

## Reporting error analysis honestly

Two failure modes in the reporting:

- **Cherry-picked failures.** Showing ten failure
  examples without saying they are the worst-K is
  misleading. State the sampling: "these are the ten
  highest-loss rows on validation".
- **Speculative causes.** "The model fails on short
  sentences because embedding models struggle with short
  context." Maybe. Test it — build a slice, measure,
  compare. Otherwise the report is a story rather than a
  claim.

The template for the section in a model report:

```
Error Analysis
--------------
Method:   sort by per-row log loss on validation
Top 20:   [table of highest-loss rows, with true/pred/proba]
Slices:   [top 5 worst-performing slices with n and metric]
Clusters: [K clusters of errors; brief description of each]
Hypotheses:
  1. Short-title queries have F1=0.62 vs population 0.78.
     Volume 8% of validation traffic.  Impact if closed: +1.3pt aggregate.
  2. ...
Regressions vs previous model:
  - <slice>: +/-<metric points>
```

## Common pitfalls

1. **Only looking at aggregate metrics.** The whole point
   of error analysis is that aggregates hide the
   important information. If the report ends at the
   confusion matrix, you have not done error analysis.
2. **Fixing the noisiest slice.** A slice with 30 rows
   and 40% error rate is noisy; a slice with 3000 rows
   and 10% error rate is where the metric moves. Order
   hypotheses by volume × gap.
3. **Reading only the extreme errors.** The middle of
   the loss distribution (uncertain rows near the
   threshold) is where calibration and threshold tuning
   pay off. Look at those too.
4. **Analysing on the training set only.** Training
   errors are a floor on model capacity but not a story
   about generalisation. Both matter — analyse both.
5. **Not saving the analysis.** A one-off notebook is
   forgotten by the next iteration. Save the analysis
   dataframe as an MLflow artefact so you can compare
   error sets between runs (move 5).
6. **Skipping the label-error check.** If 10% of your
   "errors" are actually label errors, the model may
   already be near the achievable ceiling. Sample and
   spot-check before iterating.
7. **Treating clusters as ground truth.** Clusters are
   hypotheses; verify by reading examples.

## Summary

- Error analysis is the discipline of turning an
  aggregate metric into concrete, testable claims about
  what the model is doing wrong.
- Five moves, in order of effort: confusion matrix,
  top-K worst errors, per-slice metrics, cluster the
  errors, compare error sets across models.
- The top 20 highest-loss rows are the highest-leverage
  read; label errors, systematic feature gaps, and
  recognisable patterns emerge there first.
- The output of error analysis is not "the model has
  problems" but a ranked list of hypotheses with
  volume × gap estimates. Address the highest-leverage
  first, and re-run analysis after each iteration.
- Analyse both training and validation, and save the
  analysis (per-row losses, slice breakdowns) as an
  artefact so you can compare error sets between models.

## Where this goes next

Chapter 11 is a more disciplined version of the "per-slice
metric" move, focused on the slices that matter for
fairness and robustness — sensitive-attribute subgroups,
noise / perturbation robustness, distribution shift — at
the level of thoroughness appropriate for the ML
engineering role at this level.
