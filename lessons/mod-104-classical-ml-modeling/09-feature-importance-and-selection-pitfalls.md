# 09 — Feature importance and selection pitfalls

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 04–06 (trees and ensembles), 07 (CV).

## Motivation

"Which features matter?" is the second question a stakeholder asks
after "what's the accuracy?" It sounds simple. It is not.

Every widely-used feature-importance method — impurity-based
importance for trees, permutation importance, drop-column
importance, absolute coefficient magnitude for linear models — gives
answers that are subtly wrong under conditions you will absolutely
encounter: correlated features, high-cardinality categoricals,
different scales, test-set leakage. And most feature-selection
procedures built on top of these importance measures inherit the
biases and add their own.

This chapter is a catalogue of what each method actually measures,
where it lies to you, and how to use several methods together to
get an answer you can defend. Chapter 10 covers SHAP separately —
it is not immune to these issues either, but it deserves its own
chapter.

## Method 1: coefficient magnitude (linear models only)

For a linear or logistic regression, the natural importance is
`|β_j|`.

Caveats:

- **Scale-dependent.** `|β_j|` depends on the scale of `x_j`.
  Always compare on standardised features. Chapter 03 harps on this;
  the reason it matters here is that "top-10 coefficients" reported
  from an unscaled fit is nonsense.
- **Conditional on other features.** `β_j` is the effect of `x_j`
  holding all other features constant. In the presence of
  correlated features, individual coefficients can be very small
  even when the underlying feature is strongly predictive — its
  effect is "explained by" the correlated partner.
- **Unstable under correlated features.** Ridge stabilises;
  Lasso zeros arbitrarily. Bootstrap to get a stability estimate
  (chapter 03).

For non-linear models, coefficient magnitude does not apply.
`RandomForest.feature_importances_` is a different quantity, covered
next.

## Method 2: impurity-based importance (trees and forests)

`RandomForestClassifier.feature_importances_` and its friends
compute the total impurity reduction attributable to each feature,
summed over all splits in all trees and normalised to sum to 1.

```python
import pandas as pd
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(20))
```

This is fast (computed at fit time, no additional passes) and
almost always the first importance measure you look at. It is also
biased in three specific ways, all of which are documented in
`sklearn`'s user guide:

### Bias toward high-cardinality features

A feature with many unique values (a continuous variable, a
high-cardinality categorical) offers more candidate split points.
More candidate splits mean more chances of finding one that
happens to reduce impurity on the training set — even for a
feature that is pure noise.

The result: impurity-based importance systematically overestimates
the importance of high-cardinality features and underestimates
low-cardinality ones. In an experiment where the "important"
features are all binary and the "irrelevant" features are all
continuous with many uniques, impurity importance can rank the
noise above the signal. Strobl et al. (2007) documented this and
proposed permutation importance as a partial fix — same story
applies to modern implementations.<!-- needs-research: cite Strobl,
Boulesteix, Zeileis, Hothorn (2007) "Bias in random forest
variable importance measures: illustrations, sources and a
solution", BMC Bioinformatics:
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25 -->

### Bias toward the "top of the tree"

A feature that wins the root split gets credit for a huge impurity
reduction on the whole dataset. Features used deeper in the tree
each account for smaller reductions on fewer rows. The importance
signal is amplified for the top split winner.

### Computed on the training set

Impurity reduction is measured on the same data the tree was fit
to. A feature that overfits — reduces training impurity by
memorising noise — will look important even when it does not
generalise.

The `sklearn` user guide explicitly warns:

> The impurity-based feature importances computed on tree-based
> models suffer from two flaws that can lead to misleading
> conclusions. First they are computed on statistics derived from
> the training dataset and therefore do not necessarily inform us
> on which features are most important to make good predictions on
> held-out dataset. Secondly, they favor high cardinality features,
> that is features with many unique values.

Use it as a smoke test, not as the deliverable.

## Method 3: permutation importance

For each feature `j`:

1. Score the fitted model on a hold-out set.
2. Permute the values of column `j` in the hold-out set.
3. Score again.
4. Report the drop in score as feature `j`'s importance.

Repeat with several permutations per feature for a mean and std.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    scoring="roc_auc",
    random_state=0,
    n_jobs=-1,
)

import pandas as pd
imp = pd.DataFrame({
    "mean":  result.importances_mean,
    "std":   result.importances_std,
}, index=X.columns).sort_values("mean", ascending=False)
```

Advantages over impurity-based:

- **Computed on held-out data.** Reflects generalisation, not
  training fit.
- **Model-agnostic.** Works for any estimator with a `.score` or a
  `.predict` and a metric.
- **Directly interpretable.** "How much does the score drop when
  this feature is randomised?"

Caveats — permutation importance has its own failure modes:

### Correlated features share credit

If two features are highly correlated, permuting one still leaves
the model with the other. Both features look unimportant. This is
the "correlated feature dilution" problem.

Two mitigations:

- **Group-permute correlated features together.** Cluster features
  by correlation (or hierarchical clustering on the correlation
  matrix) and permute clusters, not individual features. Reports
  cluster-level importance.
- **Report both group and per-feature importance** with a note.

### Requires a re-scored model per feature per permutation

`n_features × n_repeats` scoring passes. Cheap on
small-medium data; slow on 100k+ rows × 100+ features. Sub-sample
the hold-out if it matters.

### Extrapolation into unseen combinations

Permuting a feature can create combinations of `(x_j, x_{-j})` that
never appeared in training. If the model behaves wildly on such
inputs (a tree extrapolating, a neural net in an unexplored region),
the reported importance can be exaggerated. This is a research area;
the practical fix is to look at *conditional* permutation importance
or to fall back to SHAP for correlated features.

### Test set vs. training set

Permutation on the *test* set measures generalisation importance —
"how much does test accuracy rely on this feature." Permutation on
the *training* set measures fitted-model importance — "how much
does the fitted model use this feature." Both are useful; report
which one you used.

## Method 4: drop-column importance

The most conceptually clean: refit the model without feature `j`
and compare CV scores. Rarely used because it costs `n_features ×
cv` refits (expensive on gradient boosting), and because it, too,
handles correlated features badly (drop one and the correlated
partner picks up the slack).

If you have the budget for it — say, `< 30` features and a fast
estimator — it is the least biased of the model-agnostic methods.

## Method 5: SHAP

Chapter 10 covers this at length. Two points here for completeness:

- SHAP values are the theoretically-grounded generalisation of
  "importance", giving each feature a per-row contribution to the
  prediction that sums to the model's output.
- TreeSHAP (for trees / ensembles) is exact and fast. `KernelSHAP`
  and `PermutationExplainer` (for other models) are approximate
  and slower.

SHAP does not solve the correlated-features problem — no method
does, cleanly — but it is the most rigorous of the tools available.

## Comparing methods on the same problem

The right check is not "which method is correct" but "do the
methods agree?" A feature that ranks in the top 10 by impurity,
permutation, and SHAP is almost certainly important. A feature
that ranks top by one and bottom by another needs a closer look.

```python
import pandas as pd
from sklearn.inspection import permutation_importance

impurity = pd.Series(rf.feature_importances_, index=X.columns)
perm     = pd.Series(
    permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=0).importances_mean,
    index=X.columns,
)
compare = pd.DataFrame({"impurity": impurity, "permutation": perm})
compare["impurity_rank"]    = compare["impurity"].rank(ascending=False)
compare["permutation_rank"] = compare["permutation"].rank(ascending=False)
compare["rank_diff"]        = compare["impurity_rank"] - compare["permutation_rank"]
print(compare.sort_values("rank_diff", key=abs, ascending=False).head(10))
```

Big rank differences are the interesting rows: features flagged by
impurity but not by permutation (high-cardinality noise), features
flagged by permutation but not by impurity (buried by a stronger
splitter but genuinely predictive on the hold-out).

## Feature selection: methods and their traps

Given importance scores, the natural next step is to drop
"unimportant" features. Four common approaches, each with its own
pitfall.

### 1. Filter methods

Drop features whose univariate association with the target is weak
(e.g. `f_classif`, `mutual_info_classif`, `chi2`).

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=20)
```

**Pitfall.** Univariate methods miss feature interactions. Two
features that predict the target only in combination each look
useless individually and both get dropped. Cheap and fast, but not
appropriate for tabular ML where interactions are the point.

### 2. Wrapper methods

Iteratively add or remove features, evaluating with CV. Recursive
feature elimination (RFE) is the canonical example.

```python
from sklearn.feature_selection import RFECV
selector = RFECV(estimator=LogisticRegression(), cv=5, scoring="roc_auc")
```

**Pitfall.** Expensive (`n_features` refits per step) and prone to
selection bias — the CV score of the selected feature set is
optimistic because the selection used the same data.

### 3. Embedded methods

The estimator itself does the selection. Lasso zeros out
coefficients; tree ensembles report zero-importance features.

```python
from sklearn.feature_selection import SelectFromModel
selector = SelectFromModel(estimator=LassoCV())
```

**Pitfall.** Selection stability. Lasso on correlated features
selects one and drops the correlated partner arbitrarily — the
"selected" feature list changes across bootstrap samples. Report
selection *frequency* over resampling, not a single selected list.

### 4. `SelectFromModel(GradientBoosting…)`

The most common in tabular practice: fit a gradient booster, keep
the top-k features by impurity, refit downstream models on the
reduced set.

**Pitfall — biggest of all.** Impurity-based importance is biased
(see above). "Top-k by impurity" carries every one of those
biases into the selected feature set. Use permutation importance
or SHAP for the ranking, not impurity.

## The train-then-select-then-train-again trap

The single most common selection bug:

```python
# WRONG
selector.fit(X, y)                           # fits on the whole dataset
X_selected = selector.transform(X)
scores = cross_val_score(clf, X_selected, y, cv=5)   # CV sees "selected" features
```

The selector saw every row, including the CV validation folds. The
selected feature set has been *chosen using the validation folds'
targets*. The subsequent CV score is optimistic. This is textbook
target leakage.

The fix: wrap selection in a `Pipeline`:

```python
# RIGHT
pipe = Pipeline([
    ("select", SelectFromModel(estimator=RandomForestClassifier(n_estimators=200))),
    ("clf",    LogisticRegression(max_iter=1000)),
])
scores = cross_val_score(pipe, X, y, cv=5)   # selector refits per fold
```

Now the selector is refit inside each CV training fold; the
validation fold's targets never influence the selected features.

This same trap eats a lot of Kaggle-tutorial pipelines. It is
worth grepping for `fit_transform` on the whole dataset in any
codebase you inherit.

## Stability selection

Because most selection methods are unstable, the honest thing to
report is not "the 20 selected features" but "the 20 features
selected in ≥ 80% of bootstrap resamples":

```python
from sklearn.linear_model import LassoCV
from sklearn.utils import resample
import numpy as np, pandas as pd

selections = []
for _ in range(200):
    X_b, y_b = resample(X_train, y_train)
    lasso = LassoCV(cv=5, max_iter=10_000).fit(X_scaled_b, y_b)
    selections.append(lasso.coef_ != 0)

freq = pd.Series(np.mean(selections, axis=0), index=X.columns)
stable = freq[freq >= 0.8].sort_values(ascending=False)
```

Meinshausen and Bühlmann (2010) — "Stability selection" — is the
canonical reference for this technique.<!-- needs-research: cite
Meinshausen & Bühlmann 2010, JRSS-B:
https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00740.x -->

Stability selection over Lasso is the go-to when you need to *name*
the important features for a scientific / regulatory audience.

## When feature selection helps, and when it doesn't

Feature selection is often oversold. On tabular problems with a
few hundred features, the score improvement from selection is
typically small (a percent or less) — modern tree ensembles are
robust to noise features. Where selection actually pays off:

- **Serving cost.** Fewer features = smaller lookup table = cheaper
  inference. Common in ranking systems where the model is called
  millions of times per hour.
- **Data collection cost.** Some features come from expensive
  vendors or slow queries. Dropping them saves real money.
- **Interpretability.** A 10-feature model is easier to explain
  than a 100-feature one. Regulators care.
- **Very wide problems (`n_features ≫ n_samples`).** Genomics,
  fMRI, text with term-frequency features. Here selection is
  necessary and Lasso (or stability-selection Lasso) is the
  standard tool.

For a typical tabular business problem, do not spend a week on
feature selection. Fit your gradient booster, drop features whose
permutation importance is indistinguishable from zero, and move on.

## Pitfalls collected

1. **Reading `RandomForest.feature_importances_` as ground truth.**
   Biased toward high-cardinality features. Cross-check with
   permutation importance.
2. **Ranking features from an unscaled linear model by
   `|coef|`.** Scale-dependent nonsense. Standardise first.
3. **Feature selection on the full dataset before CV.** Target
   leakage. Wrap in `Pipeline`.
4. **Univariate filtering on interaction-rich problems.** Drops
   features that only predict when combined. Prefer embedded /
   model-based selection.
5. **Trusting a single selection run on correlated features.**
   Which correlated feature survives is coin-flip stable.
   Bootstrap and report selection frequency.
6. **Permutation importance on the training set + not
   understanding what you measured.** `permutation_importance(X_train,
   y_train)` measures how much the fitted model uses each feature;
   `(X_test, y_test)` measures generalisation. Both valid; be
   explicit.
7. **Reporting `feature_importances_` from a model tuned with
   validation-set leakage.** Everything downstream inherits the
   leak.
8. **Dropping "unimportant" features and reporting a higher CV
   score.** If your CV was correctly nested around the selection,
   score should be similar or slightly worse; if it went up
   substantially, you have contaminated the CV.

## Summary

- Impurity-based importance is fast, first-pass, biased. Do not
  ship it.
- Permutation importance is model-agnostic and hold-out-based.
  Preferred for reporting; be careful with correlated features
  (permute clusters, or use SHAP).
- Drop-column importance is the cleanest but most expensive.
- Every feature-selection method has to be inside a `Pipeline` and
  refit per CV fold. Otherwise you are selecting on the validation
  set.
- Selection on correlated features is inherently unstable.
  Bootstrap and report selection *frequency*, not a single list.
- Selection usually gains little in raw accuracy; it wins on
  serving cost, data cost, interpretability, and on very wide
  problems.

## Where this goes next

Chapter 10 introduces SHAP — the additive feature-attribution
framework that is currently the most rigorous tool for explaining
individual predictions, and for aggregating those explanations into
a global importance ranking that avoids most of chapter 09's
pitfalls.
