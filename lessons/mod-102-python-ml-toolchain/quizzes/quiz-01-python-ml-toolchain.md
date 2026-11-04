# quiz-01 — Python ML Toolchain: NumPy, Pandas, scikit-learn, Jupyter

> **Format:** 12 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–08 of this module.
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter the point
> comes from — go re-read that section if you missed the question.

---

## Questions

### 1. (Chapter 01) Why does chapter 01 argue the packaging and testing parts of this module are worth a whole module on their own, alongside the libraries?

A. Because `pip` is being removed from Python 3.12 and every project needs
   a replacement.
B. Because idioms compound (two-line vectorised NumPy replaces 20-line
   loops at ~100× speed), and reproducibility is a first-class
   engineering concern — models that "worked yesterday" and no longer
   reproduce are a load-bearing failure mode of the field.
C. Because scikit-learn's API is deprecated and packaging tools are
   the long-term replacement.
D. Because `uv` and `poetry` are a job-interview topic and the
   curriculum optimises for interviews.

### 2. (Chapter 02) You run `X.sum()` on a 2-D array `X` of shape `(1000, 32)`, expecting a per-column total. What is the problem and the fix?

A. There is no problem — `X.sum()` sums along axis 0 by default.
B. `X.sum()` sums **all** elements and returns a scalar. The chapter's
   rule is "always pass the explicit `axis=` you intend" — use
   `X.sum(axis=0)`.
C. `X.sum()` sums along axis 1 by default; swap in `X.sum(axis=0)` to
   get the correct direction.
D. `X.sum()` raises on 2-D input; you must call `X.sum(axis=0)` to make
   it work at all.

### 3. (Chapter 02) Which call pattern does the chapter recommend for making stochastic NumPy code reproducible, and why?

A. `np.random.seed(0)` followed by `np.random.randn(...)`, because the
   legacy global API is the officially supported one.
B. `rng = np.random.default_rng(seed=0)`, because the `Generator` API
   is explicit, threadable, and reproducible — the legacy global is
   discouraged for new code because it relies on hidden global state.
C. `random.seed(0)` from the Python standard library, because NumPy's
   RNG delegates to it.
D. Setting `PYTHONHASHSEED=0` in the environment, which seeds both
   NumPy and Python RNGs.

### 4. (Chapter 03) You run `users.merge(orders, on="user_id", how="left")` and get ten times as many rows as `users` had. Which single change to the merge would have failed loudly at the point where the assumption broke, instead of silently producing a Cartesian blow-up?

A. Add `how="inner"`.
B. Add `sort=True`.
C. Add `validate="one_to_many"`.
D. Replace `merge` with `join`; `join` is stricter by default.

### 5. (Chapter 03) You want to add a column `age_demeaned` that equals each row's `age` minus the mean `age` within its `country`. Which idiom from the chapter is the right tool?

A. `df.groupby("country")["age"].agg("mean")` and manually broadcast back.
B. `df.apply(lambda row: row["age"] - df[df.country == row.country]["age"].mean(), axis=1)`.
C. `df["age"] - df.groupby("country")["age"].transform("mean")` — `transform`
   broadcasts a per-group statistic back to every input row.
D. `df.pivot(index="country", values="age", aggfunc="mean")` and merge
   back on `country`.

### 6. (Chapter 04) A teammate's notebook produces nice-looking plots on screen, but when you hit **Restart and Run All** half the cells raise `NameError`. What does chapter 04 say is going on and how should you respond?

A. This is a bug in Jupyter's execution order; filing an issue upstream
   is the right fix.
B. This is the "hidden state" failure mode — cell state tracks what was
   *run*, not what is *on screen*. Treat a notebook that does not
   survive Restart and Run All as a bug, and fix it now while you
   remember which cell does what.
C. This is expected — notebooks are inherently irreproducible, and the
   plots are what matters.
D. This is an `autoreload` issue; adding `%autoreload 2` will fix it.

### 7. (Chapter 04) You find yourself copy-pasting the same 20-line "clean and featurise this DataFrame" cell into a third notebook. According to the chapter's "promote out" discipline, what is the right next move?

A. Nothing — copy-paste across notebooks is normal exploratory workflow.
B. Leave it in notebooks but add a comment saying "DO NOT EDIT" above
   each copy.
C. Move the logic into a `.py` module under `src/`, import it from
   each notebook, and (once a colleague wants to reuse it) give it a
   unit test and a real function signature.
D. Freeze the notebook outputs with `nbstripout` so the cells cannot
   drift.

### 8. (Chapter 05) You write a custom transformer. Which of the following best matches scikit-learn's conventions as described in the chapter?

A. `__init__` validates and transforms its arguments (e.g. clamping
   `lower` into `[0, 1]`); learned state is stored on
   `self.fitted_state`; `fit` returns `None`.
B. `__init__` stores hyperparameters unchanged under the same names
   (`self.lower = lower`); learned state is stored on attributes ending
   with an underscore (`self.lower_bounds_`); `fit` returns `self`.
C. `__init__` is empty; every parameter is passed into `fit`; learned
   state lives on class attributes so it is shared across instances.
D. `__init__` accepts `*args, **kwargs` so the transformer is
   forward-compatible; `fit` mutates `X` in place and returns `X`.

### 9. (Chapter 05) Why does the chapter insist on passing an integer `random_state=` to any scikit-learn estimator that uses randomness?

A. Because `random_state=None` is being removed in scikit-learn 2.0.
B. Because `random_state` also controls the learning rate on tree-based
   models.
C. Because without it, two runs of the same code on the same data
   disagree on the third decimal place — the resulting "which bug is
   this?" debugging day is pure waste, and reproducible numbers are
   the building block `mod-106` (experiment tracking) rests on.
D. Because `n_jobs=-1` only parallelises when `random_state` is set.

### 10. (Chapter 06) You run `cross_val_score(estimator, X_train, y_train, cv=5)` after `X_train["age"] = X_train["age"].fillna(X_train["age"].median())`. Which statement about leakage applies?

A. No leakage — the median was computed from `X_train` only, and
   `X_test` was never touched.
B. Leakage, but only if `X_test` had NaNs.
C. Leakage across folds: every CV fold uses the **same** median
   computed from the full training set, so each inner-validation fold
   is imputed with a statistic that saw its own rows. The fix is to
   put `SimpleImputer` inside a `Pipeline` so it is re-fit on each
   inner training fold.
D. Not leakage, but inefficient — `transform` would have been faster.

### 11. (Chapter 06) In a `GridSearchCV` over the pipeline `Pipeline([("preprocess", ColumnTransformer([("num", num_pipe, numeric_cols)])), ("clf", LogisticRegression())])`, which key tunes the imputer's `strategy` inside `num_pipe`'s `impute` step?

A. `"preprocess.num.impute.strategy"`
B. `"preprocess/num/impute/strategy"`
C. `"preprocess__num__impute__strategy"`
D. `"strategy"` — the grid search auto-resolves nested names.

### 12. (Chapter 07) Your CI pipeline currently runs `pip install -r requirements.txt`. The requirements file was generated from `pip freeze`. Which chapter-07 critique applies most directly, and what is the recommended fix?

A. `pip freeze` captures *what is installed* but not the *resolution
   graph*, and leaks platform-specific wheels; it is not a real
   lockfile. Switch to `uv` / `poetry`, commit the real lockfile
   (`uv.lock` / `poetry.lock`), and run `uv sync --frozen` (or
   `poetry install --no-update`) in CI — which refuses to update the
   lockfile and fails the build if `pyproject.toml` has drifted from
   it.
B. Nothing — `pip freeze` is the recommended lockfile mechanism in the
   chapter.
C. Switch to `conda`, which is the only reproducible option on Linux.
D. Delete `requirements.txt`; letting `pip` resolve "latest" on every
   run is more reproducible because you always pick up security fixes.

### 13. (Chapter 08) You already have unit tests on each transformer in your pipeline. Your teammate argues this makes a golden test redundant. According to chapter 08, what do golden tests catch that per-transformer unit tests tend to miss?

A. Nothing — they are strictly redundant with good unit tests.
B. Golden tests catch *numerical drift* in end-to-end predictions
   caused by subtle changes nobody thought to unit-test — the imputer's
   median changed because the input schema changed, the encoder's
   category order flipped, the scaler's mean is now computed across the
   whole input instead of per-column. The interface is still intact;
   the numbers quietly moved. Golden tests declare that end-to-end
   output will not change without you noticing.
C. Golden tests only catch performance regressions, not correctness
   regressions.
D. Golden tests replace type checking; with a golden test in place,
   `mypy` adds no value.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **B.** Chapter 01 ("Why a whole module on this") lists exactly two
   reasons: idioms compound (one vectorised expression replaces many
   loops and runs ~100× faster; one well-shaped `Pipeline` replaces a
   200-line imperative script *and* eliminates a class of leakage
   bugs), and reproducibility is a first-class engineering concern in
   ML — "worked yesterday" bugs from upgraded transitive dependencies
   are a load-bearing failure mode. The packaging and testing parts of
   the module exist specifically to prevent that.

2. **B.** Chapter 02 ("Pitfalls collected", item 2): `x.sum()` with no
   axis sums *all* elements — "rarely what you mean when `x` is 2-D".
   The chapter's rule is "always pass the explicit `axis=` you intend",
   so `X.sum(axis=0)` is the fix for per-column totals.

3. **B.** Chapter 02 ("The random API: use `default_rng`, not the
   legacy global") recommends `np.random.default_rng(seed)` explicitly:
   "explicit, threadable, reproducible". The legacy
   `np.random.seed(...)` / `np.random.randn(...)` relies on hidden
   global state and is called out as a reproducibility hazard.

4. **C.** Chapter 03 ("Joins (`merge`)"): `validate=` takes values like
   `"one_to_one"`, `"one_to_many"`, `"many_to_one"`, `"many_to_many"`
   and raises if your cardinality assumption is wrong. The chapter
   singles it out as "this single argument has caught many a 'row count
   exploded silently because the key wasn't unique' bug".

5. **C.** Chapter 03 ("Group-by and aggregate"): `transform` returns a
   value for every input row, broadcasting the group statistic back,
   so `df["age"] - df.groupby("country")["age"].transform("mean")`
   computes the demeaned column in one expression. The chapter calls
   out `df.apply(..., axis=1)` as a code smell.

6. **B.** Chapter 04 ("Why notebooks are dangerous"): hidden state is
   the first failure mode listed — cell execution is order-independent
   but kernel state tracks what was run, not what is on screen.
   "Notebooks that work when you walk away can be irreproducible the
   moment the kernel restarts." The recommended response is to fix it
   now, treating a non-Restart-and-Run-All-clean notebook as a bug.

7. **C.** Chapter 04 ("The 'promote out' discipline"): "The moment you
   copy-paste a cell, that logic earns the right to live in a `.py`
   module." Reuse by a colleague earns it a unit test. The mental model
   is that a notebook is a *consumer* of library code, not the source
   of it.

8. **B.** Chapter 05 ("Conventions" and "Custom transformers"):
   hyperparameters go to `__init__` and are stored unchanged under the
   same name (so `get_params` is faithful), learned state lives on
   `self.<name>_` (trailing underscore), and `fit` must return `self`
   — pipelines depend on it. Breaking any of these means the rest of
   the ecosystem (cross-validation, grid search, pipelines) silently
   stops working with your estimator.

9. **C.** Chapter 05 ("Random state: reproducibility and
   `random_state`"): "Without `random_state`, two runs will disagree
   on the third decimal place and you will burn a day debugging it."
   It is the building block `mod-106` (experiment tracking) rests on.

10. **C.** Chapter 06 ("Leakage, concretely"): computing the median on
    the full training set before `cross_val_score` means every CV fold
    is imputed with a statistic that has already seen the fold's own
    rows. The chapter's recommended fix is to wrap the imputer in a
    `Pipeline`, so it is re-fit on the inner training portion of every
    CV split.

11. **C.** Chapter 06 ("`Pipeline`: a chain of named steps"):
    scikit-learn flattens nested parameter names with double
    underscores — `"<step>__<param>"`, composing with nested pipelines
    as `"preprocess__num__impute__strategy"`. The chapter notes "the
    double-underscore is not a typo; it's how scikit-learn flattens
    nested parameter names".

12. **A.** Chapter 07 ("Pitfalls collected", item 2): `pip freeze` is
    explicitly called out as *not* a substitute for a real lockfile —
    "it captures what is installed but not the resolution graph. It
    also leaks platform-specific wheels." The chapter's recommended
    pattern is `uv` / `poetry` + committed lockfile + `uv sync
    --frozen` (or `poetry install --no-update`) enforced in CI.

13. **B.** Chapter 08 ("Golden / snapshot tests for models and
    pipelines"): the chapter's motivation is that many refactor bugs
    in ML pipelines do not break the interface — the pipeline still
    returns an array of probabilities — but break the numerics. Unit
    tests miss these; golden tests "declare what should not change
    without your noticing".

</details>
