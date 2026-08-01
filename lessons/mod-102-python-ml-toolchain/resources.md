# Resources for mod-102 — Python ML Toolchain: NumPy, Pandas, scikit-learn, Jupyter

A curated short list. The goal is depth-on-demand for the specific
chapter you got stuck on, not a comprehensive bibliography. Primary
documentation is listed first; textbooks, articles, and tooling
references follow.

## NumPy (chapter 02)

- **NumPy — official user guide.**
  <https://numpy.org/doc/stable/user/index.html>
  The "absolute basics for beginners" and "NumPy fundamentals"
  sections cover shape, dtype, and indexing at exactly the altitude
  of chapter 02.
- **NumPy — broadcasting.**
  <https://numpy.org/doc/stable/user/basics.broadcasting.html>
  Read once carefully. This page is responsible for a large fraction
  of shape bugs disappearing from your code once you understand it.
- **NumPy — indexing (views vs. copies).**
  <https://numpy.org/doc/stable/user/basics.indexing.html>
  The canonical reference for the basic-vs-fancy-indexing distinction
  chapter 02 flags as a gotcha.
- **NumPy — random `Generator`.**
  <https://numpy.org/doc/stable/reference/random/generator.html>
  The modern `default_rng()` API. Skim the "quick start" section; use
  it everywhere.
- **NumPy — legacy vs. new random API (NEP 19).**
  <https://numpy.org/neps/nep-0019-rng-policy.html>
  Explains *why* `default_rng()` is preferred over the legacy global
  `np.random.seed` / `np.random.randn` API. Useful context if you
  are convincing a teammate.

## Pandas (chapter 03)

- **Pandas — user guide.**
  <https://pandas.pydata.org/docs/user_guide/index.html>
  Long, but section-navigable. "Indexing and selecting data",
  "Missing data", "Group by", and "Merge" are the four sections that
  correspond directly to chapter 03.
- **Pandas — categorical data.**
  <https://pandas.pydata.org/docs/user_guide/categorical.html>
- **Pandas — nullable integer / boolean / string types.**
  <https://pandas.pydata.org/docs/user_guide/integer_na.html>
- **Pandas — merge validation.**
  <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html>
  The `validate=` argument is documented here. Under-used and
  under-taught.
- **Pandas cookbook.**
  <https://pandas.pydata.org/docs/user_guide/cookbook.html>
  Short recipes. When you need to remember how to do a rolling
  windowed groupby, this is where to look.
- **Modern Pandas (Tom Augspurger).**
  <https://tomaugspurger.net/posts/modern-1-intro/>
  A seven-part blog series on writing idiomatic Pandas. Older but
  the idioms have aged well.

## Jupyter (chapter 04)

- **Jupyter — official documentation.**
  <https://docs.jupyter.org/en/latest/>
- **JupyterLab — user guide.**
  <https://jupyterlab.readthedocs.io/en/stable/user/interface.html>
- **IPython — built-in magic commands.**
  <https://ipython.readthedocs.io/en/stable/interactive/magics.html>
  The complete list of `%` and `%%` magics. `%load_ext autoreload`,
  `%timeit`, `%debug`, `%env` are the ones worth memorising.
- **`nbstripout`.** <https://github.com/kynan/nbstripout>
  Strip notebook outputs on commit so `.ipynb` diffs stay readable.
- **`jupytext`.** <https://jupytext.readthedocs.io/>
  Pair every notebook with a diff-friendly `.py` mirror.
- **`papermill`.** <https://papermill.readthedocs.io/>
  Parameterised, headless notebook execution for reports and reruns.
- **Joel Grus — "I Don't Like Notebooks" (JupyterCon 2018 talk).**
  <https://www.youtube.com/watch?v=7jiPeIFXb6U>
  A famous, opinionated talk that names the notebook failure modes
  chapter 04 warns about. Watch even if — especially if — you love
  notebooks.

## scikit-learn (chapters 05, 06)

- **scikit-learn — user guide.**
  <https://scikit-learn.org/stable/user_guide.html>
  The whole thing is well-written. The most relevant sections for
  this module: 4 (Pipelines), 5 (Cross-validation), 6 (Preprocessing),
  and the tutorial on custom transformers.
- **scikit-learn — `Pipeline`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>
- **scikit-learn — `ColumnTransformer`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>
- **scikit-learn — developing custom estimators.**
  <https://scikit-learn.org/stable/developers/develop.html>
  The canonical guide to the conventions chapter 05 summarises
  (trailing-underscore attributes, `__init__` stores only, `fit`
  returns `self`, etc.).
- **scikit-learn — `check_estimator`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html>
  The conformance test suite for custom estimators.
- **scikit-learn — pipeline visualisation and `set_output`.**
  <https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html>
  Worked example showing `set_output(transform="pandas")`.
- **scikit-learn — common pitfalls and recommended practices.**
  <https://scikit-learn.org/stable/common_pitfalls.html>
  A short official page that covers the exact leakage traps chapter
  06 warns about. Bookmark it.
- **Buitinck et al. — "API design for machine learning software:
  experiences from the scikit-learn project" (2013).**
  <https://arxiv.org/abs/1309.0238>
  The paper that describes *why* the scikit-learn API is shaped the
  way it is. Short and worth reading once — many downstream Python
  libraries copy this design.

## Reproducible environments (chapter 07)

- **`uv` — official documentation.**
  <https://docs.astral.sh/uv/>
  Read at least "Getting started" and "Projects → Working on
  projects". As of 2026 this is the fastest mainstream Python
  project manager.
- **Poetry — official documentation.**
  <https://python-poetry.org/docs/>
  If you inherit a Poetry project, the "Basic usage" and
  "Dependency specification" pages are the essential reading.
- **PEP 440 — version specifiers.**
  <https://peps.python.org/pep-0440/>
  The formal grammar behind `pandas>=2.2,<3` etc.
- **PEP 621 — project metadata in `pyproject.toml`.**
  <https://peps.python.org/pep-0621/>
  The modern standard for declaring project metadata; both `uv` and
  Poetry (as of Poetry 2.x) follow it.
- **PyPA — Python packaging user guide.**
  <https://packaging.python.org/en/latest/>
  The umbrella reference. The "Tutorials → Packaging Python
  Projects" chapter covers the `src/` layout you should be using.
- **PyTorch — install matrix.**
  <https://pytorch.org/get-started/locally/>
  The authoritative source of the CUDA / ROCm / CPU-only wheel URLs
  for a given PyTorch version. Required reading before `mod-105`.
- **Astral — `setup-uv` GitHub Action.**
  <https://github.com/astral-sh/setup-uv>
  Drop-in `uv` install for GitHub Actions. Handles caching.

## Testing, lint, type-check (chapter 08)

- **`pytest` — documentation.**
  <https://docs.pytest.org/en/stable/>
  Skim "Getting started", "How-to guides → Fixtures", and
  "How-to guides → Parametrising tests". That covers most of what
  ML tests need.
- **`pytest` — good practices.**
  <https://docs.pytest.org/en/stable/explanation/goodpractices.html>
  Test layout, conftest, discovery — the pieces that get confusing
  the first time.
- **`hypothesis` — documentation.**
  <https://hypothesis.readthedocs.io/>
  The "Quick start" is enough to write your first property test.
- **`hypothesis.extra.numpy` / `hypothesis.extra.pandas`.**
  <https://hypothesis.readthedocs.io/en/latest/numpy.html>
  Strategies for generating arrays and DataFrames with controlled
  shapes and dtypes. Essential for property-testing ML code.
- **`ruff` — documentation.**
  <https://docs.astral.sh/ruff/>
  "Configuration" and the "Rules" catalogue are the two pages you
  will refer back to.
- **`mypy` — documentation.**
  <https://mypy.readthedocs.io/en/stable/>
- **`pyright` — documentation.**
  <https://microsoft.github.io/pyright/>
- **NumPy — typing.**
  <https://numpy.org/doc/stable/reference/typing.html>
  Documents `numpy.typing.NDArray` and the state of dtype/shape-aware
  typing in the ecosystem.
- **Pandas — type stubs (`pandas-stubs`).**
  <https://github.com/pandas-dev/pandas-stubs>
  Optional but strongly recommended if you type-check code that
  touches Pandas.
- **`pre-commit`.** <https://pre-commit.com/>
  Framework for running `ruff`, `mypy`, `nbstripout`, etc. as git
  hooks. Configure once per repo.
- **Google — "Rules of Machine Learning: Best Practices for ML
  Engineering".**
  <https://developers.google.com/machine-learning/guides/rules-of-ml>
  Not testing-specific, but rules #2 ("first, make metrics"), #23
  ("do not overfit the metric"), and the "Refactoring your pipeline"
  section frame *why* golden tests matter. Free.

## Books

- **VanderPlas, Jake. *Python Data Science Handbook* (2nd ed., O'Reilly,
  2023).** Free HTML at <https://jakevdp.github.io/PythonDataScienceHandbook/>.
  Chapters 2 (NumPy), 3 (Pandas), and 5 (scikit-learn) are exactly at
  the altitude of this module and can substitute for the corresponding
  chapters if you want a longer treatment.
- **McKinney, Wes. *Python for Data Analysis* (3rd ed., O'Reilly,
  2022).** Free HTML at <https://wesmckinney.com/book/>. Written by
  the creator of Pandas; chapters 4–8 are the deepest treatment of
  the Pandas idioms in chapter 03.
- **Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn,
  Keras, and TensorFlow* (3rd ed., O'Reilly, 2022).** The
  scikit-learn chapters (2, 3, 4) walk through end-to-end
  preprocessing + pipeline construction on real data. A useful
  companion to chapters 05–06.
- **Huyen, Chip. *Designing Machine Learning Systems* (O'Reilly,
  2022).** Chapter 8 ("Data Distribution Shifts and Monitoring")
  and chapter 6 ("Model Development and Offline Evaluation") pair
  well with the golden-testing discussion in chapter 08 — testing
  is what catches shift you did not anticipate.
- **Percival, Harry, and Bob Gregory. *Architecture Patterns with
  Python* (O'Reilly, 2020).** Free HTML at <https://www.cosmicpython.com/>.
  Not ML-specific, but the chapters on testing and dependency
  management transfer directly. Read chapters 4 ("Our First Use
  Case: Flask API and Service Layer") and 5 ("TDD in High Gear and
  Low Gear") for the testing mindset.

## Cheat-sheets and quick references

- **NumPy — cheat sheet (official, DataCamp mirror).**
  <https://numpy.org/doc/stable/user/quickstart.html>
  The official "quickstart tutorial" — better maintained than most
  third-party cheat sheets.
- **Pandas — cheat sheet (official PDF).**
  <https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf>
- **scikit-learn — choosing the right estimator.**
  <https://scikit-learn.org/stable/tutorial/machine_learning_map.html>
  The flowchart you will return to whenever you need to remember
  which estimator to try first on a new problem.

<!-- needs-research: on each autonomous research cycle, verify these URLs still resolve and swap in the current stable-version page when a project promotes a new major (e.g., scikit-learn 1.5 → 1.6, Pandas 2.x → 3.x). -->
