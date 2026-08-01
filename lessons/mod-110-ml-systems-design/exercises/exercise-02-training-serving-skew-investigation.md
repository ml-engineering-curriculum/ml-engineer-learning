# exercise-02 — Training-serving skew investigation

> **Estimated effort:** 3 hours.
> **Companion chapter:** [03 — Training-serving skew](../03-training-serving-skew.md).

## Objective

Take a small, realistic ML system with a **deliberately
injected skew bug** and produce two artifacts:

- A **written diagnosis** that names the exact cause
  (which of chapter 03's four canonical causes) and
  points at the specific offending code path.
- A **prevention harness**: a shared transform library,
  a serialised artifact contract, and a CI row-parity
  test that would have caught the bug before
  promotion.

The exercise is not "spot the bug." It is "*build the
tools that would have made spotting the bug automatic*."
Any team can find a skew bug when they know one is
there; the discipline being drilled is that no future
skew bug should slip past the harness you leave behind.

## Prerequisites

- Chapter 03 read.
- Python 3.11+; `scikit-learn ≥ 1.4`, `pandas ≥ 2`,
  `numpy`, `pytest`, `fastapi`, `httpx` (or
  `starlette.testclient`), `joblib`.
- The `mod-108` FastAPI-service scaffold or an
  equivalent scikit-learn-model-behind-FastAPI setup.
  If you don't have one, this exercise's starter code
  gives you a minimal one.

## The starter system (you build or receive)

A small end-to-end system with three components:

- **`training/train.py`** — reads a CSV of user
  activity, fits a preprocessing pipeline plus a
  `LogisticRegression` model, evaluates offline,
  saves `artifact/model.joblib` and a
  `artifact/metrics.json`.
- **`app/serving.py`** — a FastAPI service exposing
  `POST /predict`, which receives a JSON body of
  the same features, preprocesses them, and
  returns a probability.
- **`app/preprocess_serving.py`** — the serving
  path's copy of the preprocessing code (this is
  where the bug lives).

The starter's preprocessing code has one of the
following four bugs **injected on purpose**:

1. The serving path uses a re-instantiated
   `StandardScaler()` on the request payload
   instead of loading the fitted one from the
   artifact. (Cause #3 in chapter 03.)
2. The training path computes a rolling window
   `events_last_30d` from the events warehouse via
   pandas; the serving path re-implements the same
   window differently (off-by-one on the day
   boundary; UTC vs local time). (Cause #2.)
3. The training path one-hot-encodes `plan_tier`
   with a fixed vocabulary derived from the training
   set; the serving path re-derives the vocabulary
   from the current request payload, producing a
   different column order. (Cause #1.)
4. The training path reads `country` from the
   `users` warehouse table; the serving path reads
   `country` from the `sessions` warehouse table.
   The two are usually the same and occasionally
   differ. (Cause #4.)

You can either **implement the starter yourself**
(in which case you pick which bug to inject and hide
it from your test harness) or **use the starter
provided in the paired solutions repo's
`starter/exercise-02/` directory** (which contains a
mystery bug you are meant to diagnose).

## Requirements

### Part A — Reproduce the skew (≈ 30 min)

Write `investigate/reproduce.py`. It should:

1. Load the same 20 canonical fixture rows from
   `investigate/fixtures/skew_probe.json`.
2. Run each row through the **training-time**
   preprocessor (loaded from the artifact) and
   through the **serving-time** preprocessor (the
   FastAPI service's transform), and collect both
   feature vectors.
3. Print a table showing, per row, the
   training-time vector, the serving-time vector,
   and the element-wise diff. Highlight rows where
   the two disagree.
4. Save the table as `investigate/reproduce.md`.

The fixture must cover:

- 5 "normal" rows.
- 5 rows with a missing value in at least one
  feature.
- 5 rows on a day boundary (e.g., activity
  timestamps at 23:59 and 00:01 UTC on the same
  calendar day).
- 5 rows with an unusual category value.

Any of the four injected bugs should surface in one
of these four buckets.

### Part B — Diagnose (≈ 45 min)

Write `investigate/DIAGNOSIS.md`. Structure:

1. **Symptom.** One paragraph: what you observed
   (e.g., "on 8 of the 20 fixture rows, the
   `events_last_30d` feature differs by up to 3
   between the training and serving paths; on
   fixture rows where the timestamp crosses a day
   boundary, the disagreement is always +1 on the
   serving side").
2. **Root cause.** Name which of chapter 03's four
   canonical causes this is; point at the exact
   file:line in the serving code (or the training
   code) that produces the divergence; explain in
   one paragraph why it produces this specific
   symptom.
3. **The offline metric that did not catch it.**
   Look at `artifact/metrics.json` from the
   training run. Explain in one paragraph why the
   offline metric was fine despite the bug.
4. **A hypothetical incident.** Sketch what the
   production incident would look like — which
   users would be affected, how, and how long
   until anyone noticed. Reference `mod-109`
   chapter 07's playbook shape.

Length target: one page.

### Part C — Fix the bug (≈ 30 min)

Write the fix. The specific pattern depends on
which cause you diagnosed:

- **Cause #1 or #2 (two implementations / time-
  window mismatch):** create a `features/`
  package with a single implementation of the
  transform; import it from both the training
  code and the serving code; delete the
  duplicate.
- **Cause #3 (missing fitted preprocessor):**
  ensure the fitted preprocessor is inside the
  `sklearn.Pipeline` that gets saved to
  `artifact/model.joblib`; ensure the serving
  path loads the pipeline (not a fresh
  preprocessor).
- **Cause #4 (feature source drift):** rename the
  source in one place, document the canonical
  source in `features/README.md`, and add a
  smoke test that reads from both sources and
  asserts equivalence on a sampled window.

Commit the fix in a separate commit from the
prevention harness (Part D). The diff should be
small — usually 10-50 lines.

### Part D — Prevention harness (≈ 60 min)

Write `tests/test_row_parity.py`. It must:

1. Load the same fixture from Part A.
2. Load the training-time pipeline from the
   artifact.
3. Load the serving-time pipeline from the FastAPI
   app's import path (not by re-implementing it).
4. For each fixture row, assert byte-equal (or
   `np.allclose(..., atol=1e-9)`) between the two
   pipelines' output feature vectors.
5. On failure, print the row ID, the training
   vector, the serving vector, and the element
   diff — the same format Part A's reproducer
   uses.

Additionally, extend the CI pipeline (if you have
one; otherwise, describe it in `README.md`) to:

- **Fail the promotion** if `test_row_parity`
  fails.
- **Compute the reference profile** (`mod-109`
  chapter 02's format) as part of the training
  job, and include it in the artifact so the
  feature-anomaly counter has a baseline.
- **Emit a signed manifest** that lists every
  file in the artifact directory and their
  SHA256s — the artifact's integrity contract.

Finally, write a `PREVENTION.md` that lists all
four of chapter 03's canonical causes and explains,
one paragraph each, how the harness you just built
would catch each one (or explicitly, would *not*
catch some of them and what additional check is
needed).

### Part E — Shadow-mode plan (≈ 15 min)

Write `SHADOW.md` — a one-page rollout plan for the
fixed model. Include:

- What "shadow mode" means for this service
  (candidate model runs alongside; predictions
  logged; no user impact).
- What the shadow-mode step measures (per-request
  feature parity, prediction distribution
  similarity, latency of the candidate).
- What the pass/fail criteria are for exiting
  shadow mode and entering the A/B test
  (chapter 05).
- What alert fires if shadow-mode reveals a
  divergence.

## Starter guidance

- **The reproducer is the doorway.** Get Part A
  working before you look at Part B. If your
  reproducer doesn't show a divergence, you
  cannot debug the bug; if it shows a divergence
  on every row, the reproducer itself has a bug.
- **Look at the diff, not the values.** Two long
  vectors with one different element look the
  same at a glance. Print the elementwise diff.
- **The row-parity test in Part D is the point
  of the exercise.** Everything else supports it.
  If the test is not blocking-in-CI, the harness
  isn't real.
- **Do not silence the failure.** A tempting
  shortcut is to `atol=1e-3` when the real
  tolerance is `1e-9`; the loose tolerance hides
  the next bug. Fix the bug; keep the tolerance
  strict.
- **The fix commit and the harness commit are
  separate.** Reviewers should be able to look at
  the harness commit alone and be convinced that
  it would have caught the original bug —
  without the fix in the tree.

## Acceptance criteria

- `investigate/reproduce.py` runs against the
  fixture and prints a diff table showing at
  least one row of divergence between the
  training-time and serving-time feature vectors.
- `investigate/DIAGNOSIS.md` names which of
  chapter 03's four canonical causes the bug is;
  points at file:line for the divergence; and
  explains the specific symptom-to-cause link.
- The fix commit changes the offending code path.
  Running the reproducer after the fix shows the
  divergence is gone.
- `tests/test_row_parity.py` **fails** on the
  pre-fix commit and **passes** on the post-fix
  commit. Verify this by running the test at
  each commit.
- `PREVENTION.md` walks all four of chapter 03's
  canonical causes and names the harness step
  that catches each (or names the additional
  check needed).
- `SHADOW.md` describes a one-page rollout plan
  with pass/fail criteria and alerting.
- The artifact contract is signed — a manifest of
  files and SHA256s ships with the artifact.

## Stretch goals

- **A second injected bug.** Inject a second bug
  (from a different one of chapter 03's four
  causes) and run the harness. Does it catch the
  new bug? If not, extend the harness. Write up
  what changed.
- **A feature-anomaly counter test.** Extend the
  serving code with the counter from `mod-109`
  chapter 02; write a test that a serving request
  with an unknown categorical value increments
  the counter.
- **A shadow-mode simulation.** Actually implement
  shadow mode: run the candidate and control
  models against the same fixture of "production"
  requests; produce a report of prediction
  distribution similarity (KS, JSD) and
  per-request feature parity.
- **A cross-language reproducer.** If your team
  has a serving path in a language other than
  Python (Go, Java), write a small test that
  runs the same fixture through the alternate
  language's implementation and compares to the
  Python training-time output. This is the
  hardest and most valuable form of row-parity
  testing.

## What to hand in

```
exercise-02/
├── training/
│   └── train.py                            # from starter or your own
├── app/
│   ├── serving.py                          # from starter
│   ├── preprocess_serving.py               # from starter (buggy)
│   └── main.py                             # from starter
├── features/                               # NEW — shared transform (post-fix)
│   ├── __init__.py
│   ├── transform.py
│   └── README.md
├── artifact/
│   ├── model.joblib
│   ├── metrics.json
│   ├── reference_profile.json              # NEW — from mod-109 ch 02
│   └── MANIFEST.sha256                     # NEW — signed manifest
├── investigate/
│   ├── reproduce.py                        # NEW
│   ├── reproduce.md                        # NEW — diff table
│   ├── DIAGNOSIS.md                        # NEW
│   └── fixtures/
│       └── skew_probe.json                 # NEW
├── tests/
│   ├── test_row_parity.py                  # NEW
│   └── test_service.py                     # from starter
├── PREVENTION.md                           # NEW
├── SHADOW.md                               # NEW
├── pyproject.toml
└── README.md                               # how to run everything
```

Solutions live in the paired `ml-engineer-solutions`
repo — do not peek until you have submitted yours.
