# exercise-04 — Production regression post-mortem

> **Estimated effort:** 3 hours.
> **Companion chapters:** [07 — Debugging a production
> regression end-to-end](../07-debugging-a-production-regression.md),
> plus the whole module as background.

## Objective

Simulate a production ML regression end-to-end and
produce the artifact the module trained you for: **a
short, actionable post-mortem with dated, owned action
items and a runbook update that would have caught the
regression faster.**

Unlike the other exercises, this one is largely
narrative and diagnostic. You are not building a new
system component; you are exercising the muscle of
"alert fires → hypothesis → data query → root cause →
mitigation → follow-through → post-mortem." The
artifact is the write-up. The write-up has to be good
enough that a colleague who was not on-call can
reproduce your reasoning.

By the end you should have:

- A `scenarios/` directory containing a synthetic
  incident (prediction logs, feature histograms, and
  a labels stream) that mimics one of the failure
  patterns from chapter 07.
- A `INVESTIGATION.md` walkthrough of your
  step-by-step diagnosis — every query you ran, every
  hypothesis you rejected, and why.
- A `POSTMORTEM.md` in the template from chapter 07
  with a timeline, "what went well / poorly", and
  ≥ 4 dated, owned action items.
- A `RUNBOOK.md` update that would have shortened the
  time-to-detection had it been in place.

## Prerequisites

- Chapter 07 (mandatory). Chapters 02, 03, 05, and 06
  as reference.
- Exercises 01–03 (recommended but not required —
  their artifacts make good starting scaffolding).
- Python 3.11+; `pandas`, `pyarrow`, `numpy`,
  `matplotlib` (for the post-mortem's chart), `jq`
  for log inspection.

## Problem statement

### Part A — Pick a scenario (≈ 15 min)

Pick **one** of the seven regression patterns from
chapter 07's gallery. In `SCENARIO.md`, name the
pattern in one line and add a paragraph explaining
what would make this scenario realistic for a specific
model (the churn model is a fine default; use a
different one if you're building on top of your own
mod-108 work).

The seven patterns, briefly:

1. Upstream schema change (new categorical value).
2. New segment of users (real drift, not a bug).
3. Model rollout regression (v2 worse than v1).
4. Latent preprocessing bug (predictions collapse to
   a constant).
5. Delayed labels arriving from a different source.
6. A retrain that made things worse.
7. Labeler drift (guidelines changed silently).

If you build on top of exercises 01–03, pick a
pattern that exercises the specific instrumentation
you already have. Otherwise, any pattern is fair game.

### Part B — Synthesise the incident (≈ 60 min)

`scenarios/generate.py` — a script that emits the raw
data an on-call ML engineer would see:

- **`scenarios/predictions.parquet`** — 30 days of
  synthetic prediction log rows, with the
  chapter 02 schema. The last 7 days contain the
  incident.
- **`scenarios/features_daily.parquet`** — per-day
  feature summaries (mean, quantiles, category
  counts) so you can drilldown without keeping
  raw features.
- **`scenarios/predictions_labeled.parquet`** — the
  chapter 05 labels-joined table, up to but not
  including the drifted cohort (labels are still
  maturing).
- **`scenarios/drift_scores.parquet`** — the
  chapter 03 drift job's output for each of the 30
  days, with your calibrated thresholds and a
  `breached` flag.
- **`scenarios/retraining_decisions.parquet`** —
  the chapter 06 trigger's outputs for each day.
- **`scenarios/alerts.jsonl`** — the alert events
  that fired.
- **`scenarios/service_logs.jsonl`** — a *sample* of
  raw service logs (chapter 02 format) from the
  incident window, so `jq` queries in Part C have
  something to bite on.

Requirements:

- The incident should be **discoverable**: the alert
  in `alerts.jsonl` names a plausible symptom; the
  drift scores, feature summaries, or labels table
  contain the evidence trail that leads to root
  cause.
- The incident should be **plausible**: values within
  the realistic range of the feature; the timing
  consistent with the scenario ("started on Tuesday
  at 14:03 UTC after a product launch").
- The incident should have at least one **red
  herring**: another feature that looks like it
  drifted a little but was a coincidence. Real
  incidents rarely have exactly one moving part.

You may want to write this generator by starting from
Exercise-01's reference profile and injecting the
incident on top; that's fine.

### Part C — Investigate (≈ 45 min)

Working purely from the files in `scenarios/` (do not
peek at `generate.py`'s incident-specific code while
investigating), execute the chapter 07 procedure and
write `INVESTIGATION.md` in real time:

- **Step 1 — Confirm.** What tooling checks did you
  run? What ruled out a false alert?
- **Step 2 — Localise.** Which features / slices /
  windows did you drill into? Include the actual
  `jq` and `pandas` snippets you used.
- **Step 3 — Trace upstream.** Where did you go next?
  What did you find?
- **Step 4 — Diagnose.** State the root cause in one
  sentence. Then explain your evidence.
- **Include your dead ends.** Any hypothesis you
  rejected and *why*. This is the most useful part
  for the next on-call.
- **Include the queries verbatim.** Not paraphrased.
  The queries are the artifact.

Add one chart (from `matplotlib`, `plotly`, or an
export from your favourite BI tool) that
would have made the diagnosis faster if it had been
on a dashboard. Save it to `scenarios/diagnosis.png`
and reference it from `INVESTIGATION.md`.

### Part D — Mitigate (≈ 15 min)

In `INVESTIGATION.md`, sketch the mitigation you would
have deployed at the time. It doesn't need to run
code; it needs to be specific:

- What is the mitigation (a serving-time fallback, a
  routing rule, a downgraded model version, a
  disabled feature)?
- What's the criterion for removing it?
- What's the instrumentation that tells you it's
  still in effect?
- Who signs off on removing it?

### Part E — Post-mortem (≈ 45 min)

`POSTMORTEM.md` in the chapter 07 template. Sections
must include:

- **TL;DR** — one paragraph. Impact, detection,
  mitigation, resolution.
- **Timeline** — UTC, minute-level for the incident
  window, less granular for the days before.
- **What went well** — three to five bullets.
- **What went poorly** — three to five bullets, each
  actionable (i.e., each paired with an action item).
- **Action items** — at least four, each with:
  - A specific, verifiable outcome ("add an alert on
    `ml_feature_anomaly_total` rate for `plan_tier`,
    threshold X, over window Y").
  - An owner (a role name is fine — "ML on-call
    rotation", "data engineering lead" — you don't
    need to invent people).
  - A due date (a relative date is fine — "within 2
    weeks").
- **Lessons for future on-calls** — three to five
  bullets, phrased as instructions to the next
  person, not as complaints.

### Part F — Runbook update (≈ 20 min)

`RUNBOOK.md` — the file that would have been linked
from the alert. Structure:

- **When this alert fires.** What signal, what
  threshold, what window.
- **First 3 things to check.** Ordered.
- **Common root causes and how to disambiguate
  them.** A short table.
- **Mitigation options.** With criteria for each.
- **Escalation.** Who / where.
- **Post-mortem template link.** So the next on-call
  is not writing from scratch.

Even if you inherited a runbook, you should be able
to point to the specific line you added / changed
because of this incident.

## Acceptance criteria

- `scenarios/generate.py` produces the seven files
  listed in Part B; running it twice with a fixed
  seed produces byte-identical output (for
  reproducibility).
- The incident in the data is discoverable from the
  files alone, without reading `generate.py`.
- `INVESTIGATION.md` walks the chapter 07 procedure
  step-by-step, includes verbatim queries, and
  names at least one rejected hypothesis.
- `scenarios/diagnosis.png` is a real chart that
  would have accelerated the diagnosis.
- `POSTMORTEM.md` follows the chapter 07 template
  and contains ≥ 4 action items with owners and
  due dates.
- `RUNBOOK.md` contains the six sections in Part F
  and includes the specific detection improvement
  that would have caught this incident sooner.
- The whole write-up is understandable to a
  colleague who has read chapter 07 but not the
  scenario code. Verify by having someone else
  read it and try to reconstruct the incident.

## Stretch goals (optional)

- **Two scenarios instead of one.** Pick two
  different chapter-07 patterns, generate both, and
  compare how the same runbook procedure works (or
  doesn't) for each. Add a "when to use this
  runbook / when to escalate to a new runbook"
  section.
- **Chaos-engineering hook.** Wrap
  `scenarios/generate.py` in a small "game-day"
  runner that injects one of several incidents at
  random and asks the reader to diagnose it
  cold. Useful for onboarding.
- **A `docs/timeline.png`.** Render the timeline
  from the post-mortem as a horizontal chart
  (matplotlib or plotly). Cheap; makes the
  post-mortem far more scannable.
- **A cross-team retro.** In the post-mortem, add a
  section identifying the *organisational* changes
  you would recommend (a shared launch checklist, a
  new alert ownership boundary, a new SLA). Frame
  these as proposals with owners, not as demands.
- **Automated regression test.** Turn the incident
  scenario into a scheduled test that runs weekly
  in CI, feeds the synthetic incident through your
  chapter 06 trigger, and asserts the trigger
  produces the expected action. Regressions in the
  trigger's behaviour then get caught before the
  next real incident.

## What to hand in

```
exercise-04/
├── SCENARIO.md                        # one-line pattern + realism paragraph
├── scenarios/
│   ├── generate.py                    # incident synthesiser (reproducible)
│   ├── predictions.parquet
│   ├── features_daily.parquet
│   ├── predictions_labeled.parquet
│   ├── drift_scores.parquet
│   ├── retraining_decisions.parquet
│   ├── alerts.jsonl
│   ├── service_logs.jsonl
│   └── diagnosis.png                  # the chart
├── INVESTIGATION.md                   # step-by-step diagnosis
├── POSTMORTEM.md                      # chapter 07 template
├── RUNBOOK.md                         # what the alert links to
├── pyproject.toml
└── README.md                          # how to run generate.py + reading order
```

Solutions live in the paired `ml-engineer-solutions`
repo — do not peek until you have submitted yours.
