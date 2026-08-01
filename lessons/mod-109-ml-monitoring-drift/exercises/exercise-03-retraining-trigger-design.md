# exercise-03 — Retraining trigger design

> **Estimated effort:** 3 hours.
> **Companion chapters:** [05 — Ground-truth collection loops](../05-ground-truth-collection.md),
> [06 — Retraining triggers](../06-retraining-triggers.md).

## Objective

Design and implement an auditable retraining trigger
that reads monitoring state, applies a decision rule,
and produces a well-scoped, idempotent trigger event.

By the end you should have:

- A `MonitoringState` schema and a `TriggerEvent`
  schema — pydantic (or dataclass) models, small,
  precise, versioned.
- A `decide_action` pure function that maps
  `MonitoringState → Action` (one of `DO_NOTHING`,
  `INVESTIGATE`, `RETRAIN`, `ROLLBACK`).
- A test suite that codifies each decision path with
  representative inputs.
- A `simulate.py` that replays 90 days of synthetic
  monitoring history through the trigger and produces
  a `retraining_decisions.parquet` audit log.
- A **written decision rubric** in `DECISIONS.md`
  that a code reviewer could evaluate without
  reading the code.

The exercise is design-heavy on purpose. Half the value
is defending your rule; the other half is building the
machinery that makes it safe to change.

## Prerequisites

- Chapters 05 and 06 (mandatory) and chapters 03–04
  for context on the signals.
- Python 3.11+; `pydantic ≥ 2`, `pandas`, `pyarrow`,
  `pytest`, `PyYAML`.
- Exercises 01 and 02 (or equivalent) to have realistic
  shapes of `MonitoringState`. If you skipped them,
  synthesise plausible values from the chapter 06
  narrative.

## Problem statement

### Part A — Data model (≈ 30 min)

In `trigger/models.py`, define the schemas:

```python
class MonitoringState(BaseModel):
    at:                          datetime
    model_name:                  str
    model_version:               str
    days_since_last_retrain:     int

    # Drift signals (chapter 03)
    multivariate_drift:          float
    n_feature_alerts:            int
    consecutive_drift_windows:   int
    top_drifted_features:        list[str]

    # Prediction shift (chapter 04)
    prediction_shift:            float

    # Measured accuracy (chapter 05)
    rolling_accuracy:            float | None       # None if no matured labels
    reference_accuracy:          float
    reference_accuracy_std:      float
    slice_accuracy:              dict[str, float]   # slice name → accuracy
    slice_reference:             dict[str, float]

    # Previous version, if a recent rollout
    previous_version:            str | None
    previous_version_accuracy:   float | None

class Action(str, Enum):
    DO_NOTHING = "do_nothing"
    INVESTIGATE = "investigate"
    RETRAIN = "retrain"
    ROLLBACK = "rollback"

class TriggerEvent(BaseModel):
    decided_at:                  datetime
    decision_rule_version:       str    # git SHA or semver
    action:                      Action
    trigger_reason:              str    # human-readable
    state:                       MonitoringState
    training_data_window:        tuple[date, date] | None
    reference_holdout_id:        str | None
    incumbent_model_version:     str
```

Field-level validators: values in `[0, 1]` where the
signal is bounded; accuracy fields must be non-
negative; slice dicts must have matching keys between
current and reference.

### Part B — The decision rule (≈ 45 min)

In `trigger/rule.py`:

```python
def decide_action(state: MonitoringState,
                  policy: TriggerPolicy) -> TriggerEvent:
    ...
```

Design constraints your rule must respect:

- **Rollback wins.** If measured accuracy drops
  sharply below the previous version's accuracy,
  `ROLLBACK` regardless of other signals.
- **Drift alone does not retrain.** A drift signal
  without a corroborating accuracy drop, prediction
  shift, or sustained duration → `INVESTIGATE`, not
  `RETRAIN`.
- **Cooldown.** If `days_since_last_retrain < policy.
  cooldown_days` and the action would be `RETRAIN`,
  downgrade to `INVESTIGATE` (unless it's a
  rollback).
- **Debounce.** If `consecutive_drift_windows <
  policy.debounce_windows`, do not `RETRAIN` on drift
  alone.
- **Hysteresis.** If the model was in a "drift" state
  and is now below threshold, do not immediately
  cancel — require N clean windows first (encode
  this in the `policy`, not in the signal itself).
- **Scheduled floor.** If `days_since_last_retrain >=
  policy.scheduled_retrain_days`, `RETRAIN` (subject
  to the guardrail evaluation gate downstream —
  which is not this exercise, but your `trigger_
  reason` should make the reason explicit).
- **Slice regression.** If any slice's accuracy has
  regressed by more than `policy.slice_regression_
  threshold` vs. its reference, `RETRAIN` even if
  aggregate accuracy is stable.

Encode the policy as a `TriggerPolicy` pydantic model
loaded from `policy.yaml`. The rule is a pure function
of `(state, policy)`; no I/O.

### Part C — Tests (≈ 45 min)

`tests/test_rule.py` — one test per named path in the
rule. At minimum:

- **`test_do_nothing_when_all_signals_calm`.**
- **`test_scheduled_retrain_fires_at_floor`.**
- **`test_rollback_on_sharp_accuracy_drop_vs_previous`.**
- **`test_drift_alone_produces_investigate_not_retrain`.**
- **`test_debounce_blocks_retrain_before_n_windows`.**
- **`test_cooldown_downgrades_retrain_to_investigate`.**
- **`test_slice_regression_forces_retrain`.**
- **`test_multivariate_drift_with_accuracy_drop_retrains`.**
- **`test_missing_measured_accuracy_falls_back_to_estimated_or_defers`.**
- **`test_trigger_event_carries_reason_and_rule_version`.**

Each test builds a representative `MonitoringState` and
asserts the resulting `TriggerEvent.action` and the
`trigger_reason` field.

Use `pydantic.BaseModel.model_copy(update={...})` (or
your framework's equivalent) to keep the test setup
tight. Property tests (`hypothesis`) are welcome but
not required.

### Part D — The simulation (≈ 45 min)

`trigger/simulate.py`:

- Generates 90 days of synthetic `MonitoringState`
  timesteps at hourly (or daily — your choice)
  cadence.
- Includes at least three planted incidents:
  1. A sudden multivariate drift spike on day 20,
     lasting 5 days.
  2. A measured accuracy drop on day 45 that
     recovers naturally by day 55 (the drift was
     benign).
  3. A rollout regression: a new `model_version`
     ships on day 70; measured accuracy on the new
     version is materially worse than the previous
     one; the trigger should `ROLLBACK`.
- Runs `decide_action` on each state; appends the
  resulting `TriggerEvent` to
  `retraining_decisions.parquet`.
- Prints a short summary at the end: number of
  `RETRAIN`, `ROLLBACK`, `INVESTIGATE`, `DO_NOTHING`;
  the median time from incident onset to first
  `RETRAIN` or `ROLLBACK`; false-positive rate
  (retraining events not attributable to a planted
  incident).

### Part E — The decision rubric (≈ 30 min)

`DECISIONS.md` — the doc a code reviewer reads first.
Structure:

- **What signals does the rule watch?** One-line
  description of each field of `MonitoringState`.
- **What actions does the rule emit?** The four,
  with one-line semantics.
- **What are the thresholds?** The values in your
  `policy.yaml`, with a justification for each — why
  is `multivariate_drift ≥ 0.20` the retrain
  threshold and not `0.15`? Reference the calibration
  procedure from exercise-02 where relevant.
- **What are the edge cases?** The `MonitoringState`
  shapes that surprised you when writing tests, and
  how the rule handles them.
- **What is not automated?** The specific decisions
  that route to `INVESTIGATE` and require a human.
  A one-paragraph "here is what the human should
  check" for each.
- **How is the rule versioned and changed?** The
  process for updating `policy.yaml` (or the rule
  itself) — how does the audit log capture the
  change, and what tests are required?
- **How is the rule audited?** The
  `retraining_decisions.parquet` schema and the
  queries a data scientist would run to reconstruct
  a specific past decision.

The rubric is graded on clarity, not comprehensiveness.
A colleague on-call at 03:14 has to be able to read it
in 5 minutes and understand what the trigger will (and
won't) do.

## Acceptance criteria

- `trigger/models.py`, `trigger/rule.py`, and
  `trigger/simulate.py` exist and are typed.
- `decide_action` is a pure function: no I/O; same
  input → same output; no hidden state.
- Every named path in `decide_action` has at least
  one test in `tests/test_rule.py`; `pytest -q`
  reports zero failures.
- Running `python -m trigger.simulate` produces
  `retraining_decisions.parquet` with rows for each
  timestep and a printed summary.
- The three planted incidents in the simulation are
  each handled correctly (the summary line names
  each incident and the action taken).
- `DECISIONS.md` covers each of the six rubric
  sections; a colleague could read it in ≤ 5 minutes
  and explain the rule's behaviour back to you.
- The `policy.yaml` is present, loaded by the rule,
  and the tests use it (do not hardcode thresholds
  inside the rule).

## Stretch goals (optional)

- **Coverage report.** Add `pytest --cov=trigger`
  and target ≥ 90% coverage on `rule.py`.
- **Property tests.** Use `hypothesis` to generate
  `MonitoringState` and assert invariants: the
  rule never emits `ROLLBACK` when
  `previous_version is None`; `INVESTIGATE` and
  `RETRAIN` are never emitted with the same
  `trigger_reason`; etc.
- **Web hook / notification integration.** When the
  action is `RETRAIN`, `ROLLBACK`, or
  `INVESTIGATE`, POST the `TriggerEvent` to a
  configurable webhook (mock in tests). Include a
  link to the runbook from chapter 07.
- **Bandit test.** Run the simulation twice — once
  with your rule, once with a strawman "retrain on
  any drift" rule. Compare the number of retrains,
  false-positive rate, and time-to-detection.
  Write up the trade-off in `DECISIONS.md`.
- **Policy diff report.** Add a script that takes
  two `policy.yaml` files and re-runs the
  simulation with each; diffs the resulting
  actions. Useful when you change a threshold and
  want to see the downstream effect.
- **Human-in-the-loop UI.** A minimal Streamlit
  page (or Jupyter notebook) that surfaces
  pending `INVESTIGATE` events and lets a
  reviewer approve → convert to `RETRAIN`. Not
  production-grade; the point is to think through
  the review UX.

## What to hand in

```
exercise-03/
├── trigger/
│   ├── __init__.py
│   ├── models.py                       # pydantic schemas
│   ├── rule.py                         # decide_action + TriggerPolicy
│   └── simulate.py                     # 90-day replay + planted incidents
├── tests/
│   └── test_rule.py                    # per-path tests
├── policy.yaml                         # thresholds
├── retraining_decisions.parquet        # generated by simulate.py
├── pyproject.toml
├── README.md                           # how to run
└── DECISIONS.md                        # rubric doc
```

Solutions live in the paired `ml-engineer-solutions`
repo — do not peek until you have submitted yours.
