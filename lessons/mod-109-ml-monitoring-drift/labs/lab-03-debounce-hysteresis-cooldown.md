# lab-03 — Debounce, hysteresis, and cooldown on a noisy drift time series

> **Duration:** ~35–45 minutes.
> **Companion chapter:** [06 — Retraining triggers](../06-retraining-triggers.md).

## Objective

Reproduce chapter 06's three anti-noise patterns — **debounce**,
**hysteresis**, and **cooldown** — on a 90-day synthetic drift signal you
can run on a laptop. By the end you will have (a) generated a drift-score
time series with per-day noise, one **short false-positive spike**, and
one **sustained real incident**, (b) run a **naïve "retrain on any
above-threshold day"** decision rule and watched it retrain many times —
once per spike plus a whole cluster during the incident — (c) added
**debounce** ("N consecutive windows") and watched the false-positive
spikes disappear while the real incident still fires once, (d) added
**hysteresis** (separate enter/exit thresholds) and watched the trigger
stop flapping when the incident signal wobbles around a single number,
(e) added **cooldown** (M days of silence after any retrain) and watched
the flurry of alerts inside a sustained incident collapse into a single
retrain, and (f) written an auditable `retraining_decisions` row for
every firing — the shape chapter 06 insists on so that "why did the
model retrain last Tuesday?" is a query, not archaeology.

This lab is deliberately narrower than **exercise-03** (which builds a
full `decide_action` pure function, a `TriggerPolicy` pydantic model, a
per-path test suite, a 90-day simulation with three planted incidents
including a `ROLLBACK` scenario, and a `DECISIONS.md` rubric). The lab is
the isolation chamber where you *see* each of the three anti-noise
patterns subtract exactly one failure mode from the raw "retrain on
threshold" rule; the exercise then wires all three into a reviewable,
testable, auditable trigger.

## Prerequisites

- Python 3.11+.
- `numpy`, `pandas`, `matplotlib` installed (`uv venv && uv pip install
  numpy pandas matplotlib` is fine).
- Chapter 06 read through once. Chapter 03 (for the drift-score
  vocabulary) and chapter 04 (for the "prediction shift is a symptom"
  context) are useful background. Lab-01 (PSI calibration) and lab-02
  (the Scenario-A ramp) are feeders; neither is required.
- A CPU is enough.

## Setup

Create the lab directory and a single file:

```bash
mkdir -p lab-03-anti-noise && cd lab-03-anti-noise
touch trigger_lab.py
```

Start `trigger_lab.py` with imports and a seed:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=0)
```

## Steps

### Step 1 — Generate a 90-day noisy drift signal with a short spike and a real incident (≈ 7 min)

Chapter 06's anti-noise patterns are designed against two specific shapes
of input: (a) a single-window **false-positive spike** (a lone noisy
sample, a window boundary issue, a transient upstream blip) that a naïve
rule would retrain on, and (b) a **sustained real incident** that lasts
days and should fire exactly one retrain, not one per day inside it.

Synthesise both in a single 90-day time series — baseline drift ~ `0.08`
with daily noise, a single-day spike to `0.35` on day 20 (false
positive), and a 7-day plateau at `0.30` on days 55–61 (real incident).
You'll alert against a single threshold `0.20` — a value the lab-01
calibration procedure would output on a feature whose backfill p99 is
about `0.12`.

Append:

```python
N_DAYS = 90
BASELINE = 0.08
DAILY_NOISE_SIGMA = 0.02

scores = BASELINE + RNG.normal(0.0, DAILY_NOISE_SIGMA, N_DAYS)

# One-day false-positive spike (think: an upstream ETL job wrote a
# duplicate partition; the drift job's window contained double the real
# mass of a tail bucket).
scores[20] = 0.35

# A 7-day plateau — a real sustained drift incident starting day 55. The
# signal also wobbles around the threshold inside the incident — some
# days hit 0.28, some touch 0.19 — which is what hysteresis exists for.
incident_days = range(55, 62)
incident_values = [0.28, 0.32, 0.19, 0.31, 0.22, 0.30, 0.27]
for day, val in zip(incident_days, incident_values):
    scores[day] = val

# A later, briefer incident that isn't quite long enough to be real
# (two days, not three) — debounce should swallow it.
scores[80] = 0.24
scores[81] = 0.22

timeline = pd.DataFrame({"day": np.arange(N_DAYS), "score": scores})

# Alert threshold — the value lab-01's calibration procedure would emit
# on a feature whose backfill p99 is about 0.12.
THRESHOLD = 0.20

print("days above threshold:",
      sum(scores > THRESHOLD),
      "  (expected: ~8–10 — spike + real incident + brief blip)")
```

Draw the signal:

```python
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(timeline["day"], timeline["score"], marker="o", markersize=3)
ax.axhline(THRESHOLD, color="red", linestyle="--",
           label=f"threshold={THRESHOLD}")
ax.set_xlabel("day"); ax.set_ylabel("drift score")
ax.set_title("ch 06 — synthetic drift signal")
ax.legend()
fig.tight_layout()
fig.savefig("signal.png", dpi=120)
```

**Expected output.** A time series with a flat baseline around 0.08,
occasional wobbles around 0.10, one isolated point at 0.35 on day 20, a
dense cluster of points in the 0.19–0.32 range on days 55–61, and a
short two-day bump on days 80–81. The red threshold line sits at 0.20.
The printed count is ~8–10 above-threshold days.

**Checkpoint.** You have three explicit things a decision rule has to
handle: (1) the day-20 spike (one window; should **not** retrain), (2)
the days-55–61 sustained incident (seven windows; should retrain
**once**), and (3) the days-80–81 blip (two windows, still below the
"three consecutive" debounce threshold; should **not** retrain).

### Step 2 — The naïve "retrain on any threshold cross" rule (≈ 5 min)

Chapter 06's starting point is a rule as simple as possible: `score >
threshold → RETRAIN`. Run it, count the retrains, and see which ones are
false positives.

Append:

```python
def naive_rule(timeline: pd.DataFrame, threshold: float) -> list[dict]:
    """Chapter 06's anti-pattern: retrain on any above-threshold day.
    No debounce, no hysteresis, no cooldown. The rule the chapter spends
    the rest of its length patching."""
    events = []
    for _, row in timeline.iterrows():
        if row["score"] > threshold:
            events.append({
                "day": int(row["day"]),
                "action": "RETRAIN",
                "trigger_reason": f"score={row['score']:.3f} > {threshold}",
            })
    return events


naive = naive_rule(timeline, THRESHOLD)
print(f"naive rule — retrains: {len(naive)}")
for e in naive:
    print(f"  day {e['day']:>2}: {e['action']}  ({e['trigger_reason']})")
```

**Expected output.** Something like:

```
naive rule — retrains: 9
  day 20: RETRAIN  (score=0.350 > 0.2)
  day 55: RETRAIN  (score=0.280 > 0.2)
  day 56: RETRAIN  (score=0.320 > 0.2)
  day 58: RETRAIN  (score=0.310 > 0.2)
  day 60: RETRAIN  (score=0.300 > 0.2)
  day 61: RETRAIN  (score=0.270 > 0.2)
  day 80: RETRAIN  (score=0.240 > 0.2)
  day 81: RETRAIN  (score=0.220 > 0.2)
  # (plus any day inside the real incident whose score was ≥ threshold)
```

Three distinct failure modes in one output: **day 20** fires on a single
noisy sample (false positive — chapter 06 pitfall #2 "No cooldown"
meets chapter 06 pitfall #1 "Retraining on drift alone"); **days
55/56/58/60/61** fire **multiple retrains inside one incident** (the
"self-reinforcing bad-retrain loop" starts here); **days 80–81** fire on
a too-short blip. Nine retrains to cover one real incident is not a
monitoring surface anyone would trust.

**Checkpoint.** The raw rule is honest — the signal did cross the
threshold — but every anti-noise pattern from chapter 06 is subtracting
a specific bug on top of this honesty. Steps 3–5 add them one at a
time.

### Step 3 — Add debounce: require N consecutive windows (≈ 7 min)

Chapter 06: *"Debounce. Require the signal to be present in N consecutive
windows before firing. A single hour of drift is not a retrain; three
consecutive days probably is."*

Append:

```python
def debounced_rule(timeline: pd.DataFrame, threshold: float,
                   debounce_windows: int) -> list[dict]:
    """Chapter 06 debounce: require `debounce_windows` consecutive days
    above threshold before firing."""
    events = []
    streak = 0
    for _, row in timeline.iterrows():
        if row["score"] > threshold:
            streak += 1
            if streak == debounce_windows:
                events.append({
                    "day": int(row["day"]),
                    "action": "RETRAIN",
                    "trigger_reason": (
                        f"{debounce_windows} consecutive windows > "
                        f"{threshold}; current={row['score']:.3f}"
                    ),
                })
        else:
            streak = 0
    return events


debounced = debounced_rule(timeline, THRESHOLD, debounce_windows=3)
print(f"debounced rule (N=3) — retrains: {len(debounced)}")
for e in debounced:
    print(f"  day {e['day']:>2}: {e['action']}  ({e['trigger_reason']})")
```

**Expected output.** Something like:

```
debounced rule (N=3) — retrains: 1 or 2
  day 57 or 58: RETRAIN  (3 consecutive windows > 0.2; current=0.310)
  # (plus any additional day inside days 59–61 that re-arms and crosses
  # the threshold again — hence 1 or 2)
```

- **Day 20** (single spike) — gone. Debounce needs three consecutive
  windows; one does not qualify.
- **Days 80–81** (two-day blip) — gone, for the same reason.
- The **real incident on days 55–61** — fires. But if the signal
  **wobbles** inside the incident (and in this lab, day 57 sits at 0.19,
  *below* the threshold), the debounce streak resets and the rule may
  re-arm before firing again. That re-arming is the flapping behaviour
  hysteresis fixes in the next step.

**Checkpoint.** Debounce subtracts the "one noisy sample" and "too-short
blip" failure modes. It does **not** subtract the "signal wobbles
around a single threshold inside the incident" problem — that one is
hysteresis.

### Step 4 — Add hysteresis: separate enter and exit thresholds (≈ 7 min)

Chapter 06: *"Hysteresis. The threshold to enter the alerting state is
higher than the threshold to exit it. Prevents the trigger from flapping
when the signal wobbles around a single threshold."*

Append:

```python
def hysteresis_rule(timeline: pd.DataFrame,
                    enter_threshold: float,
                    exit_threshold: float,
                    debounce_windows: int) -> list[dict]:
    """Chapter 06 hysteresis + debounce.
    Enter alerting state when the score crosses `enter_threshold` upward
    and stays for `debounce_windows` consecutive days. Stay in the
    alerting state until the score falls below `exit_threshold`.
    Requires enter_threshold > exit_threshold."""
    assert enter_threshold > exit_threshold
    events = []
    in_alert = False
    streak = 0
    for _, row in timeline.iterrows():
        if not in_alert:
            if row["score"] > enter_threshold:
                streak += 1
                if streak == debounce_windows:
                    events.append({
                        "day": int(row["day"]),
                        "action": "RETRAIN",
                        "trigger_reason": (
                            f"{debounce_windows} consecutive > "
                            f"{enter_threshold}; current={row['score']:.3f}"
                        ),
                    })
                    in_alert = True
                    streak = 0
            else:
                streak = 0
        else:
            # Stay in alert until the signal falls below exit_threshold.
            if row["score"] < exit_threshold:
                in_alert = False
    return events


hysteretic = hysteresis_rule(
    timeline,
    enter_threshold=0.20,
    exit_threshold=0.15,          # chapter 06: strict less-than
    debounce_windows=3,
)
print(f"debounced + hysteretic — retrains: {len(hysteretic)}")
for e in hysteretic:
    print(f"  day {e['day']:>2}: {e['action']}  ({e['trigger_reason']})")
```

**Expected output.** Something like:

```
debounced + hysteretic — retrains: 1
  day 57 or 58: RETRAIN  (3 consecutive > 0.2; current=0.310)
```

Hysteresis holds the trigger in the "already alerted" state across the
day-57 wobble to `0.19` (still > the exit threshold of `0.15`), so the
rule does not re-arm on day 58's `0.31`. The whole seven-day incident
produces **one** firing. Chapter 06's "**Without these, a busy
monitoring system can retrain every day and never converge**" turns into
a one-row audit log entry.

**Checkpoint.** You have now subtracted three of the four chapter-06
anti-noise failure modes (single spike, too-short blip, flapping inside
the alerting state). The one remaining failure mode — retraining again
immediately after a retrain while the signal is still above threshold
— is cooldown's job in the next step.

### Step 5 — Add cooldown: M days of silence after any retrain (≈ 5 min)

Chapter 06: *"Cooldown. After a retrain, no new retrain for M days (or
until the next labelled cohort matures). Stops the trigger from firing
on the *new* model's cold-start noise."*

The debounced+hysteretic rule in step 4 already produces one firing
for the days-55–61 incident. But a **more adversarial signal** — the
same incident but shaped so the signal resets *below* the exit
threshold and re-crosses the enter threshold four days later — would
produce two firings, the second one sitting inside the cold-start
window of the first retrained model (chapter 06's "self-reinforcing
bad-retrain loop" opening pattern). Add cooldown and verify it
collapses the double fire into one.

Append:

```python
# Craft an adversarial signal variant to exercise cooldown: two short
# plateaus separated by a 2-day dip below the exit threshold.
adv_scores = BASELINE + RNG.normal(0.0, DAILY_NOISE_SIGMA, N_DAYS)
# First plateau days 55–57 at 0.28; dip to 0.10 days 58–59; second
# plateau days 60–62 at 0.26. Without cooldown, the hysteretic rule
# fires on day 57 and again on day 62. With cooldown=14 days, the
# second firing is swallowed.
for day, val in [(55, 0.28), (56, 0.30), (57, 0.26),
                 (58, 0.10), (59, 0.09),
                 (60, 0.25), (61, 0.27), (62, 0.26)]:
    adv_scores[day] = val
adv_timeline = pd.DataFrame({"day": np.arange(N_DAYS), "score": adv_scores})


def full_rule(timeline: pd.DataFrame,
              enter_threshold: float,
              exit_threshold: float,
              debounce_windows: int,
              cooldown_days: int) -> list[dict]:
    """Chapter 06 debounce + hysteresis + cooldown — the three
    anti-noise patterns in one function."""
    assert enter_threshold > exit_threshold
    events = []
    in_alert = False
    streak = 0
    last_retrain_day = -10_000
    for _, row in timeline.iterrows():
        day = int(row["day"])
        if day - last_retrain_day < cooldown_days:
            # In the cooldown window. Keep tracking alert state so the
            # audit log reflects reality, but swallow any RETRAIN.
            streak = 0
            in_alert = False
            continue
        if not in_alert:
            if row["score"] > enter_threshold:
                streak += 1
                if streak == debounce_windows:
                    events.append({
                        "day": day,
                        "action": "RETRAIN",
                        "trigger_reason": (
                            f"{debounce_windows} consec > "
                            f"{enter_threshold}; current={row['score']:.3f}"
                        ),
                    })
                    in_alert = True
                    last_retrain_day = day
                    streak = 0
            else:
                streak = 0
        else:
            if row["score"] < exit_threshold:
                in_alert = False
    return events


without_cd = hysteresis_rule(adv_timeline,
                             enter_threshold=0.20,
                             exit_threshold=0.15,
                             debounce_windows=3)
with_cd = full_rule(adv_timeline,
                    enter_threshold=0.20,
                    exit_threshold=0.15,
                    debounce_windows=3,
                    cooldown_days=14)

print(f"adversarial signal — no cooldown: {len(without_cd)} retrain(s)")
for e in without_cd:
    print(f"  day {e['day']:>2}: {e['action']}  ({e['trigger_reason']})")
print(f"adversarial signal — with cooldown=14d: {len(with_cd)} retrain(s)")
for e in with_cd:
    print(f"  day {e['day']:>2}: {e['action']}  ({e['trigger_reason']})")
```

**Expected output.** Two retrains without cooldown (one on each plateau,
days 57 and 62 ± debounce); **one** retrain with cooldown. The first
firing arms a 14-day silence; the second plateau falls inside it and is
swallowed.

**Checkpoint.** Four chapter-06 failure modes, four patches: debounce
(single spike), debounce again (too-short blip), hysteresis (flapping
inside the alerting state), cooldown (re-firing during the new model's
cold start). With all three patterns on, the rule's retrain count is
dominated by *real* incidents — nothing else.

### Step 6 — Write the audit row the chapter insists on (≈ 5 min)

Chapter 06: *"Every decision writes a row to a `retraining_decisions`
table with the state that produced it. When someone asks 'why did the
model retrain last Tuesday?', the answer is a query."*

Promote the `events` dicts you've been printing into a persisted table
so the audit discipline is real, not a list comprehension.

Append:

```python
# Run all four rules on the ORIGINAL (non-adversarial) 90-day timeline
# and build one unified audit log capturing (rule_version, day, action,
# trigger_reason) — chapter 06's "every decision writes an audit row".
rules = {
    "v0-naive":                        naive_rule(timeline, THRESHOLD),
    "v1-debounce":                     debounced_rule(timeline, THRESHOLD,
                                                      debounce_windows=3),
    "v2-debounce+hysteresis":          hysteresis_rule(
                                           timeline,
                                           enter_threshold=0.20,
                                           exit_threshold=0.15,
                                           debounce_windows=3),
    "v3-debounce+hysteresis+cooldown": full_rule(
                                           timeline,
                                           enter_threshold=0.20,
                                           exit_threshold=0.15,
                                           debounce_windows=3,
                                           cooldown_days=14),
}
audit_rows = []
for rule_version, events in rules.items():
    for e in events:
        audit_rows.append({
            "rule_version":  rule_version,
            "day":           e["day"],
            "action":        e["action"],
            "trigger_reason": e["trigger_reason"],
        })
audit = pd.DataFrame(audit_rows)
audit.to_parquet("retraining_decisions.parquet", index=False)
print("wrote retraining_decisions.parquet")
print(
    audit.groupby("rule_version")
    .agg(retrains=("action", "count"),
         days=("day", lambda d: sorted(d.tolist())))
    .reset_index()
    .to_string(index=False)
)
```

**Expected output.** A summary table showing the retrain count collapsing
from `v0 → v1 → v2 → v3`:

```
                    rule_version  retrains                                       days
                       v0-naive         8  [20, 55, 56, 58, 60, 61, 80, 81]   (ish)
                    v1-debounce         1 or 2                                 [57-61]
         v2-debounce+hysteresis         1                                      [57-61]
v3-debounce+hysteresis+cooldown         1                                      [57-61]
```

And a `retraining_decisions.parquet` on disk, keyed by `(rule_version,
day)`, with the audit `trigger_reason` string for each firing. Chapter
06 pitfall #5 ("The trigger fires; nobody knows why") is now a query:

```python
# Preview the chapter-06 audit query from chapter 07's catalogue.
import pandas as pd
df = pd.read_parquet("retraining_decisions.parquet")
print(df[df["rule_version"] == "v3-debounce+hysteresis+cooldown"])
```

**Checkpoint.** You have a reproducible artifact of the rule's behaviour
across four versions on the same input. A reviewer who wants to know
whether to loosen `debounce_windows` from 3 to 2 can diff the resulting
audit tables; a policy change is reviewable from the row delta, not from
a meeting note. Chapter 06's "Changes go through review" discipline
rests on this file existing.

### Step 7 — The one-line summary of what each pattern actually bought you (≈ 3 min)

Chapter 06's three patterns map onto specific failure modes; write them
down in a `NOTES.md` so the mapping is in the repo next to the code, not
in your head.

```markdown
# lab-03 — what each anti-noise pattern subtracted

| pattern     | subtracts                                                       | parameter in this lab            |
|-------------|-----------------------------------------------------------------|-----------------------------------|
| debounce    | single-window spikes and too-short blips                        | `debounce_windows=3`             |
| hysteresis  | flapping when the signal wobbles around a single threshold      | `enter=0.20`, `exit=0.15`        |
| cooldown    | re-firing during the new model's cold-start / unlabelled window | `cooldown_days=14`               |

Rules of thumb from chapter 06:
- **Debounce**: three consecutive windows is a good default for daily
  windows; drop to 2 for hourly windows if the fast-window noise is
  low.
- **Hysteresis**: `exit = 0.75 * enter` is a reasonable starting
  point; tune from the backfill (lab-01) by asking "how far does the
  real signal wobble below the threshold once the incident is
  underway?"
- **Cooldown**: match it to the model's **cold-start window** —
  roughly the time it takes for the new version's prediction
  distribution to stabilise and for the first mature labels of the
  new version to arrive. Usually 7–30 days. Too short and chapter 06
  pitfall #2 ("no cooldown — circular") comes back; too long and
  real new incidents get silenced.

Rollback is **not** subject to cooldown. Chapter 06 "Rollback wins
over everything": `rolling_accuracy_delta < -0.05` fires `ROLLBACK`
immediately regardless of these three patterns. Rollback is a
first-class action, not a special case of retrain.

Audit: every firing writes a row to `retraining_decisions.parquet`
with `(rule_version, day, action, trigger_reason)`. A policy change
diffs the rows, not the meeting notes.
```

**Checkpoint.** You have the chapter-06 three-pattern vocabulary with
specific numbers, specific failure modes each pattern subtracts, and a
clean separation between retrain (subject to the patterns) and rollback
(not).

## Expected final artefact

A lab directory containing:

- `trigger_lab.py` — the one-file simulation: noisy 90-day signal,
  four rule variants, adversarial variant, audit-log output.
- `signal.png` — the drift-score time series with the threshold line
  overlaid.
- `retraining_decisions.parquet` — the audit log for all four rule
  variants on the 90-day timeline.
- `NOTES.md` — the three-pattern mapping table with starting
  parameters.

Running `python trigger_lab.py` prints the per-rule retrain counts and
the adversarial-signal before/after-cooldown comparison, and writes the
parquet audit log and `signal.png`.

## Final checkpoint

Without re-reading chapter 06, answer:

1. In step 2 the naïve rule retrained **9 times** across the 90 days
   even though only **one** real incident happened. Which of chapter
   06's five listed decision-rule pitfalls (1 "retraining on drift
   alone", 2 "no cooldown", 3 "stale baseline", 4 "no slice
   metrics", 5 "trigger fires; nobody knows why") did the naïve rule
   violate, and which of those three anti-noise patterns is
   responsible for each surplus retrain?
2. In step 4 the hysteresis rule used `enter_threshold=0.20,
   exit_threshold=0.15`. State in one sentence what would have
   happened if you had set `exit_threshold=0.21` (above the enter
   threshold) and what would have happened if you had set them
   equal. Which "pattern" in chapter 06 vocabulary are you breaking
   in each case?
3. In step 5 cooldown collapsed the adversarial two-plateau signal
   into **one** retrain. What specific chapter-06 named failure
   mode — the one chapter 06 describes as the model retraining on a
   drifted window, the new predictions being worse, the labels
   reflecting that, the next drift check firing again, the pipeline
   retraining again — does cooldown prevent?
4. Chapter 06's `decide_action` pseudocode checks `ROLLBACK` **before**
   `RETRAIN`, and this lab's cooldown logic does **not** apply to
   rollback. State in one sentence why "rollback wins over
   everything" is enforced in that specific order, and name the
   signal field (e.g. `rolling_accuracy_delta`, `multivariate_drift`,
   `consecutive_drift_windows`) chapter 06 keys the rollback branch
   on.
5. In step 6 you wrote a `retraining_decisions.parquet` with
   `(rule_version, day, action, trigger_reason)` rows. Which
   chapter-06 pitfall does *this specific file* prevent, and name one
   chapter-07 query from the debugging catalogue that would read from
   it.

If you can answer all five, chapter 06's `(signal, decision, action)`
anatomy and the three-pattern anti-noise machinery are internalised and
you are ready to apply them in exercise-03's `decide_action` pure
function, `policy.yaml`, and `simulate.py` 90-day replay.

## Where to go next

- **`exercise-03` ("Retraining trigger design")** turns the three
  rule variants here into one pure `decide_action` function driven
  by a `TriggerPolicy` pydantic model from `policy.yaml`, extended
  with the four actions (`DO_NOTHING`, `INVESTIGATE`, `RETRAIN`,
  `ROLLBACK`), slice-regression handling, and scheduled-floor logic
  — plus the per-path tests and the 90-day planted-incident
  simulation.
- **`exercise-04` ("Production regression post-mortem")** uses the
  "retrain that made things worse" pattern from chapter 07's
  gallery — a scenario this lab's naïve rule (step 2) would
  cheerfully produce. The exercise is the post-mortem write-up for
  that exact chain.
- **`lab-01` ("PSI zero-bin and threshold calibration")** is the
  upstream feeder: the `0.20` threshold and the `0.15` hysteresis
  exit threshold in this lab both come from a per-feature
  calibration procedure lab-01 walks through.
- **`lab-02` ("Prediction-distribution shift from a new-segment
  ramp")** is the orthogonal feeder: the drift score feeding this
  lab's time series is exactly the kind of signal lab-02 shows
  firing in Scenario A without any actual accuracy regression —
  which is why debounce + hysteresis exist and why the correct
  action is often `INVESTIGATE` rather than `RETRAIN`.
