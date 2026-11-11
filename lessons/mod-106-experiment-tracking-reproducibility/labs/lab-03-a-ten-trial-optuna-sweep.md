# lab-03 — A 10-trial Optuna sweep, end to end

> **Duration:** ~30–45 minutes.
> **Companion chapter:** [08 — Hyperparameter sweeps with Optuna](../08-sweeps-with-optuna.md).

## Objective

Make chapter 08's three primitives — **study**, **trial**, and the
**sampler** — concrete by running the smallest useful Optuna sweep on a
closed-form objective (no model to train) and reading the three
diagnostic plots the chapter asks you to produce. By the end you will be
able to point at:

- A persistent `optuna.db` SQLite study that survives a process restart
  and that `optuna.load_study` reopens.
- A deterministic sampler (`TPESampler(seed=0)`) whose trial sequence
  reproduces byte-for-byte across runs.
- The three `optuna.visualization` plots (optimisation history, param
  importances, slice) and what each is for.
- The reason why `MedianPruner` with `n_startup_trials=5` would not have
  pruned anything in a 10-trial run — and when pruning is worth turning
  on at all.

This lab is deliberately narrower than **exercise-04** (which runs the
full 30+ trial sweep against a real training script, wires every trial
into MLflow as a nested run, runs a seed sweep on the winner, registers
the champion in the Model Registry, and produces the chapter-09
`REPORT.md`). Here there is **no model**, **no MLflow**, and **no
registry** — just Optuna itself, on a function you can compute in your
head. The purpose is to get the sweep mechanics into muscle memory
before layering the rest on top of them.

## Prerequisites

- Python 3.11+, `optuna >= 3.5` (`uv add "optuna>=3.5" plotly` or
  `pip install "optuna>=3.5" plotly`). `plotly` is required by Optuna's
  built-in visualisations.
- A CPU is enough; nothing in this lab touches a GPU.
- Chapter 08 read through once, especially "Hello, Optuna", "The search
  space — practical rules", and "Analysing a completed sweep".
- `lab-01` and `lab-02` are helpful but not required.

## Setup

```bash
mkdir -p lab-03-optuna && cd lab-03-optuna
touch sweep.py analyse.py
```

Verify Optuna and plotly are both on the path:

```bash
python -c "import optuna, plotly; print(optuna.__version__, plotly.__version__)"
```

**Expected output.** Two version strings, no `ImportError`. Any
`optuna >= 3.5` is fine.

## Steps

### Step 1 — Define a closed-form objective (≈ 5 min)

Chapter 08's examples train real models inside the objective. For the
mechanics it is enough to optimise a function whose answer we already
know. We use a **mildly multi-modal 2-D function** so that TPE has
something to learn from across trials — a strictly convex quadratic
would be solved in 2 trials and would tell you nothing about the
sampler.

Put in `sweep.py`:

```python
# sweep.py
import math
import optuna
from optuna.samplers import TPESampler


def objective(trial: optuna.Trial) -> float:
    """
    Minimum-seeking 2-D test objective.

    The function is small outside a narrow 'valley' around
    (learning_rate=1e-2, hidden=64) and spikes elsewhere; log-uniform
    sampling on learning_rate is essential because the valley spans
    two orders of magnitude.
    """
    lr     = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
    hidden = trial.suggest_int("hidden", 8, 256, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Signal: a narrow log-scale valley around lr=1e-2, hidden=64.
    valley = (math.log10(lr) - math.log10(1e-2)) ** 2 \
           + (math.log2(hidden) - math.log2(64)) ** 2 * 0.3

    # Bonus: moderate dropout helps a little.
    reg = 0.2 * (dropout - 0.1) ** 2

    # Minimise. Lower is better.
    return valley + reg
```

Nothing to run yet — we hook this into a study next.

**Checkpoint.** Three things from chapter 08's "The search space —
practical rules":
(a) `learning_rate` is sampled on a **log scale**
(`log=True`) because the valley spans two orders of magnitude; without
`log=True`, Optuna's uniform sampling would spend 99% of its budget in
the top decade.
(b) `hidden` is also on a log scale and is an **int**, so it
discretises — exactly the shape chapter 08 recommends for widths that
span powers of two (8, 16, 32, ..., 256).
(c) `dropout` is a plain uniform float — small range, so no log scale.

### Step 2 — Create a seeded, persistent study and run 10 trials (≈ 10 min)

Append to `sweep.py`:

```python
if __name__ == "__main__":
    study = optuna.create_study(
        study_name="lab-03-toy",
        direction="minimize",
        sampler=TPESampler(seed=0),           # chapter 08: seed every sampler
        storage="sqlite:///optuna.db",        # durable — survives restarts
        load_if_exists=True,                  # resumable
    )
    study.optimize(objective, n_trials=10, show_progress_bar=False)

    print("\nnumber of finished trials:", len(study.trials))
    print("best value:", f"{study.best_value:.6f}")
    print("best trial number:", study.best_trial.number)
    print("best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k} = {v}")
```

Run it:

```bash
python sweep.py
```

**Expected output.** Ten lines of Optuna progress logs (one per
trial, each with the suggested params and the returned value), then
the summary block printing the best value (should be a small positive
number, around `0.01`–`0.3` depending on how close TPE got in 10
trials), and the three best-params close to `learning_rate ≈ 1e-2`,
`hidden ≈ 64`, `dropout ≈ 0.1`.

An `optuna.db` file now exists next to `sweep.py`.

**Checkpoint.** Chapter 08 ("Hello, Optuna") says a study needs five
things to be useful: a name, a direction, a sampler, a storage, and
the `load_if_exists` flag. You have all five. The sampler is
`TPESampler(seed=0)` — chapter 08 is explicit: "Always
`TPESampler(seed=...)`" (pitfall #2). Storage is SQLite — "`sqlite:///
optuna.db` at minimum" (pitfall #1). Without the sampler seed, the
study would pick different trials next time; without the storage, the
study would disappear when the process ends.

### Step 3 — Prove reproducibility: rerun the exact study on a fresh DB (≈ 5 min)

Chapter 08's five-condition list says a sweep is reproducible when the
sampler is seeded, storage is durable, the objective is deterministic,
the data is pinned, and the environment is pinned. We have a
deterministic objective (it's pure arithmetic) and a pinned
environment (your current `uv.lock`). Prove reproducibility by deleting
the DB and rerunning:

```bash
rm optuna.db
python sweep.py
```

**Expected output.** Same ten trials, in the same order, with the same
values. The "best trial number" should match what you saw in step 2
exactly; the printed "best params" should be identical to the
sixteenth decimal.

**Checkpoint.** Chapter 08 ("Reproducing a sweep"): "Under those five
conditions, running the same sweep script against a fresh (or resumed)
study produces the same trial outcomes in the same order". You just
observed the "fresh" variant. The "resumed" variant is step 4.

### Step 4 — Resume a study and watch the sampler use its history (≈ 5 min)

Run the sweep again without deleting the DB:

```bash
python sweep.py
```

**Expected output.** Optuna adds 10 **more** trials on top of the
existing 10 (total 20) because the `load_if_exists=True` flag opened
the existing study and `optimize(n_trials=10)` runs 10 *more* trials.
Watch the per-trial logs: the best value should drop further than it
did in step 3, because the TPE sampler is now choosing based on **20
prior observations instead of 0**, concentrating samples near the
valley.

To confirm from a fresh Python session:

```bash
python -c "
import optuna
s = optuna.load_study(study_name='lab-03-toy', storage='sqlite:///optuna.db')
print('trials in study:', len(s.trials))
print('best value:',       f'{s.best_value:.6f}')
print('best trial number:', s.best_trial.number)
"
```

**Expected output.** `trials in study: 20`, a best value lower than
step 3's, and the best trial number is somewhere in `[0, 19]`. The
`optuna.load_study` call is what chapter 08's acceptance test for
exercise-04 uses to prove the SQLite storage is actually durable.

**Checkpoint.** Chapter 08 ("Parallel and distributed sweeps"): "All
four workers point at the same `storage="sqlite:///optuna.db"`. They
coordinate via the storage: each trial is claimed atomically, and the
shared history informs each worker's sampler." You've observed the
single-worker version of this: `optimize` resumed the study because
`load_if_exists=True` and `storage=` were both set.

### Step 5 — Plot the three diagnostic figures chapter 08 names (≈ 10 min)

Put in `analyse.py`:

```python
# analyse.py
import optuna
import optuna.visualization as vis

study = optuna.load_study(study_name="lab-03-toy",
                          storage="sqlite:///optuna.db")

figs = {
    "optimization_history":  vis.plot_optimization_history(study),
    "param_importances":     vis.plot_param_importances(study),
    "slice_learning_rate":   vis.plot_slice(study, params=["learning_rate"]),
}

for name, fig in figs.items():
    fig.write_html(f"plots/{name}.html")
    print(f"wrote plots/{name}.html")
```

Make the output directory and run:

```bash
mkdir -p plots
python analyse.py
```

**Expected output.** Three lines of `wrote plots/<name>.html`. The
`plots/` directory now contains three standalone HTML files with
interactive Plotly figures (open them in a browser —
`open plots/optimization_history.html` on macOS, `xdg-open ...` on
Linux).

What you should see in each plot:

- **`optimization_history.html`** — x-axis is trial number (0–19), one
  dot per trial (the objective value) and a line connecting the **best
  value seen so far**. The best-so-far line should drop sharply in the
  first few trials and flatten by the end — chapter 08 reads a flat
  tail as "you converged" and a still-rising tail as "you undersized
  `n_trials`". On this 20-trial toy, expect a visible but partial
  flattening.
- **`param_importances.html`** — a horizontal bar chart. On this
  objective **`learning_rate` should dominate**, because log-uniform
  sampling across two orders of magnitude gives the sampler the
  biggest variance to attribute. `hidden` should come second; `dropout`
  a distant third (its contribution to the objective is tiny by
  construction). Chapter 08 says "Optuna computes this via a small
  random forest fit on trial history. Not gospel — but useful for the
  next sweep's search space."
- **`slice_learning_rate.html`** — a scatter of trials in
  (learning_rate, objective) space. The **valley at `learning_rate ≈
  1e-2`** should be visible as a cluster of low objective values;
  outside it, the objective climbs. Chapter 08 calls out that slice
  plots "reveal plateaus and multi-modal responses".

**Checkpoint.** Chapter 08's "Analysing a completed sweep" section
names four plot helpers; this lab uses three of them (parallel
coordinates is the fourth — try adding it as a stretch). These plots
are the main thing exercise-04 asks you to log as MLflow artefacts on
the parent run. Reading them well is a skill: the optimisation history
tells you *how far* to extend the search, param importances tell you
*which axes* to narrow, slice plots tell you *where in each axis* the
optimum lives. If any of your three plots is empty or looks wrong,
re-run `python sweep.py` a couple more times to pad out the trial
count.

### Step 6 — Would `MedianPruner` have helped? (≈ 5 min)

Chapter 08 ("Pruning — stopping bad trials early") says pruning is
"the single biggest cost saver" for deep-learning sweeps where each
trial has many epochs, and "conservatively (or not at all) when a trial
takes seconds".

This lab's objective is **a single evaluation per trial** — there are
no intermediate steps, no `trial.report(val, step=epoch)` calls, so
`MedianPruner` has **nothing to prune on**. Prove it. Append a quick
diagnostic to `analyse.py`:

```python
print("\npruned trials:",
      sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED))
print("complete trials:",
      sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE))
```

Rerun:

```bash
python analyse.py
```

**Expected output.** `pruned trials: 0`, `complete trials: 20`.

Without a per-epoch metric, a pruner is inert. In exercise-04's real
training sweep the objective calls `trial.report(val_metric,
step=epoch)` and `if trial.should_prune(): raise optuna.TrialPruned()`
each epoch — only then can `MedianPruner` compare an in-progress
trial against prior trials at the same step and kill it early.

**Checkpoint.** Chapter 08 ("Pruning — stopping bad trials early"):
`n_startup_trials=5` means "don't prune the first 5 trials — we don't
have enough baseline to judge against" and `n_warmup_steps=3` means
"don't prune within the first 3 epochs of any trial — performance is
too noisy that early". Chapter 08 pitfall #3: "Pruning without
`n_startup_trials`. The first trials get pruned before the pruner has
any signal — you optimise on noise." On this lab's one-shot objective
neither parameter fires; on exercise-04's deep-learning sweep they are
the two numbers you tune first.

### Step 7 — Record the outcome (≈ 5 min)

Create `NOTES.md`:

```markdown
# Lab 03 — 10 (then 20) trial Optuna sweep

## Study

- `study_name`: `lab-03-toy`
- `storage`: `sqlite:///optuna.db`
- `sampler`: `TPESampler(seed=0)`
- `direction`: `minimize`
- `n_trials` per run: 10 (ran twice → 20 total trials on disk)

## Winner (after 20 trials)

- Best value: `{{copy from step 4}}`
- Best params:
  - `learning_rate = {{...}}`
  - `hidden = {{...}}`
  - `dropout = {{...}}`
- Best trial number: `{{...}}`

## Reproducibility check

- Deleted `optuna.db`, re-ran `python sweep.py`, the first 10 trials
  landed in the same order with the same values: `yes / no`
- `optuna.load_study` in a fresh Python session re-opens the study:
  `yes / no`

## What each diagnostic plot told me

- **optimization_history** — {{describe the shape of the best-so-far
  line; does it look converged by trial 20?}}
- **param_importances** — ranking:
  1. {{...}}
  2. {{...}}
  3. {{...}}
- **slice (learning_rate)** — {{where is the valley; roughly which
  log-decade}}

## Pruning (would have been inert because...)

- Pruned trials: 0 / 20. Reason: the objective reports a single value
  per trial, with no `trial.report(..., step=...)` calls, so
  `MedianPruner` has no intermediate signal to compare against prior
  trials.
```

**Checkpoint.** This is a miniature version of chapter 09's
reproducibility report — the training-procedure section (§ 5) uses
exactly these fields (sampler, pruner, search space, winner, best
trial number) plus the seed-sweep addition that exercise-04 layers on
top. Writing them once, here, on a toy objective means exercise-04's
report has one fewer thing to invent.

## Expected final artefact

```
lab-03-optuna/
├── sweep.py                      # steps 1–4 (objective, study, 10 trials, resume)
├── analyse.py                    # steps 5–6 (three plots + pruning diagnostic)
├── optuna.db                     # persistent SQLite study, 20 trials by the end
├── plots/
│   ├── optimization_history.html
│   ├── param_importances.html
│   └── slice_learning_rate.html
└── NOTES.md                      # step 7 result table
```

Running `python sweep.py && python analyse.py` from a fresh directory
(after `rm optuna.db`) should take under 10 seconds and produce the
same best trial number as the NOTES.md you wrote.

## Final checkpoint

Without re-reading chapter 08, answer:

1. In step 2 the study was constructed with
   `sampler=TPESampler(seed=0)`. Chapter 08 names the pitfall that
   happens when you omit the `seed=` argument. State the pitfall in one
   sentence and explain why "I ran the same sweep twice and got
   different winning hyperparameters" is diagnostic of it.
2. In step 4 `python sweep.py` added 10 new trials on top of the 10
   already in `optuna.db`, giving 20 total. Which two Optuna arguments
   make this behaviour possible, and what would have happened instead
   if `storage=` had been `None` (the default)?
3. In step 5, `plot_param_importances` ranked `learning_rate` highest
   by a large margin. Chapter 08 says the importance plot is "not
   gospel — but useful for the next sweep's search space." What
   concrete change to the search space would you make in the follow-up
   sweep given this ranking, and why?
4. In step 6 you confirmed no trials were pruned. Chapter 08 says
   pruning is "the single biggest cost saver" for deep-learning
   sweeps. What single line do you have to add inside the `objective`
   for `MedianPruner` to start working, and what do the two arguments
   `MedianPruner(n_startup_trials=5, n_warmup_steps=3)` protect you
   against (chapter 08 pitfall #3)?

If you can answer all four, chapter 08's sweep mechanics are now
compile-time knowledge, and exercise-04's full 30+ trial sweep against
a real training script is a matter of plugging in a model and an
MLflow run rather than learning Optuna from scratch.

## Where to go next

- **`exercise-04` ("Hyperparameter sweep with Optuna")** reuses the
  study / sampler / storage pattern you just built, wraps the sweep in
  one parent MLflow run with each trial as a nested child, runs a
  **seed sweep** on the winner (chapter 04's `seed_everything` × 5
  seeds), and registers the champion in the Model Registry. The
  deliverable is the chapter-09 `REPORT.md`.
- **`lab-01` ("Dissecting `mlruns/`")** made chapter 02's object model
  concrete; useful context before you wire MLflow logging into the
  Optuna objective.
- **`lab-02` ("Finding the non-determinism")** built chapter 04's
  reproducibility harness by hand; the same harness is what makes an
  Optuna objective deterministic (chapter 08's five-condition list).
