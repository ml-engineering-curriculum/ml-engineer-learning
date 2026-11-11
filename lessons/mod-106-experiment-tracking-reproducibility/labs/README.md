# Labs for mod-106-experiment-tracking-reproducibility (Experiment Tracking & Reproducibility)

Labs are short, guided, hands-on sessions. Each one isolates one piece
of the module — MLflow's on-disk layout, the RNG layers a training run
touches, or Optuna's study/sampler mechanics — so that the primitive is
visible before it appears inside the full exercises.

| # | Lab | Duration | Companion chapter |
|---|---|---|---|
| 01 | [Dissecting `mlruns/`: what a run actually stores on disk](lab-01-dissecting-mlruns-on-disk.md) | 30–45 min | 02 |
| 02 | [Finding the non-determinism: one seed at a time](lab-02-finding-the-nondeterminism.md) | 30–45 min | 04 |
| 03 | [A 10-trial Optuna sweep, end to end](lab-03-a-ten-trial-optuna-sweep.md) | 30–45 min | 08 |

Labs are deliberately distinct from the module's exercises (which
instrument real training scripts, version data and models, write a
determinism harness and tests, and run full hyperparameter sweeps
against a real workload with the chapter-09 `REPORT.md` as the
deliverable). A lab makes one primitive concrete; an exercise wires
several of them together on a realistic workload.

Suggested order: do each lab shortly **after** reading its companion
chapter, and **before** starting the exercise that uses that piece.
Lab 01 pairs with exercise-01 (MLflow instrumentation); lab 02 pairs
with exercise-03 (seeded reruns and determinism); lab 03 pairs with
exercise-04 (Optuna sweep + report).
