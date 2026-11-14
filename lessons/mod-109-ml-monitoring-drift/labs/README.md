# Labs for mod-109-ml-monitoring-drift (ML Monitoring & Drift Detection)

Labs are short, guided, hands-on sessions. Each one isolates one
primitive from the module — a drift distance's failure mode, a
label-free signal that is a symptom rather than a cause, a decision-rule
anti-noise pattern — so the mechanics are visible and testable before
they appear inside the larger, open-ended exercises.

| # | Lab | Duration | Companion chapters |
|---|---|---|---|
| 01 | [PSI on synthetic drift: the zero-bin bomb and a backfill-calibrated threshold](lab-01-psi-zero-bin-and-threshold-calibration.md) | 35–45 min | 03 |
| 02 | [Prediction-distribution shift from a new-segment ramp: why "retrain on shift" is a trap](lab-02-prediction-shift-is-not-concept-drift.md) | 35–45 min | 04, 06 |
| 03 | [Debounce, hysteresis, and cooldown on a noisy drift time series](lab-03-debounce-hysteresis-cooldown.md) | 35–45 min | 06 |

Labs are deliberately distinct from the module's exercises (which build
a full ML-metrics surface with a reference profile and a cardinality
audit, a scheduled drift job with calibrated thresholds and
synthetic-incident tests, an auditable retraining trigger with planted
incidents, and a full production-regression post-mortem with runbook).
A lab makes one primitive concrete on synthetic data in a single file;
an exercise wires several of them together on a realistic workload.

Suggested order: do each lab shortly **after** reading its companion
chapter, and **before** starting the exercise that uses that piece —
lab-01 feeds exercise-02 (the `drift/distances.py` and
`drift/calibrate.py` modules), lab-02 feeds exercise-03 (the
`decide_action` rule's "drift alone is not a retrain" branch), and
lab-03 feeds exercise-03 (the three anti-noise patterns inside
`decide_action` and the `retraining_decisions.parquet` audit log). The
three labs together form the compact tour of the module's
distance / signal-correlation / anti-noise vocabulary.
