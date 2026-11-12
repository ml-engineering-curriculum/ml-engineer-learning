# Labs for mod-107-model-evaluation-validation (Model Evaluation & Validation)

Labs are short, guided, hands-on sessions. Each one isolates one
primitive from the module — a metric divergence, a leakage mechanism, a
calibration diagnostic — so the mechanics are visible and testable before
they appear inside the larger, open-ended exercises.

| # | Lab | Duration | Companion chapters |
|---|---|---|---|
| 01 | [Accuracy, ROC-AUC, and PR-AUC disagree under class imbalance](lab-01-accuracy-roc-pr-disagree.md) | 30–45 min | 03, 09 |
| 02 | [Preprocessing leakage: scaler before vs inside the `Pipeline`](lab-02-preprocessing-leakage.md) | 30–45 min | 06 |
| 03 | [Reliability diagram and recalibrating a biased classifier](lab-03-reliability-diagram-and-recalibration.md) | 30–45 min | 04 |

Labs are deliberately distinct from the module's exercises (which build
full metric rubrics, nested-CV pipelines, baseline-and-error-analysis
reports, and three-dimensional slice tables on real datasets). A lab
makes one primitive concrete on synthetic or small-sample data; an
exercise wires several of them together on a realistic workload.

Suggested order: do each lab shortly **after** reading its companion
chapter, and **before** starting the exercise that uses that piece. The
three labs together form the compact tour of the module's metric /
leakage / calibration vocabulary.
