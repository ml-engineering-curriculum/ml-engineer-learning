# Labs for mod-110-ml-systems-design (ML Systems Design)

Labs are short, guided, hands-on sessions. Each one isolates one
primitive from the module — a feedback loop closing on a toy
recommender, the sample-size formula and the peeking trap behind an
A/B read-out — so the mechanics are visible and testable before they
appear inside the larger, open-ended exercises.

| # | Lab | Duration | Companion chapters |
|---|---|---|---|
| 01 | [The recommender loop collapses without exploration: catalogue coverage, entropy, and the mitigation](lab-01-recommender-loop-collapse-and-exploration.md) | 35–45 min | 04 |
| 02 | [A/B power analysis, the "run for 2 days" trap, and the false-positive cost of peeking](lab-02-power-analysis-and-the-peeking-trap.md) | 35–45 min | 05 |

Labs are deliberately distinct from the module's exercises (which
produce a full blueprint plus reference-architecture diagram plus
chapter-07-checklist walk, a skew investigation with a diagnosis
packet and a row-parity harness, an A/B design packet with a
pre-registered design doc and read-out template, and a build-vs-buy
packet with a nine-component decision table and peer-track delegation
memos). A lab makes one primitive concrete on synthetic data in a
single file; an exercise wires several of them together on a
realistic workload.

Suggested order: do each lab shortly **after** reading its companion
chapter, and **before** starting the exercise that uses that piece —
lab-01 feeds exercise-01's one-paragraph loop analysis and chapter
07 Section D's audit, and lab-02 feeds exercise-03's `power.py`
script and Part C's SRM risk entry. The two labs together form the
compact tour of the module's loop-mechanics and
power-analysis-plus-peeking vocabulary — the two places chapter
05 and chapter 04 each expect a number, not a vibe.
