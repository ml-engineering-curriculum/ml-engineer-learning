# Labs for mod-103-data-engineering-for-ml (Data Engineering for ML: Ingestion, Cleaning, Feature Engineering)

Short, guided hands-on sessions. Each lab is prescriptive by design —
numbered steps, expected output at each checkpoint — and complements
(rather than duplicates) the longer `exercises/` in this module.

| # | Lab | Companion chapters | Duration |
|---|---|---|---|
| 01 | [Parquet partition pruning and column projection](lab-01-parquet-partition-pruning.md) | 02, 04 | ~30–45 min |
| 02 | [Point-in-time features with `merge_asof`](lab-02-point-in-time-merge-asof.md) | 06, 07 | ~30–45 min |
| 03 | [Target encoding done right: hand-roll vs. `TargetEncoder`](lab-03-target-encoding-done-right.md) | 06, 07 | ~30–45 min |

## How to use the labs

- Read the companion chapters first. Labs assume you have the vocabulary
  from the chapter in front of you; they are drill, not introduction.
- Follow the steps in order; each step has a **checkpoint** you can
  verify before moving on.
- The exercises in `../exercises/` go deeper (3 hours each) and are
  where you build real muscle memory. Treat the labs as the shorter,
  guided version of the same ideas:
  - Lab 01 isolates the pushdown mechanics that exercise-01 relies on.
  - Lab 02 isolates the point-in-time feature construction that
    exercise-03 Part D plugs into a full pipeline.
  - Lab 03 isolates the target-encoding leak that exercise-03 Part C
    prevents by construction.
