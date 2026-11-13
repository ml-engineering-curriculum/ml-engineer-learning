# Labs for mod-108-model-deployment-serving (Model Packaging & Deployment)

Labs are short, guided, hands-on sessions. Each one isolates one primitive
from the module — a lifecycle signal, a concurrency pitfall, a load-test
executor choice — so the mechanics are visible and testable before they
appear inside the larger, open-ended exercises.

| # | Lab | Duration | Companion chapters |
|---|---|---|---|
| 01 | [Liveness vs readiness: watch a slow-starting service survive (or crash-loop)](lab-01-liveness-vs-readiness.md) | 30–40 min | 04 |
| 02 | [`async def` with a blocking predict call: watch the event loop deadlock](lab-02-async-blocking-event-loop.md) | 30–40 min | 03, 09 |
| 03 | [Reading the latency-throughput curve: arrival-rate vs VUs hides saturation](lab-03-arrival-rate-vs-vus.md) | 35–45 min | 09, 10 |

Labs are deliberately distinct from the module's exercises (which build a
full packaged + served model, a multi-stage Docker image with a CI
pipeline, a batch + hybrid inference path, and a full load-test rig with a
defended capacity report). A lab makes one primitive concrete on a tiny
FastAPI app with no real model behind it; an exercise wires several of them
together on a real workload.

Suggested order: do each lab shortly **after** reading its companion
chapter, and **before** starting the exercise that uses that piece — lab-01
feeds exercise-01 (the `/readyz` endpoint), lab-02 feeds exercise-01 (the
handler shape), and lab-03 feeds exercise-04 (the load-test rig). The
three labs together form the compact tour of the module's lifecycle /
concurrency / capacity vocabulary.
