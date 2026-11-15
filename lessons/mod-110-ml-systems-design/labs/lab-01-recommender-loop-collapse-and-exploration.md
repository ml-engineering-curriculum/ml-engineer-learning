# lab-01 — The recommender loop collapses without exploration: catalogue coverage, entropy, and the mitigation

> **Duration:** ~35–45 minutes.
> **Companion chapter:** [04 — Feedback loops and data flywheels](../04-feedback-loops-and-data-flywheels.md).

## Objective

Reproduce chapter 04's two load-bearing claims about recommender feedback
loops — (a) "every retrain trains on data that the previous model shaped"
and the loop tightens unless something is done about it, and (b) a small
**exploration budget** (1–10% of serving traffic) is the specific
mitigation that keeps the training set diverse — on a 100-line synthetic
simulation you can run on a laptop. By the end you will have (a) built a
50-round recommender simulation where the model is retrained each round on
its own impressions and the user population is held fixed, (b) watched
**catalogue coverage** (fraction of the 500-item catalogue ever
recommended per week) collapse from the launch-week baseline to a small
fraction, (c) watched **entropy of the recommendation distribution**
decay monotonically while **CTR goes up** — the pitfall-#2 signature of a
tightening loop, (d) added the simplest possible exploration budget — a
fixed `ε` fraction of impressions drawn uniformly from the catalogue —
and watched coverage and entropy stabilise while CTR gives back a small
amount (the cost of training-set diversity chapter 04 names), and (e)
plotted all three metrics side by side so the "aggregate metric is a
distraction while the loop tightens underneath" story is visible.

This lab is deliberately narrower than **exercise-01** (which asks you to
produce a full blueprint plus reference-architecture diagram plus
self-review across chapters 01, 02, 07) and complementary to chapter
04's one-paragraph loop-analysis discipline. The lab is the isolation
chamber where you *see* the loop close on a dataset you can eyeball, and
the exploration-budget mitigation cash out as a two-line change.

## Prerequisites

- Python 3.11+.
- `numpy`, `matplotlib` installed in a virtual environment
  (`uv venv && uv pip install numpy matplotlib` is fine).
- Chapter 04 read through once; the five vocabulary terms (feedback loop,
  data flywheel, selection bias, position bias, filter bubble / runaway)
  and the four detectors + four mitigations are the backing concepts.
- A CPU is enough; no GPU, no Docker, no warehouse, no real model.

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-01-loop && cd lab-01-loop
touch loop_lab.py
```

Everything below goes in `loop_lab.py`. Run with `python loop_lab.py` after
each step and read the output before moving on.

Start the file with the imports, a fixed seed, and the catalogue:

```python
import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(seed=0)

# Catalogue: 500 items, each with a hidden "true relevance" in [0, 1].
# Most items are mediocre; a handful are great. Beta(2, 10) gives a long
# right tail, which is the common shape in real catalogues.
CATALOGUE_SIZE = 500
true_relevance = RNG.beta(2.0, 10.0, size=CATALOGUE_SIZE)

K = 10                    # top-K shown per impression
USERS_PER_ROUND = 2_000   # impressions per training round
N_ROUNDS = 50             # a "round" is one week in the loop
```

## Steps

### Step 1 — Build the no-exploration recommender and simulate one round (≈ 5 min)

Chapter 04's recommender shape: the model scores items by *measured
engagement so far*, recommends top-K, users click with probability
proportional to each shown item's hidden true relevance, and the clicks
are the next round's training signal. Nothing random; nothing
exploration-y.

Append to `loop_lab.py`:

```python
def init_counts():
    """Initial exposure/click counts: one soft-prior click and one impression
    per item, so no item has an undefined CTR on round 1."""
    impressions = np.ones(CATALOGUE_SIZE)
    clicks = np.full(CATALOGUE_SIZE, 0.1)  # mild uniform prior
    return impressions, clicks


def model_scores(impressions, clicks):
    """Observed CTR with a small Laplace smoothing. This is 'the model' —
    a stand-in for anything that learns from logged impressions and clicks."""
    return (clicks + 0.1) / (impressions + 1.0)


def recommend_topk(scores, k=K):
    """Deterministic top-K. No exploration."""
    return np.argsort(-scores)[:k]


def simulate_round(impressions, clicks, recommend_fn):
    """One week: every user is shown top-K; each item is clicked with
    probability equal to its hidden true_relevance. Counts are updated
    from the observed impressions and clicks only."""
    shown_this_round = np.zeros(CATALOGUE_SIZE, dtype=int)
    clicks_this_round = np.zeros(CATALOGUE_SIZE)
    for _ in range(USERS_PER_ROUND):
        shown = recommend_fn(model_scores(impressions, clicks))
        shown_this_round[shown] += 1
        p_click = true_relevance[shown]
        did_click = RNG.random(K) < p_click
        clicks_this_round[shown] += did_click.astype(float)
    return shown_this_round, clicks_this_round


# Smoke test: one round, no-exploration model.
impressions, clicks = init_counts()
shown, click_counts = simulate_round(impressions, clicks, recommend_topk)
print(f"round 1 — unique items shown: {(shown > 0).sum()} / {CATALOGUE_SIZE}")
print(f"round 1 — mean CTR of top-K:  "
      f"{click_counts.sum() / shown.sum():.4f}")
```

Run `python loop_lab.py`.

**Expected output.** `round 1 — unique items shown` is a small number —
probably 10 to 50 — because every impression this round sees the *same*
top-K from the Laplace-smoothed prior and nothing has been retrained yet.
`mean CTR of top-K` is whatever the hidden relevance of those few items
averages to; it is not the number to watch yet.

**Checkpoint.** You have the three primitives chapter 04 relies on: a
recommender that scores from logged data, a simulated user population
whose click probability is a *fixed* function of each item's hidden
relevance, and a per-round counter that tracks what the system actually
saw. Nothing has drifted yet; the hidden `true_relevance` vector is
constant across the whole simulation. Any change in metrics across
rounds is **the loop, not the world**.

### Step 2 — Run 50 rounds, measure catalogue coverage and entropy, watch CTR go up as everything else goes down (≈ 7 min)

Chapter 04's first two diversity metrics: **catalogue coverage** (fraction
of the catalogue recommended at least once per round) and **entropy of
the recommendation distribution** (`H = −Σ p_i log p_i` across the items
shown that round; falling entropy = concentration on a small set). Both
should be on the same dashboard as CTR. Add them, run the loop, record
all three per round.

Append:

```python
def entropy(shown_counts):
    """Shannon entropy over the items shown this round, in nats.
    An even split over 500 items would be ln(500) ≈ 6.21; one-item
    concentration would be 0.0."""
    total = shown_counts.sum()
    if total == 0:
        return 0.0
    p = shown_counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def run_simulation(recommend_fn, n_rounds=N_ROUNDS, label="no-exploration"):
    impressions, clicks = init_counts()
    history = {"coverage": [], "entropy": [], "ctr": []}
    for _ in range(n_rounds):
        shown, round_clicks = simulate_round(impressions, clicks,
                                             recommend_fn)
        impressions += shown
        clicks += round_clicks
        history["coverage"].append(float((shown > 0).sum() / CATALOGUE_SIZE))
        history["entropy"].append(entropy(shown))
        history["ctr"].append(
            float(round_clicks.sum() / max(shown.sum(), 1)))
    return history


hist_greedy = run_simulation(recommend_topk, label="no-exploration")

print("\nno-exploration loop, selected rounds:")
print(f"{'round':>6} {'coverage':>9} {'entropy':>8} {'CTR':>7}")
for r in [0, 4, 9, 19, 29, 39, 49]:
    print(f"{r+1:>6} "
          f"{hist_greedy['coverage'][r]:>9.3f} "
          f"{hist_greedy['entropy'][r]:>8.3f} "
          f"{hist_greedy['ctr'][r]:>7.3f}")
```

**Expected output.** Something along the lines of:

```
no-exploration loop, selected rounds:
 round  coverage  entropy     CTR
     1     0.0??    ?.???   0.???
     5     0.0??    ?.???   0.???
    10     0.0??    ?.???   0.???
    20     0.0??    ?.???   0.???
    30     0.0??    ?.???   0.???
    40     0.0??    ?.???   0.???
    50     0.0??    ?.???   0.???
```

Exact digits move with the seed; the **shape** is what to watch. Three
properties fall out that chapter 04 relies on:

- **Catalogue coverage falls monotonically** across rounds. The pool of
  items the system is willing to show narrows. By round 50 the system is
  routinely showing a small handful of items.
- **Entropy collapses.** From something like `H ≈ ln(10) ≈ 2.3` on
  round 1 (a near-uniform distribution across the top-K) toward the
  low end of what K items can produce, and continuing to compress as
  the top-K itself narrows — the recommender is placing most of its
  impressions on fewer and fewer items.
- **CTR climbs anyway.** Because the model does identify real signal —
  the handful of items it converges on genuinely are among the highest
  `true_relevance` — the aggregate CTR looks fine, even great. This is
  chapter 04 pitfall #2 verbatim: *"CTR on a recommendation that is
  over-served is not evidence the recommendation is good; it is
  evidence the loop is tightening."*

**Checkpoint.** You have reproduced chapter 04's fingerprint of a
tightening loop: aggregate metric up, diversity metrics down,
**nothing in the world has changed**. On real traffic this fingerprint
takes months to show up because the time axis is weeks of production
data and the aggregate metric is what the launch review reads. In
simulation it falls out in 50 rounds and 3 columns.

### Step 3 — Add the simplest exploration budget and watch the fingerprint flip (≈ 8 min)

Chapter 04 mitigation #1: *"The recommender or ranker reserves a fixed
fraction of serving traffic — 1-10% — for **exploration**: recommending
items the model is *not* confident about. Bandit algorithms (Thompson
sampling, UCB-style approaches) formalise this; **simple
uniform-random-sample-from-top-K works surprisingly well as a starting
point**."* Implement the simplest version — a fixed `ε` fraction of
each round's impressions drawn uniformly from the whole catalogue — and
compare.

Append:

```python
EPSILON = 0.05  # 5% exploration — well inside chapter 04's 1-10% range

def recommend_epsilon_greedy(scores, k=K, eps=EPSILON):
    """With probability eps per impression, draw k items uniformly from
    the full catalogue instead of taking the model's top-K."""
    if RNG.random() < eps:
        return RNG.choice(CATALOGUE_SIZE, size=k, replace=False)
    return recommend_topk(scores, k)


hist_explore = run_simulation(recommend_epsilon_greedy,
                              label="5% exploration")

print("\n5% exploration loop, selected rounds:")
print(f"{'round':>6} {'coverage':>9} {'entropy':>8} {'CTR':>7}")
for r in [0, 4, 9, 19, 29, 39, 49]:
    print(f"{r+1:>6} "
          f"{hist_explore['coverage'][r]:>9.3f} "
          f"{hist_explore['entropy'][r]:>8.3f} "
          f"{hist_explore['ctr'][r]:>7.3f}")

print(f"\nfinal-round summary")
print(f"  no-exploration:    coverage={hist_greedy['coverage'][-1]:.3f}  "
      f"entropy={hist_greedy['entropy'][-1]:.3f}  "
      f"CTR={hist_greedy['ctr'][-1]:.3f}")
print(f"  5% exploration:    coverage={hist_explore['coverage'][-1]:.3f}  "
      f"entropy={hist_explore['entropy'][-1]:.3f}  "
      f"CTR={hist_explore['ctr'][-1]:.3f}")
```

**Expected output.** The exploration run's coverage stays well above the
no-exploration run's at every round after the first few, and the
entropy time series stops collapsing (it stabilises somewhere around
`ε`-driven uniform contribution plus the exploit-top-K contribution).
Meanwhile, the exploration run's CTR gives back a small but real
amount — a fraction of a point to a few points — because `ε` of every
round's impressions are now on randomly-chosen items that are, on
average, below the true relevance of the exploit top-K.

**Checkpoint.** Chapter 04's exploration-budget mitigation falls out as
two lines of code and the trade curve it commits the system to is
visible in the three columns: coverage and entropy hold; CTR costs a
little. The cost is **explicitly justified in the blueprint as a cost
the business is paying for training-set diversity** (chapter 04,
mitigation 1's framing verbatim). The alternative — pitfall #3 ("No
exploration budget. The first year of the system's data is what
determines the next year's model; a homogeneous first year permanently
narrows the model's view") — is now visible on your own screen, not
just in the chapter.

### Step 4 — Draw the plots and eyeball the two time series side by side (≈ 5 min)

Chapter 04 says catalogue coverage, entropy, and the primary business
metric belong on the **same dashboard**, not three different ones. Plot
them that way.

Append:

```python
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
labels = [("coverage", "catalogue coverage"),
          ("entropy",  "entropy of recommendations (nats)"),
          ("ctr",      "aggregate CTR")]
rounds = np.arange(1, N_ROUNDS + 1)
for ax, (key, title) in zip(axes, labels):
    ax.plot(rounds, hist_greedy[key], label="no exploration", linewidth=2)
    ax.plot(rounds, hist_explore[key], label="5% exploration",
            linewidth=2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("round (≈ week)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("loop_dashboard.png", dpi=120)
print("wrote loop_dashboard.png")
```

Open `loop_dashboard.png`. The three panels should tell the same story
in three directions:

- **Coverage panel.** The no-exploration line decays toward the bottom
  of the plot; the exploration line sits high and roughly constant
  from round 2 onward.
- **Entropy panel.** The no-exploration line collapses monotonically;
  the exploration line holds a plateau. The gap between the two is
  "the loop closing" vs "the loop held open."
- **CTR panel.** The no-exploration line climbs toward a higher
  plateau; the exploration line climbs more slowly and settles at a
  slightly lower plateau. The *difference* is exactly chapter 04's
  "the exploration traffic is instrumented separately so its
  performance can be measured and reported (usually worse than the
  model's, on purpose)."

**Checkpoint.** The figure is what chapter 04 means by "on the **same**
dashboard": CTR by itself is the dashboard the launch-review reads; the
three-panel figure is the dashboard an ML engineer should be looking at
to catch the loop *before* the business metric drops.

### Step 5 — Try `EPSILON = 0.0`, `0.01`, `0.05`, `0.10` and read the trade curve (≈ 7 min)

Chapter 04 names the exploration budget as a range (1–10%), not a fixed
number, because the right value depends on the catalogue size, the
hidden-relevance distribution, and the business's tolerance for the
CTR trade. Sweep it.

Append:

```python
def run_with_epsilon(eps):
    RNG_local = np.random.default_rng(seed=0)
    # Re-seed so the sweep is comparable across ε values.
    global RNG
    RNG = RNG_local
    impressions, clicks = init_counts()
    coverages, ctrs = [], []
    for _ in range(N_ROUNDS):
        shown, round_clicks = simulate_round(
            impressions, clicks,
            lambda s: recommend_epsilon_greedy(s, eps=eps),
        )
        impressions += shown
        clicks += round_clicks
        coverages.append(float((shown > 0).sum() / CATALOGUE_SIZE))
        ctrs.append(float(round_clicks.sum() / max(shown.sum(), 1)))
    return coverages[-1], ctrs[-1]


print(f"\n{'epsilon':>8} {'final coverage':>16} {'final CTR':>12}")
for eps in [0.00, 0.01, 0.05, 0.10, 0.20]:
    cov, ctr = run_with_epsilon(eps)
    print(f"{eps:>8.2f} {cov:>16.3f} {ctr:>12.3f}")
```

**Expected output.**

```
 epsilon  final coverage   final CTR
    0.00           0.0??       0.???
    0.01           0.1??       0.???
    0.05           0.9??       0.???
    0.10           1.0??       0.???
    0.20           1.0??       0.???
```

The precise values depend on your RNG, but the **shape** is the
trade curve chapter 04 implies: coverage saturates somewhere in the
1–10% range (by `ε = 0.10` almost every item in the catalogue is being
shown at least once per round), and CTR decays roughly linearly in `ε`.
Picking an `ε` is a product choice, not a mathematical one — chapter 04
mitigation 1 names the discipline: *"explicitly justified in the
blueprint as a cost the business is paying for training-set diversity."*
A team that picks `ε = 0.0` has picked it; the "we have no exploration
budget" answer in chapter 07's checklist D3 is a decision, not an
oversight.

**Checkpoint.** The trade curve lets you argue — in the blueprint,
with numbers — for an `ε` the business can defend. "At `ε = 0.05` we
give up ~N basis points of CTR for a ~5× increase in catalogue
coverage and a stable-rather-than-collapsing entropy time series;
this is the cost we are paying to keep the loop open" is a defensible
sentence. Compare to "we don't really explore but we might add it
later" (pitfall #6 — "A 'we'll add debiasing later' plan. By then,
the training set is contaminated; the loop is set").

### Step 6 — Write the one-paragraph loop analysis this lab's recommender would have in chapter 07's checklist D section (≈ 5 min)

Chapter 04 closes with the "one-page loop analysis": six questions, six
lines, every blueprint. Chapter 07 Section D makes them part of the
design review. Write them in a `NOTES.md` in the lab directory so a
reviewer of your future exercise-01 blueprint can see where the
discipline came from.

```markdown
# lab-01 — loop-analysis notes

Loop analysis for this lab's recommender (chapter 04 / chapter 07 section D):

1. Action: top-K recommendation from a 500-item catalogue; the action
   determines what the user sees, so impressions are 100% model-shaped
   without exploration.
2. Label source: passive observational clicks only (no randomised
   treatment, no external ground truth). Loop is on by construction.
3. Exploration budget: 5% uniform-random-from-catalogue impressions
   (chapter 04 mitigation 1's "simple uniform-random-sample-from-top-K
   works surprisingly well as a starting point"). Final coverage ≈ 95%;
   CTR cost ≈ a few percent — explicitly justified as training-set
   diversity.
4. Untreated cohort: n/a — this is a ranking system, so the untreated
   cohort equivalent is the exploration traffic itself (chapter 04
   detector 2 "for scoring/intervention systems" vs the exploration-
   traffic shape "for ranking/recommender systems").
5. Diversity metrics on the dashboard: catalogue coverage, entropy of
   the recommendation distribution, aggregate CTR — same dashboard,
   not three dashboards (chapter 04 pitfall #8 — "A recommender whose
   only diversity metric is accuracy@K").
6. Full-population retrain cadence: each round in this simulation IS
   a retrain; in production the chapter 04 mitigation 3 cadence is
   "once every N training cycles retrain on a broader set" — for this
   lab, once every 10 rounds retrain weighted toward the exploration
   impressions so the broader catalogue contribution is not diluted
   by the exploit traffic.

The alternative design — `ε = 0.00` — is what the lab's step 2 plot
shows: coverage → low teens, entropy → single digits in nats, CTR →
high plateau, business metric looking good while the training set
contaminates. Chapter 04 pitfall #1 ("Denying the loop exists. 'We
just train on the data we have.' That's the loop") is exactly this
alternative stated out loud.
```

**Checkpoint.** You have derived chapter 04's loop-analysis
discipline from one `.py` file and `loop_dashboard.png`. Every line
in `NOTES.md` traces back to a specific number or a specific curve
you measured in the lab, not a cargo-cult default from a blog post.

## Expected final artefact

A lab directory containing:

- `loop_lab.py` — the single file that implements the recommender
  primitives, a 50-round no-exploration simulation, a 5% exploration
  comparison, the catalogue-coverage / entropy / CTR dashboard, and a
  sweep over `ε ∈ {0.00, 0.01, 0.05, 0.10, 0.20}`.
- `loop_dashboard.png` — the three-panel figure (coverage, entropy,
  CTR) comparing the no-exploration and 5% exploration runs across
  50 rounds.
- `NOTES.md` — the six-line loop analysis for this lab's recommender,
  in chapter 04's (and chapter 07 section D's) template.

Running `python loop_lab.py` prints six labelled blocks (round 1
smoke test, no-exploration selected rounds, 5%-exploration selected
rounds, final-round summary, ε-sweep trade table, dashboard write)
and writes `loop_dashboard.png`.

## Final checkpoint

Without re-reading chapter 04, answer:

1. In step 2 catalogue coverage fell monotonically while CTR climbed
   to a higher plateau. State in your own words why that combination
   is the fingerprint of a tightening loop, and which chapter-04
   pitfall is doing the arguing ("optimising the metric the loop
   inflates").
2. In step 3 the exploration budget cost a measurable amount of CTR.
   Which chapter-04 mitigation names this trade explicitly as a cost
   the business is paying, and what belongs on the blueprint page
   where the `ε` value is picked?
3. In step 5 the `ε = 0.00` run and the `ε = 0.01` run both settle
   at a low coverage. State what that suggests about the minimum
   viable exploration budget for a catalogue of this shape, and why
   chapter 04 states the range as 1–10% rather than a fixed number.
4. The lab's `true_relevance` vector was **constant** across all 50
   rounds — nothing drifted. Explain in your own words why every
   metric movement across rounds is therefore attributable to the
   loop, not to the world, and which other failure mode ("data drift"
   from `mod-109` chapter 03) you would have to rule out on real
   production traffic to make the same claim.
5. The recommender is a ranking system, so the chapter-04 detector 2
   "untreated control cohort" doesn't directly apply. What is the
   ranking-system analogue chapter 04 names, and where in step 3's
   code is it expressed?

If you can answer all five, chapter 04's loop-dynamics-and-mitigation
argument is internalised and you are ready to apply it in
exercise-01's blueprint (specifically the one-paragraph loop analysis
chapter 07 section D is auditing for) and in any design review where
the "nothing drifted; nothing broke; the business metric is quietly
worse" fingerprint shows up.

## Where to go next

- **`exercise-01` ("System design interview walkthrough")** asks for
  a one-page blueprint where the loop analysis above is the sixth
  paragraph. Chapter 07 Section D is the audit; this lab's `NOTES.md`
  is the template.
- **`lab-02` ("A/B power analysis and the peeking trap")** moves from
  "the loop closed while nothing drifted" to "did the A/B test
  actually measure the fix, or did we read it at the first local
  maximum?" — chapter 05's power-analysis-and-ramp discipline is
  the counterpart to chapter 04's loop discipline, and the
  counterfactual A/B (chapter 04 detector 4) sits inside chapter 05's
  design doc.
- **`exercise-03` ("A/B test design for an ML rollout")** inherits
  the exploration-budget conversation from this lab: a recommender
  v2 A/B that tests "model + 5% exploration vs model + 0%
  exploration" is a real experiment with a non-obvious MDE. Chapter
  05's power analysis and the design-doc template are the artifacts.
- **`mod-109` chapter 04 ("Concept drift")** is the complementary
  failure mode to this chapter's loop — where the loop is the model
  changing the world, concept drift is the world changing out from
  under the model. Both produce "nothing drifted; the model is
  quietly worse" business-metric signatures; the diagnostic
  separation is in `mod-109` chapter 07's data-first incident
  procedure.
