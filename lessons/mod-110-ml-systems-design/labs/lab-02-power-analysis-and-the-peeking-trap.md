# lab-02 — A/B power analysis, the "run for 2 days" trap, and the false-positive cost of peeking

> **Duration:** ~35–45 minutes.
> **Companion chapter:** [05 — A/B testing ML systems](../05-ab-testing-ml-systems.md).

## Objective

Reproduce chapter 05's three load-bearing claims about A/B testing — (a)
the sample size and duration fall out of a four-input formula ("baseline,
MDE, α, power"), (b) a test that says "run for 2 days" is almost always
underpowered, and (c) **peeking** (stopping a test the moment the primary
metric looks significant) inflates the false-positive rate well above `α`
— on a 100-line Python simulation you can run on a laptop. By the end
you will have (a) implemented chapter 05's whiteboard formula
`n ≈ (z_{α/2} + z_β)² × 2 × σ² / Δ²` and reproduced the worked example
(`≈ 184,000` per variant for a 0.3pp lift on a 12% baseline), (b)
produced an MDE-vs-duration table that makes the "run for 2 days" trap
visible, (c) simulated 2,000 **null** A/B tests (treatment effect = 0)
and measured the actual false-positive rate under three stopping rules —
read-at-planned-duration, peek-and-stop-the-first-time-p<0.05, and peek-
twice — showing how the FPR climbs well above `α`, and (d) implemented a
Sample-Ratio-Mismatch (SRM) χ² check and shown it passes on a clean
50/50 routing and **fails loudly** on a routing bug that drops 2% of
treatment users.

This lab is deliberately narrower than **exercise-03** (which asks for a
full pre-registered design packet: design doc, power-analysis script,
risk register, read-out template, simulator, and dry-run reviewer
script). The lab is the isolation chamber where you *see* the sample-
size formula produce chapter 05's worked number, the peeking trap
inflate the FPR on a dataset you can eyeball, and the SRM check behave
on both a clean and a buggy routing layer.

## Prerequisites

- Python 3.11+.
- `numpy`, `scipy` installed in a virtual environment
  (`uv venv && uv pip install numpy scipy` is fine).
- Chapter 05 read through once; the four power-analysis inputs
  (baseline, MDE, α, power), the ramp pattern
  (shadow → canary → test → read-out → launch/roll-back), the four
  canonical read-out traps (novelty, Simpson's, SRM, guardrail-shrug),
  and the pre-registered decision-rule discipline are the backing
  concepts.
- A CPU is enough; no GPU, no Docker, no warehouse, no real experiment
  platform.

## Setup

Create a lab directory and a single Python file:

```bash
mkdir -p lab-02-ab && cd lab-02-ab
touch ab_lab.py
```

Everything below goes in `ab_lab.py`. Run with `python ab_lab.py` after
each step and read the output before moving on.

Start the file with the imports and a fixed seed:

```python
import math
import numpy as np
from scipy import stats

RNG = np.random.default_rng(seed=0)
```

## Steps

### Step 1 — Implement chapter 05's whiteboard sample-size formula and reproduce the worked example (≈ 5 min)

Chapter 05 Section 4 states: *"the per-variant sample size, for a
two-sample z-test on a proportion, is approximately
`n ≈ (z_{α/2} + z_β)² × 2 × σ² / Δ²`"* with the worked example
`7.85 × 2 × 0.12 × 0.88 / 0.003² ≈ 184,000` (baseline conversion
`p₀ = 0.12`, absolute MDE `Δ = 0.003`, α = 0.05 two-sided, power 0.80).

Append to `ab_lab.py`:

```python
def sample_size_two_proportions(p0: float, mde_abs: float,
                                 alpha: float = 0.05,
                                 power: float = 0.80,
                                 two_sided: bool = True) -> int:
    """Chapter 05 Section 4 formula: n per variant for a two-sample
    z-test on a proportion."""
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided \
        else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    # Use the baseline's variance on both sides (chapter 05's approximation).
    sigma_sq = p0 * (1 - p0)
    n = (z_alpha + z_beta) ** 2 * 2 * sigma_sq / mde_abs ** 2
    return math.ceil(n)


# Reproduce chapter 05's worked example.
n_worked = sample_size_two_proportions(p0=0.12, mde_abs=0.003)
print(f"chapter 05 worked example: n per variant = {n_worked:,}")
print(f"chapter 05 states          n per variant ≈  184,000")
```

Run `python ab_lab.py`.

**Expected output.**

```
chapter 05 worked example: n per variant = 184,???
chapter 05 states          n per variant ≈  184,000
```

The result lands within a few hundred of chapter 05's quoted
`≈ 184,000`. The small difference comes from the chapter rounding
`(z_{α/2} + z_β)² ≈ 7.85` vs the exact value `(1.96 + 0.84)² ≈ 7.8485`
— the formula on the whiteboard is the right order of magnitude, and
the numbers it produces are the ones the platform team will quote
back at you.

**Checkpoint.** You have chapter 05 Section 4's whiteboard calculation
in one function. Every number downstream — "weeks to read-out," "break-
even MDE for a 3-week deadline," "the design is not feasible in this
timeframe, renegotiate scope" — is produced by this function with
different inputs.

### Step 2 — Produce the MDE-vs-duration table and make the "run for 2 days" trap visible (≈ 7 min)

Chapter 05's practical rule: *"A test that says 'run for 2 days' is
almost always underpowered or has a novelty problem waiting to bite.
Never read a test in under a week."* Reproduce the number — for a
modest MDE at a plausible eligible-user volume, the duration is weeks,
not days — by sweeping MDE and listing the resulting sample size and
duration.

Append:

```python
# Scenario: chapter 05's worked example at full 50/50 allocation.
BASELINE = 0.12                # 12% conversion rate
ELIGIBLE_PER_DAY = 250_000     # the chapter's worked example
RAMP_FRACTION_FULL = 0.50      # test ramp at 50/50
RAMP_FRACTION_CANARY = 0.10    # alternative 10/10 canary split

def duration_weeks(n_per_variant: int, ramp_fraction: float,
                   eligible_per_day: int = ELIGIBLE_PER_DAY) -> float:
    eligible_per_variant_per_day = eligible_per_day * ramp_fraction
    days = n_per_variant / eligible_per_variant_per_day
    return days / 7


print(f"\n{'MDE abs':>8} {'MDE rel':>9} {'N/variant':>12} "
      f"{'weeks @50/50':>14} {'weeks @10/10':>14}")
for mde_abs in [0.001, 0.002, 0.003, 0.005, 0.010]:
    n = sample_size_two_proportions(BASELINE, mde_abs)
    rel = mde_abs / BASELINE
    w_full = duration_weeks(n, RAMP_FRACTION_FULL)
    w_can = duration_weeks(n, RAMP_FRACTION_CANARY)
    print(f"{mde_abs:>8.3f} {rel:>9.1%} {n:>12,} "
          f"{w_full:>14.1f} {w_can:>14.1f}")
```

**Expected output.** A table along the lines of:

```
 MDE abs   MDE rel    N/variant   weeks @50/50   weeks @10/10
   0.001      0.8%    1,654,???            9.5           47.3
   0.002      1.7%      413,???            2.4           11.8
   0.003      2.5%      184,???            1.1            5.3
   0.005      4.2%       66,???            0.4            1.9
   0.010      8.3%       16,???            0.1            0.5
```

Two chapter-05 lessons fall out of this table on inspection:

- **The "run for 2 days" trap.** Only the `MDE = 0.010` row (an 8.3%
  relative lift — enormous for a mature product) is actually readable
  in under a week at 50/50; everything smaller runs for weeks. A
  stakeholder who says "the test will tell us in 2 days" is implicitly
  claiming an 8%+ relative lift — which chapter 05 Section 4's
  rule-of-thumb three ("a test whose primary metric is a rare event
  needs longer than you think") plus Section 5's "Never read a test
  in under a week" directly target.
- **Ramp allocation dominates duration.** The 10/10 column is roughly
  5× the 50/50 column at the same MDE — if you're testing at a canary
  allocation because the treatment carries risk, the test is 5× longer
  to power. This is the trade chapter 05 Section 5 implies but doesn't
  tabulate: the ramp is `shadow → 1% canary → designed split → read`
  and the designed split picks the accumulation rate.

**Checkpoint.** You have chapter 05's practical rules of thumb in a
table you generated. A line like "A test that requires more users
than the entire eligible population per month is a signal that the
MDE is too small; renegotiate scope before starting" has a concrete
meaning when you can show the 9.5-weeks-at-50/50 cell to the
stakeholder.

### Step 3 — Simulate 2,000 null A/B tests and measure the false-positive rate at the designed duration (≈ 7 min)

Chapter 05 Section 6 names peeking as "*the* classic way to fool
yourself" and states that *"a test read at a random time is
meaningfully more likely to be false-positive than one read at its
planned duration."* The claim deserves a simulation. First, confirm
that at the **designed duration** the false-positive rate is `≈ α` as
advertised — the baseline the next step compares against.

Append:

```python
# Two-sample z-test for proportions, returning a two-sided p-value.
def z_test_proportions(x1: int, n1: int, x2: int, n2: int) -> float:
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


N_SIMS = 2_000
N_PER_VARIANT = 10_000      # small enough to be fast; both arms draw i.i.d.
ALPHA = 0.05

def simulate_null_tests(n_sims: int, n_per_variant: int,
                        rng: np.random.Generator):
    """Each 'experiment': generate n draws per variant from the SAME
    Bernoulli(BASELINE). No real effect. Return the final-read p-value."""
    p_values = np.empty(n_sims)
    for i in range(n_sims):
        a = rng.random(n_per_variant) < BASELINE
        b = rng.random(n_per_variant) < BASELINE
        p_values[i] = z_test_proportions(int(a.sum()), n_per_variant,
                                          int(b.sum()), n_per_variant)
    return p_values


rng = np.random.default_rng(seed=1)
p_designed = simulate_null_tests(N_SIMS, N_PER_VARIANT, rng)
fpr_designed = float((p_designed < ALPHA).mean())
print(f"\n{N_SIMS} null tests, read at planned duration:")
print(f"  observed FPR = {fpr_designed:.3f}  (nominal α = {ALPHA})")
```

**Expected output.**

```
2000 null tests, read at planned duration:
  observed FPR = 0.0??  (nominal α = 0.05)
```

The observed FPR lands close to the nominal `0.05`, within Monte-Carlo
noise. This is the contract: at the pre-registered read time, the
probability of a false positive is `α` by construction.

**Checkpoint.** The baseline is established: with the designed-duration
read rule, the observed FPR matches `α`. Any inflation in the next step
is attributable to the **peeking rule**, not to the simulation.

### Step 4 — Enable peeking and watch the false-positive rate climb well above α (≈ 8 min)

Chapter 05: "*the per-look false-positive rate compounds across the
many looks.*" Simulate the same null null A/B tests, but now allow the
experimenter to *peek* at a sequence of intermediate sample sizes and
call the test as soon as `p < α` at any of them.

Append:

```python
def simulate_peeking_tests(n_sims: int, n_final: int,
                            n_peeks: int,
                            rng: np.random.Generator) -> float:
    """Each 'experiment': accumulate samples up to n_final, performing a
    z-test at n_peeks evenly spaced checkpoints. Call 'significant' the
    FIRST time any checkpoint's p < α (chapter 05's peeking rule). Return
    the fraction of experiments that were ever called significant."""
    checkpoints = np.linspace(n_final / n_peeks, n_final, n_peeks,
                              dtype=int)
    rejected = 0
    for _ in range(n_sims):
        a_all = rng.random(n_final) < BASELINE
        b_all = rng.random(n_final) < BASELINE
        for n_so_far in checkpoints:
            p = z_test_proportions(
                int(a_all[:n_so_far].sum()), int(n_so_far),
                int(b_all[:n_so_far].sum()), int(n_so_far),
            )
            if p < ALPHA:
                rejected += 1
                break
    return rejected / n_sims


print(f"\npeeking inflates the false-positive rate:")
print(f"{'# peeks':>8} {'FPR observed':>14} {'(nominal α = 0.05)':>20}")
for n_peeks in [1, 2, 5, 10, 20, 50]:
    rng = np.random.default_rng(seed=2 + n_peeks)
    fpr = simulate_peeking_tests(N_SIMS, N_PER_VARIANT, n_peeks, rng)
    print(f"{n_peeks:>8} {fpr:>14.3f}")
```

**Expected output.** Something along the lines of:

```
peeking inflates the false-positive rate:
 # peeks   FPR observed   (nominal α = 0.05)
       1          0.0??
       2          0.0??
       5          0.1??
      10          0.1??
      20          0.1??
      50          0.2??
```

The exact numbers vary with the seed and with the number of samples
per experiment; the **shape** is what to read. Two properties fall
out that chapter 05 relies on:

- **FPR climbs with every peek.** `n_peeks = 1` reproduces step 3's
  `≈ α` baseline (the "peek" lines up with the designed read). `n_peeks
  = 2` is already above `α`. `n_peeks = 50` — a team that watches the
  dashboard hourly for 50 hours — can run 3–5× the false-positive rate
  they advertised.
- **The inflation is not a bug.** The underlying test is correct at
  every checkpoint. The problem is the **stopping rule**: "stop the
  first time any checkpoint rejects" is a different hypothesis test
  than "test once at the designed time," and it has a different FPR.

**Checkpoint.** Chapter 05 pitfall #1 ("Reading the test early. Peeking
inflates false-positive rate. If your platform doesn't support
sequential testing, you read at the designed duration, period.") is now
a number you measured, not a bullet in the chapter. Chapter 05's
suggested escape hatches — **SPRT, always-valid p-values, mixture
supermartingales** — exist specifically to make continuous monitoring
safe; this simulation is why they exist.

### Step 5 — Implement the SRM check and watch it pass on a clean 50/50 and fail loudly on a 2%-drop routing bug (≈ 8 min)

Chapter 05 Section 6: *"Sample-ratio mismatch (SRM). The design says
50/50; the actual ratio at read-out is 51.3/48.7. If the mismatch is
statistically significant, something is wrong with the assignment. … SRM
invalidates the test's statistical claims entirely; the fix is to figure
out why the assignment is asymmetric, not to interpret the result.
Every mature experimentation platform runs an automated SRM check."*
Implement one.

Append:

```python
def srm_check(n_control: int, n_treatment: int,
              expected_control_fraction: float = 0.5,
              alpha: float = 0.001) -> dict:
    """Chi-squared goodness-of-fit against the designed assignment ratio.
    Returns observed fractions and the test's p-value. Chapter 05's note:
    'alert on the read-out dashboard' — the alpha is conventionally far
    tighter than the experiment's alpha (1e-3 is common) so the SRM alarm
    only fires on routing bugs, not on sampling noise."""
    n_total = n_control + n_treatment
    expected_c = n_total * expected_control_fraction
    expected_t = n_total * (1 - expected_control_fraction)
    chi2 = ((n_control - expected_c) ** 2 / expected_c
            + (n_treatment - expected_t) ** 2 / expected_t)
    p_value = float(1 - stats.chi2.cdf(chi2, df=1))
    return {
        "observed_control_fraction": n_control / n_total,
        "observed_treatment_fraction": n_treatment / n_total,
        "chi2": chi2,
        "p_value": p_value,
        "srm_detected": p_value < alpha,
    }


# Scenario 1: clean 50/50 routing. Each user independently assigned
# with probability 0.5 to each arm.
N_USERS = 200_000
rng = np.random.default_rng(seed=10)
assignments_clean = (rng.random(N_USERS) < 0.5).astype(int)  # 1 = treatment
n_c_clean = int((assignments_clean == 0).sum())
n_t_clean = int((assignments_clean == 1).sum())

# Scenario 2: routing bug drops 2% of the treatment arm (users bounce
# back to control due to a stale cookie, say).
rng = np.random.default_rng(seed=11)
assignments_bug = (rng.random(N_USERS) < 0.5).astype(int)
drop_mask = (assignments_bug == 1) & (rng.random(N_USERS) < 0.02)
assignments_bug = np.where(drop_mask, 0, assignments_bug)
n_c_bug = int((assignments_bug == 0).sum())
n_t_bug = int((assignments_bug == 1).sum())

for label, n_c, n_t in [("clean 50/50 routing", n_c_clean, n_t_clean),
                        ("2% drop routing bug", n_c_bug, n_t_bug)]:
    result = srm_check(n_c, n_t)
    print(f"\n{label}:")
    print(f"  control: {n_c:,}   treatment: {n_t:,}")
    print(f"  observed control fraction = "
          f"{result['observed_control_fraction']:.4f}")
    print(f"  chi² = {result['chi2']:.2f}   p = {result['p_value']:.3g}")
    print(f"  SRM detected: {result['srm_detected']}")
```

**Expected output.**

```
clean 50/50 routing:
  control: 99,???   treatment: 100,???
  observed control fraction = 0.50??
  chi² = ?.??   p = 0.???
  SRM detected: False

2% drop routing bug:
  control: 100,???   treatment: 99,???
  observed control fraction = 0.50??  (but shifted toward control)
  chi² = ??.??   p = 1.??e-??
  SRM detected: True
```

The clean routing produces `control_fraction ≈ 0.500` within sampling
noise — the SRM check's χ² is small and the p-value is well above the
`1e-3` alarm threshold. The 2%-drop bug produces `control_fraction ≈
0.505` — a tiny shift, far smaller than most eyeballing-the-dashboard
intuitions, and still a p-value that fires the alarm loudly because
`N` is large enough. Chapter 05's note that *"every mature
experimentation platform runs an automated SRM check"* is exactly this:
on modest `N`, a 0.5% shift in the routing ratio is statistically
screaming that the routing layer has a bug.

**Checkpoint.** Chapter 05's SRM instruction is now one function, two
assertions, and one alpha threshold. Pitfall #7 ("No SRM check. The
routing layer is buggy; the test's assumptions don't hold; nobody
notices") is a one-line check away. When an SRM alarm fires on a
production A/B, chapter 05's rule applies: *"the fix is to figure out
why the assignment is asymmetric, not to interpret the result."*

### Step 6 — Write the one-page NOTES.md tying the three numbers together (≈ 5 min)

Chapter 05 Section 7's design doc and Section 8's read-out template
both reference specific numbers — the power analysis, the ramp plan,
the stopping rules. Write down the ones you produced in this lab so
a reviewer of your future exercise-03 design-doc can see where the
discipline came from.

```markdown
# lab-02 — A/B power / peeking / SRM notes

Three chapter-05 claims, measured on a laptop:

Power analysis (chapter 05 Section 4):
- Formula: n ≈ (z_{α/2} + z_β)² × 2 × σ² / Δ²
- Worked example reproduced: 12% baseline, 0.3pp absolute MDE, α=0.05
  two-sided, power=0.80 → n ≈ 184k per variant (matches chapter 05's
  quoted number within rounding).
- MDE-vs-duration table shows: the "run for 2 days" ask is implicitly
  claiming an 8%+ relative lift; anything smaller runs for weeks at
  50/50 and ~5× longer at 10/10 canary.

Peeking trap (chapter 05 Section 6, pitfall #1):
- Designed-duration read: observed FPR ≈ α (contract satisfied).
- One peek before the planned read: observed FPR > α.
- Twenty peeks (an hourly dashboard watch): observed FPR 2–4× α.
- Fifty peeks: observed FPR 3–5× α.
- The inflation is in the STOPPING RULE, not the test. Mitigations:
  read at designed duration (chapter 05 pitfall #1), or use
  sequential-testing methods (SPRT, always-valid p-values) that make
  continuous monitoring safe.

SRM check (chapter 05 Section 6, pitfall #7):
- Chi-squared goodness-of-fit against designed assignment ratio with
  α = 1e-3 (far tighter than the experiment's α, so only routing bugs
  fire — not sampling noise).
- Clean 50/50 routing: SRM not detected.
- 2% drop on treatment arm (routing bug — users bounce back to
  control): SRM detected loudly (p ≪ 1e-3) even though the raw
  control-fraction shift is only ~0.005.
- Rule chapter 05 states: fix the routing; do not interpret the result.

Other chapter 05 traps THIS LAB did NOT address — handled in exercise-03:
- Novelty effects (distrust the aggregate if the effect is not
  sustained over the last third of the test's duration).
- Simpson's paradox (slice-level metrics must agree with aggregate).
- Guardrail-shrug (pre-registered stopping rule is the arbiter of
  guardrail breaches, not the vibe of the read-out meeting).
```

**Checkpoint.** You have derived chapter 05's three most load-bearing
numeric claims from one `.py` file. Every line in `NOTES.md` traces
back to a specific number you measured in the lab, not a cargo-cult
default.

## Expected final artefact

A lab directory containing:

- `ab_lab.py` — the single file that implements the sample-size
  formula, the MDE-vs-duration sweep, the 2,000-run null-A/B
  simulation at planned duration, the peeking-rule simulation across
  1–50 peeks, and the SRM χ² check on both a clean and a buggy
  routing.
- `NOTES.md` — the one-page summary of the three chapter-05 claims
  the lab measured, with pointers to the exercise-03 topics the lab
  did not cover (novelty, Simpson's, guardrail-shrug).

Running `python ab_lab.py` prints five labelled blocks (worked-example
sample size, MDE-vs-duration table, designed-duration FPR, peeking
FPR table, SRM results for both routing scenarios).

## Final checkpoint

Without re-reading chapter 05, answer:

1. In step 1 you reproduced n ≈ 184,000 per variant from the four
   inputs. Which specific input would you change — and in which
   direction — if the stakeholder says "the launch has a 3-week
   deadline"? Chapter 05 Section 4's **break-even MDE** question is
   this inversion.
2. In step 2 the `MDE = 0.001` row needed 9.5 weeks at 50/50 and
   47.3 weeks at 10/10 canary. State in your own words why chapter
   05 recommends running the designed test allocation at 50/50
   after the shadow / canary phases, not a 10/10 canary the whole
   way.
3. In step 4 fifty peeks pushed the observed FPR to roughly
   3–5× the nominal α. Which chapter 05 Section 5 discipline is
   the primary prevention (short of implementing SPRT /
   always-valid p-values), and which pitfall number names this
   failure mode verbatim?
4. In step 5 the 2%-drop routing bug produced `control_fraction ≈
   0.505` — eyeballing-the-dashboard would call that "close
   enough." State in your own words why an SRM check with `α =
   1e-3` can reliably reject the null anyway, and why chapter 05
   insists SRM is a **test-invalidating** failure rather than a
   results-adjusting one.
5. The lab's power analysis is for a two-sample z-test on a
   proportion. Chapter 05 names **CUPED** and **stratified
   sampling** as two variance-reduction techniques that cut the
   required sample without biasing the estimate. State in your own
   words what each is doing at a high level and why the required
   `n` shrinks under each.

If you can answer all five, chapter 05's power-analysis-and-stopping-
rule argument is internalised and you are ready to apply it in
exercise-03's design doc (Section 4 "Power analysis" and the
pre-registered decision rule) and in any launch-review conversation
where the question "why can't we read the test early?" comes up.

## Where to go next

- **`exercise-03` ("A/B test design for an ML rollout")** inherits
  this lab's power analysis and SRM check — Part B asks for a
  reproducible `power.py` and a `POWER.md` table (which is a
  more-featured version of step 2's output), Part C's RISKS.md
  asks for a specific pre-registered SRM alerting threshold
  (which is step 5's `α = 1e-3`), and Part D's `TEMPLATE.md`
  reads the result of a fully-powered run at designed duration
  (which is step 3's baseline, not step 4's inflated peeking
  simulation).
- **`lab-01` ("Recommender loop collapse and exploration")**
  produces the fingerprint of a loop that chapter 05's A/B test
  is one of the only tools to directly measure — a bandit or
  random-variant A/B against the model (chapter 04 detector 4)
  uses exactly the machinery this lab built.
- **`mod-109` chapter 06 ("The retraining trigger")** uses
  chapter 05's shadow-mode / canary / kill-switch mechanics
  inside its evaluation gate — specifically, "the candidate
  must beat a stable pre-drift holdout, not the recent window"
  and the slice-level checks sit downstream of this lab's
  sample-size-and-power discipline.
