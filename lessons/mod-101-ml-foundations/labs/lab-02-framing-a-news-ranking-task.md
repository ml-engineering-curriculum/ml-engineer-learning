# lab-02 — Framing walk-through: ranking news articles

> **Duration:** ~30–45 minutes.
> **Companion chapters:** [07 — Framing a problem](../07-framing-an-ml-problem.md), [08 — Regression vs. classification vs. ranking](../08-regression-classification-ranking.md).

## Objective

Practise the framing conversation from chapters 07 and 08 on a single,
compact scenario, in the time it would take to actually have the
conversation with a stakeholder. By the end you will have a one-page
written framing of a realistic request, consciously **choosing between
regression, classification, and ranking** before touching any
modelling tool.

This lab is deliberately short and conversation-shaped. The full
rubric authoring exercise and three deep-dive scenarios live in
`exercise-04`; this lab is a *warm-up* on a different scenario, meant
to be done the first time you read chapters 07/08.

## Prerequisites

- Chapters 07 and 08 read through once.
- No code required. A plain text / markdown editor is enough.
- Optional: skim the "decision tree for framing" box in chapter 07
  and the "How to choose: a checklist" box in chapter 08 before you
  start.

## The scenario

> *Stakeholder:* "Our product-news app shows users a feed of ~20
> articles the first time they open the app each morning. We want ML
> to pick which articles to show, and in what order, so people read
> more. Right now we show the newest 20 articles. Can ML do better?"

Available signals you can draw on (do not invent new ones for this
lab):

- A **catalogue** of 500–2,000 fresh articles per day, each with
  text (headline + body), publication time, source, and editorial
  topic tags.
- **Impression / click logs** from the current newest-20 feed over
  the last 12 months. For each (user, article) impression you have
  whether the user clicked, how long they read for, and whether
  they shared the article.
- **User profiles** — coarse interest tags inferred from past
  reading plus registration country.
- An **editorial team** that labels a small ongoing stream (~200
  articles / week) with a hand-assigned "quality" score from 1 to 5.

A hard constraint you cannot negotiate away: the user is shown
**exactly 20 articles** per morning open, in order.

## Setup

Create a lab directory and a working file:

```bash
mkdir -p lab-02-news-framing && cd lab-02-news-framing
touch FRAMING.md
```

Everything you write in this lab goes in `FRAMING.md`. Keep it to one
page.

## Steps

### Step 1 — Rephrase the stakeholder ask as a decision (≈ 3 min)

At the top of `FRAMING.md`, write one sentence that states the
**business decision this model will drive**, *without* the words
"machine learning" or "AI". Chapter 07's framing checklist calls this
out as the single most clarifying question.

**Example of what to write.** "For each user's morning open, pick 20
articles out of the day's catalogue of ~1,500 and decide what order to
show them in."

**Checkpoint.** Does your sentence name the *unit* of prediction (per
user per morning), the *size* of the thing being produced (20), and
the *action* (select + order)? If any of those three are missing,
rewrite it before continuing.

### Step 2 — Walk the chapter-07 decision tree (≈ 5 min)

In `FRAMING.md` under a `## Framing: supervised / unsupervised / RL`
heading, answer each branch of chapter 07's decision tree in one
sentence:

1. **Do you have labelled targets?** Which signals in the "Available
   signals" list are usable as labels, and how clean are they?
2. **Do the actions you take materially change the inputs you see
   later?** The ranked list you show today influences the clicks you
   log tomorrow; is that enough to flip you to RL?
3. **If supervised, is this a prediction or a representation task?**
   (Chapter 07 lists clustering, dim-reduction, anomaly detection,
   self-supervised pre-training as alternatives — are any of those
   actually what the stakeholder is asking for?)

Land on one of **supervised**, **unsupervised**, or **RL**. Chapter 07
is unambiguous about the heuristic: "frame as supervised unless the
action your model takes meaningfully changes the future inputs it
sees."

**Checkpoint.** If you ended up at RL, re-read the "When RL is the
*wrong* framing" paragraph in chapter 07 and justify in one sentence
why your scenario really is the exception.

### Step 3 — Rule in or rule out each supervised framing (≈ 10 min)

Under a `## Framing: regression / classification / ranking` heading in
`FRAMING.md`, write a short pro-and-con for each of the three
supervised sub-framings on this scenario. Use chapter 08's checklist
("How to choose") verbatim as your scaffolding — specifically its four
questions:

1. **What does the downstream decision look like?**
2. **What signal do you actually have?**
3. **What is the natural noise model?**
4. **What metric will the business actually inspect?**

For each sub-framing, write *at most* 4 bullets:

- **Regression.** E.g. "predict expected time-on-article"; "would use
  MSE or Huber"; "pro: calibrated magnitude; con: optimises absolute
  error, not order."
- **Classification.** E.g. "predict `P(click | user, article)` per
  pair"; "would use binary cross-entropy"; "pro: clean signal from the
  log; con: pointwise, doesn't know items share a slate."
- **Ranking.** E.g. "score each candidate, rank, keep top-20";
  "pairwise or listwise loss"; "pro: matches the operational reality
  (fixed slate); con: needs pairwise preference signal — can we build
  that from click vs. no-click within a shown slate?"

**Checkpoint.** You should now have three mini-sections, each ≤ 4
bullets. If one section is noticeably longer than the others, you
probably haven't thought hard enough about the others — go back.

### Step 4 — Recommend one, with the thing you'd change tomorrow (≈ 5 min)

Under a `## Recommendation` heading, write **two paragraphs** (one
short, one shorter):

1. **The pick.** One of {regression, classification, ranking} — plus
   one sentence of justification tied to chapter 08's criteria.
   Chapter 08's own worked example (support-ticket triage) ends on a
   deliberate "there is no right answer in the abstract" — your pick
   is right if the justification is right, not if it matches a key.
2. **The first concrete thing you'd do tomorrow.** This is almost
   never "train a model." It is more likely "instrument the current
   newest-20 feed to log impression position so we can later debias
   the click signal", or "ask the editorial team whether the quality
   score is per-article or per-slot."

**Checkpoint.** Your "first thing tomorrow" bullet should not require
any training infrastructure. If it does, you are already modelling.

### Step 5 — Call out two failure modes (≈ 5 min)

Chapter 07 and chapter 08 both emphasise that framing is partly about
*naming the ways this will fail before you start*. Under a `## How
this fails` heading, pick exactly **two** failure modes from the
following list (or add your own, if you have one more specific) and
write one sentence of mitigation for each:

- The i.i.d. assumption between training logs and serving breaks
  (chapter 04): e.g. breaking-news days look nothing like the training
  window.
- Label bias from the current ranker: you only see clicks on articles
  the newest-20 heuristic already picked.
- Class imbalance: most impressions are no-clicks (chapter 08).
- Calibration vs. order: a ranker that gets order right can be badly
  mis-calibrated as a classifier, and vice versa (chapter 08).
- A sceptic-proof metric does not exist in the available logs.

**Checkpoint.** Each mitigation should name something *specific* you
would do (an experiment, a logging change, a metric choice) — not
"we'd need to be careful about this."

### Step 6 — Self-review against the chapter 08 "conflict" test (≈ 3 min)

Chapter 08 ends on a specific test: *if your answers to the four
checklist questions conflict with each other, that is information*.
Re-read your own `## Framing: regression / classification / ranking`
section. Do the four questions all point at the same framing? If
not, briefly note the conflict in `FRAMING.md` under
`## Open questions`.

## Expected final artefact

A single `FRAMING.md` file, one page or less, structured roughly as:

```
# News-feed ranking: framing

<Step 1 — one-sentence decision>

## Framing: supervised / unsupervised / RL
<Step 2 — three sentences>

## Framing: regression / classification / ranking
### Regression
<Step 3 bullets>
### Classification
<Step 3 bullets>
### Ranking
<Step 3 bullets>

## Recommendation
<Step 4 — two short paragraphs>

## How this fails
- <Step 5 — failure mode 1 + mitigation>
- <Step 5 — failure mode 2 + mitigation>

## Open questions
<Step 6 — any conflicts you noticed>
```

## Final checkpoint

Before you close `FRAMING.md`, read it top-to-bottom and ask:

1. Could a junior ML engineer pick up this doc and start work tomorrow
   without needing to re-ask any of the questions in chapter 07's
   decision tree or chapter 08's checklist?
2. Have you made the trade-off between the three supervised framings
   *visible*, rather than just picking one and justifying it?
3. Did you resist suggesting a specific model / library / library
   function anywhere in the doc?

If all three are yes, you've done the exercise correctly. If (3) is
no, re-read chapter 07's warning: "A common trap is to slide into
modelling decisions. You are framing, not modelling."

## Where to go next

- `exercise-04` turns this walk-through into a **rubric** you can use
  on any new ML request, plus three deeper framing scenarios
  (support-ticket triage, ad-spend forecasting, factory defect
  detection) that each take ~30 minutes.
- The "metric a sceptic would accept" bullet in Step 4 is the hook
  into `mod-107`, which owns metric design and validation.
