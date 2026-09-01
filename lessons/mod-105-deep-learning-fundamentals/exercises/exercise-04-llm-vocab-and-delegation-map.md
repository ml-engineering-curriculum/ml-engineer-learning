# exercise-04 — LLM vocab and delegation map

> **Estimated effort:** 3 hours.
> **Companion chapter:** [10 — Transformer LLMs and when to delegate](../10-transformer-llms-and-when-to-delegate.md).

## Objective

Produce a personal, structured reference document — a
**vocab-and-delegation map** — that you (and a future
teammate) will consult for years. This exercise is deliberately
short on code and long on *organised writing*. The `mod-105`
LLM objective is "gain working familiarity with transformer
LLMs and where to delegate to specialist tracks." That is a
written, not an implemented, artefact.

By the end you will have:

- A glossary of the LLM vocabulary from chapter 10, in your
  own words, cross-linked and calibrated against primary
  sources (papers and vendor docs).
- A **decision tree / delegation map** you can hand a
  colleague when they ask "should we use an LLM here, and
  what specialist do we need?"
- A short set of **model cards** you have read and can
  compare — so that "read an LLM model card and know what
  every field means" (chapter 10) is a thing you have
  actually done.
- One tiny piece of hands-on code (Part D) that grounds the
  vocabulary in a real API call. No fine-tuning, no self-
  hosting — that is the specialist track's job.

There is **no from-scratch model training** in this
exercise. If you find yourself building a model, you are
outside the scope of the deliverable.

## Prerequisites

- Chapter 10 read carefully.
- Chapters 07–09 read (you need to be able to speak in
  terms of "attention", "AdamW", "LR warmup", "mixed
  precision" without hedging).
- Python 3.11+, an internet connection, and a text editor.
- **Optional but very useful:** an API key for one hosted
  LLM provider (Anthropic, OpenAI, Google, an open-source
  gateway like OpenRouter, or a local Ollama install). Part
  D is much more informative if you can actually make a call.
  If you do not want to pay for API access, use a free-tier
  provider or your local Ollama; a call to a 1B–7B open-
  source model is enough to make the vocabulary concrete.

## Problem statement

Everything in this exercise goes into a single directory that
looks like a small documentation site. The default entry
point is `README.md`; the substantive pieces live in
`glossary.md`, `delegation_map.md`, `model_cards.md`, and
`prompt_playground/`.

### Part A — Glossary (≈ 60 min)

In `glossary.md`, take **every term** from chapter 10's "The
vocabulary, alphabetically-ish" section and write your own
definition. For each term, produce:

- A **one-sentence, plain-language** definition. Assume the
  reader has heard the term but does not know what it means.
- A **two-to-four-sentence** expansion covering the
  mechanics or the operational trade-off. What it costs,
  what it enables, when it is the wrong choice.
- **At least one primary source** — a link to a paper,
  vendor documentation, or PyTorch / Hugging Face reference
  that backs the definition. Not blog posts. Not Wikipedia
  (as a primary source; you may cite alongside).
- **Cross-links** to other glossary entries where the terms
  touch. Markdown wiki-links or plain sectioned anchors,
  your call — but they must resolve.

The list, at minimum (from chapter 10 — add any that come up
in your reading):

- Token; tokenizer (BPE, WordPiece, SentencePiece);
  vocabulary.
- Context length / context window.
- Parameters; memory footprint at inference.
- Pre-training; base model; next-token prediction; masked
  language modelling.
- Post-training; supervised fine-tuning (SFT); preference
  optimisation (RLHF, DPO).
- Fine-tuning; parameter-efficient fine-tuning (PEFT); LoRA;
  QLoRA.
- Prompt; system prompt; few-shot examples; in-context
  learning.
- Sampling / decoding: temperature; top-k; top-p / nucleus;
  beam search.
- Logits; logprobs.
- Instruction-tuned / chat model vs. base model.
- Prompt engineering.
- RAG (retrieval-augmented generation); embedding model;
  vector database.
- Function calling / tool use; agent.
- KV cache; paged attention; prefix caching.
- Quantization (INT8, INT4, GPTQ, AWQ, bitsandbytes).
- FLOPs; training compute; inference compute.
- MoE (Mixture of Experts).
- Prompt tokens vs. completion tokens; billing.
- Distillation.
- Evaluation benchmarks (MMLU, GSM8K, HumanEval, HellaSwag,
  MT-Bench, chatbot arena).
- Hallucination.

You may add other terms you encounter while writing the
delegation map (Part B) or the model cards (Part C); if you
add one, add it to the list in `README.md` too.

**Style constraint.** No LLM-generated glossary passes this
exercise. If a paragraph reads like a marketing page — vague,
symmetrical, hedged — rewrite it. The value of the glossary is
that it is *yours* and it is opinionated where it should be.

### Part B — Delegation map (≈ 45 min)

In `delegation_map.md`, expand chapter 10's delegation section
into a document you would hand a teammate. Include, at
minimum:

1. **A decision tree** (a rendered flowchart or a
   well-structured nested list) that answers the meta-
   question: "a task lands on my desk that involves text /
   language. Where does it go?" Distinguish:
   - Stays in `mod-105`.
   - Delegates to LLM Engineering.
   - Delegates to MLOps / Deployment.
   - Delegates to ML Infra / Distributed Training.
   - Delegates to Applied Vision, Speech, or Research.
   - "Not an LLM problem at all" (chapter 10's decision-map
     path).

   The tree should have at least **eight leaves** covering
   distinct outcomes, not four leaves with hedging.

2. **Five worked scenarios**, each 3–5 sentences. Pick
   plausible scenarios you might actually see — do not
   invent them all from scratch, borrow from real projects,
   news, or your own experience. For each: the ask, the
   right delegation target, and the one line of reasoning
   that got you there. Suggested prompts:
   - "We want to add a chatbot to our customer support
     portal."
   - "Our internal wiki is 40k pages and nobody can find
     anything."
   - "The mobile app is doing on-device Whisper for voice
     input and it's too slow."
   - "Marketing wants an image generator on our
     e-commerce product pages."
   - "We're getting flagged for hallucinated legal citations
     in the summarisation feature."
   - "The API bill for GPT-4-class calls hit six figures
     last month; can we self-host?"

   Feel free to substitute your own scenarios if you have
   more relevant ones.

3. **A "when not to use an LLM" section.** Concrete list;
   chapter 10's decision map is the base, but add the
   scenarios you have seen or read about where an LLM was
   the wrong tool. At least three items.

4. **An "escalation criteria" section.** A short list of
   signals that mean "this has grown beyond `mod-105` — hand
   it off." e.g., cost per query above a threshold; latency
   floor that self-hosted serving would break; data
   sensitivity that rules out third-party APIs; regulatory
   audit requirements; specific capabilities (long context,
   tool use at high concurrency) that the current stack
   can't meet.

### Part C — Model cards read and compared (≈ 30 min)

In `model_cards.md`, read and summarise **three model cards**
from at least two different provider families, following
chapter 10's "Reading an LLM paper productively" template.

Pick from (or beyond):

- **Meta Llama 3** family cards (Llama-3.1-8B or Llama-3.3-
  70B). GitHub or the Hugging Face model page.
- **Mistral** — Mistral-7B, Mixtral-8x7B (a real MoE),
  Mistral-Nemo, or the newer Mistral Small / Medium.
- **Alibaba Qwen** — Qwen2.5-7B, Qwen3, or a Qwen-Coder
  variant.
- **DeepSeek** — DeepSeek-V3 or DeepSeek-R1 (reasoning
  model — good contrast with a non-reasoning card).
- **Google Gemma 2** or **Gemma 3**.
- **Anthropic Claude** system card (Claude 3.7 Sonnet or
  Claude 4 family).
- **OpenAI** — a GPT-4-class or GPT-5-class model card, or
  the GPT-OSS 20B / 120B open-weights cards.

Pick a **mix of open-weights and closed-API** if possible;
the two flavours of card look quite different, and noticing
that is part of the point.

For each card, produce a comparison-friendly summary with:

| Field | Value |
|---|---|
| Model name and version | |
| Provider | |
| Release date | |
| Total parameters (and active, for MoE) | |
| Layers, heads, `d_model` | |
| Context length | |
| Tokenizer vocab size | |
| Positional encoding (learned / sinusoidal / RoPE / ALiBi) | |
| Attention variant (MHA / GQA / MQA) | |
| Training corpus (size in tokens; mixture, if disclosed) | |
| Post-training method (SFT + RLHF / DPO / other) | |
| Reported benchmark headlines (MMLU, GSM8K, HumanEval, etc.) | |
| License (open-weight? commercial use?) | |
| Known limitations (from the card itself) | |

Some fields will be **"not disclosed"** on closed-API cards —
record that. The gap between "here is the recipe" and "here
is what it can do" is one of the interesting differences
between open and closed cards; call it out in one short
paragraph beneath each table.

End the file with a **one-page comparison paragraph**: given
these three cards, which would you pick for a task that
needs (a) low-cost high-throughput classification, (b) long
document summarisation with 100k+ token context, and (c) a
code-assistant integration — and why?

### Part D — Prompt / API playground (≈ 30 min)

In `prompt_playground/`, write a **tiny** script that makes
one LLM API call and does something interesting with the
response. The goal is to *ground the vocabulary in code you
have actually run*, not to build an application.

Required elements:

1. `call.py` — the script. Uses **one** provider (Anthropic
   SDK, OpenAI SDK, `ollama`, `openrouter`, `together`, or
   `google-generativeai` — pick one). Reads the API key from
   an environment variable, never hard-coded.
2. Sends a prompt with:
   - A **system prompt** (persona / constraint).
   - **One or two few-shot examples** (in-context learning).
   - A user turn.
3. Reads back at least two things from the response:
   - The generated text.
   - The **prompt-token and completion-token counts** (every
     provider's SDK exposes these; know how to find them for
     yours).
4. Runs the same prompt at **three temperatures** — `0.0`,
   `0.7`, `1.2` — and prints the outputs side by side.
   Discuss what changed in `NOTES.md`.
5. In `NOTES.md`, answer:
   - Which fields from Part C's model card did you actually
     use / see reflected in the response?
   - What did the temperature sweep show?
   - What did the token counts imply about cost? Multiply
     by the provider's public price-per-1M-tokens and
     estimate what 10,000 similar calls would cost.
   - Would you use this model for the task described in the
     prompt in production? Why or why not?

**Ethics / safety.** Do not use this exercise to generate
harmful content, spam, or something you would not send from
your own account. The prompt should be work-appropriate. If
the provider blocks the prompt, that is itself an
interesting result — record it in `NOTES.md`.

**Cost cap.** Set yourself a $1 budget for this part. If
your test spend exceeds that, something is wrong with the
prompt.

**If you cannot get API access:** Use Ollama locally with a
1B–7B parameter open-weights model
(<https://ollama.com/library>). The exercise still works
end-to-end; note in `NOTES.md` that costs and rate limits
were not exercised.

### Part E — Wire it together (≈ 15 min)

Write a `README.md` for the deliverable that:

1. Names the exercise and its objective.
2. Links to each part (`glossary.md`, `delegation_map.md`,
   `model_cards.md`, `prompt_playground/`).
3. Includes a **one-page summary** of the delegation map so
   that a colleague opening the repo can get the answer in
   30 seconds without reading the full document.
4. Notes anything you would revisit later — chapter 10 will
   age faster than the earlier chapters, and this exercise
   is meant to be updated. Include a "last-refreshed" date
   in the front matter.

## Starter guidance

- **Do the glossary first.** The rest of the exercise
  assumes you can write the terms without looking them up.
  If Part B feels hand-wavy, come back to Part A first.
- **Cite as you write.** It is much slower to add sources
  afterwards. When you write "RoPE", link the paper on the
  same line.
- **Use the primary source, not a blog.** The Llama or
  Claude or Mistral model card *is* the source; a
  Medium post about it is a secondary source at best.
- **Read at least one card that surprises you.** Cards for
  reasoning models (DeepSeek-R1, Claude with extended
  thinking, o1/o3) are structurally different from
  instruction-tuned cards; read at least one of each family
  if you can.
- **The prompt playground is for *understanding*, not for
  building.** Do not extend it into an agent, a RAG system,
  or a chat UI. Every one of those belongs in the LLM
  Engineering track; you are marking the boundary here, not
  crossing it.
- **When in doubt about a boundary, delegate.** For this
  exercise, "delegates to LLM Engineering" is always a valid
  answer. The delegation map is *for you*; the
  conservative bias in it is a feature.

## Acceptance criteria

- `README.md` exists and links to each part with a working
  path.
- `glossary.md` covers every term in the required list, with
  the four required elements per entry (plain definition,
  expansion, cited primary source, cross-links). Any LLM-
  generated blandness has been rewritten.
- `delegation_map.md` contains:
  - A decision tree with at least eight distinct leaves.
  - Five worked scenarios with delegation targets and
    reasoning.
  - A "when not to use an LLM" section (≥ 3 items).
  - An "escalation criteria" section (≥ 4 items).
- `model_cards.md` contains three completed card summaries
  with the required fields, and the one-page comparison
  paragraph.
- `prompt_playground/call.py` runs and produces output for
  three temperatures. `NOTES.md` answers all four required
  questions.
- No API key or personal secret is committed. Verify with
  `git diff --cached | grep -iE 'sk-|api[_-]key|bearer'`
  returning nothing.
- The whole deliverable reads like something you'd send a
  new hire on day one. If it does not, you have not made it
  personal enough.

## Reflection questions (answer in `README.md`)

- Which term in chapter 10 turned out to mean something
  different from what you assumed before reading? What was
  the misconception?
- Which of your five delegation scenarios was hardest to
  place? What made it ambiguous, and would a more senior
  engineer have decided differently?
- Reading three model cards, what did the cards *not* tell
  you that you wish they had? (This is the calibration for
  future card-reading — you will notice omissions faster.)
- Given what you now know: on your current or most recent
  project, is there a task that was solved with an LLM
  where a smaller model or non-ML approach would have been
  better? Or vice versa?

## Stretch goals (optional)

- **Extend the glossary to a "compare across providers"
  section.** e.g., a table of "what Anthropic, OpenAI, and
  Google each call their reasoning-tuned models" or "which
  providers support the OpenAI chat-completions schema
  vs. their own".
- **Write a two-page briefing** for a non-technical
  stakeholder ("what is an LLM, what can it do, what does
  it cost, and what are its failure modes") based on your
  glossary. This is a real deliverable most engineers end
  up writing eventually.
- **RAG sketch.** Without building it, describe in one page
  the shape of a production RAG system for your company's
  internal wiki: ingestion, chunking, embedding model
  choice, retriever, reranker, prompt template, evaluation.
  Cite the LLM Engineering references you would consult.
- **Cost calculator.** In the playground, extend `NOTES.md`
  with a spreadsheet or script that estimates monthly cost
  given a queries-per-minute number and prompt/completion
  token estimates for the three model tiers you looked at.
- **Read the safety card / responsible-scaling document**
  for one of the providers (Anthropic's Responsible Scaling
  Policy, OpenAI's system cards, Meta's Llama 3 safety
  section). Summarise in half a page what the provider says
  they will and will not do; this frames later work in the
  LLM Engineering track.
- **Try one call each against a base model and an
  instruction-tuned model** of the same family (e.g., Llama
  3.1 8B and Llama-3.1-8B-Instruct via Ollama). Contrast the
  outputs on the same prompt. This makes the "base model vs.
  chat model" distinction concrete.

## What to hand in

```
exercise-04/
├── README.md                   # entry point, one-page summary, reflection answers
├── glossary.md                 # every term from chapter 10, in your words, cited
├── delegation_map.md           # decision tree + scenarios + boundaries
├── model_cards.md              # three completed card summaries + comparison
└── prompt_playground/
    ├── call.py                 # the tiny API script
    ├── NOTES.md                # temperature sweep, token counts, cost, reflection
    └── requirements.txt        # or pyproject.toml — one dep, the SDK you chose
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.

<!-- needs-research: the model list in Part C names 2024–2026-era model
families; on each authoring refresh, swap in the current
generation of each vendor's flagship and open-weight releases
(Llama 3.x → 4.x, Claude 3.x/4.x → next major, etc.). -->
