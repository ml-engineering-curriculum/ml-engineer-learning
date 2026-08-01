# 10 — Transformer LLMs and when to delegate

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 07 (Transformer encoder).

## Motivation

Chapter 07 built a Transformer encoder. Every modern **large
language model (LLM)** — GPT-4/5, Claude, Gemini, Llama,
Mistral, DeepSpeed models, and their open-source relatives —
is a Transformer *decoder* trained on internet-scale text.
The architectural distance between what you built and what
runs a chatbot is smaller than the size difference suggests:
same attention, same feed-forward, same residual + LayerNorm.
The difference is scale, data, training recipe, and post-
training.

This chapter's goals are narrower than the others:

1. Give you the **vocabulary** to read an LLM paper, a
   model card, or a deployment guide without having to
   pause and Google every acronym.
2. Give you the **delegation map** — what belongs in the
   LLM Engineering track, what belongs in MLOps, what
   belongs in Applied Vision, what belongs in Research.

We do not train an LLM in this module. We do not fine-tune
one in this module. The scaffolding for both is here so that
when a task lands, you know where to send it or where to
learn it.

## From encoder to decoder in one page

An **encoder** attends bidirectionally: every position can
see every other. Good for classification, embedding, and
representation.

A **decoder** attends causally: position `i` can only see
positions `≤ i`. Good for generation — at inference the
model predicts one token at a time and each prediction only
depends on the past.

The mechanics change in exactly one place: the attention
mask. `F.scaled_dot_product_attention(q, k, v,
is_causal=True)` or an upper-triangular `-inf` mask on the
attention scores.

The **encoder–decoder** architecture (the original 2017
paper; T5; Whisper) combines both: an encoder builds a
representation of an input sequence, and a decoder attends
to that representation while generating.

The current landscape:

- **Decoder-only** (GPT-family, Llama, Mistral, Claude,
  Gemini) — dominates general-purpose text generation.
- **Encoder-only** (BERT, RoBERTa) — best for classification
  and embedding tasks; smaller and cheaper.
- **Encoder–decoder** (T5, BART, Whisper, Flan-T5) — best
  for sequence-to-sequence tasks where input and output are
  distinct (translation, summarisation, speech recognition).

Choosing between them is task-driven: "generate free text
conditioned on a prompt" → decoder. "Predict a class from a
document" → encoder. "Turn an input sequence into an output
sequence with a hard boundary between them" → encoder–
decoder.

## The vocabulary, alphabetically-ish

The terms you will hear repeatedly. Each in one paragraph.

**Token.** The atomic unit the model processes. Not a word,
not a character — a subword produced by a **tokenizer**
(BPE, WordPiece, SentencePiece). "Hello, world" might be
`["Hello", ",", " world"]` — three tokens. English text
averages roughly 4 characters per token; other languages
vary. A tokenizer is trained on a corpus; the resulting
**vocabulary** is a fixed table of a few tens of thousands
of tokens.

**Context length / context window.** The maximum sequence
length the model can attend over in one forward pass. A model
with 8k context can be given 8,192 tokens of input;
everything beyond that is either truncated or requires
special handling (RAG, long-context extensions like RoPE
scaling, or paged attention). Modern models range from 4k
(older) to 200k–2M (Claude, Gemini) — with real accuracy
trade-offs at the long end.

**Parameters.** The learnable weights, counted in millions
(M) or billions (B). "Llama-3.1-8B" has ~8 billion. Parameter
count is a rough proxy for capability, memory footprint at
inference (2× parameter count in bytes for bf16), and
training cost.

**Pre-training.** Training the base model from random weights
on a massive corpus (trillions of tokens) with a
self-supervised objective — usually next-token prediction
for decoder-only, masked language modelling for BERT-style
encoders. Takes millions of dollars and thousands of GPUs.
The result is a **base model** that can predict the next
token but is not aligned to any specific use.

**Post-training.** The recipe that turns a base model into
an assistant. Usually two stages:

- **Supervised fine-tuning (SFT)** on curated
  instruction-response pairs.
- **Preference optimisation** — RLHF (reinforcement
  learning from human feedback), DPO (direct preference
  optimisation), and newer variants — that teach the model
  to prefer responses humans rate higher.

Post-training is where most of the "assistant" behaviour
comes from. Fine-tuning a well-post-trained model on your
task is qualitatively different from fine-tuning a base
model.

**Fine-tuning.** Taking a pre-trained (or post-trained)
model and continuing to train it on your task with your
data. Full fine-tuning updates all parameters and requires
the full memory footprint. Parameter-efficient variants
(next paragraph) are what most teams actually use.

**Parameter-efficient fine-tuning (PEFT).** A family of
methods that adapt an LLM by training a small number of
extra parameters while leaving the base model frozen. The
most common is **LoRA** (Low-Rank Adaptation): insert small
trainable rank-`r` matrices into each layer; train those
only. Storage-cheap (adapters are megabytes, not
gigabytes), fast to train, easily swapped between tasks.
**QLoRA** runs the frozen base model in 4-bit precision;
lets you fine-tune 65B-parameter models on a single 24 GB
GPU.

**Prompt / system prompt / few-shot.** The text handed to
the model at inference. A **system prompt** sets persona
and constraints; the user turn asks the actual question;
**few-shot examples** show the model 1–5 worked examples of
the task in the prompt before asking. In-context learning:
the model adapts its behaviour from prompt alone, without
weight updates.

**Sampling / decoding.** How the model turns per-token
probability distributions into text. Common knobs:
**temperature** (0 = deterministic argmax; 1 = raw
distribution; > 1 = flatter), **top-k** (sample from the k
most likely tokens), **top-p / nucleus** (sample from the
smallest set whose cumulative probability ≥ p), **beam
search** (multiple hypotheses; more common in translation
than chat). Repetition penalties, min-p, and various
constrained-decoding methods layer on top.

**Logits, logprobs.** The raw pre-softmax output; the log of
the softmax. Some APIs return them so you can compute
custom metrics or do constrained generation.

**Instruction-tuned / chat model.** A model that has been
post-trained to follow chat-style instructions. Distinct
from a **base model**, which will happily continue a
document but does not know it is a chat assistant.

**Prompt engineering.** The practice of finding phrasings,
examples, and structures that make an LLM produce useful
output. Reasonable first stop for many tasks; ceilings
exist. See the RAG and fine-tuning options for what to
reach for next.

**RAG (retrieval-augmented generation).** The model doesn't
memorise everything. RAG retrieves relevant documents from a
vector database (typically indexed by an embedding model)
at query time and injects them into the prompt. The
canonical way to give an LLM access to private / current /
long-tail knowledge without fine-tuning.

**Embedding model.** A (usually much smaller) Transformer
that turns text into a fixed-dimensional vector, such that
semantically similar texts have close vectors. The workhorse
for RAG retrieval, semantic search, clustering, and
similarity.

**Vector database / vector store.** A database indexed on
approximate-nearest-neighbour search over embedding vectors.
FAISS, pgvector (Postgres extension), Chroma, Weaviate,
Pinecone, Qdrant. All solve the same "find the k nearest
vectors to this query" problem with different operational
trade-offs.

**Function calling / tool use / agents.** Give the model a
list of functions (in JSON) it can request the client to
call. The client runs the function; the result goes back
into the conversation. An **agent** is a loop where the
model calls tools, sees the results, and decides what to do
next. Common tools: web search, code execution,
database queries.

**KV cache.** During generation, each new token requires
computing attention over all previous tokens. The keys and
values from previous tokens are cached to avoid recomputing
them. This dominates the memory footprint of inference;
optimising KV cache use (paged attention, prefix caching)
is a major topic in serving.

**Quantization.** Reducing the precision of weights (and
sometimes activations) below fp16/bf16 for smaller memory
footprint and faster inference. **INT8** is standard; **INT4**
is common via GPTQ, AWQ, or bitsandbytes. Modest accuracy
loss; often 2–4× throughput on the same hardware.

**FLOPs.** Floating-point operations. Used to compare model
compute. A 7B model doing one forward pass does about
`2 × 7 × 10⁹` FLOPs per token. Training FLOPs (parameters ×
tokens) is the relevant scale metric for pre-training
budgets.

**MoE (Mixture of Experts).** A sparsely activated model
where each token is routed to a subset of experts (small
MLPs). Total parameter count is high; per-token compute is
much lower. Mixtral 8x7B is 47B total, ~13B active per
token.

**Prompt tokens vs. completion tokens.** The two halves of
an API's billing model. Prompt tokens (input) are usually
cheaper than completion tokens (output). Cache-hit prompt
tokens are cheaper still (some providers).

**Prompt caching / prefix caching.** Server-side reuse of KV
caches for common prompt prefixes (system prompts, few-shot
examples, retrieved documents). Materially cheaper and
faster for repeated queries with similar prefixes.

**Context stuffing / context distillation.** Approaches for
using very long contexts efficiently — either by cramming
everything relevant into the prompt (stuffing) or by
compressing / summarising it first.

**Distillation.** Training a smaller model to mimic a larger
one's outputs. Cheaper inference; usually some accuracy
loss.

**Evaluation:** MMLU, GSM8K, HumanEval, HellaSwag,
BIG-Bench, MT-Bench, chatbot arena — the well-known
benchmarks. Real production evaluation is mostly custom,
task-specific, and usually a mix of automated metrics and
human raters. `mod-107` (model evaluation) covers the
general practice; LLM-specific eval methodology is in the
LLM Engineering track.

**Hallucination.** The model produces confident, wrong
output. Not a solved problem. RAG reduces it for knowledge-
intensive tasks; system prompts asking for citations help;
model choice matters (larger models hallucinate less on
average). Domain-critical applications need retrieval,
verification, or human review.

## The delegation map

This module gives you the vocabulary and the encoder
mechanics. Everything below is a specialist track. When a
task lands, this map tells you where it belongs.

### Stays in `mod-105`

- Reading a Transformer paper and understanding the
  architecture.
- Building a small encoder for a bespoke classification or
  embedding task on a mid-sized dataset.
- Fine-tuning a small pre-trained encoder (BERT-family) on
  a classification task — cite chapter 07's worked example
  and the resources chapter.
- Using an LLM via API for prototypes (no
  fine-tuning, no self-hosted serving).

### Delegates to the **LLM Engineering** track

- Training an LLM (base or continued pre-training).
- Fine-tuning a large chat model, including LoRA / QLoRA
  workflows.
- Building a production RAG pipeline (retrieval, embedding,
  reranking, chunking strategies, evaluation).
- Agent frameworks (tool use, planning loops, evaluation).
- Prompt engineering at production scale.
- LLM evaluation methodology beyond simple benchmarks.
- Self-hosted LLM serving (vLLM, SGLang, TGI, TensorRT-LLM),
  including quantization workflows.
- Safety alignment (RLHF / DPO), red-teaming, guardrails.

### Delegates to the **MLOps / Deployment** track (or `mod-108`)

- Model serving at scale (throughput, latency, cost).
- Batching strategies (dynamic batching, continuous batching).
- Multi-GPU inference (tensor parallelism, pipeline
  parallelism).
- KV-cache management in a shared serving environment.
- Monitoring token usage, cost, and latency per request.
- Autoscaling GPU pools.

### Delegates to the **ML Infra / Distributed Training** track

- Training on multiple GPUs (DDP, FSDP) and multiple nodes.
- ZeRO / DeepSpeed / Megatron / TP / PP configurations for
  training models above single-GPU memory.
- Storage and dataset streaming at PB scale.
- Cluster orchestration for training jobs.

### Delegates to the **Applied Vision** track

- Object detection, segmentation (Mask R-CNN, DETR, SAM).
- Vision Transformers (ViT, DINO, CLIP) beyond fine-tuning.
- Diffusion models (Stable Diffusion, Imagen).
- Multimodal models (CLIP, LLaVA, GPT-4V-style vision-
  language systems) — mostly. The vision half sits in this
  track; the LLM half in LLM Engineering.

### Delegates to the **Speech / Audio** track

- Whisper-style speech recognition (encoder–decoder
  Transformer applied to spectrograms).
- Speech synthesis (VITS, XTTS, MusicGen).
- Speaker diarisation.

### Delegates to **ML Research**

- Novel architectures, novel optimisers, novel training
  recipes.
- Interpretability research (mechanistic interpretability,
  probing).
- Scaling laws studies.

The pattern: `mod-105` is a fluency floor. If a task requires
knowing what these terms mean and how the pieces roughly
fit, `mod-105` was enough. If it requires *doing* one of the
specialist things, delegate.

## A decision map for "should I use an LLM here?"

Not every problem is an LLM problem. A short decision map.

1. **Is the task structured** (classification, regression,
   ranking, extraction into a fixed schema)? If yes, and the
   data is tabular or a modest amount of text, `mod-104` or a
   small encoder (chapter 07) is usually cheaper and better.
2. **Is the task unstructured generation** (writing an
   email, summarising a document, answering an open
   question)? If yes, an LLM API is the first move.
3. **Do you need the model to know things about your
   private data?** If yes, RAG. Do not fine-tune first —
   RAG solves this cheaper and with less risk.
4. **Do you need the model to *behave* differently
   (persona, format, tone) or to solve tasks the base
   model does poorly?** Prompt engineering, then few-shot,
   then fine-tuning if those don't suffice.
5. **Is latency, cost, or privacy prohibitive with an API?**
   Self-hosted (small model + LoRA + serving stack) becomes
   worth the engineering. Chapter 09 and the LLM
   Engineering track cover the mechanics.
6. **Is the task worth an LLM at all?** Regex, a lookup
   table, or a rule-based system beats a $0.02-per-call LLM
   call almost every time when the task is structured
   enough for one.

The mistake to avoid: reflexively reaching for an LLM
because the input is text. Text classification of tweets is
still a text classification problem; a fine-tuned
DistilBERT or a well-tuned logistic regression on TF-IDF
will beat GPT-5 in accuracy per dollar, latency, and
determinism, most of the time.

## Reading an LLM paper productively

The typical modern paper (Llama, Mistral, DeepSeek, Qwen)
follows a template. What to read first:

1. **The model card / recipe table.** Parameters, layers,
   heads, `d_model`, context length, vocabulary size,
   activation, positional encoding (RoPE / ALiBi), tokenizer.
   This is 80% of the "what is it".
2. **Data section.** Corpus size in tokens, mixture, filter
   pipeline. Data explains a lot of the capability
   differences between models of similar size.
3. **Training section.** Optimiser (AdamW, almost always),
   LR schedule, batch size, warmup, gradient clipping. The
   knobs from chapters 08 and 09, applied at scale.
4. **Post-training.** SFT dataset, preference method (DPO /
   RLHF), safety tuning.
5. **Evaluation.** The benchmarks used, and the honest
   footnotes on what the model does badly.

If a paper omits one of these entirely, you are looking at a
release announcement, not a paper. Extract what you can and
move on.

## What "working familiarity" means for this module

You do not need to be able to train an LLM after this
chapter. You *do* need to be able to:

- Read an LLM model card and know what every field means.
- Pick a base model for a downstream task and know how
  much VRAM you will need to fine-tune it.
- Distinguish "prompt this via API" from "fine-tune this"
  from "train from scratch" and pick the cheapest option
  that works.
- Recognise when a problem is not an LLM problem.
- Point at the correct specialist track for anything beyond
  the four bullets above.

Exercise-04 is a short written deliverable that operationalises
this. When you can produce that document, this module's LLM
objective is complete.

## Summary

- Every modern LLM is a Transformer decoder trained on
  internet-scale text. The architecture is what chapter 07
  built, with a causal mask and 100–10,000× more parameters.
- The vocabulary in this chapter — tokens, context, PEFT,
  RAG, KV cache, quantization, MoE — is the reading
  fluency you need to follow the field.
- This module does not train, fine-tune, serve, or evaluate
  LLMs at production scale. The LLM Engineering, MLOps,
  Vision, and Speech tracks do.
- Not every text problem is an LLM problem. Match the tool
  to the task.

## Where this goes next

Exercises 01–03 in this module exercise the loop, CNN, and
Transformer encoder you have now seen chapters for. Exercise
04 turns this chapter's vocabulary and delegation map into a
personal reference you will consult for years. Then
`mod-106` opens the experiment-tracking and reproducibility
practices that make everything you just built into an
engineering artefact rather than a notebook.
