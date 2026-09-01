# exercise-03 — Small Transformer encoder

> **Estimated effort:** 4 hours.
> **Companion chapters:** [04 — Datasets, DataLoaders, collate](../04-datasets-dataloaders-and-collate.md), [07 — A small Transformer encoder](../07-a-small-transformer-encoder.md), [08 — Regularisation and LR schedules](../08-regularisation-and-lr-schedules.md), [09 — Mixed precision and performance](../09-mixed-precision-and-training-performance.md).

## Objective

Build a **small Transformer encoder from scratch** and train
it on a real text classification task. Then compare it to two
baselines you will build alongside it:

1. **From-scratch encoder** — the model from chapter 07,
   assembled by hand from `nn.Linear`, `nn.LayerNorm`,
   `nn.Embedding`, and either your own multi-head attention
   or `F.scaled_dot_product_attention`.
2. **PyTorch built-in** — the same architecture, but
   substituting `nn.TransformerEncoderLayer` /
   `nn.TransformerEncoder`. Verifies your from-scratch build
   matches a well-tested reference to within a reasonable
   accuracy margin.
3. **Fine-tuned pre-trained model** — a small BERT-family
   checkpoint (DistilBERT or similar) via Hugging Face
   `transformers`, fine-tuned on the same split.

The training loop from exercise-01 is reused. What is new is
the model, the tokenizer + padded-collate pipeline (chapter
04), and the Transformer-specific training tricks from
chapters 07–08 (warmup, gradient clipping, initialisation).

By the end you should have:

- Three trained models on the same train/val/test split.
- A `REPORT.md` comparing accuracy, parameter count, and
  training wall-clock time across the three.
- A concrete, first-hand answer to "how much of modern NLP
  performance is architecture vs. pre-training?"

## Prerequisites

- Chapters 04, 07, 08, and 09.
- Exercise-01 completed (you will reuse its training loop
  and testing patterns).
- Python 3.11+, PyTorch ≥ 2.1 (needed for
  `F.scaled_dot_product_attention`),
  `tokenizers` (Hugging Face), `transformers`, `datasets`,
  `pytest`, `matplotlib`.
- A GPU is helpful but not required. On CPU, budget
  ~5 minutes per epoch for the from-scratch model on AG News
  at `d_model=128, n_layers=4`, and roughly 10× that for the
  DistilBERT fine-tune (halve the epochs and note it in
  `REPORT.md` if CPU-only).

## Dataset

Pick **one** of the following English text classification
datasets. All are small, well-known, and load in one line via
Hugging Face `datasets`.

- **AG News.** 4-class news topic classification, 120k train
  / 7.6k test, single-sentence-ish. The default worked example
  in chapter 07. `datasets.load_dataset("ag_news")`.
- **IMDb.** Binary sentiment, 25k train / 25k test, long
  reviews (truncate). `datasets.load_dataset("imdb")`.
- **DBpedia-14.** 14-class Wikipedia article classification,
  560k train / 70k test. Larger; the model needs to
  discriminate more classes.
  `datasets.load_dataset("dbpedia_14")`.
- **TREC.** 6-class question classification, ~5.5k train
  total — very small; the from-scratch model will underfit
  quickly, which makes the fine-tune comparison stark.
  `datasets.load_dataset("trec")`.

Whichever you choose, record it in `REPORT.md` with class
distribution and a couple of example rows.

**Do not** use a dataset that requires you to train a
tokenizer larger than a few tens of thousands of tokens — the
scope of this exercise is one afternoon, not one week.

## Problem statement

### Part A — Tokenization and collate (≈ 40 min)

In `data.py`, build the text pipeline.

Two acceptable tokenizer paths — pick one and note it in
`REPORT.md`:

- **Path 1: train a BPE tokenizer.** Use
  `tokenizers.Tokenizer` with a `BPE` model and a
  `ByteLevelPreTokenizer`. Train on the training split only
  (no leakage). Vocab size 8k–32k. Serialise to disk so the
  test suite can load it.
- **Path 2: reuse a pretrained tokenizer's vocab.** e.g.
  `AutoTokenizer.from_pretrained("distilbert-base-uncased")`,
  but *tokenize only* — do not use its model. Simpler for a
  first pass. Makes the "same tokenization" comparison
  against the DistilBERT baseline cleaner.

Write:

```python
def build_tokenizer(train_texts: list[str], vocab_size: int = 16000): ...
def encode(text: str, tokenizer, max_len: int = 128) -> list[int]: ...

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 128): ...
    def __getitem__(self, idx) -> dict: ...

def collate_pad(batch, pad_id: int = 0) -> dict: ...
def build_loaders(...): ...
```

Requirements:

1. `__getitem__` returns a dict with at least `ids` (Long
   tensor of token ids) and `label` (Long tensor scalar).
2. `collate_pad` pads each batch's sequences to the length
   of the **longest sequence in that batch** (not to
   `max_len` globally). Return a dict with:
   - `ids`: `(B, L_max)` Long
   - `padding_mask`: `(B, L_max)` Bool, `True` where
     padded (see chapter 07 for the convention)
   - `label`: `(B,)` Long
3. `pad_id=0` and reserve index 0 for `[PAD]` when training
   your own tokenizer. In `nn.Embedding`, set
   `padding_idx=0`.
4. Truncate on the right when a document exceeds `max_len`.
5. Split first, tokenize second. If you split *after*
   tokenizing on the whole corpus, statistics leak. Keep the
   tokenizer trained on train only.

### Part B — From-scratch encoder (≈ 60 min)

In `models/scratch.py`, implement the encoder from chapter 07.
You may lift chapter 07's snippets verbatim as a starting
point; the exercise is to make them work end to end.

Requirements:

1. `MultiHeadSelfAttention` module implemented by hand
   (single fused QKV projection, then chunk / reshape). May
   internally call `F.scaled_dot_product_attention` for the
   attention math *or* implement `softmax(QKᵀ/√d)V` by hand
   — pick one and note the choice in `REPORT.md`. The
   pedagogical point of implementing the softmax yourself is
   real; make sure you have done it *at least once* somewhere
   in this exercise.
2. `TransformerBlock` uses **pre-LN** (`x + Sublayer(LN(x))`),
   two residual connections, `GELU` in the feed-forward,
   `d_ff = 4 * d_model`.
3. `Embedding` combines a token embedding
   (`padding_idx=pad_id`) and a **learned** positional
   embedding, added.
4. `TransformerEncoder` stacks `n_layers` blocks, applies a
   final `LayerNorm`, **mean-pools over non-padded
   positions**, and projects to `n_classes` with a `Linear`.
5. Weight init as in chapter 07's `_init_weights` — normal
   with `std=0.02` for `Linear` and `Embedding`, ones/zeros
   for `LayerNorm`.
6. `padding_mask` is threaded through attention (score
   masking before softmax) *and* through the pooling.
7. Default hyperparameters, unless you have a reason:
   `d_model=128, n_heads=4, d_ff=512, n_layers=4,
   max_len=128, dropout=0.1`. Report the parameter count with
   `count_parameters(model)`; expect ~1M for AG News with an
   8k–16k vocab.

### Part C — PyTorch built-in reference (≈ 20 min)

In `models/reference.py`, build the same architecture using
`nn.TransformerEncoderLayer` / `nn.TransformerEncoder`. Wrap
the same embedding, pooling, and classification head from
Part B around it so the only difference is the block
implementation.

Requirements:

1. Use `nn.TransformerEncoderLayer(..., batch_first=True,
   norm_first=True, activation="gelu")` to match Part B's
   pre-LN GELU convention.
2. **Match the mask convention.** PyTorch's built-in
   attention uses `src_key_padding_mask` with `True` = padded
   (the opposite of some other libraries). Verify by running
   a small forward and checking the model does not "see"
   the padded positions.
3. Same hyperparameters as Part B (`d_model`, `n_heads`,
   `d_ff`, `n_layers`, `dropout`).

### Part D — Pre-trained fine-tune (≈ 30 min)

In `models/pretrained.py`, wire up a small pre-trained
checkpoint from Hugging Face.

Requirements:

1. Use `AutoTokenizer` + `AutoModelForSequenceClassification`
   with a **small** checkpoint — `distilbert-base-uncased`
   is the canonical choice (66M parameters). `bert-base` and
   `roberta-base` also work if you have GPU headroom.
2. Set `num_labels=n_classes` on `from_pretrained`.
3. Use the checkpoint's own tokenizer for its own inputs. Do
   not re-use your BPE tokenizer here — vocab mismatch
   silently breaks everything.
4. **You may reuse exercise-01's training loop unchanged** —
   pass the batch through the model as
   `model(input_ids=..., attention_mask=..., labels=...)` and
   read `outputs.loss` and `outputs.logits`. This is one of
   the moments where a well-designed loop earns its keep.
5. **Use a lower learning rate:** `2e-5` to `5e-5` for
   fine-tuning; `weight_decay=0.01`.

### Part E — Training with Transformer-specific tricks (≈ 40 min)

Reuse exercise-01's `train.py` loop. Extend it (or subclass /
wrap it — your call) to add three things every real
Transformer training needs:

1. **Warmup + decay schedule.** Either `OneCycleLR` with
   `pct_start=0.1`, or a hand-rolled linear warmup for the
   first ~10% of steps then cosine decay to zero. Log the LR
   per step and confirm on the first few hundred steps that
   it looks right. See chapter 08.
2. **Gradient clipping.** `torch.nn.utils.clip_grad_norm_(
   model.parameters(), max_norm=1.0)` after `.backward()`,
   before `optimizer.step()`. Log the pre-clip norm every
   `log_every` batches — it should not routinely exceed 10;
   spikes to 100 mean something is wrong.
3. **AdamW** with `weight_decay=0.01`. Not vanilla `Adam`.

Suggested training budget per model (halve on CPU):

| Model | Peak LR | Epochs |
|---|---|---|
| From-scratch encoder | `3e-4` | 5–10 |
| PyTorch reference | `3e-4` | 5–10 |
| DistilBERT fine-tune | `3e-5` | 2–3 |

Save best-val-loss checkpoints per model.

### Part F — Evaluation and comparison (≈ 30 min)

Once the three models are trained, produce for each:

1. **Test accuracy** on the held-out test split.
2. **Macro-F1** (matters if the classes are imbalanced).
3. **Confusion matrix** as `confusion_<model>.png`.
4. **Per-class accuracy** row-by-row.
5. **Parameter count.** Reported as
   `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
6. **Wall-clock training time** on your hardware. Note the
   hardware in `REPORT.md`.
7. **Ten interesting predictions** — five where every model
   agrees and gets it right, five where they disagree.
   Discuss the disagreements briefly.

Optional but valuable: **attention visualisation** for a few
examples of the from-scratch model. Take the last block's
attention weights (average over heads) for one sequence and
render as a heatmap. Save under `plots/`.

### Part G — Tests (≈ 20 min)

In `tests/`, cover at minimum:

1. **`test_scratch_matches_reference_output_shape`.** Feed a
   batch of `(4, 32)` token ids through both models; assert
   both return `(4, n_classes)`.
2. **`test_padding_mask_zeros_padded_positions`.** Construct
   a batch where two sequences differ only in the number of
   pad tokens appended. Assert the model's output for the
   unpadded content is (approximately) invariant to the
   number of trailing pads. This is the mask-correctness
   smoke test; without it, silent padding leakage is
   hard to catch.
3. **`test_causal_vs_bidirectional`.** Feed the same batch
   with and without a causal mask (either your model's or
   PyTorch's). Assert the outputs differ — this proves the
   mask is actually being used.
4. **`test_overfit_tiny_batch`.** Take 16 short examples,
   train the from-scratch encoder for 200 steps; assert loss
   drops to ≤ 1% of initial. Use a small `max_len` (≤ 32)
   to keep runtime tolerable.
5. **`test_positional_matters`.** Take one example; feed it
   forward. Shuffle its (non-pad) tokens; feed forward
   again. Assert the outputs differ. Without positional
   embeddings this test fails, which is the whole point.
6. **`test_grad_clip_norm_reports_correctly`.** Verify the
   value returned by `clip_grad_norm_` is finite and
   non-zero after one training step.
7. **`test_warmup_lr_ramp`.** Instantiate the scheduler.
   Step it `warmup_steps` times; assert the LR is at (or
   near) `peak_lr` at that point and ≪ `peak_lr` at step 1.

Keep the test suite under a minute total. `pytest -q` should
be green.

## Starter guidance

- **Start with the tiny-batch overfit test on the
  from-scratch model.** If it cannot overfit 16 examples to
  near-zero loss with the "correct" hyperparameters
  (`d_model=64, n_heads=4, n_layers=2, lr=3e-4`), the model
  or the mask is wrong. Fix that before you touch a real
  dataset.
- **Compare your from-scratch attention to
  `F.scaled_dot_product_attention` on the same inputs
  once.** Same `Q`, `K`, `V`; same output to within
  `atol=1e-5`. This one test would have saved everyone at
  least a night's debugging.
- **Do the DistilBERT baseline first.** It works out of the
  box in ~30 minutes and gives you a north-star number to
  aim at with the from-scratch model. Working backwards
  from a known-good pipeline is faster than trying to
  bootstrap two things at once.
- **Use `nn.Embedding(padding_idx=pad_id)` and keep
  `pad_id=0`.** Cheap correctness that removes an entire
  class of bugs where pad tokens' embeddings drift.
- **Log the four numbers every 50 batches:** running train
  loss, current LR, gradient norm pre-clip, step wall-clock.
  A plot of any of them tells you when something has gone
  sideways.
- **Do not train from scratch on a serious document
  classification task expecting to beat a fine-tuned BERT.**
  You won't. The comparison *is* the point.
- **`nn.MultiheadAttention` is a trap on first read.** Its
  mask conventions are inverted vs. `key_padding_mask` in
  `F.scaled_dot_product_attention`. If you use it, read the
  docs carefully — chapter 07's pitfall #8 is real.

## Acceptance criteria

- All three models train end to end and produce
  checkpoints.
- The from-scratch encoder and the `nn.TransformerEncoder`
  reference reach **within 3 percentage points** of each
  other on test accuracy at matched hyperparameters. If they
  do not, the from-scratch implementation has a bug — track
  it down before submitting.
- The DistilBERT fine-tune reaches the reasonable
  validation-accuracy floor for the chosen dataset (below).
- Overfit-tiny-batch test brings loss to ≤ 1% of initial.
- All listed tests pass. `pytest -q` is green.
- Warmup + gradient clipping are wired and *tested* (Part G
  tests 6 and 7).
- `REPORT.md` records: dataset, tokenizer path, hyper-
  parameters, parameter counts, test accuracy / macro-F1,
  confusion matrices, wall-clock times, the ten predictions
  discussion, and the reflection answers.

**Reasonable test-accuracy floors** for the suggested
datasets. Rough, not a leaderboard:

| Dataset | Small from-scratch encoder | DistilBERT fine-tune |
|---|---|---|
| AG News (4-class) | ≥ 0.88 | ≥ 0.93 |
| IMDb (binary) | ≥ 0.85 | ≥ 0.90 |
| DBpedia-14 | ≥ 0.95 | ≥ 0.98 |
| TREC (6-class) | ≥ 0.75 | ≥ 0.90 |

If the from-scratch model is well below the floor: first
check the mask, then the LR, then the warmup, then the
initialisation. Debug order matters — chapter 07's "Debugging
a Transformer that won't train" section is the right
reference.

## Reflection questions (answer in `REPORT.md`)

- The from-scratch encoder and the `nn.TransformerEncoder`
  reference should be within a few points of each other. If
  they weren't on your first run, what was the bug and how
  did you find it?
- How large was the gap between the from-scratch encoder
  and the DistilBERT fine-tune? How much of that gap do you
  attribute to *architecture* (DistilBERT is deeper /
  wider), how much to *training data* (millions of tokens
  of English pre-training), and how much to *tokenizer* /
  *initialisation*?
- Read the LR / gradient-norm log. Was there a moment where
  the loss almost diverged? What did the LR schedule or the
  clipping do at that step to save it?
- If your dataset were 100× smaller (say 1,000 examples
  instead of 100,000), which of the three models would you
  reach for first, and why? (Chapter 10's decision map is
  the right frame.)

## Stretch goals (optional)

- **Try `[CLS]` pooling** in a second variant of the
  from-scratch model. Prepend a learned `[CLS]` token id;
  use its final-layer representation instead of mean
  pooling. Compare test accuracy.
- **Swap learned positional embeddings for sinusoidal**
  (chapter 07). Same architecture, same hyperparameters;
  measure the difference on a very long-input dataset.
- **Rotary positional embeddings (RoPE).** Implement the
  half-dimension rotation on `q` and `k` inside your
  attention module. Verify against a reference
  implementation. This is a serious hour of work; skip if
  you are running short.
- **Ablate warmup, clipping, and LayerNorm placement.** Run
  four configurations of the from-scratch model:
  post-LN + no warmup + no clip; post-LN + warmup + no clip;
  pre-LN + no warmup + clip; pre-LN + warmup + clip. Plot
  the training curves and discuss.
- **Compare to a linear model.** Train a
  `sklearn.linear_model.LogisticRegression` on TF-IDF
  features for the same task. Report accuracy and training
  time. Chapter 10's "not every text problem is an LLM
  problem" is worth verifying first-hand.
- **`torch.compile(model)`** on the from-scratch model.
  Measure the speedup on the training loop; note in
  `REPORT.md`. Requires PyTorch 2.x.
- **Mixed-precision training** (chapter 09). Wrap the forward
  in `torch.autocast(device_type="cuda",
  dtype=torch.bfloat16)`. Measure speedup and memory
  savings.

## What to hand in

```
exercise-03/
├── data.py                     # tokenizer, dataset, collate_pad, loaders
├── models/
│   ├── __init__.py
│   ├── scratch.py              # MultiHeadSelfAttention, TransformerBlock, TransformerEncoder
│   ├── reference.py            # same, using nn.TransformerEncoder
│   └── pretrained.py           # DistilBERT wrapper
├── train.py                    # extends exercise-01's loop with warmup + clipping
├── evaluate.py                 # accuracy, macro-F1, confusion, ten predictions
├── tests/
│   ├── test_scratch_vs_reference.py
│   ├── test_padding_and_positional.py
│   ├── test_overfit.py
│   └── test_schedule_and_clipping.py
├── plots/
│   ├── loss_curve_scratch.png
│   ├── loss_curve_reference.png
│   ├── loss_curve_distilbert.png
│   ├── confusion_scratch.png
│   ├── confusion_reference.png
│   └── confusion_distilbert.png
├── history_scratch.json
├── history_reference.json
├── history_distilbert.json
├── tokenizer/                  # serialised tokenizer artefacts
├── pyproject.toml
└── REPORT.md
```

Solutions live in the paired `ml-engineer-solutions` repo —
do not peek until you have submitted yours.
