# 07 — A small Transformer encoder

> **Reading time:** ~45 minutes.
> **Prerequisite chapters:** 02–06.

## Motivation

Almost every impressive model of the last several years is a
Transformer or a variant. The original 2017 paper *Attention
Is All You Need* (Vaswani et al.) introduced the
architecture for machine translation; BERT applied the
encoder half to language understanding; GPT applied the
decoder half to language generation; Vision Transformer (ViT)
applied it to images; Whisper applied it to speech. Whatever
your next model is, it is more likely than not a variant of
this one.

This chapter builds a **small Transformer encoder from
scratch**, one component at a time, and trains it on a real
task. The encoder is the more useful half to understand
first — it handles classification, embedding, and
representation-learning tasks, and the decoder is a small
extension of the same primitives (which chapter 10 sketches
without implementing).

Two goals for this chapter:

1. You can read a Transformer implementation and know what
   each `nn.Module` is doing.
2. When you reach for `nn.TransformerEncoder` from
   `torch.nn`, or `AutoModel` from Hugging Face, you know
   what it is doing under the hood.

## The core idea: attention

A convolution mixes information across a **local** spatial
window (`k×k` pixels). A recurrent net mixes information
across a **sequential** history (one token at a time).
Attention mixes information across **all positions
simultaneously, with learnable weights** — each output
position is a weighted average of all input positions,
where the weights are computed from the content.

That last part is the trick. In a convolution, the weights
are the same regardless of the input. In attention, the
weights are computed from the input itself — position `i`
looks at position `j` more strongly if the *content* at `i`
and `j` says they should relate.

The scaled dot-product attention formula:

```
attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

Where:

- **Q** (queries), **K** (keys), **V** (values) are each
  `(B, L, d)` tensors — one vector per input token.
- `Q Kᵀ` gives an `(L, L)` matrix of pairwise scores — how
  much position `i` should attend to position `j`.
- Divide by `√d_k` to keep the softmax from saturating for
  large `d`.
- `softmax` (row-wise) turns scores into attention weights
  that sum to 1.
- Multiply by `V` to get, for each output position, a
  weighted average of value vectors.

Everything else in this chapter is variations on that
formula.

## Multi-head attention

Instead of one attention operation, run several ("heads") in
parallel, each with its own smaller `Q, K, V` projections,
then concatenate. Each head can specialise on a different
kind of relationship (positional, syntactic, semantic).

The math with `h` heads and full model dimension `d_model`:

- Project `x` (shape `(B, L, d_model)`) into `Q, K, V`, each
  `(B, L, d_model)`.
- Reshape each into `(B, h, L, d_head)` where
  `d_head = d_model / h`.
- Run attention independently per head.
- Concatenate back to `(B, L, d_model)` and project.

Implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv  = nn.Linear(d_model, 3 * d_model)      # one big projection
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        B, L, _ = x.shape
        qkv = self.qkv(x)                                 # (B, L, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)                    # each (B, L, d_model)

        # Reshape to (B, n_heads, L, d_head)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores, then softmax
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if padding_mask is not None:
            # padding_mask: (B, L) True where padded
            scores = scores.masked_fill(
                padding_mask[:, None, None, :], float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values, then reshape back
        out = attn @ v                                    # (B, n_heads, L, d_head)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.proj(out)
```

Notes for reading this:

- **Single `nn.Linear` for `qkv`** and a chunk is a small
  performance win — one matmul instead of three.
- **The mask**. Padded positions must contribute nothing to
  attention. Set their scores to `-inf` before softmax so
  they get weight zero. Chapter 04's `padding_mask`
  connects here.
- In production code, prefer **`F.scaled_dot_product_attention`**
  (introduced in PyTorch 2.0). It picks the fastest kernel
  available (Flash Attention 2 on new NVIDIA hardware,
  memory-efficient attention elsewhere) and handles masking
  correctly. The version above is written for pedagogy.

### The `F.scaled_dot_product_attention` version

For anything you would run in production:

```python
attn_out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,                                       # or an (B, 1, L, L) additive mask
    key_padding_mask=padding_mask if padding_mask is not None else None,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=False,                                      # True for decoder self-attn
)
```

Same math, much faster on real hardware. Reach for it once
the "from scratch" version above makes sense.

## The Transformer block

An encoder block is: multi-head self-attention → feed-forward
network, each with a residual connection and layer norm.

Two conventions exist for where LayerNorm goes:

- **Post-LN** (original 2017 paper). `LayerNorm(x + Sublayer(x))`.
- **Pre-LN** (used in most modern implementations because it
  trains more stably). `x + Sublayer(LayerNorm(x))`.

Pre-LN version, which we will use:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.norm1(x), padding_mask=padding_mask)
        x = x + self.ff(self.norm2(x))
        return x
```

Design decisions in one place:

- **`d_ff = 4 * d_model`** is the classic ratio (768 → 3072
  for BERT-base). Larger widens the block; costs quadratic
  parameters in `d_ff`.
- **`GELU`** activation in the feed-forward. `ReLU` also
  works; `GELU` is now conventional.
- **Two residual connections** — one around attention, one
  around FFN. Both are essential for training stability.
- **LayerNorm, not BatchNorm.** LN normalises per-sample
  across features, which makes it insensitive to batch size
  and to variable sequence lengths.

## Positional information

Self-attention is a set operation: shuffle the tokens and
the outputs shuffle with them. Language and images are not
sets — position matters. We inject positional information
before the first block.

Two common ways:

- **Sinusoidal positional encoding** (2017 paper). Fixed,
  non-learned. Extends to arbitrary sequence lengths.
- **Learned positional embeddings** (BERT, ViT, GPT-2).
  Simple; capped at the maximum sequence length seen at
  training time.

Modern LLMs use **RoPE** (rotary positional embeddings) or
**ALiBi**, which apply position information inside the
attention operation itself and generalise better to long
contexts. RoPE is worth a dedicated read (see resources); we
will use learned embeddings for the small model here.

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, ids: torch.Tensor):
        # ids: (B, L)
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return self.tok(ids) + self.pos(pos)
```

`padding_idx=0` tells the embedding to keep the zero vector
frozen at index 0 — the padding token's embedding never
updates. Cheap correctness.

## The full model

Putting it together for a classification task:

```python
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        max_len:     int,
        d_model:     int = 128,
        n_heads:     int = 4,
        d_ff:        int = 512,
        n_layers:    int = 4,
        n_classes:   int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.embed  = Embedding(vocab_size, max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, ids: torch.Tensor, padding_mask: torch.Tensor | None = None):
        x = self.embed(ids)                                      # (B, L, d_model)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.norm(x)

        # Sequence pooling: mean over non-padded positions.
        if padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = (~padding_mask).unsqueeze(-1).float()          # (B, L, 1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.classifier(pooled)
```

Count the parameters and you will land near a million for the
default hyperparameters — small by modern standards but
enough to learn a sentence-classification task on
mid-thousands of examples.

### Pooling strategies

For classification, you need to reduce the sequence of token
representations to one vector. Three common choices:

- **`[CLS]` token pooling** (BERT). Prepend a special `[CLS]`
  token to every input; take its final-layer representation.
- **Mean pooling over non-padded positions** (used above).
  Robust; no special token needed.
- **Max pooling.** Rare; sometimes helpful.

For this module: mean pooling. It is the simplest correct
choice.

## Initialization matters here

Transformers are more sensitive to init than the models in
chapters 05 and 06. The convention from the original paper
and BERT:

```python
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
```

Call it once in `__init__` after building the modules. Skip
this and training may diverge on the first LR you try.

## Training tricks specific to Transformers

Two knobs to reach for early:

1. **Learning-rate warmup.** Transformers do not train
   stably at their target LR from step 1. The standard
   schedule linearly ramps up from 0 to the peak LR over
   the first few thousand steps, then decays (linear or
   cosine). Chapter 08 covers the mechanic.
2. **Gradient clipping.** `torch.nn.utils.clip_grad_norm_(
   model.parameters(), max_norm=1.0)` after `backward` and
   before `step`. Without it, occasional large gradients
   detonate the model.

You will also usually want:

- **AdamW** (not vanilla Adam), with `weight_decay=0.01`.
- **`lr=1e-4`** to `3e-4` for training from scratch; `2e-5`
  to `5e-5` when fine-tuning a pre-trained checkpoint.
- **`dropout=0.1`** — the default in almost every paper.

## A worked example: text classification

Assembling the pieces for AG News (a small text
classification benchmark with 4 classes, ~120k training
examples):

```python
# 1. Tokenise. For a from-scratch small model, a word-level
#    or BPE tokenizer works; Hugging Face's `tokenizers`
#    library gives you a trainable BPE in a few lines.
#    Assume `ids: (B, L)` and `padding_mask: (B, L)` are the
#    outputs of the tokenizer + collate.

# 2. Model (as above).
model = TransformerEncoder(
    vocab_size=vocab.size,
    max_len=128,
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_layers=4,
    n_classes=4,
).to(device)

# 3. Optimizer + scheduler.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=epochs * len(train_loader),
    pct_start=0.1,           # 10% warmup
)

# 4. Training loop (chapter 03).
loss_fn = nn.CrossEntropyLoss()
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        ids   = batch["ids"].to(device)
        mask  = batch["padding_mask"].to(device)
        y     = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ids, padding_mask=mask)
        loss   = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

Nothing here is Transformer-specific to the *loop*; the
loop is chapter 03's loop. What is new is the model, the
scheduler, and the mask handling.

## Using `nn.TransformerEncoder` (the batteries-included version)

Once you understand the pieces, PyTorch's built-in modules
are what you use in practice:

```python
layer = nn.TransformerEncoderLayer(
    d_model=128, nhead=4, dim_feedforward=512,
    dropout=0.1, activation="gelu",
    batch_first=True, norm_first=True,        # pre-LN
)
encoder = nn.TransformerEncoder(layer, num_layers=4)

# Same signature idea; PyTorch's mask conventions are the
# opposite of Hugging Face's — read the docs before wiring.
```

`nn.TransformerEncoder` gives you a well-tested
implementation with the fast attention kernel path. Prefer
it over the hand-rolled version above for anything but
learning.

For any pre-trained model on a downstream task, prefer
**Hugging Face `transformers`**:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok   = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4,
)
```

Fine-tuning DistilBERT for 3 epochs on AG News will
outperform your from-scratch model in a fraction of the
compute. Chapter 10 revisits why.

## The decoder, in one paragraph

An encoder attends bidirectionally: any position can look at
any other. A **decoder** is the same block with a **causal
mask** — position `i` can only attend to positions `≤ i`. Set
`is_causal=True` in `F.scaled_dot_product_attention`, or add
a triangular `-inf` mask to the attention scores. The decoder
also adds a second attention layer (**cross-attention**) that
attends to the encoder's output; this is what makes
sequence-to-sequence (translation, summarisation) work. GPT-
family models are decoder-only — they use only self-attention
with a causal mask. Chapter 10 covers where to go next for
decoder-focused work.

## Debugging a Transformer that won't train

The failure modes overlap with earlier chapters, plus:

- **Loss stuck at `ln(C)`.** The model is outputting uniform
  probabilities. Usually: LR too high (try 10× lower), or
  gradients exploding (add clipping), or no positional
  encoding (all positions look identical to attention).
- **Loss NaN after a few hundred steps.** Warmup missing;
  gradient clipping missing; init too aggressive. Fix in
  that order.
- **Padding leaks into predictions.** Check every mask
  path: `masked_fill(-inf)` before softmax, mask in the
  pooling. A missing mask is silent — the loss still drops,
  just to a wrong local minimum.
- **Attention weights are one-hot.** Softmax has saturated;
  scores are on the wrong scale. Check the `/ √d_k`
  scaling.
- **Model plateaus early.** Try widening `d_model`, adding a
  layer, or increasing warmup length. And run the
  "overfit a batch of 8" test (chapter 03).

## Pitfalls collected

1. **No positional encoding.** Model becomes permutation-
   invariant; training goes nowhere.
2. **Wrong LayerNorm placement.** Post-LN trains less
   stably; pre-LN is the modern default.
3. **Forgetting the `/ √d_k` scaling.** Softmax saturates,
   gradients vanish.
4. **Missing padding mask.** Padding tokens leak into
   attention and pooling.
5. **Skipping LR warmup.** Loss diverges in the first few
   hundred steps.
6. **Skipping gradient clipping.** One bad batch kills the
   training run.
7. **Training from scratch when a pre-trained model exists.**
   Fine-tuning DistilBERT will beat your from-scratch
   Transformer on any small text classification benchmark.
8. **Using `nn.MultiheadAttention` without reading its docs.**
   PyTorch's mask conventions ("`True` means mask out")
   trip everyone up at least once. Same for
   `attn_mask` vs `key_padding_mask`.

## Summary

- Attention lets every position attend to every position
  with learnable weights; multi-head runs several such
  attentions in parallel.
- A Transformer block = (self-attention + residual + LN) +
  (feed-forward + residual + LN). Stack N of them.
- Positional information must be injected explicitly —
  learned embeddings, sinusoidal, or (modern) RoPE / ALiBi.
- Transformers need: AdamW, weight decay, LR warmup,
  gradient clipping, and correct masking. Skip any and
  training may diverge.
- For real work: use `nn.TransformerEncoder` or Hugging
  Face `transformers`. The from-scratch build is for
  learning.

## Where this goes next

Chapter 08 covers the regularisation and LR-schedule
knobs mentioned throughout this chapter, in one place.
