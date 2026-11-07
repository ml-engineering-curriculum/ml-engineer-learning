# quiz-01 — Deep Learning Fundamentals with PyTorch

> **Format:** 12 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–10 of this module.
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter the point
> comes from — go re-read that section if you missed the question.

---

## Questions

### 1. (Chapter 01) A product team has 4,000 rows of fully tabular data (`income`, `country`, `is_returning_user`, a dozen more semantic features) and a binary label. A junior engineer proposes training a 3-layer MLP first "because deep learning is state of the art". Which option best matches how chapter 01 frames this decision?

A. Build the MLP first; the three-signal rule says deep learning wins on any
   classification problem.
B. Build the MLP with heavy regularisation and data augmentation on tabular
   features. The regularisation techniques in chapter 08 eliminate the
   small-data disadvantage.
C. Reach for a `mod-104` gradient-boosted tree first. The features are already
   tabular and semantic, the dataset is small (< 10k rows), and chapter 01
   explicitly says "beat a gradient-boosted tree before you commit to a deep
   network." The three *use* signals (high-structure input, lots of data,
   relevant pre-trained checkpoint) are all absent; the three *don't* signals
   (tabular/semantic features, < 10k rows, interpretability-friendly) all
   apply.
D. Train a Transformer encoder from chapter 07 directly on the rows — any
   modern problem should start from a Transformer.

### 2. (Chapter 02) A new engineer writes the first two steps of a training loop. On step 1 they call `loss.backward()` then `optimizer.step()`. On step 2 they call `loss.backward()` and `optimizer.step()` again, without any other call between the two steps. The validation loss is nonsense. Which chapter-02 fact explains the bug, and what is the fix?

A. Nothing is wrong. Two gradient steps always compose correctly.
B. The issue is that `requires_grad` propagates; wrap the forward pass in
   `torch.no_grad()` to prevent gradient build-up.
C. Gradients accumulate on the `.grad` attribute of leaf tensors across
   `.backward()` calls — the step-2 gradient is the sum of step 1 and step 2's
   gradients, so the optimiser takes a step in the wrong direction. Insert
   `optimizer.zero_grad()` at the top of every iteration (the modern default
   `set_to_none=True` is slightly faster than filling with zeros).
D. The parameters were not moved to the device before the optimiser was
   constructed; `.backward()` therefore writes to orphan CPU tensors.

### 3. (Chapter 02) Which statement correctly reflects chapter 02's rule about validation code and inference code?

A. Any code path that does not call `.backward()` should be inside
   `torch.no_grad()` or `@torch.inference_mode()`, and `model.eval()` should
   be set first; the two toggles are orthogonal and both matter (dropout and
   BatchNorm change behaviour; autograd graph is suppressed).
B. `.train()` / `.eval()` are enough on their own — they already turn off
   autograd for you.
C. `torch.no_grad()` alone is enough — the model stays in whatever mode it was
   in, and that is fine for validation.
D. `torch.inference_mode()` is strictly weaker than `torch.no_grad()`; use
   `no_grad` whenever possible.

### 4. (Chapter 03) A teammate's PR applies `nn.Softmax(dim=-1)` to the logits coming out of the classifier head, then passes the result into `nn.CrossEntropyLoss()`. Training looks "sort of" like it learns, but the final accuracy is well below the known baseline. Which chapter-03 pitfall is this, and what is the fix?

A. The loss function is being called with the wrong `reduction`. Pass
   `reduction="mean"` to `CrossEntropyLoss`.
B. `CrossEntropyLoss` fuses `log_softmax` and negative log-likelihood, so the
   PR is effectively applying softmax twice — once explicitly and once inside
   the loss. The fix is to pass **raw logits** to the loss (and symmetrically,
   pass raw logits to `BCEWithLogitsLoss`, never sigmoid-then-BCE).
C. `CrossEntropyLoss` expects the labels to be floats; the real bug is a dtype
   mismatch unrelated to the softmax.
D. Softmax belongs before the loss; the real issue is that the softmax is
   applied along the wrong axis.

### 5. (Chapter 04) An image dataset opens an HDF5 file handle inside `__init__`. The team sets `num_workers=4` on the training `DataLoader`. Once training starts, batches intermittently return corrupted tensors. Which chapter-04 fact explains the failure, and what is the recommended fix?

A. `num_workers > 0` is unsupported for HDF5; drop to `num_workers=0`
   permanently and accept the slower pipeline.
B. The DataLoader workers each `fork` (or `spawn`) subprocesses that inherit
   the parent's open file descriptors; they then share and corrupt the same
   HDF5 handle. Chapter 04 says to open the handle **lazily inside
   `__getitem__`** (or use a `worker_init_fn` that opens a fresh handle per
   worker), and to be similarly careful with un-seeded augmentation randomness
   across workers.
C. The DataLoader is batching dicts incorrectly; add a custom `collate_fn`.
D. The HDF5 file is corrupt on disk; nothing inside PyTorch could cause this.

### 6. (Chapter 05) Inside an `nn.Module`, an engineer writes `self.layers = [nn.Linear(10, 10), nn.Linear(10, 10)]`. Training runs without errors, but the loss never moves and the second layer's weights never change. Which chapter-05 rule is being broken, and what is the fix?

A. `nn.Linear` is not a trainable module in a plain Python list; convert both
   to `nn.Parameter` directly.
B. A plain Python list of `nn.Module`s is **not** auto-registered on the
   parent `Module` — the parameters do not appear in `.parameters()`,
   `.state_dict()`, or `.to(device)`, and the optimiser never sees them.
   Replace the list with `nn.ModuleList([...])` (or `nn.Sequential(...)` if the
   stack is linear). The same lesson applies to `nn.ParameterList` for raw
   tensors.
C. The `__init__` method forgot `super().__init__()`; once added, the Python
   list of modules works fine.
D. `nn.Linear` layers need to be stored on `self` with a numeric suffix (e.g.
   `self.fc_1`, `self.fc_2`) for registration to trigger.

### 7. (Chapter 06) Chapter 06 argues that a stack of **two** `nn.Conv2d(..., kernel_size=3, padding=1)` layers is preferred over a single `nn.Conv2d(..., kernel_size=5, padding=2)` layer in modern CNNs. Which reasoning matches the chapter?

A. The 5×5 conv is strictly more expressive and should always be preferred;
   chapter 06 recommends it on any CNN that fits in memory.
B. Two 3×3 convs have the same receptive field as one 5×5 conv (5×5 pixels of
   the input) with **fewer parameters and an extra nonlinearity between them**.
   Chapter 06 says "modern CNNs use 3×3 almost exclusively" because deep
   stacks of small filters grow the receptive field cheaply; downsampling
   (strided conv or pooling) gets you there faster than depth alone.
C. The two conventions are numerically equivalent; the difference is only
   cosmetic.
D. 5×5 convs are incompatible with `nn.BatchNorm2d`; the 3×3 path is required
   to use BN.

### 8. (Chapter 07) Which single statement most accurately captures what chapter 07 says Transformers need in order to train stably from scratch?

A. Transformers train fine with the same hyperparameters as an MLP — SGD,
   `lr=1e-3`, no scheduler. Any extra machinery is premature optimisation.
B. The attention implementation already handles position, so no positional
   encoding is required; sinusoidal / learned / RoPE schemes are only for
   very long sequences.
C. Transformers need **positional information added before block 1**,
   **AdamW + weight decay**, **LR warmup then decay**, **gradient clipping**
   (`max_norm=1.0`), the `/√d_k` scaling inside attention, and a correct
   **padding mask** that is threaded through both attention (set padded
   positions to `-inf` before the softmax) and the pooling. Skip any of these
   and training may diverge or silently learn the wrong thing.
D. Post-LN (`LayerNorm(x + Sublayer(x))`) is the modern default because it
   trains more stably than pre-LN.

### 9. (Chapter 08) A teammate configures `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)`, passing every parameter — weights, biases, LayerNorm scale/bias, and token embeddings — through the single `weight_decay` setting. Chapter 08 flags two issues. Which option names both?

A. Nothing is wrong; `AdamW` already excludes biases and norms internally.
B. (i) `AdamW` is the wrong optimiser for weight decay; use `Adam` with
   `weight_decay` instead. (ii) Weight decay should be in the range `1e-6` to
   `1e-8`.
C. (i) `Adam(weight_decay=...)` and `AdamW(weight_decay=...)` **differ** —
   plain Adam folds the decay into the gradient before adaptive scaling, so
   parameters with large gradients get *less* decay; `AdamW` decouples the
   decay from the gradient update (the Loshchilov & Hutter result the module
   cites), which is why "use `AdamW` when you want weight decay." (ii) The
   convention is to build **two parameter groups**: one with decay for matrix
   weights (`p.ndim >= 2`), one with `weight_decay=0.0` for biases, LayerNorm
   / BatchNorm parameters, and embeddings — because decaying the latter
   slightly hurts performance.
D. (i) Weight decay must equal the learning rate. (ii) `weight_decay` must be
   set on the LR scheduler, not on the optimiser.

### 10. (Chapter 09) An engineer wants to use mixed-precision training. They add `torch.autocast(device_type="cuda", dtype=torch.float16)` around the forward and loss, then call `loss.backward()` and `optimizer.step()` directly — no `GradScaler`. Training stops improving after a few hundred steps and gradients become identically zero. Chapter 09 explains both why this happens and the two valid fixes. Which option names them correctly?

A. `float16` is unsupported for training; use `float32` only.
B. `float16` has a much narrower exponent range than `float32` and
   **small gradients can underflow to zero**. There are two valid fixes:
   (i) wrap the loss in `torch.amp.GradScaler("cuda")` — `scaler.scale(loss).
   backward()` scales the loss up so gradients are representable, and
   `scaler.step(optimizer)` / `scaler.update()` unscale before the parameter
   update (and `scaler.unscale_(opt)` first if you want to `clip_grad_norm_`);
   or (ii) **switch the autocast dtype to `torch.bfloat16`**, which has
   `float32`'s exponent range so gradients don't underflow — no scaler needed.
   Chapter 09's rule: "if your hardware supports bf16, use bf16."
C. `float16` autocast requires calling `optimizer.step()` twice per batch.
D. The fix is to call `.backward()` inside the autocast context; autocast
   handles scaling automatically in that case.

### 11. (Chapter 10) Which option most accurately describes **LoRA** (and **QLoRA**) as chapter 10 presents them?

A. LoRA is a new activation function; QLoRA is its quantized variant.
B. LoRA and QLoRA are preference-optimisation algorithms used in the
   post-training stage in place of RLHF or DPO.
C. LoRA is a parameter-efficient fine-tuning (PEFT) method that inserts small
   trainable low-rank matrices into each layer while **leaving the base
   model's weights frozen**; only the adapters train, so storage is
   megabytes-per-task instead of gigabytes and adapters can be swapped
   between tasks. **QLoRA** additionally runs the frozen base model in 4-bit
   precision, which is what makes it possible to fine-tune a 65B-parameter
   model on a single 24 GB GPU.
D. LoRA is a chunking strategy for retrieval-augmented generation (RAG) and
   is unrelated to fine-tuning.

### 12. (Chapter 10) A product manager asks: "Users keep complaining that our LLM-based summarisation feature invents legal citations. Can we just fine-tune the model to stop hallucinating?" Which answer most faithfully matches chapter 10's decision map?

A. Yes — a single round of fine-tuning on examples labelled "no hallucination"
   is the canonical fix.
B. Hallucination is a solved problem; switching to any larger closed-API
   model guarantees citations are real.
C. Hallucination is not a solved problem. For knowledge-intensive tasks like
   this one, chapter 10 says RAG (retrieve the actual citations from a
   trusted vector-indexed source and inject them into the prompt) is what
   reduces hallucination, that system prompts asking for citations help, and
   that domain-critical applications need retrieval, verification, or human
   review on top. Building a production RAG pipeline — chunking, embedding,
   retriever, reranker, evaluation — belongs in the **LLM Engineering** track;
   `mod-105` is where you recognise the shape and delegate.
D. Hallucination only affects image models; text models cannot hallucinate.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **C.** Chapter 01 ("When to *not* reach for deep learning") lists three
   signals — tabular/semantic features, dataset under ~10k rows, and an
   interpretability-critical stakeholder — any one of which is enough to stay
   with the `mod-104` toolkit. The chapter also states, and chapter 05
   repeats: "beat a gradient-boosted tree before you commit to a deep
   network." None of the three *reach-for* signals (structured input, scale
   in the hundreds of thousands to millions, relevant pre-trained checkpoint)
   applies here.

2. **C.** Chapter 02's section "Autograd in one figure" states explicitly:
   "Gradients accumulate. If you call `.backward()` twice, `w.grad` sums the
   two runs. The training loop resets grads at the start of each step." The
   same point appears in chapter 02's "Pitfalls collected" and chapter 03's
   "The four lines, one by one"; the default is `set_to_none=True` in
   PyTorch 2.x.

3. **A.** Chapter 02 ("Turning autograd off" and "Train mode vs eval mode")
   states the rule: "Any code path that does not call `.backward()` should be
   inside `torch.no_grad()` or `@torch.inference_mode()`." It then
   separately: "`.train()` / `.eval()` do **not** affect autograd. That is
   what `torch.no_grad()` is for. The two are orthogonal and you use both."
   The chapter also notes that `inference_mode` is *stricter* than
   `no_grad` (not weaker) — ruling out D.

4. **B.** Chapter 03 (table of losses, followed by "`CrossEntropyLoss`
   combines log-softmax and NLL. Do not apply `softmax` first — you will
   double-softmax and the loss will be numerically wrong. Pass raw logits.")
   and pitfall #2: "Applied `softmax` before `CrossEntropyLoss`. Double
   softmax; nonsense loss." Chapter 05 ("Output layers by task") repeats the
   rule for `BCEWithLogitsLoss` as well: output raw logits, always.

5. **B.** Chapter 04 ("Workers, GILs, and forking gotchas") names this exact
   failure: "Every worker inherits the parent's file descriptors. Opening a
   database or an HDF5 file *in `__init__`* corrupts across workers. Open
   lazily inside `__getitem__`, or override `worker_init_fn` to open a fresh
   handle per worker." Pitfall #2 of the same chapter restates it, and
   pitfall #3 covers the related "un-seeded augmentation across workers"
   case.

6. **B.** Chapter 05 ("The anatomy of an `nn.Module`", rule 3, and pitfall #1
   "Registering layers in a Python list. Parameters vanish. `nn.ModuleList`
   or `nn.Sequential`.") The chapter's example contrasts the wrong
   (`self.layers = [nn.Linear(10, 10), nn.Linear(10, 10)]`) against the right
   (`self.layers = nn.ModuleList([...])` or `nn.Sequential(...)`). Chapter 02
   makes the same point for `nn.Parameter` and `nn.ParameterList`.

7. **B.** Chapter 06 ("Receptive field") makes this argument explicitly:
   "A deep stack of small filters (3×3) matches the receptive field of a
   shallow stack of large filters (7×7) with fewer parameters and more
   nonlinearities in between. This is why modern CNNs use 3×3 almost
   exclusively." The chapter also distinguishes receptive-field growth from
   downsampling: "Downsampling (strided conv or pooling) gets you there
   faster than depth alone."

8. **C.** Chapter 07's "Training tricks specific to Transformers" and
   pitfalls 1–6 list each of these prerequisites. Positional encoding
   (pitfall #1, "No positional encoding. Model becomes permutation-
   invariant; training goes nowhere"); the `/√d_k` scaling in the attention
   formula (pitfall #3); correct padding mask in attention *and* pooling
   (pitfall #4, and the mean-pool code in the chapter); LR warmup (pitfall
   #5); gradient clipping with `max_norm=1.0` (pitfall #6); and AdamW with
   weight decay in the "Training tricks" section. Chapter 07 also explicitly
   prefers **pre-LN** over post-LN, ruling out option D.

9. **C.** Chapter 08 ("Weight decay") says: "`Adam(weight_decay=...)` and
   `AdamW(weight_decay=...)` implement it differently. In `Adam`, the
   weight-decay term gets folded into the gradient before Adam's adaptive
   scaling, which effectively *reduces* weight decay for parameters with
   large gradients. In `AdamW`, weight decay is applied to the parameter
   update *after* Adam's scaling — decoupled." The same section gives the
   two-parameter-group code and the rule "Don't decay biases, LayerNorms, or
   embeddings. Only matrix weights." Pitfall #2 of chapter 08 restates it.

10. **B.** Chapter 09 ("bf16 vs fp16: choose bf16 when available") and
    pitfall #1: "fp16 without `GradScaler`. Gradients underflow; training
    doesn't converge. Either add the scaler or switch to bf16." The
    "autocast + GradScaler pattern (fp16)" section walks through
    `scaler.scale(loss).backward()`, `scaler.unscale_(opt)` before clipping,
    and `scaler.step(opt)` / `scaler.update()`. The "bf16 pattern" section
    shows the simpler scaler-free version. Chapter 09's rule: "If your
    hardware supports bf16, use bf16. It is simpler and less error-prone."

11. **C.** Chapter 10 ("Parameter-efficient fine-tuning (PEFT)") describes
    LoRA exactly this way: "insert small trainable rank-`r` matrices into
    each layer; train those only. Storage-cheap (adapters are megabytes,
    not gigabytes), fast to train, easily swapped between tasks." And then:
    "**QLoRA** runs the frozen base model in 4-bit precision; lets you
    fine-tune 65B-parameter models on a single 24 GB GPU." Options A, B, D
    conflate LoRA with other concepts chapter 10 catalogues separately.

12. **C.** Chapter 10 ("Hallucination") says: "The model produces confident,
    wrong output. Not a solved problem. RAG reduces it for knowledge-
    intensive tasks; system prompts asking for citations help; model choice
    matters (larger models hallucinate less on average). Domain-critical
    applications need retrieval, verification, or human review." The
    decision map and the delegation map together say that **production
    RAG** — retrieval, embedding, reranking, chunking strategies,
    evaluation — delegates to the **LLM Engineering** track; `mod-105`'s job
    is to recognise which specialist to call.

</details>
