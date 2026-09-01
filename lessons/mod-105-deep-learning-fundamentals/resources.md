# Resources for mod-105 — Deep Learning Fundamentals with PyTorch

A curated short list. The goal is depth-on-demand for the
chapter you got stuck on, not a comprehensive bibliography.
Primary documentation is listed first; textbooks, papers, and
tooling references follow.

## Cross-cutting: PyTorch itself

- **PyTorch — official documentation.**
  <https://pytorch.org/docs/stable/index.html>
  The main reference. Every module named in chapters 02–09
  (`nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`,
  `nn.TransformerEncoder`, `torch.autocast`, `GradScaler`,
  `DataLoader`, `torch.compile`) is documented here with the
  full signature and examples.
- **PyTorch — tutorials.**
  <https://pytorch.org/tutorials/>
  The "Learn the Basics" and "Introduction to PyTorch on
  YouTube" tracks are the fastest way in for someone who
  has read chapters 02–04 and wants a second pass on the
  same material.
- **PyTorch — `torch.nn` API reference.**
  <https://pytorch.org/docs/stable/nn.html>
  Every layer, loss, and container. Read the "Notes" boxes;
  they call out the gotchas the docstring can't.
- **PyTorch — `torch.autograd` mechanics.**
  <https://pytorch.org/docs/stable/notes/autograd.html>
  The technical reference behind chapter 02's autograd
  explanation. Read once you have written the training loop
  from scratch.
- **PyTorch — reproducibility notes.**
  <https://pytorch.org/docs/stable/notes/randomness.html>
  What `torch.manual_seed`, `torch.use_deterministic_algorithms`,
  and cuDNN benchmark mode actually guarantee (and don't).
- **PyTorch — release notes.**
  <https://github.com/pytorch/pytorch/releases>
  Worth skimming when you upgrade a major version. The 2.0
  release introduced `torch.compile` and
  `F.scaled_dot_product_attention`, both cited throughout
  chapters 07 and 09.
- **`torchvision` — documentation.**
  <https://pytorch.org/vision/stable/index.html>
  Datasets, model zoo, transforms — all cited by chapter 06.
- **Hugging Face `datasets` — documentation.**
  <https://huggingface.co/docs/datasets/>
  The dataset loader exercise-03 uses for AG News, IMDb,
  and friends.

## Cross-cutting: textbooks

- **Goodfellow, Ian, Yoshua Bengio, and Aaron Courville.
  *Deep Learning* (MIT Press, 2016).** Free HTML at
  <https://www.deeplearningbook.org/>.
  The canonical academic reference. Chapters 6–9 (feed-
  forward nets, regularisation, optimisation, convolutional
  nets) map onto chapters 05–08 of this module. A bit dated
  on Transformers (published pre-2017) — use Prince or Zhang
  for that.
- **Prince, Simon J. D. *Understanding Deep Learning* (MIT
  Press, 2023).** Free PDF at
  <https://udlbook.github.io/udlbook/>.
  The modern successor to *Deep Learning*. Covers
  Transformers, diffusion, and modern training practice.
  Chapters 3–12 are the natural pair with `mod-105`.
- **Zhang, Aston, Zack Lipton, Mu Li, and Alex Smola. *Dive
  into Deep Learning* (Cambridge University Press, 2023).**
  Free HTML at <https://d2l.ai/>.
  Textbook-plus-notebooks. Every concept is paired with
  runnable PyTorch (and MXNet, and TensorFlow) code. The
  chapters on attention and Transformers (10–11) are the
  most useful cross-read for chapter 07.
- **Bishop, Christopher M., and Hugh Bishop. *Deep Learning:
  Foundations and Concepts* (Springer, 2024).** Free HTML at
  <https://www.bishopbook.com/>.
  A very readable modern textbook. Chapters 6 (deep nets)
  and 8 (CNNs) are the natural pair with chapters 05–06;
  chapter 12 is the Transformer counterpart.
- **Chollet, François. *Deep Learning with Python* (2nd
  ed., Manning, 2021).** Keras-oriented but the concepts
  translate. Chapter 3 (the training loop) is the friendliest
  pass over chapter 03's material anywhere.
- **Howard, Jeremy, and Sylvain Gugger. *Deep Learning for
  Coders with Fastai and PyTorch* (O'Reilly, 2020).** Free
  HTML at <https://course.fast.ai/Resources/book.html>. The
  practitioner-oriented companion. Chapters 4 (SGD from
  scratch), 5 (image classification), and 12 (NLP deep dive)
  are the direct analogues.
- **Karpathy, Andrej. "Neural Networks: Zero to Hero" (video
  lecture series).**
  <https://karpathy.ai/zero-to-hero.html>
  The "makemore" and "GPT from scratch" videos are the
  single best video companion to chapter 07 and chapter 10.

## Chapter 02 — Tensors, autograd, and devices

- **PyTorch — tensor creation.**
  <https://pytorch.org/docs/stable/torch.html#creation-ops>
  Every `torch.tensor`, `torch.zeros`, `torch.arange`,
  `torch.linspace` variant on one page.
- **PyTorch — CUDA semantics.**
  <https://pytorch.org/docs/stable/notes/cuda.html>
  Device transfer, streams, memory management, pinned memory
  — the background for chapter 02's device section.
- **PyTorch — MPS (Apple Silicon) backend.**
  <https://pytorch.org/docs/stable/notes/mps.html>
  What works and what doesn't on M-series Macs.
- **PyTorch — autograd tutorial.**
  <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>
  A gentle intro to `requires_grad`, `backward`, and the
  computation graph.
- **PyTorch — extending autograd (`torch.autograd.Function`).**
  <https://pytorch.org/docs/stable/notes/extending.html>
  Read once you have hit the limits of the built-in ops.

## Chapter 03 — The training loop from scratch

- **PyTorch — optimizer reference (`torch.optim`).**
  <https://pytorch.org/docs/stable/optim.html>
  `SGD`, `Adam`, `AdamW`, and the schedulers referenced in
  chapters 03 and 08.
- **PyTorch — loss functions (`torch.nn` losses).**
  <https://pytorch.org/docs/stable/nn.html#loss-functions>
  Every built-in loss (`CrossEntropyLoss`, `MSELoss`,
  `BCEWithLogitsLoss`, `HuberLoss`, and the rest).
- **PyTorch — saving and loading models.**
  <https://pytorch.org/tutorials/beginner/saving_loading_models.html>
  Canonical guidance on `state_dict()`, `torch.save`,
  `torch.load`, and the `map_location` argument. Cited by
  chapter 03's checkpointing section.
- **Karpathy, Andrej. "A Recipe for Training Neural
  Networks."** <https://karpathy.github.io/2019/04/25/recipe/>
  The single best debugging checklist for a training run
  that isn't working. Chapter 03's "debug in this order" is
  descended from it.
- **Kingma, Diederik P., and Jimmy Ba. "Adam: A Method for
  Stochastic Optimization" (ICLR 2015).**
  <https://arxiv.org/abs/1412.6980>
  The Adam paper. Section 4 (algorithm) is the sourced
  version of every `Adam`/`AdamW` implementation.
- **Robbins, Herbert, and Sutton Monro. "A Stochastic
  Approximation Method" (Annals of Mathematical Statistics,
  1951).**
  <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full>
  The original paper introducing what we now call SGD. Short
  and historically worth knowing.

## Chapter 04 — Datasets, DataLoaders, and collate

- **PyTorch — `torch.utils.data` reference.**
  <https://pytorch.org/docs/stable/data.html>
  `Dataset`, `IterableDataset`, `DataLoader`, `Sampler`,
  `BatchSampler`, `collate_fn` — the full API.
- **PyTorch — data loading tutorial.**
  <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>
  The worked example behind chapter 04's `Dataset` pattern.
- **PyTorch — writing a custom Dataset.**
  <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>
  The "beginner basics" version — worth skimming before
  chapter 04 if you have never subclassed `Dataset`.
- **PyTorch — `DataLoader` performance notes.**
  <https://pytorch.org/docs/stable/data.html#multi-process-data-loading>
  The section on `num_workers`, `persistent_workers`,
  `pin_memory`, and worker init functions. Directly relevant
  to the "why is my GPU at 15%?" debugging in chapter 04.

## Chapter 05 — Building an MLP

- **PyTorch — `nn.Linear`, `nn.ReLU`, `nn.Sequential`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>
  <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>
  <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>
- **PyTorch — `nn.Embedding`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>
  The `padding_idx` argument matters — see chapter 07's
  padding discussion.
- **Rosenblatt, Frank. "The Perceptron: A Probabilistic
  Model for Information Storage and Organization in the
  Brain" (Psychological Review, 1958).**
  <https://doi.org/10.1037/h0042519>
  The historical origin of the artificial neuron. Cited by
  every deep-learning textbook.
- **Rumelhart, David E., Geoffrey E. Hinton, and Ronald J.
  Williams. "Learning representations by back-propagating
  errors" (Nature, 1986).**
  <https://www.nature.com/articles/323533a0>
  The backpropagation paper. What autograd (chapter 02)
  automates.
- **Nair, Vinod, and Geoffrey E. Hinton. "Rectified Linear
  Units Improve Restricted Boltzmann Machines" (ICML 2010).**
  <https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf>
  The paper that made ReLU the default activation.
- **Hendrycks, Dan, and Kevin Gimpel. "Gaussian Error Linear
  Units (GELUs)" (2016).**
  <https://arxiv.org/abs/1606.08415>
  The GELU paper. GELU is now the default in Transformers.
- **He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  "Delving Deep into Rectifiers: Surpassing Human-Level
  Performance on ImageNet Classification" (ICCV 2015).**
  <https://arxiv.org/abs/1502.01852>
  The paper behind Kaiming initialisation — the default for
  ReLU networks in PyTorch.
- **Glorot, Xavier, and Yoshua Bengio. "Understanding the
  difficulty of training deep feedforward neural networks"
  (AISTATS 2010).**
  <https://proceedings.mlr.press/v9/glorot10a.html>
  The Xavier / Glorot initialisation paper. Complements
  Kaiming; the default for tanh / sigmoid networks.

## Chapter 06 — Convolutional neural networks

- **PyTorch — convolution layers.**
  <https://pytorch.org/docs/stable/nn.html#convolution-layers>
  `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose*`. The formula
  for output shape is on each page.
- **PyTorch — `nn.BatchNorm2d`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>
  Read the "buffers vs. parameters" note; it explains why
  BN behaves differently in `train()` and `eval()` mode.
- **PyTorch — `nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>
  <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>
- **`torchvision` — datasets.**
  <https://pytorch.org/vision/stable/datasets.html>
  `CIFAR10`, `CIFAR100`, `MNIST`, `ImageNet`, `Imagenette`,
  and the rest. Cited by exercise-02.
- **`torchvision` — models with pretrained weights.**
  <https://pytorch.org/vision/stable/models.html>
  The model zoo (ResNet, MobileNet, EfficientNet, ViT, ...)
  and the `weights.transforms()` idiom exercise-02 uses.
- **`torchvision` — transforms v2.**
  <https://pytorch.org/vision/stable/transforms.html>
  The augmentation pipeline. `RandAugment`, `AutoAugment`,
  `MixUp`, `CutMix`, `RandomErasing` are all here.
- **LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick
  Haffner. "Gradient-Based Learning Applied to Document
  Recognition" (Proc. IEEE, 1998).**
  <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>
  The LeNet paper — the origin of the modern CNN.
- **Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton.
  "ImageNet Classification with Deep Convolutional Neural
  Networks" (NeurIPS 2012).**
  <https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>
  AlexNet. The paper that started the modern deep-learning
  era.
- **Simonyan, Karen, and Andrew Zisserman. "Very Deep
  Convolutional Networks for Large-Scale Image Recognition"
  (ICLR 2015).**
  <https://arxiv.org/abs/1409.1556>
  VGG. The paper behind "stack lots of 3×3 convs" as the
  standard building pattern.
- **He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  "Deep Residual Learning for Image Recognition" (CVPR 2016).**
  <https://arxiv.org/abs/1512.03385>
  ResNet. The residual-connection paper. The most-cited
  computer-vision paper of the decade; every modern
  architecture (including Transformers) uses residuals for
  the same reason.
- **Ioffe, Sergey, and Christian Szegedy. "Batch
  Normalization: Accelerating Deep Network Training by
  Reducing Internal Covariate Shift" (ICML 2015).**
  <https://arxiv.org/abs/1502.03167>
  The BatchNorm paper. Read section 3 for the algorithm
  chapter 06 uses.
- **Howard, Andrew G., et al. "Searching for MobileNetV3"
  (ICCV 2019).**
  <https://arxiv.org/abs/1905.02244>
  The MobileNetV3 paper. Relevant for exercise-02's
  fine-tune option on constrained hardware.
- **Dosovitskiy, Alexey, et al. "An Image is Worth 16x16
  Words: Transformers for Image Recognition at Scale" (ICLR
  2021).**
  <https://arxiv.org/abs/2010.11929>
  ViT. The paper that showed convolutions are not necessary
  for image classification if you have enough data.

## Chapter 07 — A small Transformer encoder

- **PyTorch — `nn.TransformerEncoder`,
  `nn.TransformerEncoderLayer`, `nn.MultiheadAttention`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html>
  <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>
  <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>
  Read the mask-convention notes carefully; chapter 07's
  pitfall #8 is exactly this.
- **PyTorch — `F.scaled_dot_product_attention`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>
  The fused-kernel attention introduced in PyTorch 2.0. The
  fastest correct default for anything you would ship.
- **PyTorch — `nn.LayerNorm`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>
- **Vaswani, Ashish, et al. "Attention Is All You Need"
  (NeurIPS 2017).**
  <https://arxiv.org/abs/1706.03762>
  The original Transformer paper. Sections 3.2 (scaled
  dot-product attention) and 3.3 (multi-head attention) are
  the source for chapter 07's core equations.
- **Devlin, Jacob, et al. "BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding"
  (NAACL 2019).**
  <https://arxiv.org/abs/1810.04805>
  The BERT paper. The canonical encoder-only Transformer;
  every architectural choice in chapter 07 either matches
  BERT or explicitly diverges from it.
- **Sanh, Victor, et al. "DistilBERT, a distilled version of
  BERT" (NeurIPS EMC² workshop, 2019).**
  <https://arxiv.org/abs/1910.01108>
  The DistilBERT paper. The model exercise-03 fine-tunes.
- **Xiong, Ruibin, et al. "On Layer Normalization in the
  Transformer Architecture" (ICML 2020).**
  <https://arxiv.org/abs/2002.04745>
  The pre-LN vs. post-LN paper. Section 3 explains why
  chapter 07 uses pre-LN.
- **Su, Jianlin, et al. "RoFormer: Enhanced Transformer with
  Rotary Position Embedding" (Neurocomputing, 2023).**
  <https://arxiv.org/abs/2104.09864>
  The RoPE paper. The modern default for positional
  encoding. Chapter 07 mentions it; exercise-03's stretch
  goal implements it.
- **Press, Ofir, Noah A. Smith, and Mike Lewis. "Train
  Short, Test Long: Attention with Linear Biases Enables
  Input Length Extrapolation" (ICLR 2022).**
  <https://arxiv.org/abs/2108.12409>
  The ALiBi paper. The other modern positional-encoding
  approach.
- **Dao, Tri, et al. "FlashAttention: Fast and Memory-
  Efficient Exact Attention with IO-Awareness" (NeurIPS
  2022).**
  <https://arxiv.org/abs/2205.14135>
  The FlashAttention paper. The reason
  `F.scaled_dot_product_attention` is fast on modern NVIDIA
  hardware. FlashAttention-2 (<https://arxiv.org/abs/2307.08691>)
  and 3 (<https://arxiv.org/abs/2407.08608>) are worth
  knowing exist.
- **Hugging Face — `tokenizers` library.**
  <https://huggingface.co/docs/tokenizers/>
  The library exercise-03 uses to train a BPE tokenizer.
- **Hugging Face — `transformers` library.**
  <https://huggingface.co/docs/transformers/>
  The library behind `AutoModelForSequenceClassification`.
  The "Fine-tuning a pretrained model" tutorial is directly
  relevant to exercise-03's DistilBERT baseline.
- **Karpathy, Andrej. "nanoGPT."**
  <https://github.com/karpathy/nanoGPT>
  A ~300-line, from-scratch GPT implementation. The
  decoder-side counterpart to chapter 07. Read after
  chapter 07 if you want to see the causal version end to
  end.

## Chapter 08 — Regularisation and LR schedules

- **PyTorch — learning-rate schedulers.**
  <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>
  `StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `LambdaLR`,
  `ReduceLROnPlateau`, and the warmup helpers. Cited by
  chapter 08.
- **PyTorch — `nn.Dropout`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html>
- **PyTorch — `torch.nn.utils.clip_grad_norm_`.**
  <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>
  The gradient-clipping helper chapter 08 (and every
  Transformer training run) uses.
- **Srivastava, Nitish, et al. "Dropout: A Simple Way to
  Prevent Neural Networks from Overfitting" (JMLR, 2014).**
  <https://jmlr.org/papers/v15/srivastava14a.html>
  The Dropout paper. Section 3.2 explains why inference is
  scaled differently from training.
- **Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
  "Layer Normalization" (2016).**
  <https://arxiv.org/abs/1607.06450>
  The LayerNorm paper. Chapter 07 uses LayerNorm; chapter 08
  cites the batch-size-invariance argument from section 3.
- **Loshchilov, Ilya, and Frank Hutter. "Decoupled Weight
  Decay Regularization" (ICLR 2019).**
  <https://arxiv.org/abs/1711.05101>
  The AdamW paper. The reason `AdamW` is the default in
  chapters 07–09.
- **Smith, Leslie N. "Cyclical Learning Rates for Training
  Neural Networks" (WACV 2017).**
  <https://arxiv.org/abs/1506.01186>
  The paper behind PyTorch's `OneCycleLR` and the LR-range-
  test methodology exercise-01's stretch goal exercises.
- **Smith, Leslie N., and Nicholay Topin. "Super-Convergence:
  Very Fast Training of Neural Networks Using Large Learning
  Rates" (2018).**
  <https://arxiv.org/abs/1708.07120>
  The follow-up paper. Chapter 08 cites `pct_start=0.1`
  from this one.
- **Loshchilov, Ilya, and Frank Hutter. "SGDR: Stochastic
  Gradient Descent with Warm Restarts" (ICLR 2017).**
  <https://arxiv.org/abs/1608.03983>
  The paper behind `CosineAnnealingLR` and its warm-restart
  variant.
- **Goyal, Priya, et al. "Accurate, Large Minibatch SGD:
  Training ImageNet in 1 Hour" (2017).**
  <https://arxiv.org/abs/1706.02677>
  The paper that introduced the "linear scaling rule" and
  the linear-warmup schedule. The warmup mechanic every
  Transformer training run uses is descended from here.
- **Zhang, Hongyi, et al. "mixup: Beyond Empirical Risk
  Minimization" (ICLR 2018).**
  <https://arxiv.org/abs/1710.09412>
  The MixUp paper. Exercise-02's stretch goal implements
  it.
- **Cubuk, Ekin D., et al. "RandAugment: Practical
  automated data augmentation with a reduced search space"
  (CVPR 2020 workshops).**
  <https://arxiv.org/abs/1909.13719>
  The RandAugment paper. The augmentation
  `torchvision.transforms.v2.RandAugment` implements.
- **Szegedy, Christian, et al. "Rethinking the Inception
  Architecture for Computer Vision" (CVPR 2016).**
  <https://arxiv.org/abs/1512.00567>
  Section 7 introduces label smoothing — the
  `label_smoothing=0.1` argument in exercise-02.

## Chapter 09 — Mixed precision and training performance

- **PyTorch — automatic mixed precision (AMP).**
  <https://pytorch.org/docs/stable/amp.html>
  `torch.autocast`, `GradScaler`, and the op-list of which
  ops are cast to fp16 / bf16 automatically.
- **PyTorch — AMP recipe.**
  <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
  The step-by-step tutorial that chapter 09's code is based
  on.
- **PyTorch — `torch.compile` documentation.**
  <https://pytorch.org/docs/stable/torch.compiler.html>
  The compilation stack introduced in PyTorch 2.0. Chapter
  09's speedup discussion draws on this.
- **PyTorch — profiler.**
  <https://pytorch.org/docs/stable/profiler.html>
  The `torch.profiler` reference and TensorBoard viewer.
- **PyTorch — gradient checkpointing
  (`torch.utils.checkpoint`).**
  <https://pytorch.org/docs/stable/checkpoint.html>
  The memory-vs-compute trade-off for large models.
- **Micikevicius, Paulius, et al. "Mixed Precision Training"
  (ICLR 2018).**
  <https://arxiv.org/abs/1710.03740>
  The paper behind AMP. Sections 3–4 explain the loss-
  scaling trick that `GradScaler` implements for fp16.
- **NVIDIA — Training with Mixed Precision User Guide.**
  <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>
  The vendor reference. The best explanation of when to use
  fp16 + `GradScaler` vs. bf16 without.
- **Kalamkar, Dhiraj, et al. "A Study of BFLOAT16 for Deep
  Learning Training" (2019).**
  <https://arxiv.org/abs/1905.12322>
  The paper behind chapter 09's "prefer bf16 when the
  hardware supports it" advice.
- **Rasley, Jeff, et al. "DeepSpeed: System Optimizations
  Enable Training Deep Learning Models with Over 100 Billion
  Parameters" (KDD 2020).**
  <https://dl.acm.org/doi/10.1145/3394486.3406703>
  The DeepSpeed / ZeRO paper. Beyond this module's scope
  (delegated to distributed-training track), but worth
  knowing the name for chapter 09's "what to reach for
  next".

## Chapter 10 — Transformer LLMs and when to delegate

Note: LLM-family model cards move fast. The links below are
stable references (papers, official docs). Model-name-
specific links in this chapter and in exercise-04 may need
to be refreshed on future authoring cycles.

### Foundational LLM papers

- **Radford, Alec, et al. "Language Models are Unsupervised
  Multitask Learners" (2019).**
  <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
  The GPT-2 paper. The clearest early statement of what a
  large language model *is*.
- **Brown, Tom B., et al. "Language Models are Few-Shot
  Learners" (NeurIPS 2020).**
  <https://arxiv.org/abs/2005.14165>
  The GPT-3 paper. Section 2 (approach) is the source for
  the "in-context learning" / few-shot vocabulary.
- **Hoffmann, Jordan, et al. "Training Compute-Optimal Large
  Language Models" (NeurIPS 2022).**
  <https://arxiv.org/abs/2203.15556>
  The Chinchilla paper. The scaling-law argument for how to
  trade off parameters and training tokens.
- **Kaplan, Jared, et al. "Scaling Laws for Neural Language
  Models" (2020).**
  <https://arxiv.org/abs/2001.08361>
  The earlier scaling-law paper. Chinchilla's predecessor;
  read together.
- **Touvron, Hugo, et al. "LLaMA: Open and Efficient
  Foundation Language Models" (2023).**
  <https://arxiv.org/abs/2302.13971>
  The original Llama paper. The architectural choices
  (RoPE, SwiGLU, RMSNorm) that became the modern default.
- **Meta AI. "The Llama 3 Herd of Models" (2024).**
  <https://arxiv.org/abs/2407.21783>
  The Llama 3 paper. A worked example of a modern open-
  weights training recipe end to end — a useful "recipe" to
  read for exercise-04.
- **DeepSeek-AI. "DeepSeek-V3 Technical Report" (2024).**
  <https://arxiv.org/abs/2412.19437>
  A detailed public account of a large MoE model. The MoE
  discussion in chapter 10 draws on this.
- **DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning
  Capability in LLMs via Reinforcement Learning" (2025).**
  <https://arxiv.org/abs/2501.12948>
  The reasoning-model recipe.

### Post-training and alignment

- **Ouyang, Long, et al. "Training language models to follow
  instructions with human feedback" (NeurIPS 2022).**
  <https://arxiv.org/abs/2203.02155>
  The InstructGPT paper. The canonical RLHF reference.
- **Bai, Yuntao, et al. "Training a Helpful and Harmless
  Assistant with Reinforcement Learning from Human Feedback"
  (2022).**
  <https://arxiv.org/abs/2204.05862>
  Anthropic's account of RLHF. Complements Ouyang et al.
- **Rafailov, Rafael, et al. "Direct Preference Optimization:
  Your Language Model is Secretly a Reward Model" (NeurIPS
  2023).**
  <https://arxiv.org/abs/2305.18290>
  The DPO paper. The alternative to RLHF that chapter 10
  mentions.
- **Bai, Yuntao, et al. "Constitutional AI: Harmlessness
  from AI Feedback" (2022).**
  <https://arxiv.org/abs/2212.08073>
  The Constitutional AI paper. A distinct approach to
  alignment worth knowing.

### Fine-tuning and PEFT

- **Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of
  Large Language Models" (ICLR 2022).**
  <https://arxiv.org/abs/2106.09685>
  The LoRA paper. Every PEFT tutorial descends from this.
- **Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of
  Quantized LLMs" (NeurIPS 2023).**
  <https://arxiv.org/abs/2305.14314>
  The QLoRA paper. The recipe that lets you fine-tune 65B
  models on a single 24 GB GPU.
- **Hugging Face — `peft` library.**
  <https://huggingface.co/docs/peft/>
  The practical entry point to LoRA / QLoRA / adapter tuning
  on Hugging Face models.
- **Hugging Face — TRL (Transformer Reinforcement Learning).**
  <https://huggingface.co/docs/trl/>
  The library for SFT, DPO, PPO, GRPO on Hugging Face
  models. The applied companion to the RLHF / DPO papers
  above.

### Retrieval, embedding, and RAG

- **Lewis, Patrick, et al. "Retrieval-Augmented Generation
  for Knowledge-Intensive NLP Tasks" (NeurIPS 2020).**
  <https://arxiv.org/abs/2005.11401>
  The RAG paper. The name and the pattern that chapter 10
  uses.
- **Karpukhin, Vladimir, et al. "Dense Passage Retrieval for
  Open-Domain Question Answering" (EMNLP 2020).**
  <https://arxiv.org/abs/2004.04906>
  The DPR paper. The embedding-based retriever that most
  modern RAG systems descend from.
- **Reimers, Nils, and Iryna Gurevych. "Sentence-BERT:
  Sentence Embeddings using Siamese BERT-Networks" (EMNLP
  2019).**
  <https://arxiv.org/abs/1908.10084>
  The Sentence-BERT paper. The embedding model behind most
  open-source RAG systems for years.
- **Douze, Matthijs, et al. "The Faiss library" (2024).**
  <https://arxiv.org/abs/2401.08281>
  The FAISS paper. The most-used vector-search library.

### Serving and inference

- **Kwon, Woosuk, et al. "Efficient Memory Management for
  Large Language Model Serving with PagedAttention" (SOSP
  2023).**
  <https://arxiv.org/abs/2309.06180>
  The vLLM / PagedAttention paper. The KV-cache management
  chapter 10 mentions.
- **vLLM — documentation.**
  <https://docs.vllm.ai/>
- **SGLang — documentation.**
  <https://sgl-project.github.io/>
  Two of the modern open-source LLM serving stacks.
- **Frantar, Elias, et al. "GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers"
  (ICLR 2023).**
  <https://arxiv.org/abs/2210.17323>
  The GPTQ paper. INT4 quantization for LLMs.
- **Lin, Ji, et al. "AWQ: Activation-aware Weight
  Quantization for LLM Compression and Acceleration" (MLSys
  2024).**
  <https://arxiv.org/abs/2306.00978>
  The AWQ paper. The other common INT4 quantization
  approach.

### Evaluation

- **Hendrycks, Dan, et al. "Measuring Massive Multitask
  Language Understanding" (ICLR 2021).**
  <https://arxiv.org/abs/2009.03300>
  The MMLU paper.
- **Cobbe, Karl, et al. "Training Verifiers to Solve Math
  Word Problems" (2021).**
  <https://arxiv.org/abs/2110.14168>
  The GSM8K paper.
- **Chen, Mark, et al. "Evaluating Large Language Models
  Trained on Code" (2021).**
  <https://arxiv.org/abs/2107.03374>
  The HumanEval paper.
- **Zheng, Lianmin, et al. "Judging LLM-as-a-Judge with
  MT-Bench and Chatbot Arena" (NeurIPS 2023).**
  <https://arxiv.org/abs/2306.05685>
  MT-Bench and Chatbot Arena — the "human preference" side
  of LLM evaluation.
- **LMSYS Chatbot Arena — leaderboard.**
  <https://lmarena.ai/>
  The live human-preference leaderboard. Referenced in
  chapter 10.

### Vendor / provider references (for exercise-04 model cards)

Cards move over time. These are the stable landing pages;
individual model cards are linked from them.

- **Meta — Llama model families.**
  <https://www.llama.com/>
  <https://ai.meta.com/llama/>
- **Mistral AI — models.**
  <https://mistral.ai/models>
- **Alibaba — Qwen models on Hugging Face.**
  <https://huggingface.co/Qwen>
- **DeepSeek — models on Hugging Face.**
  <https://huggingface.co/deepseek-ai>
- **Google — Gemma models.**
  <https://ai.google.dev/gemma>
- **Google — Gemini API documentation.**
  <https://ai.google.dev/gemini-api/docs>
- **Anthropic — Claude documentation.**
  <https://docs.anthropic.com/>
- **Anthropic — Claude model cards / system cards.**
  <https://www.anthropic.com/claude>
- **OpenAI — API documentation and model index.**
  <https://platform.openai.com/docs/models>
- **OpenAI — GPT-OSS open-weights models.**
  <https://huggingface.co/openai>
- **Hugging Face — Open LLM Leaderboard.**
  <https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard>

## Datasets used in the exercises

- **CIFAR-10 and CIFAR-100.**
  <https://www.cs.toronto.edu/~kriz/cifar.html>
  Krizhevsky, Alex. "Learning Multiple Layers of Features
  from Tiny Images" (Tech report, University of Toronto,
  2009). <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>.
  Loadable via `torchvision.datasets.CIFAR10` /
  `CIFAR100`. Default for exercise-02.
- **AG News.**
  Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level
  Convolutional Networks for Text Classification" (NeurIPS
  2015). <https://arxiv.org/abs/1509.01626>. Loadable via
  `datasets.load_dataset("ag_news")`.
- **IMDb.**
  Maas, Andrew L., et al. "Learning Word Vectors for
  Sentiment Analysis" (ACL 2011).
  <https://aclanthology.org/P11-1015/>. Loadable via
  `datasets.load_dataset("imdb")`.
- **DBpedia-14.**
  Same Zhang et al. NeurIPS 2015 paper as AG News.
  Loadable via `datasets.load_dataset("dbpedia_14")`.
- **TREC.**
  Li, Xin, and Dan Roth. "Learning Question Classifiers"
  (COLING 2002). <https://aclanthology.org/C02-1150/>.
  Loadable via `datasets.load_dataset("trec")`.
- **MNIST and Fashion-MNIST** (worked examples in chapter
  05).
  <http://yann.lecun.com/exdb/mnist/>
  <https://github.com/zalandoresearch/fashion-mnist>
  Loadable via `torchvision.datasets.MNIST` and
  `FashionMNIST`.

<!-- needs-research: on each autonomous research cycle, verify these URLs still resolve. In particular: (a) PyTorch major-version pages (2.x → 3.x will move); (b) Hugging Face model-family landing pages; (c) vendor model cards (Llama 3.x → 4.x, Claude 3.x/4.x → next major, DeepSeek Vn → Vn+1, Qwen n → n+1). Swap in current-generation links when a vendor promotes a new flagship. -->
