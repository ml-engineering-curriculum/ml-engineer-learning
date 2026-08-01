# 04 — Datasets, DataLoaders, and collate functions

> **Reading time:** ~30 minutes.
> **Prerequisite chapter:** 03.

## Motivation

The training loop from chapter 03 assumed data would appear
in the shape `(xb, yb)` inside its `for` loop. This chapter
covers the other half: the `Dataset` you read from and the
`DataLoader` that batches, shuffles, parallelises, and
delivers examples to the loop.

Get this right and training scales cleanly from a laptop
prototype to a multi-GPU cluster with no changes to the loop.
Get it wrong and you will spend hours diagnosing why the GPU
is at 15% utilisation — because the data pipeline is the
bottleneck, not the model.

## The two Dataset kinds

PyTorch has two `Dataset` protocols. Almost every deep
learning project you will write uses the first; the second
exists for streaming.

### `torch.utils.data.Dataset` (map-style)

A map-style dataset implements two methods:

- `__len__(self) -> int`
- `__getitem__(self, idx: int) -> sample`

It behaves like a list of examples. The `DataLoader` picks
indices, calls `__getitem__` for each, and stacks the results
into a batch.

Minimal example:

```python
from torch.utils.data import Dataset
import pandas as pd
import torch

class TitanicDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, label_col):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[label_col].values,    dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
```

Precomputing into two tensors in `__init__` is fine for
small tabular data. For anything larger — images on disk,
text that needs tokenisation — do the work lazily in
`__getitem__` so you don't materialise the whole dataset in
memory.

Lazy version (images):

```python
from PIL import Image
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)                 # (3, H, W)
        return img, self.labels[idx]
```

`__getitem__` runs in the worker processes (see below), so
IO cost is parallelised.

### `torch.utils.data.IterableDataset` (streaming)

An `IterableDataset` implements `__iter__` and yields samples.
Use when the dataset is too big to enumerate up front (a Kafka
topic, an on-the-fly synthetic generator, a very large
sharded corpus with no index).

The tricky part is multi-worker duplication: without extra
work, each worker will yield the whole stream, so batches
duplicate. Use `torch.utils.data.get_worker_info()` to shard
the iteration inside your `__iter__`:

```python
class MyStream(torch.utils.data.IterableDataset):
    def __init__(self, shards):
        self.shards = shards
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        my_shards = (self.shards if info is None
                     else self.shards[info.id :: info.num_workers])
        for shard in my_shards:
            for record in read(shard):
                yield record
```

Prefer map-style for anything under a few million examples;
`IterableDataset` for stream-oriented pipelines.

## Splits

Two idioms:

**By index** (deterministic; the default when you want
reproducibility):

```python
from torch.utils.data import Subset
train_ds = Subset(dataset, train_indices)
val_ds   = Subset(dataset, val_indices)
```

**By random split** (a shortcut; still deterministic if you
pass a generator with a fixed seed):

```python
from torch.utils.data import random_split
generator = torch.Generator().manual_seed(0)
train_ds, val_ds = random_split(dataset, [0.8, 0.2], generator=generator)
```

Do the split **once**, at the start of the run, and pass the
subsets into two DataLoaders. Splitting inside the loop is a
data-leakage bug (validation shifts every epoch).

For time-series or grouped data, do the split with a Pandas /
scikit-learn `GroupShuffleSplit` on the raw indices *before*
constructing the Dataset. See `mod-104` chapter 07.

## DataLoader: the batching engine

The `DataLoader` wraps a `Dataset` and yields batches:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
```

Every argument here matters. In order:

- **`batch_size`.** How many samples per gradient step. Larger
  is faster per epoch but changes the training dynamics (see
  chapter 08). Powers of two are conventional (32, 64, 128,
  256) but not required.
- **`shuffle=True`** on training only. Reshuffles indices at
  the start of each epoch. On validation and test:
  `shuffle=False` for reproducibility.
- **`num_workers`.** The number of *subprocesses* used to
  call `__getitem__`. `0` (default) means the main process
  does it — synchronous, slow, but easy to debug. `4–8` is a
  good starting point on a modern machine; more if the CPU has
  spare cores and the transforms are heavy. **`num_workers`
  interacts with your Dataset:** if `__getitem__` opens a
  database connection or a file handle, do it lazily inside
  `__getitem__` (per-worker), not in `__init__`, or workers
  will share and corrupt state.
- **`pin_memory=True`** if you use CUDA. Copies batches into
  page-locked memory so the CPU→GPU transfer can overlap with
  compute. No effect on CPU-only.
- **`drop_last=True`** on training if BatchNorm is present or
  the model is sensitive to batch size. Skips the final,
  smaller batch. On validation: `drop_last=False` — you want
  every example counted.
- **`persistent_workers=True`** keeps workers alive between
  epochs. Saves the per-epoch startup cost, which dominates
  for short epochs. Default `False` for backward
  compatibility; you almost always want `True`.

### Batches, on the loop side

The loader yields whatever `__getitem__` returns, batched. If
`__getitem__` returns `(tensor_x, tensor_y)`, the loader
returns `(tensor_x_batch, tensor_y_batch)` with an extra
leading batch dimension. If it returns a dict, the loader
returns a dict of stacked tensors:

```python
def __getitem__(self, idx):
    return {"pixel_values": img, "label": lbl, "id": self.ids[idx]}
```

```python
batch = next(iter(loader))
# {"pixel_values": (B, 3, H, W), "label": (B,), "id": [B items]}
```

Anything not a tensor is returned as a list. Prefer dicts for
anything more complex than `(x, y)` — reading `batch["label"]`
is easier to maintain than `batch[3]`.

## Collate functions

The default collate stacks tensors on the leading dim. That
works when every sample has the same shape. When it doesn't
— variable-length sequences, variable-size images, ragged
records — you need a `collate_fn`.

### Variable-length sequences (padding)

The canonical case: NLP where each example is a sequence of
token IDs and lengths vary.

```python
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    """batch: list of (token_ids: 1-D LongTensor, label: int)"""
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    lengths = torch.tensor([s.size(0) for s in seqs])
    labels  = torch.tensor(labels)
    return padded, lengths, labels
```

Pair with a `padding_mask` in the model (chapter 07) so the
padding tokens contribute neither to attention nor to the
loss:

```python
padding_mask = (padded == 0)                    # (B, L)
# Or, more robustly, using the lengths:
padding_mask = torch.arange(padded.size(1))[None, :] >= lengths[:, None]
```

### Bucketed / grouped batches

If sequence lengths vary widely and padding is dominating
compute, group similar-length samples into the same batch.
Two approaches:

- A `Sampler` (see below) that yields indices in length
  order.
- Sort *within* the collate function each batch.

The `Sampler` approach is more efficient — you can also
enforce a maximum tokens-per-batch budget rather than a fixed
batch size.

### Ragged records (dicts of tensors)

Sometimes samples have optional fields. Write a defensive
collate:

```python
def dict_collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out
```

## Samplers

The `sampler` argument to `DataLoader` controls the order of
indices. Two you will use in practice:

- **`SequentialSampler`** (default when `shuffle=False`).
- **`RandomSampler`** (default when `shuffle=True`). Both are
  produced automatically by the `shuffle` argument.
- **`WeightedRandomSampler`** for class-imbalanced datasets.
  Give each sample a weight inversely proportional to its
  class frequency and the loader will draw balanced batches:

  ```python
  from torch.utils.data import WeightedRandomSampler
  weights = 1.0 / class_counts[labels]        # per-sample
  sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
  loader = DataLoader(dataset, batch_size=64, sampler=sampler)
  ```

  Never pass both `sampler=` and `shuffle=True` — the loader
  raises. With a custom sampler, the sampler owns ordering.

- **`DistributedSampler`** for multi-GPU (DDP) training.
  Shards indices across ranks so each rank sees a distinct
  slice. Out of scope for this module; the interface is worth
  knowing.

For length-bucketed batching or per-class balanced batches,
subclass `Sampler`:

```python
from torch.utils.data import Sampler
class ByLengthSampler(Sampler):
    def __init__(self, lengths, batch_size):
        self.batches = [
            sorted_indices[i : i + batch_size]
            for sorted_indices in [sorted(range(len(lengths)), key=lambda i: lengths[i])]
            for i in range(0, len(sorted_indices), batch_size)
        ]
    def __iter__(self): return iter(self.batches)
    def __len__(self):  return len(self.batches)
```

(For most cases the built-ins are enough. Reach for custom
samplers only when profiling shows a real problem.)

## Workers, GILs, and forking gotchas

Multi-worker DataLoaders `fork` (or `spawn`, on Windows and
macOS by default) subprocesses. Two common traps:

1. **Shared open file handles.** Every worker inherits the
   parent's file descriptors. Opening a database or an HDF5
   file *in `__init__`* corrupts across workers. Open lazily
   inside `__getitem__`, or override `worker_init_fn` to open
   a fresh handle per worker.
2. **Randomness needs seeding per worker.** Without it, every
   worker generates the same "random" augmentations. Fix:

   ```python
   def worker_init_fn(worker_id):
       import numpy as np, random, torch
       seed = torch.initial_seed() % 2**32
       np.random.seed(seed)
       random.seed(seed)

   loader = DataLoader(..., worker_init_fn=worker_init_fn)
   ```

3. **CUDA initialisation in workers.** Do not use CUDA
   tensors inside `__getitem__` unless you know what you're
   doing. Move to GPU inside the training loop.

## Sizing `num_workers` and `batch_size`

Both are hyperparameters you tune to your hardware. Rules of
thumb (all confirmable with `nvidia-smi` and `htop`):

- **If GPU utilisation is below ~90%,** the data pipeline is
  the bottleneck. Try more `num_workers`, larger
  `prefetch_factor`, or moving CPU-heavy work off the hot
  path (precompute embeddings, cache decoded images).
- **If CPU utilisation is at 100% across cores,** you have
  enough workers; you are CPU-bound on transforms. Simplify
  transforms or move them to GPU (`torchvision.transforms.v2`
  supports GPU-side ops).
- **If GPU memory is at 100%,** batch size is the ceiling.
  Chapter 09's gradient-accumulation trick lets you simulate a
  larger effective batch size on the same hardware.
- **`persistent_workers=True`** avoids the per-epoch worker-
  spinup tax; noticeable when epochs are short (< 30 s).

## A concrete tabular pipeline

Putting the pieces together on a small tabular problem:

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

df = pd.read_parquet("titanic.parquet")
feature_cols = ["age", "fare", "pclass", "sibsp", "parch"]
label_col    = "survived"

# 1. Split by index, before constructing the Dataset.
train_idx, val_idx = train_test_split(
    df.index, test_size=0.2, stratify=df[label_col], random_state=0
)

# 2. Dataset over the full frame; Subsets pick the rows.
dataset = TitanicDataset(df, feature_cols, label_col)
train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)

# 3. Two loaders — same batch shape, different shuffling.
train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True,
)
val_loader = DataLoader(
    val_ds, batch_size=256, shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=True,
)

# 4. Plug into chapter 03's training loop.
train(model, train_loader, val_loader, ...)
```

The `TitanicDataset` from earlier is reused unchanged. This
is the shape every chapter's example will follow — the
`Dataset` differs by modality, the loader arguments and the
loop do not.

## A note on `torchvision` / `torchaudio` / `torchtext`

The domain libraries provide ready-made Datasets:

- `torchvision.datasets.CIFAR10(root=..., download=True,
  transform=...)` — chapter 06 uses this.
- `torchaudio.datasets.LIBRISPEECH` — speech.
- Hugging Face `datasets` is the current standard for text —
  it interoperates with PyTorch DataLoaders cleanly and gives
  you streaming, caching, and versioned splits for free.

For any well-known dataset, prefer the domain library over
hand-rolling. It saves boilerplate and matches the schema
that every published model in that domain expects.

## Pitfalls collected

1. **Splitting inside the training loop.** The validation set
   changes every epoch — silent data leakage.
2. **Opening a DB / file handle in `__init__` with
   `num_workers > 0`.** Workers corrupt shared state. Open
   inside `__getitem__` or in `worker_init_fn`.
3. **Un-seeded augmentation across workers.** Every worker
   applies the same "random" transform. Use `worker_init_fn`
   as above.
4. **Forgetting `pin_memory=True` on CUDA.** Slower host-to-
   device transfers.
5. **`drop_last=False` on training with BatchNorm.** The
   final tiny batch skews statistics; loss curves twitch at
   epoch ends.
6. **Same batch size for train and val.** Val can be much
   larger (no gradients, no autograd graph, so less memory).
   Doubling the val batch halves wall-clock validation cost.
7. **Custom `Sampler` and `shuffle=True` together.** Raises.
   Choose one.

## Summary

- `Dataset` is the "list of examples" API. Use map-style for
  anything with a knowable length; `IterableDataset` for
  streams.
- `DataLoader` batches, shuffles, and parallelises. Tune
  `num_workers`, `pin_memory`, `persistent_workers`, and
  `drop_last` to your hardware and task.
- A `collate_fn` handles variable-length inputs — padding,
  bucketing, ragged records. Pair with padding masks in the
  model.
- Samplers control iteration order. `WeightedRandomSampler`
  for imbalance; `DistributedSampler` for multi-GPU.
- The `Dataset` / `DataLoader` interface is what makes the
  training loop from chapter 03 modality-agnostic. Chapters
  05–07 substitute in new Datasets and models; the loop does
  not change.

## Where this goes next

Chapter 05 builds the first of the three canonical
architectures — the multi-layer perceptron — and trains it
on a real tabular problem with everything from chapters
02–04.
