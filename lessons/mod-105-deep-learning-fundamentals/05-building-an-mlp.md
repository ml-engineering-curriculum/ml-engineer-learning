# 05 — Building an MLP

> **Reading time:** ~30 minutes.
> **Prerequisite chapters:** 02, 03, 04.

## Motivation

The multi-layer perceptron (MLP) is the simplest neural
network with more than one layer, and it is where every
`nn.Module` idiom you will use for the rest of the module
gets introduced. Once you can write an MLP, a CNN is a
subclass with different layers, and a Transformer is another
one on top of that.

MLPs are also — importantly — the correct model for a
narrower set of problems than a beginner might think.
Chapter 01's caveat applies here in force: **on clean tabular
data, a gradient-boosted tree from `mod-104` will usually win.**
Build the tree baseline before you build the MLP.

Where MLPs earn their place:

- As the final "head" on top of a CNN or a Transformer
  (chapters 06, 07).
- On tabular problems with hundreds of features and
  hundreds of thousands of rows, especially when you have
  categorical embeddings you want the network to learn.
- As the vehicle for internalising the mechanics of
  `nn.Module`, layers, activations, and initialisation before
  the harder architectures land.

## The anatomy of an `nn.Module`

Every model in this module is an `nn.Module`. The recipe:

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
```

Three rules:

1. **`super().__init__()` first.** Without it, `nn.Module`'s
   internal state isn't initialised and parameter registration
   silently fails.
2. **Assign layers to `self.<attr>`.** Any `nn.Module` (or
   `nn.Parameter`) so assigned is registered — it appears in
   `.parameters()`, `.state_dict()`, and `.to(device)`.
3. **Anything variable-length goes in `nn.ModuleList` or
   `nn.Sequential`.** A plain Python list of modules does
   **not** register — parameters vanish.

```python
# WRONG — params not registered:
self.layers = [nn.Linear(10, 10), nn.Linear(10, 10)]

# RIGHT:
self.layers = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10)])
# or
self.layers = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
```

## Sequential vs custom

For linear stacks, `nn.Sequential` is fine:

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 3),
)
```

For anything with branching, skip connections, or a
non-trivial forward (multiple outputs, dict outputs), write a
custom `Module`. The `MLP` above is a custom `Module` even
though it is linear — the readability trade-off tips toward
`Sequential` when the depth is short and there is no state
you want to name.

## Activation functions

The nonlinearity between layers is what makes a network more
than a linear transformation. The safe defaults:

- **`ReLU`** (`x if x > 0 else 0`). Cheap, gradient-friendly
  (constant 1 for positives), the historical default. Suffers
  from "dying ReLU" — a neuron whose inputs push it into the
  negative region gets zero gradient and never recovers.
- **`GELU`** (Gaussian Error Linear Unit). Smoother than
  ReLU; the default in Transformers and increasingly common
  elsewhere. `nn.GELU()`.
- **`SiLU` / `Swish`** (`x * sigmoid(x)`). Smooth, non-
  monotonic, popular in modern vision and language models.
  `nn.SiLU()`.
- **`LeakyReLU(negative_slope=0.01)`** — variant of ReLU that
  passes a small gradient for negative inputs. Fixes dying
  ReLU at low cost.

Two you will see and one to avoid:

- **`Sigmoid`** — only for binary output layers (and then,
  fold into `BCEWithLogitsLoss` — do not apply it manually).
- **`Tanh`** — legacy; still shows up in RNNs and GAN
  discriminators. Fine but rarely the first choice today.
- **`Softmax`** as an activation between layers — no. Softmax
  is a normalisation for output logits; it kills gradients
  when used mid-network.

For MLPs in this module: **use `ReLU` or `GELU`.** They will
not be the reason your model fails to train.

## Initialisation

Random initialisation matters. A network initialised with
values that are too small produces vanishing activations; too
large produces exploding gradients. PyTorch's defaults for
`nn.Linear` — `kaiming_uniform_` — are sensible for ReLU-ish
networks, but for a Transformer or an unusual activation you
may want to override.

The two initialisations you should know:

- **Kaiming (He)** — scaled for ReLU-family activations.
  Preserves variance of activations across layers.
  `nn.init.kaiming_normal_(w, nonlinearity="relu")`.
- **Xavier (Glorot)** — scaled for tanh/sigmoid.
  `nn.init.xavier_normal_(w)`.

To override in a custom `Module`:

```python
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# call it once in __init__ after building the layers
```

For chapter 05's MLPs on tabular data, PyTorch's defaults
work. For the Transformer in chapter 07, initialisation
matters more — that chapter revisits it.

## Depth and width

An MLP has two dials: the number of layers (depth) and the
number of hidden units per layer (width).

Some empirical rules that hold in practice:

- **Start narrow, add depth.** A 2-layer MLP with 128 hidden
  units is a fine baseline. If it under-fits, go 3–4 layers
  before you widen.
- **Wider is easier to optimise; deeper is easier to
  generalise.** Both have limits. Beyond ~6 fully connected
  layers, gains taper without residual connections.
- **Residual connections** (`x + f(x)`) let you go deeper.
  ResNet's contribution to vision was this trick;
  Transformers use it in every block. Chapter 06 and 07 make
  it concrete.

Sizes to try first on a tabular problem: `[128, 128]`,
`[256, 128]`, `[128, 64, 32]`. Grid over 3–5 of these; do
not build a Bayesian-optimised search for a first pass.

## Output layers by task

The final layer's shape and the loss are coupled. Match them
carefully.

| Task | Final layer | Output size | Loss |
|---|---|---|---|
| Regression (scalar) | `Linear(H, 1)` | `(B, 1)` | `MSELoss` / `SmoothL1Loss` / `L1Loss` |
| Multi-output regression | `Linear(H, K)` | `(B, K)` | `MSELoss` |
| Binary classification | `Linear(H, 1)` | `(B, 1)` | `BCEWithLogitsLoss` |
| Multi-class classification | `Linear(H, C)` | `(B, C)` | `CrossEntropyLoss` |
| Multi-label classification | `Linear(H, C)` | `(B, C)` | `BCEWithLogitsLoss` |

**Output raw logits, always.** Any `Sigmoid` / `Softmax`
belongs inside the loss (`BCEWithLogitsLoss`,
`CrossEntropyLoss`) for numerical stability. Applying the
activation manually and then using `BCELoss` / `NLLLoss`
works but is more error-prone and loses precision.

## Preprocessing tabular data for an MLP

MLPs are sensitive to feature scale in a way trees are not.
Do this before training:

- **Numerical features** — `StandardScaler` (mean 0, unit
  variance). Fit on train, transform on val/test.
- **Categorical features** — either one-hot (chapter 07 of
  `mod-103`) if cardinality is low, or an **embedding
  lookup** if high.

Embeddings are worth spelling out because they are your
first taste of "learn a representation":

```python
class TabularMLP(nn.Module):
    def __init__(self, n_numeric: int, cat_cardinalities: list[int], hidden: int, out: int):
        super().__init__()
        emb_dims = [(c, min(50, (c + 1) // 2)) for c in cat_cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in emb_dims])
        emb_total = sum(d for _, d in emb_dims)

        self.body = nn.Sequential(
            nn.Linear(n_numeric + emb_total, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num: (B, n_numeric); x_cat: (B, n_cat) int64
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x    = torch.cat([x_num, *embs], dim=1)
        return self.body(x)
```

Notes on that:

- `nn.Embedding(num_embeddings, embedding_dim)` is a lookup
  table with `num_embeddings × embedding_dim` learnable
  parameters. Backprop updates the rows that appeared in the
  batch.
- The `(c + 1) // 2` heuristic for embedding dimension is a
  reasonable starting point (fastai's default rule). Common
  ranges: 4–50 dims. Tune if the model plateaus.
- Reserve index 0 for "unknown" / "missing" so unseen
  categories at inference time have a bucket.

## A worked end-to-end example

A full training run on a small tabular problem, wiring up
chapters 02–04 with the MLP from this chapter. This is the
shape every subsequent chapter's worked example follows.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Data — the classic 'adult' income dataset for a quick demo.
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
X = X.select_dtypes(include="number").fillna(0.0)          # keep it simple
y = (y == ">50K").astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
scaler = StandardScaler().fit(X_tr)
X_tr = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
X_te = torch.tensor(scaler.transform(X_te), dtype=torch.float32)
y_tr = torch.tensor(y_tr.values, dtype=torch.long)
y_te = torch.tensor(y_te.values, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_te, y_te), batch_size=512)

# 2. Model
class MLP(nn.Module):
    def __init__(self, in_features, hidden, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden),      nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out),
        )
    def forward(self, x): return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
model = MLP(in_features=X_tr.shape[1], hidden=128, out=2).to(device)

# 3. Loss + optimizer
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 4. Training loop (from chapter 03)
for epoch in range(10):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    model.eval()
    correct, val_loss = 0, 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            val_loss += loss_fn(logits, yb).item() * xb.size(0)
            correct  += (logits.argmax(dim=1) == yb).sum().item()

    print(
        f"epoch {epoch} | "
        f"train {train_loss/len(train_loader.dataset):.4f} | "
        f"val {val_loss/len(val_loader.dataset):.4f} | "
        f"acc {correct/len(val_loader.dataset):.4f}"
    )
```

Compare against a `mod-104` baseline (`HistGradientBoostingClassifier`
or `LGBMClassifier`) on the same split. On this dataset with
only numeric features, the tree will likely edge out the MLP
— which is the point. **Do this comparison every time.**

## Counting parameters

An MLP with input 10, hidden 128, output 3:

- `fc1`: 10 × 128 weights + 128 biases = 1,408
- `fc2`: 128 × 3 weights + 3 biases = 387
- **Total: 1,795 parameters.**

Count them in code:

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))
```

Handy for two things: sanity-checking a Module is the size you
expect, and comparing model sizes when you tune width.

## Debugging an MLP that won't learn

If loss doesn't drop:

1. **Overfit a batch of 8** (chapter 03). If that fails, the
   loop is wrong. If it succeeds, the data is the problem.
2. **Check input shapes and dtypes at the boundary.** Assert
   in `forward`.
3. **Check that all weights have gradients after one step.**

   ```python
   for name, p in model.named_parameters():
       print(name, p.grad is not None, p.grad.norm().item() if p.grad is not None else None)
   ```

   Any parameter with `grad is None` after `.backward()` is
   disconnected from the loss.
4. **Learning rate.** Try 1e-2, 1e-3, 1e-4, 1e-5 — a decade
   apart. One of them will move the loss.
5. **Loss is on the right scale.** For classification,
   uniform predictions give loss ≈ `ln(C)`. If your loss is
   two orders of magnitude bigger, something is wrong with
   the label shape.

## Pitfalls collected

1. **Registering layers in a Python list.** Parameters
   vanish. `nn.ModuleList` or `nn.Sequential`.
2. **Forgetting `super().__init__()`.** Registration silently
   fails; parameters may not appear.
3. **Softmax mid-network.** Kills gradients. Softmax is for
   output normalisation only, and even then belongs inside
   `CrossEntropyLoss`.
4. **Unscaled numeric features into an MLP.** Trains slowly
   or not at all. Standardise.
5. **Building an MLP for 500 rows of tabular data.** A tree
   will train in 0.5 seconds and be better. Build the tree
   first (chapter 01).
6. **Embedding dimension too large.** Ten-thousand-dim
   embeddings for a 5-category column overfit. Keep it modest
   — 4 to 50, informed by cardinality.

## Summary

- An `nn.Module` is a container of layers with automatic
  parameter registration; `nn.Sequential` handles linear
  stacks, custom `Module` handles anything else.
- Activations: default to `ReLU` or `GELU`. Sigmoid and
  softmax stay in the loss for numerical stability.
- Output layer shape and loss are coupled — memorise the
  table above.
- MLPs on tabular data need scaled numeric features and
  either one-hot or learned-embedding categoricals.
- Build the `mod-104` tree baseline first. If the MLP does
  not beat it, use the tree.

## Where this goes next

Chapter 06 introduces convolutions — the layers that let a
network see spatial structure in images without the parameter
explosion of a fully-connected model.
