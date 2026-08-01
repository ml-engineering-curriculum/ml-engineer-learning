# 09 — Image feature engineering

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 06.

## Motivation

Image data — the pixels themselves — is not the same object as the
tabular DataFrames of the previous chapters. An image is:

- A 3-D array of pixel intensities (`height × width × channels`)
- Stored as a compressed file (JPEG, PNG, WebP) on disk or in an
  object store
- Interpreted differently by different libraries (RGB vs BGR;
  `uint8 [0, 255]` vs `float32 [0, 1]` vs `float32 [-1, 1]`)
- Rarely modelled as raw pixels — the near-universal pattern is to
  extract features via a pretrained convolutional or transformer
  backbone

This chapter covers the *data engineering* half of image work:
loading, decoding, resizing, normalising, augmenting, and turning a
directory of image files into a feature matrix a shallow model can
consume. End-to-end vision model training — the loss functions, the
optimisers, the architectures — is `mod-105`'s territory.

## Where the images live

Two shapes are common.

1. **Directory of image files.** `data/cats/cat_001.jpg`,
   `data/dogs/dog_042.jpg`. Fine at small scale (thousands of
   files). Slow at large scale because opening many small files is
   the object-store rate-limit disaster from chapter 04.

2. **Sharded archive files.** A `WebDataset` `.tar` shard, a
   TFRecord, an HDF5 file, or a Parquet dataset with an image bytes
   column. Batches of images in one file; reads are sequential and
   fast.

For any dataset larger than a few tens of thousands of images,
convert to shards *once*. PyTorch's `torchdata` and
[`webdataset`](https://github.com/webdataset/webdataset) both
support the sharded pattern; Hugging Face `datasets` handles it
transparently for the datasets it hosts.

A typical `webdataset` shard:

```
train-000000.tar
├── 000000000.jpg
├── 000000000.json     # metadata / label
├── 000000001.jpg
├── 000000001.json
...
```

Readers stream shards from S3 / GCS / local disk and yield
`(image_bytes, metadata)` tuples without ever creating a temp file.

## Loading and decoding

Three Python libraries do the heavy lifting:

- **Pillow (`PIL`)** — the venerable, pure-Python image library.
  `Image.open(path).convert("RGB")`. Slower than the alternatives;
  the most compatible.
- **OpenCV (`cv2`)** — faster than Pillow; defaults to BGR (not
  RGB) channel order, which is a perennial source of bugs.
- **`torchvision.io.read_image`** / **`torch.image`** — returns a
  `torch.uint8` tensor in `(C, H, W)` layout, ready for the rest
  of the PyTorch pipeline.

```python
from PIL import Image
img = Image.open("cat.jpg").convert("RGB")   # PIL Image, uint8, HWC-ish
```

```python
import cv2
bgr = cv2.imread("cat.jpg")                  # ndarray, uint8, HWC, BGR
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # convert to RGB
```

```python
from torchvision.io import read_image
tensor = read_image("cat.jpg")               # torch.uint8, (3, H, W), RGB
```

Pick one library per project and stay in it. Mixing PIL and OpenCV
in the same pipeline is how a "channels swapped" bug ends up in
production.

### The decoding cost

JPEG decoding is not free. On CPU, decoding an ImageNet-scale image
takes single-digit milliseconds. For a training loop that iterates
millions of images per epoch, the decode is often the bottleneck,
not the model. Two mitigations:

- **Decode on the GPU** with NVIDIA's `nvJPEG` (via DALI or
  `torchvision.io.decode_jpeg(device="cuda")`).
- **Preprocess once, store decoded and resized.** A Parquet
  dataset of `(image_bytes_resized_224x224, label)` reads much
  faster than raw JPEGs on every epoch.

## Resizing, cropping, padding

Nearly every model wants a fixed input size. Common canonical sizes:

- **224 × 224** — ImageNet default, most CNNs and ViTs.
- **256 → 224** — resize to 256 on shorter side, then centre-crop
  224. The historical ResNet convention.
- **384 × 384**, **512 × 512** — larger transformer variants.
- **1024 × 1024+** — modern segmentation and detection models.

Two operations to know:

- **`Resize`** — squash to the target size. Distorts aspect ratio
  if width ≠ height. Rarely what you want.
- **`Resize + CenterCrop`** — resize to preserve aspect ratio
  (shorter side → target), then centre-crop the target square.
  The standard classification recipe.
- **`Resize + Pad`** — resize preserving aspect ratio, then pad
  with a solid colour to reach the target. Preserves the entire
  image; used in detection.

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),   # ImageNet stats
])
```

The `Normalize` mean and standard deviation values are the
per-channel ImageNet statistics. When using a pretrained model, you
must normalise with the *same* statistics the model was trained
with. Read the model's model card; different backbones use
different values (in particular, CLIP and DINO variants do not use
ImageNet stats).

## Data augmentation, briefly

Augmentation randomly modifies training images each epoch to
increase effective data volume and regularise the model. Common
transforms:

- **Horizontal flip.** `RandomHorizontalFlip(p=0.5)`. Almost
  always safe except when direction is meaningful (medical
  images, some text-in-image cases).
- **Random crop.** `RandomResizedCrop(224)`. The default for
  ImageNet-style training. Take a random crop of a random size
  and resize.
- **Colour jitter.** `ColorJitter(brightness=0.4, contrast=0.4,
  saturation=0.4, hue=0.1)`. Robust to lighting variation.
- **Random erasing.** `RandomErasing(p=0.5)`. Cutout / dropout in
  pixel space.
- **Mixup / CutMix.** Blend two training images and their labels.
  Not available in vanilla torchvision transforms; implemented in
  `timm` and in most modern training libraries.

Two hard rules:

1. **Augmentation runs at training time only.** The validation and
   test transform pipelines *do not* include random augmentation.
   Otherwise your metric will vary run-to-run for no reason. The
   test pipeline is deterministic: `Resize(256) → CenterCrop(224)
   → ToTensor → Normalize`.
2. **Augmentation should preserve the label.** Rotating a "6" to a
   "9" is not augmentation; it is a labelling bug. Think about
   whether each transform makes semantic sense for the task.

## Serving-time preprocessing must match training

The train/serve skew problem from chapter 06 hits especially hard
in vision because there are so many parameters that can drift:

- **Input size.** Off-by-one in the crop.
- **Channel order.** RGB in training, BGR at serving.
- **Normalisation stats.** ImageNet stats at training, some other
  stats at serving.
- **Dtype.** `uint8 [0, 255]` in training, `float32 [0, 1]` at
  serving.
- **Colour space.** sRGB in training, linear at serving.
- **Interpolation method.** `bilinear` in training, `nearest` at
  serving.

Every one of these has caused a production incident somewhere.
Mitigation: **the exact same transform pipeline is invoked at
training and serving time,** loaded from the same code path.
`torchvision.transforms.Compose` objects can be pickled and
serialised alongside the model.

## Feature extraction with a pretrained backbone

For most non-research vision tasks, the winning workflow is:

1. Pass every image through a pretrained backbone (ResNet-50,
   ViT-B/16, CLIP, DINOv2 — whatever fits your latency budget).
2. Take the penultimate layer as a feature vector (typically
   512 – 2048 dim).
3. Train a shallow model (`LogisticRegression`, gradient booster,
   small MLP) on those features.

This is orders of magnitude cheaper than fine-tuning the backbone
and often produces competitive accuracy on small-to-medium datasets.

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval().to("cuda")
model.fc = torch.nn.Identity()                # keep the 2048-dim penultimate

preprocess = weights.transforms()             # returns the correct transform pipeline

with torch.no_grad():
    features = []
    for batch in dataloader:
        batch = batch.to("cuda")
        features.append(model(batch).cpu().numpy())
features = np.concatenate(features, axis=0)   # (n_images, 2048)
```

Notes:

- `weights.transforms()` returns the exact preprocessing the
  pretrained weights expect — resize / crop / normalise. Use it
  directly rather than hand-rolling. Removes the class of skew
  bugs above.
- `model.fc = torch.nn.Identity()` replaces the classification
  head with a no-op so the forward pass returns the penultimate
  activation.
- `torch.no_grad()` avoids autograd overhead during pure
  inference — cuts memory and speeds things up.

For open-vocabulary text-and-image tasks (search, zero-shot
classification), **CLIP** gives you image and text encoders that
share an embedding space. `open_clip` is the mature open-source
implementation.

For general-purpose visual features with no supervision required,
**DINOv2** (Meta) is a strong choice; the model card lists the
expected preprocessing.

## Caching the features

Feature extraction is expensive; the shallow model on top is cheap.
The winning storage pattern:

```
features/
  ds=train/
    part-0000.parquet   # columns: image_id, feature_vector (list<float32>[2048])
    part-0001.parquet
    ...
  ds=val/
    part-0000.parquet
```

Once the features exist as Parquet, iterating on the shallow model
(feature selection, hyperparameter search, calibration) is a
seconds-per-experiment loop instead of hours-per-experiment.

## Metadata as features

Images almost always come with metadata: capture time, camera
model, EXIF GPS, upload user, source URL. These often carry as
much signal as the pixels themselves for tasks like content
moderation or spam detection. Treat them as tabular features
(chapter 07) and concatenate with the image embeddings.

```python
X = np.hstack([image_features, tabular_features])
```

## Class imbalance

Image datasets are often heavily imbalanced (many photos of cats,
few of dogs). Two responses:

- **Weighted sampling at training time.** `WeightedRandomSampler`
  in PyTorch, `class_weight="balanced"` in scikit-learn's linear
  models.
- **Data augmentation weighted toward the minority class.** More
  augmented variants of rare-class examples per epoch.

For evaluation, use metrics robust to imbalance (per-class F1,
macro-averaged accuracy, PR-AUC for binary problems). `mod-107`
covers this in depth.

## Pitfalls collected

1. **Channel order.** OpenCV reads BGR; PIL and torchvision use
   RGB. Feeding BGR to a model trained on RGB produces
   colour-inverted predictions that "sort of work". Assert on the
   channel order at pipeline boundaries.
2. **Wrong normalisation stats.** ImageNet stats for a CLIP model,
   or vice versa. Always use the transform helper the model's
   weights object provides.
3. **Applying augmentation to the validation set.** The metric
   becomes noisy for no good reason. Train transforms and
   eval transforms are different `Compose` objects.
4. **Reading many small images from S3.** Rate-limit disaster.
   Shard and read shards sequentially.
5. **Not using `torch.no_grad()` during feature extraction.**
   Silent waste of memory and time.
6. **Assuming pretrained features generalise.** Domain-shifted
   images (medical, satellite, industrial) may need a specialised
   backbone or fine-tuning. Measure before assuming.
7. **Casting between `uint8` and `float32` without dividing by
   255.** A `float32` tensor of pixel values `[0, 255]` looks like
   an image but has drastically different normalisation behaviour
   than one in `[0, 1]`.

## Summary

- Image data engineering is: load / decode → resize / crop /
  normalise → optionally augment → feed to model or feature
  extractor.
- Shard for scale (>10k images). Cache decoded/resized bytes to
  skip repeated JPEG decode.
- Use the model's own recommended preprocessing transform
  (`weights.transforms()` in torchvision) — this eliminates the
  most common source of train/serve skew.
- Feature extraction with a pretrained backbone + shallow model is
  the pareto-optimal starting point for small-to-medium image
  tasks. Cache the feature vectors as Parquet.
- Augmentation is a *training-only* transform. Validation and test
  pipelines are deterministic.
- Combine image embeddings with tabular metadata for the strongest
  baselines on real-world problems.

## Where this goes next

The final chapter, 10, turns everything from chapters 05–09 into
automated *dataset validation*: schema checks with `pandera`,
distribution checks with Great Expectations, and how to wire them
into a training pipeline so a bad dataset never silently trains a
bad model.
