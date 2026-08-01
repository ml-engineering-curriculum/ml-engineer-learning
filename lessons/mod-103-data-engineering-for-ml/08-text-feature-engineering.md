# 08 — Text feature engineering

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 06, 07.

## Motivation

Text is not naturally consumable by a scikit-learn model. The
pipeline needs a *vectorisation* step that turns a document into a
numeric vector. This chapter covers the four representations you
will actually reach for on tabular-adjacent problems:

1. **Bag-of-words / count vectorisation** — every document is a
   sparse count vector over a fixed vocabulary.
2. **TF-IDF** — weighted counts that down-weight ubiquitous words.
3. **Hashing** — the vocabulary-free alternative for streaming or
   very-large corpora.
4. **Sentence / token embeddings** — dense vectors from a
   pre-trained model.

Deep-learning-native tokenisation (BPE, WordPiece) and end-to-end
LLM fine-tuning are `mod-105`'s territory. This chapter is about
turning short-to-medium text into features you can hand to a
`LogisticRegression`, a gradient booster, or a shallow neural net.

## Preprocessing: don't overdo it

Text pipelines historically had a long preprocessing chain:
lowercase, strip punctuation, remove stop-words, stem, lemmatise.
Modern practice is much lighter:

- **Lowercase** — usually helpful; `TfidfVectorizer(lowercase=True)`
  by default.
- **Strip punctuation** — often unnecessary; the vectoriser's
  tokeniser handles it.
- **Stop-word removal** — do not do this reflexively. Stop words
  carry more signal than tradition claims, and `TfidfVectorizer`
  down-weights them naturally.
- **Stemming / lemmatisation** — was important with tiny vocabularies
  and simple models. With modern hashing / TF-IDF / embeddings,
  rarely worth it.

The pareto-optimal preprocessing for most tasks is: lowercase, let
the vectoriser tokenise, and use `min_df=` / `max_df=` to trim the
long tail.

Where preprocessing does matter:

- **Domain-specific normalisation.** Phone numbers, URLs, currency
  amounts often benefit from being replaced with placeholder
  tokens (`<PHONE>`, `<URL>`) so the vectoriser doesn't produce
  10,000 near-identical features.
- **HTML stripping.** If your source is scraped web pages, run it
  through `BeautifulSoup(text, "html.parser").get_text()` first.
  Otherwise every document's most common "word" is `<div>`.
- **Unicode normalisation.** `unicodedata.normalize("NFC", text)`
  to collapse visually-identical forms.

## `CountVectorizer`: bag of words

The starting point. Turns a corpus into a `documents × vocabulary`
sparse matrix of token counts.

```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b",   # default: 2+ char word tokens
    ngram_range=(1, 2),               # unigrams and bigrams
    min_df=5,                         # a token must appear in ≥5 docs
    max_df=0.9,                       # drop tokens in >90% of docs
    max_features=50_000,              # cap the vocabulary
)
X = vec.fit_transform(train_texts)
```

Four flags that control almost all the interesting behaviour:

- **`ngram_range=(1, 2)`** — include word bigrams alongside
  unigrams. Bigrams often carry substantially more signal than
  unigrams at a moderate cost in vocabulary size.
- **`min_df=`** — drop tokens seen in fewer documents than this.
  Removes typos and hapax legomena. Set to 2–10 for most corpora.
- **`max_df=`** — drop tokens seen in more than this fraction of
  documents. A cheaper alternative to a stop-word list.
- **`max_features=`** — cap the vocabulary size. Deterministic
  under a specified `sort_key` (largest counts win).

**Leakage rule.** `CountVectorizer.fit` learns the vocabulary from
the training data. Fitting on the combined train+test corpus leaks
test-vocabulary presence into training. As always: `Pipeline` +
train-only fit.

## `TfidfVectorizer`: down-weight the ubiquitous

`TfidfVectorizer(**kwargs)` is `CountVectorizer(**kwargs)` +
`TfidfTransformer()` in a single step. Per-token weight becomes
`term-frequency × log(inverse-document-frequency)`, which
down-weights words that appear in nearly every document.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    max_features=50_000,
    sublinear_tf=True,                # 1 + log(tf) instead of raw tf
    norm="l2",                        # L2-normalise each row
)
```

`sublinear_tf=True` is a nearly-free improvement — it prevents very
repetitive documents from dominating.

The output is a sparse matrix of shape `(n_docs, n_vocab)`. Feed
it directly to a `LogisticRegression`, `LinearSVC`, `SGDClassifier`,
or (with `.todense()` or a tree ensemble that accepts sparse input)
a gradient booster.

Baseline text classification pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True)),
    ("clf",   LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)),
])
```

For many text classification tasks, this three-line pipeline gets
you 80% of the way to a fine-tuned transformer's accuracy at 1% of
the compute.

## `HashingVectorizer`: no `fit` step

`HashingVectorizer` maps tokens to a fixed number of buckets via a
hash function. No vocabulary learned; no `fit` needed; the
resulting features are stable across processes and machines.

```python
from sklearn.feature_extraction.text import HashingVectorizer

vec = HashingVectorizer(
    n_features=2**20,       # ~1M buckets, sparse
    ngram_range=(1, 2),
    alternate_sign=False,   # keep counts non-negative
)
```

Trade-offs:

- **Pros.** Streaming-friendly (no vocabulary to update);
  serialises to zero bytes; scales to any corpus; deterministic
  across runs.
- **Cons.** Collisions inject noise; no way to invert `hash(word) →
  word` for interpretability; feature importances are per-bucket,
  not per-word.

Reach for it when the corpus is much larger than memory, when you
need online / streaming updates, or when you want a text classifier
that has no fitted vocabulary state to serve.

## Character n-grams

For short text (tweets, product names, URLs) or noisy text (typos,
truncations), character n-grams often outperform word n-grams:

```python
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
```

`analyzer="char_wb"` gives you character n-grams *within* word
boundaries — the token `hello ` (with the trailing space) produces
`_he`, `hel`, `ell`, `llo`, `lo_`. Robust to misspellings; the
model learns `"pand4s"` and `"pandas"` share substrings.

## Sentence and token embeddings

Modern practice for anything non-trivial: encode each document as a
dense vector produced by a pretrained transformer. Two access
patterns are common.

### Sentence-Transformers

`sentence-transformers` gives you a simple API over a large
selection of pretrained models fine-tuned for producing sentence-
level embeddings. Typical usage:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")     # 384-dim, fast, general-purpose
embeddings = model.encode(
    texts,
    batch_size=128,
    convert_to_numpy=True,
    normalize_embeddings=True,   # L2-normalise
)   # shape (n_docs, 384)
```

Use `normalize_embeddings=True` if you plan to compare with cosine
similarity — with L2-normalised vectors, cosine similarity is just
the dot product.

Model choice notes:

- **`all-MiniLM-L6-v2`** — 384 dim, 22M params, fast on CPU. Solid
  default.
- **`all-mpnet-base-v2`** — 768 dim, 110M params, higher quality;
  slower. Good default when quality matters and you have a GPU.
- **`multi-qa-mpnet-base-dot-v1`** — tuned for question / passage
  retrieval; use for search / retrieval use cases.
- **Multilingual: `paraphrase-multilingual-MiniLM-L12-v2`** —
  covers 50+ languages.

For domain-specific text (medical, legal, financial, code),
sometimes a domain-fine-tuned model on Hugging Face outperforms the
general-purpose ones. Try a general model first; upgrade only if
metrics warrant.

### Embedding as a feature

Once you have embeddings, they slot into a `Pipeline` like any
other feature matrix:

```python
import numpy as np

X_train_emb = model.encode(train_texts, batch_size=128, convert_to_numpy=True)
X_test_emb  = model.encode(test_texts,  batch_size=128, convert_to_numpy=True)

clf = LogisticRegression(C=1.0, max_iter=2000)
clf.fit(X_train_emb, y_train)
```

For a fully-`Pipeline`-integrated version, wrap `model.encode` in a
custom transformer:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class SentenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=128):
        self.model_name = model_name
        self.batch_size = batch_size
    def fit(self, X, y=None):
        from sentence_transformers import SentenceTransformer
        self.model_ = SentenceTransformer(self.model_name)
        return self
    def transform(self, X):
        return self.model_.encode(list(X), batch_size=self.batch_size,
                                  convert_to_numpy=True, normalize_embeddings=True)
```

**Leakage note.** The pretrained model itself was fit on a huge
external corpus; that is not leakage in the sense of chapter 06.
What *would* be leakage: fine-tuning the encoder on your training
data and then using it to embed the test set. If you fine-tune,
the fine-tuned encoder must be fit on the training portion only —
same discipline as any other learned transformer.

### Caching embeddings

Encoding a large corpus is expensive. Cache the output:

```python
import numpy as np
from pathlib import Path

cache = Path("embeddings.npz")
if cache.exists():
    data = np.load(cache)
    X = data["X"]
else:
    X = model.encode(texts, batch_size=128, convert_to_numpy=True)
    np.savez(cache, X=X)
```

For production, the cache is usually a Parquet dataset keyed by
`(document_id, model_name, model_version)`. Recompute only when the
model changes.

## Combining text features with tabular features

Most real-world models have text *and* tabular features side-by-
side (a product listing has a title, a description, a price, a
category, a seller rating). `ColumnTransformer` composes them:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

preprocess = ColumnTransformer([
    ("title", TfidfVectorizer(min_df=5, ngram_range=(1, 2)), "title"),
    ("desc",  TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20_000), "description"),
    ("num",   Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale",  StandardScaler())]),
              ["price", "seller_rating"]),
    ("cat",   OneHotEncoder(handle_unknown="ignore", min_frequency=25),
              ["category", "brand"]),
])
```

Notes:

- `ColumnTransformer` passes a single column (as a `Series`) to a
  transformer whose column argument is a string, and a `DataFrame`
  slice to one given a list. `TfidfVectorizer` expects a `Series`,
  which is what a single-column selector produces.
- The concatenated output is sparse if any component is sparse.
  Feed it to a linear model or gradient booster that handles
  sparse input.
- The final feature matrix can be extremely wide (100k+ TF-IDF
  columns). `LogisticRegression(solver="liblinear")` or `"saga"`
  handles this well; tree ensembles less so.

## Multilingual text

If your corpus spans languages, three choices, in decreasing order
of quality and cost:

1. **Multilingual sentence embeddings** (`paraphrase-multilingual-...`).
   Best quality; handles code-switching; single model.
2. **Language-specific TF-IDF pipelines,** routed by a
   language-detection step. Reasonable for a handful of
   well-represented languages.
3. **Character n-gram TF-IDF.** Language-agnostic; loses some
   semantics; requires no per-language tuning. Underrated fallback.

If you don't know the language distribution of your corpus, run
`fasttext` or `langdetect` on a sample first. A "multilingual" corpus
that turns out to be 98% English does not need special handling.

## Pitfalls collected

1. **Fitting `TfidfVectorizer` on `X = train + test`.** Test-set
   vocabulary and IDF weights bleed into training. Wrap in a
   `Pipeline`, fit on `X_train` only.
2. **Aggressive stop-word removal.** English stop words (`the`,
   `a`, `and`) sometimes carry sentiment or intent. TF-IDF already
   down-weights them; hard-removing them can hurt.
3. **Not setting `min_df`.** A 100M-document corpus with
   `min_df=1` produces a vocabulary of 20M+ tokens, most of them
   typos. Set `min_df=5` at minimum.
4. **Serving with a different vectoriser than training.** The
   fitted `Pipeline` (saved via `joblib`) is the antidote. Never
   pickle just the model.
5. **Cosine similarity on unnormalised embeddings.** If the vectors
   are not L2-normalised, cosine similarity requires dividing by
   the norms — cheap but often forgotten. Pass
   `normalize_embeddings=True` at encode time.
6. **Assuming a bigger embedding model is always better.** For many
   classification tasks, MiniLM matches larger models. Measure.
7. **Embedding at request time in production.** Embedding a
   sentence on a CPU can take tens of milliseconds. Cache /
   precompute for latency-sensitive services.

## Summary

- Text vectorisation options, in ascending order of complexity and
  descending order of vocabulary independence: bag-of-words →
  TF-IDF → hashing → dense embeddings.
- `TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True)`
  + `LogisticRegression` is a shockingly strong baseline for most
  text classification tasks.
- Sentence embeddings (`sentence-transformers`) give you dense
  representations from a pretrained model in one line. Cache
  aggressively; L2-normalise if you plan to compare with cosine.
- `ColumnTransformer` composes text and tabular features cleanly.
  Everything belongs inside a `Pipeline` — the leakage rules from
  chapter 06 apply to text too.
- Preprocessing lightly is fine. Do not stem / lemmatise / strip
  stop-words reflexively.

## Where this goes next

Chapter 09 extends the same framing to images — decoding, resizing,
normalisation, augmentation, and pretrained-embedding features.
