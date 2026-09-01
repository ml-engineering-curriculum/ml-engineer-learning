# Resources for mod-103 — Data Engineering for ML: Ingestion, Cleaning, Feature Engineering

A curated short list. The goal is depth-on-demand for the chapter
you got stuck on, not a comprehensive bibliography. Primary
documentation is listed first; textbooks, articles, and tooling
references follow.

## Ingestion — files (chapter 02)

- **Pandas — IO tools (`read_csv`, `read_json`, `read_parquet`,
  …).**
  <https://pandas.pydata.org/docs/user_guide/io.html>
  The canonical reference for every `read_*` and `to_*` function
  chapter 02 uses. Skim once; then keep the `read_csv` and
  `read_parquet` sections open on a second monitor.
- **Pandas — `read_csv` full parameter list.**
  <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>
  The `dtype`, `parse_dates`, `na_values`, `chunksize`, and
  `dtype_backend` arguments are the ones chapter 02 spends time on.
- **Apache Parquet — format documentation.**
  <https://parquet.apache.org/docs/>
  Explains the on-disk layout, row groups, and column chunks that
  make partition pruning and column projection cheap.
- **Apache Arrow / PyArrow — user guide.**
  <https://arrow.apache.org/docs/python/index.html>
  The engine that Pandas, Polars, DuckDB, and every modern lake
  format sit on top of.
- **PyArrow — datasets API.**
  <https://arrow.apache.org/docs/python/dataset.html>
  Partitioned reads, filter pushdown, column projection — the
  primitives chapter 02 relies on for large Parquet directories.
- **`pyarrow.dataset.write_dataset`.**
  <https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html>
  Reference for the atomic-write / `_SUCCESS`-marker pattern
  chapter 02 recommends.
- **JSON Lines (NDJSON).**
  <https://jsonlines.org/>
  Two-page spec of the "one JSON object per line" format that
  chapter 02 treats as the good shape for JSON at scale.
- **Wes McKinney — "Apache Arrow and the '10 Things I Hate About
  pandas'" (blog post, 2017).**
  Searchable at <https://wesmckinney.com/>.
  <!-- needs-research: verify current permalink; the post has moved
  URLs at least once. -->
  Context on why the Arrow-backed `dtype_backend` matters — from
  the creator of Pandas.

## Ingestion — SQL (chapter 03)

- **SQLAlchemy 2 — tutorial.**
  <https://docs.sqlalchemy.org/en/20/tutorial/index.html>
  The `Engine`, `text`, and bound-parameter idioms chapter 03
  insists on. If you learned SQLAlchemy pre-2.0, this is the
  migration path.
- **SQLAlchemy — engine configuration and connection pooling.**
  <https://docs.sqlalchemy.org/en/20/core/engines.html>
  The reference for `pool_pre_ping`, pool sizing, and the URL
  builders (`URL.create`) chapter 03 uses instead of string
  interpolation.
- **Pandas — `read_sql`, `read_sql_query`, `read_sql_table`.**
  <https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html>
  Documents the `params=`, `chunksize=`, `dtype=`, and
  `parse_dates=` arguments chapter 03 covers.
- **`psycopg` (Postgres driver, v3).**
  <https://www.psycopg.org/psycopg3/docs/>
  The modern successor to `psycopg2`. `psycopg[binary]` is the
  install form chapter 03 uses.
- **Snowflake — Python connector docs.**
  <https://docs.snowflake.com/en/developer-guide/python-connector/python-connector>
  The `fetch_pandas_all()` method is the fast path chapter 03
  mentions for warehouse reads.
- **Google BigQuery — client libraries for Python.**
  <https://cloud.google.com/bigquery/docs/reference/libraries>
  Pair with `pandas-gbq` for DataFrame-shaped reads.
- **DuckDB — Python API.**
  <https://duckdb.org/docs/api/python/overview>
  "SQL that runs in-process against your Parquet files or Pandas
  DataFrames." Increasingly common as a local-analysis layer;
  chapter 03 flags it as the escape hatch when the warehouse is
  too slow.
- **`testcontainers-python`.**
  <https://testcontainers-python.readthedocs.io/en/latest/>
  Spin up a real Postgres in a test — the exercise-01 pattern for
  writing SQL integration tests without a cloud account.

## Ingestion — object stores (chapter 04)

- **`fsspec` — documentation.**
  <https://filesystem-spec.readthedocs.io/en/latest/>
  The abstraction over local, S3, GCS, Azure, HTTP, and a dozen
  others. Pandas, PyArrow, Polars, and Dask all route through it.
- **`s3fs` — documentation.**
  <https://s3fs.readthedocs.io/en/latest/>
  `fsspec` backend for Amazon S3 and S3-compatible stores
  (MinIO, Ceph, Cloudflare R2). `storage_options=` reference.
- **`gcsfs` — documentation.**
  <https://gcsfs.readthedocs.io/>
- **`adlfs` — documentation.**
  <https://github.com/fsspec/adlfs>
  Azure Data Lake / Blob backend.
- **Amazon S3 — user guide.**
  <https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html>
  The IAM, bucket-policy, and request-rate documentation chapter 04
  points at for the "credentials come from the SDK's default
  chain" rule.
- **Amazon S3 — best practices for prefixes and request rates.**
  <https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html>
  Explains the 5,500 GET / 3,500 PUT per prefix rate limits
  chapter 04 warns about.
- **MinIO — quickstart.**
  <https://min.io/docs/minio/container/index.html>
  Local S3-compatible server for a `docker compose` dev loop.
- **Delta Lake — documentation.**
  <https://docs.delta.io/latest/index.html>
- **Apache Iceberg — documentation.**
  <https://iceberg.apache.org/docs/latest/>
  The two mainstream open table formats chapter 04 names as the
  layer above raw Parquet in lakehouse deployments.
- **`polars` — reading from object stores.**
  <https://docs.pola.rs/user-guide/io/cloud-storage/>
  Polars uses the object-store Rust crate rather than `fsspec`;
  worth a look if you want to escape the Pandas GIL for ingestion.

## Profiling and cleaning (chapter 05)

- **Pandas — working with missing data.**
  <https://pandas.pydata.org/docs/user_guide/missing_data.html>
  Includes the semantics of `pd.NA`, `NaN`, and `NaT` and the
  nullable-dtype story that chapter 05 relies on.
- **Pandas — duplicated / drop_duplicates.**
  <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html>
- **`ydata-profiling` (formerly `pandas-profiling`).**
  <https://docs.profiling.ydata.ai/>
  The "one-line HTML report" tool chapter 05 recommends for the
  first-pass profile.
- **scikit-learn — outlier detection (User Guide §2.7).**
  <https://scikit-learn.org/stable/modules/outlier_detection.html>
  Covers `IsolationForest`, `LocalOutlierFactor`, and
  `OneClassSVM` — the three "throw ML at outliers" methods chapter
  05 catalogues.
- **scikit-learn — `IsolationForest`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>
- **Little, Roderick J. A., and Donald B. Rubin. *Statistical
  Analysis with Missing Data* (3rd ed., Wiley, 2019).**
  The canonical academic reference for the MCAR / MAR / MNAR
  taxonomy chapter 05 uses. Chapter 1 is enough for the working
  vocabulary; the rest is for when you need to defend an
  imputation strategy in a review.
- **van Buuren, Stef. *Flexible Imputation of Missing Data* (2nd
  ed., CRC Press, 2018).** Free HTML at <https://stefvanbuuren.name/fimd/>.
  Practical treatment of multiple imputation with the `mice` R
  package; the Python equivalent is `sklearn.experimental.IterativeImputer`.
- **Tukey, John W. *Exploratory Data Analysis* (Addison-Wesley,
  1977).** The origin of the boxplot and the IQR-based outlier
  rule chapter 05 cites. Still worth the read for the mindset.

## Leakage and label quality (chapter 06)

- **scikit-learn — common pitfalls and recommended practices.**
  <https://scikit-learn.org/stable/common_pitfalls.html>
  The official page on data leakage in preprocessing. Bookmarked
  alongside chapter 06.
- **scikit-learn — cross-validation (User Guide §5).**
  <https://scikit-learn.org/stable/modules/cross_validation.html>
  The full catalogue of splitters — including `GroupKFold`,
  `StratifiedGroupKFold`, and `TimeSeriesSplit` — that chapter 06
  matches to leakage types.
- **scikit-learn — `TimeSeriesSplit`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
- **scikit-learn — `GroupKFold` / `StratifiedGroupKFold`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html>
- **Kaufman, Shachar, Saharon Rosset, and Claudia Perlich.
  "Leakage in Data Mining: Formulation, Detection, and Avoidance"
  (SIGKDD 2011).**
  <https://dl.acm.org/doi/10.1145/2020408.2020496>
  The paper chapter 06's taxonomy is descended from. Reads as the
  war stories that named the failure modes.
- **Kapoor, Sayash, and Arvind Narayanan. "Leakage and the
  Reproducibility Crisis in ML-based Science" (2022).**
  <https://reproducible.cs.princeton.edu/>
  Contemporary catalogue of leakage patterns spotted in the
  academic literature. Sobering; useful vocabulary.
- **`cleanlab` — documentation.**
  <https://docs.cleanlab.ai/>
  The library chapter 06 names for automatic label-error
  detection. `cleanlab.filter.find_label_issues` is the workhorse
  entry point.
- **Northcutt, Curtis G., Anish Athalye, and Jonas Mueller.
  "Pervasive Label Errors in Test Sets Destabilize Machine
  Learning Benchmarks" (NeurIPS 2021).**
  <https://arxiv.org/abs/2103.14749>
  Companion paper to `cleanlab`. Documents label-error rates in
  MNIST, ImageNet, and eight other benchmark test sets. The
  "your labels have 3–10% noise" claim in chapter 06 is grounded
  here.
- **Krippendorff, Klaus. "Computing Krippendorff's Alpha-
  Reliability" (2011).**
  <https://repository.upenn.edu/asc_papers/43/>
  The interrater-agreement metric chapter 06 mentions for
  multi-annotator label pipelines.

## Tabular feature engineering (chapter 07)

- **scikit-learn — preprocessing (User Guide §6).**
  <https://scikit-learn.org/stable/modules/preprocessing.html>
  The main-line reference for every transformer chapter 07
  catalogues.
- **scikit-learn — `Pipeline`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>
- **scikit-learn — `ColumnTransformer` and
  `make_column_selector`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>
  <https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html>
- **scikit-learn — `TargetEncoder`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html>
  The docstring's "How does target encoding avoid data leakage?"
  section is *the* reference for the cross-fitting behaviour
  chapter 07 (and exercise-03) rely on.
- **scikit-learn — `OneHotEncoder` (with `min_frequency` and
  `handle_unknown`).**
  <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>
- **scikit-learn — `HistGradientBoostingClassifier` /
  `HistGradientBoostingRegressor`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>
  The `categorical_features="from_dtype"` path and native
  missing-value support are the "you can skip half the pipeline"
  features chapter 07 uses.
- **`category_encoders` — documentation.**
  <https://contrib.scikit-learn.org/category_encoders/>
  Deeper catalogue of categorical encoders (Weight-of-Evidence,
  M-estimate, CatBoost-style) than what ships in scikit-learn.
  Use when the built-ins are not enough.
- **Kuhn, Max, and Kjell Johnson. *Feature Engineering and
  Selection: A Practical Approach for Predictive Models* (CRC
  Press, 2019).** Free HTML at <http://www.feat.engineering/>.
  The most complete treatment of the transformations chapter 07
  covers — cyclical encodings, splines for numeric features,
  supervised encodings — with worked examples.
- **Zheng, Alice, and Amanda Casari. *Feature Engineering for
  Machine Learning* (O'Reilly, 2018).** A shorter, task-oriented
  companion. Chapter 5 (categorical encodings) and chapter 8
  (deep-dive on text) are the ones to cross-read with 07 and 08.

## Text feature engineering (chapter 08)

- **scikit-learn — text feature extraction (User Guide §6.2).**
  <https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction>
  `CountVectorizer`, `TfidfVectorizer`, and `HashingVectorizer`,
  with the `analyzer`, `ngram_range`, `min_df` / `max_df`, and
  `sublinear_tf` arguments chapter 08 uses.
- **scikit-learn — `TfidfVectorizer`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>
- **scikit-learn — `HashingVectorizer`.**
  <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html>
- **Sentence Transformers — documentation.**
  <https://www.sbert.net/>
  The library chapter 08 names for pretrained dense sentence
  embeddings. `SentenceTransformer("all-MiniLM-L6-v2")` is the
  strong small-model default.
- **Hugging Face — `transformers` documentation.**
  <https://huggingface.co/docs/transformers/index>
  When you need embeddings from a model that isn't in
  `sentence-transformers`, this is the fallback.
- **Hugging Face — `tokenizers` documentation.**
  <https://huggingface.co/docs/tokenizers/index>
  The library behind BPE / WordPiece tokenisation. Chapter 08
  points here for the subword-tokenisation coverage that `mod-105`
  deepens.
- **`spaCy` — documentation.**
  <https://spacy.io/usage>
  Production-grade tokenisation, sentence splitting, POS tagging,
  NER. Chapter 08 uses it as the drop-in for the "actual linguistic
  preprocessing" tier above what `TfidfVectorizer` does.
- **Weinberger, Kilian, et al. "Feature Hashing for Large Scale
  Multitask Learning" (ICML 2009).**
  <https://alex.smola.org/papers/2009/Weinbergeretal09.pdf>
  The paper behind `HashingVectorizer`. Short and quantitative on
  when the hash-collision cost is negligible.
- **Mikolov et al. "Distributed Representations of Words and
  Phrases and their Compositionality" (2013).**
  <https://arxiv.org/abs/1310.4546>
  The word2vec paper. Historical context for why dense-vector text
  representations replaced sparse bag-of-words in most pipelines.

## Image feature engineering (chapter 09)

- **`Pillow` (PIL fork) — documentation.**
  <https://pillow.readthedocs.io/en/stable/>
  The default image loader used by nearly every Python data-loading
  stack. `Image.open`, `.convert`, `.resize` are chapter 09's
  starting points.
- **`torchvision` — transforms v2.**
  <https://pytorch.org/vision/stable/transforms.html>
  The pretrained-model preprocessing conventions (`Resize`,
  `CenterCrop`, `Normalize` with the ImageNet mean/std) chapter
  09 covers. The v2 transforms replace the v1 ones as of PyTorch
  2.x.
- **`torchvision.models` — pretrained model weights.**
  <https://pytorch.org/vision/stable/models.html>
  Each model page lists the exact preprocessing pipeline it was
  trained with — chapter 09's "use the weights' own preprocessing
  or don't use the weights" rule.
- **`albumentations` — documentation.**
  <https://albumentations.ai/docs/>
  Faster, richer augmentations than `torchvision.transforms`. The
  library chapter 09 recommends for real training pipelines.
- **`webdataset` — documentation.**
  <https://webdataset.github.io/webdataset/>
  The sharded-tar dataset format chapter 09 names as the answer to
  the "millions of small image files" object-store problem.
- **Hugging Face `datasets` — documentation.**
  <https://huggingface.co/docs/datasets/index>
  The library chapter 09 defaults to for hosted image datasets
  (loads streaming, handles sharding, integrates with `torchvision`).
- **He, Kaiming, et al. "Deep Residual Learning for Image
  Recognition" (CVPR 2016).**
  <https://arxiv.org/abs/1512.03385>
  The ResNet paper. Cited in chapter 09 as the archetypal
  "extract features from a pretrained backbone" reference model.
- **Radford, Alec, et al. "Learning Transferable Visual Models
  From Natural Language Supervision" (ICML 2021, CLIP).**
  <https://arxiv.org/abs/2103.00020>
  The paper behind the "CLIP image embedding" default chapter 09
  points at for zero-shot / multimodal feature extraction.

## Dataset validation (chapter 10)

- **`pandera` — documentation.**
  <https://pandera.readthedocs.io/en/stable/>
  The `DataFrameModel`, `Field`, `Check`, `check_input`, and
  `check_output` APIs chapter 10 uses.
- **`pandera` — schema models tutorial.**
  <https://pandera.readthedocs.io/en/stable/dataframe_models.html>
  The class-based schema idiom (as opposed to the older
  `DataFrameSchema` dict form). This is the form chapter 10 and
  exercise-04 teach.
- **`pandera` — lazy validation and error reports.**
  <https://pandera.readthedocs.io/en/stable/lazy_validation.html>
  Explains what `lazy=True` changes about the failure UX.
- **Great Expectations (v1.x) — documentation.**
  <https://docs.greatexpectations.io/docs/home/>
  The current stable version's docs. If a tutorial you find uses
  `context.add_expectation_suite(...)` or `BatchRequest`, it is
  v0.18-era; the v1.x equivalents are documented here.
- **Great Expectations — Expectation Gallery.**
  <https://greatexpectations.io/expectations/>
  The full catalogue of built-in expectations, including
  `ExpectColumnKlDivergenceToBeLessThan`,
  `ExpectColumnMeanToBeBetween`, and the ~50 others chapter 10
  references.
- **`evidently` — documentation.**
  <https://docs.evidentlyai.com/>
  Open-source data-drift and model-monitoring toolkit. Chapter 10
  names it as the ready-made alternative to hand-rolling PSI / KS.
  `Report([DataDriftPreset()])` is the workhorse entry point.
- **`whylogs` — documentation.**
  <https://docs.whylabs.ai/docs/whylogs-overview>
  Data-logging library whose profiles are the input to the WhyLabs
  drift-detection service. Chapter 10 names it alongside
  `evidently` for the drift-detection tier.
- **Population Stability Index (PSI) — academic reference.**
  <!-- needs-research: cite the primary academic treatment of PSI.
  Candidate: Yurdakul, B. "Statistical Properties of Population
  Stability Index" (dissertation, Western Michigan University,
  2018). Verify title, year, and canonical URL before promoting to
  a real citation. -->
  The PSI thresholds chapter 10 uses (< 0.10 stable, 0.10–0.25
  moderate, > 0.25 severe) are practitioner rules of thumb from the
  credit-risk literature; treat them as calibration starting
  points, not universal cutoffs.
- **`scipy.stats.ks_2samp`.**
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html>
  The two-sample Kolmogorov–Smirnov test chapter 10 (and
  exercise-04) use for numeric drift.
- **`scipy.stats.chisquare`.**
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html>
  The categorical drift test.

## Cross-cutting: books and long reads

- **Kleppmann, Martin. *Designing Data-Intensive Applications*
  (O'Reilly, 2017).** Not ML-specific, but chapters 3 (storage
  and retrieval), 4 (encoding and evolution), and 11 (stream
  processing) are the systems-level backdrop for everything mod-103
  does with files, warehouses, and object stores.
- **Huyen, Chip. *Designing Machine Learning Systems* (O'Reilly,
  2022).** Chapter 3 ("Data Engineering Fundamentals") and
  chapter 4 ("Training Data") map almost one-to-one onto mod-103's
  objectives. Chapter 8 ("Data Distribution Shifts and Monitoring")
  is the natural pairing with chapter 10.
- **Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn,
  Keras, and TensorFlow* (3rd ed., O'Reilly, 2022).** Chapter 2
  ("End-to-End Machine Learning Project") walks through a complete
  ingest-clean-engineer cycle on a real dataset — a useful worked
  example alongside chapters 02, 05, and 07.
- **Kandel, Sean, et al. "Enterprise Data Analysis and
  Visualization: An Interview Study" (IEEE VAST 2012).**
  <https://vis.stanford.edu/papers/enterprise-analysis-interviews>
  Empirical study of how data scientists actually spend their
  time. The "80% of the work is data wrangling" folk claim traces
  back to this and adjacent papers; useful when you need to defend
  the amount of module-time chapter 01 spends on data.
- **Sculley, D., et al. "Hidden Technical Debt in Machine
  Learning Systems" (NeurIPS 2015).**
  <https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  The paper that named the "boring plumbing" problem that mod-103
  addresses. Section 3 ("Data Dependencies Cost More than Code
  Dependencies") is the direct motivation for chapter 10.
- **Polyzotis, Neoklis, et al. "Data Management Challenges in
  Production Machine Learning" (SIGMOD 2017).**
  <https://dl.acm.org/doi/10.1145/3035918.3054782>
  Google research group's overview of the same problem space,
  three years later. Reads as a survey.

## Datasets used in the exercises

- **UCI Adult (Census Income).**
  <https://archive.ics.uci.edu/dataset/2/adult>
  Also loadable via `sklearn.datasets.fetch_openml("adult",
  version=2, as_frame=True)`.
- **Ames Housing.**
  <https://www.openml.org/d/42165>
  Loadable via `sklearn.datasets.fetch_openml("house_prices",
  as_frame=True)`. De Cock, Dean. "Ames, Iowa: Alternative to the
  Boston Housing Data as an End of Semester Regression Project"
  (*JSE*, 2011). <https://jse.amstat.org/v19n3/decock.pdf>
- **NYC Taxi trip records (TLC).**
  <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
  Public Parquet dataset; the canonical playground for time-series
  and messy-data exercises.
- **Chicago Taxi Trips.**
  <https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew>
- **Titanic (Kaggle).**
  <https://www.kaggle.com/competitions/titanic>

<!-- needs-research: on each autonomous research cycle, verify these URLs still resolve and swap in the current stable-version page when a project promotes a new major (pandera 0.x → 1.x, GX 1.x → 2.x, scikit-learn 1.x, Pandas 2.x → 3.x, PyTorch 2.x → 3.x, etc.). -->
