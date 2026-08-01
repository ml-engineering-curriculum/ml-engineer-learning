# 04 — Ingesting from object stores

> **Reading time:** ~35 minutes.
> **Prerequisite chapters:** 01, 02.

## Motivation

Object storage is where large data actually lives. Every major cloud
has one: **Amazon S3**, **Google Cloud Storage (GCS)**, **Azure Blob
Storage**. Every large-scale data lake — Delta, Iceberg, Hudi —
sits on top of one. Warehouses' bulk export commands write to one.
When "the dataset" is bigger than "a laptop", it is almost certainly
a collection of Parquet or JSON files in an object-store bucket.

Reading from object storage is *almost* like reading from disk, but
the differences matter: latency is measured in tens of milliseconds
per request, credentials come from an IAM system, filenames are keys
in a flat namespace pretending to be a filesystem, and object
versions can silently change out from under you.

This chapter covers the working idioms: credentials, `fsspec`,
partitioned Parquet datasets, and the etiquette of large reads.

## The universal Python interface: `fsspec`

`fsspec` is the abstraction every modern Python data tool uses to
talk to filesystems and object stores. You install a per-backend
package (`s3fs`, `gcsfs`, `adlfs`) and then all of the following
"just work" because Pandas, PyArrow, Polars, and Dask route
through `fsspec` under the hood:

```python
import pandas as pd

df = pd.read_parquet("s3://bucket/path/events.parquet")
df = pd.read_parquet("gs://bucket/path/events.parquet")
df = pd.read_parquet("az://container/path/events.parquet")
df = pd.read_parquet("file:///local/path/events.parquet")   # also fsspec
```

The URL scheme selects the backend. Under the hood, `pandas` calls
`fsspec.open(...)` on the URL and hands the file handle to PyArrow.

Two implications:

- **A URL is a valid path anywhere Pandas expects one.** No
  boilerplate `boto3` code for the common case.
- **Backend-specific options** (region, endpoint, KMS key,
  requester-pays) go into a `storage_options=` dict:

  ```python
  df = pd.read_parquet(
      "s3://bucket/path/events.parquet",
      storage_options={"anon": False, "client_kwargs": {"region_name": "us-west-2"}},
  )
  ```

## Credentials: don't hardcode them

Every cloud SDK reads credentials from a standardised chain:

- **AWS.** In order: environment variables (`AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`), shared config
  files (`~/.aws/credentials`), container / EC2 instance profile,
  IAM role via IRSA or Pod Identity on EKS.
- **GCP.** Application Default Credentials: `GOOGLE_APPLICATION_CREDENTIALS`
  pointing at a service-account JSON, or Workload Identity /
  attached service account on GKE/GCE, or `gcloud auth
  application-default login` for local dev.
- **Azure.** `DefaultAzureCredential` walks environment variables,
  managed identity, `az login`, and Visual Studio Code auth in
  turn.

**Do not put access keys in code, notebooks, or config that ships in
the repo.** Every cloud vendor has a leaked-key detector that will
mail you a warning within minutes of a push, and the reputational /
security consequences can be significant.

The right shape for a training job:

- Locally: log in once (`aws sso login`, `gcloud auth application-default
  login`, `az login`). The SDK picks up your credentials.
- In CI: use an OIDC-backed role (GitHub Actions can assume an IAM
  role without long-lived secrets).
- In production: an IAM role attached to the pod / VM / task. The
  application never sees a raw key.

## The bucket / key / version model

Object stores are not filesystems, despite the `s3://bucket/dir/file`
notation. Three things are worth internalising:

1. **The namespace is flat.** `s3://bucket/foo/bar/baz.parquet` is
   a single key `foo/bar/baz.parquet` in `bucket`. Directories are
   a UI convention. Listing "a directory" is a prefix scan.
2. **Every operation is a network round-trip.** `os.path.exists`
   is free; `s3.head_object` is a request. Reading the same
   1 KB metadata a hundred times is a hundred requests. Cache
   locally when appropriate.
3. **Objects can be versioned.** With versioning enabled, a key's
   contents at a specific `versionId` is immutable. Without
   versioning, `PUT` overwrites and the old bytes are gone.

For reproducibility, especially in a training job, favour one of:

- Address a specific version (`s3://bucket/key?versionId=abcd1234`
  in APIs that support it, or the equivalent in the SDK).
- Address an immutable filename convention
  (`s3://bucket/dataset/dt=2026-08-01/part-0.parquet`) and never
  overwrite.
- Point the job at a **manifest** file that lists the specific keys
  in the dataset, and version the manifest.

## Partitioned Parquet datasets

Chapter 02 introduced Hive-style partitioning. This is where it
pays off. A dataset in S3 organised as

```
s3://bucket/events/
  date=2026-07-30/
    part-0.parquet
    part-1.parquet
  date=2026-07-31/
    part-0.parquet
  date=2026-08-01/
    part-0.parquet
```

can be read *without downloading the days you do not need*, because
`pyarrow.dataset` prunes the directory listing by the filter you
apply:

```python
import pyarrow.dataset as ds

dataset = ds.dataset(
    "s3://bucket/events/",
    format="parquet",
    partitioning="hive",
)

table = dataset.to_table(
    filter=(ds.field("date") == "2026-08-01") & (ds.field("country") == "US"),
    columns=["user_id", "ts", "amount"],
)
df = table.to_pandas()
```

Three optimisations compose here:

- **Partition pruning.** The reader lists only the `date=2026-08-01/`
  prefix.
- **Row-group filtering.** Inside each Parquet file, row groups
  whose min/max statistics do not overlap the filter are skipped.
- **Column projection.** Only the requested columns' bytes are
  downloaded.

The net effect: a dataset with terabytes across years of partitions
can serve a "yesterday, three columns, one country" read in seconds.

### Choosing a partition key

Rules of thumb:

- Partition by a **low-cardinality** column that queries frequently
  filter on. Date is the archetype.
- Do not partition on high-cardinality columns (`user_id`) — you
  get millions of tiny files, and object stores charge per request.
- Aim for partitions of roughly **128 MB to a few GB each**. Too
  small and metadata dominates read time; too large and you lose
  parallelism.

If a partition is empty for a given filter, the reader must still
list it to know that; keeping the partition count sensible matters.

## Writing partitioned datasets

`pyarrow.dataset.write_dataset` is the canonical way to write a
partitioned Parquet dataset. Pandas `.to_parquet(partition_cols=...)`
routes through it for the common case:

```python
df.to_parquet(
    "s3://bucket/events/",
    partition_cols=["date"],
    engine="pyarrow",
    compression="snappy",
    index=False,
)
```

Two habits to build:

1. **Do not include the partition column in the file body.** It is
   already encoded in the directory name; writing it inside the
   file wastes space and can go out of sync. `partition_cols=`
   handles this correctly.
2. **Write once, atomically.** For a dataset that will be read by
   many downstream consumers, write to a temporary prefix and then
   rename / copy to the final location, or write a
   `_SUCCESS` marker file that consumers check before reading.
   Partial writes are a common source of "the pipeline read half
   the data" bugs.

## Streaming download vs. materialised local copy

Sometimes you need to keep a local copy of the dataset — because
the training job runs many epochs over it, because the object store
is far away and per-request latency is painful, or because a
downstream tool doesn't speak `fsspec`.

For a whole dataset, the CLI is fastest:

```
aws s3 sync s3://bucket/events/date=2026-08-01/ ./data/2026-08-01/
gsutil -m rsync -r gs://bucket/events/date=2026-08-01/ ./data/2026-08-01/
azcopy sync https://acct.blob.core.windows.net/container/events/date=2026-08-01/ ./data/2026-08-01/
```

For a *very large* dataset that does not fit on the local disk,
mount the bucket read-only via `s3fs-fuse`, `gcsfuse`, or
`blobfuse2`. This is a common pattern in GPU training pods.

For most ML training, the sweet spot is: keep the dataset in the
object store, use `pyarrow.dataset` with column projection and
partition pruning, and only cache locally what you touch repeatedly.

## Rate limits, retries, and cost

Object stores rate-limit per-prefix. On S3, "3,500 PUT/COPY/POST/DELETE
or 5,500 GET/HEAD requests per second per partitioned prefix" is the
documented steady-state limit, and it's a common source of pain when
a training loop tries to open thousands of tiny files in parallel.

Mitigations:

- **Increase the file size** — batch tiny records into 128 MB Parquet
  parts.
- **Spread the workload across prefixes** — good partitioning helps.
- **Increase parallelism gradually** and back off on 503s. All the
  SDKs implement exponential backoff by default; do not disable it.
- **Watch egress cost.** Egressing a TB from S3 to a different
  region — or worse, out of AWS — is expensive. Run the training
  job in the same region as the bucket.

## Pitfalls collected

1. **Reading one tiny file at a time.** A directory of 100,000
   1-KB Parquet files is a rate-limit disaster and a metadata
   nightmare. Compact into fewer, larger files before consuming.
2. **Silently overwriting a versioned key.** Without versioning
   enabled, `PUT` overwrites and the previous bytes are gone. Turn
   on versioning for any bucket that holds training data.
3. **Trusting `LastModified`.** Object timestamps can jump forward
   or backward when you copy files between buckets or restore from
   backup. Use content hashes for change detection, not
   timestamps.
4. **Requester-pays buckets.** Some public datasets are "requester
   pays" — you get charged for downloading them. Pass
   `storage_options={"requester_pays": True}` (or the SDK
   equivalent) explicitly to opt in.
5. **Eventual consistency, in the past.** S3 has been strongly
   consistent for reads-after-writes since 2020; GCS always was;
   Azure Blob strongly-consistent per container. Older code and
   older tutorials may still recommend consistency workarounds
   (e.g. "wait 5 seconds before reading what you wrote"). You
   don't need them anymore.
6. **`ls` on a huge prefix.** Listing 1M keys is slow *and*
   expensive. Use the partition structure to only list the subset
   you need, or maintain a manifest file.

## Summary

- Object stores (S3, GCS, Azure Blob) are where ML data lives at
  scale. `fsspec` (via `s3fs` / `gcsfs` / `adlfs`) lets Pandas,
  PyArrow, and Polars read `s3://…` / `gs://…` / `az://…` URLs
  directly.
- Credentials come from the SDK's default chain — environment,
  config file, instance role. Never hardcoded.
- Partitioned Parquet datasets combine partition pruning, row-group
  filtering, and column projection to make reads on huge datasets
  cheap. Use `pyarrow.dataset` for filter pushdown.
- Choose partition columns for low cardinality and query patterns.
  Aim for 128 MB–a few GB per partition.
- Write once, atomically. Address by immutable path or by
  version ID. Do not overwrite the file the training job is
  reading.
- Watch out for tiny files, rate limits, and cross-region egress.

## Where this goes next

You now have three chapters' worth of tools for turning "the source
of truth" into a DataFrame in memory. Chapter 05 opens the second
arc of the module: making that DataFrame *clean* enough to feed to
a model.
