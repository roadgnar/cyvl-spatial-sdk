# Accessing the Hackathon S3 Bucket

The dataset lives in a public, read-only S3 bucket:

```
cyvl-hackathon   (region: us-east-1)
```

For the duration of the hackathon **anyone can read it** — list and download
every object, no Cyvl account required. Writes are blocked for everyone. This
guide covers every way to get at the data.

## What's in the bucket

```
cyvl-hackathon/
├── README.md          # dataset overview — read this first
├── index.md           # file index
├── schemas.md         # column/field schemas
├── data/              # GeoJSON assets
├── parquet/           # columnar tables
└── pointclouds/       # LiDAR point clouds
```

## The one thing to understand first

The bucket policy grants read access to **everyone** (`Principal: *`). That
covers both authenticated (signed) and anonymous (unsigned) requests. But the
*client* — the AWS CLI or boto3 — decides whether to sign a request, and **by
default it always tries to sign and will refuse to send anything if it can't
find credentials.** A public bucket does not change that.

So which path you take depends entirely on whether you already have AWS
credentials:

| Your situation | What to do |
| --- | --- |
| You have AWS credentials (Cyvl SSO, or *any* AWS account) | Use the CLI / boto3 normally — nothing special |
| You have no AWS credentials at all | Add an "anonymous" flag, or just use `curl` / a browser |

Both paths reach the exact same data. Pick whichever matches your setup.

---

## Path A — you have AWS credentials

If `aws sts get-caller-identity` returns an identity, you're set. The bucket
behaves like any normal bucket — your signed requests are authorized because
the policy allows everyone. **No special flags needed.**

It does not matter *which* account your credentials belong to. Any valid AWS
principal works, because the policy trusts `*`.

```bash
# List
aws s3 ls s3://cyvl-hackathon/ --region us-east-1
aws s3 ls s3://cyvl-hackathon/data/ --region us-east-1

# Download a single file
aws s3 cp s3://cyvl-hackathon/README.md . --region us-east-1

# Sync a whole prefix locally
aws s3 sync s3://cyvl-hackathon/parquet/ ./parquet/ --region us-east-1
```

```python
# boto3 — default client, uses your ambient credentials
import boto3

s3 = boto3.client("s3", region_name="us-east-1")
s3.download_file("cyvl-hackathon", "README.md", "README.md")

for obj in s3.list_objects_v2(Bucket="cyvl-hackathon", Prefix="data/")["Contents"]:
    print(obj["Key"], obj["Size"])
```

---

## Path B — you have no AWS credentials

A plain `aws s3 ...` or default boto3 client will fail here with
`Unable to locate credentials` / `NoCredentialsError` — **even though the
bucket is public** — because the client won't send an unsigned request unless
you explicitly tell it to. You have three options.

### B1. AWS CLI with `--no-sign-request`

Tells the CLI to skip signing and make an anonymous request:

```bash
aws s3 ls s3://cyvl-hackathon/ --region us-east-1 --no-sign-request
aws s3 cp s3://cyvl-hackathon/README.md . --region us-east-1 --no-sign-request
aws s3 sync s3://cyvl-hackathon/data/ ./data/ --region us-east-1 --no-sign-request
```

### B2. boto3 with `UNSIGNED` signature config

```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(signature_version=UNSIGNED),  # anonymous — no creds needed
)
s3.download_file("cyvl-hackathon", "README.md", "README.md")
```

### B3. Plain HTTPS — no AWS tooling at all

Every object has a public URL. Use `curl`, `wget`, a browser, or any HTTP
library:

```
https://cyvl-hackathon.s3.amazonaws.com/<key>
```

```bash
curl -O https://cyvl-hackathon.s3.amazonaws.com/README.md
curl -O https://cyvl-hackathon.s3.amazonaws.com/data/aboveGroundAssets_v2.geojson
```

To browse the object listing as XML in a browser or with `curl`:

```bash
curl "https://cyvl-hackathon.s3.amazonaws.com/?list-type=2"
curl "https://cyvl-hackathon.s3.amazonaws.com/?list-type=2&prefix=parquet/"
```

---

## Reading data in place (no download)

The Parquet and GeoJSON files can be queried directly over S3.

**DuckDB** (works with credentials, or anonymously via the HTTPS URLs):

```sql
-- With AWS credentials configured:
INSTALL httpfs; LOAD httpfs;
SELECT * FROM 's3://cyvl-hackathon/parquet/*.parquet' LIMIT 10;

-- No credentials? Read over plain HTTPS instead:
SELECT * FROM 'https://cyvl-hackathon.s3.amazonaws.com/parquet/<file>.parquet' LIMIT 10;
```

**pandas / geopandas** over HTTPS (no AWS setup):

```python
import pandas as pd
df = pd.read_parquet("https://cyvl-hackathon.s3.amazonaws.com/parquet/<file>.parquet")

import geopandas as gpd
gdf = gpd.read_file("https://cyvl-hackathon.s3.amazonaws.com/data/aboveGroundAssets_v2.geojson")
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Unable to locate credentials` | No creds, and you didn't opt into anonymous mode | Add `--no-sign-request` (CLI) / `UNSIGNED` config (boto3), or use the HTTPS URL |
| `An error occurred (AccessDenied)` on upload | Writes are blocked for everyone — the bucket is read-only | Expected; the dataset is download-only |
| `Could not connect ... region` | Wrong region | It's `us-east-1` |
| Expired-token / SSO errors | Stale AWS credentials | Refresh your SSO login, **or** just use `--no-sign-request` to bypass creds entirely |

## A note on lifetime

Public access is on **only for the hackathon**. After the event the bucket is
re-locked, after which anonymous access stops working and only Cyvl accounts
can read it.
