# Flight Delay Big Data Pipeline — Azure Port

<p align="center">
  <img src="https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C?logo=apachespark&logoColor=white" />
  <img src="https://img.shields.io/badge/Azure-Databricks%20%2B%20ADLS%20Gen2-0078D4?logo=microsoftazure&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-Flexible%20Server-4169E1?logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/status-code%20port%2C%20not%20deployed-yellow" />
</p>

## What this is

This repository is an **architecture port**, not a new project. The original pipeline — [`flight-delay-bigdata-pipeline`](https://github.com/somepage9770/flight-delay-bigdata-pipeline) — was built on Oracle Cloud Infrastructure (OCI) for a Big Data course at Universidad Panamericana: PySpark batch processing on OCI Data Flow, OCI Object Storage for the data lake, and PostgreSQL for the curated tables.

Here, the same PySpark logic (cleaning, feature engineering, the streaming simulation, and the custom Naive Bayes log-odds model) is adapted to run on Azure instead: **ADLS Gen2** for storage, **Azure Databricks** for the Spark jobs, and **Azure Database for PostgreSQL Flexible Server** for the curated tables.

**Status: this is a code adaptation with a deployment guide, not a project that has been run on Azure.** I ported the storage URIs (`oci://` → `abfss://`), the storage authentication, and the PostgreSQL connection/secrets handling to their Azure equivalents, and wrote up the steps to actually stand it up — but I have not provisioned Azure resources or executed this against a live Azure account (mainly a cost/time tradeoff, not a technical blocker). Treat the scripts as reviewed, self-consistent code, and the guide below as untested instructions rather than a confirmed runbook.

---

## Why port it instead of just describing it

Talking about "portable architecture" is easy; showing the actual diff between two cloud providers' storage APIs, auth models, and secrets handling is a more concrete way to demonstrate it. This repo is that diff, applied to a project I already built and understand end to end.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                             Azure                                 │
│                                                                    │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  ADLS Gen2 │    │  Azure       │    │  ADLS Gen2            │   │
│  │  (raw/)    │───▶│  Databricks  │───▶│  (clean / curated)    │   │
│  └────────────┘    │  (PySpark)   │    └──────────┬────────────┘  │
│                     └──────────────┘               │               │
│                                                     ▼               │
│                                        ┌─────────────────────────┐ │
│                                        │ Azure Database for       │ │
│                                        │ PostgreSQL Flexible      │ │
│                                        │ Server                   │ │
│                                        └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## What changed from the OCI version, and why

| Concern | OCI version | Azure version | Why |
|---|---|---|---|
| Object storage | OCI Object Storage | Azure Data Lake Storage Gen2 (ADLS Gen2) | Azure's equivalent with hierarchical namespace support |
| Compute | OCI Data Flow (serverless Spark) | Azure Databricks | Azure's managed Spark offering |
| Storage URI | `oci://bucket@namespace/path` | `abfss://container@account.dfs.core.windows.net/path` | Provider-specific filesystem scheme |
| Storage auth | OCI resource principal / API key | Storage account access key, read from `AZURE_STORAGE_KEY` env var (or `dbutils.secrets` in a Databricks notebook) | Matches how Databricks clusters authenticate against ADLS Gen2 |
| Database | PostgreSQL (local instance, hardcoded `postgres`/`postgres` credentials) | Azure Database for PostgreSQL Flexible Server, credentials from `AZURE_PG_*` env vars, `sslmode=require` | A hardcoded local password was acceptable for a local academic setup; a database reachable over the internet needs credentials out of the source and TLS enforced. This isn't an Azure-specific requirement — the same change would apply moving the original project's Postgres instance anywhere network-reachable |
| PySpark logic (cleaning, feature engineering, log-odds model) | — | Unchanged | The transformation logic is provider-agnostic; only the I/O layer and secrets handling needed to change |

---

## Repository structure

```
flight-delay-bigdata-pipeline-azure/
├── process_flights_azure.py       # Batch processing & cleaning (ADLS Gen2)
├── streaming_flights_azure.py     # Monthly micro-batch streaming simulation
├── model_flights_azure.py         # Custom Naive Bayes log-odds prediction model
├── SQLTablesCreation_azure.py     # Loads curated CSVs into Azure PostgreSQL
└── README.md
```

The dataset, data dictionary, and model result artifacts from the original run live in the [OCI repo](https://github.com/somepage9770/flight-delay-bigdata-pipeline) — they aren't duplicated here since nothing has been re-run on Azure to produce new ones.

---

## Requirements

- An Azure subscription (a free-tier trial covers this comfortably — see the cost note below)
- Python 3.8+ with `pandas`, `sqlalchemy`, `psycopg2-binary` for the local loader script
- PySpark jobs run inside Azure Databricks and don't need a local Spark install

## Cost note before you start

None of this is permanently free. Azure gives new accounts $200 USD in credit for 30 days plus some always-free service tiers, which is enough to run this whole pipeline a few times over. Set a spending alert, and **delete the resource group when you're done** so nothing keeps billing after the trial credit runs out — deleting the resource group deletes everything inside it in one shot.

---

## Deployment guide

This walks through provisioning the Azure resources and running the pipeline against them. None of these steps have been executed yet — follow them as a first attempt, not a verified runbook.

### Step 1 — Create a Storage Account with ADLS Gen2 enabled

1. Go to [portal.azure.com](https://portal.azure.com) → **Storage accounts** → **Create**
2. Resource group: create a new one, e.g. `rg-flight-delay-pipeline` (keeps cleanup a one-click operation later)
3. Storage account name: something globally unique, e.g. `flightdelaymax` — this is `STORAGE_ACCOUNT` in the scripts
4. Region: whichever is closest to you
5. Redundancy: **LRS** (cheapest, fine for a personal project)
6. On the **Advanced** tab: enable **Hierarchical namespace** — this is what makes it ADLS Gen2 instead of plain Blob Storage, and it's required for the `abfss://` paths used in the scripts
7. Create it

Once created:
- **Containers** → **+ Container** → name it `flight-delay-data` — this is `CONTAINER` in the scripts
- Inside the container, create `raw/`, `clean/`, `curated/` (or just upload into `raw/` and let the pipeline create the rest)
- Upload the source dataset (`flights_sample_3m.csv` from the original repo) into `raw/`

Grab the access key: **Storage account → Access keys → key1 → Show → Copy**. This is `AZURE_STORAGE_KEY`.

### Step 2 — Create an Azure Databricks workspace

1. **Create a resource** → search **Azure Databricks** → **Create**
2. Same resource group as above
3. Pricing tier: **Premium** (needed for trial features; a 14-day Databricks trial is available on top of the Azure credit)
4. Once deployed, **Launch Workspace**

Inside Databricks:
1. **Compute** → **Create compute** → single-node cluster, smallest available instance type, Databricks Runtime with Spark 3.x
2. **Workspace** → **Create** → **Notebook**, language Python, attached to that cluster

### Step 3 — Provide the storage key

**Quick version** (fine for a personal project): in the notebook, before running the pipeline code:

```python
import os
os.environ["AZURE_STORAGE_KEY"] = "paste-your-key-here"
```

**Production-shaped version**: use the [Databricks CLI](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/) to create a secret scope, then in the notebook:

```python
os.environ["AZURE_STORAGE_KEY"] = dbutils.secrets.get(scope="flight-pipeline", key="storage-account-key")
```

### Step 4 — Run the pipeline

Copy each script into its own notebook cell (or import as notebook files), in order:

1. `process_flights_azure.py` — batch processing. Check that `clean/` and `curated/` appear in the storage container afterward.
2. `streaming_flights_azure.py` — streaming simulation. Check that `curated/streaming/month_1/` through `month_12/` appear.
3. `model_flights_azure.py` — the prediction model. Check that the risk profile folders and `curated/predictions/` appear.

Before running each script, update the two config lines at the top:

```python
STORAGE_ACCOUNT = "flightdelaymax"   # your actual storage account name
CONTAINER = "flight-delay-data"      # your actual container name
```

### Step 5 — Set up Azure Database for PostgreSQL and load the curated data

1. **Create a resource** → **Azure Database for PostgreSQL** → **Flexible server**
2. Same resource group, cheapest **Burstable** tier
3. Set an admin username and password, note them down
4. Under **Networking**, allow your current client IP
5. Download the curated CSVs from ADLS Gen2 to your machine (Storage account → Containers → `curated/...` → Download)
6. Set the environment variables and run the loader script locally:

```powershell
$env:AZURE_PG_HOST = "flight-delay-db.postgres.database.azure.com"
$env:AZURE_PG_USER = "pgadmin"
$env:AZURE_PG_PASSWORD = "your-password"
python SQLTablesCreation_azure.py
```

### Step 6 — Clean up

**Resource groups → rg-flight-delay-pipeline → Delete resource group**. This removes the storage account, Databricks workspace, and PostgreSQL server together, so nothing keeps billing.

---

## Related

- Original project (OCI, full results, dataset, and report): [flight-delay-bigdata-pipeline](https://github.com/somepage9770/flight-delay-bigdata-pipeline)
