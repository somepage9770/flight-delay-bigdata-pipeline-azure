"""
Phase 4 — Batch processing & cleaning (Azure version)

Same logic as the original process_flights.py, but reading from and
writing to Azure Data Lake Storage Gen2 (ADLS Gen2) instead of OCI
Object Storage.

Run this as a notebook cell in Azure Databricks / Synapse Spark, or
with spark-submit against a cluster that has network access to your
storage account.

Before running, set the AZURE_STORAGE_KEY environment variable (or,
if you're in a Databricks notebook, use dbutils.secrets instead —
see the note at the bottom of this file). Never hardcode the key.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, month, when, round

# ── Config: change these two to match your Azure resources ──────────
STORAGE_ACCOUNT = "flightdelaymax"        # <-- your storage account name
CONTAINER = "flight-delay-data"           # <-- your ADLS Gen2 container name
# ──────────────────────────────────────────────────────────────────

spark = SparkSession.builder.appName("FlightDelayPipeline").getOrCreate()

# Authenticate against ADLS Gen2 using the storage account access key.
# Read it from an environment variable — never commit a key to git.
storage_key = os.environ["AZURE_STORAGE_KEY"]
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key,
)


def abfss_path(layer: str) -> str:
    """Build an abfss:// URI for a given data lake layer/path."""
    return f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net/{layer}"


# STEP 1: Load raw dataset
print("=== STEP 1: Loading raw dataset ===")
df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(abfss_path("raw/flights_sample_3m.csv"))
)

df.show(5)
df.printSchema()
print("Total raw records:", df.count())

# STEP 2: Cleaning
print("=== STEP 2: Cleaning data ===")

df_clean = df.filter(
    col("DEP_DELAY").isNotNull()
    & col("ARR_DELAY").isNotNull()
    & col("ORIGIN").isNotNull()
    & col("DEST").isNotNull()
    & col("AIRLINE_CODE").isNotNull()
)

df_clean = df_clean.withColumn(
    "IS_DELAYED", when(col("DEP_DELAY") > 15, 1).otherwise(0)
)

df_clean = df_clean.withColumn(
    "DELAY_CAUSE",
    when(col("DELAY_DUE_CARRIER") > 0, "Carrier")
    .when(col("DELAY_DUE_WEATHER") > 0, "Weather")
    .when(col("DELAY_DUE_NAS") > 0, "NAS")
    .when(col("DELAY_DUE_SECURITY") > 0, "Security")
    .when(col("DELAY_DUE_LATE_AIRCRAFT") > 0, "Late Aircraft")
    .otherwise("No Delay"),
)

print("Records after cleaning:", df_clean.count())
df_clean.show(5)

# STEP 3: Save clean layer
print("=== STEP 3: Saving clean layer ===")
df_clean.write.mode("overwrite").option("header", True).csv(
    abfss_path("clean/flights_clean/")
)
print("Clean layer saved successfully.")

# STEP 4: Delay by airline
print("=== STEP 4: Analysis 1 - Delay by airline ===")
delay_by_airline = (
    df_clean.groupBy("AIRLINE_CODE")
    .agg(
        round(avg("DEP_DELAY"), 2).alias("avg_dep_delay_min"),
        round(avg("ARR_DELAY"), 2).alias("avg_arr_delay_min"),
        count("*").alias("total_flights"),
    )
    .orderBy("avg_dep_delay_min", ascending=False)
)
delay_by_airline.show()
delay_by_airline.write.mode("overwrite").option("header", True).csv(
    abfss_path("curated/delays_by_airline/")
)

# STEP 5: Top 10 routes with the most delay
print("=== STEP 5: Analysis 2 - Delay by route ===")
delay_by_route = (
    df_clean.groupBy("ORIGIN", "DEST")
    .agg(
        round(avg("DEP_DELAY"), 2).alias("avg_delay_min"),
        count("*").alias("total_flights"),
    )
    .orderBy("avg_delay_min", ascending=False)
    .limit(10)
)
delay_by_route.show()
delay_by_route.write.mode("overwrite").option("header", True).csv(
    abfss_path("curated/delays_by_route/")
)

# STEP 6: Trend by month
print("=== STEP 6: Analysis 3 - Delay trend by month ===")
delay_by_month = (
    df_clean.groupBy(month("FL_DATE").alias("month"))
    .agg(
        round(avg("DEP_DELAY"), 2).alias("avg_delay_min"),
        count("*").alias("total_flights"),
    )
    .orderBy("month")
)
delay_by_month.show()
delay_by_month.write.mode("overwrite").option("header", True).csv(
    abfss_path("curated/delays_by_month/")
)

# STEP 7: Delay causes
print("=== STEP 7: Analysis 4 - Delay by cause ===")
delay_by_cause = (
    df_clean.groupBy("DELAY_CAUSE")
    .agg(
        count("*").alias("total_flights"),
        round(avg("DEP_DELAY"), 2).alias("avg_delay_min"),
    )
    .orderBy("total_flights", ascending=False)
)
delay_by_cause.show()
delay_by_cause.write.mode("overwrite").option("header", True).csv(
    abfss_path("curated/delays_by_cause/")
)

print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
spark.stop()

# ──────────────────────────────────────────────────────────────────
# Note on auth in an actual Databricks notebook:
# If you're running this as a Databricks notebook cell (not
# spark-submit), it's more idiomatic to skip the AZURE_STORAGE_KEY
# env var and instead register the key as a Databricks secret, then:
#
#   storage_key = dbutils.secrets.get(scope="flight-pipeline", key="storage-account-key")
#
# That keeps the key out of both the code AND your shell environment.
# Mention this in an interview — it shows you know the difference
# between "works" and "how you're supposed to do it in production."
# ──────────────────────────────────────────────────────────────────
