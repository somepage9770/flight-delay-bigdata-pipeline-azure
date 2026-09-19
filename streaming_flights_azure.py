"""
Phase 5 — Streaming simulation (Azure version)

Same monthly micro-batch simulation as the original streaming_flights.py,
reading from / writing to Azure Data Lake Storage Gen2 instead of OCI.
Run after process_flights_azure.py has produced the clean/ layer.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, month, round

STORAGE_ACCOUNT = "flightdelaymax"
CONTAINER = "flight-delay-data"

spark = SparkSession.builder.appName("FlightStreamingSimulation").getOrCreate()

storage_key = os.environ["AZURE_STORAGE_KEY"]
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key,
)


def abfss_path(layer: str) -> str:
    return f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net/{layer}"


print("=== Simulated Streaming: Loading clean dataset ===")
df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(abfss_path("clean/flights_clean/"))
)

months = list(range(1, 13))

for m in months:
    batch = df.filter(month("FL_DATE") == m)
    batch_count = batch.count()

    if batch_count == 0:
        print(f"Month {m}: no data, skipping.")
        continue

    print(f"=== Processing batch: Month {m} | Records: {batch_count} ===")

    summary = batch.agg(
        round(avg("DEP_DELAY"), 2).alias("avg_dep_delay"),
        round(avg("ARR_DELAY"), 2).alias("avg_arr_delay"),
        count("*").alias("total_flights"),
    )
    summary.show()

    batch.write.mode("overwrite").option("header", True).csv(
        abfss_path(f"curated/streaming/month_{m}/")
    )
    print(f"Month {m} saved to curated/streaming/month_{m}/")

print("=== Streaming Simulation Completed ===")
spark.stop()
