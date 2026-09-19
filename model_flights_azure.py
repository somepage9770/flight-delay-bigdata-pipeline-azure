"""
Phase 8 — Prediction model (Azure version)

Same custom Naive Bayes log-odds classifier as the original
model_flights3.py — no MLlib, no scikit-learn — just reading from and
writing to Azure Data Lake Storage Gen2 instead of OCI.
"""

import os
import math

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, month, dayofweek, log, lit
from pyspark.sql.functions import round as spark_round

STORAGE_ACCOUNT = "flightdelaymax"        # <-- your storage account name
CONTAINER = "flight-delay-data"           # <-- your ADLS Gen2 container name

spark = (
    SparkSession.builder.appName("FlightDelayPrediction")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

storage_key = os.environ["AZURE_STORAGE_KEY"]
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key,
)


def abfss_path(layer: str) -> str:
    return f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net/{layer}"


# STEP 1: Load data
print("=== STEP 1: Loading clean dataset ===")
df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(abfss_path("clean/flights_clean/"))
)
print("Total records:", df.count())

# STEP 2: Feature engineering
print("=== STEP 2: Feature engineering ===")
df = (
    df.withColumn("DEP_HOUR", (col("CRS_DEP_TIME") / 100).cast("int"))
    .withColumn("FL_MONTH", month(col("FL_DATE")))
    .withColumn("FL_DOW", dayofweek(col("FL_DATE")))
    .withColumn("DIST_BUCKET", ((col("DISTANCE") - 1) / 500).cast("int"))
)

# STEP 3: Train/validation/test split
print("=== STEP 3: Train/validation/test split (70/15/15) ===")
train_df, validation_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=42)
train_df.cache()
validation_df.cache()
n_train = train_df.count()
n_validation = validation_df.count()
n_test = test_df.count()
print(f"Train: {n_train:,} | Validation: {n_validation:,} | Test: {n_test:,}")

# STEP 4: Global delay rate
print("=== STEP 4: Global delay rate ===")
n_delayed_train = train_df.filter(col("IS_DELAYED") == 1).count()
global_p = n_delayed_train / n_train
print(f"Global delay rate: {global_p:.4f}")

prior_lo = math.log(global_p / (1.0 - global_p))
print(f"Prior log-odds: {prior_lo:.4f}")

prior_strength = 50.0
NUM_FEATURES = 12


# STEP 5: Per-feature conditional log-odds
print("=== STEP 5: Per-feature conditional log-odds ===")


def log_odds_table(df_tr, group_cols, alias):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    return (
        df_tr.groupBy(*group_cols)
        .agg(
            count("*").alias("_n"),
            count(when(col("IS_DELAYED") == 1, 1)).alias("_nd"),
        )
        .withColumn(
            "_smoothed_p",
            (col("_nd") + lit(global_p * prior_strength))
            / (col("_n") + lit(prior_strength)),
        )
        .withColumn(alias, log(col("_smoothed_p") / (lit(1.0) - col("_smoothed_p"))))
        .drop("_n", "_nd", "_smoothed_p")
    )


airline_lo = log_odds_table(train_df, "AIRLINE_CODE", "lo_airline")
route_lo = log_odds_table(train_df, ["ORIGIN", "DEST"], "lo_route")
hour_lo = log_odds_table(train_df, "DEP_HOUR", "lo_hour")
month_lo = log_odds_table(train_df, "FL_MONTH", "lo_month")
dow_lo = log_odds_table(train_df, "FL_DOW", "lo_dow")
dist_lo = log_odds_table(train_df, "DIST_BUCKET", "lo_dist")
origin_lo = log_odds_table(train_df, "ORIGIN", "lo_origin")
dest_lo = log_odds_table(train_df, "DEST", "lo_dest")

airline_hour_lo = log_odds_table(train_df, ["AIRLINE_CODE", "DEP_HOUR"], "lo_airline_hour")
origin_hour_lo = log_odds_table(train_df, ["ORIGIN", "DEP_HOUR"], "lo_origin_hour")
route_month_lo = log_odds_table(train_df, ["ORIGIN", "DEST", "FL_MONTH"], "lo_route_month")
airline_dow_lo = log_odds_table(train_df, ["AIRLINE_CODE", "FL_DOW"], "lo_airline_dow")

# STEP 6: Risk profiles (for curated layer)
print("=== STEP 6: Risk profiles ===")
airline_risk = train_df.groupBy("AIRLINE_CODE").agg(
    spark_round(avg("DEP_DELAY"), 2).alias("airline_avg_delay"),
    spark_round((count(when(col("IS_DELAYED") == 1, 1)) / count("*") * 100), 2).alias(
        "airline_delay_rate"
    ),
)
route_risk = train_df.groupBy("ORIGIN", "DEST").agg(
    spark_round(avg("DEP_DELAY"), 2).alias("route_avg_delay"),
    spark_round((count(when(col("IS_DELAYED") == 1, 1)) / count("*") * 100), 2).alias(
        "route_delay_rate"
    ),
)
hour_risk = train_df.groupBy("DEP_HOUR").agg(
    spark_round(avg("DEP_DELAY"), 2).alias("hour_avg_delay"),
    spark_round((count(when(col("IS_DELAYED") == 1, 1)) / count("*") * 100), 2).alias(
        "hour_delay_rate"
    ),
)
month_risk = train_df.groupBy("FL_MONTH").agg(
    spark_round(avg("DEP_DELAY"), 2).alias("month_avg_delay"),
    spark_round((count(when(col("IS_DELAYED") == 1, 1)) / count("*") * 100), 2).alias(
        "month_delay_rate"
    ),
)

hour_risk.orderBy("hour_delay_rate", ascending=False).show()
month_risk.orderBy("FL_MONTH").show()

# STEP 7: Scoring function
prior_correction = (NUM_FEATURES - 1) * prior_lo
print(f"Prior correction term: {prior_correction:.4f}")

LO_FILL = {
    "lo_airline": prior_lo, "lo_route": prior_lo, "lo_hour": prior_lo,
    "lo_month": prior_lo, "lo_dow": prior_lo, "lo_dist": prior_lo,
    "lo_origin": prior_lo, "lo_dest": prior_lo,
    "lo_airline_hour": prior_lo, "lo_origin_hour": prior_lo,
    "lo_route_month": prior_lo, "lo_airline_dow": prior_lo,
}


def score_dataset(dataset):
    return (
        dataset.join(airline_lo, on="AIRLINE_CODE", how="left")
        .join(route_lo, on=["ORIGIN", "DEST"], how="left")
        .join(hour_lo, on="DEP_HOUR", how="left")
        .join(month_lo, on="FL_MONTH", how="left")
        .join(dow_lo, on="FL_DOW", how="left")
        .join(dist_lo, on="DIST_BUCKET", how="left")
        .join(origin_lo, on="ORIGIN", how="left")
        .join(dest_lo, on="DEST", how="left")
        .join(airline_hour_lo, on=["AIRLINE_CODE", "DEP_HOUR"], how="left")
        .join(origin_hour_lo, on=["ORIGIN", "DEP_HOUR"], how="left")
        .join(route_month_lo, on=["ORIGIN", "DEST", "FL_MONTH"], how="left")
        .join(airline_dow_lo, on=["AIRLINE_CODE", "FL_DOW"], how="left")
        .fillna(LO_FILL)
        .withColumn(
            "SCORE",
            col("lo_airline") + col("lo_route") + col("lo_hour")
            + col("lo_month") + col("lo_dow") + col("lo_dist")
            + col("lo_origin") + col("lo_dest")
            + col("lo_airline_hour") + col("lo_origin_hour")
            + col("lo_route_month") + col("lo_airline_dow")
            - lit(prior_correction),
        )
    )


# STEP 8: Threshold search on validation data
print("=== STEP 8: Optimizing threshold on validation data ===")
train_scored = score_dataset(train_df)
validation_scored = score_dataset(validation_df)
train_scored.cache()
validation_scored.cache()

probs = [i / 40.0 for i in range(1, 40)]
thresholds = validation_scored.approxQuantile("SCORE", probs, 0.002)

agg_exprs = []
for i, t in enumerate(thresholds):
    agg_exprs += [
        count(when((col("IS_DELAYED") == 1) & (col("SCORE") > t), 1)).alias(f"tp_{i}"),
        count(when((col("IS_DELAYED") == 0) & (col("SCORE") > t), 1)).alias(f"fp_{i}"),
        count(when((col("IS_DELAYED") == 1) & (col("SCORE") <= t), 1)).alias(f"fn_{i}"),
        count(when((col("IS_DELAYED") == 0) & (col("SCORE") <= t), 1)).alias(f"tn_{i}"),
    ]

stats = validation_scored.agg(*agg_exprs).collect()[0]

best_threshold = 0.0
best_bal_acc = 0.0
print("\nThreshold search (validation data):")
for i, t in enumerate(thresholds):
    tp_ = stats[f"tp_{i}"]; fp_ = stats[f"fp_{i}"]
    fn_ = stats[f"fn_{i}"]; tn_ = stats[f"tn_{i}"]
    total_ = tp_ + fp_ + fn_ + tn_
    acc_ = (tp_ + tn_) / total_ if total_ > 0 else 0
    prec_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
    rec_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
    f1_ = 2 * prec_ * rec_ / (prec_ + rec_) if (prec_ + rec_) > 0 else 0
    bal_acc = (
        ((tp_ / (tp_ + fn_)) + (tn_ / (tn_ + fp_))) / 2
        if (tp_ + fn_) > 0 and (tn_ + fp_) > 0
        else 0
    )
    print(f"  t={t:7.3f}  acc={acc_:.4f}  bal_acc={bal_acc:.4f}  f1={f1_:.4f}  prec={prec_:.4f}  rec={rec_:.4f}")
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_threshold = t

print(f"\nBest threshold: {best_threshold:.4f}  (validation bal_acc={best_bal_acc*100:.2f}%)")

# STEP 9: Evaluate on test set
print("=== STEP 9: Evaluating on test set ===")
test_scored = score_dataset(test_df).withColumn(
    "PREDICTED_DELAY", when(col("SCORE") > best_threshold, 1).otherwise(0)
)

total_t = test_scored.count()
tp = test_scored.filter((col("IS_DELAYED") == 1) & (col("PREDICTED_DELAY") == 1)).count()
tn = test_scored.filter((col("IS_DELAYED") == 0) & (col("PREDICTED_DELAY") == 0)).count()
fp = test_scored.filter((col("IS_DELAYED") == 0) & (col("PREDICTED_DELAY") == 1)).count()
fn = test_scored.filter((col("IS_DELAYED") == 1) & (col("PREDICTED_DELAY") == 0)).count()

accuracy = (tp + tn) / total_t * 100
precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
balanced_accuracy = (
    ((tp / (tp + fn)) if (tp + fn) > 0 else 0) + ((tn / (tn + fp)) if (tn + fp) > 0 else 0)
) / 2 * 100
baseline_accuracy = (tn + fp) / total_t * 100 if total_t > 0 else 0

print(f"\nTest Results:")
print(f"  Total records:  {total_t:,}")
print(f"  Accuracy:       {accuracy:.2f}%")
print(f"  Baseline Acc.:  {baseline_accuracy:.2f}%")
print(f"  Balanced Acc.:  {balanced_accuracy:.2f}%")
print(f"  Precision:      {precision:.2f}%")
print(f"  Recall:         {recall:.2f}%")
print(f"  F1 Score:       {f1_score:.2f}%")
print(f"\n  Confusion Matrix:")
print(f"    TP={tp:,}  FP={fp:,}")
print(f"    FN={fn:,}  TN={tn:,}")

# STEP 10: Save results
print("=== STEP 10: Saving results ===")

airline_risk.orderBy("airline_delay_rate", ascending=False).write.mode(
    "overwrite"
).option("header", True).csv(abfss_path("curated/airline_risk_profile/"))

route_risk.filter(col("route_avg_delay").isNotNull()).orderBy(
    "route_delay_rate", ascending=False
).write.mode("overwrite").option("header", True).csv(
    abfss_path("curated/route_risk_profile/")
)

hour_risk.orderBy("DEP_HOUR").write.mode("overwrite").option("header", True).csv(
    abfss_path("curated/hour_risk_profile/")
)

month_risk.orderBy("FL_MONTH").write.mode("overwrite").option("header", True).csv(
    abfss_path("curated/month_risk_profile/")
)

test_scored.select(
    "AIRLINE_CODE", "ORIGIN", "DEST", "FL_MONTH", "DEP_HOUR",
    "DISTANCE", "SCORE", "IS_DELAYED", "PREDICTED_DELAY",
).write.mode("overwrite").option("header", True).csv(abfss_path("curated/predictions/"))

print(
    f"=== MODEL COMPLETED - Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% "
    f"| Recall: {recall:.2f}% | F1: {f1_score:.2f}% ==="
)
spark.stop()
