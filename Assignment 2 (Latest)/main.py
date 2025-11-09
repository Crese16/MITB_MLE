# ============================================================
# MAIN PIPELINE — BRONZE → SILVER → GOLD (Local or Jupyter Mode)
# ============================================================

import os
import glob
import pandas as pd
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

# ============================================================
# Import Layer Functions
# ============================================================
from utils.data_processing_bronze_table import process_bronze_tables
from utils.data_processing_silver_table import process_silver_tables
from utils.data_processing_gold_table import process_gold

# ============================================================
# Spark Setup
# ============================================================
spark = (
    pyspark.sql.SparkSession.builder
    .appName("feature_store_pipeline_local")
    .master("local[*]")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("🚀 FEATURE STORE PIPELINE STARTED (Local Mode)\n")

# ============================================================
# CONFIGURATION
# ============================================================
START_DATE = "2023-01-01"
END_DATE = "2024-12-01"
DPD_THRESHOLD = 30  # Days Past Due threshold

bronze_directories = {
    "lms": "datamart/bronze/lms/",
    "financials": "datamart/bronze/financials/",
    "attributes": "datamart/bronze/attributes/",
    "clickstream": "datamart/bronze/clickstream/",
}

silver_directories = {
    "lms": "datamart/silver/lms/",
    "financials": "datamart/silver/financials/",
    "attributes": "datamart/silver/attributes/",
    "clickstream": "datamart/silver/clickstream/",
}

gold_feature_store_directory = "datamart/gold/feature_store/"
gold_label_store_directory = "datamart/gold/label_store/"

# Create folders if they don’t exist
for folder in [
    *bronze_directories.values(),
    *silver_directories.values(),
    gold_feature_store_directory,
    gold_label_store_directory,
]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# Generate First-of-Month Snapshot Dates
# ============================================================
def generate_first_of_month_dates(start_date_str, end_date_str):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    current = datetime(start.year, start.month, 1)
    result = []
    while current <= end:
        result.append(current.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return result

dates_str_lst = generate_first_of_month_dates(START_DATE, END_DATE)
print(f"📅 Processing {len(dates_str_lst)} months: {dates_str_lst[0]} → {dates_str_lst[-1]}\n")

# ============================================================
# BRONZE LAYER
# ============================================================
print("🟤 Starting BRONZE layer...")

for date_str in dates_str_lst:
    try:
        process_bronze_tables(
            snapshot_date_str=date_str,
            bronze_directories=bronze_directories,
            spark=spark
        )
        print(f"✅ Bronze {date_str} completed.\n")
    except Exception as e:
        print(f"❌ Bronze failed for {date_str}: {e}")

print("🏁 Bronze layer completed.\n")

# ============================================================
# SILVER LAYER
# ============================================================
print("🥈 Starting SILVER layer...")

for date_str in dates_str_lst:
    try:
        process_silver_tables(
            snapshot_date_str=date_str,
            bronze_dirs=bronze_directories,
            silver_dirs=silver_directories,
            spark=spark
        )
        print(f"✅ Silver {date_str} completed.\n")
    except Exception as e:
        print(f"❌ Silver failed for {date_str}: {e}")

print("🏁 Silver layer completed.\n")

# ============================================================
# GOLD LAYER
# ============================================================
print("🥇 Starting GOLD layer...")

for date_str in dates_str_lst:
    try:
        process_gold(
            snapshot_date_str=date_str,
            silvers=silver_directories,
            gold_features_dir=gold_feature_store_directory,
            gold_labels_dir=gold_label_store_directory,
            spark=spark,
            dpd_threshold=DPD_THRESHOLD
        )
        print(f"✅ Gold {date_str} completed.\n")
    except Exception as e:
        print(f"❌ Gold failed for {date_str}: {e}")

print("🏁 Gold layer completed.\n")

# ============================================================
# VALIDATION — Feature Store Summary
# ============================================================
print("\n📊 VALIDATION — FEATURE STORE SUMMARY\n")

try:
    feature_files = glob.glob(os.path.join(gold_feature_store_directory, "*.parquet"))
    if not feature_files:
        print("⚠️ No Gold feature store files found.")
    else:
        df = spark.read.parquet(gold_feature_store_directory + "*.parquet")
        total_rows = df.count()
        unique_customers = df.select("Customer_ID").distinct().count()
        print(f"✅ Loaded {len(feature_files)} monthly files.")
        print(f"📈 Total rows: {total_rows}")
        print(f"👥 Unique customers: {unique_customers}")

        if "label" in df.columns:
            label_dist = df.groupBy("label").count().orderBy("label").collect()
            print("\nLabel distribution:")
            for row in label_dist:
                print(f"  Label {row['label']}: {row['count']} rows ({row['count']/total_rows*100:.1f}%)")

except Exception as e:
    print(f"⚠️ Validation failed: {e}")

# ============================================================
# CLEANUP
# ============================================================
spark.stop()
print("\n✅ PIPELINE COMPLETED SUCCESSFULLY ✅")
print("=" * 80)
