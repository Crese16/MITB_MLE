import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# Import our custom processing modules
import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("datamart_pipeline") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

print("="*80)
print("DATAMART PIPELINE - MEDALLION ARCHITECTURE")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# Label definition parameters
DPD_THRESHOLD = 30  # Days past due threshold for default
MOB_LABEL = 6       # Month on book to check for default (prediction target)


# ============================================================================
# HELPER FUNCTION: Generate date list for processing
# ============================================================================
def generate_first_of_month_dates(start_date_str, end_date_str):
    """Generate list of first-of-month dates between start and end dates"""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(f"\nProcessing {len(dates_str_lst)} monthly snapshots from {start_date_str} to {end_date_str}")


# ============================================================================
# BRONZE LAYER - RAW DATA INGESTION
# ============================================================================
print("\n" + "="*80)
print("BRONZE LAYER - RAW DATA INGESTION")
print("="*80)

bronze_directory = "datamart/bronze/"
os.makedirs(bronze_directory, exist_ok=True)

for date_str in dates_str_lst:
    print(f"\nProcessing bronze for {date_str}...")
    bronze.process_bronze_table(date_str, bronze_directory, spark)


# ============================================================================
# SILVER LAYER - DATA CLEANING AND TRANSFORMATION
# ============================================================================
print("\n" + "="*80)
print("SILVER LAYER - DATA CLEANING AND TRANSFORMATION")
print("="*80)

# Create silver directories
silver_clickstream_dir = "datamart/silver/clickstream/"
silver_attributes_dir = "datamart/silver/attributes/"
silver_financials_dir = "datamart/silver/financials/"
silver_loans_dir = "datamart/silver/loans/"

for directory in [silver_clickstream_dir, silver_attributes_dir, silver_financials_dir, silver_loans_dir]:
    os.makedirs(directory, exist_ok=True)

# Define bronze source directories
bronze_clickstream_dir = "datamart/bronze/feature_clickstream/"
bronze_attributes_dir = "datamart/bronze/feature_attributes/"
bronze_financials_dir = "datamart/bronze/feature_financials/"
bronze_loans_dir = "datamart/bronze/lms_loan_daily/"

# Process silver tables
for date_str in dates_str_lst:
    print(f"\nProcessing silver for {date_str}...")
    silver.process_silver_clickstream(date_str, bronze_clickstream_dir, silver_clickstream_dir, spark)
    silver.process_silver_attributes(date_str, bronze_attributes_dir, silver_attributes_dir, spark)
    silver.process_silver_financials(date_str, bronze_financials_dir, silver_financials_dir, spark)
    silver.process_silver_loans(date_str, bronze_loans_dir, silver_loans_dir, spark)


# ============================================================================
# GOLD LAYER - FEATURE STORE AND LABEL STORE (DATAMART)
# ============================================================================
print("\n" + "="*80)
print("GOLD LAYER - DATAMART (FEATURE STORE + LABEL STORE)")
print("="*80)

# Create gold directories
gold_feature_store_dir = "datamart/gold/feature_store/"
gold_label_store_dir = "datamart/gold/label_store/"

for directory in [gold_feature_store_dir, gold_label_store_dir]:
    os.makedirs(directory, exist_ok=True)

# Process gold feature store (features at MOB=0)
print("\nCreating Feature Store (MOB=0)...")
for date_str in dates_str_lst:
    print(f"  {date_str}:")
    gold.process_gold_feature_store(
        date_str,
        silver_clickstream_dir,
        silver_attributes_dir,
        silver_financials_dir,
        silver_loans_dir,
        gold_feature_store_dir,
        spark
    )

# Process gold label store (labels at MOB=6)
print("\nCreating Label Store (MOB=6)...")
for date_str in dates_str_lst:
    print(f"  {date_str}:")
    gold.process_gold_label_store(
        date_str,
        silver_loans_dir,
        gold_label_store_dir,
        spark,
        dpd=DPD_THRESHOLD,
        mob=MOB_LABEL
    )


# ============================================================================
# VALIDATION - INSPECT GOLD TABLES (DATAMART)
# ============================================================================
print("\n" + "="*80)
print("VALIDATION - DATAMART TABLES")
print("="*80)

# Load feature store
feature_pattern = os.path.join(gold_feature_store_dir, "*.parquet")
df_features = spark.read.parquet(feature_pattern)
print(f"\nFeature Store (MOB=0):")
print(f"  Total Records: {df_features.count():,}")
print(f"  Total Features: {len(df_features.columns)}")
print(f"  Unique Loans: {df_features.select('loan_id').distinct().count():,}")
print(f"  Unique Customers: {df_features.select('Customer_ID').distinct().count():,}")
print(f"  Date Range: {df_features.agg(F.min('snapshot_date'), F.max('snapshot_date')).collect()[0]}")

# Load label store
label_pattern = os.path.join(gold_label_store_dir, "*.parquet")
df_labels = spark.read.parquet(label_pattern)
print(f"\nLabel Store (MOB=6):")
print(f"  Total Records: {df_labels.count():,}")
print(f"  Unique Loans: {df_labels.select('loan_id').distinct().count():,}")
print(f"  Unique Customers: {df_labels.select('Customer_ID').distinct().count():,}")
print(f"  Date Range: {df_labels.agg(F.min('snapshot_date'), F.max('snapshot_date')).collect()[0]}")

# Label distribution
print("\n  Label Distribution:")
df_labels.groupBy("label").count().orderBy("label").show()

# Calculate default rate
total_loans = df_labels.count()
defaulted_loans = df_labels.filter(col("label") == 1).count()
default_rate = (defaulted_loans / total_loans * 100) if total_loans > 0 else 0
print(f"  Default Rate: {default_rate:.2f}% ({defaulted_loans:,}/{total_loans:,})")

# DPD statistics
print("\n  DPD Statistics:")
df_labels.select("dpd").summary("count", "mean", "stddev", "min", "max").show()

# Summary
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nDatamart Summary:")
print(f"  Feature Store (MOB=0): {len(df_features.columns)} columns, {df_features.count():,} records")
print(f"  Label Store (MOB=6):   {len(df_labels.columns)} columns, {df_labels.count():,} records")
print(f"  Default Rate:          {default_rate:.2f}%")

print("\n" + "="*80)
print("DATA ARCHITECTURE:")
print("  Bronze  → Raw data ingestion")
print("  Silver  → Cleaned data (MOB=0 + All MOB)")
print("  Gold    → Datamart (Feature Store at MOB=0 + Label Store at MOB=6)")
print("="*80)
print("\nDatamart Location:")
print(f"  Features: {gold_feature_store_dir}")
print(f"  Labels:   {gold_label_store_dir}")
print("\nNext Steps:")
print("  1. Query the datamart for analysis and reporting")
print("  2. Create training datasets by joining features + labels with time offset")
print("  3. Build ML models using the datamart")
print("  4. Monitor data quality and refresh regularly")
print("="*80)

# Stop Spark session
spark.stop()