# ============================================================
# BRONZE LAYER CREATION — Monthly Snapshot (YYYY_MM_DD)
# ============================================================

import os
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from datetime import datetime
import argparse


def process_bronze_tables(snapshot_date_str, bronze_directories, spark):
    print("=" * 80)
    print(f"PROCESSING BRONZE TABLES for snapshot date: {snapshot_date_str}")
    print("=" * 80)

    # Snap date to first of month
    try:
        dt_in = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    except ValueError:
        dt_in = datetime.strptime(snapshot_date_str, "%d/%m/%Y")

    month_start = datetime(dt_in.year, dt_in.month, 1)
    target_date_iso = month_start.strftime("%Y-%m-%d")
    date_suffix = month_start.strftime("%Y_%m_%d")

    datasets_config = {
        "lms": {
            "file_path": "data/lms_loan_daily.csv",
            "directory": bronze_directories["lms"],
            "prefix": "bronze_loan_daily_",
            "date_column": "snapshot_date",
        },
        "financials": {
            "file_path": "data/features_financials.csv",
            "directory": bronze_directories["financials"],
            "prefix": "bronze_financials_",
            "date_column": "snapshot_date",
        },
        "attributes": {
            "file_path": "data/features_attributes.csv",
            "directory": bronze_directories["attributes"],
            "prefix": "bronze_attributes_",
            "date_column": "snapshot_date",
        },
        "clickstream": {
            "file_path": "data/feature_clickstream.csv",
            "directory": bronze_directories["clickstream"],
            "prefix": "bronze_clickstream_",
            "date_column": "snapshot_date",
        },
    }

    for dataset, cfg in datasets_config.items():
        print(f"\n--- Processing {dataset.upper()} ---")
        os.makedirs(cfg["directory"], exist_ok=True)

        if not os.path.exists(cfg["file_path"]):
            print(f"⚠️ Missing file: {cfg['file_path']}")
            continue

        df = spark.read.csv(cfg["file_path"], header=True, inferSchema=False)
        if cfg["date_column"] in df.columns:
            df = df.withColumn(
                "normalized_snapshot_date",
                F.coalesce(
                    F.to_date(col(cfg["date_column"]), "yyyy-MM-dd"),
                    F.to_date(col(cfg["date_column"]), "d/M/yyyy"),
                ),
            )
            df = df.filter(F.col("normalized_snapshot_date") == target_date_iso)
        else:
            df = df.withColumn("normalized_snapshot_date", F.lit(target_date_iso))

        df = (
            df.withColumn("bronze_ingestion_timestamp", F.current_timestamp())
            .withColumn("bronze_processing_date", F.lit(target_date_iso))
        )

        outpath = os.path.join(cfg["directory"], f"{cfg['prefix']}{date_suffix}.csv")
        df.toPandas().to_csv(outpath, index=False)
        print(f"✅ Saved {dataset} → {outpath} ({df.count()} rows)")

    print("=" * 80)
    print("✅ BRONZE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("BronzeLayer").getOrCreate()
    bronze_dirs = {
        "lms": "/opt/airflow/datamart/bronze/lms/",
        "financials": "/opt/airflow/datamart/bronze/financials/",
        "attributes": "/opt/airflow/datamart/bronze/attributes/",
        "clickstream": "/opt/airflow/datamart/bronze/clickstream/",
    }
    process_bronze_tables(args.snapshotdate, bronze_dirs, spark)
    spark.stop()
