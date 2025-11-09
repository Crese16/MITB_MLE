# ============================================================
# SILVER LAYER CREATION — Monthly Snapshot (YYYY_MM_DD)
# ============================================================

import os, shutil, pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType, DoubleType
from datetime import datetime
import argparse


def process_silver_tables(snapshot_date_str, bronze_dirs, silver_dirs, spark):
    print("=" * 80)
    print(f"PROCESSING SILVER TABLES for snapshot date: {snapshot_date_str}")
    print("=" * 80)

    try:
        dt_in = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    except ValueError:
        dt_in = datetime.strptime(snapshot_date_str, "%d/%m/%Y")

    month_start = datetime(dt_in.year, dt_in.month, 1)
    date_suffix = month_start.strftime("%Y_%m_%d")
    snapshot_date_str = month_start.strftime("%Y-%m-%d")

    datasets = ["lms", "financials", "attributes", "clickstream"]

    for ds in datasets:
        bronze_file = f"bronze_{ds if ds!='lms' else 'loan_daily'}_{date_suffix}.csv"
        bronze_path = os.path.join(bronze_dirs[ds], bronze_file)
        if not os.path.exists(bronze_path):
            print(f"⚠️ Missing Bronze: {bronze_path}")
            continue

        df = spark.read.csv(bronze_path, header=True, inferSchema=True)
        print(f"✅ Loaded {ds}: {df.count()} rows")

        if ds == "lms":
            df = process_lms(df)
        elif ds == "clickstream":
            df = process_click(df)
        elif ds == "financials":
            df = process_fin(df)
        elif ds == "attributes":
            df = process_attr(df)

        df = df.withColumn("silver_processing_date", F.lit(snapshot_date_str))
        outpath = os.path.join(silver_dirs[ds], f"silver_{ds}_{date_suffix}.parquet")
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        df.write.mode("overwrite").parquet(outpath)
        print(f"✅ Saved {ds} → {outpath}")

    print("=" * 80)
    print("✅ SILVER COMPLETE")
    print("=" * 80)


def process_lms(df):
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn("dpd", when(col("overdue_amt") > 0, 30).otherwise(0))
    return df


def process_fin(df):
    return df


def process_attr(df):
    return df


def process_click(df):
    feat_cols = [c for c in df.columns if c.startswith("fe_")]
    for c in feat_cols:
        df = df.withColumn(c, col(c).cast(DoubleType()))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("SilverLayer").getOrCreate()
    bronze_dirs = {
        "lms": "/opt/airflow/datamart/bronze/lms/",
        "financials": "/opt/airflow/datamart/bronze/financials/",
        "attributes": "/opt/airflow/datamart/bronze/attributes/",
        "clickstream": "/opt/airflow/datamart/bronze/clickstream/",
    }
    silver_dirs = {
        "lms": "/opt/airflow/datamart/silver/lms/",
        "financials": "/opt/airflow/datamart/silver/financials/",
        "attributes": "/opt/airflow/datamart/silver/attributes/",
        "clickstream": "/opt/airflow/datamart/silver/clickstream/",
    }
    process_silver_tables(args.snapshotdate, bronze_dirs, silver_dirs, spark)
    spark.stop()
