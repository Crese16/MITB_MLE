# ============================================================
# GOLD LAYER CREATION — Monthly Feature + Label Stores
# ============================================================

import os
import shutil
import argparse
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when


# ============================================================
# Utility helpers
# ============================================================

def safe_drop(df, colname_list):
    """Safely drop one or more columns if they exist."""
    if df is None:
        return None
    for c in colname_list:
        if c in df.columns:
            df = df.drop(c)
    return df


def process_gold(snapshot_date_str, silvers, gold_features_dir, gold_labels_dir, spark, dpd_threshold=30):
    # ============================================================
    # Normalize snapshot date to first-of-month
    # ============================================================
    dt_in = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    month_start = datetime(dt_in.year, dt_in.month, 1)
    tag = month_start.strftime("%Y_%m_%d")

    print("=" * 80)
    print(f"🚀 Processing Gold Layer for Snapshot {month_start.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # ============================================================
    # Load Silver datasets
    # ============================================================
    silver_lms_path = os.path.join(silvers["lms"], f"silver_lms_{tag}.parquet")
    silver_fin_path = os.path.join(silvers["financials"], f"silver_financials_{tag}.parquet")
    silver_attr_path = os.path.join(silvers["attributes"], f"silver_attributes_{tag}.parquet")
    silver_clk_path = os.path.join(silvers["clickstream"], f"silver_clickstream_{tag}.parquet")

    if not os.path.exists(silver_lms_path):
        raise FileNotFoundError(f"❌ Silver LMS not found for snapshot {snapshot_date_str}: {silver_lms_path}")

    df_lms = spark.read.parquet(silver_lms_path)
    print(f"✅ Loaded LMS records: {df_lms.count()} rows")

    df_fin = spark.read.parquet(silver_fin_path) if os.path.exists(silver_fin_path) else None
    df_attr = spark.read.parquet(silver_attr_path) if os.path.exists(silver_attr_path) else None
    df_clk = spark.read.parquet(silver_clk_path) if os.path.exists(silver_clk_path) else None

    # ============================================================
    # Drop system columns to prevent duplicates
    # ============================================================
    system_cols = [
        "bronze_ingestion_timestamp",
        "bronze_processing_date",
        "silver_processing_date",
        "normalized_snapshot_date",
        "snapshot_date",  # 🧩 NEW FIX
    ]
    df_lms = safe_drop(df_lms, system_cols)
    df_fin = safe_drop(df_fin, system_cols)
    df_attr = safe_drop(df_attr, system_cols)
    df_clk = safe_drop(df_clk, system_cols)

    # ============================================================
    # Label store: 1 if any record has dpd >= threshold
    # ============================================================
    label_df = (
        df_lms.groupBy("Customer_ID")
        .agg(F.max(when(col("dpd") >= dpd_threshold, 1).otherwise(0)).alias("label"))
    )

    # ============================================================
    # Feature store: join across all silvers
    # ============================================================
    feat_df = df_lms.select("Customer_ID", "loan_amt", "tenure", "mob").dropDuplicates()

    if df_fin:
        feat_df = feat_df.join(df_fin, "Customer_ID", "left")
    if df_attr:
        feat_df = feat_df.join(df_attr, "Customer_ID", "left")
    if df_clk:
        feat_df = feat_df.join(df_clk, "Customer_ID", "left")

    # Drop any residual duplicate columns post-join
    feat_df = safe_drop(feat_df, ["snapshot_date", "normalized_snapshot_date"])

    # ============================================================
    # Final Gold dataset
    # ============================================================
    gold_df = feat_df.join(label_df, "Customer_ID", "left").fillna({"label": 0})
    gold_df = gold_df.withColumn("gold_processing_date", F.current_timestamp())

    # Drop again before writing
    gold_df = safe_drop(gold_df, ["snapshot_date", "normalized_snapshot_date"])

    # ============================================================
    # Write outputs (overwrite mode)
    # ============================================================
    features_out = os.path.join(gold_features_dir, f"gold_feature_store_{tag}.parquet")
    labels_out = os.path.join(gold_labels_dir, f"gold_label_store_{tag}.parquet")

    for path in [features_out, labels_out]:
        if os.path.exists(path):
            shutil.rmtree(path)

    gold_df.write.mode("overwrite").parquet(features_out)
    label_df.write.mode("overwrite").parquet(labels_out)

    print(f"✅ Saved Gold Feature Store → {features_out}")
    print(f"✅ Saved Gold Label Store → {labels_out}")
    print("=" * 80)
    print("🏁 GOLD LAYER COMPLETED SUCCESSFULLY")
    print("=" * 80)


# ============================================================
# Main entrypoint
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gold Feature & Label Stores.")
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--silver_lms_dir", required=True)
    parser.add_argument("--silver_financials_dir", required=True)
    parser.add_argument("--silver_attributes_dir", required=True)
    parser.add_argument("--silver_clickstream_dir", required=True)
    parser.add_argument("--gold_features_dir", required=True)
    parser.add_argument("--gold_labels_dir", required=True)
    parser.add_argument("--dpd_threshold", type=int, default=30)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("GoldLayer").getOrCreate()

    silvers = {
        "lms": args.silver_lms_dir,
        "financials": args.silver_financials_dir,
        "attributes": args.silver_attributes_dir,
        "clickstream": args.silver_clickstream_dir,
    }

    process_gold(
        args.snapshotdate,
        silvers,
        args.gold_features_dir,
        args.gold_labels_dir,
        spark,
        args.dpd_threshold,
    )

    spark.stop()
