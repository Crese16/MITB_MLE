import os
from datetime import datetime
from pyspark.sql.functions import col, when, lit, count
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
import pyspark.sql.functions as F


def process_gold_feature_store(snapshot_date_str, 
                                silver_clickstream_directory,
                                silver_attributes_directory, 
                                silver_financials_directory,
                                silver_loans_directory,
                                gold_feature_store_directory, 
                                spark):
    """
    Create gold-level feature store by joining all feature sources at MOB=0.
    
    Features at MOB=0 represent information available at loan application time.
    This prevents temporal leakage by ensuring we only use data available
    when the lending decision must be made.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    formatted_date = snapshot_date_str.replace('-', '_')
    
    # Load MOB=0 loan data (application-time base features)
    df_loans = spark.read.parquet(os.path.join(silver_loans_directory, f"silver_loans_mob0_{formatted_date}.parquet"))
    
    # Load customer features (all at application time)
    df_clickstream = spark.read.parquet(os.path.join(silver_clickstream_directory, f"silver_clickstream_{formatted_date}.parquet"))
    df_attributes = spark.read.parquet(os.path.join(silver_attributes_directory, f"silver_attributes_{formatted_date}.parquet"))
    df_financials = spark.read.parquet(os.path.join(silver_financials_directory, f"silver_financials_{formatted_date}.parquet"))
    
    # Start with loan base features
    df_features = df_loans.select("loan_id", "Customer_ID", "loan_amt", "tenure", "snapshot_date")
    
    # Left join customer features (some customers may not have all feature types)
    df_features = df_features.join(df_clickstream.drop("snapshot_date"), on="Customer_ID", how="left")
    df_features = df_features.join(df_attributes.drop("snapshot_date"), on="Customer_ID", how="left")
    df_features = df_features.join(df_financials.drop("snapshot_date"), on="Customer_ID", how="left")
    
    # Add metadata columns
    df_features = df_features.withColumn("feature_date", col("snapshot_date"))
    df_features = df_features.withColumn("feature_version", lit("v1").cast(StringType()))
    
    # Ensure we have valid records
    df_features = df_features.filter(col("loan_id").isNotNull() & col("Customer_ID").isNotNull())
    
    # Save feature store
    filepath = os.path.join(gold_feature_store_directory, f"gold_feature_store_{formatted_date}.parquet")
    df_features.write.mode("overwrite").parquet(filepath)
    
    print(f"  Feature store created: {df_features.count()} records with {len(df_features.columns)} features")
    
    return df_features


def process_gold_label_store(snapshot_date_str, 
                              silver_loans_directory,
                              gold_label_store_directory, 
                              spark, 
                              dpd=30, 
                              mob=6):
    """
    Create gold-level label store from silver loan data at MOB=6.
    
    Labels at MOB=6 represent the outcome we want to predict - whether
    the loan defaulted within 6 months. This is calculated using loan
    performance data observed at the 6-month mark.
    
    Parameters:
    - dpd: Days past due threshold for default (default=30)
    - mob: Month on book to check for default (default=6)
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    formatted_date = snapshot_date_str.replace('-', '_')
    
    # Read ALL loan data from silver layer (contains all MOB records)
    partition_name = f"silver_loans_all_{formatted_date}.parquet"
    filepath = os.path.join(silver_loans_directory, partition_name)
    df = spark.read.parquet(filepath)
    
    # Filter for MOB=6 to get outcome labels
    df = df.filter(col("mob") == mob)
    
    # Calculate days past due (DPD)
    # DPD logic: If overdue_amt > 0, calculate how many days since first missed payment
    df = df.withColumn("installments_missed", 
                       F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    
    df = df.withColumn("first_missed_date", 
                       when(col("installments_missed") > 0, 
                            F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
                       .cast(DateType()))
    
    df = df.withColumn("dpd", 
                       when(col("overdue_amt") > 0.0, 
                            F.datediff(col("snapshot_date"), col("first_missed_date")))
                       .otherwise(0).cast(IntegerType()))

    # Create binary default label (1 = default, 0 = good)
    df = df.withColumn("label", 
                       when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    
    # Add label definition metadata
    df = df.withColumn("label_def", 
                       lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    # Select relevant columns for label store
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date", "dpd")

    # Save label store
    filepath = os.path.join(gold_label_store_directory, f"gold_label_store_{formatted_date}.parquet")
    df.write.mode("overwrite").parquet(filepath)
    
    # Print summary statistics
    total = df.count()
    defaults = df.filter(col("label") == 1).count()
    default_rate = (defaults / total * 100) if total > 0 else 0
    print(f"  Label store created: {total} records, {defaults} defaults ({default_rate:.2f}%)")
    
    return df


def create_training_dataset(gold_feature_store_directory,
                            gold_label_store_directory,
                            gold_training_directory,
                            spark):
    """
    Create training dataset by joining features (MOB=0) with labels (MOB=6).
    Uses time-shifted join: MOB=0 from month X joins with MOB=6 from month X+6.
    """
    
    # Load ALL feature store files (MOB=0)
    feature_pattern = os.path.join(gold_feature_store_directory, "*.parquet")
    df_features = spark.read.parquet(feature_pattern)
    
    # Load ALL label store files (MOB=6)
    label_pattern = os.path.join(gold_label_store_directory, "*.parquet")
    df_labels = spark.read.parquet(label_pattern)
    
    # Add month offset to feature snapshot_date (MOB=0 + 6 months = MOB=6)
    df_features = df_features.withColumn(
        "label_snapshot_date",
        F.add_months(col("snapshot_date"), 6)
    )
    
    # Join on loan_id, Customer_ID, and time-shifted date
    df_training = df_features.join(
        df_labels.select(
            "loan_id", 
            "Customer_ID", 
            col("snapshot_date").alias("label_snapshot_date"),
            "label", 
            "label_def", 
            "dpd"
        ),
        on=["loan_id", "Customer_ID", "label_snapshot_date"],
        how="inner"
    )
    
    # Rename columns for clarity
    df_training = df_training.withColumnRenamed("snapshot_date", "feature_date")
    df_training = df_training.withColumnRenamed("label_snapshot_date", "label_date")
    
    # Save combined training dataset
    training_path = os.path.join(gold_training_directory, "gold_training_dataset.parquet")
    df_training.write.mode("overwrite").parquet(training_path)
    
    # Print statistics
    feature_count = df_features.count()
    label_count = df_labels.count()
    training_count = df_training.count()
    
    print(f"\n  Training Dataset Join Summary:")
    print(f"    Features (MOB=0): {feature_count:,} records")
    print(f"    Labels (MOB=6):   {label_count:,} records")
    print(f"    Matched:          {training_count:,} records")
    print(f"    Match rate:       {(training_count/feature_count*100):.1f}% of features")
    
    if training_count > 0:
        defaults = df_training.filter(col("label") == 1).count()
        default_rate = (defaults / training_count * 100)
        print(f"    Default rate:     {default_rate:.2f}% ({defaults:,} defaults)")
    
    return df_training