import os
from datetime import datetime
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DateType, StringType


def process_bronze_table(snapshot_date_str, bronze_directory, spark):
    """
    Process all raw data sources into bronze tables with proper partitioning.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    formatted_date = snapshot_date_str.replace('-', '_')

    sources = {
        "lms_loan_daily": "data/lms_loan_daily.csv",
        "feature_attributes": "data/features_attributes.csv",
        "feature_financials": "data/features_financials.csv",
        "feature_clickstream": "data/feature_clickstream.csv"
    }

    dfs = {}
    
    for name, path in sources.items():
        if name == "feature_financials":
            df = spark.read.csv(path, header=True, inferSchema=False)
            if "SSN" in df.columns:
                df = df.withColumn("SSN", col("SSN").cast(StringType()))
            if "Name" in df.columns:
                df = df.withColumn("Name", col("Name").cast(StringType()))
            if "Occupation" in df.columns:
                df = df.withColumn("Occupation", col("Occupation").cast(StringType()))
        else:
            df = spark.read.csv(path, header=True, inferSchema=True)
        
        if "snapshot_date" in df.columns:
            df = df.filter(col("snapshot_date") == snapshot_date_str)
        else:
            df = df.withColumn("snapshot_date", lit(snapshot_date).cast(DateType()))
        
        partition_name = f"bronze_{name}_{formatted_date}.csv"
        filepath = os.path.join(bronze_directory, name, partition_name)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.toPandas().to_csv(filepath, index=False)
        dfs[name] = df
    
    return dfs