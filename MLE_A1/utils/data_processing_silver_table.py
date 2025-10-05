import os
from datetime import datetime
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType
import pyspark.sql.functions as F


def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Clean and transform clickstream data.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    formatted_date = snapshot_date_str.replace('-', '_')
    
    partition_name = f"bronze_feature_clickstream_{formatted_date}.csv"
    filepath = os.path.join(bronze_clickstream_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=False)

    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column in df.columns:
        if column in column_type_map:
            df = df.withColumn(column, col(column).cast(column_type_map[column]))
        elif column not in ["Customer_ID", "snapshot_date"]:
            df = df.withColumn(column, col(column).cast(DoubleType()))

    df = df.filter(col("Customer_ID").isNotNull())
    
    numeric_columns = [c for c in df.columns if c not in ["Customer_ID", "snapshot_date"]]
    df = df.fillna({col_name: 0.0 for col_name in numeric_columns})

    partition_name = f"silver_clickstream_{formatted_date}.parquet"
    filepath = os.path.join(silver_clickstream_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    
    return df


def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Clean and transform customer attributes.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    formatted_date = snapshot_date_str.replace('-', '_')
    
    partition_name = f"bronze_feature_attributes_{formatted_date}.csv"
    filepath = os.path.join(bronze_attributes_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=False)

    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
        "SSN": StringType(),
        "Name": StringType(),
        "Occupation": StringType(),
    }

    for column in df.columns:
        if column in column_type_map:
            df = df.withColumn(column, col(column).cast(column_type_map[column]))
        elif column not in ["Customer_ID", "snapshot_date"]:
            try:
                df = df.withColumn(column, col(column).cast(DoubleType()))
            except:
                df = df.withColumn(column, col(column).cast(StringType()))

    df = df.filter(col("Customer_ID").isNotNull())

    numeric_columns = [c for c in df.columns if c not in ["Customer_ID", "snapshot_date", "SSN", "Name", "Occupation"] 
                      and df.schema[c].dataType in [IntegerType(), FloatType(), DoubleType()]]
    df = df.fillna({col_name: 0 for col_name in numeric_columns})

    partition_name = f"silver_attributes_{formatted_date}.parquet"
    filepath = os.path.join(silver_attributes_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    
    return df


def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Clean and transform financial data.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    formatted_date = snapshot_date_str.replace('-', '_')
    
    partition_name = f"bronze_feature_financials_{formatted_date}.csv"
    filepath = os.path.join(bronze_financials_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    
    if "SSN" in df.columns:
        df = df.withColumn("SSN", col("SSN").cast(StringType()))

    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
        "Name": StringType(),
        "Occupation": StringType(),
        "Age": DoubleType(),
        "Annual_Income": DoubleType(),
        "Monthly_Inhand_Salary": DoubleType(),
        "Num_Bank_Accounts": DoubleType(),
        "Num_Credit_Card": DoubleType(),
        "Interest_Rate": DoubleType(),
        "Num_of_Loan": DoubleType(),
        "Type_of_Loan": DoubleType(),
        "Delay_from_due_date": DoubleType(),
        "Num_of_Delayed_Payment": DoubleType(),
        "Changed_Credit_Limit": DoubleType(),
        "Num_Credit_Inquiries": DoubleType(),
        "Credit_Mix": DoubleType(),
        "Outstanding_Debt": DoubleType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "Credit_History_Age": DoubleType(),
        "Payment_of_Min_Amount": DoubleType(),
        "Total_EMI_per_month": DoubleType(),
        "Amount_invested_monthly": DoubleType(),
        "Payment_Behaviour": DoubleType(),
        "Monthly_Balance": DoubleType(),
    }

    for column in df.columns:
        if column == "SSN":
            continue
        elif column in column_type_map:
            df = df.withColumn(column, col(column).cast(column_type_map[column]))
        elif column not in ["Customer_ID", "snapshot_date", "Name", "Occupation"]:
            df = df.withColumn(column, col(column).cast(DoubleType()))

    df = df.filter(col("Customer_ID").isNotNull())
    
    numeric_columns = [c for c in df.columns 
                      if c not in ["Customer_ID", "snapshot_date", "Name", "SSN", "Occupation"]]
    df = df.fillna({col_name: 0.0 for col_name in numeric_columns})

    partition_name = f"silver_financials_{formatted_date}.parquet"
    filepath = os.path.join(silver_financials_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    
    return df


def process_silver_loans(snapshot_date_str, bronze_loans_directory, silver_loans_directory, spark):
    """
    Clean and transform loan data with DPD calculation. Save two versions:
    1. silver_loans_all_{date}.parquet - ALL MOB data with DPD for analysis
    2. silver_loans_mob0_{date}.parquet - Only MOB=0 for feature store
    
    DPD (Days Past Due) Calculation Logic:
    - Calculate installments_missed = ceiling(overdue_amt / due_amt)
    - Estimate first_missed_date by going back N months
    - Calculate DPD = days between snapshot_date and first_missed_date
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    formatted_date = snapshot_date_str.replace('-', '_')
    
    partition_name = f"bronze_lms_loan_daily_{formatted_date}.csv"
    filepath = os.path.join(bronze_loans_directory, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)

    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # Add MOB column
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    
    # Filter out invalid records
    df = df.filter(col("loan_id").isNotNull() & col("Customer_ID").isNotNull())
    df = df.filter(col("loan_amt") > 0)
    
    # ========================================================================
    # CALCULATE DPD (Days Past Due)
    # ========================================================================
    
    # Step 1: Calculate installments missed
    # Handle division by zero - if due_amt is 0, set installments_missed to 0
    df = df.withColumn("installments_missed", 
                       when(col("due_amt") > 0,
                            F.ceil(col("overdue_amt") / col("due_amt")))
                       .otherwise(0)
                       .cast(IntegerType()))
    
    # Step 2: Estimate first missed payment date
    # Go back N months from snapshot_date where N = installments_missed
    df = df.withColumn("first_missed_date", 
                       when(col("installments_missed") > 0, 
                            F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
                       .cast(DateType()))
    
    # Step 3: Calculate DPD as days between snapshot_date and first_missed_date
    df = df.withColumn("dpd", 
                       when(col("overdue_amt") > 0.0, 
                            F.datediff(col("snapshot_date"), col("first_missed_date")))
                       .otherwise(0)
                       .cast(IntegerType()))
    
    # Ensure DPD is never negative (data quality check)
    df = df.withColumn("dpd", 
                       when(col("dpd") < 0, 0)
                       .otherwise(col("dpd")))
    
    # ========================================================================
    # SAVE TWO VERSIONS
    # ========================================================================
    
    # Save ALL MOB data with DPD (needed for MOB analysis and label calculation)
    filepath_all = os.path.join(silver_loans_directory, f"silver_loans_all_{formatted_date}.parquet")
    df.write.mode("overwrite").parquet(filepath_all)
    
    # Filter and save MOB=0 only (needed for feature store - application time)
    df_mob0 = df.filter(col("mob") == 0)
    filepath_mob0 = os.path.join(silver_loans_directory, f"silver_loans_mob0_{formatted_date}.parquet")
    df_mob0.write.mode("overwrite").parquet(filepath_mob0)
    
    return df