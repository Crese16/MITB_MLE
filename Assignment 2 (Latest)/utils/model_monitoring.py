# ============================================================
# MODEL MONITORING — PSI, KS-test & Drift Alerts
# Auto-fallback if OOT not found
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import ks_2samp
from pyspark.sql import SparkSession

# -------------------------------
# CONFIG PATHS
# -------------------------------
OOT_DIR = "/opt/airflow/datamart/gold/oot/"
GOLD_DIR = "/opt/airflow/datamart/gold/feature_store/"
PRED_DIR = "/opt/airflow/datamart/predictions/"
ALERT_LOG = "/opt/airflow/datamart/monitoring/alerts.log"
os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def latest_file(path, suffix=".parquet"):
    """Return the latest .parquet file path, or None if not found."""
    if not os.path.exists(path):
        print(f"⚠️ Directory not found: {path}")
        return None
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(suffix)]
    if not files:
        print(f"⚠️ No {suffix} files found in {path}")
        return None
    latest = max(files, key=os.path.getctime)
    print(f"✅ Using file: {latest}")
    return latest

def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index (PSI) for numerical columns."""
    def scale_range(input_arr):
        return (input_arr - np.min(input_arr)) / (np.max(input_arr) - np.min(input_arr) + 1e-8)
    
    expected_perc, _ = np.histogram(scale_range(expected), bins=buckets)
    actual_perc, _ = np.histogram(scale_range(actual), bins=buckets)
    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)
    psi = np.sum((actual_perc - expected_perc) * np.log((actual_perc + 1e-8) / (expected_perc + 1e-8)))
    return psi

def compute_drift_metrics(df_ref, df_curr):
    """Compute PSI + KS-test for overlapping numeric columns."""
    results = []
    numeric_cols = [c for c in df_ref.columns if c in df_curr.columns and np.issubdtype(df_ref[c].dtype, np.number)]
    for c in numeric_cols:
        ref_vals = df_ref[c].dropna()
        cur_vals = df_curr[c].dropna()
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        psi_val = calculate_psi(ref_vals.values, cur_vals.values)
        ks_val, ks_p = ks_2samp(ref_vals.values, cur_vals.values)
        results.append((c, psi_val, ks_val, ks_p))
    return pd.DataFrame(results, columns=["feature", "psi", "ks_stat", "ks_pvalue"])

def log_alerts(df):
    """Append drift alerts to log file."""
    if df.empty:
        with open(ALERT_LOG, "a") as f:
            f.write(f"[{datetime.utcnow()}] ✅ No drift detected.\n")
        print("✅ No drift detected.")
        return

    alerts = df[(df["psi"] > 0.2) | (df["ks_pvalue"] < 0.05)]
    if alerts.empty:
        with open(ALERT_LOG, "a") as f:
            f.write(f"[{datetime.utcnow()}] ✅ Metrics stable.\n")
        print("✅ Metrics stable.")
        return

    with open(ALERT_LOG, "a") as f:
        f.write(f"\n[{datetime.utcnow()}] 🚨 Drift detected in:\n")
        for _, row in alerts.iterrows():
            f.write(f" - {row.feature}: PSI={row.psi:.3f}, KS_p={row.ks_pvalue:.3f}\n")
    print("🚨 Drift detected:")
    print(alerts)


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ModelMonitoring").getOrCreate()

    oot_path = latest_file(OOT_DIR)
    gold_path = latest_file(GOLD_DIR)
    pred_path = latest_file(PRED_DIR)

    # 1️⃣ Load reference (training or gold) data
    ref_path = oot_path or gold_path
    if not ref_path:
        raise FileNotFoundError("❌ No OOT or Gold parquet files found for reference.")
    df_ref = spark.read.parquet(ref_path).toPandas()

    # 2️⃣ Load latest inference results
    if not pred_path:
        raise FileNotFoundError("❌ No prediction files found.")
    df_pred = spark.read.parquet(pred_path).toPandas()

    print(f"✅ Reference sample: {df_ref.shape}, Predictions: {df_pred.shape}")

    # 3️⃣ Merge on Customer_ID if possible
    if "Customer_ID" in df_ref.columns and "Customer_ID" in df_pred.columns:
        df_merged = df_ref.merge(df_pred, on="Customer_ID", how="inner", suffixes=("_ref", "_pred"))
    else:
        df_merged = df_ref.copy()

    # 4️⃣ Compute drift metrics
    print("⚙️ Computing PSI and KS-test for numeric features...")
    drift_df = compute_drift_metrics(df_ref, df_merged)
    print(drift_df.head(10))

    # 5️⃣ Log alerts if drift detected
    log_alerts(drift_df)

    spark.stop()
    print("✅ Monitoring completed successfully.")
