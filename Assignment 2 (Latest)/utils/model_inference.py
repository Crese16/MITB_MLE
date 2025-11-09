# ============================================================
# Inference — schema-safe: auto-add missing features expected by the model
# ============================================================

import os, glob, sys
from datetime import datetime
import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml import PipelineModel

from pyspark.sql import SparkSession

def get_latest_gold_feature_store(base_dir: str):
    paths = glob.glob(os.path.join(base_dir, "gold_feature_store_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No gold_feature_store_*.parquet found under {base_dir}")
    # pick latest by ctime
    latest = max(paths, key=os.path.getctime)
    return latest

def get_expected_input_cols(pipeline_model: PipelineModel):
    # Find first VectorAssembler in stages; if multiple, union their inputs
    expected = []
    for st in pipeline_model.stages:
        cname = st.__class__.__name__
        # Works for VectorAssembler and also SQLTransformer isn't needed
        if cname == "VectorAssembler":
            expected.extend(st.getInputCols())
    return list(dict.fromkeys(expected))  # de-dup, keep order

def ensure_columns(df, required_cols):
    """
    Ensure all required columns exist and are DoubleType for assembler.
    Missing columns filled with 0.0. Non-numeric are cast to Double where possible.
    """
    missing = [c for c in required_cols if c not in df.columns]
    for c in missing:
        df = df.withColumn(c, F.lit(0.0))
    # Cast all required to DoubleType
    for c in required_cols:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
    if missing:
        print(f"⚠️ Added missing columns with defaults: {missing}")
    return df

def maybe_map_equivalents(df):
    """
    Optional: map close equivalents if present (rename/copy).
    Example: if model expects 'num_applications_this_month' but we only have 'num_loans'.
    Adjust this mapping as your schema evolves.
    """
    mappings = {
        # "model_expected_col": "existing_close_col"
        # e.g. "num_applications_this_month": "num_loans",
    }
    for target, source in mappings.items():
        if target not in df.columns and source in df.columns:
            df = df.withColumn(target, F.col(source))
    return df

if __name__ == "__main__":
    # CLI: python model_inference.py "YYYY-MM-DD"
    snapshot_date = sys.argv[1] if len(sys.argv) > 1 else None

    spark = (SparkSession.builder
             .appName("ModelInference")
             .getOrCreate())

    # -------------------------------
    # 1) Load trained pipeline model
    # -------------------------------
    # Adjust if you maintain a symlink 'latest' -> model folder.
    model_dir = "/opt/airflow/model_bank"
    # If you already know the exact folder, keep it; otherwise, choose newest one
    candidates = [p for p in glob.glob(os.path.join(model_dir, "*")) if os.path.isdir(p)]
    if not candidates:
        raise FileNotFoundError(f"No model folders under {model_dir}")
    model_path = max(candidates, key=os.path.getctime)
    print(f"✅ Using model: {model_path}")

    model = PipelineModel.load(model_path)

    # -------------------------------
    # 2) Load latest Gold features
    # -------------------------------
    gold_base = "/opt/airflow/datamart/gold/feature_store"
    latest_gold = get_latest_gold_feature_store(gold_base)
    print(f"✅ Using features: {latest_gold}")
    df = spark.read.parquet(latest_gold)

    # Optional: narrow to a specific snapshot if you pass one
    if snapshot_date and "snapshot_date" in df.columns:
        df = df.filter(F.col("snapshot_date") == snapshot_date)

    # Optional mapping of near-equivalent columns
    df = maybe_map_equivalents(df)

    # -------------------------------
    # 3) Schema guard — satisfy model inputs
    # -------------------------------
    required_cols = get_expected_input_cols(model)
    if not required_cols:
        raise RuntimeError("Could not find VectorAssembler inputs in the saved pipeline model.")
    print(f"ℹ️ Model expects {len(required_cols)} features. Ensuring presence and dtypes...")
    df = ensure_columns(df, required_cols)

    # -------------------------------
    # 4) Predict
    # -------------------------------
    pred = model.transform(df)

    # Keep a small output (Customer_ID + prediction)
    out_cols = [c for c in ["Customer_ID", "prediction", "probability", "rawPrediction"] if c in pred.columns]
    if not out_cols:
        out_cols = ["Customer_ID", "prediction"] if "Customer_ID" in pred.columns else ["prediction"]

    # Save predictions
    out_dir = "/opt/airflow/datamart/predictions"
    os.makedirs(out_dir, exist_ok=True)
    ts_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"predictions_{ts_tag}.parquet")
    pred.select(*out_cols).write.mode("overwrite").parquet(out_path)
    print(f"✅ Predictions saved to: {out_path}")

    spark.stop()
