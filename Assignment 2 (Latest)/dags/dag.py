# ============================================================
# AIRFLOW DAG — Monthly MLE Pipeline
# ============================================================

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator as DummyOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "dag",
    default_args=default_args,
    description="Monthly Medallion pipeline (Bronze→Silver→Gold→Model→Monitor)",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    catchup=True,
    max_active_runs=1,
    concurrency=1,
    tags=["MLE", "pipeline"],
) as dag:

    start = DummyOperator(task_id="start")

    bronze = BashOperator(
        task_id="bronze",
        bash_command=(
            'cd /opt/airflow/utils && '
            'python3 data_processing_bronze_table.py --snapshotdate "{{ ds }}"'
        ),
    )

    silver = BashOperator(
        task_id="silver",
        bash_command=(
            'cd /opt/airflow/utils && '
            'python3 data_processing_silver_table.py --snapshotdate "{{ ds }}"'
        ),
    )

    gold = BashOperator(
        task_id="gold",
        bash_command=(
            'cd /opt/airflow/utils && '
            'python3 data_processing_gold_table.py '
            '--snapshotdate "{{ ds }}" '
            '--silver_lms_dir "/opt/airflow/datamart/silver/lms/" '
            '--silver_financials_dir "/opt/airflow/datamart/silver/financials/" '
            '--silver_attributes_dir "/opt/airflow/datamart/silver/attributes/" '
            '--silver_clickstream_dir "/opt/airflow/datamart/silver/clickstream/" '
            '--gold_features_dir "/opt/airflow/datamart/gold/feature_store/" '
            '--gold_labels_dir "/opt/airflow/datamart/gold/label_store/" '
            '--dpd_threshold 30'
        ),
    )

    model = DummyOperator(task_id="model")
    monitor = DummyOperator(task_id="monitor")
    end = DummyOperator(task_id="end")

    start >> bronze >> silver >> gold >> model >> monitor >> end
