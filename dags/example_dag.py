# dags/ingest_bronze_dag.py
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import os

# ---- Configuration ----
DAG_ID = "ingest_bronze_taxi"
DATASET_PATH = "/opt/airflow/data/dataset.parquet"
BRONZE_PATH = "/opt/airflow/data/bronze_taxi.parquet"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 3),
    "retries": 1,
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description="Ingestion dataset NYC Taxi et stockage Bronze",
    schedule_interval=None,
    catchup=False,
)

# ---- Fonction Python ----
def ingest_bronze():
    """Lire le dataset brut et le sauvegarder comme Bronze"""

    df = pd.read_parquet(DATASET_PATH)


    os.makedirs(os.path.dirname(BRONZE_PATH), exist_ok=True)


    df.to_parquet(BRONZE_PATH, index=False)
    print(f"Dataset Bronze sauvegardé à {BRONZE_PATH}")


task_ingest_bronze = PythonOperator(
    task_id="ingest_bronze",
    python_callable=ingest_bronze,
    dag=dag,
)


task_ingest_bronze
