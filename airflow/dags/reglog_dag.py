from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Definir parámetros básicos del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Crear el DAG
dag = DAG(
    'reglog_mlflow',
    default_args=default_args,
    description='DAG para ejecutar la regresión logística en MLflow',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# Tarea para ejecutar el script RegLog.py usando BashOperator
run_mlflow_task = BashOperator(
    task_id='run_mlflow_script',
    bash_command='python /opt/airflow/scripts/RegLog.py',
    dag=dag,
)

# Dependencias de tareas
run_mlflow_task
