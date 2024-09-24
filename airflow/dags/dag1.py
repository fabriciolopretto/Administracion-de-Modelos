import datetime
import mlflow
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve, f1_score



default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

#   DAG DECLARATION
dag = DAG(
    'TP_pipeline',
    default_args=default_args,
    description='Pipeline Airflow para TP Integrador',
    schedule_interval=None,
    start_date=days_ago(0),
    tags=['Dag_1']
)

#   FUNCTIONS
def obtain_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    path = './data.csv'
    df.to_csv(path, index=False)
    return path



def preprocess_and_split_data(**kwargs):
    ti = kwargs['ti']
    path_input = ti.xcom_pull(task_ids='obtain_data')
    
    if path_input:
        df = pd.read_csv(path_input)

        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train.to_csv("./X_train.csv", index=False)
        X_test.to_csv("./X_test.csv", index=False)
        y_train.to_csv("./y_train.csv", index=False)
        y_test.to_csv("./y_test.csv", index=False)



def normalize_data():
    """
    Estandarizamos los datos
    """
    X_train = pd.read_csv("./X_train.csv")
    X_test = pd.read_csv("./X_test.csv")

    sc_X = StandardScaler(with_mean=True, with_std=True)
    X_train_arr = sc_X.fit_transform(X_train)
    X_test_arr = sc_X.transform(X_test)

    X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

    X_train.to_csv("./X_train.csv", index=False)
    X_test.to_csv("./X_test.csv", index=False)

    

def train_model():
    import mlflow 
    X_train = pd.read_csv("./X_train.csv")
    X_test = pd.read_csv("./X_test.csv")
    y_train = pd.read_csv("./y_train.csv")
    y_test = pd.read_csv("./y_test.csv")

    model = mlflow.experiments.models.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)





#   OPERATORS
obtain_data_operator = PythonOperator(
    task_id = 'obtain_data',
    python_callable = obtain_data,
    dag = dag,
)

preprocess_and_split_data_operator = PythonOperator(
    task_id = 'Analisys_Dataframe_and_split',
    python_callable = preprocess_and_split_data,
    dag = dag,
)

""" check_dataset_operator = PythonOperator(
    task_id = 'Check_dataset',
    python_callable = check_dataset,
    dag = dag,
) """

normalize_data_operator = PythonOperator(
    task_id = 'Normalice_data',
    python_callable = normalize_data,
    dag = dag,
)

train_model_operator = PythonOperator(
    task_id = 'Train_Model',
    python_callable = train_model,
    dag = dag,
)


obtain_data_operator >>  preprocess_and_split_data_operator >> normalize_data_operator
normalize_data_operator >> train_model_operator




