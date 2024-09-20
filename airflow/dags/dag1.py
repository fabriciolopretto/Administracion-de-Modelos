import datetime

import pandas as pd

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import mlflow
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



def check_dataset(**kwargs):
    ti = kwargs['ti']
    path_input = ti.xcom_pull(task_ids='obtain_data')
    
    def duplicados(df):
        """
        Verifica si el DataFrame ingresado tiene
        registros duplicados.

        :param:
        :df: DataFrame con los atributos.
        """
        hay_duplicados = df.duplicated().any()

        if hay_duplicados:
            print("El DataFrame tiene registros duplicados.")
        else:
            print("El DataFrame no tiene registros duplicados.")

    if path_input: 
        dataset = pd.read_csv(path_input)




def preprocess_and_split_data(**kwargs):
    ti = kwargs['ti']
    path_input = ti.xcom_pull(task_ids='obtain_data')
    
    if path_input:
        df = pd.read_csv(path_input)


        """ check_dataset(**kwargs) """



        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train.to_csv("./X_train.csv", index=False)
        X_test.to_csv("./X_test.csv", index=False)
        y_train.to_csv("./y_train.csv", index=False)
        y_test.to_csv("./y_test.csv", index=False)


##  DEJO ESTE METODO POR EL IF, NO SE SI ES NECESARIO O NO
def split_dataset(**kwargs):
    """
    Genera el dataset y obtiene set de testeo y evaluación
    """
    ti = kwargs['ti']
    # Leemos el mensaje del DAG anterior
    dummies_output = ti.xcom_pull(task_ids='make_dummies_variables')

    dataset = pd.read_csv("./data_clean_dummies.csv")

    if dataset.shape[0] == dummies_output["observations"] and dataset.shape[1] == dummies_output["columns"]:

        X = dataset.drop(columns="num")
        y = dataset[["num"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

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
    X_train = pd.read_csv("./X_train.csv")
    X_test = pd.read_csv("./X_test.csv")
    y_train = pd.read_csv("./y_train.csv")
    y_test = pd.read_csv("./y_test.csv")


    model = SVC(kernel='linear', probability=True)
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
    task_id = 'Analisys Dataframe and split',
    python_callable = preprocess_and_split_data,
    dag = dag,
)

normalize_data_operator = PythonOperator(
    task_id = 'Normalice data',
    python_callable = normalize_data,
    dag = dag,
)

train_model_operator = PythonOperator(
    task_id = 'Train Model',
    python_callable = train_model,
    dag = dag,
)


obtain_data_operator >> preprocess_and_split_data_operator >> normalize_data_operator
normalize_data_operator >> [train_model_operator]




