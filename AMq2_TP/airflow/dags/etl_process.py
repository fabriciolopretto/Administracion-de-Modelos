from airflow.decorators import dag, task
import datetime

markdown_text = """
A completar

"""

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=60),
    'is_active': True
}

@dag(
    dag_id="etl_process_cancer_mama2",
    description="ETL de características de tumores de cáncer de mama.",
    doc_md=markdown_text,
    tags=["ETL", "Cáncer de mama", "Tumores"],
    default_args=default_args,
    catchup=False,
    schedule_interval='0 * * * *',  # Corre cada hora
    start_date=datetime.datetime(2024, 10, 21),
    is_paused_upon_creation=False
)
def process_etl_cancermama():

    @task.virtualenv(
        task_id="extract_data",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def extract_data():
        import awswrangler as wr
        import pandas as pd

        from sklearn.datasets import load_breast_cancer

        tumors_df = load_breast_cancer()
        
        # Se guardar el dataframe como CSV en S3 para poder pasarlo entre tareas
        data_path = "s3://data/raw/cancer_mama.csv"
        print('previo guarda en def extract')
        wr.s3.to_csv(df=tumors_df, path=data_path, index=False)

    @task.virtualenv(
        task_id="transform_data",
        requirements=["scikit-learn==1.2.2", "awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def transform_data():
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from sklearn.impute import SimpleImputer
        import boto3
        import botocore.exceptions
        import mlflow
        import json
        import datetime

        data_path = "s3://data/raw/cancer_mama.csv"
        tumors_original_df = wr.s3.read_csv(data_path)

        tumors_df = tumors_original_df.copy()

        VARIABLE_SALIDA = 'target'

        data_end_path = "s3://data/raw/cancer_mama_corregido.csv"
        print('previo guarda en def transf')
        wr.s3.to_csv(df=tumors_df, path=data_end_path, index=False)

        # Se guarda la metadata en S3
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            data_dict = json.loads(result["Body"].read().decode())
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                data_dict = {}
            else:
                raise e

        data_dict['columns'] = tumors_original_df.columns.to_list()
        data_dict['target_col'] = VARIABLE_SALIDA
        data_dict['columns_dtypes'] = {k: str(v) for k, v in tumors_original_df.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_transform'] = {k: str(v) for k, v in tumors_df.dtypes.to_dict().items()}

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)
        client.put_object(Bucket='data', Key='data_info/data.json', Body=data_string)

        # Se registra el experimento en MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Cancer de mama")
        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                        experiment_id=experiment.experiment_id,
                        tags={"experiment": "etl", "dataset": "tumors cancer mama"})
        mlflow_dataset = mlflow.data.from_pandas(tumors_original_df.sample(10),
                                                source="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
                                                targets=VARIABLE_SALIDA,
                                                name="tumors_data_complete")
        mlflow_dataset_transformed = mlflow.data.from_pandas(tumors_df.sample(10),
                                                        source="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
                                                        targets=VARIABLE_SALIDA,
                                                        name="tumors_data_transformed")
        mlflow.log_input(mlflow_dataset, context="Dataset")                        
        mlflow.log_input(mlflow_dataset_transformed, context="Dataset")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0", "scikit-learn"],
        system_site_packages=True
    )
    def split_dataset():
        import awswrangler as wr
        import pandas as pd
        from sklearn.model_selection import train_test_split

        data_transformed_path = "s3://data/raw/cancer_mama_corregido.csv"
        tumors_df = wr.s3.read_csv(data_transformed_path)

        # Se separa el target del resto de las features
        X = tumors_df.drop(columns='target')
        y = tumors_df['target']

        # Se separa el conjunto de entrenamiento y testeo en 75% y 25% respectivamente.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

        def save_to_csv(df, path):
            print('previo guarda en def split')
            wr.s3.to_csv(df=df, path=path, index=False)

        save_to_csv(X_train, "s3://data/final/train/tumors_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/tumors_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/tumors_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/tumors_y_test.csv")

    @task.virtualenv(
        task_id="normalize_data",
        requirements=["awswrangler==3.6.0", "scikit-learn"],
        system_site_packages=True
    )
    def normalize_data():

        import json
        import mlflow
        import boto3
        import botocore.exceptions
        import awswrangler as wr
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        def save_to_csv(df, path):
            print('previo guarda en def normaliz')
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        X_train = wr.s3.read_csv("s3://data/final/train/tumors_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/tumors_X_test.csv")

        sc_X = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = sc_X.fit_transform(X_train)
        X_test_arr = sc_X.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        save_to_csv(X_train, "s3://data/final/train/tumors_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/tumors_X_test.csv")

        # Se guarda información del dataset
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                raise e

        data_dict['standard_scaler_mean'] = sc_X.mean_.tolist()
        data_dict['standard_scaler_std'] = sc_X.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Cancer de mama")

        # Se obtiene el último run_id del experimento para loguear la información
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
            mlflow.log_param("Standard Scaler scale values", sc_X.scale_)
        
    
    # Se definen las dependencias
    extract_data() >> transform_data() >> split_dataset() >> normalize_data()
    
dag = process_etl_cancermama()
