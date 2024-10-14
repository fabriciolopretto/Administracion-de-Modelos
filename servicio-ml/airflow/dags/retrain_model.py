import datetime

from airflow.decorators import dag, task

markdown_text = """
### Entrena el modelo para el dataset "Cancer de mama"

Entrena el modelo, compara contra el actual y si es mejor lo reemplaza en producción.

"""

default_args = {
    'owner': "Airflow",
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="train_model_cancer_mama",
    description="Entrena el modelo, compara contra el actual y si es mejor lo reemplaza en producción.",
    doc_md=markdown_text,
    tags=["Train", "Cáncer de mama", "Tumores"],
    default_args=default_args,
    schedule_interval='30 * * * *',  # Corre el primer día media hora desfasado con "etl_process"
    start_date=datetime.datetime(2024, 10, 21),
    catchup=False,
    is_paused_upon_creation=True
)
def processing_dag():

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def train_the_challenger_model():
        import datetime
        import mlflow
        import awswrangler as wr

        from sklearn.base import clone
        from sklearn.metrics import f1_score
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri('http://mlflow:5000')

        def load_the_champion_model():

            model_name = "cancer_mama_model_prod"
            alias = "champion"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            champion_version = mlflow.sklearn.load_model(model_data.source)

            return champion_version

        def load_the_train_test_data():
            X_train = wr.s3.read_csv("s3://data/final/train/tumors_X_train.csv")
            y_train = wr.s3.read_csv("s3://data/final/train/tumors_y_train.csv")
            X_test = wr.s3.read_csv("s3://data/final/test/tumors_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/tumors_y_test.csv")

            import pandas as pd
            import random

            # A modo de pruebas se cargan sólo 10000 ejemplo para entrenar
            # (para máquinas locales con bajos recursos computacionales)
            n_samples = 10000

            # Se genera una lista de índices aleatorios
            indices = random.sample(range(len(X_train)), n_samples)

            # Se generan muestras usando los índices aleatorios
            X_train = X_train.iloc[indices].reset_index(drop=True)
            y_train = y_train.iloc[indices].reset_index(drop=True)

            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):

            # Se realiza el track del experimento en MLflow
            experiment = mlflow.set_experiment("Cancer de mama")

            mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                             experiment_id=experiment.experiment_id,
                             tags={"experiment": "challenger models", "dataset": "Cancer de mama"},
                             log_system_metrics=True)

            params = model.get_params()
            params["model"] = type(model).__name__

            mlflow.log_params(params)

            artifact_path = "model"

            signature = infer_signature(X, model.predict(X))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format='cloudpickle',
                registered_model_name="cancer_mama_model_dev",
                metadata={"model_data_version": 1}
            )

            # Se obtiene la URI del modelo
            return mlflow.get_artifact_uri(artifact_path)

        def register_challenger(model, f1_score, model_uri):

            client = mlflow.MlflowClient()
            name = "cancer_mama_model_prod"

            # Se guardan los parámetros del modelo como tags y también la métrica f1-score
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["f1-score"] = f1_score

            # Se guarda la versión del modelo
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )

            # El alias se setea como "challenger"
            client.set_registered_model_alias(name, "challenger", result.version)

        champion_model = load_the_champion_model()

        challenger_model = clone(champion_model)

        X_train, y_train, X_test, y_test = load_the_train_test_data()

        challenger_model.fit(X_train, y_train.to_numpy().ravel())

        # Se obtiene métrica del modelo entrenado
        y_pred = challenger_model.predict(X_test)
        f1_score = f1_score(y_test.to_numpy().ravel(), y_pred)

        artifact_uri = mlflow_track_experiment(challenger_model, X_train)

        # Se registra el modelo "challenger"
        register_challenger(challenger_model, f1_score, artifact_uri)


    @task.virtualenv(
        task_id="evaluate_champion_challenge",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def evaluate_champion_challenge():
        import mlflow
        import awswrangler as wr

        from sklearn.metrics import f1_score

        mlflow.set_tracking_uri('http://mlflow:5000')

        def load_the_model(alias):
            model_name = "cancer_mama_model_prod"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            model = mlflow.sklearn.load_model(model_data.source)

            return model

        def load_the_test_data():
            X_test = wr.s3.read_csv("s3://data/final/test/tumors_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/tumors_y_test.csv")

            return X_test, y_test

        def promote_challenger(name):

            client = mlflow.MlflowClient()

            # Se baja el actual modelo "champion"
            client.delete_registered_model_alias(name, "champion")

            # Se carga el modelo "challenger"
            challenger_version = client.get_model_version_by_alias(name, "challenger")

            # Se le quita el alias "challenger"
            client.delete_registered_model_alias(name, "challenger")

            # Se le setea el alias "champion"
            client.set_registered_model_alias(name, "champion", challenger_version.version)

        def demote_challenger(name):

            client = mlflow.MlflowClient()
            client.delete_registered_model_alias(name, "challenger")

        champion_model = load_the_model("champion")

        challenger_model = load_the_model("challenger")

        # Se carga el dataset
        X_test, y_test = load_the_test_data()

        # Se obtienen las métricas para "challenger" y "champion"

        y_pred_champion = champion_model.predict(X_test)
        f1_score_champion = f1_score(y_test.to_numpy().ravel(), y_pred_champion)

        y_pred_challenger = challenger_model.predict(X_test)
        f1_score_challenger = f1_score(y_test.to_numpy().ravel(), y_pred_challenger)

        experiment = mlflow.set_experiment("Cancer de mama")

        # Se obtiene el último run_id del experimento para loguear la nueva información
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_f1_challenger", f1_score_challenger)
            mlflow.log_metric("test_f1_champion", f1_score_champion)

            if f1_score_challenger > f1_score_champion:
                mlflow.log_param("Winner", 'Challenger')
            else:
                mlflow.log_param("Winner", 'Champion')

        name = "cancer_mama_model_prod"
        if f1_score_challenger > f1_score_champion:
            promote_challenger(name)
        else:
            demote_challenger(name)

    @task.virtualenv(
        task_id="notify_api_about_new_model",
        requirements=["requests"],
        system_site_packages=True
    )
    def notify_api_about_new_model():
        import requests

        url = "http://fastapi:8800/update-model"
        response = requests.post(url)
        print(response.text)


    train_the_challenger_model() >> evaluate_champion_challenge() >> notify_api_about_new_model()


my_dag = processing_dag()
