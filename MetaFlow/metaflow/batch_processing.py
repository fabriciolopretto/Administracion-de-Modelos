import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from metaflow import FlowSpec, step, S3
import hashlib
import redis
import pickle
import joblib
import boto3

# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"

class CombinedAndBatchProcessing(FlowSpec):

    @step
    def start(self):
        print("Starting Combined Model Training and Batch Processing")
        self.next(self.load_data, self.load_models)

    @step
    def load_data(self):
        """
        Paso para cargar los datos de entrada de S3
        """
        import pandas as pd

        # Se utiliza el objeto S3 para acceder a los datos desde el bucket en S3.
        s3 = S3(s3root="s3://amqtp/")
        data_obj = s3.get("data/breast_cancer.csv")
        self.X_batch = pd.read_csv(data_obj.path)

        # Cargar el scaler utilizado durante el entrenamiento
        scaler_obj = s3.get("scaler.pkl")
        with open(scaler_obj.path, 'rb') as f:
            self.scaler = pickle.load(f)
         # Escalar los datos utilizando el scaler cargado
        data_scaled = self.scaler.transform(self.X_batch)
        self.X_batch = data_scaled
        self.next(self.batch_processing)


    @step
    def load_models(self):
        """
        Paso para cargar los modelos previamente entrenados desde S3.
        """
        s3 = S3(s3root="s3://amqtp/")
        self.loaded_models = {}

        for model_name in ["tree", "svc", "knn", "reglog"]:
            try:
                # Obtener el archivo del modelo de S3
                model_obj = s3.get(f"{model_name}_model.pkl")
                with open(model_obj.path, 'rb') as f:
                    self.loaded_models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model successfully from S3.")
            except Exception as e:
                print(f"Error while loading {model_name}: {e}")
                raise

        print("All models loaded from S3")
        self.next(self.batch_processing)

    @step
    def batch_processing(self, previous_tasks):
        """
        Paso para realizar el procesamiento por lotes con los modelos cargados.
        """
        import numpy as np

        print("Obtaining predictions from both models")

        # Inicializa las variables para datos y modelos
        data = None
        models = {}

        # Recorre las tareas previas para obtener los datos y los modelos
        for task in previous_tasks:
            if hasattr(task, 'X_batch'):
                data = task.X_batch
            if hasattr(task, 'loaded_models'):
                models = task.loaded_models  # Accede a todos los modelos cargados

        # Asegúrate de que se hayan encontrado ambos
        if data is None or not models:
            raise ValueError("Data or models not found in previous tasks.")
        
        # Realiza las predicciones para ambos modelos
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(data)
        label_map = {0: "Maligno", 1: "Benigno"}  # Ajusta según tus clases


        data_df = pd.DataFrame(data)

        # Genera un hash para cada fila de datos
        data_df['key'] = data_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        data_df['hashed'] = data_df['key'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

        # Preparamos los datos para ser enviados a Redis
        dict_redis = {}
        for index, row in data_df.iterrows():
            # Guarda las predicciones de ambos modelos en el diccionario
            dict_redis[row["hashed"]] = {
                'tree': label_map.get(predictions['tree'][index]),
                'svc': label_map.get(predictions['svc'][index])
            }
        self.redis_data = dict_redis

        self.next(self.ingest_redis)

    @step
    def ingest_redis(self):
        """
        Paso para ingestar los resultados en Redis.
        """
        import redis

        print("Ingesting predictions into Redis")
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)

        # Comenzamos un pipeline de Redis
        pipeline = r.pipeline()

        # Se pre-ingresan los datos en Redis para ambos modelos ('tree' y 'svc')
        for key, value in self.redis_data.items():
            # Guardamos los resultados de ambos modelos con sus respectivas claves
            pipeline.hset(f"predictions:{key}", mapping=value)

        # Ejecutamos el pipeline para insertar todos los datos de manera eficiente
        pipeline.execute()
        print("Predictions successfully ingested into Redis")
        self.next(self.end)

    @step
    def end(self):
        """
        Paso final del flujo. Imprime un mensaje de finalización.
        """
        print("Finished processing")

if __name__ == "__main__":
    CombinedAndBatchProcessing()
