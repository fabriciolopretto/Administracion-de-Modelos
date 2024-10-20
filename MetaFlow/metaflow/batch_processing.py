"""
Este script define un flujo de procesamiento de datos y modelos de aprendizaje
automático mediante Metaflow, integrando almacenamiento en AWS S3 (MinIO),
predicciones con varios modelos y la ingestión de resultados en Redis.
"""

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
    """
    Clase que implementa un flujo para el procesamiento por lotes y 
    la predicción usando modelos cargados desde S3, con la posibilidad 
    de almacenar los resultados en Redis.
    """

    @step
    def start(self):
        """
        Inicio del flujo.
        
        Este paso imprime un mensaje indicando el inicio del flujo y avanza a 
        los siguientes pasos para la carga de datos y modelos.
        """
        print("Starting Combined Model Training and Batch Processing")
        self.next(self.load_data, self.load_models)

    @step
    def load_data(self):
        """
        Carga los datos desde S3 y aplica un escalador previamente entrenado.
        
        Este paso carga los datos de entrada desde un archivo CSV almacenado en S3 y 
        utiliza un scaler almacenado en S3 para escalar los datos antes de 
        procesarlos. Luego avanza al paso de procesamiento por lotes.
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
        Carga los modelos previamente entrenados desde S3.
        
        Este paso obtiene los modelos almacenados en S3, que incluyen un árbol de 
        decisión, SVM, KNN y regresión logística. Avanza al siguiente paso para 
        procesar los datos en lotes.
        """
        s3 = S3(s3root="s3://amqtp/")
        self.loaded_models = {}

        for model_name in ["tree", "svc", "knn", "reglog"]:
            try:
                # Obtener el archivo del modelo de S3
                model_obj = s3.get(f"{model_name}_model.pkl")
                with open(model_obj.path, 'rb') as f:
                    self.loaded_models[model_name] = pickle.load(f)
                    print(self.loaded_models[model_name])
                print(f"Loaded {model_name} model successfully from S3.")
            except Exception as e:
                print(f"Error while loading {model_name}: {e}")
                raise

        print("All models loaded from S3")
        self.next(self.batch_processing)

    @step
    def batch_processing(self, previous_tasks):
        """
        Realiza el procesamiento por lotes con los modelos cargados.
        
        Este paso toma los datos escalados y los utiliza para obtener predicciones 
        de los modelos cargados previamente. Las predicciones se almacenan en un 
        diccionario y se mapean las clases como "Maligno" o "Benigno".
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


        # Genera las cadenas y hashes 
        string_values = [' '.join(map(str, row)) for row in data]
        hashed_values = [hashlib.sha256(substring.encode()).hexdigest() for substring in string_values]


        print(hashed_values)

        # Preparamos los datos para ser enviados a Redis
        dict_redis = {}
        for index, hash_value in enumerate(hashed_values):
            # Guarda las predicciones de ambos modelos en el diccionario
            dict_redis[hash_value] = {
                'tree': label_map.get(predictions['tree'][index]),
                'svc': label_map.get(predictions['svc'][index]),
                'knn': label_map.get(predictions['knn'][index]),
                'reglog': label_map.get(predictions['reglog'][index])
            }
        self.redis_data = dict_redis

        self.next(self.ingest_redis)

    @step
    def ingest_redis(self):
        """
        Ingresa las predicciones en Redis.
        
        Este paso toma los resultados del procesamiento por lotes y los almacena 
        en Redis utilizando un pipeline para mayor eficiencia.
        """
        import redis

        print("Ingesting predictions into Redis")
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)

        # Comenzamos un pipeline de Redis
        pipeline = r.pipeline()

        # Se pre-ingresan los datos en Redis para todos los modelos
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
        Finaliza el flujo de procesamiento.
        
        Este paso imprime un mensaje indicando que el procesamiento ha terminado 
        y realiza una prueba de escritura en Redis para verificar la conectividad.
        """
        print("Finished processing")

        import redis

        print("Ingesting predictions into Redis")
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)

        # Prueba de escritura en Redis
        r.set("test_key", "test_value")

        # Recupera el valor para verificar
        print(r.get("test_key"))

if __name__ == "__main__":
    CombinedAndBatchProcessing()
