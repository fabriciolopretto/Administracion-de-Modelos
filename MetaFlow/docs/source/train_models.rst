train_models module
=============

Descripción:
------------
Este script implementa un flujo de trabajo de machine learning utilizando la librería Metaflow,
que permite el entrenamiento y la evaluación de modelos de clasificación sobre el conjunto
de datos de cáncer de mama.

**Elementos:**

1 - class Metric(Base):
    Clase para representar la tabla de métricas en la base de datos.

    id (int): Identificador único de la métrica.
    model (str): Nombre del modelo.
    precision (float): Precisión del modelo.
    recall (float): Recall del modelo.
    f1_score (float): F1 Score del modelo.
   
2 - def check_database_connection():
    Verifica la conexión a la base de datos.
    Lanza una excepción si no se puede conectar.

3 - class ConfusionMatrixFlow(FlowSpec):
    Flujo para la creación y evaluación de modelos
    de clasificación con matrices de confusión.
    Incluye la carga de datos, entrenamiento de modelos,
    evaluación y almacenamiento de métricas.

4 - def start(self):
    Paso inicial que inicia el flujo.

5 - def load_and_prepare_data(self):
    Carga y prepara los datos del conjunto de datos de cáncer de mama.
    Se divide el conjunto de datos en entrenamiento y prueba, y se
    escalan las características.
    También se suben los archivos CSV a S3 y se guarda el escalador.

6 - def train_tree_model(self):
    Entrena un modelo de árbol de decisión y almacena las métricas.
    Se realiza una búsqueda de hiperparámetros con validación
    cruzada y se almacenan las métricas del modelo entrenado.

7 - def train_svc_model(self):
    Entrena un modelo de máquina de soporte vectorial (SVC)y almacena las métricas.
    Se realiza una búsqueda de hiperparámetros con validación cruzada y se almacenan
    las métricas del modelo entrenado.

8 - def train_knn_model(self):
    Entrena un modelo de k-vecinos más cercanos (KNN) y almacena las métricas.
    Se realiza una búsqueda de hiperparámetros con validación cruzada y se almacenan
    las métricas del modelo entrenado.

9 - def train_reglog_model(self):
    Entrena un modelo de regresión logística y almacena las métricas.
    Se realiza una búsqueda de hiperparámetros con validación cruzada y se almacenan
    las métricas del modelo entrenado.

10 - def join_models(self, inputs):
     Combina los resultados de los modelos entrenados y almacena las métricas en la base de datos.

11 - def evaluate(self):
     Evalúa el rendimiento de los modelos entrenados mediante métricas de clasificación, 
     incluyendo la matriz de confusión y la curva ROC.
     Para cada modelo, se predicen las clases en el conjunto de prueba y se calculan las probabilidades
     para la clase positiva. Luego, se generan y almacenan las visualizaciones de la matriz de confusión 
     y la curva ROC en un bucket de S3.
     Las métricas calculadas (precisión, recall, F1 y AUC) se almacenan en un diccionario para su 
     posterior uso.
     Almacena las matrices de confusión como imágenes en S3, curvas ROC como imágenes en S3 y
     métricas de evaluación.

12 -  def store_results(self):
      Almacena las métricas de evaluación de los modelos en una base de datos PostgreSQL.

      Utiliza la sesión de SQLAlchemy para agregar las métricas (precisión, recall y F1) 
      de cada modelo al modelo de datos. Confirma la transacción y cierra la sesión.

13 -  def end(self):
      Finaliza el flujo de evaluación de la matriz de confusión.

      Imprime los resultados de las métricas almacenadas en la base de datos, 
      verificando que los datos se han guardado correctamente.

      Se consultan los resultados de la base de datos y se imprime la información 
      de cada métrica almacenada.

14 - def upload_to_s3(self, file_name, bucket, object_name=None):
      Sube un archivo a un bucket de Amazon S3.

      file_name (str): Ruta del archivo a subir.
      bucket (str): Nombre del bucket de S3.
      object_name (str, opcional): Nombre del objeto en el bucket.

      Captura y muestra un mensaje de error si la subida del archivo falla.
