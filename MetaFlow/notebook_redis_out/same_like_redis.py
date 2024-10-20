"""
Este script realiza un flujo de procesamiento de datos
para predicciones de modelos y visualización de resultados.

Entre las tareas se tiene:

1 - Conexión a Redis: Establece una conexión a un servidor Redis
que está ejecutándose localmente en el puerto 6379.

2 - Configuración de AWS S3 (Minio): Configura las credenciales y la URL
para conectarse a un servidor S3 local usando Minio.

3 - Carga de datos: Lee un archivo CSV de datos de cáncer de mama
sin encabezados y toma una muestra aleatoria de 50 registros.

4 - Carga de un scaler preentrenado: Descarga un objeto scaler.pkl
desde un bucket S3 y lo carga usando pickle.

5 - Escalado de los datos: Aplica el scaler cargado a los datos de prueba.

6 - Generación de hashes: Convierte los valores escalados en cadenas
y luego genera un hash SHA-256 para cada una de ellas.
Esto facilita la identificación de predicciones almacenadas en Redis.

7 - Recuperación de predicciones de Redis: Para cada hash, busca las
predicciones correspondientes de diferentes modelos (como tree y svc)
almacenadas en Redis y las guarda en un diccionario model_outputs.

8 - Impresión de predicciones: Muestra las predicciones de los modelos
tree y svc para las primeras 5 entradas.

9 - Creación de DataFrame: Convierte los valores de prueba en un DataFrame
y añade las predicciones de cada modelo como nuevas columnas.

10 - Conversión de predicciones a numéricas: Convierte las predicciones
de los modelos a valores numéricos para facilitar el análisis.

11 - Visualización de predicciones: Crea gráficos de dispersión
(scatter plots) para visualizar las predicciones de cada modelo.
Usa las primeras dos columnas de los datos como los ejes x y y,
y colorea los puntos según las predicciones de cada modelo.
Genera una cuadrícula de gráficos para mostrar las predicciones
de todos los modelos.
Crea un gráfico adicional con las etiquetas basadas en el primer
modelo para representar los datos de prueba.
"""

import os
import redis
import hashlib
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from metaflow import FlowSpec, step, S3

# Conectamos al servidor redis (asegúrate de que el docker compose esté corriendo)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"

# Cargamos los datos
df = pd.read_csv("./breast_cancer.csv", header=None)

# Sampleamos 50 valores al azar
df_temp = df.sample(50)
test_values = df_temp.values.tolist()

# Cargamos el scaler previamente guardado
s3 = S3(s3root="s3://amqtp/")
scaler_obj = s3.get("scaler.pkl")
with open(scaler_obj.path, 'rb') as f:
    scaler = pickle.load(f)

# Aplicamos el scaler a los datos
scaled_values = scaler.transform(df_temp)

# Convertimos los valores escalados a cadenas y generamos el hash
string_values = [' '.join(map(str, sublist)) for sublist in scaled_values]
hashed_values = [hashlib.sha256(substring.encode()).hexdigest() for substring in string_values]

# Inicializamos un diccionario para almacenar las salidas del modelo
model_outputs = {}

# Obtenemos las predicciones almacenadas en Redis
for hash_key in hashed_values:
    model_outputs[hash_key] = r.hgetall(f"predictions:{hash_key}")

# Mostramos las salidas de los modelos para las primeras 5 entradas
print("Salidas de los modelos para las primeras 5 entradas:")
for index, test_value in enumerate(test_values[:5]):
    hash_key = hashed_values[index]
    tree_prediction = model_outputs[hash_key].get('tree', 'No disponible')
    svc_prediction = model_outputs[hash_key].get('svc', 'No disponible')
    
    print(f"\nPara la entrada: {test_value}")
    print(f"El modelo tree predice: {tree_prediction}")
    print(f"El modelo svc predice: {svc_prediction}")

print("\nSe han mostrado las predicciones para las primeras 5 entradas.")

# Creamos un DataFrame con los valores de test
test_values_df = pd.DataFrame(test_values)

# Obtenemos la lista de modelos disponibles
models = list(set().union(*[set(output.keys()) for output in model_outputs.values()]))

# Agregamos las predicciones de cada modelo al DataFrame
for model in models:
    test_values_df[model] = [model_outputs[hash_key].get(model, 'No disponible') for hash_key in hashed_values]

# Renombramos las columnas
test_values_df.columns = [str(i) for i in range(len(test_values_df.columns) - len(models))] + models

# Convertimos las predicciones a valores numéricos
for model in models:
    test_values_df[model] = pd.to_numeric(test_values_df[model], errors='coerce')

# Creamos los scatter plots
n_cols = 3  # Número de columnas en la cuadrícula de gráficos
n_rows = (len(models) + n_cols - 1) // n_cols  # Número de filas necesarias

plt.figure(figsize=(6*n_cols, 5*n_rows))

for i, model in enumerate(models, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.scatterplot(x="0", y="1", hue=model, data=test_values_df, palette="viridis")
    plt.title(f"Predicciones del modelo {model}")

plt.tight_layout()
plt.show()

# Representemos graficamente a estos datos
test_values_df["label"] = test_values_df[models[0]]  # Usamos el primer modelo como ejemplo para las etiquetas
sns.scatterplot(x="0", y="1", hue="label", data=test_values_df)
plt.title("Predicciones del modelo en los datos de prueba")
plt.show()
