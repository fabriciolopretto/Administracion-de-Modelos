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