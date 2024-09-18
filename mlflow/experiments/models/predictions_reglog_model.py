"""
Este script permite utilizar el modelo de regresión logística
del experiemento "modelos" en MLFlow, y generar predicciones.
"""

# Importa las librerias y modulos necesarios
import pandas as pd
import mlflow
import mlflow.sklearn
import configparser
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

# Crea el experimento en MLflow (si no existe)
experiment_name = "inferencia_modelo_reglog"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException:
    # Si el experimento ya existe, recupera su ID
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Define el experimento en MLflow
mlflow.set_experiment(experiment_name)

# Inicia una nueva corrida en MLflow
with mlflow.start_run(run_name="Inferencia_RegLog", tags={"proyecto": "prueba_inferencia"}) as run:

    # Ruta al script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Lee el archivo .ini
    config = configparser.ConfigParser()
    config.read(script_dir + r'/config.ini')

    # Obtiene las credenciales desde la seccion [postgresql]
    db_user = config['postgresql']['user']
    db_password = config['postgresql']['password']
    db_host = config['postgresql']['host']
    db_port = config['postgresql']['port']
    db_name = config['postgresql']['dbname']

    # Cambia 'localhost' a 'host.docker.internal'
    DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    # Crea una conexion con PostgreSQL
    engine = create_engine(DATABASE_URL)

    # Carga el dataset desde sklearn.datasets 
    data = load_breast_cancer()

    # Convierte en dataframe
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Separa caracteristicas y objetivo
    X = df.drop(columns=['target'])  # Todas las caracteristicas
    y = df['target']  # Objetivo (benigno o maligno)

    # Divide el conjunto de datos en conjunto de entrenamiento y prueba (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Escala las caracteristicas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Identifica el modelo a utilizar para las predicciones
    run_id = "dc3711c11565409aa77077dcdc1d5f4b"
        
    # Carga el modelo
    model_uri = model_uri = f'runs:/{run_id}/artefactos_extra'
    reg_log_model = mlflow.sklearn.load_model(model_uri=model_uri)

    # Realiza predicciones
    y_pred = reg_log_model.predict(X_test_scaled)

    # Registra las predicciones
    mlflow.log_metric("cantidad_predicciones", len(y_pred))

    # Convierte los valores de numpy a enteros estándar de Python
    y_pred = [int(val) for val in y_pred]
    y_test = [int(val) for val in y_test]

    # Crea una tabla SQL si no existe
    with engine.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS salida_modelo (
                id INT PRIMARY KEY,
                real INT,
                prediction INT
            );
        """))

        # Inserta los datos en la tabla
        for i, (real, pred) in enumerate(zip(y_test, y_pred)):
            connection.execute(text("""
                INSERT INTO salida_modelo (id, real, prediction) VALUES (:id, :real, :prediction)
                ON CONFLICT (id) DO UPDATE SET real = EXCLUDED.real, prediction = EXCLUDED.prediction;
            """), {'id': i, 'real': real, 'prediction': pred})

        # Exporta la tabla SQL a un archivo
        sql_file_path = "/app/salida_modelo.sql"
        with open(sql_file_path, 'w') as file:
            result = connection.execute(text("SELECT * FROM salida_modelo;"))
            for row in result:
                file.write(f"{row}\n")

    # Loguea el archivo SQL como un artefacto en MLflow
    mlflow.log_artifact("/app/salida_modelo.sql")
