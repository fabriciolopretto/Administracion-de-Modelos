"""
Este script permite registrar la corrida "RegLog" en el
experiemento "modelos", en MLFlow. Registrando el modelo
de regresión logística sobre el dataset de características
de cáncer de mama.
"""

# Importa las librerias y modulos necesarios
import mlflow
from mlflow import MlflowClient


# Crea el experimento en MLflow (si no existe)
experiment_name = "Registro_Modelos"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException:
    # Si el experimento ya existe, recupera su ID
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Define el experimento en MLflow
mlflow.set_experiment(experiment_name)

# Define la corrida dentro del experimento
with mlflow.start_run(run_name="RegLog", tags={"proyecto": "prueba"}) as run:

    client = MlflowClient()
    model_name = "Reg_Log"
    
    # Verifica si el modelo ya está registrado
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            print(f"El modelo '{model_name}' ya está registrado. Procediendo a crear una nueva versión.")
        else:
            print(f"Error al crear el modelo registrado: {e}")
            raise  # Vuelve a lanzar la excepción si no es un problema de existencia

    # create model version 
    run_id = "dc3711c11565409aa77077dcdc1d5f4b"
    source = f'runs:/{run_id}/artefactos_extra'

    # Crea la versión del modelo
    try:
        client.create_model_version(name=model_name, source=source, run_id=run_id)
    except mlflow.exceptions.MlflowException as e:
        print(f"Error al crear la versión del modelo: {e}")
    
    # Transición del estado de la versión del modelo
    try:
        # Aquí puedes modificar la versión según tus necesidades.
        # Por ejemplo, obtén la última versión del modelo y archívala
        versions = client.get_latest_versions(model_name, stages=["None"])
        latest_version = versions[0].version if versions else 1
        
        client.transition_model_version_stage(name=model_name, version=latest_version, stage="Archived")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error al transitar la versión del modelo a 'Archived': {e}")
