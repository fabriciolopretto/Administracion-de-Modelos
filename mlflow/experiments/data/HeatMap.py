"""
Este script permite generar figura como artefactos
en el experimento "features_engineering", en MLFlow.
Generando el mapa de calor de correlaciones entre los atributos
del dataset de cáncer de mama.
"""

# Importa las librerías y módulos necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer

# Crea el experimento en MLflow (si no existe)
experiment_name = "features_engineering"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException:
    # Si el experimento ya existe, recupera su ID
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Define el experimento en MLflow
mlflow.set_experiment(experiment_name)

# Define la corrida dentro del experimento
with mlflow.start_run(run_name="Correlations_HeatMap") as run:

    # Carga el dataset desde sklearn.datasets 
    data = load_breast_cancer()

    # Convierte en dataframe
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    heatmap_fig, ax = plt.subplots()
    correlation = df.corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Mapa de Correlación de Atributos')
    # Loggear la figura del gráfico de torta en MLflow
    mlflow.log_figure(heatmap_fig, "corr_heat_map.png")
    plt.close(heatmap_fig)
