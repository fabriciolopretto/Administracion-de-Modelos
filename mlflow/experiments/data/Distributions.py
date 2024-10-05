"""
Este script permite generar figuras como artefactos
en el experimento "features_engineering", en MLFlow.
Generando la distribución de características y un gráfico de torta
del dataset de cáncer de mama.
"""

# Importa las librerías y módulos necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.models.signature as model_signature
from sklearn.datasets import load_breast_cancer
from matplotlib import patches

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
with mlflow.start_run(run_name="Distributions") as run:

    # Carga el dataset desde sklearn.datasets 
    data = load_breast_cancer()

    # Convierte en dataframe
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Separa características y objetivo
    X = df.drop(columns=['target'])  # Todas las características
    y = df['target']  # Objetivo (benigno o maligno)

    # Diccionario traducción de atributos
    features_esn = {
        'mean radius' : 'Radio medio',              
        'mean texture' : 'Textura media',              
        'mean perimeter' : 'Perímetro medio',
        'mean area' : 'Área media',
        'mean smoothness' : 'Suavidad media',            
        'mean compactness' : 'Compactación media',
        'mean concavity' :  'Concavidad media',           
        'mean concave points' : 'Puntos cóncavos medios',
        'mean symmetry' : 'Simetría media',            
        'mean fractal dimension' : 'Dimensión fractal media',    
        'radius error' : 'Error estándar del radio',              
        'texture error' : 'Error estándar de la textura',             
        'perimeter error' : 'Error estándar del perímetro',            
        'area error' : 'Error estándar del área',                 
        'smoothness error' : 'Error estándar de la suavidad',           
        'compactness error' : 'Error estándar de la compactación',          
        'concavity error' : 'Error estándar de la concavidad',           
        'concave points error' : 'Error estándar de los puntos cóncavos',      
        'symmetry error' : 'Error estándar de la simetría',            
        'fractal dimension error' : 'Error estándar de la dimensión fractal',  
        'worst radius' : 'Peor radio',              
        'worst texture' : 'Peor textura',             
        'worst perimeter' : 'Peor perímetro',
        'worst area' : 'Peor área',
        'worst smoothness' : 'Peor suavidad',          
        'worst compactness' : 'Peor compactación',         
        'worst concavity' : 'Peor concavidad',      
        'worst concave points' : 'Peor número de puntos cóncavos',      
        'worst symmetry' : 'Peor simetría',            
        'worst fractal dimension' : 'Peor dimensión fractal'
    }

    # Paleta de colores para los gráficos (rojo para maligno, verde para benigno)
    palette = {
        0: '#ff3333',  # Rojo para maligno
        1: '#33cc33'   # Verde para benigno
    }

    # Columnas numéricas
    num_columns = df.select_dtypes(include=['float64']).columns

    # Número de columnas
    n_columns = 3
    # Calcular el número de filas necesario
    n_rows = (len(num_columns) + n_columns - 1) // n_columns  

    # Crear la figura y los ejes para la cuadrícula de histogramas
    disp_fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 5 * n_rows), squeeze=False)

    disp_fig.suptitle('Distribución de Características por Categoría', fontsize=16, fontweight='bold')

    # Graficar histogramas según target
    for idx, feature in enumerate(num_columns):
        row = idx // n_columns
        col = idx % n_columns
        ax = axes[row, col]
        
        sns.histplot(data=df, x=feature, hue='target', bins=50, alpha=0.7, palette=palette, ax=ax)
        ax.set_xlabel(features_esn[feature])
        ax.set_ylabel('Frecuencia')
        
        handles = [patches.Patch(color=palette[0], label='Maligno'), patches.Patch(color=palette[1], label='Benigno')]
        legend = ax.legend(handles=handles, title='Diagnóstico')
        
        ax.set_facecolor('#ffffff')  # Fondo blanco para el área de trazado
        for spine in ax.spines.values():
            spine.set_edgecolor('black')  # Borde negro en el área de trazado
            spine.set_linewidth(1)  # Grosor del borde del área de trazado
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        
        legend.set_frame_on(True)  # Habilitar el marco de la leyenda
        legend.get_frame().set_edgecolor('black')  # Color del borde de la leyenda
        legend.get_frame().set_linewidth(1)  # Grosor del borde de la leyenda
        legend.get_frame().set_facecolor('#ffffff')  # Color de fondo de la leyenda

    # Eliminar ejes vacíos si hay menos gráficos que subplots
    for idx in range(len(num_columns), n_rows * n_columns):
        row = idx // n_columns
        col = idx % n_columns
        disp_fig.delaxes(axes[row, col])

    # Ajustar el diseño para evitar solapamientos
    plt.tight_layout(rect=[0, 0, 1, 0.97]) 

    # Loggear la figura de histogramas en MLflow
    mlflow.log_figure(disp_fig, "data_distributions.png")
    plt.close(disp_fig)

    # Crear gráfico de torta para la distribución del target
    target_counts = y.value_counts()

    # Colores para el gráfico de torta
    colors = ['#ff3333', '#33cc33']  # Rojo para maligno, verde para benigno

    # Crear la figura del gráfico de torta
    pie_fig, ax = plt.subplots()
    ax.pie(target_counts, labels=['Maligno', 'Benigno'], autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Distribución del Diagnóstico', fontsize=16, fontweight='bold')

    # Loggear la figura del gráfico de torta en MLflow
    mlflow.log_figure(pie_fig, "target_pie_chart.png")
    plt.close(pie_fig)
