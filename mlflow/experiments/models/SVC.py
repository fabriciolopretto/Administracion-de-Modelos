"""
Este script permite introducir la corrida "SVC" en el
experiemento "modelos", en MLFlow. Evaluando el modelo
de SVC sobre el dataset de características de cáncer de mama.
"""

# Importa las librerias y modulos necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.models.signature as model_signature

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score


# Crea el experimento en MLflow (si no existe)
experiment_name = "modelos"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException:
    # Si el experimento ya existe, recupera su ID
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Define el experimento en MLflow
mlflow.set_experiment(experiment_name)

# Define la corrida dentro del experimento
with mlflow.start_run(run_name="SVC") as run:

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

    # Define la rejilla de hiperparametros para SVC
    param_grid_svc = {
        'C': [0.1, 1, 10],  # Parámetro de regularización
        'kernel': ['linear', 'rbf'],  # Tipo de kernel
        'gamma': ['scale', 'auto']  # Parámetro de kernel para 'rbf'
    }

    # Crea el modelo SVC
    svc = SVC(probability=True, random_state=42)

    # Define la busqueda de hiperparametros por grilla con validacion cruzada
    grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='f1')

    # Entrena el modelo con la busqueda de hiperparametros
    grid_search_svc.fit(X_train_scaled, y_train)

    # Mejor combinacion de hiperparametros
    best_params = grid_search_svc.best_params_
    mlflow.log_params(best_params)

    # Evalua el mejor modelo en el conjunto de prueba
    best_model_svc = grid_search_svc.best_estimator_

    # Predice en el conjunto de prueba
    y_pred_svc = best_model_svc.predict(X_test_scaled)

    # Obtiene la matriz de confusion
    cm_svc = confusion_matrix(y_test, y_pred_svc)

    # Crea una visualizacion de la matriz de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_svc, display_labels=['Maligno (0)', 'Benigno (1)'])
    disp_fig = disp.plot(cmap=plt.cm.Blues, values_format='d', colorbar=False).figure_
    plt.title('Matriz de Confusión')
    plt.close(disp_fig)
    mlflow.log_figure(disp_fig, "confusion_matrix_SVC.png")

    # Calculo de metricas con la clase 0 (maligno) como positiva
    precision_svc = precision_score(y_test, y_pred_svc, pos_label=0)
    recall_svc = recall_score(y_test, y_pred_svc, pos_label=0)
    f1_svc = f1_score(y_test, y_pred_svc, pos_label=0)

    # Log metricas
    mlflow.log_metric("precision", precision_svc)
    mlflow.log_metric("recall", recall_svc)
    mlflow.log_metric("f1_score", f1_svc)

    # Obtiene las probabilidades de prediccion para la clase positiva
    y_prob_svc = best_model_svc.predict_proba(X_test_scaled)[:, 1]  # Probabilidades para la clase 1 (benigno)

    # Calcula la curva ROC
    fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, y_prob_svc, pos_label=1)  # Clase 1 es la positiva

    # Calcula el AUC (Área bajo la curva ROC)
    auc_svc = roc_auc_score(y_test, y_prob_svc)
    mlflow.log_metric("auc", auc_svc)

    # Grafica la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_svc, tpr_svc, label=f'SVC (AUC = {auc_svc:.2f})', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - SVC')
    plt.legend(loc="lower right")
    mlflow.log_figure(plt.gcf(), "roc_curve_SVC.png")
    plt.close()

    # ejemplo de los datos de entrada (usando el primer ejemplo de X_test_scaled)
    input_example = pd.DataFrame(X_test_scaled[:1, :], columns=X.columns)

    # Define la firma del modelo (basada en las entradas y salidas)
    signature = model_signature.infer_signature(X_test_scaled, y_pred_svc)

    # Registra el modelo con input_example y la firma
    mlflow.sklearn.log_model(
        sk_model=best_model_svc,
        artifact_path="artefactos_extra",
        input_example=input_example,
        signature=signature
    )
