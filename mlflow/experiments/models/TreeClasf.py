"""
Este script permite introducir la corrida "TreeC" en el
experiemento "modelos", en MLFlow. Evaluando el modelo
de árbol de clasificación sobre el dataset
de características de cáncer de mama.
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
from sklearn.tree import DecisionTreeClassifier
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
with mlflow.start_run(run_name="TreeC") as run:

    # Carga el dataset desde sklearn.datasets 
    data = load_breast_cancer()

    # Convierte en dataframe
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Separa caracteristicas y objetivo
    X = df.drop(columns=['target'])  # Todas las características
    y = df['target']  # Objetivo (benigno o maligno)

    # Divide el conjunto de datos en conjunto de entrenamiento y prueba (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Escala las caracteristicas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define la rejilla de hiperparametros para arbol de decision
    param_grid_tree = {
        'criterion': ['gini', 'entropy'],  # Criterio para la división
        'max_depth': [None, 10, 20, 30],   # Profundidad máxima del árbol
        'min_samples_split': [2, 5, 10],    # Número mínimo de muestras requeridas para dividir un nodo
        'min_samples_leaf': [1, 2, 5]       # Número mínimo de muestras requeridas para estar en una hoja
    }

    # Crea el modelo de arbol de clasificacion
    tree_clf = DecisionTreeClassifier(random_state=42)

    # Define la busqueda de hiperparametros por grilla con validacion cruzada
    grid_search_tree = GridSearchCV(tree_clf, param_grid_tree, cv=5, scoring='f1')

    # Entrena el modelo con la busqueda de hiperparametros
    grid_search_tree.fit(X_train_scaled, y_train)

    # Mejor combinacion de hiperparametros
    best_params = grid_search_tree.best_params_
    mlflow.log_params(best_params)

    # Evalua el mejor modelo en el conjunto de prueba
    best_tree_clf = grid_search_tree.best_estimator_

    # Predice en el conjunto de prueba
    y_pred_tree = best_tree_clf.predict(X_test_scaled)

    # Obtiene la matriz de confusion
    cm_tree = confusion_matrix(y_test, y_pred_tree)

    # Crea una visualizacion de la matriz de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_tree, display_labels=['Maligno (0)', 'Benigno (1)'])
    disp_fig = disp.plot(cmap=plt.cm.Blues, values_format='d', colorbar=False).figure_
    plt.title('Matriz de Confusión')
    plt.close(disp_fig)
    mlflow.log_figure(disp_fig, "confusion_matrix_TreeC.png")

    # Calculo de metricas con la clase 0 (maligno) como positiva
    precision_tree = precision_score(y_test, y_pred_tree, pos_label=0)
    recall_tree = recall_score(y_test, y_pred_tree, pos_label=0)
    f1_tree = f1_score(y_test, y_pred_tree, pos_label=0)

    # Log metricas
    mlflow.log_metric("precision", precision_tree)
    mlflow.log_metric("recall", recall_tree)
    mlflow.log_metric("f1_score", f1_tree)    
    
    # Obtiene las probabilidades de prediccion para la clase positiva
    y_prob_tree_clf = best_tree_clf.predict_proba(X_test_scaled)[:, 1]  # Probabilidades para la clase 1 (benigno)

    # Calcula la curva ROC
    fpr_tree_clf, tpr_tree_clf, thresholds_tree_clf = roc_curve(y_test, y_prob_tree_clf, pos_label=1)  # Clase 1 es la positiva)

    # Calcula el AUC (Área bajo la curva ROC)
    auc_tree_clf = roc_auc_score(y_test, y_prob_tree_clf)
    mlflow.log_metric("auc", auc_tree_clf)

    # Grafica la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_tree_clf, tpr_tree_clf, label=f'Árbol de Clasificación (AUC = {auc_tree_clf:.2f})', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - Árbol de Clasificación')
    plt.legend(loc="lower right")
    mlflow.log_figure(plt.gcf(), "roc_curve_Tree.png")
    plt.close()

    # ejemplo de los datos de entrada (usando el primer ejemplo de X_test_scaled)
    input_example = pd.DataFrame(X_test_scaled[:1, :], columns=X.columns)

    # Define la firma del modelo (basada en las entradas y salidas)
    signature = model_signature.infer_signature(X_test_scaled, y_pred_tree)

    # Registra el modelo con input_example y la firma
    mlflow.sklearn.log_model(
        sk_model=best_tree_clf,
        artifact_path="artefactos_extra",
        input_example=input_example,
        signature=signature
    )
