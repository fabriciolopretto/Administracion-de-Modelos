batch_processing module
===========

Descripción:
------------

Este script define un flujo de procesamiento de datos y modelos de aprendizaje
automático mediante Metaflow, integrando almacenamiento en AWS S3 (MinIO),
predicciones con varios modelos y la ingestión de resultados en Redis.

**Elementos:**

1 - CombinedAndBatchProcessing(FlowSpec):
    Clase que implementa un flujo para el procesamiento por lotes y 
    la predicción usando modelos cargados desde S3, con la posibilidad 
    de almacenar los resultados en Redis.

2 - start(self):
    Inicio del flujo.
   
    Este paso imprime un mensaje indicando el inicio del flujo y avanza a 
    los siguientes pasos para la carga de datos y modelos.

3 - load_data(self):
    Carga los datos desde S3 y aplica un escalador previamente entrenado.
        
    Este paso carga los datos de entrada desde un archivo CSV almacenado en S3 y 
    utiliza un scaler almacenado en S3 para escalar los datos antes de 
    procesarlos. Luego avanza al paso de procesamiento por lotes.

4 - load_models(self):
    Carga los modelos previamente entrenados desde S3.
        
    Este paso obtiene los modelos almacenados en S3, que incluyen un árbol de 
    decisión, SVM, KNN y regresión logística. Avanza al siguiente paso para 
    procesar los datos en lotes.

5 - batch_processing(self, previous_tasks):
    Realiza el procesamiento por lotes con los modelos cargados.
        
    Este paso toma los datos escalados y los utiliza para obtener predicciones 
    de los modelos cargados previamente. Las predicciones se almacenan en un 
    diccionario y se mapean las clases como "Maligno" o "Benigno".

6 - ingest_redis(self):
    Ingresa las predicciones en Redis.
        
    Este paso toma los resultados del procesamiento por lotes y los almacena 
    en Redis utilizando un pipeline para mayor eficiencia.

7 - def end(self):
    Finaliza el flujo de procesamiento.
        
    Este paso imprime un mensaje indicando que el procesamiento ha terminado 
    y realiza una prueba de escritura en Redis para verificar la conectividad.
