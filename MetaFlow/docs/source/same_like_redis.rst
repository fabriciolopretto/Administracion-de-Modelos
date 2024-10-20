same_like_redis module
============

Descripción:
------------

Este script realiza un flujo de procesamiento de datos
para predicciones de modelos y visualización de resultados.

**Elementos:**

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
