# Administración de Modelos

**Descripción**  
Este proyecto permite realizar experimentos de ingeniería de datos y la posterior administración de modelos de aprendizaje de máquina. Generando artefactos y registrando modelos.

## Características

- **Interfaz Amigable:** Permite una navegación y operación intuitiva con MLflow.
- **Modularidad:** Personalizable para implementar los modelos que se desee.
- **Compatibilidad:** Funciona en Windows.

## Tecnologías

Este proyecto utiliza:

- [Python 3.11+](https://www.python.org/downloads/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Docker](https://www.docker.com/)

## Instalación

1. **Clona el repositorio:**

   Abre una terminal y clona el repositorio en tu sistema local con el siguiente comando:

   ```bash
   git clone https://github.com/fabriciolopretto/Administracion-de-Modelos.git
   cd Administracion-de-Modelos

2. **Inicializa Docker Deskopt**

3. **Corre el contenedor:**

    3.1 **Desde el directorio raíz del Dockerfile, ejecuta desde a terminal:**

        docker build -t image_name .

        docker run -it --name container_name -p 5000:5000 -v "$(pwd)/TP_Final/mlflow/experiments/models/mlruns:/app/mlruns" image_name

4. **Dentro del contenedor, ejecuta los scripts:**

    4.1 **Data**
    python Distributions.py
    python HeatMap.py
    
    4.2 **Modelos**
    python RegLog.py
    python KNN.py
    python SVC.py
    python TreeClasf.py

    4.3 **Artefactos**
    python registro_reg_log
    python predictions_reglog_model.py

5.  **Desde navegador web:**

    http://localhost:5000

## Contiene:

scprits con varios modelos de ML sobre el DataSet con datos de características a partir de imagenes de cancer de mama.
Dockerfile
requeriments.txt
config.ini para base de datos postgres
runs con artefactos
