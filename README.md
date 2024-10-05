# TP Final AMq2: Puesta en producción modelo.

## Descripción
Este trabajo permite gestionar modelos de Aprendizaje de Máquina. Generando artefactos de evaluación, bas de datos de predicciones, etc.

## Instalación
1. Clona el repositorio:
   ```bash
   git clone https://github.com/fabriciolopretto/AMq2
2. Abrir Docker Deskopt.
3. Desde la ruta del Dockerfile, ejecutar en terminal:
    docker build -t image_models .
    docker run -it --name container_name -p 5000:5000 -v "ruta local"/TP_Final/mlflow/experiments/models/mlruns:/app/mlruns image_name
4. ejecutar desde el contenedor: python RegLog.py, KNN.py, SVC.py, TreeClasf.py, etc (o usar runs previas)
5. ejecutar desde el contenedor: mlflow ui --host 0.0.0.0 --port 5000 &
6. Desde navegador web: http://localhost:5000

Contiene: scprits con varios modelos de ML sobre el DataSet con datos de características a partir de imagenes de cancer de mama.
          Dockerfile
          requeriments.txt
          config.ini para base de datos postgres
          runs con artefactos
