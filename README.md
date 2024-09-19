Pasos para utilizar:

1 - Clonar el repositorio localmente.
2 - Abrir Docker Desktop.
3 - Desde la ruta del Dockerfile, ejecutar en terminal:
    docker build -t image_models .
    docker run -it --name container_models -p 5000:5001 -v "$(pwd)/experiments/models/mlruns:/app/mlruns" image_models
    python RegLog.py, KNN.py, SVC.py, TreeClasf.py (o usar runs previas)
    mlflow ui --host 0.0.0.0 --port 5000 &
4 - python predictions_reglog_model.py (o usar runs previas)
5 - Desde navegador web:
    http://localhost:5000

Contiene: scripts con varios modelos de ML sobre el DataSet con datos de características a partir de imagenes de cancer de mama.
          Dockerfile
          requeriments.txt
          config.ini para base de datos postgres
          runs con artefactos
