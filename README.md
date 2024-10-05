Pasos para utilizar:

1 - Clonar le repositorio localmente.
2 - Abrir Docker Deskopt.
3 - Desde la ruta del Dockerfile, ejecutar en terminal:
    docker build -t image_models .
    docker run -it --name container_name -p 5000:5000 -v ruta/TP_Final/mlflow/experiments/models/mlruns:/app/mlruns image_name
    python RegLog.py, KNN.py, SVC.py, TreeClasf.py (o usar runs previas)
    mlflow ui --host 0.0.0.0 --port 5000 &
4 - python predictions_reglog_model.py (o usar runs previas)
5 - Desde navegador web:
    http://localhost:5000

Contiene: scprits con varios modelos de ML sobre el DataSet con datos de características a partir de imagenes de cancer de mama.
          Dockerfile
          requeriments.txt
          config.ini para base de datos postgres
          runs con artefactos
