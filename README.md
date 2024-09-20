## Pasos para Utilizar

1. Clonar el repositorio localmente.

2. Abrir Docker Desktop.

3. Desde la ruta del Dockerfile, ejecutar en terminal:

   ```bash
   docker build -t image_models .
   docker run -it --name container_models -p 5001:5001 -v "$(pwd)/experiments/models/mlruns:/app/mlruns" image_models
   ```

   Luego, ejecutar uno de los siguientes comandos (o usar runs previas):
   
   ```bash
   python RegLog.py
   python KNN.py
   python SVC.py
   python TreeClasf.py
   ```

   Finalmente, iniciar MLflow UI:

   ```bash
   mlflow ui --host 0.0.0.0 --port 5001 &
   ```

4. Ejecutar el script de predicciones (o usar runs previas):

   ```bash
   python predictions_reglog_model.py
   ```

5. Acceder a la interfaz web de MLflow desde el navegador:

   ```
   http://localhost:5001
   ```
