# Integrantes del equipo  
  
Gustavo Julián Rivas - a1620  
Myrna Lorena Degano - a1618  
Santiago José Olaciregui - a1611  
Fabricio Lopretto - a1616  
  
  
# Guía de comandos  
  
Crear y ejecutar los contenedores en segundo plano.  
  
```Bash  
docker-compose up airflow-init && docker-compose up -d  
```  
  
Ver estado de los contenedores  
```Bash  
docker-compose ps -a  
```  
  
Detener y eliminar todos los contenedores, redes, imágenes y volúmenes asociados al proyecto.   
  
```Bash  
docker compose down --rmi all --volumes --remove-orphans  
```  
  
Ver logs para troubleshooting  
```Bash  
docker-compose logs > logs.txt  
```  
  
# Estructura del proyecto  
  
AMq2_TP/  
│  
├── airflow/                 # Configuración de Apache Airflow - Para orquestar tareas y flujos de trabajo relacionados con el modelo.  
│   ├── config/              # Uso interno de Airflow  
│   ├── dags/                # DAGs de Airflow  
│   ├── logs/                # Uso interno de Airflow  
│   ├── plugin/              # Uso interno de Airflow  
│   ├── secrets/             # Uso interno de Airflow  
│   ├── Dockerfile           # Dockerfile para Airflow  
│   └── requirements.txt     # Dependencias de Airflow  
│  
├── app/                     # Código de la aplicación interactúa con el modelo de ML a través de una API REST.  
│   ├── main.py              # Código de la aplicación (Flask, FastAPI, etc.)  
│   ├── Dockerfile           # Dockerfile para la aplicación  
│   └── requirements.txt     # Dependencias de Python  
│  
├── minio/                   # Almacenamiento S3 compatible para guardar artefactos de ML  
│   ├── Dockerfile           # Dockerfile para MinIO  
│   └── requirements.txt     # Dependencias de MinIO  
│  
├── mlflow/                  # Configuración de MLflow - Servidor para gestionar experimentos y artefactos.  
│   ├── Dockerfile           # Dockerfile para MLFlow  
│   └── requirements.txt     # Dependencias de MLFlow  
│  
├── postgres/                # Configuración de PostgreSQL - Base de datos para almacenar información Airflow.  
│   ├── Dockerfile           # Dockerfile para PostgreSQL  
│   └── requirements.txt     # Dependencias de PostgreSQL  
│  
├── redis/                   # Configuración de Redis  
│   └── Dockerfile           # Usado para almacenamiento en caché y otros propósitos.  
│   └── requirements.txt     # Dependencias de Redis  
│  
├── web/                     # Front end web para interfaz de usuario final  
│   ├── index.php            # Home page  
│   └── Dockerfile           # Dockerfile para servidor web  
│   └── requirements.txt     # Dependencias de la app web  
│  
├── docker-compose.yml       # Archivo de configuración de Docker Compose  
├── .env                     # Variables de entorno (!) MODIFICAR AIRFLOW_UID con el ID del usuario del SO o bien 50000.  
├── README.md  
  
  
# Acceso a los servicios  
  
Fast API  
http://localhost:8000  
  
Front End  
http://localhost  
  
MLflow  
http://localhost:5000  
  
Airflow  
http://localhost:8080  
  
Minio  
http://localhost:9000  
  
  
# TO DO  
Verificar archivos requirements.txt y agregar/eliminar requerimientos necesarios de cada módulo. (descomentar Pip Install)  
Desarrollar el código en cada carpeta (aplicación web, DAGs de Airflow, etc.).  

