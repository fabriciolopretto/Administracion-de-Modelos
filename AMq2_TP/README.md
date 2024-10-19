# Integrantes del equipo  
  
Gustavo Julián Rivas - a1620  
Myrna Lorena Degano - a1618  
Santiago José Olaciregui - a1611  
Fabricio Lopretto - a1616  
  
  
# Guía de comandos  
  
Crear y ejecutar los contenedores en segundo plano.  
  
```Bash  
docker-compose up -d
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
├── airflow/                   # Configuración de Apache Airflow - Para orquestar tareas y flujos de trabajo relacionados con el modelo  
│   ├── config/                # Uso interno de Airflow  
│   ├── dags/                  # DAGs de Airflow  
│   ├── logs/                  # Uso interno de Airflow  
│   ├── plugin/                # Uso interno de Airflow  
│   ├── secrets/               # Uso interno de Airflow  
│   ├── Dockerfile             # Dockerfile para Airflow   
│   └── requirements.txt       # Dependencias de Airflow  
│  
├── app/                       # Código de la aplicación que interactúa con el modelo de ML a través de API  
│   ├── app.py                 # Código de la aplicación - Prototipo streamlit  
│   ├── fapi_app.py            # Fast API  
│   ├── Dockerfile             # Dockerfile para la aplicación    
│   ├── requirements.txt       # Dependencias de Python    
│   ├── utils/                 # Utilidades  
│   │   ├── functions.py       # Funciones auxiliares    
│   │   ├── sample_values.csv  # Datos para simulación   
│   └── model/                 # Utilidades  
│       ├── model.pkl          # Modelo entrenado    
│       └── scaler.pkl         # Standard Scaler entrenado    
│    
├── mlflow/                    # Configuración de MLflow - Servidor para gestionar experimentos y artefactos  
│   ├── Dockerfile             # Dockerfile para MLFlow  
│   └── requirements.txt       # Dependencias de MLFlow  
│    
├── postgres/                  # Configuración de PostgreSQL - Base de datos para almacenar información Airflow y MLFlow    
│   ├── Dockerfile             # Dockerfile para PostgreSQL    
│   └── requirements.txt       # Dependencias de PostgreSQL    
│    
├── docker-compose.yml         # Archivo de configuración de Docker Compose    
├── .env                       # Variables de entorno (!) MODIFICAR AIRFLOW_UID con el ID del usuario del SO o bien 50000  
└── README.md    
  
  
# Acceso a los servicios  
  
Fast API  
http://localhost:8800  
  
MLflow  
http://localhost:5000  
  
Airflow  
http://localhost:8080  
  
Minio  
http://localhost:9000  
  
Streamlit  
http://localhost:8501  


