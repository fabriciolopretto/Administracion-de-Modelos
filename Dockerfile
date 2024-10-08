FROM python:3.8-slim

# Crear directorio de trabajo
WORKDIR /opt

# Instala Git y las dependencias
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia los archivos necesarios al contenedor
COPY requirements.txt /opt

# Copiar los scripts
COPY scripts /opt
COPY dags/ /opt
COPY airflow/dags/ /opt

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 5000 y 8080
EXPOSE 5000 8080

# Comando por defecto
CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "postgresql+psycopg2://airflow:airflow@postgres/airflow", "--default-artifact-root", "/mlflow/mlruns"]
