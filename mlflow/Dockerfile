# Usa una imagen base con Python
FROM python:3.8-slim

# Configura el directorio de trabajo
WORKDIR /app

# Instala Git y las dependencias
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copia los archivos necesarios al contenedor
COPY requirements.txt ./
COPY experiments/models/ ./

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto por defecto de MLflow
EXPOSE 5000

# Define el comando por defecto para el contenedor
CMD ["/bin/bash"]

