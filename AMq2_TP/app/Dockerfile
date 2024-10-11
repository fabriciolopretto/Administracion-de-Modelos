# Usa una imagen base de Python
FROM python:3.9

# Establece el directorio de trabajo
WORKDIR ./

# Copia el archivo de requisitos e instala las dependencias
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY ./*.py .

# Expone el puerto que usará FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

