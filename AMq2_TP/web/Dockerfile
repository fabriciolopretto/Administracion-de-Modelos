FROM php:8.0-apache

# Establece el directorio de trabajo
WORKDIR ./

# Copia el archivo de requisitos e instala las dependencias
COPY ./requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# Copiar los archivos al directorio de trabajo
COPY ./*.php /var/www/html/

# Exponer el puerto
EXPOSE 80

