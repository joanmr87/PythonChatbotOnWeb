# Imagen base
FROM python:3.9-slim-buster

# Instalar build-essential para C++ y actualizar SQLite3
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget libsqlite3-dev && \
    wget https://www.sqlite.org/2023/sqlite-autoconf-3360000.tar.gz && \
    tar xvfz sqlite-autoconf-3360000.tar.gz && \
    cd sqlite-autoconf-3360000 && \
    ./configure && \
    make && \
    make install && \
    ldconfig

# Configurar directorio de trabajo
WORKDIR /app

# Copiar el código de la aplicación al contenedor
COPY . /app

# Instalar las dependencias del proyecto y la nueva versión de sqlite3
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install pysqlite3

# Exponer el puerto 5000 (Flask corre en el puerto 5000 por defecto)
EXPOSE 5000

# Ejecutar la aplicación
CMD ["python", "app.py"]
