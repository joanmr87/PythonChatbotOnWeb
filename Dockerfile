# Imagen base
FROM python:3.9-slim-buster

# Configurar directorio de trabajo
WORKDIR /app

# Copiar el código de la aplicación al contenedor
COPY . /app

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt


# Exponer el puerto 5000 (Flask corre en el puerto 5000 por defecto)
EXPOSE 5000

# Ejecutar la aplicación
CMD ["python", "app.py"]
