# Usamos la imagen oficial de Python como base
FROM python:3.11.9

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /code

# Copiamos el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instalamos las dependencias especificadas en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el contenido del directorio app al directorio /code/app
COPY app app

COPY .chainlit .chainlit/
COPY chainlit.md .
COPY public public

#EXPONEMOS EL PUERTO
EXPOSE 8005

# Comando por defecto para ejecutar la aplicación cuando se inicia el contenedor
CMD ["python", "-m", "chainlit", "run","--host", "0.0.0.0", "--port", "8005", "app/main.py"]