FROM python:3.12-slim

# Establecemos el directorio de trabajo dentro del contenedor
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /code

# Install dependencies using pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiamos todo el contenido del proyecto. .dockerignore se encarga de ignorar los archivos que no queremos copiar
COPY . .

EXPOSE 80
CMD ["python", "-m", "chainlit", "run", "app/main.py", "-h", "0.0.0.0", "-p", "80"]