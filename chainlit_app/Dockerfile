FROM python:3.12-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /code

# Mount UV and install dependencies
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv export --no-dev  > requirements.txt && \
    uv pip install --system -r requirements.txt

# Copiamos todo el contenido del proyecto. .dockerignore se encarga de ignorar los archivos que no queremos copiar
COPY . .

#EXPONEMOS EL PUERTO
EXPOSE 8005

# Comando por defecto para ejecutar la aplicación cuando se inicia el contenedor
CMD ["python", "-m", "chainlit", "run","--host", "0.0.0.0", "--port", "80", "app/main.py"]