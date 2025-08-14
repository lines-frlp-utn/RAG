FROM python:3.12-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /code

# Mount UV and install dependencies
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --system -r pyproject.toml

# Copiamos todo el contenido del proyecto. .dockerignore se encarga de ignorar los archivos que no queremos copiar
COPY . .

# Comando por defecto para ejecutar la aplicación cuando se inicia el contenedor
CMD ["python", "-m", "chainlit", "run","--host", "0.0.0.0", "--port", "80", "app/main.py"]