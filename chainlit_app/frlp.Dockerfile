FROM python:3.12-slim

# Establecemos el directorio de trabajo dentro del contenedor
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /code

# Mount UV and install dependencies
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --system -r pyproject.toml

# Copiamos todo el contenido del proyecto. .dockerignore se encarga de ignorar los archivos que no queremos copiar
COPY . .

COPY .chainlit/no_upload_config.toml /code/.chainlit/config.toml
COPY public-frlp /code/public

EXPOSE 80
CMD ["python", "-m", "chainlit", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80"]