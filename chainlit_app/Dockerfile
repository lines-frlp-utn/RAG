# Install uv
FROM python:3.12

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /code

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Copy the project into the image
# Copiamos todo el contenido del directorio app al directorio /code/app
COPY app app

COPY .chainlit .chainlit/
COPY chainlit.md .python-version pyproject.toml uv.lock ./
COPY public public

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

#EXPONEMOS EL PUERTO
EXPOSE 8005

# Comando por defecto para ejecutar la aplicación cuando se inicia el contenedor
CMD ["uv", "run", "python", "-m", "chainlit", "run","--host", "0.0.0.0", "--port", "8005", "app/main.py"]