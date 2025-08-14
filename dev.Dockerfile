FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1

RUN --mount=type=bind,source=./chainlit_app/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=./vectordbs/Chroma/requirements.txt,target=requirements-chroma.txt \
    --mount=type=bind,source=./vectordbs/Milvus/requirements.txt,target=requirements-milvus.txt \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r pyproject.toml && uv pip install -r requirements-chroma.txt && uv pip install -r requirements-milvus.txt

