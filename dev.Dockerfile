FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1

RUN --mount=type=bind,source=./chainlit_app/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=./vectordbs/Chroma/requirements.txt,target=requirements-chroma.txt \
    --mount=type=bind,source=./vectordbs/Milvus/pyproject.toml,target=pyproject_milvus.toml \
    --mount=type=cache,target=/root/.cache/uv \
    uv export --project pyproject_milvus.toml --format requirements-txt > requirements_milvus.txt && \
    uv pip install -r pyproject.toml -r requirements-chroma.txt -r requirements_milvus.txt

