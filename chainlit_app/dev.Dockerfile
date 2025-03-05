FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Change the working directory to the vs code workspace
WORKDIR /workspace/chainlit_app

# Create venv and install dependencies
RUN  --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync

# Fixes a bug so vscode detects venv in current directory
RUN ln -s /.venv /workspace/chainlit_app/.venv