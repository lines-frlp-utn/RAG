FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN --mount=type=bind,source=dependency-install.sh,target=dependency-install.sh \
    sh dependency-install.sh

# Install git
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
