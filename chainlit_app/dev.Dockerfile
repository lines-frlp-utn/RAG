FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Change the working directory to the `code` directory
WORKDIR /workspace/chainlit_app