FROM python:3.12-slim

WORKDIR /code

RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    uv pip install -r requirements.txt --system

COPY main.py .

CMD ["sh", "-c", "while true; do sleep 30; done"]
