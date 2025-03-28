#!/bin/bash

# Print each command before executing it (for debugging purposes)
set -x

uv pip install -r /workspace/chainlit_app/requirements.txt
uv pip install -r /workspace/vectordbs/Milvus/requirements.txt
uv pip install -r /workspace/vectordbs/Chroma/requirements.txt

# Your startup script logic goes here
echo "Starting up..."