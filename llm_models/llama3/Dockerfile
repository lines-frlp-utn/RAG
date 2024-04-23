FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Set environment variable
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON"

RUN pip install --no-cache-dir --upgrade pip

RUN pip3 install huggingface-hub
RUN huggingface-cli download bartowski/Llama-3-Smaug-8B-GGUF Llama-3-Smaug-8B-Q6_K.gguf --local-dir . --local-dir-use-symlinks False

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8007", "--reload"]
