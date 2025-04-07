FROM python:3.10-slim


RUN pip install --no-cache-dir --upgrade pip


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY main.py .


CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009", "--reload"]