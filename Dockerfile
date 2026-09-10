FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV HF_HOME=/cache/huggingface

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app ./app
COPY rag ./rag
COPY agent ./agent
COPY scripts ./scripts
COPY data ./data

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]