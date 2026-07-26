FROM python:3.11-slim

ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV HF_ENDPOINT=https://hf-mirror.com
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/model_cache && chmod -R 777 /app/model_cache
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
