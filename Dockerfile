FROM python:3.11-slim

ENV HF_ENDPOINT=https://hf-mirror.com
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载模型（构建时下载，运行时秒级启动）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
