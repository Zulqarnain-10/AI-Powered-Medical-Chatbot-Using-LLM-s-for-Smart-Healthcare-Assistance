# MedBot - Flask RAG service (Hugging Face Space, Docker SDK)
FROM python:3.10-slim

WORKDIR /app

# Writable caches for the arbitrary-uid user Spaces run containers as
ENV HF_HOME=/app/.cache \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

COPY requirements.txt setup.py ./
COPY src ./src
RUN pip install -r requirements.txt

# Bake the embedding model into the image so cold starts skip the download
RUN python -c "from src.helper import download_hugging_face_embeddings; download_hugging_face_embeddings()"

COPY . .
RUN chmod -R 777 /app

EXPOSE 8080
# Long timeout: the first boot on a fresh Pinecone project builds the index (several minutes)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "900", "app:app"]
