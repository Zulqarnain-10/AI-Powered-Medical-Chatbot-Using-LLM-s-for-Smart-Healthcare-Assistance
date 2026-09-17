# MedBot - Flask RAG service (Hugging Face Space, Docker SDK)
FROM python:3.10-slim

WORKDIR /app

# Writable caches for the arbitrary-uid user Spaces run containers as
ENV HF_HOME=/app/.cache \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# CPU-only torch first: the default PyPI wheel drags in multi-GB CUDA libraries
# this CPU-only container never uses
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt setup.py ./
RUN pip install -r requirements.txt

# Bake the embedding model into the image so cold starts skip the download;
# kept above the source COPY so code edits never invalidate this layer
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
RUN chmod -R 777 /app

EXPOSE 8080
# Long timeout: the first boot on a fresh Pinecone project builds the index (several minutes)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "900", "app:app"]
