# syntax=docker/dockerfile:1.7
# Clinical RAG API container — Phase F
#
# Single-stage on python:3.11-slim. The image bakes in:
#   - Python deps from requirements.txt
#   - Application code (RAG_chat_pipeline + platform/api)
#   - The winner sentence-transformer (BiomedNLP-PubMedBERT) so cold
#     start doesn't pay an HF Hub download
#
# At runtime the container fetches:
#   - chunked_docs + silver Parquet from Blob (USE_BLOB_DATA=true)
#   - prod-index/faiss_winner/ from Blob into /tmp/faiss_winner cache
#     (USE_BLOB_INDEX=true)
#   - LLM completions from Azure AI Foundry (TARGET_URL + MODEL_API_KEY)
#
# Auth in cloud: Managed Identity is picked up automatically by
# DefaultAzureCredential in azure_client.py + data_provider.py + the
# FAISS download in embeddings_manager.py. No keys baked into the image.
#
# Build:  docker build -t clinical-rag:dev .
# Run:    docker run --rm -p 5000:5000 --env-file .env clinical-rag:dev
#         (requires az login on host + ~/.azure mounted, OR an SP env trio
#          AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID, OR
#          deploy to ACA where Managed Identity covers it)

FROM python:3.11-slim AS runtime

# Avoid interactive prompts and noisy output, force unbuffered stdout
# so logs reach the orchestrator immediately.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# System libs needed at runtime by faiss-cpu and sentence-transformers.
# build-essential is intentionally NOT installed — every wheel we use
# (torch, faiss-cpu, pyarrow) ships compiled binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker layer caches them across code edits
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code
COPY RAG_chat_pipeline/ ./RAG_chat_pipeline/
COPY platform/ ./platform/

# Bake the winner sentence-transformer. The 9-model dir at models/ in
# the repo would balloon the image to ~6GB; copying just the winner
# keeps it near ~3.5GB total.
COPY models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/ \
     ./models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/

# Run as non-root. Create a uid that matches no host user; ACA accepts
# any non-zero uid.
RUN useradd --uid 10001 --create-home --shell /bin/false appuser \
    && mkdir -p /app/audit /tmp/faiss_winner \
    && chown -R appuser:appuser /app /tmp/faiss_winner
USER appuser

# Make the package importable from inside platform/api/ where gunicorn
# will exec. PYTHONPATH=/app means `from RAG_chat_pipeline...` works
# even though the cwd is /app/platform/api.
ENV PYTHONPATH=/app
WORKDIR /app/platform/api

EXPOSE 5000

# Healthcheck queries /health which returns 503 until the bot is ready.
# 90s start-period covers the cold-start FAISS download + load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://localhost:5000/health',timeout=3); sys.exit(0 if r.status==200 else 1)"

# 1 worker because the embedding model + FAISS index together hold ~1GB
# in process RAM — N workers means N copies = OOM.
# threads=4 handles concurrent requests; gthread worker class would be
# better for SSE streaming but adds gevent/eventlet complexity. Sync
# threads are fine for the v2 traffic profile.
# timeout=180 because gpt-5-nano with reasoning can take 60s+.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "180", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
