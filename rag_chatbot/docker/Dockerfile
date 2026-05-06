# ==================================================================
# RAG Chatbot — Dockerfile
# ==================================================================
# Multi-stage build:
#   - builder: install Python deps + download HF models
#   - runtime: slim image, non-root user, only what's needed to run
#
# Build (from project root):
#   docker build -f docker/Dockerfile -t rag-chatbot:latest .
# ==================================================================

# ------------------------------------------------------------------
# Stage 1: builder
# ------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

WORKDIR /build

# Build tools needed to compile some Python wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Isolated venv — easy to copy to runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install deps first (cached layer — only rebuilds when requirements change)
COPY requirements-api.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-api.txt

# Bake HF models into image so container starts up immediately
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base')"


# ------------------------------------------------------------------
# Stage 2: runtime
# ------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    TRANSFORMERS_CACHE=/opt/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# curl for HEALTHCHECK + libgomp1 for torch
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --shell /bin/bash --uid 1000 rag

# Copy installed deps + baked models from builder
COPY --from=builder --chown=rag:rag /opt/venv /opt/venv
COPY --from=builder --chown=rag:rag /opt/models /opt/models

WORKDIR /app

# Copy app code — data/ and models/ are excluded by .dockerignore
COPY --chown=rag:rag backend/ backend/
COPY --chown=rag:rag scripts/ scripts/

# Create writable dirs for uploads + processed metadata
RUN mkdir -p /app/data/uploads /app/data/processed && \
    chown -R rag:rag /app

USER rag

EXPOSE 8000

# Liveness probe — uses /health (no dependency check)
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
