# Multi-stage Dockerfile for RAG Enterprise Chatbot

# Stage 1: Base image with Python
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model (optional, increases image size but faster startup)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Stage 3: Production
FROM dependencies as production

# Copy application code
COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/processed /app/data/chroma_db /app/logs

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Development (optional)
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio httpx black isort flake8

# Copy all code
COPY . .

# Create directories
RUN mkdir -p /app/data/uploads /app/data/processed /app/data/chroma_db /app/logs

# Run with reload
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
