# RAG Chatbot - Ollama Local Setup

Production-ready RAG (Retrieval-Augmented Generation) chatbot system for Windows using **Ollama** for local LLM inference.

## Features

- **Ollama Integration**: Native Windows LLM inference with GPU support
- **HuggingFace Embeddings**: High-quality BGE embeddings (CPU)
- **ChromaDB**: Fast vector database for document search
- **Streaming Responses**: Real-time token streaming
- **Conversational Memory**: Multi-turn conversations with context
- **Document Processing**: PDF, DOCX, TXT, MD support
- **Resilience Patterns**: Circuit breaker, retry with backoff

## Requirements

- Windows 10/11
- NVIDIA GPU (RTX 3060 or better)
- Python 3.10+
- Ollama for Windows

## Quick Start

### 1. Install Ollama

```powershell
# Download from website
https://ollama.ai/download

# Or use winget
winget install Ollama.Ollama
```

### 2. Pull a model

```powershell
# Phi-3 (recommended, 3.8B parameters)
ollama pull phi3

# Other options
ollama pull llama3.2    # Meta's Llama 3.2
ollama pull qwen2.5     # Alibaba Qwen
ollama pull mistral     # Mistral 7B
```

### 3. Setup Python environment

```powershell
# Create conda environment
conda create -n rag_chatbot python=3.10 -y
conda activate rag_chatbot

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure .env

The `.env` file is already configured for Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=phi3
```

### 5. Start the system

```powershell
# Start Ollama + RAG Backend
scripts\start_ollama.bat
```

The script will:
1. Check Ollama installation
2. Start Ollama service
3. Download model if needed
4. Start RAG backend (port 8000)

### 6. Test

```powershell
# Health check
curl http://localhost:8000/health

# Upload document (example)
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf"

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is in the document?\", \"stream\": false}"
```

## Project Structure

```
rag_chatbot/
├── backend/
│   ├── api/          # FastAPI routes
│   ├── services/     # Core services (LLM, embeddings, retrieval)
│   └── config/       # Configuration
├── data/
│   ├── uploads/      # Uploaded documents
│   ├── chroma_db/    # Vector database
│   └── processed/    # Processed documents
├── scripts/
│   ├── start_ollama.bat      # Main startup script
│   ├── run_server.py         # Backend server
│   ├── ingest_documents.py   # Document ingestion
│   └── evaluate_rag.py       # RAG evaluation
└── .env              # Configuration
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Ollama    │────▶│  RAG API    │────▶│   ChromaDB   │
│   (GPU)     │     │  (FastAPI)  │     │  (Vectors)   │
└─────────────┘     └─────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  HuggingFace │
                    │  Embeddings  │
                    │    (CPU)     │
                    └──────────────┘
```

## Configuration

Key settings in `.env`:

```env
# LLM
LLM_PROVIDER=ollama
OLLAMA_MODEL=phi3
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512

# Embeddings
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=cpu

# Retrieval
TOP_K_RETRIEVAL=8
TOP_K_RERANK=3
USE_RERANKER=True
```

## Useful Commands

### Document Ingestion

```powershell
# Ingest single file
python scripts/ingest_documents.py path/to/document.pdf

# Ingest folder
python scripts/ingest_documents.py path/to/documents/

# With metadata
python scripts/ingest_documents.py document.pdf --department HR --tags "policy,onboarding"
```

### Evaluation

```powershell
# Run full evaluation
python scripts/evaluate_rag.py

# Quick test (5 samples)
python scripts/evaluate_rag.py --limit 5

# Save report
python scripts/evaluate_rag.py --save-report
```

### Ollama Management

```powershell
# List models
ollama list

# Check running models
ollama ps

# Remove model
ollama rm model_name

# Update model
ollama pull model_name
```

## Troubleshooting

### Ollama not responding

```powershell
# Check if Ollama service is running
curl http://localhost:11434

# Restart Ollama service
# Stop the ollama.exe process and restart
scripts\start_ollama.bat
```

### Out of GPU memory

Reduce context length in `.env`:
```env
LLM_MAX_TOKENS=256  # Reduce from 512
```

### Slow responses

1. Use smaller model: `ollama pull phi3` (3.8B)
2. Reduce retrieval: `TOP_K_RETRIEVAL=5`
3. Disable reranker: `USE_RERANKER=False`

## Documentation

- [WINDOWS_NO_DOCKER.md](WINDOWS_NO_DOCKER.md) - Detailed Windows setup guide
- [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md) - Vietnamese setup guide
- [RAG_EVALUATION_PROFESSIONAL.md](RAG_EVALUATION_PROFESSIONAL.md) - RAG evaluation metrics
- [EVALUATION_QUICK_START.md](EVALUATION_QUICK_START.md) - Quick evaluation guide

## Archived vLLM Documentation

Previous vLLM-based documentation has been archived to `backup_/` directory:
- `backup_/README.md` - Original vLLM setup
- `backup_/PRODUCTION_DEPLOY.md` - Production deployment
- `backup_/QUICKSTART.md` - Quick start guide

## License

MIT

## Support

For issues or questions, please check the documentation in the `docs/` folder or refer to [WINDOWS_NO_DOCKER.md](WINDOWS_NO_DOCKER.md) for Windows-specific guidance.
