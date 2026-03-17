# RAG Enterprise Chatbot

A **production-ready Retrieval-Augmented Generation (RAG)** chatbot system for enterprise document question answering. This project demonstrates modern AI engineering best practices with **vLLM integration** for high-performance LLM inference.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![vLLM](https://img.shields.io/badge/vLLM-0.4+-orange.svg)
![.NET](https://img.shields.io/badge/.NET-8.0-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Key Features

### Core RAG Capabilities
- **Multi-format Document Support**: PDF, DOCX, TXT, Markdown
- **Table-Aware Processing**: Intelligent extraction and representation of tabular data
- **Hybrid Search**: Combines vector similarity with keyword search (RRF fusion)
- **Reranking**: Cross-encoder model for improved retrieval accuracy
- **Streaming Responses**: Real-time token streaming for better UX
- **Conversation Memory**: Multi-turn conversation support

### Production-Ready Features
- **vLLM Integration**: High-performance LLM inference with OpenAI-compatible API
- **Circuit Breaker Pattern**: Prevents cascade failures with automatic recovery
- **Retry with Exponential Backoff**: Resilient network operations
- **OpenAI Fallback**: Automatic fallback when vLLM is unavailable
- **Prometheus Metrics**: Full observability with Grafana dashboards
- **Structured JSON Logging**: Production logging for log aggregators
- **Rate Limiting**: Protect API from overload
- **Health Checks**: Kubernetes-ready liveness/readiness probes

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WPF Desktop Client                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐│
│  │ Upload Docs │  │  Chat Input  │  │  Streaming Response Display ││
│  └─────────────┘  └──────────────┘  └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Circuit Breaker │ Rate Limiter │ Metrics │ Structured Logs   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                      API Layer                                 │ │
│  │   POST /upload-document   POST /chat   GET /metrics            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    RAG Pipeline                                │ │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────────────┐  │ │
│  │  │ Ingestion│──│ Embedding│──│ Retrieval│──│ LLM Generation │  │ │
│  │  │ Service │  │ Service  │  │ Service │  │    Service      │  │ │
│  │  └─────────┘  └──────────┘  └─────────┘  └─────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐     ┌──────────┐
   │  Chroma  │    │  Redis   │     │  vLLM    │
   │ /Qdrant  │    │  Cache   │     │ Server   │
   └──────────┘    └──────────┘     └──────────┘
                                         │
                                    ┌────┴────┐
                                    │ Fallback│
                                    │ OpenAI  │
                                    └─────────┘
```
   │ /Qdrant  │    │  Cache   │     │ /Local   │
   └──────────┘    └──────────┘     │   LLM    │
                                    └──────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- .NET 8.0 SDK (for WPF frontend)
- Redis (optional, for caching)
- NVIDIA GPU (for vLLM) or OpenAI API key (fallback)

### 1. Backend Setup

```bash
# Clone and navigate to project
cd rag_chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your settings
```

### 2. Start vLLM Server (Local LLM)

```bash
# Option A: Using deployment script (recommended)
python scripts/deploy_vllm.py start --model microsoft/Phi-3-mini-4k-instruct

# Option B: Using Docker directly
docker run -d --gpus all \
    -p 8001:8000 \
    --name rag-vllm \
    vllm/vllm-openai:latest \
    --model microsoft/Phi-3-mini-4k-instruct \
    --trust-remote-code

# Check server status
python scripts/deploy_vllm.py status

# View recommended models
python scripts/deploy_vllm.py models
```

### 3. Run Backend API

```bash
python scripts/run_server.py
```

### 4. Frontend Setup (WPF)

```bash
# Navigate to frontend folder (separate project)
cd ../fe_rag_chatbot

# Restore packages
dotnet restore

# Build
dotnet build

# Run
dotnet run
```

### 3. Using Docker

```bash
cd docker

# Start all services
docker-compose up -d

# Start with Qdrant (instead of Chroma)
docker-compose --profile qdrant up -d
```

## 📁 Project Structure

```
# Backend (Python/FastAPI)
rag_chatbot/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   └── routes/
│   │       ├── chat.py          # Chat endpoints
│   │       ├── documents.py     # Document management
│   │       └── health.py        # Health checks
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── services/
│   │   ├── ingestion.py         # Document processing
│   │   ├── document_parser.py   # PDF/DOCX parsing
│   │   ├── chunker.py           # Text chunking
│   │   ├── table_extractor.py   # Table extraction
│   │   ├── embeddings.py        # Embedding generation
│   │   ├── vector_store.py      # Vector DB operations
│   │   ├── retrieval.py         # Hybrid search
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   ├── llm.py               # LLM integration
│   │   ├── cache.py             # Redis caching
│   │   └── conversation.py      # Conversation memory
│   └── evaluation/
│       ├── evaluator.py         # RAG evaluation
│       └── metrics.py           # Quality metrics
├── scripts/
│   ├── ingest_documents.py      # CLI document ingestion
│   ├── run_evaluation.py        # Evaluation runner
│   └── run_server.py            # Server starter
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.dev.yml
├── tests/
├── requirements.txt
├── .env.example
└── README.md

# Frontend (WPF C#/.NET 8) - Separate Project
fe_rag_chatbot/
├── App.xaml                      # Application resources
├── MainWindow.xaml               # Main chat interface
├── RAGChatbot.csproj             # Project file
├── appsettings.json              # API configuration
├── ViewModels/
│   └── MainViewModel.cs          # MVVM ViewModel
├── Models/
│   └── Models.cs                  # Data models
├── Services/
│   ├── ChatService.cs             # Chat API client
│   └── DocumentService.cs         # Document API client
└── Converters/
    └── Converters.cs              # XAML converters
```

## 🔧 Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/local) | openai |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `EMBEDDING_MODEL` | Embedding model | BAAI/bge-large-en-v1.5 |
| `VECTOR_DB_TYPE` | Vector DB (chroma/qdrant) | chroma |
| `USE_RERANKER` | Enable reranking | true |
| `USE_HYBRID_SEARCH` | Enable hybrid search | true |
| `USE_CACHE` | Enable Redis caching | true |

## 📖 API Documentation

Once the server is running, access the interactive API docs:

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

### Key Endpoints

#### Upload Document
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <document_file>
department: (optional) HR
tags: (optional) quarterly,finance
```

#### Chat
```http
POST /api/v1/chat
Content-Type: application/json

{
  "question": "What was the Q1 revenue?",
  "conversation_id": null,
  "stream": true
}
```

## 🔍 Document Processing Pipeline

### Text Extraction
1. **PDF**: Uses pypdf + pdfplumber for text and tables
2. **DOCX**: Uses python-docx with structure preservation
3. **TXT/MD**: Direct parsing with header detection

### Table Handling
Tables are converted to row-based text representation:
```
Table: Sales Data

Row 1:
  Product: A
  Price: 10
  Country: Japan

Row 2:
  Product: B
  Price: 20
  Country: USA
```

### Chunking Strategy
- **Token-based**: ~500 tokens per chunk with 50-token overlap
- **Boundary-aware**: Respects paragraphs and sentences
- **Heading-preserving**: Keeps section context

## 🔬 Evaluation

Industry-standard evaluation framework: See [RAG_EVALUATION_PROFESSIONAL.md](RAG_EVALUATION_PROFESSIONAL.md)

**Comprehensive Framework (16 metrics across 3 dimensions):**
- 5 Retrieval Metrics (Precision@5, Recall@5, F1, MRR, NDCG)
- 5 Generation Metrics (BLEU, ROUGE, Semantic Sim, Relevancy, Faithfulness)
- 6 End-to-End Metrics (Context Precision/Recall/F1, Hallucination, Citation, Correctness)

**Quick Start:**
```bash
# Upload document
python scripts/ingest_documents.py paper.pdf

# Run evaluation
python scripts/evaluate_rag.py --limit 5    # Test 5 questions
python scripts/evaluate_rag.py              # Full 26 questions
python scripts/evaluate_rag.py --save-report # Save JSON report
```

**Output:** Detailed metrics table + summary statistics + JSON report

## 🎨 Frontend Features

The WPF desktop application includes:

- **Document Upload**: Drag-and-drop or file picker
- **Real-time Streaming**: Watch answers appear token by token
- **Source Citations**: See which documents were used
- **Conversation History**: Multi-turn chat support
- **Material Design**: Modern, clean interface

## 🔒 Advanced Features

### Hybrid Search (RRF)
Combines vector similarity with keyword matching using Reciprocal Rank Fusion:
```python
score = α/k+rank_vector + (1-α)/k+rank_keyword
```

### Reranking
Uses BGE-reranker cross-encoder to re-score retrieved chunks based on actual question-passage relevance.

### Conversation Memory
Maintains conversation context for follow-up questions with automatic cleanup of old conversations.

### Response Caching
Redis-based caching with MD5 query hashing for repeated questions.

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html
```

## 📊 Performance Optimization

- **Batch Embedding**: Process documents in batches of 32
- **Async Operations**: All I/O operations are async
- **Connection Pooling**: Reuse database connections
- **Model Caching**: Pre-load embedding and reranker models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG patterns inspiration
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework

---

**Built with ❤️ for AI Engineering interviews**
