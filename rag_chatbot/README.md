# RAG Chatbot — Production-Ready

**RAG (Retrieval-Augmented Generation)** chatbot với Ollama, Qdrant, BGE embeddings. Hỗ trợ PDF/DOCX/TXT, hybrid search, reranking, streaming SSE.

**Tài liệu kỹ thuật chi tiết:** xem [RAG.md](RAG.md) để hiểu kiến trúc RAG đầy đủ.

---

## Yêu cầu

- **Python 3.11+** (khuyến nghị dùng conda)
- **Ollama** (LLM local inference)
- **Docker** (cho Qdrant vector DB)
- **GPU** (optional, tăng tốc embedding/LLM)

---

## Bước 1: Setup môi trường

### 1.1 Tạo conda environment

```powershell
conda create -n rag_chatbot python=3.10 -y
conda activate rag_chatbot
```

### 1.2 Cài đặt dependencies

```powershell
cd C:\thanhhuynh\Chatbot\rag_chatbot

# Cài thư viện chính
pip install -r requirements.txt

# Cài thư viện cho evaluation (optional)
pip install ragas datasets langchain-community pyyaml
```

### 1.3 Cài Ollama

**Download:** https://ollama.com/download/windows

Sau khi cài xong:

```powershell
# Pull LLM model (mặc định: llama3.2)
ollama pull llama3.2

# Kiểm tra
ollama list
```

---

## Bước 2: Cache embedding models (BẮT BUỘC)

**⚠️ QUAN TRỌNG:** Backend sẽ **không khởi động** nếu chưa cache models.

```powershell
# Download BAAI/bge-base-en-v1.5 và BAAI/bge-reranker-base
python setup_embedding_models.py
```

**Thời gian:** ~5-10 phút (download ~750MB models về `./models/`)

**Output mong đợi:**
```
======================================================================
  EMBEDDING MODELS SETUP
======================================================================

📦 Caching models to ./models directory...

1️⃣  Downloading BAAI/bge-base-en-v1.5 (embedding model)...
   ✅ Embedding model cached successfully

2️⃣  Downloading BAAI/bge-reranker-base (reranker model)...
   ✅ Reranker model cached successfully

======================================================================
✅ Setup Complete!
======================================================================
```

**Chỉ cần chạy 1 lần.** Sau đó backend khởi động **tức thì** mà không cần download.

---

## Bước 3: Khởi động services

### Cách 1: Development (khuyến nghị cho dev)

**Terminal 1 — Qdrant (Vector DB):**
```powershell
docker run --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Terminal 2 — Ollama (LLM):**
```powershell
ollama serve
```

**Terminal 3 — Backend (FastAPI):**
```powershell
cd C:\thanhhuynh\Chatbot\rag_chatbot
conda activate rag_chatbot

uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ **Backend ready:** http://localhost:8000/api/v1/docs (Swagger UI)

### Cách 2: Docker Compose (production-like)

**Terminal 1 — Ollama (vẫn chạy trên host để dùng GPU):**
```powershell
ollama serve
```

**Terminal 2 — Docker Compose (Qdrant + Redis + Backend):**
```powershell
cd C:\thanhhuynh\Chatbot\rag_chatbot
docker-compose -f docker/docker-compose.yml up -d
```

Backend trong Docker tự kết nối Ollama qua `host.docker.internal:11434`.

**Kiểm tra:**
```powershell
# Xem logs backend
docker logs rag-backend -f

# Healthcheck
curl http://localhost:8000/health
```

---

## Bước 4: Ingest documents (đưa tài liệu vào Qdrant)

### 4.1 Chuẩn bị documents

Đặt file PDF/DOCX/TXT/MD vào `./data/techviet_docs/` (hoặc thư mục bất kỳ).

### 4.2 Chạy ingestion

```powershell
# Ingest toàn bộ thư mục
python scripts/ingest_documents.py ./data/techviet_docs/

# Hoặc ingest 1 file cụ thể
python scripts/ingest_documents.py ./data/techviet_docs/handbook.pdf
```

**Output:**
```
INFO - Processing: handbook.pdf
INFO - Parsing document...
INFO - Preprocessing (8 operations)...
INFO - Chunking (section-aware + semantic)...
INFO - Embedding 127 chunks...
INFO - Upserting to Qdrant...
INFO - ✅ Successfully ingested: handbook.pdf (127 chunks, 3 tables)
```

**Metadata sidecar** được lưu vào `./data/processed/{document_id}_meta.json` — dùng để audit.

### 4.3 Kiểm tra Qdrant

```powershell
# Hoặc vào Qdrant UI: http://localhost:6333/dashboard
```

---

## Bước 5: Test API

### 5.1 Healthcheck

```powershell
curl http://localhost:8000/health
```

### 5.2 Upload document qua API

```powershell
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@./data/techviet_docs/policy.pdf"
```

### 5.3 Chat query

**Swagger UI:** http://localhost:8000/api/v1/docs → `/api/v1/chat/chat` → Try it out

**cURL:**
```powershell
curl -X POST "http://localhost:8000/api/v1/chat/chat" `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"what is leave policy?\", \"stream\": false}'
```

**Response:**
```json
{
  "answer": "Employees are entitled to 12 days annual leave per year [Source 1: policy.pdf, Page 5]",
  "sources": [
    {"filename": "policy.pdf", "page": 5, "score": 0.87, "chunk_type": "text"}
  ],
  "conversation_id": "abc-123-def"
}
```

---

## Bước 6: Evaluation (đánh giá chất lượng RAG)

### 6.1 Chuẩn bị test dataset

File `evaluation/datasets/eval_dataset.json`:

```json
[
  {
    "question": "How many days of annual leave?",
    "ground_truth_answer": "12 days per year",
    "relevant_doc_ids": ["policy.pdf"]
  }
]
```

### 6.2 Smoke test (nhanh)

```powershell
# 5 cases, không dùng RAGAS (LLM-judged metrics)
python -m evaluation.run_evaluation --limit 5 --no-ragas
```

**Thời gian:** ~1-2 phút  
**Output:** `evaluation/reports/report_YYYYMMDD_HHMMSS.json`

### 6.3 Full evaluation

```powershell
# 50 cases, bao gồm RAGAS (Faithfulness, ContextPrecision...)
python -m evaluation.run_evaluation
```

**Thời gian:** ~10-20 phút (tùy số lượng test case + LLM speed)

### 6.4 Kết quả

```json
{
  "retrieval_metrics": {
    "hit@5": 0.92,
    "recall@5": 0.78,
    "mrr": 0.85
  },
  "ragas_metrics": {
    "faithfulness": 0.89,
    "context_precision": 0.82,
    "response_relevancy": 0.91
  }
}
```

**Chi tiết:** xem [evaluation/README.md](evaluation/README.md)

---

## Cấu trúc thư mục

```
rag_chatbot/
├── backend/           # FastAPI backend
│   ├── api/          # Endpoints (chat, documents, health)
│   ├── services/     # Core services (ingestion, retrieval, llm...)
│   ├── core/         # Resilience, logging, exceptions
│   └── config/       # Settings (Pydantic)
├── evaluation/       # RAGAS + IR metrics
├── data/
│   ├── techviet_docs/   # Input documents (PDF/DOCX/TXT)
│   ├── processed/       # Metadata sidecar JSON
│   └── uploads/         # API uploads
├── models/           # Cached HuggingFace models
├── scripts/          # CLI tools (ingest_documents.py)
├── docker/           # Dockerfile + docker-compose.yml
├── setup_embedding_models.py  # Cache models (RUN ONCE)
├── RAG.md            # 📚 Tài liệu kỹ thuật đầy đủ
└── README.md         # 👈 File này
```

---

## Tham số quan trọng (`.env`)

```env
# LLM
OLLAMA_MODEL=llama3.2
LLM_TEMPERATURE=0.0

# Embedding
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768

# Chunking
SECTION_MAX_CHUNK_TOKENS=600
SECTION_OVERLAP_SENTENCES=2
SECTION_SEMANTIC_MIN_SCORE=0.15

# Retrieval
TOP_K_RETRIEVAL=40
TOP_K_RERANK=5
HYBRID_ALPHA=0.7
RETRIEVAL_SCORE_THRESHOLD=0.3

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=documents
```

**Xem đầy đủ:** [backend/config/settings.py](backend/config/settings.py)

---

## Troubleshooting

### ❌ Backend không khởi động — "Models directory not found"

**Nguyên nhân:** Chưa chạy `setup_embedding_models.py`

**Fix:**
```powershell
python setup_embedding_models.py
```

### ❌ Ollama connection error

**Kiểm tra Ollama đang chạy:**
```powershell
curl http://localhost:11434/api/version
```

**Hoặc restart Ollama:**
```powershell
# Windows: tắt Ollama trong System Tray → mở lại
ollama serve
```

### ❌ Qdrant connection error

**Kiểm tra Qdrant đang chạy:**
```powershell
curl http://localhost:6333/collections
```

**Hoặc restart Qdrant:**
```powershell
docker stop qdrant
docker rm qdrant
docker run --name qdrant -p 6333:6333 qdrant/qdrant
```

### ❌ Ingestion lỗi — "Failed to parse PDF"

**Nguyên nhân:** File PDF scan (hình ảnh, không có text layer)

**Fix:** Dùng OCR tool (Tesseract) hoặc chuyển sang DOCX.

### ❌ Retrieval trả về kết quả không liên quan

**Kiểm tra:**
1. Document đã được ingest chưa? → `curl http://localhost:6333/collections/documents`
2. Query có quá chung chung? → Thử query cụ thể hơn
3. Threshold quá cao? → Giảm `RETRIEVAL_SCORE_THRESHOLD` từ 0.3 → 0.2

---

## Development workflow

```powershell
# 1. Activate env
conda activate rag_chatbot

# 2. Start services (3 terminals)
docker run --name qdrant -p 6333:6333 qdrant/qdrant    # Terminal 1
ollama serve                                             # Terminal 2
uvicorn backend.api.main:app --reload                   # Terminal 3

# 3. Ingest documents
python scripts/ingest_documents.py ./data/techviet_docs/

# 4. Test API
curl http://localhost:8000/api/v1/docs

# 5. Run evaluation
python -m evaluation.run_evaluation --limit 5 --no-ragas

# 6. Check logs
# Backend logs hiện trực tiếp trên terminal 3
```

---

## Production deployment

### Docker Compose (khuyến nghị)

```powershell
# 1. Cache models trên host (1 lần)
python setup_embedding_models.py

# 2. Build image
docker-compose -f docker/docker-compose.yml build

# 3. Start services
docker-compose -f docker/docker-compose.yml up -d

# 4. Ingest documents
docker exec rag-backend python scripts/ingest_documents.py /app/data/techviet_docs/

# 5. Check logs
docker logs rag-backend -f
```

### Kubernetes (nâng cao)

- Prebake models vào Docker image (xem [setup_embedding_models.py](setup_embedding_models.py))
- Chuyển `ConversationManager` từ in-memory → Redis
- Qdrant: dùng Qdrant Cloud hoặc StatefulSet
- Secrets: K8s Secret thay vì `.env` file

---

## Tài liệu liên quan

- **[RAG.md](RAG.md)** — Kiến trúc RAG chi tiết (chunking, embedding, retrieval, prompts...)
- **[evaluation/README.md](evaluation/README.md)** — Hướng dẫn đầy đủ về evaluation
- **[INTERVIEW_NOTES.md](INTERVIEW_NOTES.md)** — Câu hỏi phỏng vấn về dự án

---

## License

MIT