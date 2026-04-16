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
conda create -n rag_chatbot python=3.11 -y
conda activate rag_chatbot
```

### 1.2 Cài đặt dependencies

```powershell
cd <path to rag_chatbot>

# Cài thư viện chính
pip install -r requirements.txt

# Cài thư viện cho evaluation (optional)
pip install ragas datasets langchain-ollama langchain-huggingface
```

### 1.3 Cài Ollama

**Download:** https://ollama.com/download/windows

Sau khi cài xong:

```powershell
# Pull LLM model (mặc định: llama3.1:8b)
ollama pull llama3.1:8b

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

Trên Windows, Ollama thường tự chạy như service sau khi cài → bỏ qua bước này. Kiểm tra bằng:
```powershell
curl http://localhost:11434
# Nếu thấy "Ollama is running" → OK
```

Nếu chưa chạy thì mới cần:
```powershell
ollama serve
```

**Terminal 3 — Backend (FastAPI):**
```powershell
cd <path to rag_chatbot>
conda activate rag_chatbot

uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ **Backend ready:** http://localhost:8000/api/v1/docs (Swagger UI)

### Cách 2: Docker Compose (production-like)

**Terminal 1 — Ollama (vẫn chạy trên host để dùng GPU):**
```powershell
# Windows: Ollama tự chạy như service. Verify:
curl http://localhost:11434
```

**Terminal 2 — Docker Compose (Qdrant + Redis + Backend):**
```powershell
cd <path to rag_chatbot>
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

### 6.1 Datasets có sẵn

| Dataset | File | Số lượng |
|---|---|---|
| Single-turn | `evaluation/datasets/techviet_qa_v2.json` | 56 test cases, 12 category |
| Multi-turn  | `evaluation/datasets/techviet_multiturn_v1.json` | 16 hội thoại / 61 turns, 9 category |

Bao phủ: factual, multi-hop, comparison, temporal, negation/refusal, hard_out_of_scope, multi_intent, coreference, topic_shift, self_correction...

### 6.2 Single-turn

```powershell
# Smoke test — 5 cases, bỏ RAGAS
python -m evaluation.run_evaluation --limit 5 --no-ragas

# Full — 56 cases, có RAGAS (Faithfulness, ContextPrecision, ...)
python -m evaluation.run_evaluation
```

**Thời gian:** smoke ~1 phút, full ~20-40 phút tuỳ model (8B chậm hơn 3B).

### 6.3 Multi-turn

```powershell
# Smoke — 2 hội thoại, bỏ RAGAS
python -m evaluation.run_multiturn_evaluation --limit 2 --no-ragas

# Full — 16 hội thoại
python -m evaluation.run_multiturn_evaluation
```

Script truyền `conversation_history` giữa các turn → test thực tế `QueryRewriter` và xử lý coreference/topic-shift.

### 6.4 Kết quả

Report lưu ở `evaluation/reports/report_<timestamp>.json` (single-turn) hoặc `multiturn_report_<timestamp>.json`. Shape chính:

```json
{
  "retrieval_metrics": {
    "hit_at_k": 1.0,
    "recall_at_k": 0.97,
    "mrr": 0.95,
    "refusal_accuracy": 0.0,
    "num_answerable": 48,
    "num_unanswerable": 8
  },
  "generation_refusal_accuracy": 1.0,
  "ragas_metrics": {
    "faithfulness": 0.78,
    "answer_relevancy": 0.85,
    "context_precision": 1.0,
    "context_recall": 0.67
  },
  "ragas_coverage": {
    "faithfulness": {"scored": 46, "failed": 2, "total": 48}
  },
  "ragas_per_sample": [
    {"id": "factual_001", "status": "ok", "scores": {...}}
  ]
}
```

Multi-turn có thêm `retrieval_metrics_by_turn_position` và `retrieval_metrics_by_category`.

**Chi tiết (quy tắc chấm, judge model, debug failures):** xem [evaluation/README.md](evaluation/README.md)

---

## Cấu trúc thư mục

```
rag_chatbot/
├── backend/           # FastAPI backend
│   ├── api/          # Endpoints (chat, documents, health)
│   ├── services/     # Core services (ingestion, retrieval, llm...)
│   ├── core/         # Resilience, logging, exceptions
│   └── config/       # Settings (Pydantic)
├── evaluation/
│   ├── datasets/
│   │   ├── techviet_qa_v2.json         # 56 single-turn cases
│   │   └── techviet_multiturn_v1.json  # 16 conversations / 61 turns
│   ├── metrics.py                      # IR metrics + RAGAS wrapper
│   ├── run_evaluation.py               # Single-turn runner
│   ├── run_multiturn_evaluation.py     # Multi-turn runner
│   └── reports/                        # Report JSON
├── data/
│   ├── techviet_docs/   # Input PDFs (nằm ở project root: ../data/)
│   ├── processed/       # Metadata sidecar JSON
│   └── uploads/         # API uploads
├── models/           # Cached HuggingFace models
├── scripts/          # CLI tools (ingest_documents.py, start_ollama.bat)
├── docker/           # Dockerfile + docker-compose.yml
├── setup_embedding_models.py  # Cache models (RUN ONCE)
├── RAG.md            # 📚 Tài liệu kỹ thuật đầy đủ
└── README.md         # 👈 File này
```

---

## Tham số quan trọng (`.env`)

```env
# LLM (dùng cho chat + reuse cho RAGAS judge)
OLLAMA_MODEL=llama3.1:8b
LLM_TEMPERATURE=0.0

# RAGAS (eval-only). Rỗng = reuse OLLAMA_MODEL.
RAGAS_JUDGE_MODEL=
RAGAS_TIMEOUT_SECONDS=600
RAGAS_MAX_RETRIES=3

# Embedding
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
EMBEDDING_DEVICE=cpu

# Chunking
SECTION_MAX_CHUNK_TOKENS=600
SECTION_OVERLAP_SENTENCES=2
SECTION_SEMANTIC_MIN_SCORE=0.15

# Retrieval
TOP_K_RETRIEVAL=40
TOP_K_RERANK=10
USE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_DEVICE=cpu
USE_HYBRID_SEARCH=true
HYBRID_ALPHA=0.5
RETRIEVAL_SCORE_THRESHOLD=0.3

# Query rewriting (multi-turn)
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_MIN_TURNS=2

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

**Restart container (giữ data):**
```powershell
docker restart qdrant
```

**Nếu container chưa tồn tại:**
```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

⚠️ `docker rm qdrant` sẽ xoá container nhưng **không xoá volume** `qdrant_data` → index giữ nguyên. Chỉ rm khi bạn thực sự muốn reset.

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

# 2. Start services
docker start qdrant                                              # hoặc docker run lần đầu
# Ollama: Windows tự chạy — verify bằng: curl http://localhost:11434
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Ingest documents (lần đầu hoặc khi thêm doc mới)
python scripts/ingest_documents.py ./data/techviet_docs/

# 4. Test API
# Swagger UI: http://localhost:8000/api/v1/docs

# 5. Run evaluation
python -m evaluation.run_evaluation --limit 5 --no-ragas           # single-turn smoke
python -m evaluation.run_multiturn_evaluation --limit 2 --no-ragas # multi-turn smoke
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