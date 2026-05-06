# RAG Chatbot

RAG chatbot với Ollama, Qdrant, BGE embeddings. PDF/DOCX/TXT, hybrid search, reranking, streaming SSE, Prometheus + Grafana.

**Kiến trúc chi tiết:** [RAG.md](RAG.md)

---

## Yêu cầu

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/download/windows)
- Python 3.11 + conda — chỉ cần nếu chạy evaluation trên host

---

## Bước 1 — Cài Ollama và pull model

```bash
ollama pull llama3.1:8b
```

Windows: Ollama tự chạy như service sau khi cài. Kiểm tra: `curl http://localhost:11434`

---

## Bước 2 — Cấu hình `.env`

Mở file `.env` ở root, đổi password Grafana:

```env
GF_SECURITY_ADMIN_PASSWORD=your_password
```

Các giá trị hay chỉnh thêm:

```env
OLLAMA_MODEL=llama3.1:8b     # đổi nếu dùng model khác
EMBEDDING_DEVICE=cpu         # hoặc cuda nếu có GPU
TOP_K_RERANK=10
```

---

## Bước 3 — Chạy stack

Từ thư mục `rag_chatbot/`:

```bash
# Dev — code thay đổi → backend tự reload, không rebuild
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# Production
docker compose up -d --build
```

**Lần build đầu** mất ~10-15 phút (cài torch + bake BGE models ~750MB). Lần sau nhanh nhờ layer cache.

**Stack:** backend · Qdrant · Redis · Prometheus · Grafana · Redis Exporter

---

## Bước 4 — Kiểm tra

```bash
docker compose ps
curl http://localhost:8000/health
```

Tất cả services phải `healthy`. Backend mất ~30-60 giây sau khi start.

---

## Bước 5 — Ingest tài liệu

Thư mục data nằm ngoài `rag_chatbot/`, được mount vào container tại `/app/data/`:

```
C:\thanhhuynh\Chatbot\
├── data\
│   └── techviet_docs\    ← đặt PDF/DOCX/TXT vào đây
└── rag_chatbot\          ← project root
```

Chạy ingest:

```bash
docker compose exec backend python scripts/ingest_documents.py /app/data/techviet_docs/
```

Kiểm tra tại: http://localhost:6333/dashboard

---

## Bước 6 — Dùng API

Swagger UI: http://localhost:8000/api/v1/docs

```bash
# Chat
curl -X POST http://localhost:8000/api/v1/chat/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "what is leave policy?", "stream": false}'

# Upload document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@./data/techviet_docs/policy.pdf"
```

---

## Evaluation (chạy trên host)

Setup lần đầu:

```bash
conda create -n rag_chatbot python=3.11 -y
conda activate rag_chatbot
pip install -r requirements.txt
pip install ragas datasets
python setup_embedding_models.py    # download BGE models → ./models/ (~750MB)
```

Chạy eval (Qdrant + Redis phải đang chạy — `docker compose ps`):

```bash
conda activate rag_chatbot

# Smoke test — nhanh ~1 phút
python -m evaluation.run_evaluation --limit 5 --no-ragas
python -m evaluation.run_multiturn_evaluation --limit 2 --no-ragas

# Full — single-turn 56 cases + RAGAS (~20-40 phút)
python -m evaluation.run_evaluation

# Full — multi-turn 16 conversations
python -m evaluation.run_multiturn_evaluation
```

Report lưu tại `evaluation/reports/`.

---

## Monitoring

| URL | Mục đích |
|---|---|
| http://localhost:3000 | Grafana — `admin` / `GF_SECURITY_ADMIN_PASSWORD` |
| http://localhost:9090 | Prometheus |
| http://localhost:8000/metrics | Raw metrics |
| http://localhost:6333/dashboard | Qdrant UI |

Dashboard **"RAG Chatbot"** tự load trong Grafana.

---

## Lệnh hàng ngày

```bash
docker compose logs -f backend          # xem log realtime
docker compose ps                       # status + health
docker compose restart backend          # restart sau khi sửa .env
docker compose build backend            # rebuild sau khi thêm dependency
docker compose down                     # dừng, giữ data
docker compose down -v                  # dừng + xóa toàn bộ data
```

---

## Cấu trúc thư mục

```
C:\thanhhuynh\Chatbot\
├── data\                             # dữ liệu — mounted vào container tại /app/data
│   ├── techviet_docs\                # input PDF/DOCX/TXT
│   ├── processed\                    # metadata sidecar JSON (tự sinh khi ingest)
│   └── uploads\                      # files upload qua API
└── rag_chatbot\                      # project root
    ├── docker-compose.yml            # base stack (production)
    ├── docker-compose.dev.yml        # dev overrides (hot-reload)
    ├── .env                          # config — không commit vào git
    ├── backend/
    │   ├── api/main.py               # FastAPI app + /metrics
    │   ├── api/v1/endpoints/         # chat, documents, health
    │   ├── services/                 # ingestion, retrieval, llm, cache
    │   └── core/
    │       ├── metrics.py            # custom Prometheus metrics
    │       ├── logging.py            # JSON/console + correlation ID
    │       └── resilience.py         # CircuitBreaker + retry
    ├── evaluation/
    │   ├── datasets/
    │   │   ├── techviet_qa_v2.json           # 56 single-turn test cases
    │   │   └── techviet_multiturn_v1.json    # 16 conversations / 61 turns
    │   ├── run_evaluation.py
    │   ├── run_multiturn_evaluation.py
    │   └── reports/
    ├── models/                       # HuggingFace cache local (eval only)
    ├── scripts/ingest_documents.py
    ├── docker/
    │   ├── Dockerfile                # multi-stage, non-root, BGE models baked in
    │   ├── prometheus/prometheus.yml
    │   └── grafana/
    ├── setup_embedding_models.py     # download models local (eval only)
    └── RAG.md
```

---

## Troubleshooting

| Lỗi | Fix |
|---|---|
| Backend `unhealthy` | `docker compose logs backend` xem lỗi |
| Ollama connection refused | `curl http://localhost:11434` — nếu lỗi thì `ollama serve` |
| Grafana không có data | Gửi 1 request chat; kiểm tra `docker compose ps` |
| Prometheus target `DOWN` | Backend phải `healthy` trước |
| PDF không parse được | File scan (ảnh) → cần OCR hoặc chuyển DOCX |

---

## Checklist deploy lên server

- [ ] `.env` không commit vào git
- [ ] `GF_SECURITY_ADMIN_PASSWORD` là password mạnh
- [ ] `CORS_ORIGINS` đặt domain cụ thể, bỏ `["*"]`
- [ ] Dùng `docker compose up -d --build` (production, không dùng dev override)
- [ ] Backup volume `qdrant_data` định kỳ

---

**Tài liệu liên quan:** [RAG.md](RAG.md) · [evaluation/README.md](evaluation/README.md)
