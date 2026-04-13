# RAG Chuyên Sâu — Học Qua Dự Án RAG_Chatbot

> Tài liệu này vừa mô tả **đúng những gì dự án triển khai**, vừa bổ sung kiến thức so sánh bên ngoài để ôn phỏng vấn.
> Mọi nội dung được gắn nhãn rõ ràng để bạn không nhầm:
>
> - ✅ **[DÙNG TRONG DỰ ÁN]** — thực sự có trong code, kèm file + dòng cụ thể.
> - 📚 **[KIẾN THỨC SO SÁNH]** — kỹ thuật/khái niệm ngoài, đưa vào để hiểu bối cảnh và trả lời phỏng vấn. **Không có trong code dự án này**.
>
> Khi phỏng vấn, hãy nói rõ: "Trong dự án em chỉ dùng X. Em biết thêm Y, Z thường dùng khi…"

---

## Mục lục

1. [RAG là gì](#1-rag-là-gì)
2. [Workflow tổng thể](#2-workflow-tổng-thể)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Ingestion Pipeline](#4-ingestion-pipeline)
5. [Embedding Service](#5-embedding-service)
6. [Vector Store (Qdrant + BM25)](#6-vector-store-qdrant--bm25)
7. [Retrieval Pipeline](#7-retrieval-pipeline)
8. [LLM Service & Prompt](#8-llm-service--prompt)
9. [Conversation Manager](#9-conversation-manager)
10. [Cache (Redis)](#10-cache-redis)
11. [API Layer (FastAPI)](#11-api-layer-fastapi)
12. [Resilience](#12-resilience)
13. [Logging có Correlation ID](#13-logging-có-correlation-id)
14. [Evaluation](#14-evaluation)
15. [Docker Deployment](#15-docker-deployment)
16. [Câu hỏi phỏng vấn về dự án này](#16-câu-hỏi-phỏng-vấn-về-dự-án-này)

---

## 1. RAG là gì

**RAG (Retrieval-Augmented Generation)** = LLM + kho tri thức ngoài. Thay vì bắt LLM tự nhớ, ta:

1. **Retrieve**: tìm các đoạn văn bản liên quan đến câu hỏi từ vector DB.
2. **Augment**: nhét các đoạn đó vào prompt làm "context".
3. **Generate**: LLM trả lời **chỉ dựa vào context** đó.

Lợi ích chính: (a) giảm hallucination; (b) cập nhật tri thức mà không phải train lại; (c) có citation/audit; (d) data có thể on-prem.

### 📚 [KIẾN THỨC SO SÁNH] RAG vs Fine-tune vs Long-context

> Các tiếp cận thay thế — **không dùng trong dự án**, chỉ để biết:

| Tiêu chí | RAG (dự án này) | Fine-tune | Long-context (nhét hết vào prompt) |
|---|---|---|---|
| Cập nhật tri thức | Thêm doc là xong | Phải train lại | Thay prompt |
| Chi phí mỗi query | Rẻ (embed + LLM) | Rẻ sau khi đã train | **Rất đắt** (tokens nhiều) |
| Chi phí onboarding | Vừa | Đắt (GPU train) | Rẻ |
| Citation | Có | Không | Khó trace |
| Hallucination | Thấp | Vẫn có | Thấp nhưng "lost in the middle" |
| Thích hợp khi | Corpus lớn, cập nhật thường xuyên | Domain hẹp, cố định, cần đổi style | Corpus < ~100k tokens |

---

## 2. Workflow tổng thể

```
INGESTION (offline — khi upload doc):
  Upload (PDF/DOCX/TXT/MD)
    → DocumentParser (pdfplumber / python-docx)
    → DocumentPreprocessor (8 bước clean)
    → SectionChunker (section-aware + semantic boundaries)
    → TableChunkBuilder (whole-table + row-batch chunks)
    → EmbeddingService (BGE-base, 768-dim)
    → Qdrant (COSINE) + rebuild BM25 index
    → IngestionMetadataWriter (sidecar JSON)

QUERY (online — mỗi câu hỏi):
  Question
    → Cache check (Redis, optional)
    → QueryRewriter (nếu history ≥ 2 turns)
    → Vector search (Qdrant, top 40)
    → Keyword search (BM25, top 40)
    → RRF fusion (α=0.7, k=60)
    → Score threshold filter (≥ 0.3)
    → CrossEncoder Reranker (top 5)
    → Format context + SYSTEM_PROMPT
    → LLM (Ollama llama3.2) stream tokens qua SSE
    → Lưu conversation + cache response
```

---

## 3. Cấu trúc thư mục

```
rag_chatbot/
├── backend/
│   ├── api/
│   │   ├── main.py                         # FastAPI entrypoint + lifespan
│   │   ├── middleware.py                   # CorrelationIdMiddleware
│   │   └── v1/endpoints/                   # chat.py, documents.py, health.py
│   ├── services/
│   │   ├── ingestion/
│   │   │   ├── parser.py                   # PDF/DOCX/TXT/MD parser
│   │   │   ├── preprocessor.py             # 8 bước làm sạch
│   │   │   ├── chunker.py                  # SectionChunker
│   │   │   ├── table_extractor.py          # TableExtractor
│   │   │   ├── table_chunk_builder.py      # TableChunkBuilder
│   │   │   ├── metadata_writer.py          # IngestionMetadataWriter
│   │   │   └── pipeline.py                 # DocumentIngestionService (orchestrator)
│   │   ├── embedding/service.py            # BGE via sentence-transformers
│   │   ├── vectorstore/qdrant.py           # Qdrant + BM25
│   │   ├── retrieval/
│   │   │   ├── pipeline.py                 # RetrievalService (hybrid + rerank)
│   │   │   ├── query_rewriter.py           # LLM-based rewrite
│   │   │   └── reranker.py                 # CrossEncoder rerank
│   │   ├── llm/
│   │   │   ├── service.py                  # Ollama qua OpenAI SDK
│   │   │   └── prompts.py                  # SYSTEM_PROMPT
│   │   ├── conversation.py                 # In-memory OrderedDict
│   │   └── cache.py                        # Redis cache
│   ├── core/
│   │   ├── exceptions.py                   # RAGException hierarchy
│   │   ├── logging.py                      # setup_logging + JsonFormatter
│   │   └── resilience.py                   # CircuitBreaker + retry
│   ├── models/document.py                  # TextBlock, Table, ParsedDocument
│   └── config/settings.py                  # Pydantic BaseSettings
├── evaluation/metrics.py                   # IR metrics + RAGAS
├── scripts/ingest_documents.py             # CLI batch ingest
└── docker/                                 # Dockerfile + docker-compose.yml
```

---

## 4. Ingestion Pipeline

Orchestrator: [backend/services/ingestion/pipeline.py](backend/services/ingestion/pipeline.py) — chỉ gọi các collaborator theo thứ tự: parse → preprocess → extract tables → chunk text → build table chunks → embed → store → metadata writer.

### 4.1 Parser

**File:** [backend/services/ingestion/parser.py](backend/services/ingestion/parser.py) (import `pdfplumber` dòng 91)
**Class:** `DocumentParser`

**Thư viện thực tế:**
- PDF → `pdfplumber`
- DOCX → `python-docx`
- TXT/MD → builtin + regex

**PDF parsing — heuristics đã code:**
- Cắt header/footer theo y-coordinate (`HEADER_Y_THRESHOLD=55`, `FOOTER_Y_THRESHOLD=780` cho A4).
- Extract table bounding boxes → extract words **ngoài** các vùng table.
- Group `words → lines → paragraphs` theo vertical gap.
- Phân loại **heading vs paragraph** theo font size (heading lớn hơn body +1.5pt) vì PDF không có "style".

**Output:** `ParsedDocument(text_blocks: List[TextBlock], tables: List[Table])`.

### 4.2 Preprocessor — 8 bước

**File:** [backend/services/ingestion/preprocessor.py](backend/services/ingestion/preprocessor.py)
**Class:** `DocumentPreprocessor`

| # | Method | Làm gì |
|---|---|---|
| 1 | `_op1_unicode_repair` | Fix mojibake, ligature (`ﬁ→fi`), zero-width, NFC normalize |
| 2 | `_op2_artifact_repair` | Rejoin word bị tách (`pro`+`f` → `prof`), fix khoảng trắng trước dấu câu |
| 3 | `_op3_title_page_splitter` | Tách title page bị merge thành 1 block |
| 4 | `_op4_heading_splitter` | Tách `4. Performance Management TechViet uses...` → heading + body |
| 5 | `_op5_frequency_dedup` | Xóa text xuất hiện ở ≥40% số trang (header/footer lặp) |
| 6 | `_op6_cross_page_merge` | Nối paragraph bị cắt ngang trang |
| 7 | `_op7_small_block_merge` | Gom block <50 chars vào neighbor |
| 8 | `_rebuild_sections` | Gán lại section context top-down từ heading |

Mỗi op là pure function trên `List[TextBlock]` → dễ test, dễ debug.

### 📚 [KIẾN THỨC SO SÁNH] Các parser phổ biến khác (không dùng trong dự án)

| Library | Ưu | Khi nào chọn |
|---|---|---|
| `pdfplumber` ✅ dự án | Extract text + table + tọa độ, kiểm soát sâu | PDF có layout, cần filter header/footer |
| `PyMuPDF` (fitz) | Nhanh nhất, có OCR nhẹ | Cần tốc độ |
| `unstructured` | One-liner cho mọi format, hỗ trợ OCR | Prototype nhanh, không cần tinh chỉnh |
| `LlamaParse` (managed) | Layout phức tạp (multi-column, bảng chồng) | Trả phí, chất lượng cao nhất |
| `Tesseract` (OCR) | PDF scan (hình ảnh) | File PDF không có text layer |

Dự án chọn `pdfplumber` vì đủ chính xác cho doc office/handbook và miễn phí, kiểm soát được heuristics.

### 4.3 Chunker — Section-Aware + Semantic

**File:** [backend/services/ingestion/chunker.py](backend/services/ingestion/chunker.py)
**Class:** `SectionChunker`

**Chiến lược:** kết hợp 2 tín hiệu:
- **Structural**: heading hierarchy + paragraph boundary.
- **Semantic**: cosine similarity giữa 2 câu liên tiếp (dùng chính embedding model).

**Pipeline (analyse → plan → build):**

1. **Analyse**: flatten → câu (protect abbreviation `Dr.`, `e.g.`...). Class `_EmbeddingBoundaryScorer` tính
   ```
   boundary_score[i] = 1.0 - cosine_similarity(sent_i, sent_{i+1})
   ```
   Score cao = 2 câu khác topic → điểm cắt tốt.

2. **Plan**: đi từng câu, cộng dồn token. Khi vượt `max_chunk_tokens`, chọn split point tốt nhất trong cửa sổ `semantic_look_back=3` câu gần nhất; ưu tiên paragraph boundary (signal=1.0), rồi semantic score (phải > `semantic_min_score=0.15`).

3. **Build**: prepend heading breadcrumb (`H1 > H2 > H3`) + overlap `overlap_sentences=2` câu cuối từ chunk trước.

**Token counting:** `tiktoken` encoding `cl100k_base` (chuẩn GPT-4). Fallback `len(text)//4` nếu tiktoken không có.

**Tham số ([settings.py](backend/config/settings.py)):**

| Setting | Giá trị | Ý nghĩa |
|---|---|---|
| `SECTION_MAX_CHUNK_TOKENS` | 600 | Trần cứng (BGE-base nhận 512, có dư cho prefix) |
| `SECTION_MIN_CHUNK_TOKENS` | 80 | Nhỏ hơn → merge với neighbor |
| `SECTION_OVERLAP_SENTENCES` | 2 | Chống mất context biên |
| `SECTION_SEMANTIC_LOOK_BACK` | 3 | Cửa sổ tìm split point |
| `SECTION_SEMANTIC_MIN_SCORE` | 0.15 | Ngưỡng ưu tiên semantic split |

### 📚 [KIẾN THỨC SO SÁNH] Các chiến lược chunking khác

| Chiến lược | Cách làm | Ưu | Nhược | Dự án có dùng? |
|---|---|---|---|---|
| **Fixed-size** | Cắt theo N tokens | Đơn giản nhất | Cắt giữa câu, mất context | ❌ |
| **Sentence/Recursive** | LangChain `RecursiveCharacterTextSplitter` — thử cắt theo `\n\n`, `\n`, `.`, `space` | Giữ câu, phổ biến | Không hiểu ngữ nghĩa | ❌ |
| **Semantic only** | Embedding similarity giữa câu → cắt ở điểm drop | Topic coherent | Đắt, khó tune threshold | ❌ |
| **Section-aware only** | Cắt theo heading H1/H2/H3 | Bám structure | Section dài vẫn phải cắt thêm | ❌ |
| **Section-aware + Semantic** (hybrid) | Kết hợp cả 2 | Tốt nhất cho doc có structure | Phức tạp hơn | ✅ **dự án dùng** |
| **Parent-child / Small-to-big** | Index chunk nhỏ (chính xác retrieve), trả về chunk cha (context rộng) | Retrieval chính xác + context đủ | Phức tạp, storage gấp đôi | ❌ |
| **Propositions** | Dùng LLM extract từng mệnh đề → embed | Retrieval cực chính xác | Rất tốn LLM call ở indexing | ❌ |

### 4.4 Table Extraction & Chunking

**File:** [table_chunk_builder.py](backend/services/ingestion/table_chunk_builder.py)
**Class:** `TableChunkBuilder`

- **Whole-table chunk** (`chunk_type="table"`) cho mọi bảng.
- Bảng > `_LARGE_TABLE_ROW_THRESHOLD=10` rows: thêm **row-batch chunks** (`_ROW_BATCH_SIZE=5`, `chunk_type="table_rows"`) cho retrieval tinh hơn.
- Format text dạng key:value (không dùng CSV/Markdown):
  ```
  Table: Name (Rows 1-5)
  Row 1:
    Header A: Value A
    Header B: Value B
  ```

### 4.5 Metadata Writer

**File:** [metadata_writer.py](backend/services/ingestion/metadata_writer.py)
**Class:** `IngestionMetadataWriter`

Ghi sidecar JSON `{document_id}_meta.json` ở `settings.PROCESSED_DIR` (mặc định `./data/processed/`). Ghi `status="completed"` (kèm `chunks_count`, `tables_count`) hoặc `status="failed"` (kèm `error`). Dùng làm audit trail, không thay DB.

---

## 5. Embedding Service

**File:** [backend/services/embedding/service.py](backend/services/embedding/service.py)
**Class:** `EmbeddingService`

### Model thực tế

- **Default:** `BAAI/bge-base-en-v1.5` (settings.py:48)
- **Dimension:** 768
- **Library:** `sentence-transformers`

### Asymmetric embedding

BGE là model **bất đối xứng** — query cần **instruction prefix**, passage thì không:

```python
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
```

Chỉ áp dụng khi `"bge" in model_name.lower()` (service.py:106). `embed_query()` thêm prefix, `embed_documents()` không.

### Best practices đã code

1. **Normalize vector** (`normalize_embeddings=True`) → cosine = dot product.
2. **Batch size = 32** khi encode documents (service.py:131).
3. **Async wrapper**: `asyncio.to_thread(self.model.encode, ...)` — encoding là CPU-bound, không được block event loop của FastAPI.
4. **Lazy singleton**: init 1 lần ở `initialize_services()` (5-min timeout), dùng chung mọi request.

### 📚 [KIẾN THỨC SO SÁNH] Các embedding model khác

| Model | Dim | Ngôn ngữ | Ghi chú | Dự án dùng? |
|---|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | 768 | EN | Top MTEB, asymmetric, free | ✅ |
| `BAAI/bge-m3` | 1024 | 100+ (multi) | Multi-lingual, multi-granularity | ❌ |
| `intfloat/e5-large-v2` | 1024 | EN | Asymmetric, mạnh | ❌ |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | EN | Nhỏ, nhanh, symmetric | ❌ |
| `OpenAI text-embedding-3-small/large` | 1536/3072 | Multi | Managed API, trả phí, chất lượng cao | ❌ |
| `Cohere embed-v3` | 1024 | Multi | Managed, có reranker kèm | ❌ |

**Khi nào đổi?** Dự án dùng tiếng Việt → nên cân nhắc `bge-m3` hoặc `multilingual-e5-large`.

**Fine-tune embedding** (không có trong dự án): dùng contrastive learning với triplet `(query, positive_passage, negative_passage)` trên data domain riêng → tăng recall đáng kể khi domain hẹp.

---

## 6. Vector Store (Qdrant + BM25)

**File:** [backend/services/vectorstore/qdrant.py](backend/services/vectorstore/qdrant.py)
**Class:** `VectorStoreService`

### Qdrant

- **Client:** `qdrant-client`
- **Distance:** `COSINE` (qdrant.py:136) — khớp BGE đã normalize.
- **Collection schema:** vector size 768, payload là `{content, metadata}`.
- **Filter:** `FieldCondition + MatchValue` trên payload field (vd lọc theo `department`, `document_id`).
- **Async:** mọi lời gọi Qdrant sync wrap bằng `asyncio.to_thread()`.

### BM25 (keyword search)

- **Library:** `rank_bm25` (`BM25Okapi`) (qdrant.py:55).
- **Index:** build **trong memory**, không lưu Qdrant.
- **Khi nào build:** lúc `initialize()` (dòng 146); rebuild sau `add_chunks` (dòng 214), sau `delete_document` (dòng 282), hoặc khi detect stale (dòng 256-257).
- **Tokenize:** đơn giản — `lowercase + split()`.

BM25 là sparse retrieval (keyword). Mạnh ở chỗ dense yếu: mã sản phẩm, tên riêng, viết tắt cụ thể.

### 📚 [KIẾN THỨC SO SÁNH] Các vector DB khác

| DB | Loại | Ưu | Khi nào chọn | Dự án dùng? |
|---|---|---|---|---|
| **Qdrant** | Self-host / Cloud | Filter mạnh, Rust, gRPC nhanh | Prod on-prem | ✅ |
| **FAISS** | Library | Nhanh nhất (Meta), không có persistence/filter | Research, batch eval | ❌ |
| **Chroma** | Embedded / server | Dễ dùng, Python-native | Prototype, dev local | ❌ |
| **Pinecone** | Managed cloud | Scale tự động, không cần ops | SaaS, không muốn tự host | ❌ |
| **Weaviate** | Self-host / Cloud | Hybrid search built-in, schema mạnh | Cần hybrid out-of-box | ❌ |
| **Milvus** | Self-host | Scale ngang rất tốt, nhiều index type | Corpus rất lớn (100M+) | ❌ |
| **pgvector** | Postgres extension | Dùng PG sẵn có, ACID, filter SQL | Đã có PG, không muốn thêm DB | ❌ |
| **Elasticsearch / OpenSearch** | Full-text + vector | Hybrid native, log-friendly | Đã dùng ES | ❌ |

### 📚 [KIẾN THỨC SO SÁNH] ANN index

Qdrant mặc định dùng **HNSW** (Hierarchical Navigable Small World) — graph-based, trade-off tốc độ/recall qua param `ef_search`. Các thuật toán khác: IVF (inverted file), PQ (product quantization — nén vector), HNSW-PQ (kết hợp). Dự án không cấu hình riêng, dùng default của Qdrant.

---

## 7. Retrieval Pipeline

### 7.1 Query Rewriter

**File:** [backend/services/retrieval/query_rewriter.py](backend/services/retrieval/query_rewriter.py)

**Vấn đề giải quyết:** câu follow-up `"nó có hiệu lực khi nào?"` → embedding không biết "nó" = gì → retrieve sai. Rewriter dùng LLM viết lại thành self-contained.

**Triển khai đã code:**
- Chỉ chạy khi `history >= QUERY_REWRITE_MIN_TURNS=2` (settings.py:81).
- Gọi lại Ollama LLM với `temperature=0`, `max_tokens=120` (query_rewriter.py:112-113).
- Nếu lỗi → trả **original query** (fallback an toàn, không bao giờ block retrieval).

### 📚 [KIẾN THỨC SO SÁNH] Các kỹ thuật query transformation khác

**Không có trong dự án**, nhưng cần biết khi phỏng vấn:

| Kỹ thuật | Cách làm | Khi nào dùng |
|---|---|---|
| **Pronoun rewrite** ✅ dự án | LLM viết lại query self-contained dựa vào history | Multi-turn chat |
| **HyDE** (Hypothetical Document Embeddings) | Bắt LLM "trả lời thử" query → embed câu trả lời giả → retrieve bằng embedding đó | Query ngắn, doc dài; đảo khoảng cách asymmetric |
| **Multi-query** | Sinh N variation của query → retrieve mỗi cái → union | Query mơ hồ, cần coverage rộng |
| **Step-back prompting** | Hỏi LLM câu tổng quát hơn trước → retrieve → rồi mới trả lời câu cụ thể | Câu hỏi cần background knowledge |
| **Query decomposition** | Chia câu phức thành nhiều câu con → retrieve riêng → tổng hợp | Multi-hop question |

### 7.2 Retrieval Service (hybrid + rerank)

**File:** [backend/services/retrieval/pipeline.py](backend/services/retrieval/pipeline.py)
**Class:** `RetrievalService`

**Các bước:**

1. Rewrite query (nếu đủ điều kiện).
2. **Dense search** (Qdrant) — top `TOP_K_RETRIEVAL=40`.
3. **Sparse search** (BM25) — top 40, nếu `USE_HYBRID_SEARCH=true`.
4. **RRF Fusion** (pipeline.py:125-148):
   ```
   score(d) = α / (k + rank_dense + 1) + (1 - α) / (k + rank_keyword + 1)
   ```
   - `α = HYBRID_ALPHA = 0.7` (ưu tiên dense)
   - `k = 60` (hằng số hard-coded trong hàm)
5. **Score threshold filter**: bỏ chunk có score < `RETRIEVAL_SCORE_THRESHOLD=0.3` (pipeline.py:88-103). **Áp dụng TRƯỚC reranker**.
6. **Reranker**: lấy top ~12 → rerank → top `TOP_K_RERANK=5` (pipeline.py:110-117).

**Vì sao RRF chứ không cộng score thẳng?** Score dense (cosine) và BM25 khác scale → cộng thẳng vô nghĩa. RRF chỉ dùng **rank** (vị trí) nên an toàn và là chuẩn công nghiệp.

### 7.3 Reranker

**File:** [backend/services/retrieval/reranker.py](backend/services/retrieval/reranker.py)

- **Model:** `BAAI/bge-reranker-base` (settings.py:73).
- **Class:** `CrossEncoder` từ `sentence_transformers` (reranker.py:32, 41).
- **Scoring:** raw logit → **sigmoid normalize** về `[0, 1]` (hàm `_sigmoid` dòng 74-80).

**Khác biệt cross-encoder vs bi-encoder (embedding):**
- **Bi-encoder (BGE embedding)**: encode query và passage độc lập → dot product. Nhanh, scale được triệu doc.
- **Cross-encoder (BGE reranker)**: cho cả `[query, passage]` vào model cùng lúc → 1 score. Chính xác hơn nhưng O(N) per query → chỉ rerank top-k nhỏ.

**Two-stage retrieval** (đây là pattern industry standard):
- Stage 1 (bi-encoder): 1M docs → 40.
- Stage 2 (cross-encoder): 40 → 5.

### 📚 [KIẾN THỨC SO SÁNH] Các reranker & kỹ thuật re-ranking khác

| Kỹ thuật | Mô tả | Dự án dùng? |
|---|---|---|
| **bge-reranker-base** ✅ | Cross-encoder, 278M params, nhanh | ✅ |
| **bge-reranker-large** | Chính xác hơn, chậm hơn (~560M) | ❌ |
| **Cohere Rerank API** | Managed, chất lượng cao | ❌ |
| **ColBERT** | Late-interaction multi-vector, vừa là retriever vừa là reranker | ❌ |
| **MMR** (Maximal Marginal Relevance) | Không phải rerank by relevance — mà chọn top-k **diverse**, tránh duplicate | ❌ |
| **LLM as reranker** | Prompt LLM chấm điểm từng chunk (vd: `Score 1-10`) | ❌ |

---

## 8. LLM Service & Prompt

**File:** [backend/services/llm/service.py](backend/services/llm/service.py), [prompts.py](backend/services/llm/prompts.py)

### Backend: Ollama qua OpenAI SDK

- **Client:** `AsyncOpenAI` (service.py:46-50). Ollama expose REST API tương thích OpenAI → dùng thẳng `openai` Python SDK với `base_url="http://ollama:11434/v1"`.
- **Model default:** `llama3.2` (settings.py:28). On-prem, miễn phí, không gửi data ra ngoài.

### Tham số generation mặc định (settings.py:28-33)

| Param | Giá trị | Ý nghĩa |
|---|---|---|
| `temperature` | `0.0` | Deterministic — factual QA không cần creativity |
| `max_tokens` | `1024` | Trần độ dài câu trả lời |
| `top_p` | `0.95` | Nucleus sampling |
| `presence_penalty` | `0.2` | Giảm lặp chủ đề |
| `frequency_penalty` | `0.3` | Giảm lặp từ |

### System prompt

File [prompts.py](backend/services/llm/prompts.py) dòng 7-44, biến `SYSTEM_PROMPT`. Các rule chính:
- "Answer ONLY based on provided context"
- Synthesize từ nhiều sources
- Ưu tiên bảng cho số liệu
- Citation format `[Source N]`
- Không lặp, không filler
- Kèm few-shot examples (Ex1-3)

### Conversation context

Build message list: `[system] + last 6 turns + [user prompt with context]` (service.py:184: `conversation_history[-6:]`).

User prompt format:
```
CONTEXT:
[Source 1: handbook.pdf, Page 5]
{chunk}
---
[Source 2: policy.pdf, Page 12]
{chunk}

QUESTION: {user question}

ANSWER:
```

### Streaming

`generate_stream()` trả về `AsyncGenerator[str]` — yield từng token. Endpoint chat wrap thành SSE.

### 📚 [KIẾN THỨC SO SÁNH] LLM backend khác

| Backend | Ưu | Nhược | Dự án dùng? |
|---|---|---|---|
| **Ollama** (llama3.2) | On-prem, free, OpenAI-compatible API | Tốc độ phụ thuộc hardware, chất lượng < GPT-4 | ✅ |
| **vLLM** | Throughput cao nhất cho self-host, PagedAttention | Cần GPU, phức tạp deploy hơn Ollama | ❌ |
| **TGI** (Text Generation Inference, HuggingFace) | Production-grade self-host | Nặng | ❌ |
| **OpenAI GPT-4** | Chất lượng top, tool use tốt | Trả phí, data ra ngoài | ❌ |
| **Anthropic Claude** | Long context (200k+), instruction tốt | Trả phí | ❌ |
| **Google Gemini** | Multi-modal, context 1M+ | Trả phí | ❌ |

### 📚 [KIẾN THỨC SO SÁNH] Kỹ thuật giảm hallucination khác

Dự án dùng: prompt ràng buộc + citation + reranker + RAGAS eval.

Ngoài ra có thể thêm: (a) **Self-RAG** — LLM tự đánh giá "có đủ context không, có cần retrieve thêm?"; (b) **CRAG** (Corrective RAG) — đánh giá chất lượng chunk trước khi đưa vào prompt, trigger web search nếu kém; (c) **Guardrails / output validation** — check câu trả lời post-hoc (regex, JSON schema).

---

## 9. Conversation Manager

**File:** [backend/services/conversation.py](backend/services/conversation.py)
**Class:** `ConversationManager`

- **Storage:** `OrderedDict` **in-memory** (conversation.py:28). **Mất khi restart pod** — prod nên thay Redis/Postgres.
- **Giới hạn:**
  - `_max_messages_per_conversation = 50` (dòng 30)
  - `_max_conversations = 1000` (dòng 29) — LRU cleanup.
- **Message shape:** `{role: "user"|"assistant", content, timestamp ISO}`.

---

## 10. Cache (Redis)

**File:** [backend/services/cache.py](backend/services/cache.py)
**Class:** `CacheService`

- **Backend:** Redis, **optional** (`USE_CACHE=false` mặc định).
- **Key:** `MD5(question.lower() + json(filters sorted))` (cache.py:77) — exact-match.
- **TTL:** `CACHE_TTL=3600` giây = 1 giờ (settings.py:88).
- **Fallback:** Redis down → disable gracefully (app vẫn chạy).

### 📚 [KIẾN THỨC SO SÁNH] Semantic cache

**Không có trong dự án.** Exact-match MD5 bị miss khi câu hỏi diễn đạt khác đi chút (`"thời gian nghỉ phép"` vs `"chính sách nghỉ phép"`).

**Semantic cache:** embed query → tìm cached query gần nhất trong vector store nhỏ → hit nếu `cosine > 0.95`. Thư viện: `gptcache`, Redis Vector. Trade-off: thêm 1 lần embed + ANN search, nhưng tỷ lệ hit cao hơn nhiều.

---

## 11. API Layer (FastAPI)

**Files:** [backend/api/main.py](backend/api/main.py), [v1/endpoints/chat.py](backend/api/v1/endpoints/chat.py), [v1/endpoints/documents.py](backend/api/v1/endpoints/documents.py)

### Patterns đã code

1. **Lifespan handler** (`@asynccontextmanager`) — thay cho `on_event` deprecated. Init services ở startup, cleanup ở shutdown.
2. **Service Registry** ([services/__init__.py](backend/services/__init__.py)) — dict global, `get_service(name)` mọi nơi → tránh circular import.
3. **CorrelationIdMiddleware** ([middleware.py](backend/api/middleware.py)):
   - Nhận header `X-Request-ID` nếu client gửi (dòng 28), else generate UUID.
   - Set vào `contextvar` → mọi log trong request đều có `request_id`.
   - Echo lại trong response header để client quote khi report lỗi.
4. **Custom exceptions** ([core/exceptions.py](backend/core/exceptions.py)): `IngestionError`, `EmbeddingError`, `VectorStoreError`, `RetrievalError`, `LLMServiceError`. Endpoint bắt loại cụ thể, map sang HTTP code phù hợp.

### Streaming SSE (chat.py)

Generator yield dạng `data: {json}\n\n` (chat.py:152, 165, 173, 184). 3 loại event:
- `{type: "sources", sources: [...], conversation_id}`
- `{type: "token", content: "..."}`
- `{type: "done"}` (hoặc `{type: "done", error: "..."}` nếu fail mid-stream)

Lý do streaming: UX — user thấy chữ chạy ngay, không đợi 10s cuối câu.

---

## 12. Resilience

**File:** [backend/core/resilience.py](backend/core/resilience.py)

### CircuitBreaker (resilience.py:26-64)

3 trạng thái: `CLOSED → OPEN → HALF_OPEN`.
- Đủ `CIRCUIT_BREAKER_FAILURE_THRESHOLD=5` lỗi liên tiếp → OPEN (fail fast, không gọi downstream).
- Sau `CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30` giây → HALF_OPEN (cho 1 request thử).
- Thành công → về CLOSED.

**Áp dụng cho:** `LLMService`. Nếu Ollama crash, không có circuit breaker thì mọi request sẽ timeout → thread pool cạn → cả app chết.

### Retry với exponential backoff (resilience.py:71-86)

- `RETRY_MAX_ATTEMPTS=2` (settings.py:40)
- `RETRY_INITIAL_DELAY=1.0` giây
- `RETRY_MAX_DELAY=10.0` giây

Tránh thundering herd khi service vừa hồi phục.

---

## 13. Logging có Correlation ID

**File:** [backend/core/logging.py](backend/core/logging.py)

- `request_id_var`: contextvar được `CorrelationIdMiddleware` set mỗi request.
- `RequestIdFilter`: inject `request_id` vào mọi `LogRecord`.
- `JsonFormatter`: mỗi log line là 1 JSON `{timestamp, level, logger, message, request_id, exception?}` — log aggregator (Loki/ELK/Datadog) parse được ngay.
- `setup_logging(level, fmt)` gọi 1 lần ở `main.py`:
  - `LOG_FORMAT=json` cho Docker/prod.
  - `LOG_FORMAT=console` cho local dev (human-readable).

---

## 14. Evaluation

**File:** [evaluation/metrics.py](evaluation/metrics.py)

### IR metrics (deterministic, rẻ)

Functions (dòng 39-68):

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| `hit_at_k` | `1` nếu có ít nhất 1 relevant trong top-k | Có recall được không |
| `recall_at_k` | `\|relevant ∩ retrieved\| / \|relevant\|` | Tỷ lệ relevant lấy về |
| `reciprocal_rank` | `1 / rank_first_relevant` | Doc đúng nằm cao hay thấp |

Tổng hợp trong `compute_retrieval_metrics()` → `RetrievalMetrics` dataclass: `hit_at_k, recall_at_k, mrr, refusal_accuracy, k, num_answerable, num_unanswerable`.

### RAGAS (LLM-judged, chính xác hơn, tốn chi phí hơn)

Class `RAGASEvaluator` dùng Ollama làm judge. Metrics import thực tế (metrics.py:162-167):

| Metric | Đo |
|---|---|
| `Faithfulness` | Answer có grounded trong context không (chống hallucination) |
| `ResponseRelevancy` | Answer có đúng câu hỏi không |
| `LLMContextPrecisionWithoutReference` | Top-k chunk có thực sự relevant và xếp đúng thứ tự không |
| `LLMContextRecallWithoutReference` | Có lấy đủ context cần thiết không |

Input sample: `{user_input, response, retrieved_contexts, reference}`.

### 📚 [KIẾN THỨC SO SÁNH] Các framework đánh giá khác

| Framework | Đặc điểm | Dự án dùng? |
|---|---|---|
| **RAGAS** ✅ | Chuyên RAG, LLM-as-judge, nhiều metrics | ✅ |
| **TruLens** | Tracing + eval, visualize dependency | ❌ |
| **DeepEval** | Pytest-style, tích hợp CI dễ | ❌ |
| **LangSmith** (LangChain) | Managed, trace + eval + dataset | ❌ |
| **ARES** | Auto-generate test set từ corpus | ❌ |

### 📚 [KIẾN THỨC SO SÁNH] Kiến trúc RAG nâng cao

Dự án hiện tại là **Naive RAG** (retrieve → generate, 1 lần). Các biến thể:

| Architecture | Mô tả | Khi nào dùng |
|---|---|---|
| **Naive RAG** ✅ dự án | Retrieve → generate | QA đơn giản |
| **Advanced RAG** | + query rewrite + hybrid + rerank (dự án có) | Prod |
| **Modular RAG** | Component swap dễ dàng (dự án thiết kế theo hướng này) | Flexibility |
| **Agentic RAG** | LLM tự quyết định: retrieve, web search, hay trả lời thẳng | Query đa dạng |
| **Self-RAG** | LLM tự đánh giá "cần retrieve nữa không" | Giảm retrieve không cần thiết |
| **CRAG** (Corrective RAG) | Đánh giá chunk, fallback web search nếu kém | Corpus không đầy đủ |
| **GraphRAG** (Microsoft) | Build knowledge graph từ corpus trước, retrieve theo graph | Câu hỏi suy luận trên entity |
| **HyDE** | Embed câu trả lời giả thay vì query | Query ngắn, doc dài |

---

## 15. Docker Deployment

**Folder:** [docker/](docker/)

### docker-compose services

| Service | Image | Port |
|---|---|---|
| `ollama` | `ollama/ollama` | 11434 |
| `qdrant` | `qdrant/qdrant` | 6333, 6334 |
| `redis` | `redis:7-alpine` | 6379 |
| `rag-backend` | built từ Dockerfile | 8000 |

Backend gọi các service qua DNS internal của docker-compose network: `http://ollama:11434/v1`, `qdrant:6333`, `redis:6379`.

### Dockerfile chính

- Base `python:3.11-slim`
- Install `requirements.txt` → copy backend → expose 8000.
- Healthcheck `/health` mỗi 30s.
- CMD: `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000`.

### Checklist prod

- `LOG_FORMAT=json`, `LOG_LEVEL=INFO`.
- CORS origins giới hạn cụ thể (không `*`).
- Đổi `ConversationManager` sang Redis (in-memory mất khi restart).
- Qdrant snapshot/backup.
- Giới hạn resource cho Ollama (RAM/GPU).
- Secrets qua env/K8s Secret, không commit `.env`.

---

## 16. Câu hỏi phỏng vấn về dự án này

### Về kiến trúc / thiết kế

1. Vẽ workflow RAG 2 phase (ingestion + query) của dự án, gọi tên từng component.
2. Vì sao `DocumentIngestionService` chỉ orchestrate mà không tự làm chunking/table/metadata? → Single Responsibility, dễ test, dễ thay.
3. Vì sao dùng Service Registry (`get_service`) thay vì DI framework? → Đơn giản, đủ dùng cho monolith, tránh circular import.

### Về ingestion

4. Preprocessor có 8 bước — kể tên và mục đích mỗi bước.
5. Vì sao chunking của dự án lại dùng cả **heading** lẫn **embedding similarity**? → Structural giữ section; semantic giữ topic coherence.
6. `SECTION_MAX_CHUNK_TOKENS=600` chọn sao? → BGE-base max 512 tokens, chừa chỗ cho heading prefix + overlap.
7. Vì sao table được chunk riêng, format key:value thay vì CSV? → Giữ quan hệ header↔value, embedding bắt tốt hơn; CSV bị tokenizer vỡ.
8. Vì sao bảng >10 rows phải có thêm row-batch chunk? → Query kiểu "giá trị cột X ở row Y" chỉ cần vài dòng; whole-table chunk quá to nên retrieval dilution.

### Về embedding & retrieval

9. BGE là asymmetric — giải thích. → Query có instruction prefix, passage thì không.
10. Vì sao `normalize_embeddings=True`? → Cosine = dot product, giảm tính toán Qdrant.
11. Hybrid search trong dự án dùng dense + sparse nào? Fusion ra sao? → Qdrant COSINE + BM25Okapi; RRF với α=0.7, k=60.
12. Vì sao RRF chứ không cộng score thẳng? → Dense score và BM25 khác scale; RRF dùng rank nên độc lập scale.
13. Two-stage retrieval của dự án: bi-encoder lấy top 40, cross-encoder rerank còn 5. Vì sao phải 2 stage? → Bi-encoder scale được nhưng kém chính xác; cross-encoder chính xác nhưng O(N). Kết hợp = tốt cả hai.
14. Score threshold = 0.3 áp dụng **trước** rerank. Điều đó có nhược gì? → Có thể loại oan chunk đúng bị bi-encoder chấm thấp mà cross-encoder lại xếp cao. Trade-off: giảm noise đưa vào reranker (đắt).

### Về query rewriter

15. Khi nào rewriter chạy? → `history >= 2 turns`.
16. Nếu rewriter lỗi, pipeline xử lý sao? → Fallback về original query, không block retrieval.

### Về LLM & prompt

17. Vì sao `temperature=0.0`? → Factual QA cần deterministic, reproducible.
18. Cách giảm hallucination trong dự án? → (a) SYSTEM_PROMPT bắt "answer ONLY from context"; (b) citation `[Source N]` buộc trích nguồn; (c) reranker tốt → context chất lượng; (d) RAGAS faithfulness đo post-hoc.
19. Vì sao chỉ giữ 6 turns gần nhất trong context? → Cân bằng context length LLM + cost; 6 turn đủ cho pronoun resolution.

### Về production

20. Circuit breaker bảo vệ gì? Threshold/recovery time bao nhiêu? → Bảo vệ LLMService. 5 lỗi → OPEN 30s.
21. Correlation ID trong dự án hoạt động ra sao? → Middleware đọc/gen `X-Request-ID`, set contextvar, JsonFormatter inject vào mọi log. Debug cross-service dễ.
22. Nếu deploy K8s 3 pod backend, `ConversationManager` in-memory có vấn đề gì? → Session đi sai pod là mất history. Fix: Redis hoặc sticky session.
23. Cache key trong dự án là MD5 exact-match. Hạn chế? → Câu hỏi diễn đạt khác chút là cache miss. (Cao cấp hơn có semantic cache — không code trong dự án.)

### Về evaluation

24. IR metrics vs RAGAS — khác nhau khi nào dùng cái nào? → IR metrics cần ground-truth relevant docs, nhanh và deterministic. RAGAS không cần reference answer, nhưng tốn LLM call.
25. Trong 4 RAGAS metrics của dự án, cái nào quan trọng nhất cho RAG? → Faithfulness (chống hallucination) + Context Precision (nguồn gốc lỗi retrieval).

---

> **Mẹo phỏng vấn:** khi được hỏi "design RAG" hoặc "kể về dự án", mở tài liệu này theo trình tự — workflow → ingestion → embedding → vector store → retrieval → LLM → resilience → eval. Mỗi section chỉ ra file và tham số cụ thể → thể hiện bạn đọc code thật chứ không chỉ biết khái niệm.

---

## Phụ lục: Tổng kết dự án dùng gì / không dùng gì

### ✅ Dùng trong dự án

- **Parser:** pdfplumber, python-docx.
- **Preprocessor:** 8 bước clean custom.
- **Chunker:** Section-aware + Semantic boundary hybrid; tiktoken `cl100k_base`.
- **Table chunking:** Whole-table + row-batch cho bảng > 10 rows.
- **Embedding:** `BAAI/bge-base-en-v1.5`, asymmetric (prefix query), normalized.
- **Vector store:** Qdrant COSINE, HNSW default.
- **Sparse retrieval:** BM25 (`rank_bm25`), in-memory.
- **Fusion:** Reciprocal Rank Fusion (α=0.7, k=60).
- **Reranker:** `BAAI/bge-reranker-base` (CrossEncoder + sigmoid).
- **Query rewrite:** LLM pronoun resolution (chạy khi history ≥ 2 turns).
- **LLM:** Ollama `llama3.2` qua OpenAI SDK, streaming SSE.
- **Conversation:** OrderedDict in-memory.
- **Cache:** Redis MD5 exact-match (optional).
- **Resilience:** CircuitBreaker + retry exponential backoff.
- **Logging:** JSON structured + correlation ID qua contextvar.
- **Evaluation:** IR metrics (`hit@k, recall@k, MRR`) + RAGAS (`Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall`).
- **Deploy:** docker-compose (ollama + qdrant + redis + backend).

### ❌ Không dùng (nhưng biết để so sánh)

- Parser: PyMuPDF, unstructured, LlamaParse, Tesseract OCR.
- Chunking: RecursiveCharacterTextSplitter, Propositions, Parent-child.
- Embedding: bge-m3, e5, MiniLM, OpenAI, Cohere. Fine-tune embedding.
- Vector DB: FAISS, Chroma, Pinecone, Weaviate, Milvus, pgvector, Elasticsearch.
- ANN: IVF, PQ, HNSW-PQ.
- Query transform: HyDE, Multi-query, Step-back, Query decomposition.
- Reranker: bge-reranker-large, Cohere Rerank, ColBERT, MMR, LLM-as-reranker.
- LLM: vLLM, TGI, OpenAI, Claude, Gemini.
- Cache: Semantic cache (gptcache, Redis Vector).
- Giảm hallucination: Self-RAG, CRAG, Guardrails.
- Eval framework: TruLens, DeepEval, LangSmith, ARES.
- Kiến trúc nâng cao: Agentic RAG, GraphRAG.
