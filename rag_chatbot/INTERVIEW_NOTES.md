# RAG Chatbot — Interview Knowledge Base

> Tài liệu tổng hợp toàn bộ kiến thức cần biết về dự án RAG Chatbot này để pass phỏng vấn.
> Nội dung được trích trực tiếp từ source code thực tế trong [rag_chatbot/](rag_chatbot/).

---

## 1. RAG là gì? Tại sao cần RAG?

**RAG (Retrieval-Augmented Generation)** = Truy xuất tri thức từ kho tài liệu rồi nhồi vào prompt cho LLM trả lời.

### Vì sao không hỏi thẳng LLM?
- **Hallucination**: LLM bịa ra thông tin nó không biết.
- **Knowledge cutoff**: LLM chỉ biết tới thời điểm train, không biết tài liệu nội bộ công ty.
- **Không trích nguồn**: Không kiểm chứng được câu trả lời.
- **Fine-tuning quá đắt** và phải retrain mỗi khi data mới.

### RAG giải quyết:
1. Lưu tài liệu công ty trong một **vector DB**.
2. Khi user hỏi → tìm các đoạn (chunks) liên quan nhất.
3. Đưa các đoạn đó kèm câu hỏi vào prompt → LLM tổng hợp câu trả lời.
4. Có **trích nguồn** chính xác (filename + page).

> Đây là pattern chuẩn cho **document Q&A nội bộ doanh nghiệp** — mục tiêu của dự án này.

---

## 2. Workflow tổng thể của dự án

Dự án có **2 pipeline lớn**:

### Pipeline A — INGESTION (offline, chạy 1 lần khi có tài liệu mới)
```
File (PDF/DOCX/TXT/MD)
  → 1. Parser (pdfplumber)         — tách text + tables, lọc header/footer
  → 2. Preprocessor (8 ops)         — sửa Unicode, gộp dòng, dedup, merge cross-page
  → 3. Table Extractor              — convert table → text dạng row-based
  → 4. Section-Aware Chunker        — chia theo heading + semantic boundary
  → 5. Embedding (BGE)              — vector hóa từng chunk (768-d)
  → 6. Vector Store (Qdrant)        — lưu vector + payload, rebuild BM25 index
```

### Pipeline B — RETRIEVAL + GENERATION (online, mỗi câu hỏi)
```
User question
  → 1. Query Rewriter (LLM)         — biến follow-up question thành standalone (nếu cần)
  → 2. Embed query (BGE + instruction prefix)
  → 3a. Vector search (cosine, top-40)
  → 3b. BM25 keyword search (top-40)
  → 4. Reciprocal Rank Fusion (alpha=0.7) — gộp 2 list
  → 5. Score threshold filter
  → 6. Cross-Encoder Reranker (BGE-reranker-base) — re-rank top-12 → top-5
  → 7. Format context [Source N: file, pX]
  → 8. LLM Generate (Ollama llama3.2)
  → 9. Trả về answer + sources (streaming SSE hoặc JSON)
```

Files chính:
- Ingestion: [backend/services/ingestion/pipeline.py](rag_chatbot/backend/services/ingestion/pipeline.py)
- Retrieval: [backend/services/retrieval/pipeline.py](rag_chatbot/backend/services/retrieval/pipeline.py)
- Chat endpoint: [backend/api/v1/endpoints/chat.py](rag_chatbot/backend/api/v1/endpoints/chat.py)

---

## 3. Data — Format và đặc điểm

### Loại file hỗ trợ
`.pdf`, `.docx`, `.txt`, `.md` — khai báo trong [settings.py:60](rag_chatbot/backend/config/settings.py#L60).

### Domain
**Tài liệu nội bộ doanh nghiệp** (Employee Handbook, ADR, postmortem, policy, technical doc...). Eval dataset là `techviet_qa_v2.json` — 50 câu Q&A về một công ty giả tên TechViet.

### Đặc thù dữ liệu enterprise (lý do phải design phức tạp)
1. **PDF có header/footer** lặp đi lặp lại trên mỗi trang → phải lọc.
2. **Bảng số liệu** rất nhiều và quan trọng (revenue, salary band...) → phải extract riêng.
3. **Heading có numbering** kiểu `2.1.3 Code Review` → có hierarchy rõ.
4. **Title page** thường nhồi đủ thứ vào 1 block to (title + subtitle + version + date).
5. **Văn bản bị split giữa trang** → câu cuối trang N nối câu đầu trang N+1.
6. **PDF extraction artifacts**: chữ bị tách (`organizat ion`), ligature (`ﬁle`), hyphen line break.
7. **Multi-hop questions**: câu hỏi cần ráp thông tin từ nhiều nguồn.

→ Toàn bộ design bên dưới sinh ra để xử lý đúng những vấn đề thực tế này.

---

## 4. Document Parsing — `DocumentParser`

File: [backend/services/ingestion/parser.py](rag_chatbot/backend/services/ingestion/parser.py)

### Vì sao chọn `pdfplumber`?
- **PyPDF2** chỉ extract text thô, không có toạ độ → không lọc được header/footer, không tách được bảng.
- **Unstructured / Tika**: nặng, chậm, kết quả khó kiểm soát.
- **pdfplumber** trả lời được **toạ độ x/y, font size** của từng word → dùng để:
  - Lọc header/footer bằng ngưỡng y (`HEADER_Y_THRESHOLD=55`, `FOOTER_Y_THRESHOLD=780` cho A4 595x842).
  - Detect heading bằng **font size** (heading thường lớn hơn body 1.5pt).
  - Tách bảng riêng và **không duplicate text bảng vào phần text** (vì biết bbox của bảng).

### Pipeline parse 1 trang PDF
1. **Crop** bỏ header/footer theo y-coordinate.
2. **find_tables()** → lấy danh sách bbox của bảng.
3. **extract()** từng bảng → `Table` dataclass.
4. **extract_words()** với `extra_attrs=["size"]` → giữ font size.
5. Lọc bỏ word nằm trong bbox bảng (tránh trùng).
6. Group word → line (theo `top` cùng dòng) → paragraph (theo gap dọc).
7. Mỗi paragraph: nếu `font_size > body_size + 1.5` hoặc match regex `^\d+(\.\d+)*\.?\s` → là **heading**.

### Output
`ParsedDocument` chứa `text_blocks: List[TextBlock]`, `tables: List[Table]`, `page_count`. Định nghĩa ở [backend/models/document.py](rag_chatbot/backend/models/document.py).

---

## 5. Document Preprocessing — 8 operations

File: [backend/services/ingestion/preprocessor.py](rag_chatbot/backend/services/ingestion/preprocessor.py)

> **Nguyên tắc**: parser chỉ trung thành với source → preprocessor làm sạch. Mỗi op là **pure function trên `List[TextBlock]`** → test isolation được. Thứ tự op quan trọng (text-level → structural → noise → merging → metadata).

| # | Op | Làm gì | Vì sao cần |
|---|---|---|---|
| 1 | Unicode repair | Sửa mojibake, ligature `ﬁ→fi`, smart quote `'→'`, NFC normalize, bỏ zero-width | PDF cổ thường lưu ligature, smart-quote → embedding model không quen, làm hỏng tokenization |
| 2 | Artifact repair | Rejoin word bị split (`organizat ion → organization`), bỏ hyphen line-break, fix space-before-punct | pdfplumber detect word theo gap → từ dài hay bị tách 1-2 ký tự cuối |
| 3 | Title page splitter | Block đầu page 1 quá dài → tách title + metadata | Title page nhồi tất cả vào 1 heading → chunk to, embedding kém |
| 4 | Heading splitter | Tách `"4. Performance Management TechViet uses..."` → heading + body | Parser nhiều khi gộp heading + đoạn đầu thành 1 paragraph |
| 5 | Frequency dedup | Block xuất hiện trên `>= 40%` số trang → là header/footer còn sót → xoá | Backup cho op crop ở parser (template header/footer khác y) |
| 6 | Cross-page merger | Nối block paragraph cuối trang N với paragraph đầu trang N+1 nếu N kết thúc bằng chữ thường + N+1 bắt đầu bằng chữ thường/conjunction | Câu bị cắt giữa 2 trang sẽ hỏng câu khi chunk |
| 7 | Small-block merger | Block paragraph < 50 ký tự → merge vào block trước cùng section | Tránh chunk vụn vặt |
| 8 | Section rebuild | Re-assign `section` field cho mỗi block dựa trên heading gần nhất ở trên | Sau khi split/merge, metadata section có thể sai |

**Idempotent**: chạy preprocess 2 lần ra cùng output → an toàn để re-run.

---

## 6. Chunking — Section-Aware + Semantic Boundary

File: [backend/services/ingestion/chunker.py](rag_chatbot/backend/services/ingestion/chunker.py)

### Tại sao chunking là khâu QUAN TRỌNG NHẤT của RAG?
- Chunk quá to → embedding "loãng", LLM bị nhiễu, hết context.
- Chunk quá nhỏ → mất ngữ cảnh, không trả lời được câu hỏi cần nhiều câu.
- Chunk cắt giữa câu/giữa ý → embedding sai topic, retrieval miss.

### Các strategy chunking phổ biến và chọn cái nào?
| Strategy | Mô tả | Vấn đề |
|---|---|---|
| Fixed-size (vd 512 tokens) | Cắt cứng theo token | Cắt giữa câu, mất topic boundary |
| Recursive character split (LangChain default) | Split theo `\n\n → \n → . → space` | Vẫn không hiểu heading, không hiểu topic shift |
| Sentence-based | Mỗi câu 1 chunk | Quá nhỏ, mất context |
| **Section-aware + semantic** (CHỌN) | Tôn trọng heading + dùng embedding similarity tìm topic boundary | Phức tạp hơn nhưng chunk align với topic thật |

### Pipeline chunker (analyse → plan → build)
- **Mọi section đều qua cùng pipeline**, không có "short path" / "long path".
- **Analyse**:
  - Flatten paragraph → câu (split bằng regex `(?<=[.!?])\s+`, có whitelist viết tắt `Dr.`, `Mr.`, `e.g.`...).
  - Tính `boundary_score[i] = 1 - cosine_similarity(embed(sent_i), embed(sent_i+1))` — score cao = topic shift.
- **Plan**:
  - Walk câu, accumulate token cho đến khi vượt `max_chunk_tokens=600`.
  - Khi cần split, xét **3 candidate gần đây nhất** (look-back=3): chọn điểm có score cao nhất giữa (paragraph boundary + semantic score), miễn `>= 0.15`.
  - Paragraph boundary cộng `+1.0` (rất mạnh) → ưu tiên cắt theo cấu trúc hơn semantic.
- **Build**:
  - Mỗi chunk có **prefix `[Heading]\n\n`** trước nội dung.
  - Có overlap **2 câu cuối** sang chunk sau → tránh mất ngữ cảnh.
  - Metadata: `breadcrumb` (`1. Intro > 1.1 Scope`), `chunk_part`, `total_parts`, `page_numbers`.
- **Post-pass**: chunk < `min_chunk_tokens=80` → gộp vào chunk trước.

### Vì sao prefix `[Heading]` quan trọng?
Embedding model "nhìn thấy" heading sẽ encode chunk gần với những query có cùng topic. Đây là một trick rất rẻ nhưng tăng recall đáng kể (đặc biệt với BGE).

### Vì sao dùng `tiktoken cl100k_base` để đếm token?
Đó là tokenizer của GPT-4 → estimate gần đúng cho cả LLM mục tiêu. Có fallback `len(text)//4` nếu thiếu lib.

### Tham số mặc định ([settings.py:63-67](rag_chatbot/backend/config/settings.py#L63)):
```
SECTION_MAX_CHUNK_TOKENS = 600
SECTION_MIN_CHUNK_TOKENS = 80
SECTION_OVERLAP_SENTENCES = 2
SECTION_SEMANTIC_LOOK_BACK = 3
SECTION_SEMANTIC_MIN_SCORE = 0.15
```
**600 tokens**: vừa với context window của embedding model BGE (max 512 token nhưng tính cả prefix), nhỏ đủ để 5 chunk gộp lại còn fit context của LLM.

---

## 7. Table Extraction

File: [backend/services/ingestion/table_extractor.py](rag_chatbot/backend/services/ingestion/table_extractor.py)

### Vì sao phải xử lý table riêng?
- Table dạng CSV/Markdown → embedding hiểu kém vì cấu trúc 2D bị flatten.
- LLM trả lời số liệu sai nếu cell value bị tách khỏi header.

### Format chọn: **Row-based text**
```
Table: Salary_Bands | Section: 5. Compensation | Page 12

Row 1:
  Level: L3
  Min: 18000
  Max: 25000
Row 2:
  Level: L4
  ...
```
→ Mỗi row tự thuyết minh bằng `header: value`. Embedding model encode được semantics của từng cell.

### Bảng lớn (> 10 rows)
Tách thêm **row-batch chunks** (5 row/batch) để truy xuất chính xác hơn 1 dòng cụ thể (xem [pipeline.py:163-204](rag_chatbot/backend/services/ingestion/pipeline.py#L163)).

---

## 8. Embedding — `BAAI/bge-base-en-v1.5`

File: [backend/services/embedding/service.py](rag_chatbot/backend/services/embedding/service.py)

### Vì sao chọn BGE-base-en-v1.5?
- **MTEB benchmark top** trong class size ~110M parameters → cân bằng accuracy/cost.
- **Free, open-source**, chạy local trên CPU OK (768-d, batch 32 ~vài giây/100 chunks).
- **Hỗ trợ asymmetric encoding** (query có instruction prefix khác passage).
- **Đã train trên data Q&A** → tốt cho retrieval task.

### So sánh nhanh
| Model | Dim | Size | Note |
|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 | API | Tốn $, không local |
| `bge-base-en-v1.5` | 768 | 110M | **Chọn** |
| `bge-large-en-v1.5` | 1024 | 335M | Tốt hơn nhưng chậm hơn ~3x |
| `all-MiniLM-L6-v2` | 384 | 22M | Nhanh, accuracy thấp |

### Asymmetric encoding (key insight)
- **Query**: prefix `"Represent this sentence for searching relevant passages: "` rồi mới encode.
- **Passage**: encode raw (không prefix). Lý do: chunk đã có `[Heading]` prefix tự cho topic signal.

→ Model học được rằng "cái text có prefix kia là intent đi tìm" → vector space tách rõ "looking-for" vs "content". Recall cao hơn ~5-10% so với dùng cùng 1 prefix.

### Normalize embedding
`normalize_embeddings=True` → vector unit length → cosine similarity = dot product → Qdrant store dùng `Distance.COSINE`.

### Async pattern
Toàn bộ encode chạy qua `asyncio.to_thread()` → không block FastAPI event loop. Batch size 32 cân bằng RAM và throughput trên CPU.

---

## 9. Vector Store — Qdrant + cached BM25

File: [backend/services/vectorstore/qdrant.py](rag_chatbot/backend/services/vectorstore/qdrant.py)

### Vì sao chọn Qdrant?
| DB | Lý do (không) chọn |
|---|---|
| **Qdrant** ✅ | Production-grade, Rust → fast, hỗ trợ payload filter rich, self-host dễ (docker run), Python client tốt |
| Pinecone | Cloud-only, $$ |
| Weaviate | Heavy hơn, schema phức tạp |
| Chroma | Tốt cho dev nhưng kém scale |
| FAISS | Library chứ không phải DB, không có filter, không có upsert |
| pgvector | Tốt nếu đã có Postgres, nhưng performance kém Qdrant ở scale |

### Schema
Mỗi point = 1 chunk, có:
- `id`: UUID
- `vector`: 768-d embedding
- `payload`: metadata + `content` (text gốc)

`Distance.COSINE`, collection tự create lần đầu.

### BM25 keyword index
- **In-memory**, build 1 lần, **rebuild khi add/delete document**.
- Lý do cache: scroll toàn bộ Qdrant + tokenize trên mỗi query rất chậm (O(N)).
- Lib: `rank_bm25` (BM25Okapi).
- Tokenize đơn giản: `doc.lower().split()` — đủ cho corpus enterprise tiếng Anh.

### Pattern async
Tất cả call sync Qdrant đều gói trong `asyncio.to_thread()`.

---

## 10. Hybrid Search + Reciprocal Rank Fusion

### Hybrid = Vector + Keyword
- **Vector search** giỏi semantic similarity ("tăng lương" ~ "salary increase") nhưng thua khi query có **từ khoá hiếm** (mã sản phẩm, tên người, số điện thoại).
- **BM25** giỏi exact-match keyword nhưng không hiểu synonym.
- **Hybrid** = best of both worlds, khá là chuẩn industry hiện nay.

### Reciprocal Rank Fusion (RRF)
Code: [`_reciprocal_rank_fusion`](rag_chatbot/backend/services/retrieval/pipeline.py#L125)

```
RRF_score(doc) = alpha / (k + rank_vector) + (1-alpha) / (k + rank_keyword)
```

- `k=60` (constant industry chuẩn của paper RRF gốc — Cormack 2009).
- `alpha=0.7` → ưu tiên vector hơn (đặt trong [settings.py:76](rag_chatbot/backend/config/settings.py#L76)).
- Doc xuất hiện ở **cả 2 list** → score cao gấp đôi → boost mạnh.

**Vì sao RRF tốt hơn weighted-sum?**
- Không phụ thuộc vào scale của 2 score (BM25 không bounded, cosine ∈ [-1,1]).
- Chỉ dùng **rank**, không dùng score thô → robust.
- Đơn giản, deterministic, không cần normalize.

---

## 11. Query Rewriting (cho multi-turn)

File: [backend/services/retrieval/query_rewriter.py](rag_chatbot/backend/services/retrieval/query_rewriter.py)

### Vấn đề
User chat:
- Q1: *"Mức lương L4 là bao nhiêu?"*
- Q2: *"Còn L5 thì sao?"*

Nếu embed thẳng Q2 thì retrieval miss vì "Còn L5 thì sao" không có context "lương".

### Giải pháp
Trước khi retrieve, gọi **LLM nhẹ** với prompt:
> "Rewrite the user's follow-up question into one clear self-contained English question. Resolve all pronouns. Output ONLY the rewritten question."

Q2 → *"What is the salary range for level L5?"*

### Tại sao lại cần điều kiện `QUERY_REWRITE_MIN_TURNS=2`?
- Lượt đầu tiên không có context cũ → không cần rewrite, đỡ 1 LLM call (latency).
- Chỉ rewrite từ lượt 2 trở đi.

### Latency tối ưu
- `temperature=0`, `max_tokens=120` → chỉ vài trăm ms.
- Try/except → fail thì fallback dùng query gốc, không block retrieval.

---

## 12. Reranking — Cross-encoder

File: [backend/services/retrieval/reranker.py](rag_chatbot/backend/services/retrieval/reranker.py)

### Bi-encoder vs Cross-encoder
| | Bi-encoder (BGE) | Cross-encoder |
|---|---|---|
| Cách | Encode query và passage **độc lập** → cosine | Encode `(query, passage)` **cùng nhau** → 1 score |
| Tốc độ | Rất nhanh (precompute embedding) | Chậm (mỗi pair là 1 forward pass) |
| Accuracy | Tốt | **Tốt hơn rõ rệt** |
| Dùng ở đâu | Bước retrieval đầu (top-N lớn) | Bước rerank (top-K nhỏ) |

### Pipeline 2-stage là chuẩn
1. Bi-encoder retrieve **top 40** (rộng nhưng nhanh).
2. Cross-encoder rerank xuống **top 5** (chính xác).

→ Vừa scale tốt vừa accuracy cao.

### Model: `BAAI/bge-reranker-base`
- Cross-encoder của BGE family, 280M params, max_length 512.
- Output là logit → áp **sigmoid** để normalize về [0,1] (làm score đẹp cho UI).
- Code: rerank `top_k * 2.5 = 12` candidate xuống `top_k = 5`.

### Settings
```
TOP_K_RETRIEVAL = 40   # bi-encoder retrieve rộng
TOP_K_RERANK = 5       # cross-encoder trả về cuối cùng
RETRIEVAL_SCORE_THRESHOLD = 0.3  # bỏ chunk score quá thấp sau RRF
```

---

## 13. LLM Service — Ollama + Resilience

File: [backend/services/llm/service.py](rag_chatbot/backend/services/llm/service.py)

### Vì sao chọn Ollama?
- **Self-host**, không cần API key, không leak data nội bộ.
- **OpenAI-compatible API** → đổi sang OpenAI/Anthropic chỉ cần đổi base_url.
- **GPU/CPU auto-detect**, hỗ trợ nhiều model (llama3.2, mistral, qwen...).
- Default model: **`llama3.2`** — small + tốt + free.

### Generation params
```python
temperature = 0.0          # Q&A cần deterministic
max_tokens = 1024
top_p = 0.95
frequency_penalty = 0.3    # giảm lặp lại
presence_penalty = 0.2     # khuyến khích đề cập nhiều khía cạnh
```

### Resilience patterns
File: [backend/core/resilience.py](rag_chatbot/backend/core/resilience.py)

#### Circuit Breaker
3 trạng thái: `CLOSED → OPEN → HALF_OPEN → CLOSED`
- Đếm failure: vượt `failure_threshold=5` → OPEN.
- OPEN: từ chối request `recovery_timeout=30s`.
- Sau timeout → HALF_OPEN: cho phép vài request thử, success 3 lần → CLOSED.
- **Tránh cascading failure** khi Ollama down.

#### Retry with Exponential Backoff
- `max_attempts=2`, `initial_delay=1s`, `base=2.0`, `max=10s`.
- Retry lần n delay = `min(1 * 2^n, 10)` giây.

#### Concurrency limit
`MAX_CONCURRENT_REQUESTS=50` — counter `_active_requests` từ chối thêm khi vượt.

### Streaming response
Endpoint chat hỗ trợ **SSE** (Server-Sent Events): `text/event-stream`. Mỗi token LLM sinh ra → push qua client luôn → UX latency thấp.

---

## 14. System Prompt — Trick để control LLM

File: [backend/services/llm/prompts.py](rag_chatbot/backend/services/llm/prompts.py)

Key rules được nhồi vào system prompt:
1. **"Answer ONLY based on the provided context"** → chống hallucination.
2. **"Tables first for numerical data"** → ưu tiên số liệu từ table chunks.
3. **"Cite [Source N: filename, pX]"** → trace back được nguồn.
4. **"Include ALL specific details: names, dates, numbers, root causes"** → tránh trả lời chung chung.
5. **Format**: Verdict trước → bullet points evidence → KHÔNG kết luận sáo rỗng.
6. Cho **3 example** (simple / root-cause / missing) → few-shot learning.

→ Đây là **prompt engineering kỹ lưỡng** để LLM nhỏ (llama3.2) cũng cho output chất lượng.

---

## 15. API Layer — FastAPI

File: [backend/api/main.py](rag_chatbot/backend/api/main.py)

### Endpoints chính
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/chat` | Hỏi đáp (hỗ trợ stream + conversation history) |
| GET | `/api/v1/chat/conversations/{id}` | Lấy lịch sử |
| DELETE | `/api/v1/chat/conversations/{id}` | Xoá conversation |
| POST | `/api/v1/documents` | Upload file mới |
| GET | `/api/v1/documents` | List docs |
| GET | `/health` | K8s health check |
| GET | `/api/v1/docs` | Swagger UI |

### Lifespan pattern
`@asynccontextmanager` → khi app start: `initialize_services()` (load BGE, connect Qdrant, init reranker, init Ollama client). Khi shutdown: graceful drain active requests.

### CORS
Mở `*` (cho dev), production cần bóp lại.

---

## 16. Conversation Management

File: [backend/services/conversation.py](rag_chatbot/backend/services/conversation.py)

- **In-memory** OrderedDict (class-level → shared across instances).
- Limit **1000 conversations**, **50 messages/conversation**.
- LRU eviction khi vượt limit.
- Production note: nên đổi sang Redis/Postgres cho persistence (đã ghi trong docstring).

---

## 17. Caching Layer

File: [backend/services/cache.py](rag_chatbot/backend/services/cache.py)

- **Redis** cache cho query response.
- Key = hash của `(question, filters)`.
- TTL = 3600s (1h).
- Optional (`USE_CACHE=false` mặc định trong dev).
- Lý do: cùng câu hỏi → cùng answer → đỡ trả lại pipeline retrieval + LLM (latency 2-10s → vài ms).

---

## 18. Evaluation — Cách đo RAG có tốt không

File: [evaluation/run_evaluation.py](rag_chatbot/evaluation/run_evaluation.py), [evaluation/metrics.py](rag_chatbot/evaluation/metrics.py)

### Tại sao phải eval?
RAG có **rất nhiều hyperparameter** (chunk size, top_k, alpha RRF, reranker on/off...). Không có metrics → tune mù.

### Dataset
`evaluation/datasets/techviet_qa_v2.json` — **50 test case** chuẩn:
```json
{
  "id": "...",
  "category": "factual_easy | multi_hop | unanswerable | ...",
  "question": "...",
  "expected_answer": "...",
  "source_documents": ["file1.pdf", "file2.pdf"],
  "has_answer": true
}
```

### Metrics chia 2 nhóm

#### A. Retrieval metrics (deterministic, RẺ)
| Metric | Công thức | Ý nghĩa |
|---|---|---|
| **Hit@k** | `1` nếu top-k có ít nhất 1 doc đúng | Có lấy được không? |
| **Recall@k** | `\|retrieved ∩ relevant\| / \|relevant\|` | Lấy được bao nhiêu % doc đúng? |
| **MRR** | `1 / rank của doc đúng đầu tiên` | Doc đúng có lên TOP không? |
| **Refusal Accuracy** | Câu unanswerable có retrieve về 0 chunk không? | Có biết "không biết" không? |

→ Đo **chất lượng retriever**, không cần LLM judge → chạy nhanh, dùng để A/B test khi tune chunker/embedding/RRF.

#### B. Generation metrics — RAGAS (LLM-judged, CHẬM)
| Metric | Đo gì |
|---|---|
| **Faithfulness** | Câu trả lời có grounded trong context không? (chống hallucination) |
| **Answer Relevancy** | Trả lời có đúng câu hỏi không? |
| **Context Precision** | Chunks retrieved có relevant và đúng thứ tự không? |
| **Context Recall** | Đã retrieve đủ context cần thiết chưa? |

→ Dùng LLM (Ollama llama3.2) làm **judge** → score 0-1.
→ Chạy chậm (~10-20 phút cho 50 case) nhưng đo **toàn bộ pipeline end-to-end**.

### Ngưỡng tham khảo
| Metric | Acceptable | Good |
|---|---|---|
| Hit@6 | ≥ 0.80 | ≥ 0.90 |
| Recall@6 | ≥ 0.70 | ≥ 0.85 |
| MRR | ≥ 0.60 | ≥ 0.75 |
| Faithfulness | ≥ 0.75 | ≥ 0.85 |
| Answer Relevancy | ≥ 0.75 | ≥ 0.85 |

### Cách chạy
```bash
make evaluate-quick   # 5 cases, không RAGAS, smoke test
make evaluate         # full 50 cases + RAGAS
```

---

## 19. Tech stack tóm tắt

| Layer | Technology | Lý do |
|---|---|---|
| **Backend framework** | FastAPI | Async, fast, auto OpenAPI docs |
| **PDF parsing** | pdfplumber | Có toạ độ + font size |
| **Token counting** | tiktoken (cl100k_base) | Match GPT family |
| **Embedding** | sentence-transformers + BGE-base-en-v1.5 | Top MTEB, free, local |
| **Vector DB** | Qdrant | Production, filter rich, self-host |
| **Keyword search** | rank_bm25 (in-memory) | BM25, simple, đủ dùng |
| **Reranker** | sentence-transformers CrossEncoder + bge-reranker-base | Cross-encoder accuracy cao |
| **LLM server** | Ollama | Self-host, OpenAI-compat |
| **LLM model** | llama3.2 | Free, đủ tốt cho Q&A có context |
| **LLM client** | openai (async) | OpenAI-compatible với Ollama |
| **Cache** | Redis (optional) | Query response cache |
| **Evaluation** | RAGAS + custom IR metrics | Industry standard |
| **Container** | Docker Compose | Orchestrate Qdrant + Redis + backend |
| **Settings** | pydantic-settings | Type-safe env vars |

---

## 20. Câu hỏi phỏng vấn dễ bị hỏi & cách trả lời

### Q: Tại sao chunk size 600 token chứ không phải 1000 hay 200?
- 600 vừa với context window của BGE (~512 + prefix). Lớn hơn sẽ bị truncate, embedding mất một phần text.
- Đủ nhỏ để top-5 chunk gộp lại fit context của LLM nhỏ (llama3.2 ~8K).
- Đủ lớn để giữ vài câu liền mạch về 1 topic. Đã test bằng eval dataset.

### Q: Vì sao dùng cả vector search + BM25? Vector search không đủ à?
- Vector search miss khi query có **identifier hiếm** (mã ADR-042, tên file `runbook-v3.2`, version number...) vì model không gặp những token này khi train.
- BM25 lookup keyword chính xác.
- RRF gộp lại → tăng Hit@k 5-15% trên eval.

### Q: Reranker có thực sự cần không, đắt vậy?
- Bi-encoder cosine ~70% Hit@5, cross-encoder rerank đẩy lên 85-90%.
- Cost: chỉ rerank 12 candidates → CPU cũng chỉ ~200ms.
- Trade-off rất đáng.

### Q: Tại sao không dùng LangChain cho rapid prototype?
- LangChain abstract quá nhiều → khó debug, khó tune.
- Phiên bản API thay đổi liên tục.
- Self-implement giúp **kiểm soát từng bước** + **biết đang chạy gì** → quan trọng cho production.
- Dùng `langchain_community.ChatOllama` chỉ ở RAGAS vì RAGAS yêu cầu wrapper chuẩn của LangChain.

### Q: Làm sao chống hallucination?
1. **System prompt** ép "answer ONLY from context" + few-shot examples.
2. **Faithfulness metric** trong eval phát hiện hallucination.
3. **Score threshold** retrieval (`0.3`) → không có chunk đủ tốt thì trả "Not found".
4. **Citations** trong output → user verify được.
5. `temperature=0` → deterministic.

### Q: Multi-turn conversation hoạt động thế nào?
1. ConversationManager lưu lịch sử in-memory.
2. Mỗi request đính kèm `conversation_id`.
3. **Query Rewriter** biến follow-up thành standalone trước khi retrieve (resolve pronoun).
4. LLM được đưa **6 turn cuối** (assistant message bị cắt còn 300 char để tiết kiệm context).

### Q: Scale lên 100K documents thì sao?
- Qdrant scale tốt (HNSW index) → vector search vẫn nhanh.
- BM25 in-memory sẽ tốn RAM → đổi sang **Elasticsearch / Tantivy** hoặc dùng `qdrant sparse vectors` (BM25 trong Qdrant).
- Embedding cost lớn → batch + GPU.
- Reranker thành bottleneck → có thể dùng **distilled model** nhỏ hơn.
- Cache layer (Redis) thành quan trọng.

### Q: Tại sao chia 8 ops preprocessing? Gom chung 1 hàm không được sao?
- **Mỗi op test isolated được** → unit test rõ ràng (xem [test_preprocessing.py](rag_chatbot/tests/unit/test_preprocessing.py)).
- **Idempotent** → an toàn re-run.
- **Order matters**: text-level fix trước → structural fix sau → noise removal → merging → metadata rebuild. Đảo thứ tự sẽ hỏng (vd dedup trước khi unicode repair sẽ bỏ sót).
- **Easier to add/remove** ops khi gặp pattern data mới.

### Q: Vì sao prefix `[Heading]` cho passage mà không cho query?
- BGE asymmetric: query có **instruction prefix riêng** (`Represent this sentence...`) — đó là cách model học để biết "đây là intent search".
- Passage không cần instruction nhưng được hưởng `[Heading]` để embedding "biết" topic của chunk → tăng relevance giữa query vs passage cùng topic.
- Test trên dataset thấy recall tốt hơn khi có prefix heading.

### Q: Tại sao Cosine distance mà không Euclidean / Dot product?
- Embedding đã `normalize_embeddings=True` → unit vector → **cosine = dot product**.
- Cosine không bị ảnh hưởng bởi vector magnitude → chỉ đo direction (semantic).
- Industry standard cho text embedding.

### Q: BM25 hoạt động thế nào (very high level)?
- Score = `IDF(term) × TF(term, doc) × length_norm`.
- IDF: từ hiếm → score cao.
- TF: xuất hiện nhiều trong doc → score cao (nhưng saturate).
- Length norm: doc dài bị penalty (chống cheat).
- Tuning: `k1=1.5`, `b=0.75` (default rank_bm25).

### Q: Cross-encoder hoạt động khác bi-encoder thế nào?
- **Bi-encoder**: `embed(query)` và `embed(passage)` riêng → cosine. Có thể precompute passage → fast retrieval.
- **Cross-encoder**: input là `[CLS] query [SEP] passage [SEP]` → BERT tự attention giữa 2 → output 1 score relevance. Không precompute được → chậm. Nhưng accuracy cao hơn nhiều vì attention thấy được word-level interaction.

### Q: Tại sao RRF với `k=60`?
- Constant gốc trong paper Cormack et al. 2009. Điều chỉnh sensitivity → k lớn → top doc không quá dominant.
- Experimentally robust, hầu như không cần tune.

### Q: Có support tiếng Việt không?
- Hiện tại embedding `bge-base-en-v1.5` chỉ tốt cho English. Nếu cần VN → đổi sang `bge-m3` (multilingual) hoặc `paraphrase-multilingual-mpnet-base-v2`.
- Eval refusal phrases có support cả VN: `"không tìm thấy", "không có thông tin"` ([run_evaluation.py:118](rag_chatbot/evaluation/run_evaluation.py#L118)).

### Q: Production-ready check list?
- [x] Circuit breaker + retry
- [x] Concurrency limit
- [x] Graceful shutdown
- [x] Health endpoints
- [x] Structured logging
- [x] Async I/O không block event loop
- [x] Docker compose deploy
- [x] Eval pipeline
- [ ] Conversation persistence (đang in-memory)
- [ ] Auth/RBAC
- [ ] Rate limiting per user
- [ ] Distributed tracing (OTel)

---

## 21. Sơ đồ flow tóm tắt cho phỏng vấn

```
                          INGESTION (offline)
  ┌─────────┐     ┌──────────┐    ┌────────────┐    ┌────────┐
  │ PDF/    │ ──▶ │ Parser   │──▶ │ Pre-       │──▶ │ Section│
  │ DOCX    │     │ pdfplum  │    │ processor  │    │ Chunker│
  └─────────┘     │  +crop   │    │  (8 ops)   │    │ + sem  │
                  │  hdr/ftr │    │            │    │ bound  │
                  └──────────┘    └────────────┘    └────┬───┘
                                                         │
                  ┌──────────┐    ┌────────────┐         ▼
                  │ Qdrant   │◀── │  BGE       │◀── chunks
                  │ (cosine) │    │ embed-doc  │
                  └────┬─────┘    └────────────┘
                       │ rebuild
                       ▼
                  ┌──────────┐
                  │ BM25 idx │
                  │ in-mem   │
                  └──────────┘

                          RETRIEVAL + GEN (online)
  ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌────────┐  ┌────┐
  │ user │─▶│  Query   │─▶│ BGE      │─▶│Qdrant│─▶│  RRF   │─▶│Re- │
  │ Q    │  │ Rewriter │  │embed-q   │  │top40 │  │ fusion │  │rank│
  └──────┘  │ (LLM)    │  │+instr    │  └──────┘  │alpha=  │  │BGE │
            └──────────┘  └──────────┘  ┌──────┐  │  0.7   │  │xenc│
                                        │ BM25 │─▶│        │  │top5│
                                        │top40 │  └────────┘  └─┬──┘
                                        └──────┘                │
                                                                ▼
                                  ┌─────────────┐         ┌──────────┐
                                  │ format ctx  │◀────────│ filter   │
                                  │ [Source N]  │         │ thresh   │
                                  └──────┬──────┘         └──────────┘
                                         ▼
                                  ┌─────────────┐
                                  │ LLM         │
                                  │ Ollama      │
                                  │ llama3.2    │
                                  │ (stream SSE)│
                                  └──────┬──────┘
                                         ▼
                                    answer + sources
```

---

## 22. "Câu thần chú" 1 dòng để giới thiệu dự án

> *"Tôi build một RAG chatbot Q&A tài liệu nội bộ doanh nghiệp: pipeline ingestion 6 bước với pdfplumber + section-aware chunking dùng semantic boundary detection; retrieval 2-stage gồm BGE bi-encoder + BM25 RRF fusion + cross-encoder reranking; LLM Ollama llama3.2 với prompt engineering chống hallucination; tất cả async FastAPI có circuit breaker và streaming SSE; eval bằng Hit@k/MRR + RAGAS faithfulness."*

Nói được câu này = đã chứng minh bạn hiểu **kiến trúc, design choice, và industry best practice**. Phỏng vấn viên sẽ hỏi sâu vào 1-2 chỗ → dùng các mục 4-18 ở trên để trả lời.
