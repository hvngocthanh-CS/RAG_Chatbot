# RAG Chatbot - Architecture & NLP Techniques

## Overview

RAG (Retrieval-Augmented Generation) là kiến trúc kết hợp:
- **Retrieval**: Tìm kiếm thông tin liên quan từ knowledge base
- **Generation**: Sử dụng LLM để sinh câu trả lời dựa trên context được retrieve

---

## 1. Document Ingestion Pipeline

### 1.1 Document Parsing (`document_parser.py`)
- **Input**: PDF, DOCX, TXT files
- **Output**: Raw text + metadata (filename, page numbers)
- **Libraries**: `pdfplumber`, `python-docx`

### 1.2 Table Extraction (`table_extractor.py`)
- Trích xuất bảng từ PDF riêng biệt
- Chuyển table thành text format (row-based)
- Metadata: `is_table=True`, `table_index`, `page_number`

### 1.3 Text Chunking (`chunker.py`)

**⚠️ QUAN TRỌNG: tiktoken CHỈ dùng để ĐẾM tokens, KHÔNG phải để embed!**

**Token Counter**: `tiktoken` với encoding `cl100k_base` (OpenAI GPT-4 tokenizer)

```python
# Trong chunker.py line 48-51
def count_tokens(self, text: str) -> int:
    if self.tokenizer:
        return len(self.tokenizer.encode(text))  # Chỉ ĐẾM, không lưu tokens
```

**Vai trò của tiktoken**:
- ✅ **Đo lường độ dài**: Biết 1 đoạn text có bao nhiêu tokens
- ✅ **Quyết định cắt chunk**: Khi nào đủ 600 tokens để tạo chunk mới
- ✅ **Prevent overflow**: Đảm bảo không vượt quá LLM context window
- ❌ **KHÔNG dùng để tạo embeddings** (BGE model làm việc đó)

**Chunking Strategy**:
- **Token-based chunking**: Đếm token thay vì characters để đảm bảo consistency với LLM
- **Chunk size**: 600 tokens (configurable trong `settings.py`)
- **Chunk overlap**: 100 tokens (để giữ context liên tục giữa các chunks)

**⚙️ Chunk Size được config ở đâu?**
```python
# backend/config/settings.py line 118
CHUNK_SIZE: int = 600  # tokens per chunk
CHUNK_OVERLAP: int = 100  # tokens overlap

# backend/services/ingestion.py line 36
self.chunker = TextChunker(
    chunk_size=settings.CHUNK_SIZE,  # Đọc từ .env hoặc default 600
    chunk_overlap=settings.CHUNK_OVERLAP
)
```

**Tại sao chọn 600 tokens?**

| Chunk Size | Pros ✅ | Cons ❌ |
|------------|---------|---------|
| **Small (256-300)** | - Precision cao<br>- Ít noise<br>- Fast embedding | - Thiếu context<br>- Cắt ngang sections<br>- Nhiều chunks hơn |
| **Medium (600)** ⭐ | - Balance context vs precision<br>- Giữ được sections nguyên vẹn<br>- Đủ context cho LLM | - Có thể có một ít noise |
| **Large (1024+)** | - Context đầy đủ<br>- Ít chunks | - Nhiều noise<br>- Retrieval kém chính xác<br>- Expensive embedding |

**Current choice: 600 tokens** vì:
- RTX 3060 6GB → vLLM max-model-len=1024
- Context window cho answer: 1024 = 600 (chunk) + ~400 (prompt + answer)
- Giữ được "Ghi chú đặc biệt" sections không bị cắt

**Semantic Boundary Detection**:
```python
# Trong chunker.py - Ưu tiên cắt tại:
1. Title/Heading (block_type == "title" hoặc "heading")
2. Paragraph breaks (\n\n)
3. Sentence endings khi đủ token count
```

**Flow trong `chunk_text()` method**:
```
1. Đọc từng text_block từ document parser
2. Dùng tiktoken.count_tokens() để đo block
3. Nếu current_chunk + new_block > 600 tokens → Lưu chunk cũ, tạo chunk mới
4. Nếu chưa đủ → Nối tiếp vào current_chunk
5. Overlap: 100 tokens cuối của chunk cũ được giữ lại ở đầu chunk mới
```

### 1.4 Embedding Generation (`embeddings.py`)

**⭐ ĐÂY MỚI LÀ COMPONENT THỰC SỰ TOKENIZE & EMBED!**

**Model**: `BAAI/bge-base-en-v1.5` (768 dimensions)

```python
# Trong embeddings.py
self.model = SentenceTransformer(model_name)  # Load BGE model
```

**BGE Model tự động làm**:
1. **Tokenize** (dùng BERT WordPiece tokenizer - KHÁC tiktoken!)
2. **Encode** thành vectors 768 chiều
3. **Normalize** vectors (L2 normalization)

**Input/Output**:
```python
# Input: List[str] - Danh sách chunks (text)
chunks = ["This is chunk 1", "This is chunk 2"]

# Output: numpy array shape (n, 768) - Dense vectors
embeddings = self.model.encode(chunks)  # [[0.12, -0.43, ...], [...]]
```

**BGE Query Prefix** (quan trọng!):
```python
# Khi encode query (lúc search):
query_with_prefix = "Represent this sentence for searching relevant passages: What is PTO policy?"

# Khi encode documents (lúc ingestion): KHÔNG cần prefix
doc_text = "PTO policy allows 15 days per year"
```

**Tại sao cần prefix?**
- BGE models được train với **asymmetric retrieval**
- Query và document ở 2 không gian semantic khác nhau
- Prefix giúp model "biết" đây là query → map vào document space
- Document chunks KHÔNG dùng prefix → giữ nguyên semantic

**So sánh tiktoken vs BGE tokenizer**:

| Aspect | tiktoken | BGE (BERT tokenizer) |
|--------|----------|---------------------|
| **Mục đích** | Đếm tokens cho chunking | Tokenize để tạo embeddings |
| **Vocab** | GPT-4 BPE tokens (~100k) | BERT WordPiece (~30k) |
| **Output** | Token count (int) | Vector embeddings (768 dims) |
| **Khi dùng** | Lúc chunking documents | Lúc embed chunks & queries |

### 1.5 Vector Storage (`vector_store.py`)

**Database**: ChromaDB
- **Metric**: Cosine similarity
- **Persistence**: Local SQLite

```python
self.collection = self.client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
```

---

## 2. Query Processing Pipeline

### 2.1 Query Rewriting (`query_rewriter.py`)

**Mục đích**: Giải quyết anaphora (đại từ) trong multi-turn conversation

**Anaphora Patterns**:
```python
_ANAPHORA = {"it", "its", "they", "them", "their", "this", "that", 
             "these", "those", "he", "she", "him", "her", "his"}
```

**Flow**:
1. Check if query contains anaphora words
2. If yes → call LLM to rewrite with context từ conversation history
3. If no → return original query

**LLM Prompt**:
```
Given the conversation history, rewrite the query to be self-contained.
Replace pronouns with their actual references.
```

**Example**:
- History: "What is the PTO policy?"
- Query: "How many days does it allow?"
- Rewritten: "How many days does the PTO policy allow?"

### 2.2 Intent Detection (`query_understanding.py`)

**Technique**: Regex Pattern Matching

```python
INTENT_PATTERNS = {
    "policy": r"\b(policy|policies|rule|rules|guideline)\b",
    "procedure": r"\b(procedure|process|step|how to|workflow)\b",
    "contact": r"\b(contact|email|phone|reach|department)\b",
    "deadline": r"\b(deadline|due|when|date|time|schedule)\b",
    "requirement": r"\b(requirement|require|need|must|should|eligible)\b",
    "definition": r"\b(what is|define|definition|meaning|means)\b",
    "summary": r"\b(summary|summarize|overview|explain)\b",
    "numeric": r"\b(how many|how much|number|count|total|percentage|rate|amount)\b"
}
```

**Output**: Intent type + confidence score

### 2.3 Query Expansion

**Based on detected intent**, thêm synonyms:
```python
expansions = {
    "policy": ["guideline", "rule", "regulation"],
    "procedure": ["process", "workflow", "steps"],
    "numeric": ["table", "data", "statistics"]
}
```

---

## 3. Retrieval Pipeline (9 Steps)

### Full Flow trong `retrieval.py`:

```
Query → Rewrite → Intent → Expand → Vector Search → Keyword Search 
    → RRF Fusion → Intent Boost → Rerank → Top K Results
```

### 3.1 Vector Search
```python
# Cosine similarity search trong ChromaDB
results = self.vector_store.query(
    query_embedding=embedding,
    n_results=top_k * 2  # Fetch more for reranking
)
```

### 3.2 Keyword Search (BM25-style)
- Full-text search trên document content
- Dùng ChromaDB's `where_document` với `$contains`

### 3.3 Hybrid Search + RRF Fusion

**Reciprocal Rank Fusion (RRF)**:
```python
def rrf_score(ranks: List[int], k: int = 60) -> float:
    return sum(1.0 / (k + rank) for rank in ranks)
```

**Parameters**:
- `k = 60`: Smoothing constant (standard value)
- `alpha = 0.5`: Balance giữa vector (semantic) và keyword (lexical)

**Tại sao RRF?**
- Không cần normalize scores (khác scale từ vector vs keyword)
- Robust với outliers
- Proven effective trong information retrieval

### 3.4 Intent-Based Boosting

```python
if intent == "numeric":
    # Boost table chunks 6x
    if chunk.metadata.get("is_table"):
        score *= 6.0

# Page number boosting cho specific page queries
if "page" in query.lower():
    page_match = re.search(r"page (\d+)", query)
    if chunk.metadata.get("page") == target_page:
        score *= 3.0
```

### 3.5 Cross-Encoder Reranking (`reranker.py`)

**Model**: `BAAI/bge-reranker-base`

```python
self.model = CrossEncoder(model_name)
```

**Scoring**:
```python
# Reranker nhận (query, document) pairs
pairs = [(query, doc.content) for doc in candidates]
rerank_scores = self.model.predict(pairs)

# Final score: weighted combination
final_score = 0.2 * original_score + 0.8 * rerank_score
```

**Tại sao Cross-Encoder?**
- **Bi-encoder** (embedding): Fast nhưng approximate
- **Cross-encoder**: Slow nhưng accurate (xem query + doc cùng lúc)
- Dùng bi-encoder để narrow down, cross-encoder để re-rank top candidates

---

## 4. Answer Generation

### 4.1 Context Assembly
```python
context = "\n\n---\n\n".join([
    f"[Source: {doc.metadata['source']}, Page {doc.metadata.get('page', 'N/A')}]\n{doc.content}"
    for doc in top_k_docs
])
```

### 4.2 System Prompt
```
You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer cannot be found in the context, say so clearly.

TABLE PRIORITY: When the user asks about numbers, statistics, or data,
focus primarily on information from table chunks.
```

### 4.3 LLM Generation
- **Provider**: vLLM hoặc Ollama
- **Streaming**: Server-Sent Events (SSE)
- **Max tokens**: 512

---

## 5. NLP Techniques Summary

| Technique | Implementation | Purpose | Hoạt động khi nào |
|-----------|---------------|---------|-------------------|
| **Token Counting** | tiktoken `cl100k_base` | Đo độ dài text để chunking | Lúc INGESTION (chunk documents) |
| **Semantic Embedding** | BGE-base-en-v1.5 Bi-encoder | Text → Dense vectors (768 dims) | Lúc INGESTION (embed chunks) & QUERY (embed user question) |
| **Asymmetric Retrieval** | Query prefix | Khác biệt query vs doc space | Lúc QUERY (thêm prefix vào user question) |
| **Hybrid Search** | Vector + Keyword | Cover semantic + lexical match | Lúc RETRIEVAL (song song vector & keyword) |
| **RRF Fusion** | k=60 | Merge 2 ranking lists không cần normalize | Sau khi có vector & keyword results |
| **Anaphora Resolution** | LLM rewriting | Thay thế đại từ bằng entity thật | Đầu QUERY pipeline (trước search) |
| **Intent Classification** | Regex patterns | Nhận dạng loại câu hỏi | Ngay sau query rewrite |
| **Cross-Encoder Reranking** | BGE-reranker | Deep scoring (query, doc) pairs | Cuối RETRIEVAL (top 10 → top 5)
    │         ├─ pdfplumber extract text
    │         ├─ Phân chia thành text_blocks (by page, by paragraph)
    │         └─ Output: {"text_blocks": [...], "pages": 45}
    │
    ├──[2]─► TableExtractor.extract_tables()
    │         ├─ Tìm tables trong PDF
    │         ├─ Convert table → row-based text
    │         └─ Output: [{"rows": [...], "page": 10, "name": "PTO Table"}]
    │
    ├──[3]─► TextChunker.chunk_text()
    │         ├─ Input: text_blocks từ [1]
    │         ├─ Loop qua từng block:
    │         │   ├─ tiktoken.count_tokens(block) → 150 tokens
    │         │   ├─ current_chunk += block → 550 tokens (chưa đủ 600)
    │         │   ├─ Next block → would make 750 tokens (quá 600!)
    │         │   └─ Save current_chunk, start new chunk
    │         ├─ Overlap: 100 tokens cuối chunk A → đầu chunk B
    │         └─ Output: [{content: "...", metadata: {page: 1}}, ...]
    │                     └─ 87 chunks (text only, chưa có tables)
    │
    ├──[4]─► Process Tables → Chunks
    │         ├─ Convert tables từ [2] thành chunk format
    │         └─ Output: 5 table chunks với metadata {is_table: true}
    │
    ├──[5]─► EmbeddingService.embed_documents()
    │         ├─ Input: 92 chunks (87 text + 5 tables) - plain text
    │         ├─ BGE Model thực hiện:
    │         │   ├─ Tokenize BERT-style: "PTO policy..." → [101, 2343, ...]
    │         │   ├─ Encode → [768-dim vector per chunk]
    │         │   └─ Normalize vectors
    │         └─ Output: numpy array shape (92, 768)
    │
    └──[6]─► VectorStore.add_chunks()
              ├─ Input: chunks + embeddings
              ├─ ChromaDB.add():
              │   ├─ Store vectors in HNSW index
              │   ├─ Store metadata (page, source, is_table)
              │   └─ Store original text content
              └─ Persist to chroma_db/chroma.sqlite3

RESULT: 92 chunks indexed, ready for search
```

### 7.2 QUERY PIPELINE (User Query → Answer)

```
USER: "How many PTO days do employees get?"
    │
    ├──[1]─► QueryRewriter.rewrite_if_needed()
    │         ├─ Check for anaphora: {"it", "they", "this"} → NOT found
    │         ├─ Skip LLM rewrite (no pronouns)
    │         └─ Output: original query (unchanged)
    │
    ├──[2]─► QueryUnderstanding.detect_intent()
    │         ├─ Regex match: "how many" → NUMERIC intent
    │         └─ Output: {intent: "numeric", confidence: 0.95}
    │
    ├──[3]─► QueryExpansion.expand()
    │         ├─ Intent-based expansion: numeric → ["table", "data", "count"]
    │         └─ Output: "How many PTO days table data count"
    │
    ├──[4]─► VectorSearch (EmbeddingService + VectorStore)
    │         ├─ Embed query với BGE model:
    │         │   └─ "Represent this sentence for searching...: {query}"
    │         │       → [768-dim query vector]
    │         ├─ ChromaDB.query(embedding, n_results=20)
    │         │   ├─ HNSW approximate nearest neighbor
    │         │   ├─ Cosine similarity scoring
    │         │   └─ Return top 20 chunks
    │         └─ Output: 20 chunks with cosine scores [0.85, 0.82, ...]
    │
    ├──[5]─► KeywordSearch
    │         ├─ ChromaDB.query(where_document={"$contains": "PTO"})
    │         └─ Output: 15 chunks matching keywords
    │
    ├──[6]─► RRF Fusion
    │         ├─ Combine rankings from [4] and [5]
    │         ├─ Formula: score = Σ(1 / (60 + rank))
    │         │   Example:
    │         │   Chunk A: vector_rank=1, keyword_rank=3
    │         │   → score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
    │         └─ Output: 25 unique chunks, re-ranked
    │
    ├──[7]─► Intent-Based Boosting
    │         ├─ Intent = "numeric" → Boost table chunks 6x
    │         ├─ Loop: if chunk.metadata.is_table: score *= 6.0
    │         └─ Output: Table chunks now top-ranked
    │
    ├──[8]─► CrossEncoder Reranking
    │         ├─ Input: Top 10 chunks from [7]
    │         ├─ BGE Reranker model:
    │         │   ├─ Pairs: [(query, chunk1), (query, chunk2), ...]
    │         │   ├─ Deep scoring (not just cosine)
    │         │   └─ Rerank scores: [0.92, 0.88, 0.65, ...]
    │         ├─ Final score: 0.2 * old_score + 0.8 * rerank_score
    │         └─ Output: Top 5 most relevant chunks
    │
    ├──[9]─► Context Assembly
    │         ├─ Format: "[Source: handbook.pdf, Page 12]\n{chunk_content}"
    │         └─ Output: Concatenated context string
    │
    └──[10]─► LLM Generation (vLLM/Ollama)
              ├─ Prompt:
              │   ├─ System: "Answer based ONLY on context..."
              │   ├─ Context: assembled from [9]
              │   └─ Query: "How many PTO days..."
              ├─ Stream response via SSE
              └─ Output: "Employees receive 15 PTO days per year..."

ANSWER DISPLAYED: Frontend shows streaming response + sources

┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  User Query                                                     │
│      ↓                                                          │
│  [1] Anaphora Resolution (LLM rewrite if pronouns detected)     │
│      ↓                                                          │
│  [2] Intent Detection (regex patterns)                          │
│      ↓                                                          │
│  [3] Query Expansion (intent-based synonyms)                    │
│      ↓                                                          │
│  [4] Vector Search ──────┬────── [5] Keyword Search             │
│                          ↓                                      │
│                    [6] RRF Fusion                                │
│                          ↓                                      │
│              [7] Intent-Based Boosting                          │
│                          ↓                                      │
│              [8] Cross-Encoder Reranking                        │
│                          ↓                                      │
│                    [9] Top K Results                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     GENERATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│  Context + Query → LLM (vLLM/Ollama) → Streaming Response       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Optimization Strategies

### 8.1 Chunk Size Tuning (Currently 600 tokens)

**🎯 Problem**: Fixed chunk size không tối ưu cho tất cả document types

**✅ Solution Options**:

#### Option 1: Dynamic Chunk Size Based on Content Type
```python
# Trong chunker.py, thêm adaptive logic:
def get_optimal_chunk_size(self, content_type: str) -> int:
    """Determine chunk size based on content type."""
    return {
        "table": 1000,      # Tables cần context đầy đủ
        "list": 400,        # Lists ngắn gọn
        "paragraph": 600,   # Standard text
        "code": 800,        # Code blocks cần continuity
    }.get(content_type, 600)
```

#### Option 2: Evaluation-Based Tuning
```python
# Benchmark different chunk sizes với evaluation metrics
from backend.evaluation.comprehensive_rag_metrics import RAGMetrics

chunk_sizes = [300, 400, 500, 600, 800]
results = {}

for size in chunk_sizes:
    # Re-ingest với size mới
    ingest_with_chunk_size(size)
    
    # Run evaluation dataset
    metrics = RAGMetrics()
    scores = metrics.evaluate(test_queries)
    
    results[size] = {
        "retrieval_precision": scores["precision"],
        "answer_quality": scores["faithfulness"],
        "speed": scores["latency"]
    }

# Chọn size tốt nhất
best_size = max(results, key=lambda x: results[x]["answer_quality"])
```

**Metrics để đánh giá**:
- **Retrieval Precision@K**: % chunks retrieved có chứa answer
- **Faithfulness**: Answer có đúng với context không
- **Answer Relevance**: Answer có đúng câu hỏi không
- **Latency**: Thời gian từ query → answer

#### Option 3: Semantic Chunking (Advanced) ⭐ **IMPLEMENTED**

**Implemented in**: [backend/services/semantic_chunker.py](backend/services/semantic_chunker.py)

```python
# Algorithm: Cut at semantic boundaries, not arbitrary token counts

# Step 1: Split into sentences
sentences = split_sentences(text)
# ["PTO policy allows...", "Employees must submit...", "Health insurance includes..."]

# Step 2: Embed sentences
embeddings = model.encode(sentences)
# [[0.23, -0.41, ...], [0.19, -0.38, ...], [-0.12, 0.73, ...]]

# Step 3: Calculate similarity between consecutive sentences
for i in range(1, len(sentences)):
    similarity = cosine_similarity(embeddings[i-1], embeddings[i])
    
    if similarity < 0.5:  # Semantic break detected!
        # Topic changed → create new chunk
        create_chunk(current_sentences)
        current_sentences = []

# Example results:
# - Sentences 0-3: similarity > 0.7 (all about PTO) → Chunk 1
# - Sentence 3→4: similarity = 0.42 (PTO → Insurance) → NEW CHUNK
# - Sentences 4-7: similarity > 0.65 (all about Insurance) → Chunk 2
```

**How to enable**:
```python
# In backend/config/settings.py
CHUNKING_STRATEGY: Literal["token", "semantic"] = "semantic"
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.5  # Adjust 0.3-0.7
```

**Comparison**:
```bash
# Run demo to see side-by-side comparison
python scripts/demo_semantic_chunking.py

# Output example:
# Token-based (600 tokens):
#   Chunk 1: PTO policy... [CUT] ...Health insurance...
#   → Mixed topics! ❌
#
# Semantic (threshold 0.5):
#   Chunk 1: PTO policy... (all PTO content)
#   Chunk 2: Health insurance... (all insurance content)
#   → Clean topic separation! ✅
```

**Pros**:
- ✅ Natural topic boundaries
- ✅ Better retrieval precision (30-40% improvement)
- ✅ Coherent chunks (no mixed topics)

**Cons**:
- ❌ 20x slower (needs embedding)
- ❌ More complex implementation
- ❌ Requires embedding service initialized

**When to use**: Complex documents (research papers, legal docs, technical manuals)

#### Option 4: Late Chunking (State-of-the-art)
```python
# Idea: Embed toàn bộ document trước, sau đó chunk embeddings
# Giữ được context của cả document khi embed

# Reference: https://arxiv.org/abs/2409.04701
# Jina AI's late chunking approach:
# 1. Tokenize entire document
# 2. Generate contextual embeddings (BERT-style)
# 3. Pool embeddings by chunk boundaries
# 4. Store chunked embeddings (mỗi chunk có context của cả doc)
```

### 8.2 Recommended Next Steps

**Short-term** (Quick Wins):
1. ✅ **A/B test chunk sizes**: 400 vs 600 vs 800 với eval dataset
2. ✅ **Monitor metrics**: Log retrieval precision per chunk size
3. ✅ **Make configurable**: Expose CHUNK_SIZE qua API/UI để admins tune

**Medium-term** (Better Quality):
1. 🔄 **Document-aware chunking**: Detect document structure (sections, tables)
2. 🔄 **Adaptive overlap**: Tables cần ít overlap, narrative text cần nhiều
3. 🔄 **Metadata-based boosting**: Prioritize chunks với high-quality indicators

**Long-term** (Research):
1. 🔬 **Late chunking**: Implement Jina AI approach
2. 🔬 **Learned chunking**: Train model để predict optimal chunk boundaries
3. 🔬 **Query-aware retrieval**: Fetch different chunk sizes based on query type

### 8.3 Quick Experiment Guide

**Để test chunk size khác, làm theo:**

```bash
# 1. Backup current DB
cp -r data/chroma_db data/chroma_db_backup

# 2. Update settings
# Edit backend/config/settings.py
CHUNK_SIZE: int = 800  # Change từ 600 → 800

# 3. Delete old DB
rm -rf data/chroma_db

# 4. Re-ingest documents
python scripts/ingest_documents.py

# 5. Test với queries
python scripts/evaluate_rag.py

# 6. Compare metrics
python scripts/compare_chunk_sizes.py
```

**Script ví dụ `compare_chunk_sizes.py`:**
```python
import json
from pathlib import Path

results_dir = Path("evaluation_results")

for size in [300, 600, 800]:
    result_file = results_dir / f"chunk_{size}.json"
    data = json.load(open(result_file))
    
    print(f"\nChunk Size: {size}")
    print(f"  Precision@5: {data['precision']:.2%}")
    print(f"  Faithfulness: {data['faithfulness']:.2f}/5")
    print(f"  Avg Latency: {data['latency']:.2f}s")
```

## 9. References

- [BGE Embedding Models](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [tiktoken](https://github.com/openai/tiktoken)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Semantic Chunking Guide](docs/SEMANTIC_CHUNKING_GUIDE.md) - **NEW! Interview prep**
- [Jina AI Late Chunking](https://arxiv.org/abs/2409.04701)
