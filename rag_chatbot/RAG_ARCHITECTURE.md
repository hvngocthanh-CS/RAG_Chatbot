# RAG Chatbot — Architecture & Step-by-Step Build Guide

> A practical guide to building a production-quality RAG chatbot from scratch.
> Designed to demonstrate real-world AI engineering skills for Junior AI Engineer roles.

---

## Table of Contents

**Part I — Understanding RAG**
1. [What Is RAG and Why It Matters](#1-what-is-rag-and-why-it-matters)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)

**Part II — Building the Ingestion Pipeline (Offline)**
3. [Step 1: Document Parsing](#3-step-1-document-parsing)
4. [Step 2: Text Chunking](#4-step-2-text-chunking)
5. [Step 3: Embedding Generation](#5-step-3-embedding-generation)
6. [Step 4: Vector Storage](#6-step-4-vector-storage)

**Part III — Building the Query Pipeline (Real-Time)**
7. [Step 5: Query Understanding](#7-step-5-query-understanding)
8. [Step 6: Retrieval (Hybrid Search)](#8-step-6-retrieval-hybrid-search)
9. [Step 7: Reranking](#9-step-7-reranking)
10. [Step 8: Answer Generation with LLM](#10-step-8-answer-generation-with-llm)
11. [Step 9: Streaming & Response Delivery](#11-step-9-streaming--response-delivery)

**Part IV — Production Concerns**
12. [Step 10: Conversation Management & Caching](#12-step-10-conversation-management--caching)
13. [Observability & Monitoring](#13-observability--monitoring)
14. [Resilience Patterns](#14-resilience-patterns)

**Part V — Reference**
15. [Configuration Reference](#15-configuration-reference)
16. [Data Schemas](#16-data-schemas)
17. [API Reference](#17-api-reference)
18. [Deployment Topology](#18-deployment-topology)
19. [Design Decisions & Trade-offs](#19-design-decisions--trade-offs)
20. [What to Build Next](#20-what-to-build-next)

---

# Part I — Understanding RAG

## 1. What Is RAG and Why It Matters

### The Problem

Large Language Models (LLMs) are powerful but have critical limitations for enterprise use:

- **Hallucination**: LLMs confidently generate false information
- **Knowledge cutoff**: They don't know about your company's internal docs
- **No source attribution**: You can't verify where the answer came from
- **No access control**: They can't respect "HR-only" or "Engineering-only" documents

### The Solution: Retrieval-Augmented Generation

RAG solves all of these by adding a **retrieval step** before generation:

```
Traditional LLM:
  User Question --> LLM --> Answer (may be hallucinated)

RAG:
  User Question --> RETRIEVE relevant docs --> LLM + docs --> Grounded answer with sources
```

The LLM only answers based on the documents you give it. If the documents don't contain the answer, it says "I don't know" instead of making something up.

### Why Recruiters Care About This

RAG is the #1 most deployed GenAI pattern in enterprise. Every company with internal documents (policies, wikis, runbooks, FAQs) needs this. Building a RAG chatbot demonstrates you understand:

- **Embedding models** (how text becomes vectors)
- **Vector databases** (how similarity search works)
- **LLM orchestration** (prompt engineering, context management)
- **Information retrieval** (search relevance, ranking)
- **System design** (API design, streaming, caching, error handling)

### Technology Stack Used in This Project

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| Backend API | Python 3.12 + FastAPI | Industry standard for AI/ML APIs |
| Frontend | React 18 + TypeScript + Vite | Modern web stack; shows full-stack capability |
| LLM | Ollama (local) or vLLM | Runs locally — no API costs, data stays private |
| Embedding Model | `BAAI/bge-base-en-v1.5` (768-dim) | Open-source, optimized for retrieval, runs on CPU |
| Reranker | `BAAI/bge-reranker-base` | Cross-encoder for precision; biggest quality lever |
| Vector Database | ChromaDB (default) / Qdrant | ChromaDB is embedded (no server needed); Qdrant for scale |
| Cache | Redis (optional) | Standard caching layer; reduces repeated LLM calls |
| Doc Parsing | pypdf + pdfplumber + python-docx | Handles PDF tables, DOCX styles, Markdown headers |

---

## 2. End-to-End System Architecture

### The Two Pipelines

Every RAG system has two separate pipelines that share the vector database:

```
PIPELINE 1: INGESTION (offline — runs when documents are uploaded)

  PDF/DOCX/TXT/MD
       |
       v
  [1. Parse] --> Extract text blocks + tables
       |
       v
  [2. Chunk] --> Split into ~500-token semantic chunks
       |
       v
  [3. Embed] --> Convert chunks to 768-dim vectors
       |
       v
  [4. Store] --> Save vectors + metadata in ChromaDB
```

```
PIPELINE 2: QUERY (real-time — runs when user asks a question)

  User Question + Conversation History
       |
       v
  [5. Understand] --> Rewrite query + detect intent + expand keywords
       |
       v
  [6. Retrieve]   --> Hybrid search (vector + BM25) + RRF fusion
       |
       v
  [7. Rerank]     --> Cross-encoder rescoring (top 15 → top 6)
       |
       v
  [8. Generate]   --> LLM produces answer grounded in retrieved chunks
       |
       v
  [9. Stream]     --> SSE delivers tokens + sources to frontend
```

### Full Architecture Diagram

```
                        USER (Browser)
                             |
                             v
              +-----------------------------+
              |    FRONTEND (React + TS)    |
              |    - Chat interface          |
              |    - Document upload         |
              |    - Real-time streaming     |
              +-------------+---------------+
                            |
                   HTTP / SSE (port 3000 → 8000)
                            |
                            v
+=================================================================+
|                   FastAPI BACKEND (port 8000)                    |
|=================================================================|
|                                                                 |
|  +--------------------+    +---------------------------------+  |
|  | INGESTION PIPELINE |    | QUERY PIPELINE                  |  |
|  |                    |    |                                   |  |
|  | Upload API         |    | Chat API                         |  |
|  |   |                |    |   |                               |  |
|  |   v                |    |   v                               |  |
|  | DocumentParser     |    | ConversationManager              |  |
|  |   |                |    |   |                               |  |
|  |   v                |    |   v                               |  |
|  | TableExtractor     |    | QueryRewriter (LLM-based)        |  |
|  |   |                |    |   |                               |  |
|  |   v                |    |   v                               |  |
|  | SemanticChunker    |    | QueryUnderstanding (intent)      |  |
|  |   |                |    |   |                               |  |
|  |   v                |    |   v                               |  |
|  | EmbeddingService   |    | RetrievalService                 |  |
|  |   |                |    |   |  - Vector search              |  |
|  |   v                |    |   |  - Keyword search (BM25)      |  |
|  | VectorStore.add()  |    |   |  - RRF fusion                 |  |
|  |                    |    |   |  - Intent boosting             |  |
|  +--------------------+    |   v                               |  |
|                            | RerankerService (cross-encoder)   |  |
|                            |   |                               |  |
|                            |   v                               |  |
|                            | LLMService (generate / stream)    |  |
|                            |   |                               |  |
|                            |   v                               |  |
|                            | SSE response to frontend          |  |
|                            +---------------------------------+  |
|                                                                 |
+=================================================================+
        |                    |                    |
        v                    v                    v
  +-----------+      +-------------+      +------------+
  | ChromaDB  |      |   Ollama    |      |   Redis    |
  | (vectors) |      |   (LLM)    |      |  (cache)   |
  +-----------+      +-------------+      +------------+
```

### What Makes This Architecture "Production-Quality"

| Feature | Basic RAG (tutorial-level) | This Project |
|---------|--------------------------|-------------|
| Chunking | Fixed-size (500 chars) | Semantic chunking with embedding-based boundary detection |
| Search | Vector-only | Hybrid: vector + BM25 keyword + RRF fusion |
| Ranking | None | Cross-encoder reranking (+16% accuracy) |
| Query handling | Pass-through | Intent detection + query rewriting + expansion |
| Multi-turn | None | LLM-based query rewriting resolves pronouns |
| Delivery | Wait for full response | Token-by-token SSE streaming |
| Reliability | Crashes on LLM timeout | Circuit breaker + retry + concurrency control |
| Monitoring | print() statements | Structured JSON logging + Prometheus metrics |

---

# Part II — Building the Ingestion Pipeline (Offline)

## 3. Step 1: Document Parsing

### What You're Building

A service that takes raw files (PDF, DOCX, TXT, MD) and extracts structured text blocks and tables.

### Service: `DocumentParser` → `backend/services/document_parser.py`

### How It Works

```
Raw file
    |
    v
Detect format by file extension
    |
    ├── PDF:   pypdf for text extraction (page by page)
    |          pdfplumber for table detection (row/column parsing)
    |          Font-size analysis to detect titles vs headings vs body text
    |
    ├── DOCX:  python-docx for paragraph extraction
    |          Style detection maps Heading 1/2/3 to document structure
    |          Table extraction with header row identification
    |
    └── TXT/MD: Plain text reading
               Markdown header detection (# / ## / ###)
               Line-based block splitting
    |
    v
Output structure:
{
    "text_blocks": [
        {"text": "...", "page_number": 1, "type": "title"},
        {"text": "...", "page_number": 1, "type": "heading"},
        {"text": "...", "page_number": 1, "type": "paragraph"}
    ],
    "tables": [
        {"headers": ["Name", "Days", "Notes"], "rows": [[...]], "page_number": 3}
    ],
    "metadata": {"page_count": 12, "file_type": ".pdf"}
}
```

### Table Extraction: `TableExtractor` → `backend/services/table_extractor.py`

Tables are dense, structured data. They need special handling because embedding a raw CSV doesn't work well. Instead, tables are converted to a row-based text format:

```
Table: Employee Leave Entitlements

Row 1:
Leave Type: Annual Leave
Entitlement: 15 days/year
Carry Over: Max 5 days

Row 2:
Leave Type: Sick Leave
Entitlement: 10 days/year
Carry Over: No
```

This format lets the LLM read table data naturally and answer questions like "How many sick leave days do I get?" accurately.

### Key Interview Talking Points

> **Q: Why not just dump the entire PDF text into the LLM?**
> A: LLMs have context window limits (4K-128K tokens). A 50-page PDF won't fit. Even if it did, the LLM performs worse with irrelevant noise — retrieval + focused context beats "dump everything."

> **Q: Why extract tables separately?**
> A: Tables contain dense numerical data (salaries, leave days, deadlines) that are high-value for Q&A. They need dedicated parsing because PDF table extraction is notoriously difficult — text extraction alone loses the row/column structure.

---

## 4. Step 2: Text Chunking

### What You're Building

A service that splits large text blocks into optimally-sized chunks that balance between having enough context and being focused enough for accurate retrieval.

### The Core Trade-off

```
Too small (50 tokens):   "15 days"         → Retrieved but LLM lacks context
Too large (2000 tokens): [entire section]   → Too much noise, dilutes relevance
Sweet spot (300-600):    [complete topic]   → Enough context, focused retrieval
```

### Strategy A: Semantic Chunking (Default) → `SemanticChunker`

**File:** `backend/services/semantic_chunker.py`

This is the advanced approach. Instead of cutting at fixed intervals, it detects natural topic boundaries using embedding similarity.

```
Algorithm:
  1. Split document into individual sentences
  2. Generate an embedding for each sentence
  3. Calculate cosine similarity between consecutive sentences
  4. When similarity drops sharply → that's a topic boundary → cut here
  5. Enforce hard limits: max 600 tokens, min 80 tokens per chunk
  6. Apply 3-sentence overlap between chunks for continuity

Visualization:

  Sentence:     S1    S2    S3    S4    S5    S6    S7    S8
  Similarity:      0.92  0.88  0.35  0.91  0.87  0.29  0.85
                                 ↑                   ↑
                            TOPIC SHIFT          TOPIC SHIFT

  Result: Chunk A = [S1, S2, S3]
          Chunk B = [S3, S4, S5, S6]    ← S3 overlaps (continuity)
          Chunk C = [S6, S7, S8]        ← S6 overlaps
```

**Configuration:**

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `max_chunk_size` | 600 tokens | Hard upper limit — prevents oversized chunks |
| `min_chunk_size` | 80 tokens | Filters out tiny fragments that aren't useful |
| `similarity_threshold` | 0.4 | Lower = more aggressive splitting (more, smaller chunks) |
| `overlap_sentences` | 3 | How many sentences are shared between adjacent chunks |

### Strategy B: Token-Based Chunking (Simpler) → `TextChunker`

**File:** `backend/services/chunker.py`

Fixed-size chunking with smart boundary detection:

```
Config: chunk_size=500, chunk_overlap=100, min_chunk_size=100
Split priority: paragraph boundary > sentence boundary > word boundary
```

### Table Chunks

Tables become their own chunks with `chunk_type: "table"` in metadata. This is important later — when the user asks a numeric question ("How many days?"), the system can boost table chunks 3x.

### Why Semantic Chunking Wins

| Aspect | Semantic Chunking | Fixed-Size Chunking |
|--------|------------------|-------------------|
| Topic coherence | Each chunk = one topic | May cut mid-topic |
| Retrieval quality | Higher relevance per chunk | More noise per chunk |
| Reranking benefit | Clean signal for cross-encoder | Cross-encoder compensates but starts worse |
| Speed | Slower (embeds every sentence at ingest time) | Faster |
| Best for | Production systems | Quick prototyping |

### Key Interview Talking Points

> **Q: Why overlap between chunks?**
> A: Without overlap, if a sentence at a chunk boundary is the answer to a query, it might be split across two chunks and neither chunk alone has the full context. Overlap ensures boundary information appears in both adjacent chunks.

> **Q: What's the downside of semantic chunking?**
> A: It's slower during ingestion because you embed every sentence to compute similarities. But ingestion is offline — you pay this cost once. Query-time performance is what matters, and semantic chunks improve that.

---

## 5. Step 3: Embedding Generation

### What You're Building

A service that converts text chunks into 768-dimensional dense vectors — the mathematical representation that makes similarity search possible.

### Service: `EmbeddingService` → `backend/services/embeddings.py`

### Model: `BAAI/bge-base-en-v1.5`

| Property | Value |
|----------|-------|
| Dimensions | 768 |
| Parameters | 110M |
| Max input tokens | 512 |
| Device | CPU (configurable to CUDA/GPU) |
| Batch size | 32 documents per batch |

### Critical Concept: Asymmetric Encoding

BGE models are **asymmetric** — they use a different prefix for queries vs. documents. This is the most common mistake people make when implementing BGE embeddings:

```python
# QUERY embedding — add this prefix (tells the model "this is a search query")
input = "Represent this sentence for searching relevant passages: What is the leave policy?"

# DOCUMENT embedding — no prefix (tells the model "this is a passage to be found")
input = "Employees are entitled to 15 days of annual leave per year..."
```

**Why asymmetric?** The query embedding is optimized to *find* relevant passages. The document embedding is optimized to *be found*. This asymmetry improves retrieval by ~5-10% over symmetric encoding. Using the wrong prefix (or no prefix for queries) significantly degrades results.

### Methods

```python
# Embeds a single query with the BGE query prefix
async embed_query(text: str) -> List[float]    # Returns 768-dim vector

# Embeds multiple documents WITHOUT prefix, in batches of 32
async embed_documents(texts: List[str]) -> List[List[float]]

# Returns the model's output dimension
get_embedding_dimension() -> int  # 768
```

### Key Interview Talking Points

> **Q: Why not use OpenAI's text-embedding-ada-002?**
> A: Three reasons: (1) data privacy — with BGE, no company data leaves the network; (2) cost — BGE runs locally for free vs. $0.0001/1K tokens for ada-002; (3) we can fine-tune BGE on domain-specific data later.

> **Q: What does "768 dimensions" mean?**
> A: Each chunk becomes a point in 768-dimensional space. Semantically similar texts end up close together. "Annual leave policy" and "vacation day rules" will be nearby, even though they share no keywords.

> **Q: Why batch processing?**
> A: Embedding 1,000 chunks one-by-one takes ~30 seconds. Batching 32 at a time takes ~3 seconds — the model processes them in parallel on the hardware.

---

## 6. Step 4: Vector Storage

### What You're Building

A persistent store for your embeddings that supports fast similarity search and metadata filtering.

### Service: `VectorStore` → `backend/services/vector_store.py` (Abstract Factory pattern)

### ChromaDB (Default Implementation)

```
Storage:  Persistent on disk (./data/chroma_db)
Index:    HNSW (Hierarchical Navigable Small World graph)
Metric:   Cosine similarity
Features:
  - Metadata filtering (e.g., department="HR", chunk_type="table")
  - Built-in BM25 keyword search
  - Persistent across application restarts
  - No separate server needed (embedded in your Python process)
```

### Qdrant (Production Alternative)

```
Storage:  Self-hosted server or Qdrant Cloud
Index:    HNSW with optional scalar/binary quantization
Features:
  - Advanced filtering with nested boolean conditions
  - Horizontal scaling (sharding + replication)
  - Separate server = independent scaling from your API
```

### What Gets Stored Per Chunk

```json
{
  "id": "doc_abc123_chunk_0",
  "embedding": [0.012, -0.034, 0.089, "... 768 floats total"],
  "document": "The actual text content of this chunk...",
  "metadata": {
    "document_id": "abc123",
    "filename": "01_Employee_Handbook.pdf",
    "file_type": ".pdf",
    "file_size": 23000,
    "chunk_type": "text",
    "chunk_index": 0,
    "page_number": 5,
    "section": "2.2 Leave Entitlements",
    "department": "HR",
    "category": "Policy",
    "tags": "leave,hr,policy",
    "author": "HR Team",
    "created_at": "2026-03-15T10:30:00Z"
  }
}
```

### Core Methods

```python
async add_chunks(chunks, metadata)             # Store embeddings + metadata
async search(query_embedding, top_k, filters)  # Vector similarity search
async keyword_search(query, top_k, filters)    # BM25 text search (for hybrid)
async delete_document(document_id)             # Remove all chunks of a document
async get_collection_stats()                   # Collection count and size info
```

### Key Interview Talking Points

> **Q: What is HNSW and why does it matter?**
> A: HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor algorithm. Exact search across 1M vectors would take seconds. HNSW builds a graph structure that finds the ~95% best results in milliseconds. The trade-off is a small accuracy loss for massive speed gain.

> **Q: When would you switch from ChromaDB to Qdrant?**
> A: ChromaDB is embedded — great up to ~1M vectors on a single machine. Beyond that, or if you need horizontal scaling, production-grade monitoring, or advanced filtering, Qdrant (or Pinecone, Weaviate) makes sense.

---

# Part III — Building the Query Pipeline (Real-Time)

## 7. Step 5: Query Understanding

### What You're Building

A multi-stage query processor that transforms the raw user question into optimized search queries. This is where a basic RAG becomes a *good* RAG.

### Sub-step 5a: Query Rewriting (multi-turn conversations)

**Service:** `QueryRewriterService` → `backend/services/query_rewriter.py`

**Problem:** In a multi-turn conversation, follow-up questions use pronouns ("it", "that", "how many days?") that don't make sense without context.

```
Conversation:
  User:      "Tell me about the leave policy"
  Assistant: "TechViet provides 15 days annual leave..."
  User:      "How many days can I carry over?"
                     ↓
              Without rewriting: search for "How many days can I carry over?"
              → Retrieves chunks about many unrelated topics with "days" and "carry over"
                     ↓
              With rewriting (LLM resolves the pronoun):
              "How many annual leave days can an employee carry over to the next year
               according to TechViet's leave policy?"
              → Retrieves exactly the right leave policy chunk
```

**When it triggers:** Only when `conversation_history` has >= 2 messages. First question in a conversation skips this entirely.

**Config:** `temperature=0.0` (deterministic), `max_tokens=120` (lightweight — the rewrite should be fast).

### Sub-step 5b: Intent Detection (what kind of question is this?)

**Service:** `QueryUnderstandingService` → `backend/services/query_understanding.py`

**Approach:** Rule-based regex pattern matching. Fast (<1ms), interpretable, no LLM call needed.

```
Input: "How many sick leave days do I get per year?"
         ↓
  Pattern match: "how many" → intent = "numeric"
         ↓
Output: {
    intent_type: "numeric",
    confidence: 0.9,
    boost_metadata: {"chunk_type": "table"},   // numbers are usually in tables
    expansion_terms: ["how many", "number", "count", "total", "duration"]
}
```

**Supported intents and their retrieval effects:**

| Intent | Trigger Patterns | What It Does to Retrieval |
|--------|-----------------|--------------------------|
| `policy` | "policy", "rule", "guideline" | Boosts policy document chunks |
| `procedure` | "how to", "steps to", "process for" | Boosts procedure/guide chunks |
| `contact` | "who to contact", "email", "phone" | Boosts contact info chunks |
| `deadline` | "when", "deadline", "due date" | Boosts date-containing chunks |
| `requirement` | "required", "must", "mandatory" | Boosts requirement/checklist chunks |
| `definition` | "what is", "define", "meaning of" | Boosts definition/overview chunks |
| `summary` | "overview", "summary", "tell me about" | Prefers introductory chunks |
| `numeric` | "how many", "how much", "number" | **3x boost to table chunks** (tables contain numbers) |
| `multi_context` | Complex queries spanning topics | No special boost — relies on hybrid search |

### Sub-step 5c: Query Expansion (add synonyms for keyword search)

Based on the detected intent, expand the keyword query with related terms:

```
Original:  "What is the leave policy?"
Intent:    policy
Expanded:  "What is the leave policy rule guideline regulation compliance standard"
```

This helps the BM25 keyword search find documents that use different words for the same concept (e.g., "guideline" instead of "policy").

### Key Interview Talking Points

> **Q: Why rule-based intent detection instead of an LLM classifier?**
> A: Speed and reliability. Regex matching takes <1ms. An LLM call would add 1-2 seconds per query. For a chatbot, latency matters. The rule-based approach covers 90%+ of query patterns. You can always add an LLM fallback for the remaining edge cases later.

> **Q: Why is query rewriting important?**
> A: Without it, multi-turn conversations are broken. "How many days?" means nothing without knowing we're talking about leave policy. This is the #1 reason enterprise RAG chatbots give bad answers — they search for the literal follow-up question instead of the resolved question.

---

## 8. Step 6: Retrieval (Hybrid Search)

### What You're Building

The core retrieval engine that finds the most relevant document chunks. This step uses **two search methods** combined for the best results.

### Service: `RetrievalService` → `backend/services/retrieval.py`

### Why Hybrid Search?

```
Vector search is great at:
  "What's the remote work policy?" → finds "work from home guidelines" (semantic match)

Vector search is bad at:
  "VND 50,000 lunch allowance" → might miss the exact number (keyword match)

BM25 keyword search is great at:
  "VND 50,000" → finds the exact string (lexical match)

BM25 keyword search is bad at:
  "vacation days" → misses "annual leave" (no semantic understanding)

Hybrid = combine both → best of both worlds
```

### The Complete 10-Step Retrieval Pipeline

```
User Question + Conversation History
         |
         v
   +-----------------------+
   | Step 0: Query Rewrite |  LLM-based, multi-turn only
   +-----------+-----------+
               |
               v
   +-----------------------+
   | Step 1: Intent Detect |  Regex-based, <1ms
   +-----------+-----------+
               |
               v
   +-----------------------+
   | Step 2: Enhance Query |  Add conversation context
   +-----------+-----------+
               |
               v
   +-----------------------+
   | Step 3: Prepare Two   |  embedding_query → for vector search
   |   Search Queries      |  keyword_query  → expanded, for BM25
   +-----------+-----------+
               |
         +-----+-----+
         |           |
         v           v
   +---------+  +---------+
   | Step 4  |  | Step 7  |
   | Vector  |  | Keyword |
   | Search  |  | Search  |
   | (top 40)|  | (BM25)  |
   +---------+  +---------+
         |           |
         v           |
   +---------+       |
   | Step 5  |       |
   | Filter  |       |
   | score   |       |
   | > 0.3   |       |
   +---------+       |
         |           |
         v           |
   +---------+       |
   | Step 6  |       |
   | Intent  |       |
   | Boost   |       |
   +---------+       |
         |           |
         +-----+-----+
               |
               v
   +-----------------------+
   | Step 8: RRF Fusion    |  Combine vector + keyword results
   +-----------+-----------+
               |
               v
   +-----------------------+
   | Step 9: Rerank        |  Cross-encoder (top 15 → top 6)
   +-----------+-----------+
               |
               v
   +-----------------------+
   | Step 10: Return       |  Top-K chunks with scores + metadata
   +-----------------------+
```

### Step 4 — Vector Search (Semantic)

```python
query_embedding = EmbeddingService.embed_query(embedding_query)
results = VectorStore.search(
    query_embedding=query_embedding,
    top_k=40,               # Cast a wide net — retrieve many candidates
    filters=user_filters    # Optional: department="HR", tags="leave"
)
```

### Step 5 — Score Threshold Filtering

Remove low-confidence results before spending compute on reranking:

```python
filtered = [r for r in results if r["score"] >= 0.3]
```

### Step 6 — Intent-Based Metadata Boosting

If the user is asking a numeric question, tables probably have the answer:

```python
if intent.intent_type == "numeric":
    for chunk in results:
        if chunk["metadata"]["chunk_type"] == "table":
            chunk["score"] *= 3.0  # Triple the score for table chunks
```

### Step 7 — Keyword Search (BM25)

Run in parallel with vector search:

```python
keyword_results = VectorStore.keyword_search(
    query=keyword_query,    # Expanded with synonyms from Step 5c
    top_k=40,
    filters=user_filters
)
```

### Step 8 — RRF (Reciprocal Rank Fusion)

The standard method to combine two ranked lists:

```
For each result appearing in either list:
  rrf_vector  = 1 / (60 + rank_in_vector_results)     # 0 if not in list
  rrf_keyword = 1 / (60 + rank_in_keyword_results)     # 0 if not in list

  combined_score = 0.7 × rrf_vector + 0.3 × rrf_keyword
                   ^^^                 ^^^
                   70% weight to       30% weight to
                   semantic search     keyword search
```

**Why 60?** It's the standard RRF constant. It prevents top-ranked items from dominating too heavily.

**Why 0.7/0.3?** Semantic search is generally more useful for natural language questions. Keywords help with exact matches. This ratio is configurable via `HYBRID_ALPHA`.

### Key Interview Talking Points

> **Q: Why retrieve 40 candidates if you only need 6?**
> A: Reranking is more expensive but more accurate. You want to give the reranker a large candidate pool so it has a better chance of finding the truly best chunks. Retrieving only 6 and hoping they're all correct is risky.

> **Q: What is RRF and why not just average the scores?**
> A: Vector scores (cosine similarity) and BM25 scores are on completely different scales — you can't meaningfully average them. RRF converts both to rank-based scores (which are comparable) and then combines them. It's simple, robust, and requires no score normalization.

---

## 9. Step 7: Reranking

### What You're Building

A precision layer that re-scores the top candidates using a more powerful (but slower) model. This is the **single biggest quality improvement** you can add to a RAG system.

### Service: `RerankerService` → `backend/services/reranker.py`

### Model: `BAAI/bge-reranker-base` (cross-encoder)

### Why Reranking Matters: Bi-Encoder vs Cross-Encoder

```
Bi-Encoder (used in Step 3 — fast, scalable):
  ┌──────────┐     ┌──────────────────────┐
  │  Query   │     │     Document         │
  │ encoder  │     │     encoder          │
  └────┬─────┘     └──────────┬───────────┘
       │                      │
       v                      v
  [query vector]    [document vector]
       │                      │
       └──────── dot ─────────┘
                  │
              score: 0.82

  Problem: Query and document are encoded INDEPENDENTLY.
  They never "see" each other during encoding.
```

```
Cross-Encoder (used in Step 7 — slow, precise):
  ┌─────────────────────────────────────┐
  │  [CLS] query [SEP] document [SEP]  │
  │         ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓         │
  │     Full transformer attention      │
  │     (every token sees every other)  │
  └──────────────────┬──────────────────┘
                     │
                 score: 0.94

  Advantage: Query and document tokens INTERACT directly.
  The model understands the relationship, not just similarity.
```

### Scoring Formula

```python
raw_score = cross_encoder.predict(query, passage)
normalized = sigmoid(raw_score)                        # Map to [0, 1]
final_score = 0.3 × original_retrieval_score + 0.7 × normalized
```

### Pipeline Position

```
Input:  Top 15 chunks from hybrid search (sorted by RRF score)
Output: Top 6 chunks re-scored by cross-encoder
                                          ^^ TOP_K_RERANK setting
```

### Impact Measurement

| Metric | Without Reranker | With Reranker | Improvement |
|--------|-----------------|---------------|-------------|
| Top-1 accuracy | ~72% | ~88% | +16pp |
| MRR@5 | 0.68 | 0.84 | +24% |
| Latency added | — | +200-400ms (CPU) | Acceptable |
| Model size | — | 278M parameters | Runs on CPU |

Can be disabled with `USE_RERANKER=False` for latency-sensitive environments.

### Key Interview Talking Points

> **Q: Why not use the cross-encoder for everything?**
> A: It's too slow. A cross-encoder must process every (query, document) pair individually. For 10,000 chunks, that's 10,000 forward passes. The bi-encoder encodes chunks once at ingest time, then search is just a fast vector dot product. So we use bi-encoder to find 40 candidates cheaply, then cross-encoder to precisely rank the top 15.

> **Q: Is +200ms latency acceptable?**
> A: Yes. Users expect ~2-3 seconds for an AI response. The total pipeline (retrieval + rerank + LLM) takes ~2 seconds. 200ms for reranking is a small price for +16% accuracy.

---

## 10. Step 8: Answer Generation with LLM

### What You're Building

The final step: an LLM reads the retrieved chunks and generates a natural language answer with source citations.

### Service: `LLMService` → `backend/services/llm.py`

### Supported Providers

| Provider | When to Use | API |
|----------|------------|-----|
| Ollama (default) | Local development, data privacy, no API costs | OpenAI-compatible `/v1/chat/completions` |
| vLLM | High-throughput production, GPU cluster | OpenAI-compatible |

### The Prompt Structure

This is the most important part of the LLM integration — the system prompt:

```
System prompt:
  You are a helpful assistant for TechViet employees.
  Answer questions based ONLY on the provided context.
  If the context doesn't contain the answer, say
  "I don't have enough information to answer this question."
  Always cite your sources using [Source N] format.

Context (injected from Step 7):
  [Source 1: 01_Employee_Handbook.pdf, Page 5, Section: 2.2 Leave Entitlements]
  Employees are entitled to 15 days of annual leave per year...

  [Source 2: 01_Employee_Handbook.pdf, Page 5, Section: 2.2 Leave Entitlements]
  Maximum carry-over is 5 days to the following year...

User question:
  How many leave days can I carry over?
```

### Generation Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | 0.0 | Deterministic — same question always gets same answer (critical for enterprise) |
| `max_tokens` | 2048 | Long enough for detailed multi-part answers |
| `top_p` | 0.95 | Nucleus sampling (minimal effect at temp=0) |
| `presence_penalty` | 0.2 | Reduces topic repetition |
| `frequency_penalty` | 0.3 | Reduces word-level repetition |

### Streaming Implementation

```python
async def generate_stream(question, context, history) -> AsyncGenerator[str]:
    async for chunk in llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
        temperature=0.0,
        max_tokens=2048
    ):
        token = chunk.choices[0].delta.content
        if token:
            yield token  # Each token sent immediately to frontend
```

### Key Interview Talking Points

> **Q: Why temperature=0?**
> A: In a corporate knowledge assistant, you want consistency. If an employee asks the same question twice, they should get the same answer. Temperature=0 makes the output deterministic.

> **Q: What if the retrieved chunks don't contain the answer?**
> A: The system prompt explicitly instructs the LLM to say "I don't have enough information." This is a critical safety feature — it's better to say "I don't know" than to hallucinate a wrong policy answer.

---

## 11. Step 9: Streaming & Response Delivery

### What You're Building

Real-time delivery of the LLM's response to the user, token by token, so they see the answer appearing immediately instead of waiting 5-10 seconds for the full response.

### Protocol: Server-Sent Events (SSE)

```
Browser                                 Server
  |                                       |
  | POST /api/v1/chat                     |
  | {"question":"...", "stream":true}     |
  |-------------------------------------->|
  |                                       | [Retrieve chunks: ~500ms]
  |                                       | [Start LLM generation]
  |                                       |
  | data: {"type":"sources",              |  ← Sources sent FIRST
  |   "sources":[...],                    |    (user sees references immediately)
  |   "conversation_id":"uuid"}           |
  |<--------------------------------------|
  |                                       |
  | data: {"type":"token","content":"The"}|  ← Tokens stream one by one
  |<--------------------------------------|
  | data: {"type":"token","content":" leave"}
  |<--------------------------------------|
  | data: {"type":"token","content":" policy"}
  |<--------------------------------------|
  |        ... (continues) ...            |
  |                                       |
  | data: {"type":"done"}                 |  ← Generation complete
  |<--------------------------------------|
```

### Event Types

| Type | Payload | When Sent |
|------|---------|-----------|
| `sources` | Retrieved chunks with metadata, scores, filenames | First event — before any tokens |
| `token` | Single generated token (word fragment) | During LLM streaming |
| `error` | Error message string | On any failure |
| `done` | Empty | Generation complete |

### Non-Streaming Response (when `stream: false`)

```json
{
  "answer": "According to the Employee Handbook (Section 2.2), you can carry over up to 5 days...",
  "conversation_id": "uuid-123",
  "sources": [
    {
      "content": "Maximum carry-over is 5 days...",
      "document_id": "abc",
      "document_name": "01_Employee_Handbook.pdf",
      "page_number": 5,
      "chunk_type": "text",
      "relevance_score": 0.94
    }
  ],
  "processing_time_ms": 1850
}
```

### Key Interview Talking Points

> **Q: Why SSE instead of WebSockets?**
> A: SSE is simpler and sufficient. We only need server→client streaming (one direction). SSE works over plain HTTP, handles auto-reconnection natively, and is easier to implement than WebSockets. WebSockets would be overkill for a chat application where the user sends a message and waits for a response.

> **Q: Why send sources before the answer?**
> A: User experience. While the LLM is generating (which takes 1-5 seconds), the user can already see which documents will be cited. It also makes the interface feel responsive — something appears immediately.

---

# Part IV — Production Concerns

## 12. Step 10: Conversation Management & Caching

### Conversation Manager → `backend/services/conversation.py`

Stores multi-turn conversation state so follow-up questions have context.

```
Storage: In-memory OrderedDict (FIFO eviction when full)

Limits:
  - Max 1,000 concurrent conversations
  - Max 50 messages per conversation
  - Oldest conversations evicted first when limit reached

Lifecycle:
  1. POST /api/v1/chat (no conversation_id)   → new conversation created
  2. POST /api/v1/chat (with conversation_id)  → continues existing one
  3. GET  /api/v1/chat/conversations/{id}      → returns message history
  4. DELETE /api/v1/chat/conversations/{id}    → deletes conversation

Production note: Replace with Redis or PostgreSQL for persistence
across restarts and horizontal scaling.
```

### Cache Service → `backend/services/cache.py`

Optional Redis-based response caching for repeated queries.

```
Cache key:        MD5(question + sorted(filters))
Cache value:      Full JSON response (answer + sources)
TTL:              3,600 seconds (1 hour)
Invalidation:     Per-document (when re-indexed or deleted)

Flow:
  1. Compute cache key from question + filters
  2. Check Redis → HIT? Return cached response (skip entire pipeline)
  3. MISS? Execute full pipeline (retrieve + rerank + generate)
  4. Store result in Redis with TTL
  5. Return result to user
```

---

## 13. Observability & Monitoring

### Service: `ObservabilityModule` → `backend/services/observability.py`

### Structured JSON Logging

Every request produces structured logs that can be aggregated by ELK, Datadog, or Grafana Loki:

```json
{
  "timestamp": "2026-03-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "retrieval_service",
  "message": "Retrieval completed",
  "context": {
    "request_id": "req-a1b2c3d4",
    "trace_id": "trace-xyz789",
    "conversation_id": "conv-456"
  },
  "extra": {
    "query": "What is the leave policy?",
    "intent": "policy",
    "vector_results": 35,
    "keyword_results": 28,
    "after_rerank": 6,
    "latency_ms": 420
  }
}
```

### Prometheus Metrics (`GET /metrics`)

| Metric | Type | Description |
|--------|------|-------------|
| `llm_requests_total` | Counter | Total LLM API calls (by success/failure) |
| `llm_tokens_total` | Counter | Token consumption (input/output) |
| `llm_latency_ms` | Histogram | LLM response time distribution |
| `retrieval_chunks_count` | Gauge | Average chunks retrieved per query |
| `retrieval_latency_ms` | Histogram | Full retrieval pipeline latency |
| `reranker_latency_ms` | Histogram | Reranking step latency |
| `document_ingestion_total` | Counter | Documents processed (success/failure) |

### Health Check Endpoints

| Endpoint | Purpose | Use Case |
|----------|---------|----------|
| `GET /health` | Basic liveness check | "Is the server running?" |
| `GET /health/live` | Kubernetes liveness probe | K8s restarts pod if this fails |
| `GET /health/ready` | Kubernetes readiness probe | Checks LLM + VectorDB connections |
| `GET /health/detailed` | Full system diagnostic | Shows all services, circuit breaker states, latencies |
| `GET /metrics/json` | JSON metrics summary | Dashboard data without Prometheus |

---

## 14. Resilience Patterns

### Circuit Breaker (protects against LLM failures)

When the LLM provider goes down, the circuit breaker prevents cascading failures:

```
Normal operation:
  CLOSED → all requests go through to LLM

After 5 consecutive failures:
  CLOSED → OPEN → all requests fail fast (no LLM call, immediate error)
                   "Circuit is open" error returned in ~1ms instead of ~30s timeout

After 30 seconds:
  OPEN → HALF_OPEN → allows 1 test request through
                      |
                      ├── success → CLOSED (back to normal)
                      └── failure → OPEN (wait another 30 seconds)
```

### Retry with Exponential Backoff

For transient failures (network blips, temporary overload):

```
Attempt 1:  immediate
Attempt 2:  wait 1 second
Attempt 3:  wait 2 seconds
            (give up — surface error to user)
```

### Concurrency Control

```
Max concurrent LLM requests: 50 (via asyncio.Semaphore)
  → Prevents overwhelming a local Ollama server
  → Excess requests queue until a slot opens
```

### Timeout Management

| Operation | Timeout | Rationale |
|-----------|---------|-----------|
| LLM generation | 120 seconds | Long answers need time |
| Health check | 10 seconds | Should be fast |
| Embedding | 30 seconds | Batch processing can be slow |
| Vector search | 10 seconds | Should be fast (HNSW) |

---

# Part V — Reference

## 15. Configuration Reference

All settings via environment variables, loaded by Pydantic `BaseSettings` in `backend/config/settings.py`.

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `"ollama"` | `"ollama"` or `"vllm"` |
| `OLLAMA_BASE_URL` | `"http://localhost:11434/v1"` | Ollama API endpoint |
| `OLLAMA_MODEL` | `"phi3"` | Model name (phi3, llama3.2, qwen2.5) |
| `VLLM_BASE_URL` | `"http://localhost:8001/v1"` | vLLM API endpoint |
| `VLLM_MODEL_NAME` | `"microsoft/Phi-3-mini-4k-instruct"` | vLLM model ID |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Max output tokens |

### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `"BAAI/bge-base-en-v1.5"` | HuggingFace model ID |
| `EMBEDDING_DIMENSION` | `768` | Output vector dimensions |
| `EMBEDDING_DEVICE` | `"cpu"` | `"cpu"` or `"cuda"` |

### Vector Database Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_DB_TYPE` | `"chroma"` | `"chroma"` or `"qdrant"` |
| `CHROMA_PERSIST_DIR` | `"./data/chroma_db"` | ChromaDB storage path |
| `COLLECTION_NAME` | `"documents"` | Collection name |

### Retrieval Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNKING_STRATEGY` | `"semantic"` | `"semantic"` or `"token"` |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `0.4` | Break threshold for semantic chunking |
| `SEMANTIC_MAX_CHUNK_SIZE` | `600` | Max tokens per semantic chunk |
| `TOP_K_RETRIEVAL` | `40` | Candidate pool size |
| `TOP_K_RERANK` | `6` | Final result count after reranking |
| `USE_RERANKER` | `True` | Enable/disable cross-encoder |
| `USE_HYBRID_SEARCH` | `True` | Enable/disable BM25 + vector fusion |
| `HYBRID_ALPHA` | `0.7` | Vector weight in RRF (0=keyword only, 1=vector only) |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3` | Minimum score to keep a result |
| `QUERY_REWRITE_ENABLED` | `True` | Enable/disable LLM query rewriting |
| `QUERY_REWRITE_MIN_TURNS` | `2` | Min conversation messages before rewriting |

### Cache Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_CACHE` | `True` | Enable/disable Redis cache |
| `REDIS_HOST` | `"localhost"` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `CACHE_TTL` | `3600` | Cache time-to-live (seconds) |

---

## 16. Data Schemas

### Vector DB Document (ChromaDB)

```json
{
  "id": "doc-abc123_chunk_0",
  "embedding": [0.012, -0.034, 0.089, "...768 floats"],
  "document": "The actual text content of this chunk...",
  "metadata": {
    "document_id": "abc123",
    "filename": "01_Employee_Handbook.pdf",
    "file_type": ".pdf",
    "file_size": 23000,
    "chunk_type": "text",
    "chunk_index": 0,
    "page_number": 5,
    "section": "2.2 Leave Entitlements",
    "department": "HR",
    "category": "Policy",
    "tags": "leave,hr,policy",
    "author": "HR Team",
    "version": "3.2",
    "created_at": "2026-03-15T10:30:00Z"
  }
}
```

### Conversation State (in-memory)

```json
{
  "conversation_id": "conv-uuid-456",
  "created_at": "2026-03-15T10:30:00Z",
  "updated_at": "2026-03-15T10:35:00Z",
  "messages": [
    {"role": "user", "content": "What is the leave policy?", "timestamp": "..."},
    {"role": "assistant", "content": "According to the handbook...", "timestamp": "..."},
    {"role": "user", "content": "How many days carry over?", "timestamp": "..."},
    {"role": "assistant", "content": "Up to 5 days can be carried...", "timestamp": "..."}
  ]
}
```

### Redis Cache Entry

```
Key:   rag:query:a1b2c3d4e5f6   (MD5 of question + filters)
Value: {"answer":"...", "sources":[...], "cached_at":"2026-03-15T10:30:00Z"}
TTL:   3600 seconds
```

---

## 17. API Reference

### Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send question, get answer (streaming or sync) |
| `GET` | `/api/v1/chat/conversations/{id}` | Get conversation history |
| `DELETE` | `/api/v1/chat/conversations/{id}` | Delete conversation |

**`POST /api/v1/chat` — Request:**
```json
{
  "question": "What is the annual leave entitlement?",
  "conversation_id": null,
  "stream": true,
  "filters": {"department": "HR"}
}
```

**Streaming response (SSE):**
```
data: {"type":"sources","sources":[...],"conversation_id":"uuid"}
data: {"type":"token","content":"According"}
data: {"type":"token","content":" to"}
...
data: {"type":"done"}
```

**Sync response (`stream: false`):**
```json
{
  "answer": "According to the Employee Handbook...",
  "conversation_id": "uuid-123",
  "sources": [{"content":"...","document_name":"...","page_number":5,"relevance_score":0.94}],
  "processing_time_ms": 1850
}
```

### Document Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload and ingest a document |
| `GET` | `/api/v1/documents` | List all indexed documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/documents/{id}` | Delete document and all its chunks |

**`POST /api/v1/documents/upload` — Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | Binary | Yes | PDF, DOCX, TXT, or MD file |
| `department` | String | No | Organizational department |
| `category` | String | No | Document category |
| `author` | String | No | Document author |
| `tags` | String | No | Comma-separated tags |

### Health & Metrics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Basic health check |
| `GET` | `/health/live` | Kubernetes liveness probe |
| `GET` | `/health/ready` | Readiness probe (checks LLM + VectorDB) |
| `GET` | `/health/detailed` | Full system diagnostic |
| `GET` | `/metrics` | Prometheus-format metrics |
| `GET` | `/metrics/json` | JSON metrics summary |

---

## 18. Deployment Topology

### Development (single machine)

```
+----------------------------------------------+
|           Your Development Machine           |
|                                              |
|  Frontend   (Vite dev server :3000)          |
|  Backend    (Uvicorn :8000)                  |
|  Ollama     (LLM server :11434)             |
|  ChromaDB   (embedded, ./data/chroma_db)     |
|  Redis      (optional, :6379)               |
+----------------------------------------------+
```

### Production (containerized)

```
                   Load Balancer
                        |
            +-----------+-----------+
            |                       |
     +------+------+       +-------+-----+
     | Frontend    |       | Backend (×3) |  ← horizontal scaling
     | (Nginx/S3)  |       | FastAPI      |
     +-------------+       +------+-------+
                                   |
                    +--------------+--------------+
                    |              |              |
              +-----+----+  +----+-----+  +-----+----+
              | ChromaDB  |  |  Ollama   |  |  Redis   |
              | / Qdrant  |  |  / vLLM   |  |  Cache   |
              +-----------+  +----------+  +----------+
```

### Resource Estimates

| Component | CPU | RAM | GPU | Storage |
|-----------|-----|-----|-----|---------|
| Backend (per instance) | 2 cores | 4 GB | — | — |
| Embedding model | 2 cores | 2 GB | Optional | ~500 MB |
| Reranker model | 2 cores | 2 GB | Optional | ~1 GB |
| Ollama (phi3) | 4 cores | 8 GB | Optional | ~2 GB |
| ChromaDB | 2 cores | 4 GB | — | ~1 GB per 1M chunks |
| Redis | 1 core | 1 GB | — | — |

---

## 19. Design Decisions & Trade-offs

Every technical decision in this project was made deliberately. Understanding the *why* behind each choice is what separates a junior engineer from a tutorial-follower.

| # | Decision | What We Chose | What We Didn't Choose | Why |
|---|----------|--------------|----------------------|-----|
| 1 | Chunking strategy | Semantic (embedding-based) | Fixed-size token chunking | Topic-coherent chunks improve retrieval quality. Slower ingestion is acceptable since it's offline. |
| 2 | Embedding model | BGE (open-source, self-hosted) | OpenAI ada-002, Cohere | Data privacy — no company data leaves the network. Free. Can be fine-tuned on domain data. |
| 3 | Reranking | Cross-encoder (bge-reranker) | No reranking / LLM-based reranking | +16% accuracy for +200ms latency. Best quality/cost ratio. LLM reranking would add seconds. |
| 4 | Search method | Hybrid (vector + BM25 + RRF) | Vector-only or keyword-only | Vector catches semantic matches; BM25 catches exact terms. RRF combines both simply and effectively. |
| 5 | Intent detection | Rule-based (regex) | LLM-based classification | <1ms, interpretable, no API call. Covers 90%+ of patterns. Can add LLM fallback later. |
| 6 | LLM provider | Ollama (local) | OpenAI / Anthropic cloud API | Data privacy, zero API cost, lower latency. Trade-off: lower quality than GPT-4o/Claude. |
| 7 | Error handling | Circuit breaker pattern | Simple retry or no handling | Prevents cascading failures. Essential when LLM goes down — fail fast instead of timeout cascade. |
| 8 | Response delivery | SSE streaming | WebSocket / long polling | SSE is simpler (HTTP-based, auto-reconnect). Sufficient for one-way server→client streaming. |
| 9 | Conversation storage | In-memory (OrderedDict) | Redis / PostgreSQL | Simplicity for MVP. Trade-off: data lost on restart. Clear upgrade path to Redis when needed. |
| 10 | Vector database | ChromaDB (embedded) | Qdrant / Pinecone / pgvector | No separate server needed. Good for <1M vectors. Easy to start. Qdrant when scaling is needed. |

---

## 20. What to Build Next

These are real features that mid/large companies need. Building any of these would strengthen your portfolio further.

| Priority | Feature | What It Is | Why It Matters |
|----------|---------|-----------|----------------|
| P1 | **Query Decomposition** | Split complex questions into sub-queries, retrieve for each, then synthesize | "Compare the leave policy with the travel policy" needs two separate retrievals |
| P1 | **User Feedback Loop** | Thumbs up/down per answer → use to improve retrieval weights | Shows you understand MLOps: deploy → measure → improve cycle |
| P1 | **Role-Based Access Control** | Filter retrieval by user's department/role | Security requirement in every enterprise. HR docs shouldn't appear for engineering queries. |
| P2 | **Multi-Hop Reasoning** | Follow cross-references: "See Section 4.2" → retrieve that section too | Real documents reference each other. Single-hop retrieval misses these connections. |
| P2 | **Agentic RAG** | LLM decides: search? ask for clarification? use different filters? | Next evolution beyond static pipelines. Shows you understand AI agents. |
| P2 | **Knowledge Graph** | Extract entities + relationships → supplement vector search | Better for "Who is responsible for X?" and relationship-based queries. |
| P3 | **Fine-Tuned Embeddings** | Train BGE on your domain vocabulary | 5-15% retrieval improvement for domain-specific terminology. |
| P3 | **Multilingual Support** | Vietnamese + English cross-lingual retrieval | Real need in Vietnamese companies with English documentation. |
| P3 | **Evaluation Framework** | Automated testing with labeled Q&A pairs | Impossible to improve what you can't measure. Build before optimizing. |
| P3 | **A/B Testing** | Compare retrieval strategies, prompts, models | Data-driven optimization instead of guesswork. |
