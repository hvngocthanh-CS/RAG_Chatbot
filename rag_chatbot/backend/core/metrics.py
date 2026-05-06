"""
Prometheus metrics for the RAG pipeline.

HTTP request metrics are handled automatically by prometheus-fastapi-instrumentator.
This module defines custom business-level metrics for the RAG pipeline.
"""
from prometheus_client import Counter, Histogram, Gauge

# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------

RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "End-to-end retrieval pipeline latency (embed + search + rerank)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RERANKER_DURATION = Histogram(
    "rag_reranker_duration_seconds",
    "Cross-encoder reranking latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks_count",
    "Number of chunks returned after retrieval+rerank",
    buckets=[1, 3, 5, 7, 10, 15, 20, 30, 50, 70],
)

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

LLM_DURATION = Histogram(
    "rag_llm_generation_duration_seconds",
    "LLM generation latency (non-streaming)",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Responses served from Redis cache",
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Cache misses — pipeline executed",
)

# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

DOCUMENTS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Documents successfully ingested",
    ["file_type"],
)

DOCUMENTS_DELETED = Counter(
    "rag_documents_deleted_total",
    "Documents deleted from the vector store",
)

INGESTION_DURATION = Histogram(
    "rag_ingestion_duration_seconds",
    "Document ingestion latency (parse + chunk + embed + upsert)",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

ACTIVE_CONVERSATIONS = Gauge(
    "rag_active_conversations",
    "In-memory conversations currently tracked",
)
