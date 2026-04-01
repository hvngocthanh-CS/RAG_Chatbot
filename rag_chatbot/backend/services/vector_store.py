"""
Vector Store Service — Step 5 of the RAG pipeline.

Stores chunk embeddings in Qdrant and provides semantic + keyword search.

Design decisions:
  - Qdrant as the single vector DB (production-grade, rich filtering).
  - BM25 keyword search built on top of scrolled corpus for hybrid retrieval.
  - Cosine distance for similarity (matches BGE normalized embeddings).
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 keyword search helper
# ---------------------------------------------------------------------------

def _bm25_search(
    ids: List[str],
    docs: List[str],
    metas: List[Dict[str, Any]],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run BM25 keyword scoring over an in-memory corpus."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed — keyword search disabled")
        return []

    if not docs:
        return []

    tokenised = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenised)
    scores = bm25.get_scores(query.lower().split())

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {"id": ids[idx], "content": docs[idx], "metadata": metas[idx], "score": float(scores[idx])}
        for idx in top_indices
        if scores[idx] > 0
    ]


# ---------------------------------------------------------------------------
# Qdrant Vector Store
# ---------------------------------------------------------------------------

class VectorStoreService:
    """
    Qdrant-backed vector store with semantic + BM25 keyword search.

    Initialized once at startup via the service registry.
    """

    def __init__(self):
        self.client = None

    async def initialize(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if settings.QDRANT_API_KEY:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY,
                https=True,
            )
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )

        collections = [c.name for c in self.client.get_collections().collections]
        if settings.COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", settings.COLLECTION_NAME)

        logger.info("Qdrant initialized. Collection: %s", settings.COLLECTION_NAME)

    # --- write ---

    async def add_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        from qdrant_client.models import PointStruct

        if not chunks:
            return

        points = []
        for chunk in chunks:
            payload = {
                **chunk["metadata"],
                "content": chunk["content"],
                "created_at": datetime.utcnow().isoformat(),
            }
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload=payload,
            ))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points[i : i + batch_size],
            )

        logger.info("Added %d chunks to Qdrant", len(chunks))

    # --- semantic search ---

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )

        return [
            {
                "id": str(r.id),
                "content": r.payload.get("content", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "content"},
                "score": r.score,
            }
            for r in results
        ]

    # --- keyword search (BM25 over scrolled corpus) ---

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        records, _ = self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=self._build_filter(filters),
            limit=10_000,
            with_payload=True,
        )

        if not records:
            return []

        ids = [str(r.id) for r in records]
        docs = [r.payload.get("content", "") for r in records]
        metas = [{k: v for k, v in r.payload.items() if k != "content"} for r in records]

        return _bm25_search(ids, docs, metas, query, top_k)

    # --- delete ---

    async def delete_document(self, document_id: str) -> bool:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            self.client.delete(
                collection_name=settings.COLLECTION_NAME,
                points_selector=Filter(must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                ]),
            )
            logger.info("Deleted chunks for document %s", document_id)
            return True
        except Exception as e:
            logger.error("Error deleting document %s: %s", document_id, e)
            return False

    # --- list / get ---

    async def list_documents(self, skip: int = 0, limit: int = 20, **filters) -> List[Dict[str, Any]]:
        records, _ = self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=10_000,
            with_payload=True,
        )

        documents: Dict[str, Dict[str, Any]] = {}
        for point in records:
            doc_id = point.payload.get("document_id")
            if not doc_id:
                continue
            if doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "filename": point.payload.get("filename", ""),
                    "file_type": point.payload.get("file_type", ""),
                    "file_size": int(point.payload.get("file_size", 0)),
                    "upload_date": point.payload.get("created_at", ""),
                    "chunks_count": 0,
                    "status": "indexed",
                    "department": point.payload.get("department"),
                    "tags": point.payload.get("tags", []),
                }
            documents[doc_id]["chunks_count"] += 1

        return list(documents.values())[skip : skip + limit]

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        records, _ = self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            ]),
            limit=10_000,
            with_payload=True,
        )

        if not records:
            return None

        first = records[0]
        return {
            "id": document_id,
            "filename": first.payload.get("filename", ""),
            "file_type": first.payload.get("file_type", ""),
            "file_size": int(first.payload.get("file_size", 0)),
            "upload_date": first.payload.get("created_at", ""),
            "chunks_count": len(records),
            "status": "indexed",
            "department": first.payload.get("department"),
            "tags": first.payload.get("tags", []),
        }

    # --- health ---

    async def health_check(self) -> bool:
        try:
            self.client.get_collection(settings.COLLECTION_NAME)
            return True
        except Exception:
            return False

    # --- filter builder ---

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, Any]]):
        if not filters:
            return None

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = []
        for key, value in filters.items():
            if value is not None and value != "":
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions) if conditions else None
