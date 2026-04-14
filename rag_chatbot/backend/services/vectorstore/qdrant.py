"""
Vector Store Service — Step 5 of the RAG pipeline.

Stores chunk embeddings in Qdrant and provides semantic + keyword search.

Design decisions:
  - Qdrant as the single vector DB (production-grade, rich filtering).
  - BM25 keyword index is built once and cached in memory, rebuilt only
    when documents are added or deleted (not on every query).
  - Cosine distance for similarity (matches BGE normalized embeddings).
  - All Qdrant calls are offloaded to threads via asyncio.to_thread()
    so the FastAPI event loop is never blocked.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
)

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached BM25 Index
# ---------------------------------------------------------------------------

class _BM25Index:
    """
    In-memory BM25 index that is built once and reused across queries.

    Rebuilt only when documents change (add/delete), not on every search.
    This avoids the O(N) scroll + tokenize cost on each keyword query.
    """

    def __init__(self):
        self._bm25 = None
        self._ids: List[str] = []
        self._docs: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._dirty = True  # needs rebuild

    def mark_dirty(self):
        """Mark index as stale — will be rebuilt on next search."""
        self._dirty = True

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and not self._dirty

    def build(self, ids: List[str], docs: List[str], metas: List[Dict[str, Any]]):
        """Build/rebuild the BM25 index from a corpus."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — keyword search disabled")
            return

        self._ids = ids
        self._docs = docs
        self._metas = metas

        if not docs:
            self._bm25 = None
            self._dirty = False
            return

        tokenised = [doc.lower().split() for doc in docs]
        self._bm25 = BM25Okapi(tokenised)
        self._dirty = False
        logger.info("BM25 index built: %d documents", len(docs))

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search the cached BM25 index. Returns empty list if not ready."""
        if self._bm25 is None or not self._ids:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            {
                "id": self._ids[idx],
                "content": self._docs[idx],
                "metadata": self._metas[idx],
                "score": float(scores[idx]),
            }
            for idx in top_indices
            if scores[idx] > 0
        ]


# ---------------------------------------------------------------------------
# Qdrant Vector Store
# ---------------------------------------------------------------------------

class VectorStoreService:
    """
    Qdrant-backed vector store with semantic + cached BM25 keyword search.

    All sync Qdrant calls are wrapped in asyncio.to_thread() to avoid
    blocking the event loop.
    """

    def __init__(self):
        self.client = None
        self._bm25_index = _BM25Index()

    async def initialize(self):
        def _init_sync():
            if settings.QDRANT_API_KEY:
                client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    api_key=settings.QDRANT_API_KEY,
                    https=True,
                )
            else:
                client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                )

            collections = [c.name for c in client.get_collections().collections]
            if settings.COLLECTION_NAME not in collections:
                client.create_collection(
                    collection_name=settings.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", settings.COLLECTION_NAME)
            return client

        self.client = await asyncio.to_thread(_init_sync)
        logger.info("Qdrant initialized. Collection: %s", settings.COLLECTION_NAME)

        # Build initial BM25 index from existing data
        await self._rebuild_bm25_index()

    # --- BM25 index management ---

    async def _rebuild_bm25_index(self):
        """Scroll all documents from Qdrant and rebuild the BM25 index."""
        def _scroll_and_build():
            all_records = []
            offset = None
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    limit=1000,
                    with_payload=True,
                    offset=offset,
                )
                all_records.extend(records)
                if next_offset is None:
                    break
                offset = next_offset

            if not all_records:
                self._bm25_index.build([], [], [])
                return

            ids = [str(r.id) for r in all_records]
            docs = [r.payload.get("content", "") for r in all_records]
            metas = [
                {k: v for k, v in r.payload.items() if k != "content"}
                for r in all_records
            ]
            self._bm25_index.build(ids, docs, metas)

        await asyncio.to_thread(_scroll_and_build)

    # --- write ---

    async def add_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        if not chunks:
            return

        points = []
        for chunk in chunks:
            payload = {
                **chunk["metadata"],
                "content": chunk["content"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload=payload,
            ))

        def _upsert_sync():
            batch_size = 100
            for i in range(0, len(points), batch_size):
                self.client.upsert(
                    collection_name=settings.COLLECTION_NAME,
                    points=points[i : i + batch_size],
                )

        await asyncio.to_thread(_upsert_sync)
        logger.info("Added %d chunks to Qdrant", len(chunks))

        # Rebuild BM25 index to include new data
        await self._rebuild_bm25_index()

    # --- semantic search ---

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        qf = self._build_filter(filters)

        def _search_sync():
            return self.client.search(
                collection_name=settings.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qf,
                with_payload=True,
            )

        results = await asyncio.to_thread(_search_sync)

        return [
            {
                "id": str(r.id),
                "content": r.payload.get("content", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "content"},
                "score": r.score,
            }
            for r in results
        ]

    # --- keyword search (cached BM25) ---

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Rebuild if index is stale (e.g. after external changes)
        if not self._bm25_index.is_ready:
            await self._rebuild_bm25_index()

        # BM25 search is CPU-bound, offload to thread
        return await asyncio.to_thread(self._bm25_index.search, query, top_k)

    # --- delete ---

    async def delete_document(self, document_id: str) -> bool:
        try:
            f = Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            ])

            def _delete_sync():
                self.client.delete(
                    collection_name=settings.COLLECTION_NAME,
                    points_selector=f,
                )

            await asyncio.to_thread(_delete_sync)
            logger.info("Deleted chunks for document %s", document_id)

            # Rebuild BM25 index after deletion
            await self._rebuild_bm25_index()
            return True
        except Exception as e:
            logger.error("Error deleting document %s: %s", document_id, e)
            return False

    # --- list / get ---

    async def list_documents(self, skip: int = 0, limit: int = 20, **filters) -> List[Dict[str, Any]]:
        def _scroll_sync():
            all_records = []
            offset = None
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    limit=1000,
                    with_payload=["document_id", "filename", "file_type",
                                  "file_size", "created_at", "department", "tags"],
                    offset=offset,
                )
                all_records.extend(records)
                if next_offset is None:
                    break
                offset = next_offset
            return all_records

        records = await asyncio.to_thread(_scroll_sync)

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
        f = Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
        ])

        def _scroll_sync():
            records, _ = self.client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=f,
                limit=10_000,
                with_payload=True,
            )
            return records

        records = await asyncio.to_thread(_scroll_sync)

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
            await asyncio.to_thread(
                self.client.get_collection, settings.COLLECTION_NAME
            )
            return True
        except Exception:
            return False

    # --- filter builder ---

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, Any]]):
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            # Skip None, empty strings, empty dicts, empty lists
            if value is None or value == "" or (isinstance(value, (dict, list)) and not value):
                continue
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions) if conditions else None
