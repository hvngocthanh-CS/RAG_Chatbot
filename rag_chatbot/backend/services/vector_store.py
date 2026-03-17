"""
Vector Store Service.
Provides abstraction over different vector databases.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


# ChromaDB 0.4.x still references np.float_, which was removed in NumPy 2.x.
if not hasattr(np, "float_"):
    np.float_ = np.float64


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the vector store connection."""
        pass
    
    @abstractmethod
    async def add_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Add document chunks to the store."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        pass
    
    @abstractmethod
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 20,
        **filters
    ) -> List[Dict[str, Any]]:
        """List all documents."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store implementation.
    
    Features:
    - Persistent storage
    - Metadata filtering
    - Hybrid search support
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
    
    async def initialize(self):
        """Initialize ChromaDB client."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        # Ensure persist directory exists
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Initialize client with persistence
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"ChromaDB initialized. Collection: {settings.COLLECTION_NAME}")
    
    async def add_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Add document chunks to ChromaDB."""
        if not chunks:
            return
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{metadata['document_id']}_{i}"
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["content"])
            
            # Prepare metadata (ChromaDB requires simple types)
            chunk_meta = self._flatten_metadata(chunk["metadata"])
            chunk_meta["created_at"] = datetime.utcnow().isoformat()
            metadatas.append(chunk_meta)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )
        
        logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    
    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten metadata for ChromaDB (only supports simple types)."""
        flat = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flat[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                flat[key] = ",".join(str(v) for v in value)
            elif value is None:
                flat[key] = ""
            else:
                flat[key] = str(value)
        return flat
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in ChromaDB."""
        # Build where clause for filtering
        where = None
        if filters:
            where = self._build_where_clause(filters)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        chunks = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                chunk = {
                    "id": chunk_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                chunks.append(chunk)
        
        return chunks
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        conditions = []
        
        for key, value in filters.items():
            if value is not None and value != "":
                conditions.append({key: {"$eq": value}})
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None
    
    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Keyword-based search.
        
        Note: ChromaDB doesn't support true BM25/keyword search.
        This returns empty results - hybrid search will rely on vector search.
        For true keyword search, consider using Elasticsearch or similar.
        """
        # ChromaDB's query_texts uses its default embedding which has different dimensions
        # than our custom embeddings, so we can't use it directly.
        # Return empty list - vector search results will be used instead.
        logger.debug("Keyword search not available in ChromaDB - using vector search only")
        return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 20,
        **filters
    ) -> List[Dict[str, Any]]:
        """List all unique documents."""
        # Get all documents metadata
        where = self._build_where_clause(filters) if filters else None
        
        results = self.collection.get(
            where=where,
            include=["metadatas"]
        )
        
        # Group by document_id
        documents = {}
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                doc_id = meta.get("document_id")
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "id": doc_id,
                        "filename": meta.get("filename", ""),
                        "file_type": meta.get("file_type", ""),
                        "file_size": int(meta.get("file_size", 0)),
                        "upload_date": meta.get("created_at", ""),
                        "chunks_count": 0,
                        "status": "indexed",
                        "department": meta.get("department"),
                        "tags": meta.get("tags", "").split(",") if meta.get("tags") else []
                    }
                if doc_id:
                    documents[doc_id]["chunks_count"] += 1
        
        # Apply pagination
        doc_list = list(documents.values())
        return doc_list[skip:skip + limit]
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document."""
        results = self.collection.get(
            where={"document_id": {"$eq": document_id}},
            include=["metadatas"],
            limit=1
        )
        
        if results and results["metadatas"]:
            meta = results["metadatas"][0]
            
            # Count total chunks
            all_results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            chunks_count = len(all_results["ids"]) if all_results else 0
            
            return {
                "id": document_id,
                "filename": meta.get("filename", ""),
                "file_type": meta.get("file_type", ""),
                "file_size": int(meta.get("file_size", 0)),
                "upload_date": meta.get("created_at", ""),
                "chunks_count": chunks_count,
                "status": "indexed",
                "department": meta.get("department"),
                "tags": meta.get("tags", "").split(",") if meta.get("tags") else []
            }
        
        return None
    
    async def health_check(self) -> bool:
        """Check if ChromaDB is healthy."""
        try:
            self.collection.count()
            return True
        except Exception:
            return False


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store implementation.
    
    Features:
    - Cloud and self-hosted support
    - Advanced filtering
    - Payload storage
    """
    
    def __init__(self):
        self.client = None
    
    async def initialize(self):
        """Initialize Qdrant client."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        # Connect to Qdrant
        if settings.QDRANT_API_KEY:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY,
                https=True
            )
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )
        
        # Create collection if not exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if settings.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {settings.COLLECTION_NAME}")
        
        logger.info(f"Qdrant initialized. Collection: {settings.COLLECTION_NAME}")
    
    async def add_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Add document chunks to Qdrant."""
        from qdrant_client.models import PointStruct
        
        if not chunks:
            return
        
        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            
            payload = {
                **chunk["metadata"],
                "content": chunk["content"],
                "created_at": datetime.utcnow().isoformat()
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload
            ))
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points[i:i + batch_size]
            )
        
        logger.info(f"Added {len(chunks)} chunks to Qdrant")
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build filter
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if conditions:
                query_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )
        
        chunks = []
        for result in results:
            chunk = {
                "id": str(result.id),
                "content": result.payload.get("content", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "content"},
                "score": result.score
            }
            chunks.append(chunk)
        
        return chunks
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        try:
            self.client.delete(
                collection_name=settings.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 20,
        **filters
    ) -> List[Dict[str, Any]]:
        """List all unique documents."""
        # Scroll through all points to get unique documents
        results, _ = self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=1000,  # Get all
            with_payload=True
        )
        
        documents = {}
        for point in results:
            doc_id = point.payload.get("document_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "filename": point.payload.get("filename", ""),
                    "file_type": point.payload.get("file_type", ""),
                    "file_size": int(point.payload.get("file_size", 0)),
                    "upload_date": point.payload.get("created_at", ""),
                    "chunks_count": 0,
                    "status": "indexed",
                    "department": point.payload.get("department"),
                    "tags": point.payload.get("tags", [])
                }
            if doc_id:
                documents[doc_id]["chunks_count"] += 1
        
        doc_list = list(documents.values())
        return doc_list[skip:skip + limit]
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        results, _ = self.client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        
        if results:
            point = results[0]
            
            # Count total chunks
            all_results, _ = self.client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=10000
            )
            
            return {
                "id": document_id,
                "filename": point.payload.get("filename", ""),
                "file_type": point.payload.get("file_type", ""),
                "file_size": int(point.payload.get("file_size", 0)),
                "upload_date": point.payload.get("created_at", ""),
                "chunks_count": len(all_results),
                "status": "indexed",
                "department": point.payload.get("department"),
                "tags": point.payload.get("tags", [])
            }
        
        return None
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            self.client.get_collection(settings.COLLECTION_NAME)
            return True
        except Exception:
            return False


class VectorStoreService:
    """
    Factory service for vector store operations.
    Automatically selects the configured vector store implementation.
    """
    
    def __init__(self):
        if settings.VECTOR_DB_TYPE == "chroma":
            self._store = ChromaVectorStore()
        else:
            self._store = QdrantVectorStore()
    
    async def initialize(self):
        """Initialize the vector store."""
        await self._store.initialize()
    
    async def add_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Add chunks to the vector store."""
        await self._store.add_chunks(chunks, metadata)
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        return await self._store.search(query_embedding, top_k, filters)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document."""
        return await self._store.delete_document(document_id)
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 20,
        **filters
    ) -> List[Dict[str, Any]]:
        """List documents."""
        return await self._store.list_documents(skip, limit, **filters)
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata."""
        return await self._store.get_document(document_id)
    
    async def health_check(self) -> bool:
        """Check vector store health."""
        return await self._store.health_check()
