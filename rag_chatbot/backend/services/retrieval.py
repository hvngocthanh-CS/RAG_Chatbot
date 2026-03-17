"""
Retrieval Service.
Implements the full retrieval pipeline with hybrid search and reranking.
"""
import logging
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.services import get_service
from backend.services.vector_store import ChromaVectorStore
from backend.services.reranker import RerankerService
from backend.services.query_understanding import QueryUnderstandingService

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Advanced retrieval service with query understanding, hybrid search, and reranking.
    
    Pipeline:
    1. Query understanding (detect intent: title, author, etc.)
    2. Query expansion based on intent
    3. Vector search (semantic similarity)
    4. Keyword search (if hybrid enabled)
    5. Page/metadata boosting based on intent
    6. Apply reranking model
    7. Return top-k results
    """
    
    def __init__(self):
        self.reranker = RerankerService() if settings.USE_RERANKER else None
        self.query_understanding = QueryUnderstandingService()
    
    @property
    def embedding_service(self):
        return get_service("embedding")
    
    @property
    def vector_store(self):
        return get_service("vector_store")
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User's question
            filters: Metadata filters (document_id, department, etc.)
            conversation_history: Previous conversation for context
            top_k: Number of results to return
        
        Returns:
            List of relevant chunks with scores
        """
        top_k = top_k or settings.TOP_K_RERANK
        retrieval_k = settings.TOP_K_RETRIEVAL
        
        # Step 0: Query understanding
        intent = self.query_understanding.analyze_query(query)
        logger.info(f"Detected intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
        
        # Expand query based on intent
        expanded_query = self.query_understanding.expand_query(query, intent)
        
        # Optionally enhance query with conversation context
        enhanced_query = self._enhance_query(expanded_query, conversation_history)
        
        # Step 1: Vector search
        logger.info(f"Performing vector search for: {query[:50]}...")
        query_embedding = await self.embedding_service.embed_query(enhanced_query)
        
        # Apply intent-based filters (e.g., page 1 for title questions)
        search_filters = filters or {}
        if intent.page_filter:
            # Retrieve more results initially, then filter/boost by page
            vector_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=retrieval_k * 2,  # Get more to ensure page 1 chunks
                filters=search_filters
            )
        else:
            vector_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=retrieval_k,
                filters=search_filters
            )
        
        # Step 2: Apply intent-based boosting
        if intent.page_filter or intent.boost_metadata:
            vector_results = self._apply_intent_boosting(vector_results, intent)
        
        # Step 3: Keyword search (if hybrid enabled)
        if settings.USE_HYBRID_SEARCH:
            logger.info("Performing keyword search...")
            keyword_results = await self._keyword_search(enhanced_query, retrieval_k, search_filters)
            
            # Combine results using Reciprocal Rank Fusion
            combined_results = self._reciprocal_rank_fusion(
                vector_results,
                keyword_results,
                alpha=settings.HYBRID_ALPHA
            )
        else:
            combined_results = vector_results
        
        if not combined_results:
            logger.warning("No results found in retrieval")
            return []
        
        # Step 4: Reranking
        if self.reranker and settings.USE_RERANKER:
            logger.info(f"Reranking {len(combined_results)} results...")
            reranked_results = await self.reranker.rerank(
                query=query,
                chunks=combined_results,
                top_k=top_k
            )
            return reranked_results
        
        # Return top-k without reranking
        return combined_results[:top_k]
    
    def _apply_intent_boosting(
        self,
        results: List[Dict[str, Any]],
        intent
    ) -> List[Dict[str, Any]]:
        """
        Boost chunks based on query intent.
        
        For example:
        - Title questions → Strongly boost page 1 chunks
        - Author questions → Boost page 1 chunks
        - Technical questions → No boosting
        """
        if not results:
            return results
        
        boosted_results = []
        
        for chunk in results:
            score = chunk.get("score", 0.0)
            metadata = chunk.get("metadata", {})
            page_num = metadata.get("page_number")
            
            # Apply page-based boosting
            if intent.page_filter and page_num in intent.page_filter:
                # Strongly boost pages in filter
                boost_factor = intent.boost_metadata.get("page_number", 5.0) if intent.boost_metadata else 5.0
                score *= boost_factor
                logger.debug(f"Boosted page {page_num} chunk by {boost_factor}x (intent: {intent.intent_type})")
            
            chunk["score"] = score
            boosted_results.append(chunk)
        
        # Re-sort by boosted scores
        boosted_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        return boosted_results
    
    def _enhance_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Enhance query with conversation context for better retrieval.
        """
        if not conversation_history:
            return query
        
        # Include last few exchanges for context
        context_turns = conversation_history[-4:]  # Last 2 exchanges
        
        if not context_turns:
            return query
        
        # Build context string
        context_parts = []
        for turn in context_turns:
            role = turn.get("role", "")
            content = turn.get("content", "")[:200]  # Limit length
            if role == "user":
                context_parts.append(f"Previous question: {content}")
            elif role == "assistant":
                context_parts.append(f"Previous answer: {content}")
        
        # Combine context with current query
        enhanced = f"{' '.join(context_parts)} Current question: {query}"
        
        return enhanced
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        """
        # Use ChromaDB's built-in keyword search if available
        if isinstance(self.vector_store, ChromaVectorStore):
            return await self.vector_store.keyword_search(
                query=query,
                top_k=top_k,
                filters=filters
            )
        
        # Fallback: simple keyword matching via vector search
        # (This is a simplified approach - in production, you might use BM25 or Elasticsearch)
        return []
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        alpha: float = 0.5,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank))
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            alpha: Weight for vector results (1-alpha for keyword)
            k: RRF constant (typically 60)
        
        Returns:
            Combined and ranked results
        """
        scores = {}
        chunks = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result["id"]
            rrf_score = alpha / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            chunks[chunk_id] = result
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            chunk_id = result["id"]
            rrf_score = (1 - alpha) / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in chunks:
                chunks[chunk_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build result list
        combined = []
        for chunk_id in sorted_ids:
            result = chunks[chunk_id].copy()
            result["score"] = scores[chunk_id]
            combined.append(result)
        
        return combined
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.
        """
        # Implementation depends on vector store
        pass
