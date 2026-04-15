"""
Retrieval Service — Step 6 of the RAG pipeline.

Implements the retrieval pipeline:
  1. Conversational query rewriting (follow-up -> standalone query)
  2. Vector search (semantic similarity)
  3. Keyword search + RRF fusion (if hybrid enabled)
  4. Score threshold filtering (after fusion, on unified scores)
  5. Cross-encoder reranking
  6. Return top-k results
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.services import get_service
from .reranker import RerankerService
from .query_rewriter import QueryRewriterService

logger = logging.getLogger(__name__)


class RetrievalService:
    """Retrieval service with hybrid search and reranking."""

    def __init__(self):
        self.reranker = RerankerService() if settings.USE_RERANKER else None
        self.query_rewriter = QueryRewriterService()

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
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query."""
        top_k = top_k or settings.TOP_K_RERANK
        retrieval_k = settings.TOP_K_RETRIEVAL

        # Step 1: Conversational query rewriting
        retrieval_query = query
        if (
            settings.QUERY_REWRITE_ENABLED
            and conversation_history
            and len(conversation_history) >= settings.QUERY_REWRITE_MIN_TURNS
        ):
            retrieval_query = await self.query_rewriter.rewrite(
                question=query,
                conversation_history=conversation_history,
            )
            if retrieval_query != query:
                logger.info("Query rewritten: [%s] -> [%s]",
                            query, retrieval_query)

        # Step 2: Vector search (semantic similarity)
        query_embedding = await self.embedding_service.embed_query(retrieval_query)
        search_filters = filters or {}

        # Step 3: Run vector + keyword search in parallel when hybrid is on
        if settings.USE_HYBRID_SEARCH:
            vector_results, keyword_results = await asyncio.gather(
                self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=retrieval_k,
                    filters=search_filters,
                ),
                self.vector_store.keyword_search(
                    query=retrieval_query,
                    top_k=retrieval_k,
                    filters=search_filters,
                ),
            )
            combined_results = self._reciprocal_rank_fusion(
                vector_results, keyword_results,
                alpha=settings.HYBRID_ALPHA,
            )
        else:
            combined_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=retrieval_k,
                filters=search_filters,
            )

        # Step 4: Score threshold filtering (AFTER fusion)
        if combined_results:
            top_score = combined_results[0]["score"]
            if settings.USE_HYBRID_SEARCH and top_score < 0.1:
                min_score = top_score * 0.2
            else:
                min_score = settings.RETRIEVAL_SCORE_THRESHOLD

            before = len(combined_results)
            combined_results = [
                r for r in combined_results if r["score"] >= min_score
            ]
            filtered = before - len(combined_results)
            if filtered:
                logger.info("Score filter removed %d/%d chunks (min=%.4f)",
                            filtered, before, min_score)

        if not combined_results:
            logger.warning("No results found in retrieval")
            return []

        # Step 5: Cross-encoder reranking
        if self.reranker and settings.USE_RERANKER:
            rerank_candidates = combined_results[: int(top_k * 2.5)]
            logger.info("Reranking %d candidates...", len(rerank_candidates))
            return await self.reranker.rerank(
                query=retrieval_query,
                chunks=rerank_candidates,
                top_k=top_k,
            )

        return combined_results[:top_k]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self, vector_results, keyword_results, alpha=0.5, k=60,
    ):
        """
        Combine vector and keyword results using Reciprocal Rank Fusion.

        RRF score = alpha / (k + rank_vector) + (1-alpha) / (k + rank_keyword)
        """
        scores = {}
        chunks = {}

        for rank, result in enumerate(vector_results):
            cid = result["id"]
            scores[cid] = scores.get(cid, 0) + alpha / (k + rank + 1)
            chunks[cid] = result

        for rank, result in enumerate(keyword_results):
            cid = result["id"]
            scores[cid] = scores.get(cid, 0) + (1 - alpha) / (k + rank + 1)
            if cid not in chunks:
                chunks[cid] = result

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [{**chunks[cid], "score": scores[cid]} for cid in sorted_ids]
