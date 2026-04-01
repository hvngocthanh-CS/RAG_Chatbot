"""
Retrieval Service — Step 6 of the RAG pipeline.

Implements the full retrieval pipeline:
  0. Conversational query rewriting (follow-up → standalone query)
  1. Query understanding (detect intent: policy, procedure, contact, ...)
  2. Query expansion based on intent
  3. Enhance query with conversation context
  4. Vector search (semantic similarity)
  5. Score threshold filtering
  6. Keyword search + RRF fusion (if hybrid enabled)
  7. Intent-based metadata boosting
  8. Cross-encoder reranking
  9. Return top-k results
"""
import logging
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.services import get_service
from backend.services.reranker import RerankerService
from backend.services.query_understanding import QueryUnderstandingService
from backend.services.query_rewriter import QueryRewriterService

logger = logging.getLogger(__name__)


class RetrievalService:
    """Advanced retrieval service with hybrid search and reranking."""

    def __init__(self):
        self.reranker = RerankerService() if settings.USE_RERANKER else None
        self.query_understanding = QueryUnderstandingService()
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

        # Step 0: Conversational query rewriting
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

        # Step 1: Query understanding
        intent = self.query_understanding.analyze_query(query)
        logger.info(
            "intent=%s confidence=%.2f rewrite=%s",
            intent.intent_type, intent.confidence,
            "yes" if retrieval_query != query else "no",
        )

        # Step 2: Prepare query variants
        embedding_query = self._enhance_query(retrieval_query, conversation_history)
        keyword_query = self.query_understanding.expand_query(retrieval_query, intent)

        # Step 3: Vector search
        query_embedding = await self.embedding_service.embed_query(embedding_query)

        search_filters = filters or {}
        fetch_k = retrieval_k * 2 if intent.page_filter else retrieval_k

        vector_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            filters=search_filters,
        )

        # Step 4: Score threshold filtering
        before = len(vector_results)
        vector_results = [
            r for r in vector_results
            if r.get("score", 0) >= settings.RETRIEVAL_SCORE_THRESHOLD
        ]
        if before != len(vector_results):
            logger.info(
                "Score threshold (%.2f) filtered %d/%d chunks",
                settings.RETRIEVAL_SCORE_THRESHOLD,
                before - len(vector_results), before,
            )

        # Step 5: Intent-based boosting
        if intent.page_filter or intent.boost_metadata:
            vector_results = self._apply_intent_boosting(vector_results, intent)

        # Step 6: Hybrid search (keyword + RRF fusion)
        if settings.USE_HYBRID_SEARCH:
            keyword_results = await self.vector_store.keyword_search(
                query=keyword_query,
                top_k=retrieval_k,
                filters=search_filters,
            )
            combined_results = self._reciprocal_rank_fusion(
                vector_results, keyword_results,
                alpha=settings.HYBRID_ALPHA,
            )
        else:
            combined_results = vector_results

        if not combined_results:
            logger.warning("No results found in retrieval")
            return []

        # Step 7: Reranking
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
    # Intent boosting
    # ------------------------------------------------------------------

    def _apply_intent_boosting(self, results, intent):
        if not results:
            return results

        for chunk in results:
            score = chunk.get("score", 0.0)
            meta = chunk.get("metadata", {})
            page_num = meta.get("page_number")
            chunk_type = meta.get("chunk_type", "text")

            if intent.boost_metadata and intent.boost_metadata.get("table"):
                if chunk_type in ("table", "table_rows"):
                    score *= intent.boost_metadata.get("table", 5.0)
            elif intent.page_filter and page_num in intent.page_filter:
                factor = intent.boost_metadata.get("page_number", 5.0) if intent.boost_metadata else 5.0
                score *= factor

            chunk["score"] = score

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Query enhancement
    # ------------------------------------------------------------------

    def _enhance_query(self, query, conversation_history):
        if not conversation_history:
            return query

        last_user = None
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                last_user = turn.get("content", "").strip()
                break

        if not last_user or last_user.lower()[:80] == query.lower()[:80]:
            return query

        return f"Context: {last_user[:120]} | Question: {query}"

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(self, vector_results, keyword_results, alpha=0.5, k=60):
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
