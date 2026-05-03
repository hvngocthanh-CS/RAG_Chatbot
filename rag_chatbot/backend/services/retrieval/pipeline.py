"""
Retrieval Service — Step 6 of the RAG pipeline.

Two-mode retrieval depending on question complexity:

  Simple query (single intent):
    1. Conversational query rewriting
    2. Paraphrase expansion (N synonymous phrasings)
    3. Embed + hybrid search per variant (parallel)
    4. RRF fusion per variant → merge across variants
    5. Score threshold → cross-encoder rerank → top 7

  Complex query (multi-part / decomposed):
    1. Conversational query rewriting
    2. Sub-question decomposition (one atomic question per aspect)
    3. Embed + hybrid search per sub-question (parallel)
    4. RRF fusion per sub-question → merge across sub-questions
    5. Score threshold → cross-encoder rerank → top 20
       with a lower relevance gate (0.20 vs 0.35) so that chunks
       covering a minority sub-question are not dropped before the LLM.

Hybrid alpha is reduced from 0.70 to 0.55 for legal/compliance queries
where keyword precision (statute numbers, deadlines, article refs) is
more discriminating than semantic similarity alone.

The reranker always uses the original rewritten query — not paraphrases
or sub-questions — as its reference, so it scores by true user intent.
"""
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.services import get_service
from .reranker import RerankerService
from .query_rewriter import QueryRewriterService
from .query_expander import QueryExpanderService

logger = logging.getLogger(__name__)

# Legal/compliance terms that warrant higher BM25 weight in hybrid search
_LEGAL_SIGNALS = frozenset({
    "regulation", "comply", "compliance", "breach", "notification",
    "timeline", "deadline", "erasure", "gdpr", "pdpl", "article",
    "data", "subject", "obligation", "penalty", "statute", "clause",
    "provision", "enforcement", "violation", "disclosure", "retention",
    "controller", "processor", "supervisory", "authority", "consent",
})


def _is_legal_query(query: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", query.lower()))
    return bool(tokens & _LEGAL_SIGNALS)


class RetrievalService:
    """Retrieval service with adaptive hybrid search and cross-encoder reranking."""

    def __init__(self):
        self.reranker = RerankerService() if settings.USE_RERANKER else None
        self.query_rewriter = QueryRewriterService()
        self.query_expander = QueryExpanderService() if settings.MULTI_QUERY_ENABLED else None

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
    ) -> Dict[str, Any]:
        """
        Retrieve relevant document chunks for a query.

        Returns:
          {
            "chunks":        list of retrieved/reranked chunk dicts,
            "sub_questions": list of decomposed sub-questions (empty for simple queries),
          }

        Callers can use sub_questions to build a structured LLM prompt that
        explicitly addresses each aspect of a complex question.
        """
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
                logger.info("Query rewritten: [%s] -> [%s]", query, retrieval_query)

        # Step 2: Expansion — paraphrase (simple) or decomposition (complex).
        # ExpansionResult.is_decomposed tells us which mode was used so we can
        # adapt top_k and the reranker threshold in the steps below.
        all_queries: List[str] = [retrieval_query]
        is_decomposed = False
        sub_questions: List[str] = []

        if self.query_expander:
            expansion = await self.query_expander.expand(
                retrieval_query, n=settings.MULTI_QUERY_COUNT
            )
            if expansion.queries:
                all_queries.extend(expansion.queries)
                is_decomposed = expansion.is_decomposed
                if is_decomposed:
                    sub_questions = expansion.queries
                    logger.info(
                        "Decomposed into %d sub-questions for parallel retrieval",
                        len(sub_questions),
                    )
                else:
                    logger.info("Multi-query: %d paraphrase variants total", len(all_queries))

        # Adaptive parameters based on query complexity and domain
        effective_top_k = top_k or (
            settings.TOP_K_RERANK_COMPLEX if is_decomposed else settings.TOP_K_RERANK
        )
        reranker_threshold = (
            settings.RERANKER_SCORE_THRESHOLD_COMPLEX if is_decomposed
            else settings.RERANKER_SCORE_THRESHOLD
        )
        # Legal queries benefit from more keyword weight — statute numbers,
        # deadlines, and article references are exact-match signals that
        # dense embeddings routinely miss.
        hybrid_alpha = (
            settings.HYBRID_ALPHA_LEGAL
            if _is_legal_query(retrieval_query)
            else settings.HYBRID_ALPHA
        )

        search_filters = filters or {}

        # Step 3: Embed all queries in parallel, then search for each.
        query_embeddings: List = await asyncio.gather(
            *[self.embedding_service.embed_query(q) for q in all_queries]
        )

        # Step 3b: Keyword fallback — probe top-1 dense score for the original
        # query.  A low score means the corpus has weak semantic coverage
        # (e.g., query contains codes, numbers, or rare terms) so BM25 should
        # carry more weight.  Alpha is capped at HYBRID_ALPHA_FALLBACK (0.50).
        if settings.USE_HYBRID_SEARCH:
            _probe = await self.vector_store.search(
                query_embedding=query_embeddings[0], top_k=1, filters=search_filters,
            )
            _dense_top = _probe[0]["score"] if _probe else 0.0
            if _dense_top < settings.HYBRID_DENSE_FALLBACK_THRESHOLD:
                _fallback_alpha = min(hybrid_alpha, settings.HYBRID_ALPHA_FALLBACK)
                if _fallback_alpha < hybrid_alpha:
                    logger.info(
                        "Low dense confidence (%.3f) — boosting keyword weight α %.2f → %.2f",
                        _dense_top, hybrid_alpha, _fallback_alpha,
                    )
                    hybrid_alpha = _fallback_alpha

        if settings.USE_HYBRID_SEARCH:
            # Interleave (vector_search, keyword_search) tasks for every query
            # so asyncio.gather fires all of them in one shot.
            search_tasks = []
            for emb, q in zip(query_embeddings, all_queries):
                search_tasks.append(
                    self.vector_store.search(
                        query_embedding=emb, top_k=retrieval_k, filters=search_filters,
                    )
                )
                search_tasks.append(
                    self.vector_store.keyword_search(
                        query=q, top_k=retrieval_k, filters=search_filters,
                    )
                )
            raw = await asyncio.gather(*search_tasks)

            # Pair (vector, keyword) and fuse each pair with RRF.
            per_query_fused = [
                self._reciprocal_rank_fusion(raw[i], raw[i + 1], alpha=hybrid_alpha)
                for i in range(0, len(raw), 2)
            ]
            combined_results = self._merge_results(per_query_fused)
        else:
            search_tasks = [
                self.vector_store.search(
                    query_embedding=emb, top_k=retrieval_k, filters=search_filters,
                )
                for emb in query_embeddings
            ]
            raw = await asyncio.gather(*search_tasks)
            combined_results = self._merge_results(list(raw))

        # Step 4: Score threshold filtering.
        # - Pure vector: cosine scores are on [0, 1] → absolute threshold.
        # - Hybrid (RRF): scores depend on alpha/k, not on [0, 1] scale →
        #   relative filter: drop chunks below HYBRID_RRF_MIN_RATIO × top_score.
        if combined_results:
            if settings.USE_HYBRID_SEARCH:
                top_score = combined_results[0]["score"]
                min_score = top_score * settings.HYBRID_RRF_MIN_RATIO
                before = len(combined_results)
                combined_results = [r for r in combined_results if r["score"] >= min_score]
                logger.debug(
                    "Hybrid threshold (ratio=%.2f, floor=%.5f): %d → %d candidates",
                    settings.HYBRID_RRF_MIN_RATIO, min_score, before, len(combined_results),
                )
            else:
                combined_results = [
                    r for r in combined_results
                    if r["score"] >= settings.RETRIEVAL_SCORE_THRESHOLD
                ]

        if not combined_results:
            logger.warning("No results found after threshold filtering")
            return {"chunks": [], "sub_questions": sub_questions}

        # Step 5: Cross-encoder reranking + relevance gate.
        # Pass ALL post-filter candidates so the cross-encoder can rescue a
        # chunk that ranked poorly under RRF but is highly relevant to the
        # original query.  The reranker always uses retrieval_query (the
        # original / rewritten query), never the paraphrases or sub-questions.
        if self.reranker and settings.USE_RERANKER:
            logger.info(
                "Reranking %d candidates → top %d (threshold=%.2f, mode=%s)",
                len(combined_results), effective_top_k, reranker_threshold,
                "decomposed" if is_decomposed else "paraphrase",
            )
            reranked = await self.reranker.rerank(
                query=retrieval_query,
                chunks=combined_results,
                top_k=effective_top_k,
            )
            if reranked and reranked[0].get("rerank_score", 1.0) < reranker_threshold:
                logger.info(
                    "Top reranker score %.3f < threshold %.3f — no relevant context",
                    reranked[0]["rerank_score"], reranker_threshold,
                )
                return {"chunks": [], "sub_questions": sub_questions}
            return {"chunks": reranked, "sub_questions": sub_questions}

        return {"chunks": combined_results[:effective_top_k], "sub_questions": sub_questions}

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(self, vector_results, keyword_results, alpha=0.5, k=60):
        """
        Combine vector and keyword results using Reciprocal Rank Fusion.

        RRF score = alpha / (k + rank_vector) + (1-alpha) / (k + rank_keyword)

        RRF is used instead of direct score addition because cosine scores
        (0-1) and BM25 scores (0-∞) are on incompatible scales.  Rank-based
        combination is scale-invariant and is the industry standard.
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

    # ------------------------------------------------------------------
    # Multi-query result merger
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_results(result_lists: List[List[Dict]]) -> List[Dict]:
        """
        Merge result lists from multiple query variants into one ranked list.

        For each unique chunk ID, keep the highest score seen across all
        query variants (paraphrases or sub-questions).  A chunk that ranks
        well for several sub-questions naturally carries a higher max score;
        one that only surfaced as a false-positive for a single variant is
        not artificially boosted.

        Sorting by max score preserves the relative RRF ordering established
        within each per-query list.
        """
        best: Dict[str, Dict] = {}
        for results in result_lists:
            for r in results:
                cid = r["id"]
                if cid not in best or r["score"] > best[cid]["score"]:
                    best[cid] = r
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)
