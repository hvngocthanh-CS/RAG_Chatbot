"""
Reranker Service.
Uses cross-encoder models to improve retrieval accuracy.
"""
import asyncio
import math
import logging
from typing import List, Dict, Any

from backend.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Cross-encoder reranking service.

    Reranking uses a more powerful model to re-score retrieved
    results based on the actual question-passage relevance.
    """

    def __init__(self):
        self.model = None
        self._initialized = False

    async def initialize(self):
        """Initialize the reranker model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder
            import torch

            device = settings.RERANKER_DEVICE
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"

            logger.info("Initializing reranker on device: %s", device)
            self.model = CrossEncoder(
                settings.RERANKER_MODEL,
                max_length=512,
                device=device
            )

            self._initialized = True
            logger.info("Reranker initialized: %s on %s", settings.RERANKER_MODEL, device)

        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking disabled.")
            self.model = None
        except Exception as e:
            logger.error("Failed to initialize reranker: %s", e)
            self.model = None

    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank chunks based on relevance to query."""
        if not self._initialized:
            await self.initialize()

        if not self.model or not chunks:
            return chunks[:top_k]

        try:
            pairs = [(query, chunk.get("text") or chunk.get("content", "")) for chunk in chunks]
            scores = await asyncio.to_thread(
                self.model.predict, pairs, show_progress_bar=False
            )

            def _sigmoid(x: float) -> float:
                return 1.0 / (1.0 + math.exp(-x))

            scored_chunks = []
            for chunk, raw_score in zip(chunks, scores):
                reranked_chunk = chunk.copy()
                norm_score = _sigmoid(float(raw_score))
                reranked_chunk["rerank_score"] = norm_score
                reranked_chunk["original_score"] = chunk.get("score", 0)
                reranked_chunk["score"] = norm_score
                scored_chunks.append(reranked_chunk)

            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            logger.info("Reranked %d chunks, returning top %d", len(chunks), top_k)

            return scored_chunks[:top_k]

        except Exception as e:
            logger.error("Reranking failed: %s", e)
            return chunks[:top_k]

    async def health_check(self) -> bool:
        try:
            if not self._initialized:
                await self.initialize()
            return self.model is not None
        except Exception:
            return False
