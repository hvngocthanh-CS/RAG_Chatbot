"""
Reranker Service.
Uses cross-encoder models to improve retrieval accuracy.
"""
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
    
    Supported models:
    - BGE Reranker
    - Cross-encoder models from sentence-transformers
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
            
            # Use configured device (CPU for this service to free GPU for LLM)
            device = settings.RERANKER_DEVICE
            # Validate device availability if CUDA is requested
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
            
            logger.info(f"Initializing reranker on device: {device} (configured via RERANKER_DEVICE setting)")
            self.model = CrossEncoder(
                settings.RERANKER_MODEL,
                max_length=512,
                device=device
            )
            
            self._initialized = True
            logger.info(f"✓ Reranker initialized: {settings.RERANKER_MODEL} on {device}")
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking disabled.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.model = None
    
    async def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on relevance to query.
        
        Args:
            query: User's question
            chunks: Retrieved chunks to rerank
            top_k: Number of top results to return
        
        Returns:
            Reranked and filtered chunks
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.model or not chunks:
            return chunks[:top_k]
        
        try:
            # Prepare query-passage pairs
            pairs = [(query, chunk.get("text") or chunk.get("content", "")) for chunk in chunks]
            
            # Get reranker scores
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # Normalise cross-encoder logits to [0, 1] via sigmoid so they
            # are on the same scale as cosine-similarity scores.
            def _sigmoid(x: float) -> float:
                return 1.0 / (1.0 + math.exp(-x))

            scored_chunks = []
            for chunk, raw_score in zip(chunks, scores):
                reranked_chunk = chunk.copy()
                norm_score = _sigmoid(float(raw_score))
                reranked_chunk["rerank_score"] = norm_score
                # Use only the cross-encoder score for final ranking.
                # The original score (cosine or RRF) is on a completely
                # different scale, so blending them is meaningless.
                # The cross-encoder already sees (query, passage) jointly
                # and is strictly more accurate than the bi-encoder score.
                reranked_chunk["original_score"] = chunk.get("score", 0)
                reranked_chunk["score"] = norm_score
                scored_chunks.append(reranked_chunk)
            
            # Sort by combined score
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Reranked {len(chunks)} chunks, returning top {top_k}")
            
            return scored_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]
    
    async def health_check(self) -> bool:
        """Check if reranker is healthy."""
        try:
            if not self._initialized:
                await self.initialize()
            return self.model is not None
        except Exception:
            return False
