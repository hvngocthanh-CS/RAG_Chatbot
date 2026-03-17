"""
Reranker Service.
Uses cross-encoder models to improve retrieval accuracy.
"""
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
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = CrossEncoder(
                settings.RERANKER_MODEL,
                max_length=512,
                device=device
            )
            
            self._initialized = True
            logger.info(f"Reranker initialized: {settings.RERANKER_MODEL}")
            
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
            pairs = [(query, chunk["content"]) for chunk in chunks]
            
            # Get reranker scores
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # Combine with original chunks
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                reranked_chunk = chunk.copy()
                reranked_chunk["rerank_score"] = float(score)
                # Combine original score with rerank score
                original_score = chunk.get("score", 0)
                reranked_chunk["score"] = 0.3 * original_score + 0.7 * float(score)
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
