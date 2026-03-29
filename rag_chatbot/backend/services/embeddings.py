"""
Embedding Service.
Handles text embedding generation using various providers.
"""
import asyncio
import logging
from typing import List, Optional
import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using HuggingFace models.
    
    Supports:
    - HuggingFace models (BGE, E5, Instructor)
    
    Features:
    - Batch processing
    - Automatic model loading
    """
    
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the embedding model."""
        if self._initialized:
            return
        
        logger.info(f"Initializing embedding service with provider: {self.provider}")
        
        try:
            if self.provider == "huggingface":
                await asyncio.wait_for(self._init_huggingface(), timeout=300)  # 5 min timeout
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
            
            self._initialized = True
            logger.info("Embedding service initialized")
        
        except asyncio.TimeoutError:
            logger.error(f"Embedding model initialization timeout")
            raise RuntimeError("Embedding model initialization timed out (>300s)")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}", exc_info=True)
            raise
    
    async def _init_huggingface(self):
        """Initialize HuggingFace model."""
        try:
            logger.info("Importing sentence-transformers...")
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Use configured device (CPU for this service to free GPU for LLM)
            device = settings.EMBEDDING_DEVICE
            # Validate device availability if CUDA is requested
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
            logger.info(f"Using device: {device} (configured via EMBEDDING_DEVICE setting)")
            
            # Load model
            logger.info(f"Loading HuggingFace model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=device
            )
            
            logger.info(f"✓ Embedding model loaded successfully on {device}")
            
        except ImportError as e:
            logger.error(f"sentence-transformers not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        For query embeddings, some models use special prefixes.
        """
        if not self._initialized:
            await self.initialize()
        
        if self.provider == "huggingface":
            # BGE models use query prefix
            if "bge" in settings.EMBEDDING_MODEL.lower():
                text = f"Represent this sentence for searching relevant passages: {text}"
            
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        
        raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Uses batch processing for efficiency.
        """
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        if self.provider == "huggingface":
            return await self._embed_huggingface(texts)
        
        raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    async def _embed_huggingface(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace model."""
        # Process in batches to manage memory
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # BGE models use passage prefix for documents
            if "bge" in settings.EMBEDDING_MODEL.lower():
                batch = [f"Represent this sentence for retrieval: {t}" for t in batch]
            
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.provider == "huggingface" and self.model:
            return self.model.get_sentence_embedding_dimension()
        return settings.EMBEDDING_DIMENSION
    
    async def health_check(self) -> bool:
        """Check if the embedding service is healthy."""
        try:
            test_text = "Health check test"
            embedding = await self.embed_query(test_text)
            return len(embedding) > 0
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False
