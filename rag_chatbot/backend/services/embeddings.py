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
    Service for generating text embeddings.
    
    Supports:
    - HuggingFace models (BGE, E5, Instructor)
    - OpenAI embeddings
    
    Features:
    - Batch processing
    - Caching
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
            elif self.provider == "openai":
                await self._init_openai()
            
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
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model
            logger.info(f"Loading HuggingFace model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=device
            )
            
            logger.info(f"✓ Embedding model loaded successfully")
            
        except ImportError as e:
            logger.error(f"sentence-transformers not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
    
    async def _init_openai(self):
        """Initialize OpenAI embeddings client."""
        try:
            from openai import AsyncOpenAI
            
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(f"Initialized OpenAI embeddings: {settings.OPENAI_EMBEDDING_MODEL}")
            
        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
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
        
        elif self.provider == "openai":
            response = await self.client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
    
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
        elif self.provider == "openai":
            return await self._embed_openai(texts)
    
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
    
    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        # OpenAI has a limit of 2048 inputs per request
        batch_size = 2000
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = await self.client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=batch
            )
            
            # Sort by index to ensure correct order
            batch_embeddings = [None] * len(batch)
            for item in response.data:
                batch_embeddings[item.index] = item.embedding
            
            all_embeddings.extend(batch_embeddings)
        
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
