"""
Embedding Service — Step 4 of the RAG pipeline.

Generates dense vector representations of text using HuggingFace
sentence-transformers.  Sits between the Chunker (Step 3) and the
Vector Store (Step 5).

Design decisions:
  - Model: BAAI/bge-base-en-v1.5 (768-dim, top MTEB for its size class)
  - Asymmetric encoding: queries get an instruction prefix so the model
    knows to match them against passages, not against other queries.
    Passages are embedded raw — the Section Chunker already prepends a
    [Heading] prefix that gives the model a strong topic signal.
  - All CPU-bound work (model loading + encoding) runs via
    asyncio.to_thread() so the FastAPI event loop is never blocked.
  - Batch size 32 balances throughput vs memory on CPU.

Ref: https://huggingface.co/BAAI/bge-base-en-v1.5
"""

import asyncio
import logging
import time
import os
from typing import List

# Set cache directories BEFORE importing models
os.environ.setdefault("HF_HOME", "./models")
os.environ.setdefault("TRANSFORMERS_CACHE", "./models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "./models")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "180")

from backend.config import settings

logger = logging.getLogger(__name__)

# BGE query instruction — tells the model the text is a search query.
# Passages must NOT have this prefix (they are embedded raw).
_BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingService:
    """
    Text embedding service backed by SentenceTransformer.

    All heavy work (model load, encode) is offloaded to a thread via
    asyncio.to_thread() so the async event loop stays responsive.
    """

    def __init__(self):
        self.model = None
        self._initialized = False
        self._model_name = settings.EMBEDDING_MODEL
        self._device: str = settings.EMBEDDING_DEVICE

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self):
        """Load the embedding model from cache (must be preloaded).
        
        ⚠️  REQUIRED: Models must be cached first!
            Run: python setup_embedding_models.py
        """
        if self._initialized:
            return

        # Verify models are cached locally
        models_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME", "./models")
        if not os.path.exists(models_dir):
            raise RuntimeError(
                f"\n❌ ERROR: Models directory not found: {models_dir}\n"
                f"   Please run first:\n"
                f"   python setup_embedding_models.py\n"
            )

        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._load_model_sync),
                timeout=600,  # 10 minutes for loading from cache
            )
            self._initialized = True
            logger.info(
                "Embedding service ready  model=%s  dim=%d  device=%s",
                self._model_name,
                self.get_embedding_dimension(),
                self._device,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "Embedding model failed to load from cache (>600 s). "
                "Please ensure models are properly cached by running: "
                "python setup_embedding_models.py"
            )
        except Exception as e:
            logger.error("Failed to initialize embedding service: %s", e, exc_info=True)
            raise

    def _load_model_sync(self):
        """Synchronous model loading — called inside a thread."""
        from sentence_transformers import SentenceTransformer
        import torch

        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU")
            self._device = "cpu"

        logger.info("Loading %s on %s ...", self._model_name, self._device)
        self.model = SentenceTransformer(self._model_name, device=self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single search query.

        BGE models use an instruction prefix for queries so the vector
        space separates "looking-for" intent from "passage content".
        """
        if not self._initialized:
            await self.initialize()

        if "bge" in self._model_name.lower():
            text = f"{_BGE_QUERY_INSTRUCTION}{text}"

        embedding = await asyncio.to_thread(
            self.model.encode, text, normalize_embeddings=True,
        )
        return embedding.tolist()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed document passages in batches.

        Passages are embedded WITHOUT an instruction prefix — the
        Section Chunker's ``[Heading]`` prefix already provides topic
        context to the model.

        Progress is logged every 5 batches so large ingestion jobs
        are observable.
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        batch_size = 32
        total_batches = (len(texts) + batch_size - 1) // batch_size
        all_embeddings: List[List[float]] = []

        t0 = time.perf_counter()

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            batch = texts[start : start + batch_size]

            embeddings = await asyncio.to_thread(
                self.model.encode,
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())

            # Progress logging every 5 batches or on the last batch
            if (batch_idx + 1) % 5 == 0 or batch_idx + 1 == total_batches:
                elapsed = time.perf_counter() - t0
                done = min(start + batch_size, len(texts))
                logger.info(
                    "Embedding progress: %d/%d texts  (%.1f s elapsed)",
                    done, len(texts), elapsed,
                )

        return all_embeddings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_embedding_dimension(self) -> int:
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return settings.EMBEDDING_DIMENSION

    async def health_check(self) -> bool:
        try:
            emb = await self.embed_query("health check")
            return len(emb) == self.get_embedding_dimension()
        except Exception:
            return False
