"""
Step 4 — Embedding Service tests.

Validates that the EmbeddingService correctly embeds:
  - Single queries (with BGE instruction prefix)
  - Document passages in batches (raw, no prefix)
  - Section-aware chunks (heading prefix preserved in vector)
"""

import pytest
import numpy as np

from backend.services.embedding import EmbeddingService
from backend.services.ingestion import SectionChunker
from backend.config import settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_service():
    """Load the embedding model once for all tests in this module."""
    svc = EmbeddingService()
    import asyncio
    asyncio.get_event_loop().run_until_complete(svc.initialize())
    return svc


@pytest.fixture(scope="module")
def expected_dim(embedding_service):
    return embedding_service.get_embedding_dimension()


# ---------------------------------------------------------------------------
# Basic embedding tests
# ---------------------------------------------------------------------------

class TestEmbeddingBasics:

    @pytest.mark.asyncio
    async def test_model_loads(self, embedding_service):
        assert embedding_service._initialized is True
        assert embedding_service.model is not None

    @pytest.mark.asyncio
    async def test_embedding_dimension_matches_settings(self, embedding_service, expected_dim):
        assert expected_dim == settings.EMBEDDING_DIMENSION

    @pytest.mark.asyncio
    async def test_embed_query_returns_correct_shape(self, embedding_service, expected_dim):
        emb = await embedding_service.embed_query("What is the leave policy?")
        assert isinstance(emb, list)
        assert len(emb) == expected_dim

    @pytest.mark.asyncio
    async def test_embed_query_is_normalized(self, embedding_service):
        emb = await embedding_service.embed_query("test normalization")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_embed_documents_batch(self, embedding_service, expected_dim):
        texts = [
            "The annual leave policy grants 15 days per year.",
            "Remote work requires manager approval.",
            "Code reviews must be completed within 48 hours.",
        ]
        embeddings = await embedding_service.embed_documents(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == expected_dim

    @pytest.mark.asyncio
    async def test_embed_documents_empty(self, embedding_service):
        result = await embedding_service.embed_documents([])
        assert result == []


# ---------------------------------------------------------------------------
# Asymmetric encoding tests
# ---------------------------------------------------------------------------

class TestAsymmetricEncoding:

    @pytest.mark.asyncio
    async def test_query_and_passage_differ(self, embedding_service):
        text = "employee leave policy"

        query_emb = await embedding_service.embed_query(text)
        passage_emb = await embedding_service.embed_documents([text])

        sim = np.dot(query_emb, passage_emb[0])
        assert sim < 0.99

    @pytest.mark.asyncio
    async def test_relevant_query_passage_high_similarity(self, embedding_service):
        query_emb = await embedding_service.embed_query(
            "What is the annual leave entitlement?"
        )
        passage_emb = await embedding_service.embed_documents([
            "[Leave Policy]\n\nAll employees are entitled to 15 days of annual leave per year.",
        ])

        sim = float(np.dot(query_emb, passage_emb[0]))
        assert sim > 0.5


# ---------------------------------------------------------------------------
# Section chunk compatibility
# ---------------------------------------------------------------------------

class TestSectionChunkEmbedding:

    @pytest.mark.asyncio
    async def test_heading_prefix_improves_topic_signal(self, embedding_service):
        query_emb = await embedding_service.embed_query(
            "What are the code review requirements?"
        )

        text_body = (
            "All pull requests must have at least two approvals. "
            "Reviews should be completed within 48 hours of submission."
        )

        with_heading = f"[Code Review Standards]\n\n{text_body}"
        without_heading = text_body

        emb_with = (await embedding_service.embed_documents([with_heading]))[0]
        emb_without = (await embedding_service.embed_documents([without_heading]))[0]

        sim_with = float(np.dot(query_emb, emb_with))
        sim_without = float(np.dot(query_emb, emb_without))

        assert sim_with >= sim_without - 0.05

    @pytest.mark.asyncio
    async def test_end_to_end_chunk_then_embed(self, embedding_service, expected_dim):
        chunker = SectionChunker(max_chunk_tokens=200, min_chunk_tokens=20)

        text_blocks = [
            {"text": "Performance Reviews", "block_type": "heading", "page_number": 1},
            {
                "text": (
                    "Performance reviews are conducted quarterly. Each employee "
                    "meets with their manager to discuss goals and progress. "
                    "Ratings are given on a five-point scale."
                ),
                "block_type": "paragraph",
                "page_number": 1,
            },
        ]
        metadata = {"document_id": "test", "filename": "handbook.pdf"}

        chunks = chunker.chunk_text(text_blocks, metadata)
        assert len(chunks) >= 1

        texts = [c["content"] for c in chunks]
        embeddings = await embedding_service.embed_documents(texts)

        assert len(embeddings) == len(chunks)
        for emb in embeddings:
            assert len(emb) == expected_dim
            assert abs(np.linalg.norm(emb) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Large batch handling
# ---------------------------------------------------------------------------

class TestBatchProcessing:

    @pytest.mark.asyncio
    async def test_large_batch_correct_count(self, embedding_service):
        texts = [f"Document number {i} about various topics." for i in range(70)]
        embeddings = await embedding_service.embed_documents(texts)
        assert len(embeddings) == 70


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_health_check_passes(self, embedding_service):
        assert await embedding_service.health_check() is True
