"""
Shared fixtures for integration tests.

All external services (Qdrant, Ollama, Redis, PostgreSQL) are replaced with
in-process mocks so the test suite runs without any running infrastructure.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

import backend.services as svc_module
from backend.config import settings
from backend.services.conversation import ConversationManager


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_CHUNK = {
    "content": "Employees must complete onboarding training within 30 days of joining.",
    "metadata": {
        "document_id": "doc-fixture-001",
        "filename": "policy.pdf",
        "page_number": 1,
        "chunk_type": "text",
        "department": None,
        "category": None,
        "version": None,
    },
    "score": 0.88,
}

SAMPLE_DOCUMENT = {
    "id": "doc-fixture-001",
    "filename": "policy.pdf",
    "file_type": ".pdf",
    "file_size": 1024,
    "upload_date": "2026-01-01T00:00:00",
    "chunks_count": 5,
    "status": "indexed",
    "department": None,
    "tags": [],
}


# ---------------------------------------------------------------------------
# Autouse: reset shared state between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_conversation_store():
    ConversationManager._conversations.clear()
    yield
    ConversationManager._conversations.clear()


@pytest.fixture(autouse=True)
def _temp_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# Service mocks (function-scoped — fresh per test)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    svc = AsyncMock()
    svc.delete_document.return_value = True
    svc.list_documents.return_value = []
    svc.get_document.return_value = None
    return svc


@pytest.fixture
def mock_ingestion():
    svc = AsyncMock()
    svc.process_document.return_value = {"chunks_count": 4}
    return svc


@pytest.fixture
def mock_retrieval():
    svc = AsyncMock()
    svc.retrieve.return_value = {"chunks": [SAMPLE_CHUNK], "sub_questions": []}
    return svc


@pytest.fixture
def mock_llm():
    svc = AsyncMock()
    svc.generate.return_value = "The answer is based on the uploaded documents."

    async def _stream_tokens(*args, **kwargs):
        for token in ["The ", "answer ", "is ", "here."]:
            yield token

    svc.generate_stream = _stream_tokens
    return svc


@pytest.fixture
def mock_cache():
    svc = AsyncMock()
    # Default: cache miss — prevents the early-return path from firing in every test
    svc.get_cached_response.return_value = None
    return svc


# ---------------------------------------------------------------------------
# HTTP test client
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(mock_vector_store, mock_ingestion, mock_retrieval, mock_llm, mock_cache):
    """
    AsyncClient wired to the FastAPI app with all heavy services mocked.

    Strategy: populate the global _services dict directly, then patch
    initialize_services / cleanup_services to no-ops so the real lifespan
    does not attempt connections to Qdrant, Ollama or Redis.
    """
    injected = {
        "vector_store": mock_vector_store,
        "ingestion": mock_ingestion,
        "retrieval": mock_retrieval,
        "llm": mock_llm,
        "conversation": ConversationManager(),
        "cache": mock_cache,
    }

    async def _noop(): ...

    with (
        patch("backend.services.initialize_services", new=_noop),
        patch("backend.services.cleanup_services", new=_noop),
    ):
        # Services are available before lifespan runs
        svc_module._services.update(injected)

        from backend.api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    svc_module._services.clear()
