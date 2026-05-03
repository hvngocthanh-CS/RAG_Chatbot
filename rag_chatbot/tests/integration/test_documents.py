"""Integration tests — document management endpoints."""
import pytest

from tests.integration.conftest import SAMPLE_DOCUMENT


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_txt_document(client, mock_ingestion):
    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("report.txt", b"This is sample document content.", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["filename"] == "report.txt"
    assert "document_id" in body
    assert "4" in body["message"]  # mock returns chunks_count=4


@pytest.mark.asyncio
async def test_upload_unsupported_extension(client):
    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("malware.exe", b"binary data", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_calls_ingestion_pipeline(client, mock_ingestion):
    await client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc.txt", b"content", "text/plain")},
    )
    mock_ingestion.process_document.assert_called_once()
    call_kwargs = mock_ingestion.process_document.call_args
    _, metadata = call_kwargs.args
    assert metadata["filename"] == "doc.txt"
    assert metadata["file_type"] == ".txt"


# ---------------------------------------------------------------------------
# List & get
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_documents_empty(client):
    response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == []
    assert body["total"] == 0


@pytest.mark.asyncio
async def test_list_documents_returns_items(client, mock_vector_store):
    mock_vector_store.list_documents.return_value = [SAMPLE_DOCUMENT]
    response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["documents"][0]["filename"] == "policy.pdf"


@pytest.mark.asyncio
async def test_get_document_not_found(client):
    response = await client.get("/api/v1/documents/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_document_success(client, mock_vector_store):
    mock_vector_store.get_document.return_value = SAMPLE_DOCUMENT
    response = await client.get("/api/v1/documents/doc-fixture-001")
    assert response.status_code == 200
    assert response.json()["filename"] == "policy.pdf"


# ---------------------------------------------------------------------------
# Delete + cache invalidation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_document_success(client):
    response = await client.delete("/api/v1/documents/any-doc-id")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


@pytest.mark.asyncio
async def test_delete_document_invalidates_cache(client, mock_cache):
    doc_id = "doc-to-delete-123"
    await client.delete(f"/api/v1/documents/{doc_id}")
    mock_cache.invalidate_document_cache.assert_called_once_with(doc_id)


@pytest.mark.asyncio
async def test_delete_nonexistent_document(client, mock_vector_store):
    mock_vector_store.delete_document.return_value = False
    response = await client.delete("/api/v1/documents/ghost-id")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Full upload → delete flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_then_delete_flow(client, mock_vector_store, mock_cache):
    # Upload
    upload_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("flow.txt", b"test content", "text/plain")},
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["document_id"]

    # Delete
    delete_resp = await client.delete(f"/api/v1/documents/{doc_id}")
    assert delete_resp.status_code == 200
    mock_cache.invalidate_document_cache.assert_called_once_with(doc_id)
