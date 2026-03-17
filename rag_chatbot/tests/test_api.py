"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# We need to mock the services before importing the app
@pytest.fixture
def client():
    """Create test client with mocked services."""
    with patch('backend.services.initialize_services'):
        with patch('backend.services.cleanup_services'):
            from backend.api.main import app
            with TestClient(app) as client:
                yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestDocumentEndpoints:
    """Tests for document management endpoints."""

    def test_upload_unsupported_file(self, client):
        """Test uploading unsupported file type."""
        # Create a fake file with unsupported extension
        files = {"file": ("test.xyz", b"content", "application/octet-stream")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    @patch('backend.api.routes.documents.DocumentIngestionService')
    @patch('backend.api.routes.documents.VectorStoreService')
    def test_list_documents_empty(self, mock_vector, mock_ingestion, client):
        """Test listing documents when empty."""
        mock_vector.return_value.list_documents = MagicMock(return_value=[])
        
        response = client.get("/api/v1/documents")
        
        assert response.status_code == 200


class TestChatEndpoints:
    """Tests for chat endpoints."""

    @patch('backend.api.routes.chat.RetrievalService')
    @patch('backend.api.routes.chat.LLMService')
    @patch('backend.api.routes.chat.CacheService')
    @patch('backend.api.routes.chat.ConversationManager')
    def test_chat_no_documents(
        self, 
        mock_conv, 
        mock_cache, 
        mock_llm, 
        mock_retrieval,
        client
    ):
        """Test chat when no documents indexed."""
        # Mock retrieval to return empty
        mock_retrieval.return_value.retrieve = MagicMock(return_value=[])
        mock_cache.return_value.get_cached_response = MagicMock(return_value=None)
        mock_conv.return_value.create_conversation = MagicMock(return_value="test-id")
        mock_conv.return_value.get_history = MagicMock(return_value=[])
        
        response = client.post(
            "/api/v1/chat",
            json={"question": "Test question", "stream": False}
        )
        
        # Should return 404 when no documents found
        assert response.status_code == 404


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
