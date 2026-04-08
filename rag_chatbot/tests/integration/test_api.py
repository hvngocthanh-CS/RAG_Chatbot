"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


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
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestDocumentEndpoints:
    """Tests for document management endpoints."""

    def test_upload_unsupported_file(self, client):
        files = {"file": ("test.xyz", b"content", "application/octet-stream")}

        response = client.post("/api/v1/documents/upload", files=files)

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
