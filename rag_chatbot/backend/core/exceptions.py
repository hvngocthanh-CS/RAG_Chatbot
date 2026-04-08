"""
Custom exception hierarchy for the RAG Chatbot.
"""


class RAGException(Exception):
    """Base exception for all RAG-related errors."""


class DocumentProcessingError(RAGException):
    """Raised when document parsing, preprocessing, or chunking fails."""


class RetrievalError(RAGException):
    """Raised when the retrieval pipeline fails."""


class LLMServiceError(RAGException):
    """Raised when the LLM service is unavailable or returns an error."""


class ServiceUnavailableError(RAGException):
    """Raised when a required service (Qdrant, Redis, Ollama) is unavailable."""
