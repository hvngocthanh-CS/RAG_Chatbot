"""
Custom exception hierarchy for the RAG Chatbot.

Use these instead of raising bare `Exception`. They make error handling
explicit (callers can catch the kind of failure they care about) and let the
API layer translate domain errors into proper HTTP responses without leaking
internal details.
"""


class RAGException(Exception):
    """Base exception for all RAG-related errors."""


# --- Ingestion pipeline ----------------------------------------------------

class IngestionError(RAGException):
    """Raised when the ingestion pipeline fails as a whole."""


class DocumentParsingError(IngestionError):
    """Raised when a document cannot be parsed (corrupt file, unsupported layout)."""


class DocumentProcessingError(IngestionError):
    """Raised when preprocessing or chunking fails."""


# --- Retrieval / generation ------------------------------------------------

class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGException):
    """Raised when the vector store (Qdrant) read/write fails."""


class RetrievalError(RAGException):
    """Raised when the retrieval pipeline fails."""


class LLMServiceError(RAGException):
    """Raised when the LLM service is unavailable or returns an error."""


# --- Infrastructure --------------------------------------------------------

class ServiceUnavailableError(RAGException):
    """Raised when a required service (Qdrant, Redis, Ollama) is unavailable."""
