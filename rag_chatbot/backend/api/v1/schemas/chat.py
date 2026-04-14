"""Pydantic schemas for the Chat API."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request schema.
    
    Args:
        question: User question (1-2000 characters)
        conversation_id: Optional conversation context for multi-turn chats
        stream: Whether to stream response or get full response
        filters: Optional metadata filters (e.g., {"filename": "doc.pdf", "document_id": "123"}).
                 Leave empty dict {} or null if no filtering needed.
    
    Example:
        {
            "question": "What is the policy?",
            "conversation_id": "conv-123",
            "stream": false,
            "filters": {}
        }
    """
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn context")
    stream: bool = Field(True, description="Stream response or get full response")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters (filename, document_id, etc.)")


class SourceChunk(BaseModel):
    content: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_type: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: List[SourceChunk]
    processing_time_ms: int
