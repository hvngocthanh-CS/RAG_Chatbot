"""Pydantic schemas for the Chat API."""
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    stream: bool = True
    filters: Optional[dict] = None


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
