"""Pydantic schemas for the Documents API."""
from typing import List, Optional
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    id: str
    filename: str
    file_type: str
    file_size: int
    upload_date: str
    chunks_count: int
    status: str
    department: Optional[str] = None
    tags: List[str] = []


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    documents: List[DocumentMetadata]
    total: int


class UploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    status: str
    message: str
