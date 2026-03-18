"""
Document management endpoints.
"""
import os
import uuid
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from backend.config import settings
from backend.services.ingestion import DocumentIngestionService
from backend.services import get_service

logger = logging.getLogger(__name__)
router = APIRouter()


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


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    department: Optional[str] = Query(None, description="Department that owns this document (e.g. HR, Finance, IT)"),
    category: Optional[str] = Query(None, description="Document category (e.g. Policy, SOP, Report, Contract, Handbook)"),
    author: Optional[str] = Query(None, description="Author or document owner name"),
    version: Optional[str] = Query(None, description="Document version (e.g. v1.0, 2024-Q1)"),
    doc_date: Optional[str] = Query(None, description="Document effective/publish date (YYYY-MM-DD)"),
    tags: Optional[str] = Query(None, description="Comma-separated tags for free-text filtering")
):
    """
    Upload and process a document.
    
    Supported formats: PDF, DOCX, TXT, MD
    Maximum file size: 50MB
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {settings.SUPPORTED_EXTENSIONS}"
        )
    
    # Validate file size
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Save file to upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}{file_ext}")
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    
    # Prepare metadata
    metadata = {
        "document_id": document_id,
        "filename": file.filename,
        "file_type": file_ext,
        "file_size": len(file_content),
        "language": "en",          # English-only system
        "department": department,
        "category": category,      # Policy | SOP | Report | Contract | Handbook | …
        "author": author,
        "version": version,
        "doc_date": doc_date,
        "tags": tag_list
    }
    
    # Process document - SYNC instead of background for better error handling
    try:
        logger.info(f"Processing document: {file.filename}")
        ingestion_service = DocumentIngestionService()
        result = await ingestion_service.process_document(file_path, metadata)
        
        logger.info(f"Document processed: {file.filename} ({result.get('chunks_count')} chunks)")
        
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="success" if result.get("chunks_count", 0) > 0 else "warning",
            message=f"Document processed: {result.get('chunks_count', 0)} chunks created"
        )
    
    except Exception as e:
        logger.error(f"Failed to process document {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    department: Optional[str] = None,
    status: Optional[str] = None
):
    """
    List all indexed documents with optional filtering.
    """
    vector_store = get_service("vector_store")
    documents = await vector_store.list_documents(
        skip=skip,
        limit=limit,
        department=department,
        status=status
    )
    
    return DocumentListResponse(
        documents=documents,
        total=len(documents)
    )


@router.get("/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str):
    """
    Get metadata for a specific document.
    """
    vector_store = get_service("vector_store")
    document = await vector_store.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its embeddings from the system.
    """
    vector_store = get_service("vector_store")
    success = await vector_store.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Also delete the uploaded file
    for ext in settings.SUPPORTED_EXTENSIONS:
        file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}{ext}")
        if os.path.exists(file_path):
            os.remove(file_path)
            break
    
    logger.info(f"Document deleted: {document_id}")
    
    return {"status": "success", "message": f"Document {document_id} deleted"}
