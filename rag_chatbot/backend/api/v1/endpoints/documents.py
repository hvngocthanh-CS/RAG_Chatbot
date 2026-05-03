"""
Document management endpoints.
"""
import os
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from backend.config import settings
from backend.core.exceptions import IngestionError
from backend.services import get_service
from backend.api.v1.schemas.documents import (
    DocumentMetadata,
    DocumentListResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: Optional[str] = Query(None, description="Department that owns this document"),
    category: Optional[str] = Query(None, description="Document category"),
    author: Optional[str] = Query(None, description="Author or document owner name"),
    version: Optional[str] = Query(None, description="Document version"),
    doc_date: Optional[str] = Query(None, description="Document effective/publish date (YYYY-MM-DD)"),
    tags: Optional[str] = Query(None, description="Comma-separated tags")
):
    """Upload and process a document. Supported: PDF, DOCX, TXT, MD. Max: 50MB."""
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
        "language": "en",
        "department": department,
        "category": category,
        "author": author,
        "version": version,
        "doc_date": doc_date,
        "tags": tag_list
    }

    # Process document
    try:
        logger.info("Processing document: %s", file.filename)
        ingestion_service = get_service("ingestion")
        result = await ingestion_service.process_document(file_path, metadata)

        logger.info("Document processed: %s (%d chunks)", file.filename, result.get("chunks_count", 0))

        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="success" if result.get("chunks_count", 0) > 0 else "warning",
            message=f"Document processed: {result.get('chunks_count', 0)} chunks created"
        )

    except IngestionError as e:
        # Domain-level failure: bad/corrupt file, parsing fails, etc.
        # The pipeline already logged the full traceback.
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process document: {e}",
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    department: Optional[str] = None,
    status: Optional[str] = None
):
    """List all indexed documents with optional filtering."""
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
    """Get metadata for a specific document."""
    vector_store = get_service("vector_store")
    document = await vector_store.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings from the system."""
    vector_store = get_service("vector_store")
    success = await vector_store.delete_document(document_id)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete the uploaded file from disk
    for ext in settings.SUPPORTED_EXTENSIONS:
        file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}{ext}")
        if os.path.exists(file_path):
            os.remove(file_path)
            break

    # Invalidate cached responses that may reference this document
    cache_service = get_service("cache")
    if cache_service:
        await cache_service.invalidate_document_cache(document_id)

    logger.info("Document deleted: %s", document_id)

    return {"status": "success", "message": f"Document {document_id} deleted"}
