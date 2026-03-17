#!/usr/bin/env python
"""
Document Ingestion Script.
Use this script to ingest documents into the RAG system.

Usage:
    python ingest_documents.py <path_to_documents>
    python ingest_documents.py ./documents --department HR
    python ingest_documents.py ./report.pdf --tags "quarterly,finance"
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings
from backend.services.ingestion import DocumentIngestionService
from backend.services import initialize_services

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def ingest_file(
    file_path: str,
    department: str = None,
    tags: list = None
):
    """Ingest a single file."""
    import uuid
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    # Check extension
    if file_path.suffix.lower() not in settings.SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported file type: {file_path.suffix}")
        return False
    
    # Generate metadata
    document_id = str(uuid.uuid4())
    metadata = {
        "document_id": document_id,
        "filename": file_path.name,
        "file_type": file_path.suffix.lower(),
        "file_size": file_path.stat().st_size,
        "department": department,
        "tags": tags or []
    }
    
    logger.info(f"Ingesting: {file_path.name} (ID: {document_id})")
    
    # Process document
    ingestion_service = DocumentIngestionService()
    result = await ingestion_service.process_document(str(file_path), metadata)
    
    if result["status"] == "success":
        logger.info(f"Successfully ingested: {file_path.name} ({result['chunks_count']} chunks)")
        return True
    else:
        logger.error(f"Failed to ingest: {file_path.name}")
        return False


async def ingest_directory(
    directory_path: str,
    department: str = None,
    tags: list = None,
    recursive: bool = True
):
    """Ingest all documents in a directory."""
    directory = Path(directory_path)
    
    if not directory.is_dir():
        logger.error(f"Not a directory: {directory}")
        return
    
    # Find all supported files
    files = []
    for ext in settings.SUPPORTED_EXTENSIONS:
        if recursive:
            files.extend(directory.rglob(f"*{ext}"))
        else:
            files.extend(directory.glob(f"*{ext}"))
    
    logger.info(f"Found {len(files)} documents to ingest")
    
    success_count = 0
    for file_path in files:
        if await ingest_file(str(file_path), department, tags):
            success_count += 1
    
    logger.info(f"Ingested {success_count}/{len(files)} documents successfully")


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system"
    )
    parser.add_argument(
        "path",
        help="Path to file or directory to ingest"
    )
    parser.add_argument(
        "--department",
        "-d",
        help="Department tag for the documents"
    )
    parser.add_argument(
        "--tags",
        "-t",
        help="Comma-separated tags"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recursively process directories"
    )
    
    args = parser.parse_args()
    
    # Parse tags
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    
    # Initialize services
    logger.info("Initializing services...")
    await initialize_services()
    
    # Process path
    path = Path(args.path)
    
    if path.is_file():
        await ingest_file(str(path), args.department, tags)
    elif path.is_dir():
        await ingest_directory(
            str(path),
            args.department,
            tags,
            recursive=not args.no_recursive
        )
    else:
        logger.error(f"Path does not exist: {path}")


if __name__ == "__main__":
    asyncio.run(main())
