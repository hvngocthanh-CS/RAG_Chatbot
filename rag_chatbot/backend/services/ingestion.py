"""
Document Ingestion Service.
Handles document parsing, chunking, table extraction, and embedding generation.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from backend.config import settings
from backend.services.document_parser import DocumentParser
from backend.services.chunker import TextChunker
from backend.services.table_extractor import TableExtractor
from backend.services import get_service

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """
    Service for ingesting documents into the RAG system.
    
    Pipeline:
    1. Parse document to extract text and layout
    2. Extract tables and convert to structured format
    3. Chunk text into segments
    4. Generate embeddings for chunks and tables
    5. Store in vector database
    """
    
    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.table_extractor = TableExtractor()
    
    @property
    def embedding_service(self):
        return get_service("embedding")
    
    @property
    def vector_store(self):
        return get_service("vector_store")
    
    async def process_document(
        self, 
        file_path: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a document through the full ingestion pipeline.
        
        Args:
            file_path: Path to the document file
            metadata: Document metadata (id, filename, etc.)
        
        Returns:
            Processing results including chunk count
        """
        try:
            logger.info(f"Processing document: {metadata['filename']}")
            
            # Step 1: Parse document
            parsed_content = await self.parser.parse(file_path)
            
            # Step 2: Extract tables
            tables = self.table_extractor.extract_tables(parsed_content)
            
            # Step 3: Chunk text content
            text_chunks = self.chunker.chunk_text(
                parsed_content["text_blocks"],
                metadata
            )
            
            # Step 4: Convert tables to chunks
            table_chunks = self._process_tables(tables, metadata)
            
            # Combine all chunks
            all_chunks = text_chunks + table_chunks
            
            if not all_chunks:
                logger.warning(f"No content extracted from document: {metadata['filename']}")
                return {
                    "status": "warning",
                    "message": "No content could be extracted from document",
                    "chunks_count": 0
                }
            
            # Step 5: Generate embeddings
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
            embeddings = await self.embedding_service.embed_documents(
                [chunk["content"] for chunk in all_chunks]
            )
            
            # Add embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk["embedding"] = embedding
            
            # Step 6: Store in vector database
            await self.vector_store.add_chunks(all_chunks, metadata)
            
            # Save processing metadata
            self._save_processing_metadata(metadata, len(all_chunks), len(tables))
            
            logger.info(f"Document processed successfully: {len(all_chunks)} chunks created")
            
            return {
                "status": "success",
                "document_id": metadata["document_id"],
                "chunks_count": len(all_chunks),
                "text_chunks": len(text_chunks),
                "table_chunks": len(table_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            self._save_processing_error(metadata, str(e))
            raise
    
    def _process_tables(
        self, 
        tables: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert extracted tables to chunk format.
        
        Tables are converted to row-based text representation for better retrieval.
        """
        table_chunks = []
        
        for idx, table in enumerate(tables):
            # Convert table to row-based text representation
            table_text = self._table_to_text(table)
            
            chunk = {
                "content": table_text,
                "metadata": {
                    **metadata,
                    "chunk_type": "table",
                    "table_index": idx,
                    "table_name": table.get("name", f"Table {idx + 1}"),
                    "column_headers": table.get("headers", []),
                    "page_number": table.get("page_number"),
                    "row_count": len(table.get("rows", []))
                }
            }
            table_chunks.append(chunk)
            
            # If table is large, also create individual row chunks for better retrieval
            if len(table.get("rows", [])) > 10:
                row_chunks = self._create_row_chunks(table, metadata, idx)
                table_chunks.extend(row_chunks)
        
        return table_chunks
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """
        Convert a table to row-based text representation.
        
        Example output:
        Table: Sales Data
        
        Row 1:
        Product: A
        Price: 10
        Country: Japan
        
        Row 2:
        Product: B
        Price: 20
        Country: USA
        """
        lines = []
        
        if table.get("name"):
            lines.append(f"Table: {table['name']}")
            lines.append("")
        
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        for row_idx, row in enumerate(rows, 1):
            lines.append(f"Row {row_idx}:")
            for col_idx, cell_value in enumerate(row):
                header = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"
                lines.append(f"{header}: {cell_value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_row_chunks(
        self, 
        table: Dict[str, Any], 
        metadata: Dict[str, Any],
        table_idx: int
    ) -> List[Dict[str, Any]]:
        """Create individual chunks for table rows (for large tables)."""
        row_chunks = []
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        # Group rows in batches of 5
        batch_size = 5
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            
            lines = [f"Table: {table.get('name', f'Table {table_idx + 1}')} (Rows {i+1}-{i+len(batch)})"]
            lines.append("")
            
            for row_idx, row in enumerate(batch, i + 1):
                lines.append(f"Row {row_idx}:")
                for col_idx, cell_value in enumerate(row):
                    header = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"
                    lines.append(f"{header}: {cell_value}")
                lines.append("")
            
            chunk = {
                "content": "\n".join(lines),
                "metadata": {
                    **metadata,
                    "chunk_type": "table_rows",
                    "table_index": table_idx,
                    "row_range": f"{i+1}-{i+len(batch)}",
                    "page_number": table.get("page_number")
                }
            }
            row_chunks.append(chunk)
        
        return row_chunks
    
    def _save_processing_metadata(
        self, 
        metadata: Dict[str, Any], 
        chunks_count: int,
        tables_count: int
    ):
        """Save document processing metadata to file."""
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
        
        processing_info = {
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "processed_at": datetime.utcnow().isoformat(),
            "status": "completed",
            "chunks_count": chunks_count,
            "tables_count": tables_count
        }
        
        path = os.path.join(
            settings.PROCESSED_DIR, 
            f"{metadata['document_id']}_meta.json"
        )
        
        with open(path, "w") as f:
            json.dump(processing_info, f, indent=2)
    
    def _save_processing_error(self, metadata: Dict[str, Any], error: str):
        """Save processing error information."""
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
        
        error_info = {
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "processed_at": datetime.utcnow().isoformat(),
            "status": "failed",
            "error": error
        }
        
        path = os.path.join(
            settings.PROCESSED_DIR,
            f"{metadata['document_id']}_meta.json"
        )
        
        with open(path, "w") as f:
            json.dump(error_info, f, indent=2)
