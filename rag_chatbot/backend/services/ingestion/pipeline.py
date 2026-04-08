"""
Document Ingestion Service.
Orchestrates document parsing, preprocessing, chunking, table extraction,
and embedding generation.
"""
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from backend.config import settings
from backend.models import Table
from backend.services import get_service
from .parser import DocumentParser
from .preprocessor import DocumentPreprocessor
from .chunker import SectionChunker
from .table_extractor import TableExtractor

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """
    Service for ingesting documents into the RAG system.

    Pipeline:
    1. Parse document to extract text and layout
    2. Preprocess (clean, merge, deduplicate blocks)
    3. Chunk text into section-aware segments
    4. Extract tables and convert to text chunks
    5. Generate embeddings for all chunks
    6. Store in vector database
    """

    def __init__(self):
        self.parser = DocumentParser()
        self.preprocessor = DocumentPreprocessor()
        self.chunker = SectionChunker(
            max_chunk_tokens=settings.SECTION_MAX_CHUNK_TOKENS,
            min_chunk_tokens=settings.SECTION_MIN_CHUNK_TOKENS,
            overlap_sentences=settings.SECTION_OVERLAP_SENTENCES,
            semantic_look_back=settings.SECTION_SEMANTIC_LOOK_BACK,
            semantic_min_score=settings.SECTION_SEMANTIC_MIN_SCORE,
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
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a document through the full ingestion pipeline."""
        try:
            # Step 1: Parse document
            parsed = await self.parser.parse(file_path)

            # Step 2: Preprocess (clean text, merge blocks, deduplicate)
            parsed = self.preprocessor.preprocess(parsed)

            # Step 3: Extract tables
            tables = self.table_extractor.extract_tables(parsed)

            # Step 4: Chunk text content (section-aware + semantic embedding)
            embedding_model = self.embedding_service.model
            if embedding_model is not None:
                self.chunker.set_embedding_model(embedding_model)

            text_block_dicts = [b.to_dict() for b in parsed.text_blocks]
            text_chunks = self.chunker.chunk_text(text_block_dicts, metadata)

            # Step 5: Convert tables to chunks
            table_chunks = self._process_tables(tables, metadata)

            # Combine all chunks
            all_chunks = text_chunks + table_chunks

            if not all_chunks:
                logger.warning(
                    "No content extracted from document: %s",
                    metadata["filename"],
                )
                return {
                    "status": "warning",
                    "message": "No content could be extracted from document",
                    "chunks_count": 0,
                }

            # Step 6: Generate embeddings
            logger.info("Generating embeddings for %d chunks", len(all_chunks))
            embeddings = await self.embedding_service.embed_documents(
                [chunk["content"] for chunk in all_chunks]
            )

            # Add embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk["embedding"] = embedding

            # Step 7: Store in vector database
            await self.vector_store.add_chunks(all_chunks, metadata)

            # Save processing metadata
            self._save_processing_metadata(metadata, len(all_chunks), len(tables))

            logger.info(
                "Document processed successfully: %d chunks created",
                len(all_chunks),
            )

            return {
                "status": "success",
                "document_id": metadata["document_id"],
                "chunks_count": len(all_chunks),
                "text_chunks": len(text_chunks),
                "table_chunks": len(table_chunks),
            }

        except Exception as e:
            logger.error("Error processing document: %s", e, exc_info=True)
            self._save_processing_error(metadata, str(e))
            raise

    def _process_tables(
        self,
        tables: List[Table],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Convert extracted Table objects to chunk dicts."""
        table_chunks = []

        for idx, table in enumerate(tables):
            table_text = TableExtractor.table_to_text(table)

            chunk = {
                "content": table_text,
                "metadata": {
                    **metadata,
                    "chunk_type": "table",
                    "table_index": idx,
                    "table_name": table.name or f"Table {idx + 1}",
                    "column_headers": table.headers,
                    "page_number": table.page_number,
                    "row_count": len(table.rows),
                },
            }
            table_chunks.append(chunk)

            # Large tables: also create row-batch chunks for finer retrieval
            if len(table.rows) > 10:
                row_chunks = self._create_row_chunks(table, metadata, idx)
                table_chunks.extend(row_chunks)

        return table_chunks

    def _create_row_chunks(
        self,
        table: Table,
        metadata: Dict[str, Any],
        table_idx: int,
    ) -> List[Dict[str, Any]]:
        """Create individual chunks for table rows (for large tables)."""
        row_chunks = []
        batch_size = 5

        for i in range(0, len(table.rows), batch_size):
            batch = table.rows[i : i + batch_size]

            lines = [
                f"Table: {table.name or f'Table {table_idx + 1}'} "
                f"(Rows {i + 1}-{i + len(batch)})",
                "",
            ]

            for row_idx, row in enumerate(batch, i + 1):
                lines.append(f"Row {row_idx}:")
                for col_idx, cell_value in enumerate(row):
                    header = (
                        table.headers[col_idx]
                        if col_idx < len(table.headers)
                        else f"Column {col_idx + 1}"
                    )
                    lines.append(f"  {header}: {cell_value}")
                lines.append("")

            row_chunks.append({
                "content": "\n".join(lines),
                "metadata": {
                    **metadata,
                    "chunk_type": "table_rows",
                    "table_index": table_idx,
                    "row_range": f"{i + 1}-{i + len(batch)}",
                    "page_number": table.page_number,
                },
            })

        return row_chunks

    def _save_processing_metadata(
        self,
        metadata: Dict[str, Any],
        chunks_count: int,
        tables_count: int,
    ):
        """Save document processing metadata to file."""
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

        processing_info = {
            "document_id": metadata["document_id"],
            "filename": metadata["filename"],
            "processed_at": datetime.utcnow().isoformat(),
            "status": "completed",
            "chunks_count": chunks_count,
            "tables_count": tables_count,
        }

        path = os.path.join(
            settings.PROCESSED_DIR,
            f"{metadata['document_id']}_meta.json",
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
            "error": error,
        }

        path = os.path.join(
            settings.PROCESSED_DIR,
            f"{metadata['document_id']}_meta.json",
        )

        with open(path, "w") as f:
            json.dump(error_info, f, indent=2)
