"""
Document Ingestion Service — orchestrates the end-to-end ingestion pipeline.

Responsibilities are split across small collaborators so this file stays
short and readable:

    parse  -> preprocess -> extract tables -> chunk text
                                            -> build table chunks
                                            -> embed all chunks
                                            -> store in vector DB
                                            -> write metadata sidecar

Each collaborator lives in its own module (`parser.py`, `preprocessor.py`,
`chunker.py`, `table_extractor.py`, `table_chunk_builder.py`,
`metadata_writer.py`). This class only wires them together.
"""
import logging
from typing import Any, Dict, List

from backend.config import settings
from backend.core.exceptions import IngestionError
from backend.services import get_service

from .chunker import SectionChunker
from .metadata_writer import IngestionMetadataWriter
from .parser import DocumentParser
from .preprocessor import DocumentPreprocessor
from .table_chunk_builder import TableChunkBuilder
from .table_extractor import TableExtractor

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """High-level entry point for ingesting a document into the RAG index."""

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
        self.table_chunk_builder = TableChunkBuilder()
        self.metadata_writer = IngestionMetadataWriter()

    # Embedding & vector store are global singletons (see backend.services).
    # We pull them lazily so this class can be constructed before they exist
    # (e.g. in unit tests).
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
        """
        Run the full ingestion pipeline for one document.

        Args:
            file_path: absolute path to the uploaded file.
            metadata: must contain at least `document_id` and `filename`.
                      Forwarded into every chunk's metadata.

        Returns:
            A summary dict: status, document_id, chunks_count, breakdown.

        Raises:
            IngestionError: any step of the pipeline failed. The original
                exception is chained via `raise ... from`.
        """
        try:
            chunks = await self._build_chunks(file_path, metadata)

            if not chunks:
                logger.warning(
                    "No content extracted from document: %s", metadata["filename"],
                )
                return {
                    "status": "warning",
                    "message": "No content could be extracted from document",
                    "chunks_count": 0,
                }

            await self._embed_and_store(chunks, metadata)

            _table_types = {"table", "table_rows", "table_summary"}
            text_chunks = sum(1 for c in chunks if c["metadata"].get("chunk_type") not in _table_types)
            table_chunks = len(chunks) - text_chunks
            tables_count = sum(1 for c in chunks if c["metadata"].get("chunk_type") == "table")

            self.metadata_writer.write_success(metadata, len(chunks), tables_count)
            logger.info("Document processed: %d chunks", len(chunks))

            return {
                "status": "success",
                "document_id": metadata["document_id"],
                "chunks_count": len(chunks),
                "text_chunks": text_chunks,
                "table_chunks": table_chunks,
            }

        except Exception as e:
            logger.error("Ingestion failed for %s: %s", metadata.get("filename"), e, exc_info=True)
            self.metadata_writer.write_failure(metadata, str(e))
            raise IngestionError(str(e)) from e

    # ---- internal steps ---------------------------------------------------

    async def _build_chunks(
        self,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Parse + preprocess + chunk text + build table chunks."""
        parsed = await self.parser.parse(file_path)
        parsed = self.preprocessor.preprocess(parsed)

        tables = self.table_extractor.extract_tables(parsed)

        # The chunker can use the embedding model to pick semantic boundaries.
        embedding_model = self.embedding_service.model if self.embedding_service else None
        if embedding_model is not None:
            self.chunker.set_embedding_model(embedding_model)

        text_block_dicts = [b.to_dict() for b in parsed.text_blocks]
        text_chunks = self.chunker.chunk_text(text_block_dicts, metadata)
        table_chunks = self.table_chunk_builder.build(tables, metadata)

        return text_chunks + table_chunks

    async def _embed_and_store(
        self,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        """Generate embeddings for every chunk and persist them."""
        logger.info("Generating embeddings for %d chunks", len(chunks))
        embeddings = await self.embedding_service.embed_documents(
            [chunk["content"] for chunk in chunks]
        )
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        await self.vector_store.add_chunks(chunks, metadata)
