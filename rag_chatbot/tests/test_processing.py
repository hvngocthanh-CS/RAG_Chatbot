"""
Tests for document processing services.
"""
import pytest
from pathlib import Path

from backend.services.section_chunker import SectionChunker
from backend.services.table_extractor import TableExtractor
from backend.services.document_parser import ParsedDocument, TextBlock, Table


class TestSectionChunker:
    """Tests for the SectionChunker service."""

    def setup_method(self):
        self.chunker = SectionChunker(
            max_chunk_tokens=200, min_chunk_tokens=20, overlap_sentences=1,
        )

    def test_count_tokens(self):
        text = "Hello world, this is a test sentence."
        count = self.chunker.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_chunk_empty_input(self):
        metadata = {"document_id": "test", "filename": "test.txt"}
        chunks = self.chunker.chunk_text([], metadata)
        assert chunks == []

    def test_chunk_single_paragraph(self):
        """A single paragraph under no heading produces one chunk."""
        text_blocks = [
            {"text": "This is a short paragraph.", "type": "paragraph", "page_number": 1}
        ]
        metadata = {"document_id": "test", "filename": "test.txt"}
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        assert len(chunks) == 1
        assert "This is a short paragraph." in chunks[0]["content"]

    def test_heading_creates_section_prefix(self):
        """Heading text is prepended as [Heading] prefix in the chunk."""
        text_blocks = [
            {"text": "Introduction", "type": "heading", "page_number": 1},
            {"text": "Welcome to the guide.", "type": "paragraph", "page_number": 1},
        ]
        metadata = {"document_id": "test", "filename": "test.txt"}
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        assert len(chunks) == 1
        assert chunks[0]["content"].startswith("[Introduction]")
        assert "Welcome to the guide." in chunks[0]["content"]

    def test_multiple_sections(self):
        """Each heading starts a new chunk when sections are large enough."""
        # Use enough text so neither section is below min_chunk_tokens
        para_a = "Content of section A. " * 10
        para_b = "Content of section B. " * 10
        text_blocks = [
            {"text": "Section A", "type": "heading", "page_number": 1},
            {"text": para_a, "type": "paragraph", "page_number": 1},
            {"text": "Section B", "type": "heading", "page_number": 2},
            {"text": para_b, "type": "paragraph", "page_number": 2},
        ]
        metadata = {"document_id": "test", "filename": "test.txt"}
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        assert len(chunks) == 2
        assert "[Section A]" in chunks[0]["content"]
        assert "[Section B]" in chunks[1]["content"]

    def test_preserves_metadata(self):
        text_blocks = [
            {"text": "Test content here.", "type": "paragraph", "page_number": 1}
        ]
        metadata = {
            "document_id": "doc123",
            "filename": "test.pdf",
            "department": "HR",
        }
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        assert len(chunks) == 1
        m = chunks[0]["metadata"]
        assert m["document_id"] == "doc123"
        assert m["filename"] == "test.pdf"
        assert m["department"] == "HR"
        assert m["chunking_method"] == "section"
        assert m["chunk_index"] == 0

    def test_page_numbers_tracked(self):
        text_blocks = [
            {"text": "Heading", "type": "heading", "page_number": 3},
            {"text": "Para on page 3.", "type": "paragraph", "page_number": 3},
            {"text": "Para on page 4.", "type": "paragraph", "page_number": 4},
        ]
        metadata = {"document_id": "t", "filename": "t.pdf"}
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        assert chunks[0]["metadata"]["page_numbers"] == [3, 4]
        assert chunks[0]["metadata"]["page_number"] == 3

    def test_large_section_split(self):
        """A section exceeding max_chunk_tokens is split into multiple chunks."""
        long_text = ". ".join([f"Sentence number {i}" for i in range(80)]) + "."
        text_blocks = [
            {"text": "Big Section", "type": "heading", "page_number": 1},
            {"text": long_text, "type": "paragraph", "page_number": 1},
        ]
        metadata = {"document_id": "t", "filename": "t.pdf"}
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        assert len(chunks) > 1
        # Every chunk should carry the heading prefix
        for chunk in chunks:
            assert chunk["content"].startswith("[Big Section]")

    def test_small_chunks_merged(self):
        """Tiny trailing chunks are merged into predecessors."""
        chunker = SectionChunker(
            max_chunk_tokens=500, min_chunk_tokens=80, overlap_sentences=1,
        )
        text_blocks = [
            {"text": "Main Section", "type": "heading", "page_number": 1},
            {"text": "A " * 100, "type": "paragraph", "page_number": 1},
            {"text": "Tiny Section", "type": "heading", "page_number": 1},
            {"text": "Short.", "type": "paragraph", "page_number": 1},
        ]
        metadata = {"document_id": "t", "filename": "t.pdf"}
        chunks = chunker.chunk_text(text_blocks, metadata)
        # The tiny section should be merged into the main one
        assert len(chunks) == 1

    def test_section_metadata(self):
        text_blocks = [
            {"text": "Policy Overview", "type": "heading", "page_number": 1},
            {"text": "Details about the policy.", "type": "paragraph", "page_number": 1},
        ]
        metadata = {"document_id": "t", "filename": "t.pdf"}
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        assert chunks[0]["metadata"]["section"] == "Policy Overview"
        assert chunks[0]["metadata"]["sections"] == ["Policy Overview"]


class TestTableExtractor:
    """Tests for the TableExtractor service."""

    def setup_method(self):
        self.extractor = TableExtractor()

    def test_extract_skips_empty_table(self):
        parsed = ParsedDocument(
            tables=[Table(headers=[], rows=[])],
            page_count=1,
            file_type="pdf",
        )
        result = self.extractor.extract_tables(parsed)
        assert result == []

    def test_extract_valid_table(self):
        parsed = ParsedDocument(
            tables=[Table(
                headers=["Product", "Price"],
                rows=[["A", "10"], ["B", "20"]],
            )],
            page_count=1,
            file_type="pdf",
        )
        result = self.extractor.extract_tables(parsed)
        assert len(result) == 1
        assert len(result[0].headers) == 2

    def test_table_to_text(self):
        table = Table(
            name="Test Table",
            headers=["Name", "Value"],
            rows=[["A", "1"], ["B", "2"]],
        )
        text = TableExtractor.table_to_text(table)
        assert "Test Table" in text
        assert "Row 1:" in text
        assert "Name: A" in text


class TestIntegration:
    """Integration tests for the processing pipeline."""

    def test_full_chunking_pipeline(self):
        """Test the complete chunking pipeline with SectionChunker."""
        parsed = ParsedDocument(
            text_blocks=[
                TextBlock(text="Introduction to the document.", page_number=1, block_type="heading"),
                TextBlock(
                    text="This is the main content of the document. It contains important information.",
                    page_number=1,
                    block_type="paragraph",
                ),
            ],
            tables=[Table(headers=["Item", "Value"], rows=[["A", "100"]])],
            page_count=1,
            file_type="pdf",
        )

        # Extract tables
        extractor = TableExtractor()
        tables = extractor.extract_tables(parsed)
        assert len(tables) == 1

        # Chunk text (convert TextBlocks to dicts as the chunker expects)
        text_block_dicts = [b.to_dict() for b in parsed.text_blocks]
        chunker = SectionChunker(max_chunk_tokens=500, min_chunk_tokens=20)
        metadata = {"document_id": "test", "filename": "test.pdf"}
        chunks = chunker.chunk_text(text_block_dicts, metadata)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert "[Introduction to the document.]" in chunks[0]["content"]
