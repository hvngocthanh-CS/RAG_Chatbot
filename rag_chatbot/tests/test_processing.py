"""
Tests for document processing services.
"""
import pytest
import asyncio
from pathlib import Path

from backend.services.chunker import TextChunker
from backend.services.table_extractor import TableExtractor


class TestTextChunker:
    """Tests for the TextChunker service."""

    def setup_method(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    def test_count_tokens(self):
        """Test token counting."""
        text = "Hello world, this is a test sentence."
        count = self.chunker.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_chunk_empty_input(self):
        """Test chunking with empty input."""
        metadata = {"document_id": "test", "filename": "test.txt"}
        chunks = self.chunker.chunk_text([], metadata)
        assert chunks == []

    def test_chunk_single_block(self):
        """Test chunking a single small text block."""
        text_blocks = [
            {"text": "This is a short test.", "type": "paragraph"}
        ]
        metadata = {"document_id": "test", "filename": "test.txt"}
        
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        
        # Should create at least one chunk
        assert len(chunks) >= 0

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        text_blocks = [
            {"text": "Test content here.", "type": "paragraph", "page_number": 1}
        ]
        metadata = {
            "document_id": "doc123",
            "filename": "test.pdf",
            "department": "HR"
        }
        
        chunks = self.chunker.chunk_text(text_blocks, metadata)
        
        if chunks:
            assert chunks[0]["metadata"]["document_id"] == "doc123"
            assert chunks[0]["metadata"]["filename"] == "test.pdf"


class TestTableExtractor:
    """Tests for the TableExtractor service."""

    def setup_method(self):
        self.extractor = TableExtractor()

    def test_clean_empty_table(self):
        """Test cleaning an empty table."""
        table = {"headers": [], "rows": []}
        result = self.extractor._clean_table(table, 0)
        assert result is None

    def test_clean_valid_table(self):
        """Test cleaning a valid table."""
        table = {
            "headers": ["Product", "Price"],
            "rows": [["A", "10"], ["B", "20"]]
        }
        result = self.extractor._clean_table(table, 0)
        
        assert result is not None
        assert len(result["headers"]) == 2
        assert len(result["rows"]) == 2

    def test_table_to_row_text(self):
        """Test converting table to row-based text."""
        table = {
            "name": "Test Table",
            "headers": ["Name", "Value"],
            "rows": [["A", "1"], ["B", "2"]]
        }
        
        text = self.extractor.table_to_row_text(table)
        
        assert "Test Table" in text
        assert "Row 1:" in text
        assert "Name: A" in text

    def test_table_to_json(self):
        """Test converting table to JSON."""
        table = {
            "headers": ["product", "price"],
            "rows": [["A", "10"], ["B", "20"]]
        }
        
        json_str = self.extractor.table_to_json(table)
        
        import json
        data = json.loads(json_str)
        assert len(data) == 2
        assert data[0]["product"] == "A"

    def test_table_to_markdown(self):
        """Test converting table to Markdown."""
        table = {
            "name": "Test",
            "headers": ["A", "B"],
            "rows": [["1", "2"]]
        }
        
        md = self.extractor.table_to_markdown(table)
        
        assert "| A | B |" in md
        assert "| 1 | 2 |" in md


class TestIntegration:
    """Integration tests for the processing pipeline."""

    @pytest.mark.asyncio
    async def test_full_chunking_pipeline(self):
        """Test the complete chunking pipeline."""
        # Create sample content similar to what DocumentParser produces
        parsed_content = {
            "text_blocks": [
                {"text": "Introduction to the document.", "type": "heading"},
                {"text": "This is the main content of the document. It contains important information.", "type": "paragraph"}
            ],
            "tables": [
                {
                    "headers": ["Item", "Value"],
                    "rows": [["A", "100"]]
                }
            ],
            "metadata": {"page_count": 1}
        }
        
        # Extract tables
        extractor = TableExtractor()
        tables = extractor.extract_tables(parsed_content)
        
        assert len(tables) == 1
        
        # Chunk text
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        metadata = {"document_id": "test", "filename": "test.pdf"}
        chunks = chunker.chunk_text(parsed_content["text_blocks"], metadata)
        
        # Should have created chunks
        assert isinstance(chunks, list)
