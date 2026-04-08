"""
Domain models for document parsing and processing.

These dataclasses are shared across the ingestion pipeline:
  - DocumentParser (Step 1) produces them
  - DocumentPreprocessor (Step 2) transforms them
  - SectionChunker (Step 3) consumes them
  - TableExtractor (Step 4) operates on Table objects
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextBlock:
    """A contiguous block of text extracted from a document."""
    text: str
    page_number: Optional[int] = None
    block_type: str = "paragraph"       # "title" | "heading" | "paragraph"
    section: str = ""                   # Nearest heading above this block

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "page_number": self.page_number,
            "block_type": self.block_type,
            "section": self.section,
        }


@dataclass
class Table:
    """A table extracted from a document."""
    headers: List[str]
    rows: List[List[str]]
    page_number: Optional[int] = None
    name: str = ""
    section: str = ""

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def to_dict(self) -> dict:
        return {
            "headers": self.headers,
            "rows": self.rows,
            "page_number": self.page_number,
            "name": self.name,
            "section": self.section,
        }


@dataclass
class ParsedDocument:
    """Complete output of parsing a single file."""
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    page_count: int = 0
    file_type: str = ""

    def to_dict(self) -> dict:
        return {
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "tables": [t.to_dict() for t in self.tables],
            "metadata": {
                "page_count": self.page_count,
                "file_type": self.file_type,
                "text_block_count": len(self.text_blocks),
                "table_count": len(self.tables),
            },
        }
