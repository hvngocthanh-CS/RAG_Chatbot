"""
Table Extractor — converts parsed Table objects into text for embedding.

The row-based text format preserves table structure better than CSV/Markdown
for LLM comprehension and embedding quality.
"""
import logging
from typing import List

from backend.models import Table, ParsedDocument

logger = logging.getLogger(__name__)


class TableExtractor:
    """Converts Table objects from DocumentParser into embeddable text chunks."""

    def extract_tables(self, parsed: ParsedDocument) -> List[Table]:
        """Return only non-empty tables from a parsed document."""
        tables = [t for t in parsed.tables if not t.is_empty]
        logger.info("Extracted %d non-empty tables", len(tables))
        return tables

    @staticmethod
    def table_to_text(table: Table) -> str:
        """
        Convert a table to row-based text optimized for embedding and LLM reading.
        """
        lines = []

        title_parts = []
        if table.name:
            title_parts.append(table.name)
        if table.section:
            title_parts.append(f"Section: {table.section}")
        if table.page_number:
            title_parts.append(f"Page {table.page_number}")
        if title_parts:
            lines.append(f"Table: {' | '.join(title_parts)}")

        # Column headers give the embedding model critical context about what
        # this table is about — without them, a row chunk looks like generic
        # key:value text with no topic signal.
        if table.headers:
            lines.append(f"Columns: {', '.join(table.headers)}")
        lines.append("")

        for row_idx, row in enumerate(table.rows, 1):
            lines.append(f"Row {row_idx}:")
            for col_idx, value in enumerate(row):
                header = (
                    table.headers[col_idx]
                    if col_idx < len(table.headers)
                    else f"Column {col_idx + 1}"
                )
                lines.append(f"  {header}: {value}")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def table_summary(table: Table) -> str:
        """One-line summary for logging / metadata."""
        col_preview = ", ".join(table.headers[:4])
        if len(table.headers) > 4:
            col_preview += ", ..."
        return (
            f"'{table.name}' — {len(table.headers)} columns ({col_preview}), "
            f"{len(table.rows)} rows"
        )
