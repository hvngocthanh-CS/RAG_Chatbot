"""
Convert extracted Table objects into retrieval-ready chunk dicts.

A "chunk" is the unit the vector store indexes: a piece of text plus
metadata. Tables get three flavours of chunks:

  1. table_summary  — natural language description of the table (name,
                      columns, sample rows). Matches semantic queries like
                      "what activities on Day 4?" or "passing requirements
                      for training" without the user knowing which table
                      contains the answer.

  2. table          — full key:value rendering of every row. Good for
                      "show me everything in this table" queries and for
                      the LLM to read when constructing a detailed answer.

  3. table_rows     — row-batch slices of large tables. Good for
                      "what is column C for row R?" point-lookup queries.
                      Kept small (3 rows) so a specific row's signal is
                      not diluted by unrelated rows.
"""
from typing import Any, Dict, List

from backend.models import Table
from .table_extractor import TableExtractor

_LARGE_TABLE_ROW_THRESHOLD = 10   # tables larger than this get row-batch chunks
_ROW_BATCH_SIZE = 3               # 3 rows/batch — reduces dilution for row lookups


class TableChunkBuilder:
    """Turn `Table` objects into chunk dicts ready for embedding."""

    def build(
        self,
        tables: List[Table],
        base_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return a flat list of chunks for every table."""
        chunks: List[Dict[str, Any]] = []
        for idx, table in enumerate(tables):
            chunks.append(self._summary_chunk(table, idx, base_metadata))
            chunks.append(self._whole_table_chunk(table, idx, base_metadata))
            if len(table.rows) > _LARGE_TABLE_ROW_THRESHOLD:
                chunks.extend(self._row_batch_chunks(table, idx, base_metadata))
        return chunks

    # ------------------------------------------------------------------
    # Chunk builders
    # ------------------------------------------------------------------

    def _summary_chunk(
        self,
        table: Table,
        table_idx: int,
        base_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Natural language summary for semantic retrieval.

        Matches queries that describe the table's *purpose* or *content*
        without knowing the exact table structure or row values.
        Samples first, middle, and last rows to cover the data range.
        """
        name = table.name or f"Table {table_idx + 1}"
        section = f" in section '{table.section}'" if table.section else ""
        page = f" on page {table.page_number}" if table.page_number else ""

        col_str = ", ".join(table.headers[:8])
        if len(table.headers) > 8:
            col_str += f" (and {len(table.headers) - 8} more)"

        # Sample first / middle / last rows for semantic coverage
        n = len(table.rows)
        sample_indices = sorted({0, n // 2, n - 1}) if n > 2 else list(range(n))
        sample_parts: List[str] = []
        for si in sample_indices[:3]:
            row = table.rows[si]
            pairs = [
                f"{table.headers[ci] if ci < len(table.headers) else f'Col{ci+1}'}: {v}"
                for ci, v in enumerate(row)
                if v and str(v).strip()
            ]
            if pairs:
                sample_parts.append("; ".join(pairs[:5]))

        summary = (
            f"'{name}' table{section}{page}: "
            f"{n} rows with columns: {col_str}."
        )
        if sample_parts:
            summary += f" Sample entries — {' | '.join(sample_parts)}."

        return {
            "content": summary,
            "metadata": {
                **base_metadata,
                "chunk_type": "table_summary",
                "table_index": table_idx,
                "table_name": name,
                "column_headers": table.headers,
                "page_number": table.page_number,
                "row_count": n,
            },
        }

    def _whole_table_chunk(
        self,
        table: Table,
        table_idx: int,
        base_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "content": TableExtractor.table_to_text(table),
            "metadata": {
                **base_metadata,
                "chunk_type": "table",
                "table_index": table_idx,
                "table_name": table.name or f"Table {table_idx + 1}",
                "column_headers": table.headers,
                "page_number": table.page_number,
                "row_count": len(table.rows),
            },
        }

    def _row_batch_chunks(
        self,
        table: Table,
        table_idx: int,
        base_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Split a large table into row batches, one chunk per batch.

        Column headers are repeated in every batch header so the embedding
        model always knows what columns this data belongs to.
        """
        chunks: List[Dict[str, Any]] = []
        table_label = table.name or f"Table {table_idx + 1}"
        col_header_str = ", ".join(table.headers[:8])
        if len(table.headers) > 8:
            col_header_str += ", ..."

        for start in range(0, len(table.rows), _ROW_BATCH_SIZE):
            batch = table.rows[start : start + _ROW_BATCH_SIZE]
            end = start + len(batch)

            lines = [
                f"Table: {table_label} (Rows {start + 1}–{end})"
                f" | Columns: {col_header_str}",
                "",
            ]
            for row_offset, row in enumerate(batch, start + 1):
                lines.append(f"Row {row_offset}:")
                for col_idx, cell_value in enumerate(row):
                    header = (
                        table.headers[col_idx]
                        if col_idx < len(table.headers)
                        else f"Column {col_idx + 1}"
                    )
                    lines.append(f"  {header}: {cell_value}")
                lines.append("")

            chunks.append({
                "content": "\n".join(lines),
                "metadata": {
                    **base_metadata,
                    "chunk_type": "table_rows",
                    "table_index": table_idx,
                    "table_name": table_label,
                    "column_headers": table.headers,
                    "row_range": f"{start + 1}-{end}",
                    "page_number": table.page_number,
                },
            })

        return chunks
