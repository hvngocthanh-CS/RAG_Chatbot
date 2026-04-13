"""
Convert extracted Table objects into retrieval-ready chunk dicts.

A "chunk" is the unit the vector store indexes: a piece of text plus
metadata. Tables get two flavours of chunks:

  1. One whole-table chunk (good for "what does table X contain?" queries).
  2. For tables with many rows, additional row-batch chunks
     (good for "what is the value of column C in row R?" queries),
     so the retriever can hit a small slice instead of the entire table.
"""
from typing import Any, Dict, List

from backend.models import Table
from .table_extractor import TableExtractor

# Tables larger than this get extra row-batch chunks for finer retrieval.
_LARGE_TABLE_ROW_THRESHOLD = 10
_ROW_BATCH_SIZE = 5


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
            chunks.append(self._whole_table_chunk(table, idx, base_metadata))
            if len(table.rows) > _LARGE_TABLE_ROW_THRESHOLD:
                chunks.extend(self._row_batch_chunks(table, idx, base_metadata))
        return chunks

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
        """Split a large table into row batches, one chunk per batch."""
        chunks: List[Dict[str, Any]] = []
        table_label = table.name or f"Table {table_idx + 1}"

        for start in range(0, len(table.rows), _ROW_BATCH_SIZE):
            batch = table.rows[start : start + _ROW_BATCH_SIZE]
            end = start + len(batch)

            lines = [f"Table: {table_label} (Rows {start + 1}-{end})", ""]
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
                    "row_range": f"{start + 1}-{end}",
                    "page_number": table.page_number,
                },
            })

        return chunks
