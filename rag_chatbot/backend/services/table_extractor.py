"""
Table Extraction Service.
Handles extraction and formatting of tables from documents.
"""
import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Service for extracting and processing tables from parsed documents.
    
    Features:
    - Clean table data
    - Handle merged cells
    - Convert to multiple output formats
    - Extract table metadata
    """
    
    def __init__(self):
        pass
    
    def extract_tables(self, parsed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and clean tables from parsed document content.
        
        Args:
            parsed_content: Output from DocumentParser
        
        Returns:
            List of cleaned table dictionaries
        """
        raw_tables = parsed_content.get("tables", [])
        cleaned_tables = []
        
        for idx, table in enumerate(raw_tables):
            cleaned = self._clean_table(table, idx)
            if cleaned:
                cleaned_tables.append(cleaned)
        
        logger.info(f"Extracted {len(cleaned_tables)} tables")
        return cleaned_tables
    
    def _clean_table(self, table: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """Clean and validate a table."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        # Skip empty tables
        if not rows:
            return None
        
        # Clean headers
        headers = [self._clean_cell(h) for h in headers]
        
        # If no headers, generate column names
        if not headers or all(not h for h in headers):
            max_cols = max(len(row) for row in rows) if rows else 0
            headers = [f"Column_{i+1}" for i in range(max_cols)]
        
        # Clean rows and ensure consistent column count
        cleaned_rows = []
        for row in rows:
            cleaned_row = [self._clean_cell(cell) for cell in row]
            # Pad or truncate to match header count
            while len(cleaned_row) < len(headers):
                cleaned_row.append("")
            cleaned_row = cleaned_row[:len(headers)]
            
            # Skip empty rows
            if any(cell for cell in cleaned_row):
                cleaned_rows.append(cleaned_row)
        
        if not cleaned_rows:
            return None
        
        return {
            "name": table.get("name", f"Table_{index + 1}"),
            "headers": headers,
            "rows": cleaned_rows,
            "page_number": table.get("page_number"),
            "section": table.get("section", "")
        }
    
    def _clean_cell(self, cell: Any) -> str:
        """Clean a table cell value."""
        if cell is None:
            return ""
        
        # Convert to string
        text = str(cell).strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove newlines within cells
        text = text.replace("\n", " ")
        
        return text
    
    def table_to_row_text(self, table: Dict[str, Any]) -> str:
        """
        Convert table to row-based text representation.
        
        Example output:
        Table: Sales Data
        
        Row 1:
        Product: A
        Price: 10
        Country: Japan
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
                lines.append(f"  {header}: {cell_value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def table_to_json(self, table: Dict[str, Any]) -> str:
        """
        Convert table to JSON representation.
        
        Example output:
        [
            {"product": "A", "price": 10, "country": "Japan"},
            {"product": "B", "price": 20, "country": "USA"}
        ]
        """
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        json_rows = []
        for row in rows:
            row_dict = {}
            for col_idx, cell_value in enumerate(row):
                header = headers[col_idx] if col_idx < len(headers) else f"column_{col_idx + 1}"
                # Clean header for JSON key
                key = header.lower().replace(" ", "_").replace("-", "_")
                # Try to parse numeric values
                row_dict[key] = self._parse_value(cell_value)
            json_rows.append(row_dict)
        
        return json.dumps(json_rows, indent=2, ensure_ascii=False)
    
    def table_to_markdown(self, table: Dict[str, Any]) -> str:
        """Convert table to Markdown format."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        lines = []
        
        # Table name as heading
        if table.get("name"):
            lines.append(f"### {table['name']}")
            lines.append("")
        
        # Header row
        lines.append("| " + " | ".join(headers) + " |")
        
        # Separator
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Data rows
        for row in rows:
            # Escape pipe characters in cells
            escaped_row = [cell.replace("|", "\\|") for cell in row]
            lines.append("| " + " | ".join(escaped_row) + " |")
        
        return "\n".join(lines)
    
    def _parse_value(self, value: str) -> Any:
        """Try to parse a string value to appropriate type."""
        if not value:
            return value
        
        # Try integer
        try:
            return int(value.replace(",", ""))
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value.replace(",", ""))
        except ValueError:
            pass
        
        return value
    
    def get_table_summary(self, table: Dict[str, Any]) -> str:
        """Generate a brief summary of a table."""
        headers = table.get("headers", [])
        row_count = len(table.get("rows", []))
        
        return f"Table '{table.get('name', 'Unnamed')}' with {len(headers)} columns ({', '.join(headers[:3])}{'...' if len(headers) > 3 else ''}) and {row_count} rows"
