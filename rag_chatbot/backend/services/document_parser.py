"""
Document Parser Service.
Handles parsing of various document formats (PDF, DOCX, TXT).
"""
import os
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Multi-format document parser with layout detection.
    
    Supports:
    - PDF files (with table detection)
    - DOCX files
    - TXT/MD files
    """
    
    def __init__(self):
        self._setup_parsers()
    
    def _setup_parsers(self):
        """Initialize document parsing libraries."""
        pass  # Lazy loading to avoid heavy imports at startup
    
    async def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document and extract structured content.
        
        Args:
            file_path: Path to the document
        
        Returns:
            Dictionary containing:
            - text_blocks: List of text segments with metadata
            - tables: List of detected tables
            - metadata: Document-level metadata
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".pdf":
            return await self._parse_pdf(file_path)
        elif file_ext == ".docx":
            return await self._parse_docx(file_path)
        elif file_ext in [".txt", ".md"]:
            return await self._parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    async def _parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF document with layout detection.
        
        Uses pypdf for text extraction and pdfplumber for tables.
        """
        import pypdf
        import pdfplumber
        
        text_blocks = []
        tables = []
        
        # Extract text using pypdf
        with open(file_path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    # Detect title on page 1 (first non-empty line)
                    if page_num == 1:
                        lines = [l.strip() for l in text.split('\n') if l.strip()]
                        if lines:
                            title = lines[0]
                            # Add title as separate block
                            text_blocks.append({
                                "text": f"DOCUMENT TITLE: {title}",
                                "page_number": page_num,
                                "type": "title"
                            })
                    
                    # Add full page text
                    text_blocks.append({
                        "text": text.strip(),
                        "page_number": page_num,
                        "type": "text"
                    })
        
        # Extract tables using pdfplumber (better table detection)
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                
                for table_idx, table_data in enumerate(page_tables):
                    if table_data and len(table_data) > 1:
                        # First row is usually headers
                        headers = [str(cell) if cell else "" for cell in table_data[0]]
                        rows = [
                            [str(cell) if cell else "" for cell in row]
                            for row in table_data[1:]
                        ]
                        
                        tables.append({
                            "headers": headers,
                            "rows": rows,
                            "page_number": page_num,
                            "table_index": table_idx,
                            "name": f"Table_{page_num}_{table_idx + 1}"
                        })
        
        return {
            "text_blocks": text_blocks,
            "tables": tables,
            "metadata": {
                "page_count": len(pdf_reader.pages),
                "file_type": "pdf"
            }
        }
    
    async def _parse_docx(self, file_path: str) -> Dict[str, Any]:
        """
        Parse DOCX document.
        
        Uses python-docx for extraction.
        """
        from docx import Document
        from docx.table import Table
        
        doc = Document(file_path)
        
        text_blocks = []
        tables = []
        current_section = ""
        
        for element in doc.element.body:
            if element.tag.endswith('p'):
                # Paragraph
                para = None
                for p in doc.paragraphs:
                    if p._element is element:
                        para = p
                        break
                
                if para and para.text.strip():
                    # Check if it's a heading
                    style_name = para.style.name if para.style else ""
                    
                    if "Heading" in style_name:
                        current_section = para.text.strip()
                    
                    text_blocks.append({
                        "text": para.text.strip(),
                        "type": "heading" if "Heading" in style_name else "paragraph",
                        "section": current_section,
                        "style": style_name
                    })
            
            elif element.tag.endswith('tbl'):
                # Table
                for tbl in doc.tables:
                    if tbl._element is element:
                        table_data = self._extract_docx_table(tbl)
                        if table_data:
                            table_data["section"] = current_section
                            tables.append(table_data)
                        break
        
        return {
            "text_blocks": text_blocks,
            "tables": tables,
            "metadata": {
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables),
                "file_type": "docx"
            }
        }
    
    def _extract_docx_table(self, table) -> Dict[str, Any]:
        """Extract table data from a DOCX table object."""
        rows = []
        
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                row_data.append(cell.text.strip())
            rows.append(row_data)
        
        if not rows:
            return None
        
        # First row as headers
        headers = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        
        return {
            "headers": headers,
            "rows": data_rows,
            "name": f"Table_{len(rows)}rows"
        }
    
    async def _parse_text(self, file_path: str) -> Dict[str, Any]:
        """
        Parse plain text or markdown files.
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        text_blocks = []
        for para in paragraphs:
            # Detect markdown headers
            lines = para.split("\n")
            first_line = lines[0] if lines else ""
            
            if first_line.startswith("#"):
                block_type = "heading"
            else:
                block_type = "paragraph"
            
            text_blocks.append({
                "text": para,
                "type": block_type
            })
        
        return {
            "text_blocks": text_blocks,
            "tables": [],  # No table detection for plain text
            "metadata": {
                "character_count": len(content),
                "file_type": "text"
            }
        }
