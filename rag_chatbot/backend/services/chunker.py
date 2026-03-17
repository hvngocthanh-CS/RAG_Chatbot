"""
Text Chunking Service.
Implements intelligent text chunking with semantic boundaries.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 500  # tokens
    chunk_overlap: int = 50  # tokens
    min_chunk_size: int = 100  # tokens
    separator_pattern: str = r'\n\n|\n(?=[A-Z])|(?<=[.!?])\s+'


class TextChunker:
    """
    Intelligent text chunker that respects semantic boundaries.
    
    Features:
    - Token-based chunking (using tiktoken)
    - Respects paragraph and sentence boundaries
    - Configurable overlap for context preservation
    - Handles headings and sections
    """
    
    def __init__(
        self, 
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = 100
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to character-based estimation
            self.tokenizer = None
            logger.warning("Tiktoken not available, using character-based chunking")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: estimate 1 token per 4 characters
        return len(text) // 4
    
    def chunk_text(
        self, 
        text_blocks: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk text blocks into smaller segments.
        
        Args:
            text_blocks: List of text blocks from document parser
            metadata: Document metadata
        
        Returns:
            List of chunks with content and metadata
        """
        chunks = []
        current_chunk_text = ""
        current_chunk_meta = {
            "page_numbers": [],
            "sections": []
        }
        
        for block in text_blocks:
            block_text = block.get("text", "")
            block_type = block.get("type", "paragraph")
            page_number = block.get("page_number")
            section = block.get("section", "")
            
            # Handle title - keep as separate chunk
            if block_type == "title":
                chunks.append(self._create_chunk(
                    block_text,
                    metadata,
                    {"page_numbers": [page_number], "sections": []},
                    len(chunks)
                ))
                continue
            
            # Handle headings
            if block_type == "heading":
                # Save current chunk if exists
                if current_chunk_text and self.count_tokens(current_chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        current_chunk_text, 
                        metadata, 
                        current_chunk_meta,
                        len(chunks)
                    ))
                    current_chunk_text = ""
                    current_chunk_meta = {"page_numbers": [], "sections": []}
                
                # Start new chunk with heading
                current_chunk_text = block_text + "\n\n"
                if page_number:
                    current_chunk_meta["page_numbers"].append(page_number)
                if section:
                    current_chunk_meta["sections"].append(section)
                continue
            
            # Calculate if adding this block would exceed chunk size
            potential_text = current_chunk_text + block_text + "\n\n"
            potential_tokens = self.count_tokens(potential_text)
            
            if potential_tokens <= self.chunk_size:
                # Block fits in current chunk
                current_chunk_text = potential_text
                if page_number and page_number not in current_chunk_meta["page_numbers"]:
                    current_chunk_meta["page_numbers"].append(page_number)
                if section and section not in current_chunk_meta["sections"]:
                    current_chunk_meta["sections"].append(section)
            else:
                # Block doesn't fit, need to split
                if current_chunk_text and self.count_tokens(current_chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        current_chunk_text.strip(),
                        metadata,
                        current_chunk_meta,
                        len(chunks)
                    ))
                
                # Split the block if it's too large
                if self.count_tokens(block_text) > self.chunk_size:
                    sub_chunks = self._split_large_block(block_text, block, metadata, len(chunks))
                    chunks.extend(sub_chunks)
                    current_chunk_text = ""
                    current_chunk_meta = {"page_numbers": [], "sections": []}
                else:
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk_text)
                    current_chunk_text = overlap_text + block_text + "\n\n"
                    current_chunk_meta = {"page_numbers": [], "sections": []}
                    if page_number:
                        current_chunk_meta["page_numbers"].append(page_number)
                    if section:
                        current_chunk_meta["sections"].append(section)
        
        # Don't forget the last chunk
        if current_chunk_text and self.count_tokens(current_chunk_text) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                current_chunk_text.strip(),
                metadata,
                current_chunk_meta,
                len(chunks)
            ))
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def _split_large_block(
        self,
        text: str,
        block: Dict[str, Any],
        metadata: Dict[str, Any],
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """Split a large text block into multiple chunks."""
        chunks = []
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_text = ""
        for sentence in sentences:
            potential_text = current_text + sentence + " "
            
            if self.count_tokens(potential_text) <= self.chunk_size:
                current_text = potential_text
            else:
                if current_text:
                    chunk_meta = {
                        "page_numbers": [block.get("page_number")] if block.get("page_number") else [],
                        "sections": [block.get("section")] if block.get("section") else []
                    }
                    chunks.append(self._create_chunk(
                        current_text.strip(),
                        metadata,
                        chunk_meta,
                        start_idx + len(chunks)
                    ))
                
                # Handle sentence that's too long on its own
                if self.count_tokens(sentence) > self.chunk_size:
                    # Split by words as last resort
                    word_chunks = self._split_by_words(sentence, metadata, start_idx + len(chunks))
                    chunks.extend(word_chunks)
                    current_text = ""
                else:
                    overlap_text = self._get_overlap_text(current_text) if current_text else ""
                    current_text = overlap_text + sentence + " "
        
        if current_text.strip():
            chunk_meta = {
                "page_numbers": [block.get("page_number")] if block.get("page_number") else [],
                "sections": [block.get("section")] if block.get("section") else []
            }
            chunks.append(self._create_chunk(
                current_text.strip(),
                metadata,
                chunk_meta,
                start_idx + len(chunks)
            ))
        
        return chunks
    
    def _split_by_words(
        self,
        text: str,
        metadata: Dict[str, Any],
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """Last resort: split text by words."""
        chunks = []
        words = text.split()
        
        current_words = []
        for word in words:
            current_words.append(word)
            if self.count_tokens(" ".join(current_words)) >= self.chunk_size:
                chunk_text = " ".join(current_words[:-1])
                if chunk_text:
                    chunks.append(self._create_chunk(
                        chunk_text,
                        metadata,
                        {"page_numbers": [], "sections": []},
                        start_idx + len(chunks)
                    ))
                current_words = [word]
        
        if current_words:
            chunks.append(self._create_chunk(
                " ".join(current_words),
                metadata,
                {"page_numbers": [], "sections": []},
                start_idx + len(chunks)
            ))
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if not text or self.chunk_overlap <= 0:
            return ""
        
        # Get last N tokens worth of text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        overlap_text = ""
        for sentence in reversed(sentences):
            potential = sentence + " " + overlap_text
            if self.count_tokens(potential) <= self.chunk_overlap:
                overlap_text = potential
            else:
                break
        
        return overlap_text
    
    def _create_chunk(
        self,
        content: str,
        doc_metadata: Dict[str, Any],
        chunk_metadata: Dict[str, Any],
        chunk_index: int
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with all metadata."""
        return {
            "content": content,
            "metadata": {
                "document_id": doc_metadata.get("document_id"),
                "filename": doc_metadata.get("filename"),
                "file_type": doc_metadata.get("file_type"),
                "department": doc_metadata.get("department"),
                "tags": doc_metadata.get("tags", []),
                "chunk_type": "text",
                "chunk_index": chunk_index,
                "page_number": chunk_metadata["page_numbers"][0] if chunk_metadata["page_numbers"] else None,
                "page_numbers": chunk_metadata["page_numbers"],
                "sections": chunk_metadata["sections"],
                "token_count": self.count_tokens(content)
            }
        }
