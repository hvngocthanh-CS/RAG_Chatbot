"""
Section-Aware Chunker — Step 3 of the RAG pipeline.

Leverages document structure (headings, sections) produced by the
DocumentPreprocessor (Step 2) to create chunks that align with the
document's natural topic boundaries.

Why Section-Aware?
  - Each chunk covers exactly one topic (the section under a heading).
  - The heading is prepended as a context prefix, giving the embedding
    model a strong topic signal and improving retrieval precision.
  - No embedding cost at chunk time (unlike semantic chunkers), so
    ingestion is fast.
  - Rich metadata (section_path breadcrumb) enables downstream
    filtering and citation.

Algorithm:
  1. Walk preprocessed TextBlocks, tracking heading hierarchy.
  2. Accumulate paragraphs under the current heading into a section buffer.
  3. When a new heading arrives OR the buffer exceeds max_chunk_tokens,
     flush the buffer as one chunk (or split it by sentences if too large).
  4. Each chunk is prefixed with its section heading for context.
  5. Small trailing sections are merged into the previous chunk.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class SectionChunkConfig:
    """Configuration for section-aware chunking."""
    max_chunk_tokens: int = 600       # hard upper limit per chunk
    min_chunk_tokens: int = 80        # merge smaller chunks into neighbours
    overlap_sentences: int = 2        # sentence overlap when splitting large sections
    heading_separator: str = " > "    # breadcrumb separator for nested sections


class SectionChunker:
    """
    Section-aware text chunker optimised for RAG retrieval.

    Key properties:
      - Chunks align with document sections (one topic per chunk).
      - Each chunk carries a heading prefix for embedding quality.
      - Fast: no model calls during chunking (pure structural splitting).
      - Preserves page numbers and section hierarchy metadata.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 600,
        min_chunk_tokens: int = 80,
        overlap_sentences: int = 2,
        heading_separator: str = " > ",
        encoding_name: str = "cl100k_base",
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_sentences = overlap_sentences
        self.heading_separator = heading_separator

        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = None
            logger.warning("Tiktoken not available, using character estimation")

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4

    # ------------------------------------------------------------------
    # Sentence splitting (shared utility)
    # ------------------------------------------------------------------

    _ABBREVS_RE = re.compile(
        r"\b(Dr|Mr|Ms|Mrs|Prof|Sr|Jr|vs|etc|e\.g|i\.e|No|Vol|Fig)\."
    )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling common abbreviations."""
        safe = self._ABBREVS_RE.sub(r"\1<PERIOD>", text)
        parts = re.split(r"(?<=[.!?])\s+", safe)
        return [p.replace("<PERIOD>", ".") for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text_blocks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Chunk preprocessed text blocks using section awareness.

        Args:
            text_blocks: list of dicts with keys text, type, page_number, section
                         (output of ParsedDocument.to_dict()["text_blocks"]
                          or DocumentPreprocessor)
            metadata: document-level metadata (document_id, filename, ...)

        Returns:
            List of chunk dicts with "content" and "metadata".
        """
        chunks: List[Dict[str, Any]] = []

        # Accumulator for the current section
        current_heading: str = ""
        current_paragraphs: List[str] = []
        current_pages: List[int] = []

        for block in text_blocks:
            block_text = block.get("text", "").strip()
            block_type = block.get("block_type") or block.get("type", "paragraph")
            page = block.get("page_number")

            if not block_text:
                continue

            # ----- Heading / Title: flush current section, start new one -----
            if block_type in ("heading", "title"):
                # Flush accumulated paragraphs
                if current_paragraphs:
                    new_chunks = self._flush_section(
                        current_heading, current_paragraphs,
                        current_pages, metadata, len(chunks),
                    )
                    chunks.extend(new_chunks)
                    current_paragraphs = []
                    current_pages = []

                current_heading = block_text
                if page and page not in current_pages:
                    current_pages.append(page)
                continue

            # ----- Paragraph: accumulate under current heading -----
            current_paragraphs.append(block_text)
            if page and page not in current_pages:
                current_pages.append(page)

        # Flush remaining
        if current_paragraphs:
            chunks.extend(self._flush_section(
                current_heading, current_paragraphs,
                current_pages, metadata, len(chunks),
            ))

        # Post-pass: merge tiny trailing chunks into previous
        chunks = self._merge_small_chunks(chunks)

        logger.info("Created %d section-aware chunks", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Internal: flush one section into one or more chunks
    # ------------------------------------------------------------------

    def _flush_section(
        self,
        heading: str,
        paragraphs: List[str],
        pages: List[int],
        metadata: Dict[str, Any],
        start_idx: int,
    ) -> List[Dict[str, Any]]:
        """Convert a heading + its paragraphs into chunk(s)."""
        body = "\n\n".join(paragraphs)

        # Build the full chunk text with heading prefix
        if heading:
            full_text = f"[{heading}]\n\n{body}"
        else:
            full_text = body

        token_count = self.count_tokens(full_text)

        # Case 1: fits in a single chunk
        if token_count <= self.max_chunk_tokens:
            return [self._make_chunk(
                full_text, heading, pages, metadata, start_idx,
            )]

        # Case 2: section too large — split by sentences with heading prefix
        return self._split_large_section(
            heading, body, pages, metadata, start_idx,
        )

    def _split_large_section(
        self,
        heading: str,
        body: str,
        pages: List[int],
        metadata: Dict[str, Any],
        start_idx: int,
    ) -> List[Dict[str, Any]]:
        """Split an oversized section into multiple chunks by sentences."""
        sentences = self._split_sentences(body)
        if not sentences:
            return []

        prefix = f"[{heading}]\n\n" if heading else ""
        prefix_tokens = self.count_tokens(prefix)

        chunks: List[Dict[str, Any]] = []
        current_sents: List[str] = []
        current_tokens = prefix_tokens  # reserve room for prefix

        for sent in sentences:
            sent_tokens = self.count_tokens(sent + " ")

            if current_tokens + sent_tokens > self.max_chunk_tokens and current_sents:
                # Flush current accumulator
                chunk_body = " ".join(current_sents)
                chunk_text = prefix + chunk_body
                chunks.append(self._make_chunk(
                    chunk_text, heading, pages, metadata,
                    start_idx + len(chunks),
                ))

                # Overlap: keep last N sentences for context continuity
                overlap = current_sents[-self.overlap_sentences:]
                current_sents = overlap
                current_tokens = prefix_tokens + sum(
                    self.count_tokens(s + " ") for s in overlap
                )

            current_sents.append(sent)
            current_tokens += sent_tokens

        # Flush remainder
        if current_sents:
            chunk_body = " ".join(current_sents)
            chunk_text = prefix + chunk_body
            chunks.append(self._make_chunk(
                chunk_text, heading, pages, metadata,
                start_idx + len(chunks),
            ))

        return chunks

    # ------------------------------------------------------------------
    # Post-pass: merge small trailing chunks
    # ------------------------------------------------------------------

    def _merge_small_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge chunks that are too small into their predecessor."""
        if len(chunks) < 2:
            return chunks

        merged: List[Dict[str, Any]] = [chunks[0]]

        for chunk in chunks[1:]:
            prev = merged[-1]
            cur_tokens = chunk["metadata"]["token_count"]
            prev_tokens = prev["metadata"]["token_count"]

            # Merge if current chunk is tiny AND combined still fits
            if (
                cur_tokens < self.min_chunk_tokens
                and prev_tokens + cur_tokens <= self.max_chunk_tokens
            ):
                prev["content"] = prev["content"] + "\n\n" + chunk["content"]
                prev["metadata"]["token_count"] = self.count_tokens(prev["content"])
                # Extend page list
                for p in chunk["metadata"].get("page_numbers", []):
                    if p and p not in prev["metadata"]["page_numbers"]:
                        prev["metadata"]["page_numbers"].append(p)
            else:
                merged.append(chunk)

        # Re-index
        for i, c in enumerate(merged):
            c["metadata"]["chunk_index"] = i

        return merged

    # ------------------------------------------------------------------
    # Chunk dict builder
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        content: str,
        heading: str,
        pages: List[int],
        doc_metadata: Dict[str, Any],
        chunk_index: int,
    ) -> Dict[str, Any]:
        content = content.strip()
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
                "page_number": pages[0] if pages else None,
                "page_numbers": list(pages),
                "section": heading,
                "sections": [heading] if heading else [],
                "token_count": self.count_tokens(content),
                "chunking_method": "section",
            },
        }
