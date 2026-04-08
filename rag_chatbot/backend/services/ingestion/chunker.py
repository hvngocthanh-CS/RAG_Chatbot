"""
Section-Aware Chunker — Step 3 of the RAG pipeline.

Splits preprocessed text blocks into retrieval-optimised chunks using a
unified strategy that combines structural awareness (headings, paragraphs)
with semantic boundary detection via embedding cosine similarity.

Design decisions:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Every section goes through the SAME pipeline:                      │
  │                                                                    │
  │  1. Analyse  — count tokens, split into sentences, compute         │
  │                embedding similarity between consecutive sentences   │
  │                to detect topic shifts.                              │
  │  2. Plan     — choose split points by combining token budget       │
  │                with paragraph boundaries and semantic scores.       │
  │  3. Build    — produce chunk dicts with heading prefix,            │
  │                breadcrumb path, and part numbering.                 │
  │                                                                    │
  │ There is no "short path" vs "long path" — a 100-token section     │
  │ and a 3 000-token section both run through the same planner.       │
  │ The planner naturally produces one chunk for short sections and    │
  │ multiple for long ones.                                            │
  └─────────────────────────────────────────────────────────────────────┘

Why this matters for retrieval:
  - Chunks align with the document's own topic boundaries.
  - The heading prefix gives the embedding model a topic signal.
  - Embedding-based boundary detection captures topic shifts that
    keyword heuristics would miss.
  - Part numbering and breadcrumb paths enable downstream reassembly
    and scoped filtering.
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import tiktoken

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

@dataclass(frozen=True)
class ChunkerConfig:
    """All tuneable parameters in one place — no magic numbers elsewhere."""

    max_chunk_tokens: int = 600
    """Hard ceiling per chunk (tokens).  Chunks will never exceed this."""

    min_chunk_tokens: int = 80
    """Chunks smaller than this are merged into their neighbour."""

    overlap_sentences: int = 2
    """Number of trailing sentences to repeat in the next chunk for
    context continuity when a section is split."""

    semantic_look_back: int = 3
    """When the token budget forces a split, look back this many sentence
    boundaries for a higher-scored semantic split point."""

    semantic_min_score: float = 0.15
    """Minimum semantic score required to prefer a semantic split over
    the default token-budget split.  Lower → more aggressive semantic
    splitting.  Range 0.0–1.0.  With embedding similarity, scores are
    typically 0.05–0.4, so 0.15 is a reasonable default."""

    heading_separator: str = " > "
    """Separator for breadcrumb paths (e.g. '1. Intro > 1.1 Scope')."""


# ======================================================================
# Semantic boundary scorer (embedding similarity)
# ======================================================================

class _EmbeddingBoundaryScorer:
    """
    Scores sentence boundaries by computing cosine similarity between
    consecutive sentence embeddings.  A low similarity indicates a
    topic shift — exactly where we want to split.

    boundary_score[i] = 1.0 - cosine_similarity(sentence_i, sentence_i+1)

    Requires a SentenceTransformer model instance.
    """

    @staticmethod
    def score_boundaries(
        sentences: List[str],
        model,
    ) -> List[float]:
        """
        Return a list of length ``len(sentences) - 1``.

        ``scores[i]`` is the topic-shift strength between sentence *i*
        and sentence *i + 1*.  Higher score = more likely topic change.
        """
        if len(sentences) <= 1:
            return []

        embeddings = model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = []
        for i in range(len(embeddings) - 1):
            similarity = float(np.dot(embeddings[i], embeddings[i + 1]))
            scores.append(1.0 - similarity)

        return scores


# ======================================================================
# Heading hierarchy helper
# ======================================================================

_HEADING_LEVEL_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")


def _heading_depth(heading: str) -> int:
    """'2.1.3 Foo' → 3,  'Overview' → 1."""
    m = _HEADING_LEVEL_RE.match(heading)
    return m.group(1).count(".") + 1 if m else 1


# ======================================================================
# Main chunker
# ======================================================================

class SectionChunker:
    """
    Section-aware text chunker optimised for RAG retrieval.

    Every section — regardless of length — runs through the same
    analyse → plan → build pipeline.  The planner picks the best split
    points by combining the token budget with paragraph boundaries and
    semantic boundary scores.  Short sections naturally get one chunk;
    long sections get multiple chunks with overlap and part numbering.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 600,
        min_chunk_tokens: int = 80,
        overlap_sentences: int = 2,
        heading_separator: str = " > ",
        encoding_name: str = "cl100k_base",
        *,
        semantic_look_back: int = 3,
        semantic_min_score: float = 0.15,
    ):
        self.cfg = ChunkerConfig(
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
            overlap_sentences=overlap_sentences,
            heading_separator=heading_separator,
            semantic_look_back=semantic_look_back,
            semantic_min_score=semantic_min_score,
        )
        self._embedding_model = None
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._enc = None
            logger.warning("tiktoken unavailable — using len//4 estimate")

    def set_embedding_model(self, model) -> None:
        """Set the SentenceTransformer model for semantic boundary scoring."""
        self._embedding_model = model
        logger.info("Chunker: embedding model set for semantic boundary detection")

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text)) if self._enc else len(text) // 4

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    _ABBREVS_RE = re.compile(
        r"\b(Dr|Mr|Ms|Mrs|Prof|Sr|Jr|vs|etc|e\.g|i\.e|No|Vol|Fig)\."
    )

    def _split_sentences(self, text: str) -> List[str]:
        safe = self._ABBREVS_RE.sub(r"\1<PERIOD>", text)
        parts = re.split(r"(?<=[.!?])\s+", safe)
        return [p.replace("<PERIOD>", ".") for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Heading hierarchy
    # ------------------------------------------------------------------

    def _update_heading_stack(
        self, stack: List[str], heading: str,
    ) -> None:
        depth = _heading_depth(heading)
        while stack and _heading_depth(stack[-1]) >= depth:
            stack.pop()
        stack.append(heading)

    def _breadcrumb(self, stack: List[str]) -> str:
        if len(stack) <= 1:
            return stack[0] if stack else ""
        return self.cfg.heading_separator.join(stack)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text_blocks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Main entry point.  Accepts preprocessed text blocks and
        document-level metadata; returns a flat list of chunk dicts.
        """
        chunks: List[Dict[str, Any]] = []
        heading_stack: List[str] = []

        # Accumulator for the current section
        cur_heading = ""
        cur_paragraphs: List[str] = []
        cur_pages: List[int] = []

        def flush() -> None:
            if not cur_paragraphs:
                return
            bc = self._breadcrumb(list(heading_stack)) if cur_heading else ""
            new = self._plan_and_build(
                cur_heading, bc, cur_paragraphs, cur_pages,
                metadata, len(chunks),
            )
            chunks.extend(new)

        for block in text_blocks:
            text = block.get("text", "").strip()
            btype = block.get("block_type") or block.get("type", "paragraph")
            page = block.get("page_number")

            if not text:
                continue

            if btype in ("heading", "title"):
                flush()
                cur_paragraphs = []
                cur_pages = []
                self._update_heading_stack(heading_stack, text)
                cur_heading = text
                if page and page not in cur_pages:
                    cur_pages.append(page)
                continue

            cur_paragraphs.append(text)
            if page and page not in cur_pages:
                cur_pages.append(page)

        flush()

        chunks = self._merge_small_chunks(chunks)
        logger.info("Created %d chunks", len(chunks))
        return chunks

    # ==================================================================
    # Core pipeline: analyse → plan → build
    # ==================================================================

    def _plan_and_build(
        self,
        heading: str,
        breadcrumb: str,
        paragraphs: List[str],
        pages: List[int],
        doc_meta: Dict[str, Any],
        start_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        Unified pipeline for ANY section size.

        1. **Analyse** — flatten paragraphs into sentences, compute
           semantic boundary scores, record paragraph boundaries.
        2. **Plan** — walk sentences accumulating tokens; when the
           budget is exhausted, choose the best split point by checking
           (a) paragraph boundaries and (b) semantic scores.
        3. **Build** — turn each planned segment into a chunk dict
           with heading prefix, breadcrumb, and part numbering.
        """
        prefix = f"[{heading}]\n\n" if heading else ""
        prefix_tokens = self.count_tokens(prefix)
        budget = self.cfg.max_chunk_tokens - prefix_tokens

        # ---- Analyse ----
        sentences, para_boundaries, boundary_scores = self._analyse(
            paragraphs,
        )
        if not sentences:
            return []

        # ---- Plan ----
        segments = self._plan_splits(
            sentences, para_boundaries, boundary_scores, budget,
        )

        # ---- Build ----
        used_semantic = any(s.get("semantic") for s in segments)
        total_parts = len(segments)
        method = "section+semantic" if used_semantic else "section"

        chunks: List[Dict[str, Any]] = []
        for part_num, seg in enumerate(segments, 1):
            body = " ".join(seg["sentences"])
            content = (prefix + body).strip()
            chunk = self._make_chunk(
                content=content,
                heading=heading,
                breadcrumb=breadcrumb,
                pages=pages,
                doc_meta=doc_meta,
                chunk_index=start_idx + len(chunks),
                method=method if total_parts > 1 else "section",
                part_num=part_num,
                total_parts=total_parts,
            )
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Step 1: Analyse
    # ------------------------------------------------------------------

    def _analyse(
        self, paragraphs: List[str],
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Flatten paragraphs into sentences and compute:
        - ``para_boundaries``: set of sentence indices that start a new
          paragraph (natural structural boundary).
        - ``boundary_scores``: semantic topic-shift score for each
          inter-sentence boundary, computed via embedding cosine
          similarity (1 - similarity).
        """
        sentences: List[str] = []
        para_boundaries: List[int] = []  # sentence indices that start a new para

        for para in paragraphs:
            para_boundaries.append(len(sentences))
            sentences.extend(self._split_sentences(para))

        if self._embedding_model is not None:
            boundary_scores = _EmbeddingBoundaryScorer.score_boundaries(
                sentences, self._embedding_model,
            )
        else:
            logger.warning(
                "No embedding model set — boundary scores will be zeros"
            )
            boundary_scores = [0.0] * max(0, len(sentences) - 1)

        return sentences, para_boundaries, boundary_scores

    # ------------------------------------------------------------------
    # Step 2: Plan
    # ------------------------------------------------------------------

    def _plan_splits(
        self,
        sentences: List[str],
        para_boundaries: List[int],
        boundary_scores: List[float],
        budget: int,
    ) -> List[Dict[str, Any]]:
        """
        Walk through sentences and decide where to split.

        At each potential split point, rank candidates by:
          1. Paragraph boundary (structural signal)
          2. Semantic score (topic-shift signal)
          3. Token-budget exhaustion (hard limit, never exceeded)

        Returns a list of segments.  Each segment is a dict:
            {"sentences": [...], "semantic": bool}
        """
        para_set = set(para_boundaries)
        segments: List[Dict[str, Any]] = []
        buf: List[str] = []
        buf_tokens = 0
        used_semantic = False

        for i, sent in enumerate(sentences):
            sent_tok = self.count_tokens(sent + " ")

            # Would adding this sentence exceed the budget?
            if buf_tokens + sent_tok > budget and buf:
                split_idx, is_semantic = self._best_split(
                    buf, i, para_set, boundary_scores,
                )

                flush = buf[:split_idx]
                leftover = buf[split_idx:]

                if flush:
                    segments.append({
                        "sentences": flush,
                        "semantic": is_semantic,
                    })
                if is_semantic:
                    used_semantic = True

                # Overlap: keep last N sentences for continuity
                overlap_sents = flush[-self.cfg.overlap_sentences:]
                buf = overlap_sents + leftover
                buf_tokens = sum(self.count_tokens(s + " ") for s in buf)

            buf.append(sent)
            buf_tokens += sent_tok

        if buf:
            segments.append({"sentences": buf, "semantic": used_semantic})

        return segments

    def _best_split(
        self,
        buf: List[str],
        global_sent_idx: int,
        para_set: set,
        boundary_scores: List[float],
    ) -> Tuple[int, bool]:
        """
        Choose the best split position within ``buf``.

        Candidates are the last ``semantic_look_back`` positions.
        Returns ``(split_at, used_semantic_signal)``.
        """
        look_back = self.cfg.semantic_look_back
        n = len(buf)

        best_pos = n          # default: split right at the budget edge
        best_priority = -1.0  # higher is better
        is_semantic = False

        for offset in range(min(look_back, n)):
            candidate = n - offset
            if candidate < 1:
                break

            # Map buffer position → original sentence index
            orig_idx = global_sent_idx - n + candidate - 1

            # Score this candidate
            score = 0.0

            # Structural signal: is the next sentence a paragraph start?
            next_orig = orig_idx + 1
            if next_orig in para_set:
                score += 1.0  # paragraph boundary is very strong

            # Semantic signal
            if 0 <= orig_idx < len(boundary_scores):
                score += boundary_scores[orig_idx]

            if score > best_priority and score >= self.cfg.semantic_min_score:
                best_priority = score
                best_pos = candidate
                is_semantic = score > 0 and (next_orig not in para_set)

        return best_pos, is_semantic

    # ------------------------------------------------------------------
    # Step 3: Build (chunk dict factory)
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        content: str,
        heading: str,
        breadcrumb: str,
        pages: List[int],
        doc_meta: Dict[str, Any],
        chunk_index: int,
        method: str,
        part_num: int,
        total_parts: int,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "document_id": doc_meta.get("document_id"),
            "filename": doc_meta.get("filename"),
            "file_type": doc_meta.get("file_type"),
            "department": doc_meta.get("department"),
            "tags": doc_meta.get("tags", []),
            "chunk_type": "text",
            "chunk_index": chunk_index,
            "page_number": pages[0] if pages else None,
            "page_numbers": list(pages),
            "section": heading,
            "section_path": breadcrumb or heading,
            "sections": [heading] if heading else [],
            "token_count": self.count_tokens(content),
            "chunking_method": method,
        }
        if total_parts > 1:
            meta["chunk_part"] = part_num
            meta["chunk_total_parts"] = total_parts
        return {"content": content, "metadata": meta}

    # ------------------------------------------------------------------
    # Post-pass: merge tiny chunks
    # ------------------------------------------------------------------

    def _merge_small_chunks(
        self, chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(chunks) < 2:
            return chunks

        merged: List[Dict[str, Any]] = [chunks[0]]

        for chunk in chunks[1:]:
            prev = merged[-1]
            cur_tok = chunk["metadata"]["token_count"]
            prev_tok = prev["metadata"]["token_count"]

            if (
                cur_tok < self.cfg.min_chunk_tokens
                and prev_tok + cur_tok <= self.cfg.max_chunk_tokens
            ):
                prev["content"] += "\n\n" + chunk["content"]
                prev["metadata"]["token_count"] = self.count_tokens(
                    prev["content"],
                )
                for p in chunk["metadata"].get("page_numbers", []):
                    if p and p not in prev["metadata"]["page_numbers"]:
                        prev["metadata"]["page_numbers"].append(p)
            else:
                merged.append(chunk)

        for i, c in enumerate(merged):
            c["metadata"]["chunk_index"] = i

        return merged
