"""
Document Preprocessor — Step 2 of the RAG pipeline.

Cleans and restructures ParsedDocument output from Step 1 (DocumentParser)
to maximise downstream chunk quality for retrieval.

Pipeline (6 operations, sequential):
  1. Unicode / encoding repair
  2. PDF text-artifact repair (split words, ligatures)
  3. Heading splitter  — split long "heading+body" blocks
  4. Frequency-based dedup — remove repeated headers/footers
  5. Cross-page merger — rejoin text broken across page boundaries
  6. Small-block merger — absorb tiny fragments into neighbours

Design principles:
  • Each operation is a pure function on ParsedDocument (testable in isolation)
  • Operations are idempotent — running twice produces same result
  • Parser stays faithful to the source; preprocessor handles cleaning
"""

import re
import logging
import unicodedata
from copy import deepcopy
from collections import Counter
from typing import List, Optional, Tuple

from backend.services.document_parser import ParsedDocument, TextBlock, Table

logger = logging.getLogger(__name__)


# ============================================================================
# Main entry point
# ============================================================================

class DocumentPreprocessor:
    """
    Clean and restructure a ParsedDocument for optimal chunking.

    Usage:
        preprocessor = DocumentPreprocessor()
        clean_doc = preprocessor.preprocess(raw_doc)
    """

    def __init__(
        self,
        *,
        freq_threshold: float = 0.4,
        min_block_chars: int = 50,
        heading_max_len: int = 120,
    ):
        """
        Args:
            freq_threshold: fraction of pages a block must appear on to be
                            considered a repeating header/footer (0.4 = 40%).
            min_block_chars: blocks shorter than this (non-heading) get merged
                            into neighbours.
            heading_max_len: max length for a heading — longer blocks that were
                            classified as headings get split.
        """
        self.freq_threshold = freq_threshold
        self.min_block_chars = min_block_chars
        self.heading_max_len = heading_max_len

    def preprocess(self, doc: ParsedDocument) -> ParsedDocument:
        """Run all 6 operations and return a cleaned copy."""
        # Work on a deep copy so the original stays untouched
        doc = deepcopy(doc)

        blocks = doc.text_blocks
        tables = doc.tables

        # --- Sequential pipeline ---
        blocks = self._op1_unicode_repair(blocks)
        tables = self._op1_unicode_repair_tables(tables)

        blocks = self._op2_artifact_repair(blocks)
        tables = self._op2_artifact_repair_tables(tables)

        blocks = self._op3_heading_splitter(blocks)

        blocks = self._op4_frequency_dedup(blocks, doc.page_count)

        blocks = self._op5_cross_page_merge(blocks)

        blocks = self._op6_small_block_merge(blocks)

        # Rebuild section context after all mutations
        blocks = self._rebuild_sections(blocks)

        doc.text_blocks = blocks
        doc.tables = tables

        logger.info(
            "Preprocessed: %d text blocks, %d tables",
            len(blocks), len(tables),
        )
        return doc

    # ========================================================================
    # Op 1 — Unicode & Encoding Repair
    # ========================================================================

    # Common PDF encoding errors: mojibake sequences → correct chars
    _ENCODING_FIXES = {
        "\u00f9": "–",      # ù → en-dash (seen in step1_output)
        "\ufb01": "fi",     # ﬁ ligature
        "\ufb02": "fl",     # ﬂ ligature
        "\ufb00": "ff",     # ﬀ ligature
        "\ufb03": "ffi",    # ﬃ ligature
        "\ufb04": "ffl",    # ﬄ ligature
        "\u2019": "'",      # right single quote → apostrophe
        "\u2018": "'",      # left single quote → apostrophe
        "\u201c": '"',      # left double quote
        "\u201d": '"',      # right double quote
        "\u2013": "–",      # en-dash (keep as-is, just normalize)
        "\u2014": "—",      # em-dash (keep as-is)
        "\u00a0": " ",      # non-breaking space → regular space
        "\u200b": "",       # zero-width space → remove
        "\u200c": "",       # zero-width non-joiner → remove
        "\u200d": "",       # zero-width joiner → remove
        "\ufeff": "",       # BOM → remove
    }

    def _fix_text(self, text: str) -> str:
        """Apply Unicode fixes and normalise whitespace."""
        for bad, good in self._ENCODING_FIXES.items():
            text = text.replace(bad, good)

        # Normalize to NFC (composed form) for consistent comparisons
        text = unicodedata.normalize("NFC", text)

        # Collapse multiple spaces/tabs into single space
        text = re.sub(r"[ \t]+", " ", text)

        # Remove leading/trailing whitespace per line, then strip block
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines).strip()

        return text

    def _op1_unicode_repair(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Fix encoding artifacts in all text blocks."""
        for block in blocks:
            block.text = self._fix_text(block.text)
        return blocks

    def _op1_unicode_repair_tables(self, tables: List[Table]) -> List[Table]:
        """Fix encoding artifacts in table headers and cells."""
        for table in tables:
            table.headers = [self._fix_text(h) for h in table.headers]
            table.rows = [
                [self._fix_text(cell) for cell in row]
                for row in table.rows
            ]
        return tables

    # ========================================================================
    # Op 2 — PDF Text-Artifact Repair
    # ========================================================================

    # Common 1-2 letter English words — must NOT be merged with neighbours
    _SHORT_WORDS = frozenset({
        "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he",
        "if", "in", "is", "it", "me", "my", "no", "of", "oh", "ok", "on",
        "or", "ox", "so", "to", "up", "us", "we",
    })
    # Pattern: space before punctuation (e.g. "word .")
    _SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
    # Pattern: hyphenated line break (e.g. "imple-\nmentation")
    _HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

    def _repair_split_words(self, text: str) -> str:
        """
        Rejoin words that were split by PDF extraction.

        Only merges when the trailing fragment (1-2 chars) is NOT a common
        English word.  "Statu s" → "Status" but "Guide to" stays.
        """
        parts = text.split(" ")
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            # Look ahead: if next part is 1-2 chars and not a real word, merge
            if (
                i + 1 < len(parts)
                and len(parts[i + 1]) <= 2
                and parts[i + 1].lower() not in self._SHORT_WORDS
                and parts[i + 1].isalpha()
                and part  # not empty
                and part[-1].isalpha()  # current part ends with letter
            ):
                result.append(part + parts[i + 1])
                i += 2
            else:
                result.append(part)
                i += 1
        return " ".join(result)

    def _repair_artifacts(self, text: str) -> str:
        """Fix common PDF text extraction artifacts."""
        # Rejoin hyphenated line breaks
        text = self._HYPHEN_BREAK_RE.sub(r"\1\2", text)
        # Rejoin split words (e.g. "Statu s" → "Status")
        text = self._repair_split_words(text)
        # Remove extra space before punctuation
        text = self._SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        return text

    def _op2_artifact_repair(self, blocks: List[TextBlock]) -> List[TextBlock]:
        for block in blocks:
            block.text = self._repair_artifacts(block.text)
        return blocks

    def _op2_artifact_repair_tables(self, tables: List[Table]) -> List[Table]:
        for table in tables:
            table.headers = [self._repair_artifacts(h) for h in table.headers]
            table.rows = [
                [self._repair_artifacts(cell) for cell in row]
                for row in table.rows
            ]
        return tables

    # ========================================================================
    # Op 3 — Heading Splitter
    # ========================================================================

    # Sentence-start indicators that signal body text (not title continuation).
    # These are words that typically begin a sentence/clause, not a heading title.
    _BODY_START_WORDS = frozenset({
        "TechViet", "The", "This", "These", "Those", "Our", "We", "All",
        "As", "It", "In", "A", "An", "For", "To", "At", "By", "On",
        "Each", "Every", "Any", "Some", "No", "Not", "During", "After",
        "Before", "When", "Where", "While", "Since", "From", "With",
        "Standard", "Employees",
    })

    # Detects numbered heading prefix at start of text
    _NUMBERED_START_RE = re.compile(r"^(\d+(?:\.\d+)*\.?\s+)")

    def _find_heading_body_split(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Find where a numbered heading ends and body text begins.

        Strategy: After the number prefix, scan words. The heading is the
        title-case portion. Body starts when we encounter a word that
        signals sentence-level prose (common sentence starters, or a
        lowercase word following a capitalized title word).

        Returns (heading, body) or None if no split found.
        """
        m = self._NUMBERED_START_RE.match(text)
        if not m:
            return None

        prefix = m.group(1)  # e.g. "4. " or "1.1 "
        rest = text[m.end():]
        words = rest.split()

        if len(words) < 3:
            return None

        # Scan through words to find where heading title ends
        for i in range(1, len(words)):
            word = words[i]
            clean_word = word.rstrip(".,;:!?")

            # Body starts at a known sentence-start word that follows
            # at least one title word
            if clean_word in self._BODY_START_WORDS and i >= 1:
                heading = prefix + " ".join(words[:i])
                body = " ".join(words[i:])
                return heading.strip(), body.strip()

            # Body starts when we see a lowercase word (not a preposition
            # that could be part of a title like "and", "of", "for")
            title_joiners = {"and", "of", "for", "the", "in", "on", "to", "a", "an", "&"}
            if (
                word[0].islower()
                and word not in title_joiners
                and i >= 2
            ):
                heading = prefix + " ".join(words[:i])
                body = " ".join(words[i:])
                return heading.strip(), body.strip()

        return None

    def _op3_heading_splitter(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """
        Split blocks that contain a heading merged with body text.

        Detects pattern: "4. Performance Management TechViet uses a combination..."
        Splits into:
          - heading: "4. Performance Management"
          - paragraph: "TechViet uses a combination..."
        """
        result: List[TextBlock] = []

        for block in blocks:
            text = block.text.strip()

            # Only attempt split on long blocks
            if len(text) > self.heading_max_len:
                split = self._find_heading_body_split(text)
                if split:
                    heading_text, body_text = split

                    # Emit heading block
                    result.append(TextBlock(
                        text=heading_text,
                        page_number=block.page_number,
                        block_type="heading",
                        section=heading_text,
                    ))
                    # Emit body block
                    result.append(TextBlock(
                        text=body_text,
                        page_number=block.page_number,
                        block_type="paragraph",
                        section=heading_text,
                    ))
                    continue

            result.append(block)

        return result

    # ========================================================================
    # Op 4 — Frequency-based Dedup (header/footer removal)
    # ========================================================================

    @staticmethod
    def _normalise_for_compare(text: str) -> str:
        """Normalise text for fuzzy comparison (ignore whitespace, case, numbers)."""
        text = text.lower().strip()
        # Replace numbers with # so "Page 1" matches "Page 2"
        text = re.sub(r"\d+", "#", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _is_likely_header_footer(text: str) -> bool:
        """Headers/footers are typically short (< 120 chars)."""
        return len(text.strip()) < 120

    def _op4_frequency_dedup(
        self, blocks: List[TextBlock], page_count: int
    ) -> List[TextBlock]:
        """
        Remove text blocks that appear (near-identically) on many pages.
        These are typically running headers, footers, or page numbers.

        Only applies when page_count >= 3 (need enough pages to detect pattern).
        """
        if page_count < 3:
            return blocks

        min_appearances = max(2, int(page_count * self.freq_threshold))

        # Count how many distinct pages each normalised text appears on
        # Only consider short blocks (headers/footers are typically < 120 chars)
        text_page_map: dict[str, set[int]] = {}
        for block in blocks:
            if not self._is_likely_header_footer(block.text):
                continue
            norm = self._normalise_for_compare(block.text)
            if not norm:
                continue
            page = block.page_number or 0
            text_page_map.setdefault(norm, set()).add(page)

        # Identify repeating texts
        repeating = {
            norm
            for norm, pages in text_page_map.items()
            if len(pages) >= min_appearances
        }

        if repeating:
            logger.info("Frequency dedup: removing %d repeating patterns", len(repeating))

        return [
            block for block in blocks
            if self._normalise_for_compare(block.text) not in repeating
        ]

    # ========================================================================
    # Op 5 — Cross-page Merger
    # ========================================================================

    @staticmethod
    def _is_incomplete_end(text: str) -> bool:
        """True if text ends mid-sentence (no terminal punctuation)."""
        text = text.rstrip()
        if not text:
            return False
        return text[-1] not in ".!?:;\u201d\""

    @staticmethod
    def _is_continuation_start(text: str) -> bool:
        """True if text starts like a sentence continuation (lowercase or specific patterns)."""
        text = text.lstrip()
        if not text:
            return False
        # Starts with lowercase letter → continuation
        if text[0].islower():
            return True
        # Starts with common continuation patterns
        if re.match(r"^(and|or|but|the|a|an|that|which|who|where|when|also|however|categories)\b", text, re.IGNORECASE):
            return True
        return False

    def _op5_cross_page_merge(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """
        Merge text blocks that were split across page boundaries.

        Condition: block N ends without terminal punctuation AND block N+1
        starts with lowercase (or continuation word) AND they are on
        consecutive pages.
        """
        if len(blocks) < 2:
            return blocks

        result: List[TextBlock] = []
        skip_next = False

        for i in range(len(blocks)):
            if skip_next:
                skip_next = False
                continue

            block = blocks[i]

            if i + 1 < len(blocks):
                next_block = blocks[i + 1]

                # Only merge paragraphs (not headings)
                both_paragraphs = (
                    block.block_type == "paragraph"
                    and next_block.block_type == "paragraph"
                )

                # Check consecutive pages
                consecutive_pages = (
                    block.page_number is not None
                    and next_block.page_number is not None
                    and next_block.page_number == block.page_number + 1
                )

                if (
                    both_paragraphs
                    and consecutive_pages
                    and self._is_incomplete_end(block.text)
                    and self._is_continuation_start(next_block.text)
                ):
                    # Merge: join with space
                    merged = TextBlock(
                        text=block.text.rstrip() + " " + next_block.text.lstrip(),
                        page_number=block.page_number,
                        block_type="paragraph",
                        section=block.section,
                    )
                    result.append(merged)
                    skip_next = True
                    logger.debug(
                        "Cross-page merge: pages %s–%s",
                        block.page_number, next_block.page_number,
                    )
                    continue

            result.append(block)

        return result

    # ========================================================================
    # Op 6 — Small-block Merger
    # ========================================================================

    def _op6_small_block_merge(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """
        Merge very short paragraph blocks into their nearest neighbour
        within the same section. Headings are never merged.
        """
        if len(blocks) < 2:
            return blocks

        result: List[TextBlock] = []

        for block in blocks:
            # Never merge headings
            if block.block_type == "heading":
                result.append(block)
                continue

            # Short paragraph?
            if len(block.text) < self.min_block_chars:
                # Try to merge into previous block if same section
                if (
                    result
                    and result[-1].block_type == "paragraph"
                    and result[-1].section == block.section
                ):
                    result[-1].text = result[-1].text.rstrip() + " " + block.text.lstrip()
                    logger.debug("Merged small block (%d chars) into previous", len(block.text))
                    continue

            result.append(block)

        return result

    # ========================================================================
    # Rebuild section context
    # ========================================================================

    @staticmethod
    def _rebuild_sections(blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Re-assign section context based on the nearest heading above each block.
        Must be called after any operation that reorders or inserts blocks.
        """
        current_section = ""
        for block in blocks:
            if block.block_type == "heading":
                current_section = block.text
            block.section = current_section
        return blocks
