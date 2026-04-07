"""
Document Preprocessor — Step 2 of the RAG pipeline.

Cleans and restructures ParsedDocument output from Step 1 (DocumentParser)
to maximise downstream chunk quality for retrieval.

Pipeline (8 operations, sequential — order matters):
  1. Unicode / encoding repair       — fix mojibake, ligatures, zero-width chars
  2. PDF text-artifact repair        — rejoin split words, fix punctuation spacing
  3. Title-page splitter             — extract doc metadata from merged title blocks
  4. Heading splitter                — split "heading + body" and "heading + sub-heading"
  5. Frequency-based dedup           — remove repeated headers/footers
  6. Cross-page merger               — rejoin text broken across page boundaries
  7. Small-block merger              — absorb tiny fragments into neighbours
  8. Section rebuild                 — re-assign section context top-down

Design principles:
  - Each operation is a pure function on List[TextBlock] (testable in isolation)
  - Operations are idempotent — running twice produces same result
  - Parser stays faithful to the source; preprocessor handles cleaning
  - Operations ordered so that earlier ops create cleaner input for later ops:
      text repair → structural repair → noise removal → merging → metadata rebuild
"""

import re
import logging
import unicodedata
from copy import deepcopy
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
        title_max_len: int = 80,
    ):
        self.freq_threshold = freq_threshold
        self.min_block_chars = min_block_chars
        self.heading_max_len = heading_max_len
        self.title_max_len = title_max_len

    def preprocess(self, doc: ParsedDocument) -> ParsedDocument:
        """Run all 8 operations and return a cleaned copy."""
        doc = deepcopy(doc)

        blocks = doc.text_blocks
        tables = doc.tables

        # Phase 1: Text repair (character-level)
        blocks = self._op1_unicode_repair(blocks)
        tables = self._op1_unicode_repair_tables(tables)
        blocks = self._op2_artifact_repair(blocks)
        tables = self._op2_artifact_repair_tables(tables)

        # Phase 2: Structural repair (block-level)
        blocks = self._op3_title_page_splitter(blocks)
        blocks = self._op4_heading_splitter(blocks)

        # Phase 3: Noise removal
        blocks = self._op5_frequency_dedup(blocks, doc.page_count)

        # Phase 4: Merging
        blocks = self._op6_cross_page_merge(blocks)
        blocks = self._op7_small_block_merge(blocks)

        # Phase 5: Metadata rebuild
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

    _ENCODING_FIXES = {
        "\u00f9": "–",      # ù → en-dash (common PDF mojibake)
        "\ufb01": "fi",     # ﬁ ligature
        "\ufb02": "fl",     # ﬂ ligature
        "\ufb00": "ff",     # ﬀ ligature
        "\ufb03": "ffi",    # ﬃ ligature
        "\ufb04": "ffl",    # ﬄ ligature
        "\u2019": "'",      # right single quote → apostrophe
        "\u2018": "'",      # left single quote → apostrophe
        "\u201c": '"',      # left double quote
        "\u201d": '"',      # right double quote
        "\u2013": "–",      # en-dash (normalize)
        "\u2014": "–",      # em-dash → en-dash (normalize to single dash type)
        "\u00a0": " ",      # non-breaking space
        "\u200b": "",       # zero-width space
        "\u200c": "",       # zero-width non-joiner
        "\u200d": "",       # zero-width joiner
        "\ufeff": "",       # BOM
    }

    def _fix_text(self, text: str) -> str:
        """Apply Unicode fixes and normalise whitespace."""
        for bad, good in self._ENCODING_FIXES.items():
            text = text.replace(bad, good)
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines).strip()
        return text

    def _op1_unicode_repair(self, blocks: List[TextBlock]) -> List[TextBlock]:
        for block in blocks:
            block.text = self._fix_text(block.text)
        return blocks

    def _op1_unicode_repair_tables(self, tables: List[Table]) -> List[Table]:
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

    _SHORT_WORDS = frozenset({
        "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he",
        "if", "in", "is", "it", "me", "my", "no", "of", "oh", "ok", "on",
        "or", "ox", "so", "to", "up", "us", "we",
    })
    _SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
    _HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

    def _repair_split_words(self, text: str) -> str:
        """Rejoin words split by PDF extraction (trailing 1-2 char fragments)."""
        parts = text.split(" ")
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if (
                i + 1 < len(parts)
                and len(parts[i + 1]) <= 2
                and parts[i + 1].lower() not in self._SHORT_WORDS
                and parts[i + 1].isalpha()
                and part
                and part[-1].isalpha()
            ):
                result.append(part + parts[i + 1])
                i += 2
            else:
                result.append(part)
                i += 1
        return " ".join(result)

    def _repair_artifacts(self, text: str) -> str:
        text = self._HYPHEN_BREAK_RE.sub(r"\1\2", text)
        text = self._repair_split_words(text)
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
    # Op 3 — Title Page Splitter (NEW)
    # ========================================================================
    #
    # Problem: PDF title pages merge Title + Subtitle + Department + Version
    # into a single large heading block because all text has large font size
    # and no sufficient vertical gap.
    #
    # Example input (1 block, 176 chars):
    #   "Employee Handbook Comprehensive Guide to Policies, Benefits, and
    #    Workplace Standards Human Resources Department Version 3.2 |
    #    Effective January 1, 2026 | Approved by CEO & CHRO"
    #
    # Desired output (2 blocks):
    #   heading:   "Employee Handbook"
    #   paragraph: "Comprehensive Guide to Policies, Benefits, and Workplace
    #              Standards Human Resources Department Version 3.2 |
    #              Effective January 1, 2026 | Approved by CEO & CHRO"
    #
    # Strategy: The first heading block on page 1 that is too long is likely
    # a title page.  We split it into a short document title (first meaningful
    # phrase) and a metadata paragraph (the rest).

    # Separators that commonly appear in title page metadata
    _TITLE_SEPARATOR_RE = re.compile(
        r"(?:"
        r"(?:Comprehensive|Your Comprehensive|Internal|Guidelines for|"
        r"Standard Operating|New Features|Purchasing|Learning Paths|"
        r"Meeting Rooms|Service Level|Frequently Asked|Personal Data|"
        r"Quarterly|Key Technical|Production Database|Technical Standards|"
        r"Objectives,)"
        r")",
    )

    def _split_title_block(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Split a title-page block into (document_title, subtitle_metadata).

        Strategy: find where the actual document title ends and the subtitle
        / metadata begins.  We look for common subtitle-start patterns.
        """
        # Try to find where subtitle begins
        m = self._TITLE_SEPARATOR_RE.search(text)
        if m and m.start() > 5:
            title = text[:m.start()].strip()
            subtitle = text[m.start():].strip()
            if title and subtitle:
                return title, subtitle

        # Fallback: split at pipe | separator
        if " | " in text:
            parts = text.split(" | ", 1)
            # Only if first part is reasonably short
            if len(parts[0]) <= self.title_max_len:
                return parts[0].strip(), parts[1].strip()

        return None

    def _op3_title_page_splitter(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """
        Split the first heading on page 1 if it's a merged title-page block.
        Converts it into: short title (heading) + metadata (paragraph).
        """
        if not blocks:
            return blocks

        result: List[TextBlock] = []
        title_split_done = False

        for block in blocks:
            # Only split the FIRST long heading on page 1
            if (
                not title_split_done
                and block.page_number == 1
                and block.block_type == "heading"
                and len(block.text) > self.heading_max_len
            ):
                split = self._split_title_block(block.text)
                if split:
                    title_text, meta_text = split
                    result.append(TextBlock(
                        text=title_text,
                        page_number=1,
                        block_type="heading",
                        section=title_text,
                    ))
                    result.append(TextBlock(
                        text=meta_text,
                        page_number=1,
                        block_type="paragraph",
                        section=title_text,
                    ))
                    title_split_done = True
                    continue

            if block.page_number and block.page_number > 1:
                title_split_done = True  # past page 1

            result.append(block)

        return result

    # ========================================================================
    # Op 4 — Heading Splitter (enhanced)
    # ========================================================================
    #
    # Handles TWO patterns:
    #   A) Numbered heading merged with body text:
    #      "4. Performance Management TechViet uses a combination..."
    #      → heading: "4. Performance Management"
    #      → paragraph: "TechViet uses a combination..."
    #
    #   B) Heading merged with sub-heading:
    #      "2. Code Quality Standards 2.1 Pull Request and Code Review..."
    #      → heading: "2. Code Quality Standards"
    #      → heading: "2.1 Pull Request and Code Review..."

    _BODY_START_WORDS = frozenset({
        "TechViet", "The", "This", "These", "Those", "Our", "We", "All",
        "As", "It", "In", "A", "An", "For", "To", "At", "By", "On",
        "Each", "Every", "Any", "Some", "No", "Not", "During", "After",
        "Before", "When", "Where", "While", "Since", "From", "With",
        "Standard", "Employees", "Below", "Dr.", "Currently",
        "Data", "Under", "Both", "Although", "Despite", "Given",
        "Based", "Using", "Following", "According",
    })

    # Pattern C: heading text followed by a timestamp, number, or data marker
    # e.g. "1.1 Incident Timeline 08:00 UTC ..."
    _TIMESTAMP_BODY_RE = re.compile(
        r"^(\d+(?:\.\d+)*\.?\s+.{3,80}?)\s+"
        r"(\d{1,2}[:.]\d{2}\b.*|"             # timestamps: 08:00, 10.15
        r"\d{4}[-/]\d{2}[-/]\d{2}\b.*|"       # dates: 2025-11-28
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s.*|"
        r"Q[1-4]\s\d{4}\b.*|"                 # Q4 2025
        r"Phase\s\d\b.*|"                     # Phase 1
        r"Step\s\d\b.*|"                      # Step 1
        r"Year\s\d\b.*|"                      # Year 1
        r"Alert\s.*|"                         # Alert fires...
        r"Scenario\s\d\b.*"                   # Scenario 1
        r")$"
    )

    _NUMBERED_START_RE = re.compile(r"^(\d+(?:\.\d+)*\.?\s+)")

    # Pattern B: detect embedded sub-heading number like " 2.1 " inside text
    _EMBEDDED_SUBHEADING_RE = re.compile(
        r"^(.*?\S)\s+(\d+\.\d+(?:\.\d+)*\.?\s+.+)$",
        re.DOTALL,
    )

    def _find_heading_body_split(self, text: str) -> Optional[List[Tuple[str, str]]]:
        """
        Find split points in a long block.

        Returns list of (text, block_type) tuples, or None if no split found.
        """
        # --- Pattern C: heading + timestamp/data body ---
        # "1.1 Incident Timeline 08:00 UTC (15:00 Vietnam time) ..."
        m_ts = self._TIMESTAMP_BODY_RE.match(text)
        if m_ts:
            return [
                (m_ts.group(1).strip(), "heading"),
                (m_ts.group(2).strip(), "paragraph"),
            ]

        # --- Pattern B: heading + embedded sub-heading ---
        # "2. Code Quality Standards 2.1 Pull Request and Code Review Requirements"
        m_sub = self._EMBEDDED_SUBHEADING_RE.match(text)
        if m_sub:
            first_part = m_sub.group(1).strip()
            second_part = m_sub.group(2).strip()
            # Validate: first part should be a heading (short-ish, starts with number)
            if (
                self._NUMBERED_START_RE.match(first_part)
                and len(first_part) < 150
            ):
                return [
                    (first_part, "heading"),
                    (second_part, "heading" if len(second_part) < 120 else "paragraph"),
                ]

        # --- Pattern A: numbered heading + body text ---
        m = self._NUMBERED_START_RE.match(text)
        if not m:
            return None

        prefix = m.group(1)
        rest = text[m.end():]
        words = rest.split()

        if len(words) < 3:
            return None

        title_joiners = {"and", "of", "for", "the", "in", "on", "to", "a", "an", "&"}

        for i in range(1, len(words)):
            word = words[i]
            clean_word = word.rstrip(".,;:!?()")

            if clean_word in self._BODY_START_WORDS and i >= 2:
                body_text = " ".join(words[i:])
                if len(body_text) > 60:
                    heading = prefix + " ".join(words[:i])
                    # If heading ends with a dangling preposition/article,
                    # move it to the body (e.g. "... Meltdown On" → "On November...")
                    h_words = heading.rsplit(None, 1)
                    if len(h_words) == 2 and h_words[1].lower() in title_joiners | {"on", "at", "by", "in", "from", "with"}:
                        heading = h_words[0]
                        body_text = h_words[1] + " " + body_text
                    return [(heading.strip(), "heading"), (body_text.strip(), "paragraph")]

            if (
                word[0].islower()
                and word not in title_joiners
                and i >= 2
            ):
                body_text = " ".join(words[i:])
                if len(body_text) > 60:
                    heading = prefix + " ".join(words[:i])
                    return [(heading.strip(), "heading"), (body_text.strip(), "paragraph")]

        return None

    # Pattern: "ADR-001: Title Here Status: Accepted | Date: ..."
    # Split at " Status:" to separate the ADR title from metadata
    _METADATA_SPLIT_RE = re.compile(
        r"^(.{10,80}?)\s+(Status:\s+.+)$"
    )

    def _op4_heading_splitter(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Split blocks that contain merged heading+body or heading+sub-heading."""
        result: List[TextBlock] = []

        for block in blocks:
            text = block.text.strip()

            if len(text) > self.heading_max_len:
                # Try numbered heading split (Pattern A & B)
                parts = self._find_heading_body_split(text)
                if parts:
                    for part_text, part_type in parts:
                        result.append(TextBlock(
                            text=part_text,
                            page_number=block.page_number,
                            block_type=part_type,
                            section="",
                        ))
                    continue

                # Try metadata split (ADR pattern: "Title Status: Accepted | Date...")
                m = self._METADATA_SPLIT_RE.match(text)
                if m:
                    result.append(TextBlock(
                        text=m.group(1).strip(),
                        page_number=block.page_number,
                        block_type="heading",
                        section="",
                    ))
                    result.append(TextBlock(
                        text=m.group(2).strip(),
                        page_number=block.page_number,
                        block_type="paragraph",
                        section="",
                    ))
                    continue

            result.append(block)

        return result

    # ========================================================================
    # Op 5 — Frequency-based Dedup (header/footer removal)
    # ========================================================================

    @staticmethod
    def _normalise_for_compare(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\d+", "#", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _is_likely_header_footer(text: str) -> bool:
        return len(text.strip()) < 120

    def _op5_frequency_dedup(
        self, blocks: List[TextBlock], page_count: int
    ) -> List[TextBlock]:
        """Remove text blocks that appear on many pages (headers/footers)."""
        if page_count < 3:
            return blocks

        min_appearances = max(2, int(page_count * self.freq_threshold))

        text_page_map: dict[str, set[int]] = {}
        for block in blocks:
            if not self._is_likely_header_footer(block.text):
                continue
            norm = self._normalise_for_compare(block.text)
            if not norm:
                continue
            page = block.page_number or 0
            text_page_map.setdefault(norm, set()).add(page)

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
    # Op 6 — Cross-page Merger
    # ========================================================================

    @staticmethod
    def _is_incomplete_end(text: str) -> bool:
        text = text.rstrip()
        if not text:
            return False
        return text[-1] not in ".!?:;\u201d\""

    @staticmethod
    def _is_continuation_start(text: str) -> bool:
        text = text.lstrip()
        if not text:
            return False
        if text[0].islower():
            return True
        if re.match(
            r"^(and|or|but|the|a|an|that|which|who|where|when|also|"
            r"however|categories)\b",
            text, re.IGNORECASE,
        ):
            return True
        return False

    def _op6_cross_page_merge(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Merge text blocks split across page boundaries."""
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

                both_paragraphs = (
                    block.block_type == "paragraph"
                    and next_block.block_type == "paragraph"
                )
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
                    merged = TextBlock(
                        text=block.text.rstrip() + " " + next_block.text.lstrip(),
                        page_number=block.page_number,
                        block_type="paragraph",
                        section=block.section,
                    )
                    result.append(merged)
                    skip_next = True
                    continue

            result.append(block)

        return result

    # ========================================================================
    # Op 7 — Small-block Merger
    # ========================================================================

    def _op7_small_block_merge(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Merge very short paragraph blocks into nearest same-section neighbour."""
        if len(blocks) < 2:
            return blocks

        result: List[TextBlock] = []

        for block in blocks:
            if block.block_type == "heading":
                result.append(block)
                continue

            if len(block.text) < self.min_block_chars:
                if (
                    result
                    and result[-1].block_type == "paragraph"
                    and result[-1].section == block.section
                ):
                    result[-1].text = result[-1].text.rstrip() + " " + block.text.lstrip()
                    continue

            result.append(block)

        return result

    # ========================================================================
    # Op 8 — Rebuild section context
    # ========================================================================

    @staticmethod
    def _rebuild_sections(blocks: List[TextBlock]) -> List[TextBlock]:
        """Re-assign section context based on nearest heading above each block."""
        current_section = ""
        for block in blocks:
            if block.block_type == "heading":
                current_section = block.text
            block.section = current_section
        return blocks
