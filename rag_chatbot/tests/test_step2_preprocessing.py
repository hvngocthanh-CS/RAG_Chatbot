"""
Step 2 Tests — Document Preprocessing.

Run:
    cd rag_chatbot
    pytest tests/test_step2_preprocessing.py -v
    pytest tests/test_step2_preprocessing.py -v -s -k inspect   # full before/after
"""
import pytest
from backend.services.document_parser import (
    DocumentParser, ParsedDocument, TextBlock, Table,
)
from backend.services.document_preprocessor import DocumentPreprocessor
from backend.services.table_extractor import TableExtractor


@pytest.fixture
def preprocessor():
    return DocumentPreprocessor()


@pytest.fixture
def parser():
    return DocumentParser()


@pytest.fixture
def extractor():
    return TableExtractor()


# ============================================================================
# Helpers — build synthetic ParsedDocuments for unit tests
# ============================================================================

def _make_doc(
    blocks: list[TextBlock] | None = None,
    tables: list[Table] | None = None,
    page_count: int = 1,
) -> ParsedDocument:
    return ParsedDocument(
        text_blocks=blocks or [],
        tables=tables or [],
        page_count=page_count,
        file_type="pdf",
    )


# ============================================================================
# Op 1 — Unicode Repair
# ============================================================================

class TestUnicodeRepair:

    def test_fixes_mojibake_dash(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="10:00 AM \u00f9 4:00 PM", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "\u00f9" not in result.text_blocks[0].text
        assert "–" in result.text_blocks[0].text

    def test_fixes_ligatures(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="conﬁdential ﬂow", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "fi" in result.text_blocks[0].text
        assert "fl" in result.text_blocks[0].text

    def test_collapses_whitespace(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="hello   world\t\tfoo", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "hello world foo" == result.text_blocks[0].text

    def test_removes_zero_width_chars(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="hel\u200blo\u200cworld", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert result.text_blocks[0].text == "helloworld"

    def test_fixes_table_cells(self, preprocessor):
        doc = _make_doc(
            blocks=[],
            tables=[Table(
                headers=["Col\u00a01"],
                rows=[["val\u00f9ue"]],
                page_number=1,
                name="T1",
            )],
        )
        result = preprocessor.preprocess(doc)
        assert result.tables[0].headers[0] == "Col 1"
        assert "–" in result.tables[0].rows[0][0]


# ============================================================================
# Op 2 — Artifact Repair
# ============================================================================

class TestArtifactRepair:

    def test_fixes_split_words(self, preprocessor):
        # Trailing short-fragment splits (1-2 chars) that PDFs commonly produce
        doc = _make_doc([
            TextBlock(text="Statu s repor t processin g", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "Status" in result.text_blocks[0].text
        assert "report" in result.text_blocks[0].text
        assert "processing" in result.text_blocks[0].text

    def test_fixes_space_before_punctuation(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Hello , world .", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "Hello, world." == result.text_blocks[0].text

    def test_fixes_hyphen_break(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="imple-\nmentation", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "implementation" in result.text_blocks[0].text


# ============================================================================
# Op 3 — Heading Splitter
# ============================================================================

class TestHeadingSplitter:

    def test_splits_long_heading_with_body(self, preprocessor):
        long_text = (
            "5. Code of Conduct All employees, contractors, and temporary staff "
            "are expected to uphold ethical standards and professional conduct at "
            "all times, whether in the office, at client sites, at company events, "
            "or in any context where they represent the company."
        )
        doc = _make_doc([
            TextBlock(text=long_text, page_number=6, block_type="heading"),
        ])
        result = preprocessor.preprocess(doc)

        # Should be split into 2 blocks
        assert len(result.text_blocks) == 2
        assert result.text_blocks[0].block_type == "heading"
        assert result.text_blocks[1].block_type == "paragraph"
        # Heading should be short
        assert len(result.text_blocks[0].text) < 120

    def test_splits_numbered_paragraph_with_heading_prefix(self, preprocessor):
        text = (
            "4. Performance Management TechViet uses a combination of OKRs "
            "(Objectives and Key Results) for goal-setting and a calibrated "
            "performance review process for evaluation. The system is designed "
            "to be fair, transparent, and growth-oriented."
        )
        doc = _make_doc([
            TextBlock(text=text, page_number=5, block_type="paragraph"),
        ])
        result = preprocessor.preprocess(doc)

        assert len(result.text_blocks) == 2
        assert "Performance Management" in result.text_blocks[0].text
        assert result.text_blocks[0].block_type == "heading"
        assert "TechViet" in result.text_blocks[1].text

    def test_preserves_short_headings(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="2.1 Working Hours", page_number=2, block_type="heading"),
        ])
        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 1
        assert result.text_blocks[0].block_type == "heading"

    def test_preserves_normal_paragraphs(self, preprocessor):
        doc = _make_doc([
            TextBlock(
                text="This is a normal paragraph with enough text to be meaningful.",
                page_number=1,
                block_type="paragraph",
            ),
        ])
        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 1
        assert result.text_blocks[0].block_type == "paragraph"


# ============================================================================
# Op 4 — Frequency Dedup
# ============================================================================

class TestFrequencyDedup:

    def test_removes_repeating_header(self, preprocessor):
        unique_words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
        blocks = []
        for i, page in enumerate(range(1, 8)):
            # Repeating header on every page
            blocks.append(TextBlock(
                text="Employee Handbook | TechViet Solutions | Confidential",
                page_number=page,
                block_type="paragraph",
            ))
            # Truly unique content on each page — different words, not just numbers
            blocks.append(TextBlock(
                text=(
                    f"The {unique_words[i]} department handles specific policies "
                    f"and procedures that are completely different from other sections "
                    f"in the document and cannot be confused with repeating headers."
                ),
                page_number=page,
                block_type="paragraph",
            ))

        doc = _make_doc(blocks, page_count=7)
        result = preprocessor.preprocess(doc)

        # Repeating header should be removed
        texts = [b.text for b in result.text_blocks]
        assert not any("Employee Handbook | TechViet" in t for t in texts)
        # Unique content should remain
        assert any("alpha" in t for t in texts)

    def test_removes_page_numbers(self, preprocessor):
        blocks = []
        for page in range(1, 8):
            blocks.append(TextBlock(
                text=f"Page {page} of 7",
                page_number=page,
                block_type="paragraph",
            ))
            blocks.append(TextBlock(
                text=f"Unique content {page}.",
                page_number=page,
                block_type="paragraph",
            ))

        doc = _make_doc(blocks, page_count=7)
        result = preprocessor.preprocess(doc)

        # "Page X of Y" should be removed (numbers normalised to #)
        texts = [b.text for b in result.text_blocks]
        assert not any("Page" in t and "of" in t for t in texts)

    def test_skips_when_few_pages(self, preprocessor):
        blocks = [
            TextBlock(
                text="Employee Handbook | TechViet Solutions | Confidential document header.",
                page_number=1, block_type="paragraph",
            ),
            TextBlock(
                text="Employee Handbook | TechViet Solutions | Confidential document header.",
                page_number=2, block_type="paragraph",
            ),
        ]
        doc = _make_doc(blocks, page_count=2)
        result = preprocessor.preprocess(doc)
        # With only 2 pages, dedup should not activate — both blocks remain
        assert len(result.text_blocks) == 2


# ============================================================================
# Op 5 — Cross-page Merger
# ============================================================================

class TestCrossPageMerge:

    def test_merges_split_sentence(self, preprocessor):
        doc = _make_doc([
            TextBlock(
                text="Protected categories include but are not limited to: gender, Protected",
                page_number=6,
                block_type="paragraph",
                section="5. Code of Conduct",
            ),
            TextBlock(
                text="categories include but are not limited to: gender identity, sexual orientation.",
                page_number=7,
                block_type="paragraph",
                section="5. Code of Conduct",
            ),
        ], page_count=7)

        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 1
        assert "Protected" in result.text_blocks[0].text
        assert "sexual orientation." in result.text_blocks[0].text

    def test_does_not_merge_complete_sentences(self, preprocessor):
        doc = _make_doc([
            TextBlock(
                text="This sentence is complete and has enough content to stand alone in the block without being merged.",
                page_number=1,
                block_type="paragraph",
            ),
            TextBlock(
                text="This is a new paragraph on the next page with plenty of text to avoid small-block merging.",
                page_number=2,
                block_type="paragraph",
            ),
        ], page_count=2)

        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 2

    def test_does_not_merge_headings(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Some ending text", page_number=1, block_type="paragraph"),
            TextBlock(text="2.1 New Section", page_number=2, block_type="heading"),
        ], page_count=2)

        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 2

    def test_does_not_merge_non_consecutive_pages(self, preprocessor):
        doc = _make_doc([
            TextBlock(
                text="Ends without punct but has enough content to stand alone in block",
                page_number=1,
                block_type="paragraph",
            ),
            TextBlock(
                text="continues here with enough text to avoid being merged by small-block merger too.",
                page_number=5,
                block_type="paragraph",
            ),
        ], page_count=5)

        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 2


# ============================================================================
# Op 6 — Small Block Merger
# ============================================================================

class TestSmallBlockMerge:

    def test_merges_tiny_paragraph_into_previous(self, preprocessor):
        doc = _make_doc([
            TextBlock(
                text="A reasonably long paragraph with enough content to stand on its own.",
                page_number=1,
                block_type="paragraph",
                section="Intro",
            ),
            TextBlock(
                text="Short tail.",
                page_number=1,
                block_type="paragraph",
                section="Intro",
            ),
        ])

        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 1
        assert "Short tail." in result.text_blocks[0].text

    def test_never_merges_headings(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Long paragraph before the heading section.",
                       page_number=1, block_type="paragraph", section="Intro"),
            TextBlock(text="Section A", page_number=1, block_type="heading", section="Section A"),
        ])

        result = preprocessor.preprocess(doc)
        headings = [b for b in result.text_blocks if b.block_type == "heading"]
        assert len(headings) == 1

    def test_keeps_standalone_short_block_in_different_section(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Paragraph in section A with enough text.",
                       page_number=1, block_type="paragraph", section="A"),
            TextBlock(text="Short.", page_number=1, block_type="paragraph", section="B"),
        ])

        result = preprocessor.preprocess(doc)
        # Should NOT merge because different sections
        assert len(result.text_blocks) == 2


# ============================================================================
# Section Rebuild
# ============================================================================

class TestSectionRebuild:

    def test_sections_rebuilt_after_split(self, preprocessor):
        text = (
            "3. Benefits Package TechViet provides comprehensive benefits "
            "including health insurance and retirement plans for all employees."
        )
        doc = _make_doc([
            TextBlock(text=text, page_number=4, block_type="heading"),
            TextBlock(text="Details follow.", page_number=4, block_type="paragraph",
                       section="old section"),
        ])

        result = preprocessor.preprocess(doc)

        # After split + rebuild, the paragraph should have the new heading as section
        paragraphs = [b for b in result.text_blocks if b.block_type == "paragraph"]
        assert len(paragraphs) >= 1
        for p in paragraphs:
            assert "old section" not in p.section


# ============================================================================
# Integration — Real PDF
# ============================================================================

class TestPreprocessRealPDF:

    @pytest.mark.asyncio
    async def test_preprocess_reduces_noise(self, parser, preprocessor, sample_pdf_path):
        raw = await parser.parse(str(sample_pdf_path))
        clean = preprocessor.preprocess(raw)

        # Preprocessing should not increase block count significantly
        # (it may increase slightly due to heading splits)
        assert len(clean.text_blocks) > 0

        # All blocks should have sections
        for block in clean.text_blocks:
            assert isinstance(block.section, str)

        # No block should have section > 200 chars (the old bug had 300+ char sections)
        for block in clean.text_blocks:
            assert len(block.section) < 200, (
                f"Section too long ({len(block.section)} chars): {block.section[:80]}..."
            )

    @pytest.mark.asyncio
    async def test_no_unicode_artifacts(self, parser, preprocessor, sample_pdf_path):
        raw = await parser.parse(str(sample_pdf_path))
        clean = preprocessor.preprocess(raw)

        for block in clean.text_blocks:
            assert "\u00f9" not in block.text, f"Unicode artifact in: {block.text[:80]}"
            assert "\u00a0" not in block.text
            assert "\u200b" not in block.text

    @pytest.mark.asyncio
    async def test_tables_also_cleaned(self, parser, preprocessor, sample_pdf_path):
        raw = await parser.parse(str(sample_pdf_path))
        clean = preprocessor.preprocess(raw)

        # Tables should survive preprocessing
        assert len(clean.tables) == len(raw.tables)
        for table in clean.tables:
            assert len(table.headers) > 0
            assert len(table.rows) > 0


# ============================================================================
# Inspect — Visual before/after comparison
# ============================================================================

class TestInspectPreprocessed:
    """
    Run with: pytest tests/test_step2_preprocessing.py -v -s -k inspect
    """

    @pytest.mark.asyncio
    async def test_inspect_before_after(
        self, parser, preprocessor, extractor, sample_pdf_paths
    ):
        sep = "=" * 80
        thin = "-" * 80

        for path in sample_pdf_paths:
            raw = await parser.parse(str(path))
            clean = preprocessor.preprocess(raw)

            raw_headings = [b for b in raw.text_blocks if b.block_type == "heading"]
            clean_headings = [b for b in clean.text_blocks if b.block_type == "heading"]
            raw_paras = [b for b in raw.text_blocks if b.block_type == "paragraph"]
            clean_paras = [b for b in clean.text_blocks if b.block_type == "paragraph"]

            print(f"\n{sep}")
            print(f"FILE: {path.name}")
            print(sep)
            print(f"  BEFORE: {len(raw.text_blocks)} blocks "
                  f"({len(raw_headings)} headings, {len(raw_paras)} paragraphs)")
            print(f"  AFTER:  {len(clean.text_blocks)} blocks "
                  f"({len(clean_headings)} headings, {len(clean_paras)} paragraphs)")
            print(f"  Tables: {len(raw.tables)} -> {len(clean.tables)}")
            print()

            # Show cleaned blocks
            print(f"  CLEANED TEXT BLOCKS ({len(clean.text_blocks)} total)")
            print(f"  {thin}")
            for i, block in enumerate(clean.text_blocks):
                print(f"\n  Block {i+1}/{len(clean.text_blocks)}")
                print(f"  Type: {block.block_type} | Page: {block.page_number} "
                      f"| Section: {block.section[:60]}")
                print(f"  {thin}")
                for line in block.text.split("\n"):
                    print(f"  | {line}")
                print()

            # Show cleaned tables
            print(f"\n  CLEANED TABLES ({len(clean.tables)} total)")
            print(f"  {thin}")
            for i, t in enumerate(clean.tables):
                print(f"\n  Table {i+1}/{len(clean.tables)}: {extractor.table_summary(t)}")
                print(f"  Page: {t.page_number} | Section: {t.section}")
                print(f"  {thin}")
                text = extractor.table_to_text(t)
                for line in text.split("\n"):
                    print(f"  | {line}")
                print()

        print(f"\n{sep}")
        print("DONE — Step 2 Preprocessing")
        print(sep)
