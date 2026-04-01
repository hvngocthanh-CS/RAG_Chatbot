"""
Step 2 Tests -- Document Preprocessing.

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
# Helpers
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
# Op 1 -- Unicode Repair
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
            TextBlock(text="con\ufb01dential \ufb02ow", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "fi" in result.text_blocks[0].text
        assert "fl" in result.text_blocks[0].text

    def test_collapses_whitespace(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="hello   world   foo", page_number=1),
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
# Op 2 -- Artifact Repair
# ============================================================================

class TestArtifactRepair:

    def test_fixes_split_words(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Statu s repor t processin g", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "Status" in result.text_blocks[0].text
        assert "report" in result.text_blocks[0].text
        assert "processing" in result.text_blocks[0].text

    def test_preserves_real_short_words(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Guide to the department of engineering in Asia", page_number=1),
        ])
        result = preprocessor.preprocess(doc)
        assert "Guide to the department of engineering in Asia" == result.text_blocks[0].text

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
# Op 3 -- Title Page Splitter (NEW)
# ============================================================================

class TestTitlePageSplitter:

    def test_splits_typical_title_page(self, preprocessor):
        doc = _make_doc([
            TextBlock(
                text=(
                    "Employee Handbook Comprehensive Guide to Policies, Benefits, "
                    "and Workplace Standards Human Resources Department Version 3.2 | "
                    "Effective January 1, 2026 | Approved by CEO & CHRO"
                ),
                page_number=1,
                block_type="heading",
            ),
            TextBlock(
                text="1. Company Overview",
                page_number=1,
                block_type="heading",
            ),
        ])

        result = preprocessor.preprocess(doc)

        # First block should be short title
        assert result.text_blocks[0].block_type == "heading"
        assert result.text_blocks[0].text == "Employee Handbook"
        assert len(result.text_blocks[0].text) < 50

        # Second block should be subtitle/metadata
        assert result.text_blocks[1].block_type == "paragraph"
        assert "Comprehensive" in result.text_blocks[1].text

    def test_splits_security_policy_title(self, preprocessor):
        doc = _make_doc([
            TextBlock(
                text=(
                    "Information Security Policy Comprehensive Security Framework "
                    "for Systems, Data, and Personnel IT Security Department | "
                    "Version 4.0 | Last Updated March 2026"
                ),
                page_number=1,
                block_type="heading",
            ),
        ])

        result = preprocessor.preprocess(doc)
        assert result.text_blocks[0].text == "Information Security Policy"
        assert result.text_blocks[0].block_type == "heading"

    def test_preserves_short_headings(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Employee Handbook", page_number=1, block_type="heading"),
        ])
        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 1
        assert result.text_blocks[0].text == "Employee Handbook"

    def test_only_splits_first_heading_on_page1(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Short Title", page_number=1, block_type="heading"),
            TextBlock(
                text=(
                    "This is a very long heading on page 1 that should not be split "
                    "because the title page splitter already processed page 1 and only "
                    "splits the first long heading block it encounters on page 1."
                ),
                page_number=1,
                block_type="heading",
            ),
        ])
        result = preprocessor.preprocess(doc)
        # First stays short, second is handled by op4 heading splitter (not title splitter)
        assert result.text_blocks[0].text == "Short Title"

    def test_does_not_split_page2_headings(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Short", page_number=1, block_type="heading"),
            TextBlock(
                text=(
                    "Very Long Heading on Page 2 Comprehensive Guide to Everything "
                    "That Should Not Be Title-Split Because It Is Not on Page 1"
                ),
                page_number=2,
                block_type="heading",
            ),
        ], page_count=2)
        result = preprocessor.preprocess(doc)
        # page 2 heading should NOT be affected by title splitter
        # (may still be split by op4 heading splitter if it matches patterns)
        page2_blocks = [b for b in result.text_blocks if b.page_number == 2]
        assert len(page2_blocks) >= 1


# ============================================================================
# Op 4 -- Heading Splitter
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

        headings = [b for b in result.text_blocks if b.block_type == "heading"]
        paragraphs = [b for b in result.text_blocks if b.block_type == "paragraph"]
        assert len(headings) >= 1
        assert len(paragraphs) >= 1
        assert len(headings[0].text) < 120

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

    def test_splits_heading_with_subheading(self, preprocessor):
        """Pattern B: heading merged with sub-heading."""
        text = (
            "2. Code Quality Standards 2.1 Pull Request and Code Review "
            "Requirements Code review is one of our most important quality "
            "practices for ensuring high-quality software delivery."
        )
        doc = _make_doc([
            TextBlock(text=text, page_number=2, block_type="heading"),
        ])
        result = preprocessor.preprocess(doc)

        # Should split into two parts at "2.1"
        assert len(result.text_blocks) >= 2
        assert "2. Code Quality Standards" in result.text_blocks[0].text
        assert "2.1" in result.text_blocks[1].text

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
# Op 5 -- Frequency Dedup
# ============================================================================

class TestFrequencyDedup:

    def test_removes_repeating_header(self, preprocessor):
        unique_words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
        blocks = []
        for i, page in enumerate(range(1, 8)):
            blocks.append(TextBlock(
                text="Employee Handbook | TechViet Solutions | Confidential",
                page_number=page,
                block_type="paragraph",
            ))
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

        texts = [b.text for b in result.text_blocks]
        assert not any("Employee Handbook | TechViet" in t for t in texts)
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
        assert len(result.text_blocks) == 2


# ============================================================================
# Op 6 -- Cross-page Merger
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
# Op 7 -- Small Block Merger
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

        paragraphs = [b for b in result.text_blocks if b.block_type == "paragraph"]
        assert len(paragraphs) >= 1
        for p in paragraphs:
            assert "old section" not in p.section


# ============================================================================
# Integration -- Real PDF
# ============================================================================

class TestPreprocessRealPDF:

    @pytest.mark.asyncio
    async def test_preprocess_reduces_noise(self, parser, preprocessor, sample_pdf_path):
        raw = await parser.parse(str(sample_pdf_path))
        clean = preprocessor.preprocess(raw)

        assert len(clean.text_blocks) > 0

        for block in clean.text_blocks:
            assert isinstance(block.section, str)

        # No section should be excessively long (the old 300+ char bug)
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

        assert len(clean.tables) == len(raw.tables)
        for table in clean.tables:
            assert len(table.headers) > 0
            assert len(table.rows) > 0

    @pytest.mark.asyncio
    async def test_title_page_is_split(self, parser, preprocessor, sample_pdf_path):
        """The first block should be a short document title, not a 176-char blob."""
        raw = await parser.parse(str(sample_pdf_path))
        clean = preprocessor.preprocess(raw)

        first_heading = next(
            (b for b in clean.text_blocks if b.block_type == "heading"), None
        )
        assert first_heading is not None
        # After title-page split, the first heading should be the document title
        # (reasonably short), not the entire title+subtitle+metadata blob
        assert len(first_heading.text) < 100, (
            f"First heading still too long: {first_heading.text[:80]}..."
        )


# ============================================================================
# Full pipeline -- All 20 docs
# ============================================================================

class TestPreprocessAllDocs:

    @pytest.mark.asyncio
    async def test_all_docs_no_long_sections(self, parser, preprocessor, sample_pdf_paths):
        """After preprocessing, no section should be excessively long."""
        for path in sample_pdf_paths:
            raw = await parser.parse(str(path))
            clean = preprocessor.preprocess(raw)

            for block in clean.text_blocks:
                assert len(block.section) < 200, (
                    f"{path.name} section too long ({len(block.section)}): "
                    f"{block.section[:60]}..."
                )


# ============================================================================
# Inspect -- Visual before/after comparison
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

            print(f"  CLEANED TEXT BLOCKS ({len(clean.text_blocks)} total)")
            print(f"  {thin}")
            for i, block in enumerate(clean.text_blocks):
                sec = block.section[:60]
                print(f"\n  Block {i+1}/{len(clean.text_blocks)}")
                print(f"  Type: {block.block_type} | Page: {block.page_number} "
                      f"| Section: {sec}")
                print(f"  {thin}")
                for line in block.text.split("\n"):
                    print(f"  | {line}")
                print()

        print(f"\n{sep}")
        print("DONE -- Step 2 Preprocessing")
        print(sep)
