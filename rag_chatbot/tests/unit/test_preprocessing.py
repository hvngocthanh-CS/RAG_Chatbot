"""
Step 2 Tests -- Document Preprocessing.

Run:
    cd rag_chatbot
    pytest tests/unit/test_preprocessing.py -v
"""
import pytest
from backend.models import ParsedDocument, TextBlock, Table
from backend.services.ingestion import DocumentParser, DocumentPreprocessor, TableExtractor


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
        assert "\u2013" in result.text_blocks[0].text

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
        assert "\u2013" in result.tables[0].rows[0][0]


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
# Op 3 -- Title Page Splitter
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

        assert result.text_blocks[0].block_type == "heading"
        assert result.text_blocks[0].text == "Employee Handbook"
        assert len(result.text_blocks[0].text) < 50

        assert result.text_blocks[1].block_type == "paragraph"
        assert "Comprehensive" in result.text_blocks[1].text

    def test_preserves_short_headings(self, preprocessor):
        doc = _make_doc([
            TextBlock(text="Employee Handbook", page_number=1, block_type="heading"),
        ])
        result = preprocessor.preprocess(doc)
        assert len(result.text_blocks) == 1
        assert result.text_blocks[0].text == "Employee Handbook"


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
            assert len(block.section) < 200

    @pytest.mark.asyncio
    async def test_no_unicode_artifacts(self, parser, preprocessor, sample_pdf_path):
        raw = await parser.parse(str(sample_pdf_path))
        clean = preprocessor.preprocess(raw)

        for block in clean.text_blocks:
            assert "\u00f9" not in block.text
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
