"""
Step 1 Tests — Document Parsing.

Run:
    cd rag_chatbot
    pytest tests/test_step1_parsing.py -v
    pytest tests/test_step1_parsing.py -v -s -k inspect   # see full output
"""
import pytest
from backend.services.document_parser import DocumentParser, ParsedDocument, TextBlock, Table
from backend.services.table_extractor import TableExtractor


@pytest.fixture
def parser():
    return DocumentParser()


@pytest.fixture
def extractor():
    return TableExtractor()


class TestDocumentParser:

    @pytest.mark.asyncio
    async def test_parse_returns_parsed_document(self, parser, sample_pdf_path):
        result = await parser.parse(str(sample_pdf_path))
        assert isinstance(result, ParsedDocument)
        assert result.file_type == "pdf"
        assert result.page_count > 0

    @pytest.mark.asyncio
    async def test_text_blocks_extracted(self, parser, sample_pdf_path):
        result = await parser.parse(str(sample_pdf_path))
        assert len(result.text_blocks) > 0
        for block in result.text_blocks:
            assert isinstance(block, TextBlock)
            assert block.text.strip() != ""
            assert block.block_type in ("title", "heading", "paragraph")

    @pytest.mark.asyncio
    async def test_text_blocks_have_page_numbers(self, parser, sample_pdf_path):
        result = await parser.parse(str(sample_pdf_path))
        for block in result.text_blocks:
            assert block.page_number is not None
            assert block.page_number >= 1

    @pytest.mark.asyncio
    async def test_tables_extracted(self, parser, sample_pdf_path):
        result = await parser.parse(str(sample_pdf_path))
        assert len(result.tables) > 0
        for table in result.tables:
            assert isinstance(table, Table)
            assert len(table.headers) > 0
            assert len(table.rows) > 0

    @pytest.mark.asyncio
    async def test_tables_have_metadata(self, parser, sample_pdf_path):
        result = await parser.parse(str(sample_pdf_path))
        for table in result.tables:
            assert table.page_number is not None
            assert table.name != ""

    @pytest.mark.asyncio
    async def test_table_rows_match_header_length(self, parser, sample_pdf_path):
        result = await parser.parse(str(sample_pdf_path))
        for table in result.tables:
            for row in table.rows:
                assert len(row) == len(table.headers)

    @pytest.mark.asyncio
    async def test_to_dict_roundtrip(self, parser, sample_pdf_path):
        result = await parser.parse(str(sample_pdf_path))
        d = result.to_dict()
        assert "text_blocks" in d
        assert "tables" in d
        assert "metadata" in d
        assert d["metadata"]["page_count"] == result.page_count

    @pytest.mark.asyncio
    async def test_parse_all_sample_files(self, parser, sample_pdf_paths):
        for path in sample_pdf_paths:
            result = await parser.parse(str(path))
            assert len(result.text_blocks) > 0

    @pytest.mark.asyncio
    async def test_unsupported_format_raises(self, parser, tmp_path):
        fake_file = tmp_path / "test.xlsx"
        fake_file.write_text("fake")
        with pytest.raises(ValueError, match="Unsupported format"):
            await parser.parse(str(fake_file))


class TestTableExtractor:

    @pytest.mark.asyncio
    async def test_extract_tables_filters_empty(self, parser, extractor, sample_pdf_path):
        parsed = await parser.parse(str(sample_pdf_path))
        tables = extractor.extract_tables(parsed)
        for table in tables:
            assert not table.is_empty

    @pytest.mark.asyncio
    async def test_table_to_text_format(self, parser, extractor, sample_pdf_path):
        parsed = await parser.parse(str(sample_pdf_path))
        tables = extractor.extract_tables(parsed)
        assert len(tables) > 0
        text = extractor.table_to_text(tables[0])
        assert "Table:" in text
        assert "Row 1:" in text

    @pytest.mark.asyncio
    async def test_table_summary(self, parser, extractor, sample_pdf_path):
        parsed = await parser.parse(str(sample_pdf_path))
        tables = extractor.extract_tables(parsed)
        assert len(tables) > 0
        summary = extractor.table_summary(tables[0])
        assert "columns" in summary
        assert "rows" in summary


class TestInspectOutput:
    """
    Run with: pytest tests/test_step1_parsing.py -v -s -k inspect
    """

    @pytest.mark.asyncio
    async def test_inspect_parsed_documents(self, parser, extractor, sample_pdf_paths):
        sep = "=" * 80
        thin = "-" * 80

        for path in sample_pdf_paths:
            result = await parser.parse(str(path))
            tables = extractor.extract_tables(result)
            headings = [b for b in result.text_blocks if b.block_type == "heading"]
            paragraphs = [b for b in result.text_blocks if b.block_type == "paragraph"]

            print(f"\n{sep}")
            print(f"FILE: {path.name}")
            print(f"Pages: {result.page_count} | "
                  f"Text blocks: {len(result.text_blocks)} "
                  f"({len(headings)} headings, {len(paragraphs)} paragraphs) | "
                  f"Tables: {len(tables)}")
            print(sep)

            print(f"\n  TEXT BLOCKS ({len(result.text_blocks)} total)")
            print(f"  {thin}")
            for i, block in enumerate(result.text_blocks):
                print(f"\n  Block {i+1}/{len(result.text_blocks)}")
                print(f"  Type: {block.block_type} | Page: {block.page_number} | Section: {block.section}")
                print(f"  {thin}")
                for line in block.text.split("\n"):
                    print(f"  | {line}")
                print()

            print(f"\n  TABLES ({len(tables)} total)")
            print(f"  {thin}")
            for i, t in enumerate(tables):
                print(f"\n  Table {i+1}/{len(tables)}: {extractor.table_summary(t)}")
                print(f"  Page: {t.page_number} | Section: {t.section}")
                print(f"  {thin}")
                text = extractor.table_to_text(t)
                for line in text.split("\n"):
                    print(f"  | {line}")
                print()

        print(f"\n{sep}")
        print("DONE")
        print(sep)
