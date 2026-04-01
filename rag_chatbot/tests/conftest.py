"""
Shared test fixtures and configuration.

All test files automatically have access to fixtures defined here.
"""
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent
TEST_SAMPLES_DIR = TESTS_DIR / "test_samples"


@pytest.fixture
def test_samples_dir() -> Path:
    """Path to the test_samples/ folder containing sample PDFs."""
    assert TEST_SAMPLES_DIR.exists(), f"Missing test samples: {TEST_SAMPLES_DIR}"
    return TEST_SAMPLES_DIR


@pytest.fixture
def sample_pdf_paths(test_samples_dir) -> list[Path]:
    """All PDF files in test_samples/, sorted by name."""
    pdfs = sorted(test_samples_dir.glob("*.pdf"))
    assert len(pdfs) > 0, "No PDF files found in test_samples/"
    return pdfs


@pytest.fixture
def sample_pdf_path(sample_pdf_paths) -> Path:
    """A single sample PDF for quick tests (first file)."""
    return sample_pdf_paths[0]
