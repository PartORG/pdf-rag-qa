"""Tests for input validation."""

import pytest
from pathlib import Path

from pdf_qa.core.validators import validate_query, validate_pdf_file, validate_directory, is_describe_query
from pdf_qa.core.exceptions import EmptyQueryError, QueryTooLongError, DocumentError


class TestQueryValidation:
    """Tests for query validation."""

    def test_valid_query(self) -> None:
        assert validate_query("What is AI?") == "What is AI?"
        assert validate_query("  What is AI?  ") == "What is AI?"
        assert validate_query("What   is   AI?") == "What is AI?"

    def test_invalid_query(self) -> None:
        with pytest.raises(EmptyQueryError):
            validate_query("")
        with pytest.raises(EmptyQueryError):
            validate_query("   ")
        with pytest.raises(EmptyQueryError):
            validate_query("a")
        with pytest.raises(QueryTooLongError):
            validate_query("a" * 2500)


class TestFileValidation:
    """Tests for file/directory validation."""

    def test_pdf_file_validation(self, temp_dir: Path) -> None:
        with pytest.raises(DocumentError):
            validate_pdf_file(Path("/nonexistent.pdf"))
        with pytest.raises(DocumentError):
            validate_pdf_file(temp_dir)  # directory

        txt = temp_dir / "test.txt"
        txt.write_text("test")
        with pytest.raises(DocumentError):
            validate_pdf_file(txt)

        pdf = temp_dir / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        assert validate_pdf_file(pdf) is True

    def test_directory_validation(self, temp_dir: Path) -> None:
        with pytest.raises(DocumentError):
            validate_directory(Path("/nonexistent"))

        assert validate_directory(temp_dir, must_contain_pdfs=False) == []

        with pytest.raises(DocumentError):
            validate_directory(temp_dir, must_contain_pdfs=True)

        (temp_dir / "doc.pdf").write_bytes(b"%PDF-1.4 test")
        result = validate_directory(temp_dir)
        assert len(result) == 1


class TestDescribeQuery:
    """Tests for describe query detection."""

    def test_describe_queries(self) -> None:
        assert is_describe_query("describe the documents") is True
        assert is_describe_query("list documents") is True
        assert is_describe_query("what files do you have") is True

    def test_regular_queries(self) -> None:
        assert is_describe_query("what is machine learning") is False
        assert is_describe_query("explain the methodology") is False
