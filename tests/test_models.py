"""Tests for data models."""

import pytest
from pydantic import ValidationError
from pdf_qa.models.document import Document, PageContent, Chunk
from pdf_qa.models.qa import Answer, Source, QueryResult


class TestDocumentModels:
    """Tests for document models."""

    def test_page_content(self) -> None:
        page = PageContent(page_number=1, content="Test")
        assert page.is_empty is False
        assert PageContent(page_number=2, content="   ").is_empty is True

        with pytest.raises(ValidationError):
            PageContent(page_number=0, content="Test")

    def test_document_properties(self, sample_document: Document) -> None:
        assert sample_document.has_content is True
        assert sample_document.non_empty_page_count == 3
        assert "introduction" in sample_document.full_text.lower()

    def test_chunk(self) -> None:
        chunk = Chunk(
            content="Test", source_file="test.pdf",
            page_number=1, chunk_index=0, start_char=0, end_char=4,
        )
        assert chunk.get_chunk_id() == "test.pdf:p1:c0"
        assert chunk.to_metadata()["source_file"] == "test.pdf"

        with pytest.raises(ValidationError):
            Chunk(content="", source_file="t.pdf", page_number=1, chunk_index=0, start_char=0, end_char=0)


class TestQAModels:
    """Tests for Q&A models."""

    def test_source(self) -> None:
        source = Source(filename="doc.pdf", page_number=5, relevance_score=0.85)
        assert source.format_reference() == "doc.pdf (Page 5)"

        with pytest.raises(ValidationError):
            Source(filename="t.pdf", page_number=1, relevance_score=1.5)

    def test_answer(self, sample_answer: Answer) -> None:
        assert sample_answer.has_sources is True
        assert "test_paper.pdf" in sample_answer.unique_files

    def test_query_result(self) -> None:
        result = QueryResult(
            chunk_content="A" * 300,
            source_file="test.pdf", page_number=1, score=0.2, chunk_index=0,
        )
        source = result.to_source()
        assert source.relevance_score == 0.8
        assert source.snippet.endswith("...")
