"""Tests for text chunking."""

import pytest
from pdf_qa.rag.chunker import TextChunker
from pdf_qa.models.document import Document, PageContent


class TestTextChunker:
    """Tests for TextChunker."""

    def test_init_validation(self) -> None:
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500

        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_small_document(self, sample_document: Document) -> None:
        chunker = TextChunker(chunk_size=1000, chunk_overlap=50, min_chunk_size=10)
        chunks = chunker.chunk_document(sample_document)

        assert len(chunks) == 3
        assert chunks[0].source_file == "test_paper.pdf"
        assert chunks[0].page_number == 1

    def test_chunk_large_text(self) -> None:
        large_text = "This is a sentence. " * 100
        doc = Document(
            filename="large.pdf", filepath="/path/large.pdf",
            pages=[PageContent(page_number=1, content=large_text)],
            total_pages=1,
        )

        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 250

    def test_empty_page_skipped(self) -> None:
        doc = Document(
            filename="sparse.pdf", filepath="/path/sparse.pdf",
            pages=[
                PageContent(page_number=1, content="Content page 1 with enough chars."),
                PageContent(page_number=2, content="   "),
                PageContent(page_number=3, content="Content page 3 with enough chars."),
            ],
            total_pages=3,
        )

        chunker = TextChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=10)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 3

    def test_min_chunk_size_filtering(self) -> None:
        doc = Document(
            filename="tiny.pdf", filepath="/path/tiny.pdf",
            pages=[PageContent(page_number=1, content="Hi")],
            total_pages=1,
        )

        chunker = TextChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=50)
        assert len(chunker.chunk_document(doc)) == 0

    def test_multiple_documents(self) -> None:
        docs = [
            Document(
                filename=f"doc{i}.pdf", filepath=f"/path/doc{i}.pdf",
                pages=[PageContent(page_number=1, content=f"Content for doc {i} with enough text.")],
                total_pages=1,
            )
            for i in range(3)
        ]

        chunker = TextChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=10)
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 3
        assert chunks[0].source_file == "doc0.pdf"
