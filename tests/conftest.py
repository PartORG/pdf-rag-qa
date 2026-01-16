"""Pytest fixtures for PDF Q&A tests."""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from pdf_qa.models.document import Document, PageContent, Chunk
from pdf_qa.models.qa import Answer, Source, QueryResult


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    return Document(
        filename="test_paper.pdf",
        filepath="/path/to/test_paper.pdf",
        pages=[
            PageContent(page_number=1, content="This is the introduction to machine learning."),
            PageContent(page_number=2, content="Neural networks are computational systems."),
            PageContent(page_number=3, content="Deep learning has transformed AI research."),
        ],
        total_pages=3,
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            content="This is the introduction to machine learning.",
            source_file="test_paper.pdf",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=46,
        ),
        Chunk(
            content="Neural networks are computational systems.",
            source_file="test_paper.pdf",
            page_number=2,
            chunk_index=1,
            start_char=0,
            end_char=42,
        ),
        Chunk(
            content="Deep learning has transformed AI research.",
            source_file="test_paper.pdf",
            page_number=3,
            chunk_index=2,
            start_char=0,
            end_char=42,
        ),
    ]


@pytest.fixture
def sample_query_results() -> list[QueryResult]:
    """Create sample query results for testing."""
    return [
        QueryResult(
            chunk_content="Neural networks are computational systems.",
            source_file="test_paper.pdf",
            page_number=2,
            score=0.15,
            chunk_index=1,
        ),
        QueryResult(
            chunk_content="Deep learning has transformed AI research.",
            source_file="test_paper.pdf",
            page_number=3,
            score=0.25,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_answer() -> Answer:
    """Create a sample answer for testing."""
    return Answer(
        question="What is machine learning?",
        answer_text="Machine learning is a subset of AI that enables systems to learn from data.",
        sources=[
            Source(
                filename="test_paper.pdf",
                page_number=1,
                relevance_score=0.85,
                snippet="This is the introduction to machine learning.",
            ),
        ],
        model_used="llama3.2",
        confidence="high",
    )


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """Create a mock Ollama client."""
    client = MagicMock()
    client.embeddings.return_value = {"embedding": [0.1] * 768}
    client.chat.return_value = {"message": {"content": "Test response"}}
    client.list.return_value = {"models": [{"name": "llama3.2"}]}
    return client


@pytest.fixture
def mock_settings(temp_dir: Path) -> MagicMock:
    """Create mock settings."""
    settings = MagicMock()
    settings.ollama_host = "http://localhost:11434"
    settings.ollama_model = "llama3.2"
    settings.ollama_embedding_model = "mxbai-embed-large"
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    settings.top_k_results = 5
    settings.documents_path = temp_dir / "docs"
    settings.vector_store_path = temp_dir / "vectors"
    settings.log_level = "INFO"
    settings.generation_timeout = 120
    settings.get_index_version.return_value = "mxbai-embed-large:512:50"
    return settings
