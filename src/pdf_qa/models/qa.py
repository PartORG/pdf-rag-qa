"""Q&A response models."""

from typing import List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A source reference for an answer."""

    filename: str
    page_number: int = Field(ge=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
    snippet: str = ""

    def format_reference(self) -> str:
        return f"{self.filename} (Page {self.page_number})"


class Answer(BaseModel):
    """Complete answer with sources."""

    question: str
    answer_text: str
    sources: List[Source] = Field(default_factory=list)
    model_used: str
    confidence: Optional[str] = None

    @property
    def has_sources(self) -> bool:
        return len(self.sources) > 0

    @property
    def unique_files(self) -> List[str]:
        return list(set(s.filename for s in self.sources))


class DocumentSummary(BaseModel):
    """Summary of a document."""

    filename: str
    title: Optional[str] = None
    summary: str
    page_count: int = Field(ge=1)
    topics: List[str] = Field(default_factory=list)

    def format_short(self) -> str:
        title = self.title or self.filename
        return f"{title}: {self.summary[:100]}..."


class QueryResult(BaseModel):
    """Result from vector store retrieval."""

    chunk_content: str
    source_file: str
    page_number: int = Field(ge=1)
    score: float
    chunk_index: int = Field(ge=0)

    def to_source(self) -> Source:
        """Convert to Source for answer formatting."""
        relevance = max(0.0, min(1.0, 1.0 - self.score))
        return Source(
            filename=self.source_file,
            page_number=self.page_number,
            relevance_score=relevance,
            snippet=self.chunk_content[:200] + "..." if len(self.chunk_content) > 200 else self.chunk_content,
        )
