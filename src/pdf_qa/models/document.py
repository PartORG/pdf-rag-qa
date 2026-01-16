"""Document models for RAG pipeline."""

from typing import List

from pydantic import BaseModel, Field, computed_field


class PageContent(BaseModel):
    """Text content from a single PDF page."""

    page_number: int = Field(ge=1)
    content: str

    @computed_field
    @property
    def is_empty(self) -> bool:
        return len(self.content.strip()) == 0


class Document(BaseModel):
    """A loaded PDF document."""

    filename: str
    filepath: str
    pages: List[PageContent] = Field(default_factory=list)
    total_pages: int = Field(ge=0)

    @computed_field
    @property
    def full_text(self) -> str:
        return "\n\n".join(
            page.content for page in self.pages if not page.is_empty
        )

    @computed_field
    @property
    def has_content(self) -> bool:
        return any(not page.is_empty for page in self.pages)

    @computed_field
    @property
    def non_empty_page_count(self) -> int:
        return sum(1 for page in self.pages if not page.is_empty)


class Chunk(BaseModel):
    """A text chunk ready for embedding."""

    content: str = Field(min_length=1)
    source_file: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    def to_metadata(self) -> dict:
        """Convert to vector store metadata."""
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.content)

    def get_chunk_id(self) -> str:
        """Generate unique ID for this chunk."""
        return f"{self.source_file}:p{self.page_number}:c{self.chunk_index}"
