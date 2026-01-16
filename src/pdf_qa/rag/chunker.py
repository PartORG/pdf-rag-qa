"""Text chunking for document processing.

Splits documents into overlapping chunks suitable for embedding,
preserving metadata about source location for citation.
"""

import re
from typing import List

from pdf_qa.core.logging import get_logger
from pdf_qa.models.document import Document, Chunk, PageContent

logger = get_logger(__name__)


class TextChunker:
    """Splits documents into chunks with configurable size and overlap."""

    # Pre-compiled regex for faster break point detection
    BREAK_PATTERN = re.compile(r"([.!?]\s+|\n\n|\s+)")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Process a single document into chunks."""
        all_chunks: List[Chunk] = []
        idx = 0
        for page in document.pages:
            if page.is_empty:
                continue
            page_chunks = self._chunk_page(page, document.filename, idx)
            all_chunks.extend(page_chunks)
            idx += len(page_chunks)

        logger.debug(f"Created {len(all_chunks)} chunks from {document.filename}")
        return all_chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents\n")
        return all_chunks

    def _chunk_page(
        self,
        page: PageContent,
        source_file: str,
        start_index: int,
    ) -> List[Chunk]:
        """Split a single page into chunks."""
        text = page.content
        text_len = len(text)
        chunks: List[Chunk] = []
        idx = start_index

        # Small text fits in one chunk
        if text_len <= self.chunk_size:
            if len(text.strip()) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=text.strip(),
                    source_file=source_file,
                    page_number=page.page_number,
                    chunk_index=idx,
                    start_char=0,
                    end_char=text_len,
                ))
            return chunks

        # Process larger text with overlapping chunks
        current_pos = 0
        while current_pos < text_len:
            end = min(current_pos + self.chunk_size, text_len)

            # Find natural break point near end
            if end < text_len:
                search_start = max(current_pos, end - 100)
                search_region = text[search_start:end]
                matches = list(self.BREAK_PATTERN.finditer(search_region))
                if matches:
                    end = search_start + matches[-1].end()

            chunk_text = text[current_pos:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    source_file=source_file,
                    page_number=page.page_number,
                    chunk_index=idx,
                    start_char=current_pos,
                    end_char=end,
                ))
                idx += 1

            # Move forward with overlap
            new_pos = end - self.chunk_overlap
            if new_pos <= current_pos:
                new_pos = end  # Ensure forward progress
            current_pos = new_pos

        return chunks


def create_chunker_from_settings() -> TextChunker:
    """Create a TextChunker from application settings."""
    from pdf_qa.config.settings import get_settings

    settings = get_settings()
    return TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
