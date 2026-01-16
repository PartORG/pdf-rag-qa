"""Document retrieval from vector store."""

from typing import List, Optional

from pdf_qa.core.logging import get_logger
from pdf_qa.models.qa import QueryResult, Source
from pdf_qa.rag.embeddings import EmbeddingService
from pdf_qa.rag.store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """Retrieves relevant document chunks using semantic search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        top_k: int = 5,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> List[QueryResult]:
        """Retrieve relevant chunks using semantic search."""
        k = top_k or self.top_k

        logger.debug(f"Retrieving top {k} chunks for query: {query[:50]}...")

        query_embedding = self.embedding_service.embed_query(query)

        filter_dict = None
        if source_filter:
            filter_dict = {"source_file": source_filter}

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            filter_dict=filter_dict,
        )

        logger.debug(f"Retrieved {len(results)} chunks")
        return results

    def retrieve_with_diversity(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[QueryResult]:
        """Retrieve results with source diversity to avoid redundant chunks."""
        k = top_k or self.top_k

        fetch_k = min(k * 3, 20)
        all_results = self.retrieve(query, top_k=fetch_k)

        if len(all_results) <= k:
            return all_results

        selected: List[QueryResult] = []
        seen_sources: dict[str, int] = {}

        for result in all_results:
            source = result.source_file
            source_count = seen_sources.get(source, 0)

            max_per_source = max(1, k // 2)
            if source_count < max_per_source:
                selected.append(result)
                seen_sources[source] = source_count + 1

            if len(selected) >= k:
                break

        if len(selected) < k:
            for result in all_results:
                if result not in selected:
                    selected.append(result)
                    if len(selected) >= k:
                        break

        return selected

    def get_sources_for_results(self, results: List[QueryResult]) -> List[Source]:
        """Convert results to unique source references."""
        seen: set[tuple[str, int]] = set()
        sources: List[Source] = []

        for result in results:
            key = (result.source_file, result.page_number)
            if key not in seen:
                seen.add(key)
                sources.append(result.to_source())

        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return sources

    def build_context(
        self,
        results: List[QueryResult],
        max_context_chars: int = 4000,
    ) -> str:
        """Build context string from retrieved results for LLM."""
        context_parts: List[str] = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            chunk_text = (
                f"[Source: {result.source_file}, Page {result.page_number}]\n"
                f"{result.chunk_content}"
            )

            if total_chars + len(chunk_text) > max_context_chars:
                remaining = max_context_chars - total_chars - 100
                if remaining > 200:
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(chunk_text)
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text) + 10

        return "\n\n---\n\n".join(context_parts)


def create_retriever_from_settings() -> Retriever:
    """Create Retriever from settings."""
    from pdf_qa.config.settings import get_settings
    from pdf_qa.rag.embeddings import create_embedding_service_from_settings
    from pdf_qa.rag.store import create_vector_store_from_settings

    settings = get_settings()
    embedding_service = create_embedding_service_from_settings()
    vector_store = create_vector_store_from_settings()

    return Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=settings.top_k_results,
    )
