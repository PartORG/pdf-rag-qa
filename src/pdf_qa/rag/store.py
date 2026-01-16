"""Vector store using FAISS for similarity search."""

import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import faiss
import numpy as np

from pdf_qa.core.logging import get_logger
from pdf_qa.models.document import Chunk
from pdf_qa.models.qa import QueryResult

logger = get_logger(__name__)

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.pkl"
INFO_FILE = "index_info.json"


class VectorStore:
    """FAISS vector store for document chunks."""

    def __init__(self, persist_path: Path):
        self.persist_path = Path(persist_path)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: List[Dict[str, Any]] = []
        self._documents: List[str] = []

    def _ensure_dir(self) -> None:
        self.persist_path.mkdir(parents=True, exist_ok=True)

    def add_documents(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> int:
        """Add chunks with embeddings to the store."""
        if not chunks or not embeddings:
            return 0

        # Convert to numpy and normalize for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        # Initialize index if needed
        if self._index is None:
            dim = vectors.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # Inner product after L2 norm = cosine

        # Add vectors
        self._index.add(vectors)

        # Store metadata and documents
        for chunk in chunks:
            self._metadata.append(chunk.to_metadata())
            self._documents.append(chunk.content)

        logger.info(f"Added {len(chunks)} chunks to store\n")
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Search for similar documents."""
        if self._index is None or self._index.ntotal == 0:
            return []

        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # Search (get more if filtering)
        search_k = top_k * 3 if filter_dict else top_k
        search_k = min(search_k, self._index.ntotal)

        scores, indices = self._index.search(query, search_k)

        results: List[QueryResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            meta = self._metadata[idx]

            # Apply filter if specified
            if filter_dict:
                match = all(meta.get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue

            # Convert similarity (higher=better) to distance (lower=better)
            distance = 1.0 - float(score)

            results.append(QueryResult(
                chunk_content=self._documents[idx],
                source_file=meta["source_file"],
                page_number=meta["page_number"],
                score=distance,
                chunk_index=meta["chunk_index"],
            ))

            if len(results) >= top_k:
                break

        return results

    def get_document_count(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def get_unique_sources(self) -> List[str]:
        """Get list of unique source filenames."""
        sources = set()
        for meta in self._metadata:
            if "source_file" in meta:
                sources.add(meta["source_file"])
        return sorted(list(sources))

    def get_source_previews(self, max_chars: int = 100) -> Dict[str, str]:
        """Get first chunk preview for each source file."""
        previews: Dict[str, str] = {}
        for i, meta in enumerate(self._metadata):
            source = meta.get("source_file")
            if source and source not in previews:
                content = self._documents[i] if i < len(self._documents) else ""
                # Clean and truncate
                preview = " ".join(content.split())[:max_chars]
                if len(content) > max_chars:
                    preview = preview.rsplit(" ", 1)[0] + "..."
                previews[source] = preview
        return previews

    def exists(self) -> bool:
        return self.get_document_count() > 0

    def clear(self) -> None:
        """Clear all data."""
        self._index = None
        self._metadata = []
        self._documents = []

        # Remove files
        for fname in [INDEX_FILE, METADATA_FILE, INFO_FILE]:
            fpath = self.persist_path / fname
            if fpath.exists():
                fpath.unlink()

        logger.info("Cleared vector store\n")

    def save(self) -> None:
        """Persist index to disk."""
        if self._index is None:
            return

        self._ensure_dir()

        # Save FAISS index
        faiss.write_index(self._index, str(self.persist_path / INDEX_FILE))

        # Save metadata and documents
        with open(self.persist_path / METADATA_FILE, "wb") as f:
            pickle.dump({
                "metadata": self._metadata,
                "documents": self._documents,
            }, f)

        # Save info
        with open(self.persist_path / INFO_FILE, "w") as f:
            json.dump({
                "count": self._index.ntotal,
                "sources": self.get_unique_sources(),
            }, f, indent=2)

        logger.info(f"Saved index with {self._index.ntotal} vectors\n")

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        index_path = self.persist_path / INDEX_FILE
        meta_path = self.persist_path / METADATA_FILE

        if not index_path.exists() or not meta_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(index_path))

            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self._metadata = data["metadata"]
                self._documents = data["documents"]

            logger.info(f"Loaded index with {self._index.ntotal} vectors\n")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def save_metadata(self, index_version: str, settings_hash: str) -> None:
        """Save index metadata for version tracking."""
        info_path = self.persist_path / INFO_FILE
        info = {
            "index_version": index_version,
            "settings_hash": settings_hash,
            "count": self.get_document_count(),
            "sources": self.get_unique_sources(),
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load index metadata."""
        info_path = self.persist_path / INFO_FILE
        if not info_path.exists():
            return None
        try:
            with open(info_path, "r") as f:
                return json.load(f)
        except Exception:
            return None


def create_vector_store_from_settings() -> VectorStore:
    """Create a VectorStore from settings."""
    from pdf_qa.config.settings import get_settings
    settings = get_settings()
    store = VectorStore(persist_path=settings.vector_store_path)
    store.load()  # Try to load existing
    return store
