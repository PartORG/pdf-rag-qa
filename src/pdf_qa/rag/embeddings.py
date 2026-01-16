"""Embedding generation using Ollama."""

from typing import List, Optional

import ollama
from ollama import ResponseError

from pdf_qa.core.exceptions import (
    EmbeddingModelNotFoundError,
    EmbeddingFailedError,
    OllamaConnectionError,
)
from pdf_qa.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Generates embeddings using Ollama."""

    def __init__(
        self,
        model_name: str = "mxbai-embed-large",
        host: str = "http://localhost:11434",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.host = host
        self.batch_size = batch_size
        self._client: Optional[ollama.Client] = None

    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        embeddings: List[List[float]] = []

        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_embeddings = self._embed_batch(batch)
                embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Embedded batch {i // self.batch_size + 1}: "
                    f"{len(batch_embeddings)} texts"
                )

            return embeddings

        except ConnectionError as e:
            raise OllamaConnectionError(self.host)
        except ResponseError as e:
            if "not found" in str(e).lower():
                raise EmbeddingModelNotFoundError(self.model_name)
            raise EmbeddingFailedError(str(e))
        except Exception as e:
            raise EmbeddingFailedError(f"Unexpected error: {e}")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        embeddings = []
        for text in texts:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text,
            )
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=query,
            )
            return response["embedding"]

        except ConnectionError:
            raise OllamaConnectionError(self.host)
        except ResponseError as e:
            if "not found" in str(e).lower():
                raise EmbeddingModelNotFoundError(self.model_name)
            raise EmbeddingFailedError(str(e))
        except Exception as e:
            raise EmbeddingFailedError(f"Unexpected error: {e}")

    def is_available(self) -> bool:
        """Check if embedding model is available."""
        try:
            self.embed_query("test")
            return True
        except Exception:
            return False


def create_embedding_service_from_settings() -> EmbeddingService:
    """Create EmbeddingService from settings."""
    from pdf_qa.config.settings import get_settings

    settings = get_settings()
    return EmbeddingService(
        model_name=settings.ollama_embedding_model,
        host=settings.ollama_host,
    )
