"""Application settings using Pydantic BaseSettings.

Configuration is loaded from environment variables and .env file.
Settings are validated on application startup.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pdf_qa.core.exceptions import InvalidSettingError


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Attributes:
        ollama_host: Ollama server URL.
        ollama_model: LLM model name for text generation.
        ollama_embedding_model: Model name for embeddings.
        chunk_size: Maximum characters per text chunk.
        chunk_overlap: Overlapping characters between chunks.
        top_k_results: Number of similar chunks to retrieve.
        documents_path: Path to PDF documents directory.
        vector_store_path: Path to persist vector store.
        log_level: Logging verbosity level.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="LLM model for text generation",
    )
    ollama_embedding_model: str = Field(
        default="mxbai-embed-large",
        description="Model for generating embeddings",
    )

    # RAG Configuration
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=4000,
        description="Maximum characters per chunk",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks",
    )
    top_k_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )

    # Paths
    documents_path: Path = Field(
        default=Path("./research_papers"),
        description="Directory containing PDF documents",
    )
    vector_store_path: Path = Field(
        default=Path("./data/faiss_index"),
        description="Directory for vector store persistence",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # Generation settings
    generation_timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout for LLM generation in seconds",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise InvalidSettingError(
                message=f"Invalid log level: {v}",
                details=f"Valid levels: {', '.join(valid_levels)}",
            )
        return v.upper()

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        """Validate chunk_overlap is less than chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise InvalidSettingError(
                message="chunk_overlap must be less than chunk_size",
                details=f"Got chunk_overlap={self.chunk_overlap}, chunk_size={self.chunk_size}",
            )
        return self

    def get_index_version(self) -> str:
        """Generate a version string for the index based on settings.

        Returns:
            Version string combining model and chunking settings.
        """
        return f"{self.ollama_embedding_model}:{self.chunk_size}:{self.chunk_overlap}"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.vector_store_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Singleton Settings instance.
    """
    return Settings()
