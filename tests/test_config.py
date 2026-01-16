"""Tests for configuration."""

import pytest
from pathlib import Path
from pdf_qa.config.settings import Settings
from pdf_qa.core.exceptions import InvalidSettingError


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_values(self) -> None:
        settings = Settings()
        assert settings.ollama_model == "llama3.2"
        assert settings.ollama_embedding_model == "mxbai-embed-large"
        assert settings.chunk_size == 512

    def test_validation(self) -> None:
        # Log level validation
        assert Settings(log_level="DEBUG").log_level == "DEBUG"
        assert Settings(log_level="warning").log_level == "WARNING"
        with pytest.raises(InvalidSettingError):
            Settings(log_level="INVALID")

        # Chunk overlap < chunk size
        with pytest.raises(InvalidSettingError):
            Settings(chunk_size=100, chunk_overlap=100)

        # Bounds
        with pytest.raises(Exception):
            Settings(chunk_size=50)
        with pytest.raises(Exception):
            Settings(top_k_results=0)

    def test_index_version(self) -> None:
        settings = Settings(ollama_embedding_model="test", chunk_size=256, chunk_overlap=32)
        version = settings.get_index_version()
        assert "test" in version and "256" in version

    def test_ensure_directories(self, temp_dir: Path) -> None:
        settings = Settings(vector_store_path=temp_dir / "new" / "vectors")
        assert not settings.vector_store_path.exists()
        settings.ensure_directories()
        assert settings.vector_store_path.exists()

    def test_path_conversion(self) -> None:
        settings = Settings(documents_path="./docs", vector_store_path="./vectors")
        assert isinstance(settings.documents_path, Path)
