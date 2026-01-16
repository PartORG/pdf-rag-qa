"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from pdf_qa.cli.commands import app
from pdf_qa import __version__

runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_help_and_version(self) -> None:
        assert runner.invoke(app, ["--help"]).exit_code == 0
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0 and __version__ in result.stdout

    def test_init_help(self) -> None:
        assert runner.invoke(app, ["init", "--help"]).exit_code == 0

    def test_init_nonexistent_dir(self, temp_dir: Path) -> None:
        result = runner.invoke(app, ["init", "--dir", str(temp_dir / "nonexistent")])
        assert result.exit_code != 0

    @patch("pdf_qa.rag.store.create_vector_store_from_settings")
    def test_init_existing_index(self, mock_store: MagicMock, temp_dir: Path) -> None:
        mock_store.return_value.exists.return_value = True
        mock_store.return_value.get_unique_sources.return_value = ["doc.pdf"]
        result = runner.invoke(app, ["init", "--dir", str(temp_dir)])
        assert "already exists" in result.stdout.lower() or "force" in result.stdout.lower()

    @patch("pdf_qa.rag.store.create_vector_store_from_settings")
    def test_ask_without_index(self, mock_store: MagicMock) -> None:
        mock_store.return_value.exists.return_value = False
        result = runner.invoke(app, ["ask", "What is AI?"])
        assert result.exit_code != 0

    @patch("pdf_qa.rag.store.create_vector_store_from_settings")
    def test_chat_without_index(self, mock_store: MagicMock) -> None:
        mock_store.return_value.exists.return_value = False
        assert runner.invoke(app, ["chat"]).exit_code != 0

    def test_health_help(self) -> None:
        assert runner.invoke(app, ["health", "--help"]).exit_code == 0

    @patch("pdf_qa.rag.store.create_vector_store_from_settings")
    def test_clean_no_index(self, mock_store: MagicMock) -> None:
        mock_store.return_value.exists.return_value = False
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0

    @patch("pdf_qa.config.settings.get_settings")
    @patch("pdf_qa.rag.store.create_vector_store_from_settings")
    def test_clean_with_force(self, mock_store: MagicMock, mock_settings: MagicMock, temp_dir: Path) -> None:
        mock_settings.return_value.vector_store_path = temp_dir / "vectors"
        mock_store.return_value.exists.return_value = True
        mock_store.return_value.get_document_count.return_value = 50
        runner.invoke(app, ["clean", "--force"])
        mock_store.return_value.clear.assert_called_once()
