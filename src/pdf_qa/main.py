"""Main entry point for the PDF Q&A CLI application.

This module initializes and runs the Typer CLI application.
It can be invoked directly or via the `pdf-qa` command after installation.

Usage:
    # Via poetry
    poetry run pdf-qa --help

    # Direct execution
    python -m pdf_qa --help

    # After installation
    pdf-qa --help
"""

from pdf_qa.cli.commands import app


def main() -> None:
    """Run the PDF Q&A CLI application."""
    app()


if __name__ == "__main__":
    main()
