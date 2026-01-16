"""Input validation utilities for user queries and file paths."""

import re
from pathlib import Path
from typing import List

from pdf_qa.core.exceptions import (
    EmptyQueryError,
    QueryTooLongError,
    DocumentError,
)

# Constants
MAX_QUERY_LENGTH = 2000
MIN_QUERY_LENGTH = 2
PDF_MAGIC_BYTES = b"%PDF"


def validate_query(query: str) -> str:
    """Validate and sanitize user query.

    Args:
        query: Raw user input query.

    Returns:
        Sanitized query string.

    Raises:
        EmptyQueryError: If query is empty or whitespace-only.
        QueryTooLongError: If query exceeds maximum length.
    """
    # Strip whitespace
    cleaned = query.strip()

    # Check for empty query
    if not cleaned or len(cleaned) < MIN_QUERY_LENGTH:
        raise EmptyQueryError()

    # Check length
    if len(cleaned) > MAX_QUERY_LENGTH:
        raise QueryTooLongError(len(cleaned), MAX_QUERY_LENGTH)

    # Basic sanitization - remove excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned


def validate_pdf_file(filepath: Path) -> bool:
    """Validate that a file is a valid PDF.

    Args:
        filepath: Path to the file to validate.

    Returns:
        True if file appears to be a valid PDF.

    Raises:
        DocumentError: If file doesn't exist or isn't a valid PDF.
    """
    if not filepath.exists():
        raise DocumentError(
            message=f"File not found: {filepath}",
            details="The specified file does not exist.",
        )

    if not filepath.is_file():
        raise DocumentError(
            message=f"Not a file: {filepath}",
            details="The path points to a directory, not a file.",
        )

    # Check file extension
    if filepath.suffix.lower() != ".pdf":
        raise DocumentError(
            message=f"Invalid file extension: {filepath}",
            details=f"Expected .pdf, got {filepath.suffix}",
        )

    # Check magic bytes
    try:
        with open(filepath, "rb") as f:
            header = f.read(8)
            if not header.startswith(PDF_MAGIC_BYTES):
                raise DocumentError(
                    message=f"Invalid PDF file: {filepath}",
                    details="File does not have valid PDF header.",
                )
    except PermissionError:
        raise DocumentError(
            message=f"Permission denied: {filepath}",
            details="Cannot read the file due to permission restrictions.",
        )
    except IOError as e:
        raise DocumentError(
            message=f"Cannot read file: {filepath}",
            details=str(e),
        )

    return True


def validate_directory(dirpath: Path, must_contain_pdfs: bool = False) -> List[Path]:
    """Validate a directory path and optionally check for PDFs.

    Args:
        dirpath: Path to the directory.
        must_contain_pdfs: If True, raises error if no PDFs found.

    Returns:
        List of PDF file paths found in the directory.

    Raises:
        DocumentError: If directory doesn't exist or is empty when expected.
    """
    if not dirpath.exists():
        raise DocumentError(
            message=f"Directory not found: {dirpath}",
            details="The specified directory does not exist.",
        )

    if not dirpath.is_dir():
        raise DocumentError(
            message=f"Not a directory: {dirpath}",
            details="The path points to a file, not a directory.",
        )

    # Find PDF files
    pdf_files = list(dirpath.glob("*.pdf"))

    if must_contain_pdfs and not pdf_files:
        raise DocumentError(
            message=f"No PDF files found in: {dirpath}",
            details="The directory does not contain any .pdf files.",
        )

    return pdf_files


def is_describe_query(query: str) -> bool:
    """Check if query is asking to describe/list documents.

    Args:
        query: User query string.

    Returns:
        True if query appears to be asking for document descriptions.
    """
    query_lower = query.lower().strip()

    # Patterns that indicate a "describe documents" query
    describe_patterns = [
        r"^describe\s+(the\s+)?documents?",
        r"^(list|show|what)\s+(are\s+)?(the\s+)?documents?",
        r"^what\s+(files?|papers?|pdfs?)\s+(do\s+)?(you\s+)?have",
        r"^(summarize|overview)\s+(of\s+)?(all\s+)?(the\s+)?documents?",
        r"^tell\s+me\s+about\s+(the\s+)?documents?",
        r"^what('s|\s+is)\s+in\s+(the\s+)?(documents?|papers?|corpus)",
    ]

    for pattern in describe_patterns:
        if re.match(pattern, query_lower):
            return True

    return False
