"""Custom exception hierarchy for PDF Q&A system.

Provides specific exceptions for different failure modes with user-friendly messages.
"""


class PDFQAError(Exception):
    """Base exception for all PDF Q&A errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


# Configuration Errors
class ConfigurationError(PDFQAError):
    """Base exception for configuration-related errors."""

    pass


class InvalidSettingError(ConfigurationError):
    """Raised when a configuration value is invalid."""

    pass


# Document Errors
class DocumentError(PDFQAError):
    """Base exception for document processing errors."""

    pass


class PDFLoadError(DocumentError):
    """Raised when a PDF file cannot be loaded."""

    def __init__(self, filepath: str, reason: str):
        super().__init__(
            message=f"Failed to load PDF: {filepath}",
            details=reason,
        )
        self.filepath = filepath


class EmptyDocumentError(DocumentError):
    """Raised when a document has no extractable text."""

    def __init__(self, filepath: str):
        super().__init__(
            message=f"No text content found in: {filepath}",
            details="The PDF may be scanned/image-only or empty.",
        )
        self.filepath = filepath


class PasswordProtectedError(DocumentError):
    """Raised when a PDF is password protected."""

    def __init__(self, filepath: str):
        super().__init__(
            message=f"Password protected PDF: {filepath}",
            details="Cannot process password-protected PDFs.",
        )
        self.filepath = filepath


# Index Errors (using different name to avoid shadowing builtin)
class VectorIndexError(PDFQAError):
    """Base exception for vector store index errors."""

    pass


class IndexNotFoundError(VectorIndexError):
    """Raised when the vector store index doesn't exist."""

    def __init__(self) -> None:
        super().__init__(
            message="No index found. Please run 'pdf-qa init' first.",
            details="The vector store has not been initialized.",
        )


class IndexCorruptedError(VectorIndexError):
    """Raised when the vector store index is corrupted."""

    def __init__(self, path: str):
        super().__init__(
            message="Index appears to be corrupted.",
            details=f"Try running 'pdf-qa clean' and then 'pdf-qa init'. Path: {path}",
        )
        self.path = path


class IndexVersionMismatchError(VectorIndexError):
    """Raised when index version doesn't match current settings."""

    def __init__(self, expected: str, found: str):
        super().__init__(
            message="Index version mismatch detected.",
            details=f"Expected: {expected}, Found: {found}. Run 'pdf-qa init --force' to rebuild.",
        )
        self.expected = expected
        self.found = found


# Embedding Errors
class EmbeddingError(PDFQAError):
    """Base exception for embedding-related errors."""

    pass


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when the embedding model is not available."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Embedding model not found: {model_name}",
            details=f"Run 'ollama pull {model_name}' to download it.",
        )
        self.model_name = model_name


class EmbeddingFailedError(EmbeddingError):
    """Raised when embedding generation fails."""

    def __init__(self, reason: str):
        super().__init__(
            message="Failed to generate embeddings.",
            details=reason,
        )


# LLM Errors
class LLMError(PDFQAError):
    """Base exception for LLM-related errors."""

    pass


class OllamaConnectionError(LLMError):
    """Raised when cannot connect to Ollama service."""

    def __init__(self, host: str):
        super().__init__(
            message=f"Cannot connect to Ollama at {host}",
            details="Ensure Ollama is running: 'ollama serve'",
        )
        self.host = host


class ModelNotAvailableError(LLMError):
    """Raised when the specified model is not available."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model not available: {model_name}",
            details=f"Run 'ollama pull {model_name}' to download it.",
        )
        self.model_name = model_name


class GenerationTimeoutError(LLMError):
    """Raised when LLM response times out."""

    def __init__(self, timeout_seconds: int):
        super().__init__(
            message=f"LLM response timed out after {timeout_seconds} seconds.",
            details="Try a smaller model or increase timeout.",
        )
        self.timeout_seconds = timeout_seconds


# Query Errors
class QueryError(PDFQAError):
    """Base exception for query-related errors."""

    pass


class EmptyQueryError(QueryError):
    """Raised when query is empty or whitespace-only."""

    def __init__(self) -> None:
        super().__init__(
            message="Query cannot be empty.",
            details="Please provide a valid question.",
        )


class QueryTooLongError(QueryError):
    """Raised when query exceeds maximum length."""

    def __init__(self, length: int, max_length: int):
        super().__init__(
            message=f"Query too long: {length} characters.",
            details=f"Maximum allowed: {max_length} characters.",
        )
        self.length = length
        self.max_length = max_length
