# PDF Q&A

CLI tool for asking questions about research papers. Uses RAG with Ollama.

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Limitations](#limitations)
- [License](#license)

## Features

### CLI Tool for Research Papers
PDF Q&A is a command-line interface (CLI) tool designed to help researchers and students quickly answer questions about research papers using the Retrieval-Augmented Generation (RAG) model with Ollama. The tool allows users to index PDFs, ask questions, and receive answers complete with source references.

### Modular Architecture
The project follows a modular architecture, separating concerns into distinct packages:
- **CLI**: Handles user input and output formatting.
- **Config**: Manages application settings using Pydantic.
- **Core**: Provides shared utilities like custom exceptions, validators, and logging setup.
- **Models**: Defines data classes for documents, chunks, answers, and search results.
- **RAG**: Implements the RAG pipeline components including PDF loading, text chunking, embeddings, FAISS store, and retriever.
- **Services**: Orchestrates the pipeline to handle business logic like Ollama LLM client and QA service.

### Edge Case Handling
The tool gracefully handles common failure scenarios:
- Skips empty or scanned PDFs with a warning.
- Provides clear error messages if Ollama is not running.
- Handles no results found without hallucinating.
- Special handling for "Describe documents" queries.
- Cleans up partial indices in case of interruptions.

## How It Works

1. **Initialization (`init`)**:
   - Loads PDFs from the `./research_papers` directory or a custom directory.
   - Splits text into chunks with 50-character overlap.
   - Generates embeddings for each chunk using the `mxbai-embed-large` model.
   - Stores the embeddings in an FAISS index.

2. **Asking Questions (`ask`)**:
   - Embeds the user's question.
   - Finds similar chunks from the FAISS index.
   - Uses the Ollama LLM to generate an answer, citing relevant sources.

3. **Interactive Chat (`chat`)**:
   - Similar to `ask`, but in a loop.
   - Keeps context between questions within the same session.

## Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.11+ | Target language for maximum library stability and performance. |
| Typer + Rich | CLI tool with type hints and rich output formatting. |
| Pydantic | Data validation and settings management using environment variables and `.env` files. |
| PyMuPDF (pymupdf) | Fast PDF processing, handles complex layouts. |
| Pillow | OCR support for extracting text from images in PDFs. |
| pytesseract | OCR support (requires Tesseract installed). |
| FAISS-cpu | Fast vector store for efficient similarity search. |
| Numpy | Numerical operations and array manipulations. |
| Ollama | Local, private LLM for generating answers without API costs. |
| python-dotenv | Loads environment variables from a `.env` file. |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/download)

## Installation

### With Poetry (recommended)
```bash
pip install poetry
poetry install
```

### With pip (standard Python environment)
```bash
python3.11 -m venv .venv  # On Windows: py -3.11 -m venv .venv
source venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Ollama Models
Pull the required models before first use:
```bash
ollama pull llama3.2           # LLM for generating answers
ollama pull mxbai-embed-large  # Embeddings for semantic search
```

Copy `.env.example` to `.env` to customize settings (optional).

## Configuration

Observed environment variables and configuration files include:
- `.env`: Customizable settings.

## Quick Start

1. **Initialize the index**:
   ```bash
   poetry run pdf-qa init
   ```

2. **Ask a question**:
   ```bash
   poetry run pdf-qa ask "What is the main contribution of this paper?"
   ```

3. **Start an interactive chat session**:
   ```bash
   poetry run pdf-qa chat
   ```

## Usage

Run the CLI using Poetry or directly if installed in a Python environment.

### With Poetry
```bash
poetry run pdf-qa <command>
```

### With Python environment
```bash
pdf-qa <command>
```

### Commands

All commands support `--help` for detailed options.

| Command | Description |
|---------|-------------|
| `init` | Index PDFs from `./research_papers` directory. Creates vector embeddings for search. |
| `init --dir <path>` | Index PDFs from a custom directory. |
| `init --force` | Re-index even if index already exists. |
| `ask "question"` | Ask a single question. Returns answer with source references. |
| `chat` | Interactive Q&A session. Type questions, get answers. Type `exit` to quit. |
| `health` | Check Ollama connection and index status. |
| `clean` | Delete the vector index. Use `--force` to skip confirmation. |

## Project Structure

```plaintext
.
├── .env.example
├── .gitignore
├── README.md
├── data/.gitkeep
├── pyproject.toml
├── research_papers/
│   ├── 2510.06042v1.pdf
│   ├── 2510.06534v1.pdf
│   ├── 2510.06664v1.pdf
│   ├── 2510.06911v1.pdf
│   ├── 2510.07043v1.pdf
│   ├── 2510.07423v1.pdf
│   ├── 2510.07593v1.pdf
│   ├── 2510.07614v1.pdf
│   ├── 2510.07733v1.pdf
│   ├── 2510.07772v1.pdf
│   ├── 2510.08149v1.pdf
│   ├── 2510.08255v1.pdf
│   ├── 2510.08383v1.pdf
│   ├── 2510.08529v1.pdf
│   └── 2510.08567v1.pdf
├── src/
│   └── pdf_qa/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── commands.py
│       │   └── console.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── exceptions.py
│       │   ├── logging.py
│       │   └── validators.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── document.py
│       │   └── qa.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── chunker.py
│       │   ├── embeddings.py
│       │   ├── loader.py
│       │   ├── retriever.py
│       │   └── store.py
│       └── services/
│           ├── __init__.py
│           ├── llm.py
│           ├── prompts.py
│           └── qa.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_chunker.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_models.py
    └── test_validators.py
```

## Development

The project follows best practices for dependency management and testing. The codebase is modular, making it easy to extend and maintain.

## Testing

Unit tests cover validators, chunking, models, config, and CLI commands. Tests use mocks for Ollama calls.

```bash
# With Poetry
poetry run pytest
poetry run pytest --cov=src/pdf_qa    # with coverage

# With Python environment (venv activated)
pytest
pytest --cov=src/pdf_qa               # with coverage
```

## Limitations

- The tool assumes that the Ollama models are available and correctly configured.
- It does not handle large PDFs or complex document structures beyond what PyMuPDF can process.
- There is no support for real-time updates to the index.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.