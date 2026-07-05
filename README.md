# PDF Q&A

CLI tool for asking questions about research papers. Uses RAG with Ollama to provide accurate and context-aware answers.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/PartORG/pdf-rag-qa)](LICENSE)
[![Tests](https://github.com/PartORG/pdf-rag-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/PartORG/pdf-rag-qa/actions/workflows/tests.yml)

## Introduction

PDF Q&A is a powerful command-line interface (CLI) tool designed to help researchers and students quickly find answers to questions about research papers. It leverages the Retrieval-Augmented Generation (RAG) model, combined with Ollama's large language model (LLM), to provide accurate and context-aware responses.

The tool indexes PDFs into a vector store, allowing for efficient semantic search. When you ask a question, it retrieves relevant chunks of text from the indexed documents and uses the LLM to generate an answer, citing the source references. This makes it an invaluable resource for anyone working with large volumes of research material.

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

### Indexing PDFs

PDF Q&A allows you to index research papers into a vector store. This enables efficient semantic search and retrieval of relevant chunks of text.

```bash
pdf-qa init
```

### Asking Questions

You can ask single questions or engage in an interactive chat session to get answers with source references.

```bash
pdf-qa ask "What is the main conclusion of the paper?"
```

### Interactive Chat

For a more conversational experience, use the `chat` command to interactively ask multiple questions and receive answers.

```bash
pdf-qa chat
```

### Health Check

Check the status of your Ollama connection and index with the `health` command.

```bash
pdf-qa health
```

### Cleaning Up

Delete the vector index when you're done or need to re-index.

```bash
pdf-qa clean --force
```

## How It Works

1. **Indexing**: PDFs are loaded, split into chunks, and embeddings are generated using `mxbai-embed-large`. These embeddings are stored in a FAISS index.
2. **Question Processing**: When you ask a question, the tool embeds it and finds similar chunks from the index. The LLM then generates an answer, citing the source references.

## Technology Stack

| Technology | Purpose |
|------------|---------|
| **FAISS** | Fast vector store for efficient semantic search. |
| **mxbai-embed-large** | Large embeddings model for semantic understanding. |
| **llama3.2 via Ollama** | Local LLM for generating answers without API costs. |
| **Typer + Rich** | CLI tool with type hints and rich output formatting. |
| **Pydantic** | Data validation and settings management. |
| **PyMuPDF (pymupdf)** | PDF parsing library for handling complex layouts. |
| **Pillow** | OCR support for extracting text from images in PDFs. |
| **pytesseract** | OCR tool for recognizing text in images. |

## Requirements

- Python 3.10+
- Ollama (https://ollama.ai/download)

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

- `.env`: Customizable settings (optional)

## Quick Start

1. **Install the tool**:
    ```bash
    poetry install
    ```

2. **Index PDFs**:
    ```bash
    pdf-qa init
    ```

3. **Ask a question**:
    ```bash
    pdf-qa ask "What is the main conclusion of the paper?"
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
│   └── ...
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
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_models.py
│   └── test_validators.py
```

## Development

The codebase follows a modular architecture separating concerns into distinct packages:

- **cli/**: Command-line interface using Typer. Handles user input, output formatting with Rich.
- **config/**: Application settings via Pydantic. Loads from environment variables and `.env` file.
- **core/**: Shared utilities: custom exceptions, input validators, logging setup.
- **models/**: Data classes: Document, Chunk, Answer, SearchResult.
- **rag/**: RAG pipeline components: PDF loader, text chunker, embeddings, FAISS store, retriever.
- **services/**: Business logic: Ollama LLM client, QA service that orchestrates the pipeline.

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

- **Empty/scanned PDFs**: Skipped with warning.
- **Ollama not running**: Clear error message.
- **No results found**: Says so instead of hallucinating.
- **"Describe documents" queries**: Detected and handled specially (lists all docs).
- **Interrupted indexing**: Partial index is cleaned up.

## License

This project is licensed under the [MIT License](LICENSE).