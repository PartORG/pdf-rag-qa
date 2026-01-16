# PDF Q&A

CLI tool for asking questions about research papers. Uses RAG with Ollama.

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.ai/download).

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

### Ollama models

Pull the required models before first use:

```bash
ollama pull llama3.2           # LLM for generating answers
ollama pull mxbai-embed-large  # Embeddings for semantic search
```

Copy `.env.example` to `.env` to customize settings (optional).

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

## How It Works

Make sure Ollama is running (`ollama serve`) before using the tool.

1. **init**: Loads PDFs → splits into chunks → generates embeddings → stores in FAISS index
2. **ask**: Embeds your question → finds similar chunks → LLM generates answer citing sources
3. **chat**: Same as `ask` but in a loop. Keeps context between questions in the same session.

## Design Decisions

Key technology choices and the reasoning behind them:

| Component | Choice | Why |
|-----------|--------|-----|
| Vector store | FAISS | Fast, simple, no external server needed |
| Embeddings | mxbai-embed-large | Good semantic understanding, runs locally |
| LLM | llama3.2 via Ollama | Local, private, no API costs |
| CLI | Typer + Rich | Type hints → CLI args, nice output |
| Config | Pydantic | Validation, .env support |
| PDF parsing | PyMuPDF | Fast, handles complex layouts |

**Chunking**: 512 chars with 50 char overlap. Breaks at sentence boundaries when possible.

## Edge Cases

The tool handles common failure scenarios gracefully:

- **Empty/scanned PDFs**: Skipped with warning
- **Ollama not running**: Clear error message
- **No results found**: Says so instead of hallucinating
- **"Describe documents" queries**: Detected and handled specially (lists all docs)
- **Interrupted indexing**: Partial index is cleaned up

## Structure

The codebase follows a modular architecture separating concerns into distinct packages:

| Package | Purpose |
|---------|---------|
| `cli/` | Command-line interface using Typer. Handles user input, output formatting with Rich. |
| `config/` | Application settings via Pydantic. Loads from environment variables and `.env` file. |
| `core/` | Shared utilities: custom exceptions, input validators, logging setup. |
| `models/` | Data classes: Document, Chunk, Answer, SearchResult. |
| `rag/` | RAG pipeline components: PDF loader, text chunker, embeddings, FAISS store, retriever. |
| `services/` | Business logic: Ollama LLM client, QA service that orchestrates the pipeline. |

## Tests

Unit tests cover validators, chunking, models, config, and CLI commands. Tests use mocks for Ollama calls.

```bash
# With Poetry
poetry run pytest
poetry run pytest --cov=src/pdf_qa    # with coverage

# With Python environment (venv activated)
pytest
pytest --cov=src/pdf_qa               # with coverage
```
