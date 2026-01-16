"""CLI commands for PDF Q&A system."""

from pathlib import Path
from typing import Optional

import typer

from pdf_qa import __version__
from pdf_qa.cli.console import (
    console,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_answer,
    print_health_status,
    print_welcome,
    prompt_question,
    print_header,
    get_progress,
)
from pdf_qa.core.exceptions import PDFQAError, OllamaConnectionError, EmptyQueryError
from pdf_qa.core.logging import setup_logging, get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="pdf-qa",
    help="Q&A system for research papers using RAG and Ollama.",
    add_completion=False,
    no_args_is_help=True,
)

# Simple shutdown flag
_shutdown_requested = False


def _check_shutdown():
    """Raise KeyboardInterrupt if shutdown was requested."""
    if _shutdown_requested:
        raise KeyboardInterrupt()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"pdf-qa version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """PDF Q&A - Ask questions about your research papers."""
    pass


@app.command()
def init(
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to documents directory.",
        exists=True, file_okay=False, dir_okay=True, resolve_path=True,
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing."),
) -> None:
    """Index PDF documents for Q&A."""
    from pdf_qa.config.settings import get_settings
    from pdf_qa.rag.loader import DocumentLoader
    from pdf_qa.rag.chunker import create_chunker_from_settings
    from pdf_qa.rag.embeddings import create_embedding_service_from_settings
    from pdf_qa.rag.store import create_vector_store_from_settings

    settings = get_settings()
    setup_logging(settings.log_level)
    docs_path = directory or settings.documents_path

    console.print()
    print_header("Initializing PDF Q&A System")

    vector_store = create_vector_store_from_settings()
    if vector_store.exists() and not force:
        print_warning("Index already exists. Use --force to re-index.")
        print_info(f"Currently indexed: {len(vector_store.get_unique_sources())} documents")
        raise typer.Exit(0)

    if force and vector_store.exists():
        print_info("Clearing existing index...")
        vector_store.clear()

    try:
        loader = DocumentLoader()
        chunker = create_chunker_from_settings()
        embedding_service = create_embedding_service_from_settings()

        pdf_files = list(docs_path.glob("*.pdf"))
        if not pdf_files:
            print_error(f"No PDF files found in {docs_path}")
            raise typer.Exit(1)

        print_info(f"Found {len(pdf_files)} PDF files")
        total_chunks = 0

        with get_progress() as progress:
            task = progress.add_task("Processing...", total=len(pdf_files))

            for pdf_path in pdf_files:
                progress.update(task, description=f"Processing: {pdf_path.name}")

                try:
                    doc = loader.load_pdf(pdf_path)
                    if not doc or not doc.has_content:
                        progress.update(task, advance=1)
                        continue

                    chunks = chunker.chunk_document(doc)
                    if not chunks:
                        progress.update(task, advance=1)
                        continue

                    embeddings = embedding_service.embed_texts([c.content for c in chunks])
                    vector_store.add_documents(chunks, embeddings)
                    total_chunks += len(chunks)

                except Exception as e:
                    print_warning(f"Error processing {pdf_path.name}: {e}")

                progress.update(task, advance=1)

        vector_store.save()
        vector_store.save_metadata(
            index_version=settings.get_index_version(),
            settings_hash=str(hash(str(settings))),
        )

        console.print()
        print_success("Indexing complete!")
        print_info(f"Indexed {len(pdf_files)} files | {total_chunks} chunks")

    except KeyboardInterrupt:
        console.print()
        print_warning("Indexing interrupted.")
        vector_store.clear()
        raise typer.Exit(130)
    except OllamaConnectionError as e:
        print_error(e.message, e.details)
        raise typer.Exit(1)
    except PDFQAError as e:
        print_error(e.message, e.details)
        raise typer.Exit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details."),
) -> None:
    """Ask a question about indexed documents."""
    from pdf_qa.config.settings import get_settings
    from pdf_qa.services.qa import create_qa_service_from_settings
    from pdf_qa.rag.store import create_vector_store_from_settings

    settings = get_settings()
    setup_logging(settings.log_level if verbose else "WARNING")

    vector_store = create_vector_store_from_settings()
    if not vector_store.exists():
        print_error("No index found. Run 'pdf-qa init' first.")
        raise typer.Exit(1)

    try:
        console.print()
        with console.status("[cyan]Thinking...[/cyan]"):
            qa_service = create_qa_service_from_settings()
            answer = qa_service.ask(question)
        print_answer(answer)
        console.print()

    except EmptyQueryError as e:
        print_error(e.message)
        raise typer.Exit(1)
    except OllamaConnectionError as e:
        print_error(e.message, e.details)
        raise typer.Exit(1)
    except PDFQAError as e:
        print_error(e.message, e.details)
        raise typer.Exit(1)


@app.command()
def chat() -> None:
    """Start interactive Q&A session."""
    from pdf_qa.config.settings import get_settings
    from pdf_qa.services.qa import create_qa_service_from_settings
    from pdf_qa.rag.store import create_vector_store_from_settings

    get_settings()
    setup_logging("WARNING")

    vector_store = create_vector_store_from_settings()
    if not vector_store.exists():
        print_error("No index found. Run 'pdf-qa init' first.")
        raise typer.Exit(1)

    print_welcome()

    try:
        qa_service = create_qa_service_from_settings()

        while True:
            try:
                question = prompt_question().strip()

                if question.lower() in ("quit", "exit", "q"):
                    print_info("Goodbye!")
                    break

                if not question:
                    continue

                with console.status("[cyan]Thinking...[/cyan]"):
                    answer = qa_service.ask(question)
                print_answer(answer)
                console.print()

            except EmptyQueryError:
                print_warning("Please enter a valid question.")
            except KeyboardInterrupt:
                console.print()
                print_info("Goodbye!")
                break

    except OllamaConnectionError as e:
        print_error(e.message, e.details)
        raise typer.Exit(1)
    except PDFQAError as e:
        print_error(e.message, e.details)
        raise typer.Exit(1)


@app.command()
def health() -> None:
    """Check system health."""
    from pdf_qa.config.settings import get_settings
    from pdf_qa.rag.embeddings import create_embedding_service_from_settings
    from pdf_qa.rag.store import create_vector_store_from_settings
    from pdf_qa.services.llm import create_llm_from_settings

    settings = get_settings()
    all_healthy = True

    console.print()
    print_header("System Health Check")
    console.print()

    # Ollama
    console.print("[bold]Ollama Service[/bold]")
    try:
        llm = create_llm_from_settings()
        models = llm.list_models()
        print_health_status("Connection", bool(models), f"{len(models)} models")
    except Exception as e:
        print_health_status("Connection", False, str(e))
        all_healthy = False

    try:
        llm = create_llm_from_settings()
        print_health_status("LLM Model", llm.is_available(), settings.ollama_model)
    except Exception as e:
        print_health_status("LLM Model", False, str(e))
        all_healthy = False

    try:
        embed = create_embedding_service_from_settings()
        print_health_status("Embedding Model", embed.is_available(), settings.ollama_embedding_model)
    except Exception as e:
        print_health_status("Embedding Model", False, str(e))
        all_healthy = False

    console.print()

    # Documents
    console.print("[bold]Documents[/bold]")
    docs_path = settings.documents_path
    if docs_path.exists():
        pdf_count = len(list(docs_path.glob("*.pdf")))
        print_health_status("Directory", True, str(docs_path))
        print_health_status("PDF Files", pdf_count > 0, f"{pdf_count} files")
        if not pdf_count:
            all_healthy = False
    else:
        print_health_status("Directory", False, "Not found")
        all_healthy = False

    console.print()

    # Index
    console.print("[bold]Index[/bold]")
    try:
        vector_store = create_vector_store_from_settings()
        if vector_store.exists():
            print_health_status("Status", True, f"{vector_store.get_document_count()} chunks")
        else:
            print_health_status("Status", False, "Run 'pdf-qa init'")
            all_healthy = False
    except Exception as e:
        print_health_status("Status", False, str(e))
        all_healthy = False

    console.print()
    if all_healthy:
        print_success("All systems operational!")
    else:
        print_warning("Some issues detected.")

    raise typer.Exit(0 if all_healthy else 1)


@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Remove indexed data."""
    from pdf_qa.config.settings import get_settings
    from pdf_qa.rag.store import create_vector_store_from_settings
    import shutil

    settings = get_settings()

    console.print()
    print_header("Clean Index Data")

    vector_store = create_vector_store_from_settings()

    if not vector_store.exists():
        print_info("No index found. Nothing to clean.")
        raise typer.Exit(0)

    doc_count = vector_store.get_document_count()
    console.print()
    console.print(f"This will delete {doc_count} indexed chunks.")
    console.print()

    if not force:
        if not typer.confirm("Continue?"):
            print_info("Cancelled.")
            raise typer.Exit(0)

    try:
        vector_store.clear()
        if settings.vector_store_path.exists():
            shutil.rmtree(settings.vector_store_path)
        print_success("Index cleaned.")
    except Exception as e:
        print_error(f"Failed: {e}")
        raise typer.Exit(1)
