"""Console output utilities using Rich."""

from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.theme import Theme

from pdf_qa.models.qa import Answer, Source

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
})

console = Console(theme=custom_theme)


def print_answer(answer: Answer) -> None:
    """Display formatted answer with sources."""
    console.print()
    console.print(Panel(
        Markdown(answer.answer_text),
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    if answer.confidence:
        color = {"high": "green", "medium": "yellow", "low": "red"}.get(answer.confidence, "white")
        console.print(f"  Confidence: [{color}]{answer.confidence}[/{color}]")

    if answer.sources:
        console.print()
        _print_sources_table(answer.sources)


def _print_sources_table(sources: List[Source]) -> None:
    table = Table(title="Sources", show_header=True, header_style="bold cyan")
    table.add_column("File", style="cyan")
    table.add_column("Page", justify="center")
    table.add_column("Relevance", justify="center")

    for source in sources:
        score = source.relevance_score
        style = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
        table.add_row(source.filename, str(source.page_number), f"[{style}]{score:.0%}[/{style}]")

    console.print(table)


def print_success(msg: str) -> None:
    console.print(f"[success]+[/success] {msg}")


def print_error(msg: str, details: Optional[str] = None) -> None:
    console.print(f"[error]x Error:[/error] {msg}")
    if details:
        console.print(f"  [dim]{details}[/dim]")


def print_warning(msg: str) -> None:
    console.print(f"[warning]![/warning] {msg}")


def print_info(msg: str) -> None:
    console.print(f"[info]*[/info] {msg}")


def get_progress() -> Progress:
    """Get a progress bar for long operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )


def print_health_status(component: str, status: bool, details: Optional[str] = None) -> None:
    icon = "[green]+[/green]" if status else "[red]x[/red]"
    status_text = "[green]OK[/green]" if status else "[red]FAIL[/red]"
    if details:
        console.print(f"  {icon} {component}: {status_text} - {details}")
    else:
        console.print(f"  {icon} {component}: {status_text}")


def print_welcome() -> None:
    console.print()
    console.print(Panel(
        "[bold]Interactive Q&A Chat[/bold]\n\n"
        "Ask questions about the indexed research papers.\n"
        "Type [cyan]quit[/cyan] or [cyan]exit[/cyan] to end the session.",
        title="PDF Q&A",
        border_style="cyan",
    ))
    console.print()


def prompt_question() -> str:
    return console.input("[bold cyan]Question:[/bold cyan] ")


def print_header(text: str) -> None:
    console.print()
    console.print(f"[bold]{text}[/bold]")
    console.print("-" * len(text))
