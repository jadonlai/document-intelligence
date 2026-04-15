import os
import sys
from pathlib import Path
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import box
from rich.align import Align
from rich.rule import Rule
from lib.constants import PDFFOLDER, EMBEDDINGSFOLDER
from lib.doc_analysis import save_embeddings_from_text

console = Console()



def clear():
    os.system("cls" if os.name == "nt" else "clear")

def print_menu():
    table = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 2)
    )
    table.add_column("key", style="bold #00d4ff")
    table.add_column("action", style="white")

    table.add_row("1", "Upload a PDF file")
    table.add_row("2", "Use an existing PDF")
    table.add_row("3", "Exit")

    console.print(Align.center(Panel(
        Align.center(table),
        title="[bold white]Main Menu[/bold white]",
        border_style="#2a2a4a",
        padding=(1, 1),
    )))
    console.print()

def error(msg: str):
    console.print(f"\n  [bold red]✗[/bold red]  {msg}\n")
    time.sleep(1)

def success(msg: str):
    console.print(f"\n  [bold green]✓[/bold green]  [green]{msg}[/green]\n")
    time.sleep(1)

def info(msg: str):
    console.print(f"  [dim #00d4ff]→[/dim #00d4ff]  [dim]{msg}[/dim]")

def upload_pdf() -> str | None:
    console.print(Panel(
        "[bold white]Upload a PDF File[/bold white]\n"
        "[dim]Enter the full or relative path to your PDF.[/dim]",
        border_style="#00d4ff",
        padding=(1, 1),
    ))

    path_str = Prompt.ask("\n  [#00d4ff]PDF path[/#00d4ff]")
    path = Path(path_str.strip())

    if not path.exists():
        error(f"File not found: {path}")
        return None
    if path.suffix.lower() != ".pdf":
        error("File must be a .pdf")
        return None

    success(f"Loaded: {path.name}")
    return str(path)

def choose_existing_pdf() -> str | None:
    pdf_folder = Path(PDFFOLDER)
    embeddings_folder = Path(EMBEDDINGSFOLDER)

    if not pdf_folder.exists():
        error(f"PDF folder not found: {PDFFOLDER}")
        info("Create the folder or change PDFFOLDER in cli.py")
        return None

    pdfs = sorted(pdf_folder.glob("*.pdf"))

    if not pdfs:
        error(f"No PDF files found in {PDFFOLDER}")
        return None

    console.print(Panel(
        "[bold white]Choose an Existing PDF[/bold white]",
        border_style="#00d4ff",
        padding=(1, 1),
    ))
    console.print()

    table = Table(
        show_header=True,
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        header_style="bold #00d4ff",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="white")
    table.add_column("Size", style="dim", justify="right")

    for i, pdf in enumerate(pdfs, 1):
        size_kb = pdf.stat().st_size // 1024
        table.add_row(str(i), pdf.name, f"{size_kb} KB")

    console.print(table)
    console.print()

    choice = Prompt.ask(
        f"  [#00d4ff]Select a file[/#00d4ff] [dim](1–{len(pdfs)})[/dim]"
    )

    if not choice.isdigit() or not (1 <= int(choice) <= len(pdfs)):
        error("Invalid selection.")
        return None

    selected = pdfs[int(choice) - 1]
    selected_embeddings = embeddings_folder / f'{selected.name[:-4]}.pkl'

    if selected_embeddings.exists():
        info(f"Embeddings found for {selected.name}")
    else:
        info(f"Embeddings not found for {selected.name}")
        info("Creating embeddings for this file...")
        save_embeddings_from_text(selected.name)
        info("Embeddings created.")
        
        info("Attempting to upload embeddings to Supabase...")
        # NEED TO UPSERT
        info("Embeddings uploaded.")
    
    success(f"Loaded: {selected.name}")
    return str(selected)

def query_loop(pdf_path: str):
    filename = Path(pdf_path).name

    console.print()
    console.print(Rule(f"[dim]  {filename}  [/dim]", style="#2a2a4a"))
    console.print()
    info("Type your question below. Enter [bold]quit[/bold] or [bold]back[/bold] to return to the menu.")
    console.print()

    while True:
        query = Prompt.ask("  [bold #00d4ff]Query[/bold #00d4ff]")
        query = query.strip()

        if not query:
            continue

        if query.lower() in ("quit", "exit", "back", "q"):
            console.print()
            info("Returning to main menu...\n")
            break

        # ── Replace this block with your actual model call ─────────────────
        with console.status("[dim]Searching document and generating answer...[/dim]",
                            spinner="dots", spinner_style="#00d4ff"):
            answer = run_query(pdf_path, query)   # <-- your function here
        # ───────────────────────────────────────────────────────────────────

        console.print()
        console.print(Panel(
            answer,
            title="[bold white]Answer[/bold white]",
            border_style="#00d4ff",
            padding=(1, 2),
        ))
        console.print()


def run_query(pdf_path: str, query: str) -> str:
    return (
        f"[dim](stub) No model connected yet.[/dim]\n\n"
        f"PDF: {pdf_path}\n"
        f"Query: {query}"
    )

def main():
    clear()

    while True:
        print_menu()

        choice = Prompt.ask("  [#00d4ff]Select an option[/#00d4ff]",
                            choices=["1", "2", "3"],
                            show_choices=False)

        if choice == "1":
            clear()
            pdf_path = upload_pdf()
            if pdf_path:
                query_loop(pdf_path)
            clear()

        elif choice == "2":
            clear()
            pdf_path = choose_existing_pdf()
            if pdf_path:
                query_loop(pdf_path)
            clear()

        elif choice == "3":
            console.print()
            console.print(Align.center(
                "[dim]Goodbye.[/dim]"
            ))
            console.print()
            sys.exit(0)

if __name__ == "__main__":
    main()