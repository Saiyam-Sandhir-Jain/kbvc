# kbvc/utils/display.py
"""
Rich terminal helpers for consistent KBVC CLI output.
"""

from __future__ import annotations

from typing import Any


def print_commit_summary(commit: Any, changed_kos: dict) -> None:
    """Print the commit output block shown after kbvc commit."""
    try:
        from rich.console import Console
        from rich.tree import Tree
        console = Console()
        tree = Tree(f"[bold green]Commit {commit.display_id}[/bold green]")
        for ko_id, change in changed_kos.items():
            tree.add(
                f"[cyan]{ko_id}@{change.from_version}[/cyan] → "
                f"[cyan]@{change.to_version}[/cyan]  "
                f"({len(change.chunks_reembedded)} re-embedded)"
            )
        console.print(tree)
        console.print(
            f"  Graph: [dim]{commit.graph_snapshot}[/dim]   "
            f"Prompt: [dim]{commit.prompt_snapshot}[/dim]   "
            f"Retrieval: [dim]{commit.retrieval_snapshot}[/dim]"
        )
    except ImportError:
        # Fallback without rich
        print(f"Commit {commit.display_id}")
        for ko_id, change in changed_kos.items():
            n = len(change.chunks_reembedded)
            print(f"  ├── {ko_id}@{change.from_version} → @{change.to_version}"
                  f"  ({n} re-embedded)")
        print(f"  Graph: {commit.graph_snapshot}   "
              f"Prompt: {commit.prompt_snapshot}   "
              f"Retrieval: {commit.retrieval_snapshot}")


def print_error(msg: str) -> None:
    try:
        from rich.console import Console
        Console(stderr=True).print(f"[bold red]Error:[/bold red] {msg}")
    except ImportError:
        print(f"Error: {msg}")


def print_warning(msg: str) -> None:
    try:
        from rich.console import Console
        Console().print(f"[bold yellow]⚠[/bold yellow]  {msg}")
    except ImportError:
        print(f"⚠  {msg}")


def print_success(msg: str) -> None:
    try:
        from rich.console import Console
        Console().print(f"[bold green]✓[/bold green]  {msg}")
    except ImportError:
        print(f"✓  {msg}")
