# kbvc/commands/contradict.py
"""
kbvc contradict — Contradiction detection and resolution flow.

Phase 9 feature.

Scans `contradicts` relations in the knowledge graph and reports on them.
For each contradicting pair, shows:
  - Both KO IDs and their most recent versions
  - The relation note (if any)
  - Whether either KO has a `supersedes` relation (indicating resolution)
  - Resolution options

Usage:
    kbvc contradict list              # list all contradictions
    kbvc contradict resolve <rel_id>  # mark a contradiction as resolved
                                      # by superseding one KO with another
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


@dataclass
class ContradictionEntry:
    rel_id: str
    from_ko: str
    to_ko: str
    note: str
    confidence: float
    source: str
    resolved: bool = False          # True if a supersedes relation exists
    resolution_rel_id: Optional[str] = None


def detect_contradictions(repo: "KbvcRepo") -> List[ContradictionEntry]:
    """Return all `contradicts` relations, annotated with resolution status."""
    from kbvc.core.graph import RelationGraph

    graph = RelationGraph(repo.graph_dir, repo.current_branch())
    results = []

    # Collect all supersedes relations for resolution check
    supersedes_pairs: set[tuple] = set()
    for rel in graph.relations:
        if rel.type == "supersedes":
            supersedes_pairs.add((rel.from_id, rel.to_id))

    for rel in graph.relations:
        if rel.type != "contradicts":
            continue
        # A contradiction is "resolved" if either KO supersedes the other
        resolved = (
            (rel.from_id, rel.to_id) in supersedes_pairs
            or (rel.to_id, rel.from_id) in supersedes_pairs
        )
        # Find resolution rel_id if resolved
        resolution_rel_id = None
        if resolved:
            for s in graph.relations:
                if s.type == "supersedes" and s.from_id in (rel.from_id, rel.to_id):
                    resolution_rel_id = s.id
                    break

        results.append(ContradictionEntry(
            rel_id=rel.id,
            from_ko=rel.from_id,
            to_ko=rel.to_id,
            note=rel.note or "",
            confidence=rel.confidence,
            source=rel.source,
            resolved=resolved,
            resolution_rel_id=resolution_rel_id,
        ))

    return results


def run_contradict_list(repo: "KbvcRepo") -> None:
    import click
    from kbvc.core.ko import KOStore

    entries = detect_contradictions(repo)
    store = KOStore(repo.ko_store_path)

    if not entries:
        click.echo("✓ No contradictions found in the knowledge graph.")
        return

    unresolved = [e for e in entries if not e.resolved]
    resolved = [e for e in entries if e.resolved]

    click.echo(f"Contradictions: {len(unresolved)} unresolved  /  {len(resolved)} resolved")
    click.echo("")

    if unresolved:
        click.echo("UNRESOLVED:")
        for e in unresolved:
            from_ko = store.get(e.from_ko)
            to_ko = store.get(e.to_ko)
            from_ver = f"v{from_ko.version}" if from_ko else "?"
            to_ver = f"v{to_ko.version}" if to_ko else "?"
            click.echo(f"  {e.rel_id[:8]}  {e.from_ko} ({from_ver}) ←contradicts→ {e.to_ko} ({to_ver})")
            if e.note:
                click.echo(f"            note: {e.note}")
            click.echo(f"            confidence: {e.confidence:.2f}  source: {e.source}")
            click.echo("")
        click.echo("To resolve:")
        click.echo("  kbvc link <winner.md> <loser.md> --type supersedes")
        click.echo("  kbvc contradict resolve <rel_id>")

    if resolved:
        click.echo("RESOLVED:")
        for e in resolved:
            click.echo(f"  {e.rel_id[:8]}  {e.from_ko} ←contradicts→ {e.to_ko}  [superseded by {e.resolution_rel_id or '?'}]")


def run_contradict_resolve(repo: "KbvcRepo", rel_id: str) -> None:
    """
    Mark a contradiction as resolved by recording who supersedes whom.

    Requires that a `supersedes` relation already exists between the two KOs.
    This command verifies the pair is consistent and echoes the resolution.
    """
    import click
    from kbvc.core.graph import RelationGraph

    graph = RelationGraph(repo.graph_dir, repo.current_branch())

    # Find the contradicts relation
    target = None
    for rel in graph.relations:
        if rel.type == "contradicts" and (rel.id == rel_id or rel.id.startswith(rel_id)):
            target = rel
            break

    if target is None:
        raise click.ClickException(f"No 'contradicts' relation found with ID '{rel_id}'.")

    # Check supersedes
    winner = None
    for rel in graph.relations:
        if rel.type == "supersedes" and rel.from_id in (target.from_id, target.to_id):
            winner = rel.from_id
            loser = rel.to_id
            break

    if winner is None:
        raise click.ClickException(
            f"Cannot resolve: no 'supersedes' relation found for "
            f"'{target.from_id}' or '{target.to_id}'.\n"
            f"Run: kbvc link <winner.md> <loser.md> --type supersedes"
        )

    click.echo(f"✓ Contradiction {target.id[:8]} is resolved.")
    click.echo(f"  {winner} supersedes {loser}.")
    click.echo(f"  The contradicts relation remains in the graph as a historical record.")
    click.echo("")
    click.echo(f"Consider annotating {loser} with --reason \"superseded by {winner}\":")
    click.echo(f"  kbvc annotate knowledge/{loser}.md --reason \"superseded by {winner}\"")
