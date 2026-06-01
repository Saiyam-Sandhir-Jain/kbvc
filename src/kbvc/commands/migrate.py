# kbvc/commands/migrate.py
"""
kbvc migrate — VSAL backend-to-backend migration.

Reads all ChunkRecords from the source backend, normalises them through
the universal ChunkRecord schema, and writes them to the target backend.

The key point: KBVC never migrates raw backend objects.  It always goes:

    Source Backend
          ↓  export_chunks()
    List[ChunkRecord]   ← the universal KBVC schema
          ↓  import_chunks()
    Target Backend

This means the source and target backends never need to know about each other.

Usage:
    kbvc migrate --from qdrant --to pgvector
    kbvc migrate --from qdrant --to pgvector --dry-run
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


def run_migrate(
    repo: "KbvcRepo",
    from_backend: str,
    to_backend: str,
    dry_run: bool = False,
) -> None:
    """
    Migrate all ChunkRecords from one vector backend to another.

    Parameters
    ----------
    repo        : KbvcRepo instance (provides config + collection name).
    from_backend: Backend name to read from (e.g. "qdrant", "pgvector").
    to_backend  : Backend name to write to.
    dry_run     : If True, report what would happen without writing.
    """
    import click
    from kbvc.backends.vectordb import ChunkRecord  # noqa: F401 (ensure import)
    from kbvc.commands.migrate_helpers import load_backend

    cfg = repo.config()
    collection = cfg.get("vectordb.collection", "kbvc")

    click.echo(f"Migration: {from_backend} → {to_backend}")
    click.echo(f"Collection: {collection}")
    click.echo("")

    # ── load source backend ──────────────────────────────────────────────────
    click.echo(f"[1/4] Connecting to source backend: {from_backend}")
    src = load_backend(from_backend, cfg)

    # ── export ───────────────────────────────────────────────────────────────
    click.echo("[2/4] Exporting ChunkRecords from source…")
    try:
        chunks = src.export_chunks(collection)
    except NotImplementedError as exc:
        raise click.ClickException(str(exc))

    click.echo(f"      Exported {len(chunks)} chunk(s)")
    if not chunks:
        click.echo("Nothing to migrate.")
        return

    if dry_run:
        click.echo("")
        click.echo("Dry-run mode — no data written.")
        click.echo(f"Would migrate {len(chunks)} chunk(s) to {to_backend}.")
        _print_sample(chunks[:5])
        return

    # ── load target backend ──────────────────────────────────────────────────
    click.echo(f"[3/4] Connecting to target backend: {to_backend}")
    dst = load_backend(to_backend, cfg)

    # ── import ───────────────────────────────────────────────────────────────
    click.echo("[4/4] Importing ChunkRecords into target…")
    dims = len(chunks[0].embedding)
    dst.initialize_schema(collection, dims)
    dst.import_chunks(collection, chunks)

    click.echo("")
    click.echo(f"✓ Migrated {len(chunks)} chunk(s) from {from_backend} to {to_backend}.")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  kbvc config set vectordb.backend {to_backend}")
    click.echo("  kbvc backend info")


def _print_sample(chunks) -> None:
    """Print a sample of exported chunks for dry-run preview."""
    import click
    click.echo("")
    click.echo("Sample (first 5 chunks):")
    for c in chunks:
        click.echo(f"  {c.vector_id}  ko={c.ko_id}  ver={c.ko_version}  idx={c.chunk_index}")
