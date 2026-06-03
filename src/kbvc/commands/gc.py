# kbvc/commands/gc.py
"""
kbvc gc — Garbage-collect orphaned vectors and stale snapshots.

Phase 7 feature:

  1. Scan all vector IDs tracked in ko_store.json  →  "live" set
  2. Export all vector IDs present in the configured local vector DB
  3. IDs in DB but NOT in live set  →  orphaned; delete (unless --dry-run)
  4. Optionally prune commit snapshot files that are no longer reachable
     from HEAD (--snapshots flag).

Usage:
    kbvc gc              # remove orphaned vectors
    kbvc gc --dry-run    # preview what would be deleted
    kbvc gc --snapshots  # also prune unreachable graph/prompt/retrieval snapshots
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


def run_gc(repo: "KbvcRepo", dry_run: bool = False, prune_snapshots: bool = False) -> None:
    import click
    from kbvc.core.ko import KOStore
    from kbvc.backends import get_vectordb_backend

    cfg = repo.config()
    collection = cfg.get("vectordb.collection", "kbvc")

    # ── collect live vector IDs ──────────────────────────────────────────────
    store = KOStore(repo.ko_store_path)
    live_ids: set[str] = set()
    for ko in store.all():
        for vid in ko.vector_ids:
            live_ids.add(vid)

    click.echo(f"Live vector IDs in KO store: {len(live_ids)}")

    # ── connect to vector DB ─────────────────────────────────────────────────
    try:
        vdb = get_vectordb_backend(cfg)
    except Exception as exc:
        raise click.ClickException(f"Could not load vector backend: {exc}")

    # ── export all chunk IDs from DB ─────────────────────────────────────────
    try:
        chunks = vdb.export_chunks(collection)
    except NotImplementedError:
        raise click.ClickException(
            "The configured backend does not support export_chunks(). "
            "GC is only available for backends that implement full export "
            "(Qdrant, pgvector)."
        )
    except Exception as exc:
        raise click.ClickException(f"Could not export chunks from backend: {exc}")

    db_ids = {c.vector_id for c in chunks}
    click.echo(f"Vector IDs in backend:        {len(db_ids)}")

    orphaned = db_ids - live_ids

    if not orphaned:
        click.echo("\n✓ No orphaned vectors found. Vector DB is clean.")
    else:
        click.echo(f"\nOrphaned vectors:             {len(orphaned)}")
        if dry_run:
            click.echo("\nDry-run — would delete:")
            for vid in sorted(orphaned):
                click.echo(f"  {vid}")
        else:
            click.echo("\nDeleting orphaned vectors…")
            deleted = 0
            for vid in sorted(orphaned):
                try:
                    vdb.delete(collection, vid)
                    deleted += 1
                except Exception as exc:
                    click.echo(f"  ✗ {vid}: {exc}", err=True)
            click.echo(f"  ✓ Deleted {deleted} orphaned vector(s).")

    # ── optional snapshot pruning ─────────────────────────────────────────────
    if prune_snapshots:
        _prune_snapshots(repo, dry_run)


def _prune_snapshots(repo: "KbvcRepo", dry_run: bool) -> None:
    """Remove graph/prompt/retrieval snapshot files not reachable from HEAD."""
    import click
    from kbvc.core.commit import CommitObject

    click.echo("\nScanning snapshot reachability…")

    # Collect snapshots referenced by any reachable commit
    reachable_graphs: set[str] = set()
    reachable_prompts: set[str] = set()
    reachable_retrievals: set[str] = set()

    try:
        head_sha = repo.head_commit()
    except Exception:
        click.echo("  No commits — nothing to prune.")
        return

    from kbvc.core.commit import walk_dag
    try:
        commits = walk_dag(repo.commits_dir, head_sha)
    except Exception:
        commits = []

    for c in commits:
        if c.graph_snapshot:
            reachable_graphs.add(c.graph_snapshot)
        if c.prompt_snapshot:
            reachable_prompts.add(c.prompt_snapshot)
        if c.retrieval_snapshot:
            reachable_retrievals.add(c.retrieval_snapshot)

    pruned = 0
    directories = [
        (repo.graph_dir, "graph-v", reachable_graphs),
        (repo.kbvc_dir / "prompts", "p-v", reachable_prompts),
        (repo.kbvc_dir / "retrieval", "r-v", reachable_retrievals),
    ]
    for directory, prefix, keep_set in directories:
        if not directory.exists():
            continue
        for f in directory.iterdir():
            name = f.stem  # e.g. "graph-v3"
            if f.name.startswith(prefix) and name not in keep_set:
                if dry_run:
                    click.echo(f"  [dry-run] would prune: {f.name}")
                else:
                    f.unlink(missing_ok=True)
                    click.echo(f"  pruned: {f.name}")
                pruned += 1

    if pruned == 0:
        click.echo("  ✓ No unreachable snapshots found.")
    elif not dry_run:
        click.echo(f"  ✓ Pruned {pruned} unreachable snapshot file(s).")
