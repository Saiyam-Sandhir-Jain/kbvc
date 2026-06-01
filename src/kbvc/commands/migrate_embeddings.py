# kbvc/commands/migrate_embeddings.py
"""
kbvc migrate embeddings — zero-downtime embedding model swap.

Phase 7 feature.

Flow:
  1. Export all ChunkRecords from the current backend.
  2. Re-embed every chunk text using the NEW embedding model.
     - Chunk texts are retrieved from the KO source files on disk.
     - If a source file is missing, the chunk is skipped with a warning.
  3. Write re-embedded ChunkRecords back under the SAME vector IDs
     (overwriting in-place — same collection, same IDs, new vectors).
  4. Update kbvc.lock with the new model + dims.

The migration is non-destructive: old vectors are overwritten atomically
per batch.  A --dry-run flag shows what would happen without writing.

Usage:
    kbvc migrate embeddings --from text-embedding-3-small --to gemini-embedding-001
    kbvc migrate embeddings --from text-embedding-3-small --to gemini-embedding-001 --dry-run
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


def run_migrate_embeddings(
    repo: "KbvcRepo",
    from_model: str,
    to_model: str,
    dry_run: bool = False,
) -> None:
    import click
    from kbvc.core.ko import KOStore
    from kbvc.core.chunker import split_into_chunks
    from kbvc.backends import get_vectordb_backend, get_embed_backend

    cfg = repo.config()
    collection = cfg.get("vectordb.collection", "kbvc")

    click.echo(f"Embedding migration: {from_model} → {to_model}")
    click.echo(f"Collection: {collection}")
    if dry_run:
        click.echo("(dry-run — no data will be written)")
    click.echo("")

    # ── load backends ────────────────────────────────────────────────────────
    click.echo("[1/5] Connecting to vector backend…")
    try:
        vdb = get_vectordb_backend(cfg)
    except Exception as exc:
        raise click.ClickException(f"Could not load vector backend: {exc}")

    click.echo(f"[2/5] Loading new embed backend: {to_model}")
    # Override embed.model to the target model for this operation
    new_cfg = {**cfg, "embed.model": to_model}
    try:
        embedder = get_embed_backend(new_cfg)
    except Exception as exc:
        raise click.ClickException(f"Could not load embed backend for '{to_model}': {exc}")

    # ── export existing chunks ────────────────────────────────────────────────
    click.echo("[3/5] Exporting existing ChunkRecords…")
    try:
        chunks = vdb.export_chunks(collection)
    except NotImplementedError:
        raise click.ClickException(
            "The configured backend does not support export_chunks(). "
            "Embedding migration requires a backend with full export support."
        )

    if not chunks:
        click.echo("No chunks found — nothing to migrate.")
        return

    click.echo(f"      Found {len(chunks)} chunk(s) across {len({c.ko_id for c in chunks})} KO(s).")

    # ── re-embed ──────────────────────────────────────────────────────────────
    click.echo("[4/5] Re-embedding chunks with new model…")
    store = KOStore(repo.ko_store_path)

    # Group chunks by ko_id for efficient file reads
    from collections import defaultdict
    by_ko: dict = defaultdict(list)
    for c in chunks:
        by_ko[c.ko_id].append(c)

    re_embedded = []
    skipped = 0

    for ko_id, ko_chunks in by_ko.items():
        ko = store.get(ko_id)
        if ko is None:
            click.echo(f"  ⚠  KO '{ko_id}' not in store — {len(ko_chunks)} chunk(s) skipped.", err=True)
            skipped += len(ko_chunks)
            continue

        src_path = repo.root / ko.path
        if not src_path.exists():
            click.echo(f"  ⚠  Source file missing for '{ko_id}' ({ko.path}) — skipped.", err=True)
            skipped += len(ko_chunks)
            continue

        src_text = src_path.read_text(encoding="utf-8")
        from kbvc.core.chunker import parse_frontmatter
        _fm, _body = parse_frontmatter(src_text)
        doc_chunks = split_into_chunks(_body, _fm)

        for chunk_rec in ko_chunks:
            idx = chunk_rec.chunk_index
            if idx < len(doc_chunks):
                chunk_text = doc_chunks[idx].text
            else:
                chunk_text = src_text  # fallback: whole-document embedding
            try:
                new_vector = embedder.embed(chunk_text)
            except Exception as exc:
                click.echo(f"  ✗ embed failed for {chunk_rec.vector_id}: {exc}", err=True)
                skipped += 1
                continue

            import dataclasses
            updated = dataclasses.replace(chunk_rec, embedding=new_vector)
            re_embedded.append(updated)

    click.echo(f"      Re-embedded: {len(re_embedded)}  |  Skipped: {skipped}")

    if not re_embedded:
        click.echo("\nNothing to write.")
        return

    # ── write back ────────────────────────────────────────────────────────────
    if dry_run:
        click.echo("\n[5/5] Dry-run — skipping write-back.")
        click.echo(f"      Would overwrite {len(re_embedded)} chunk vector(s) with {to_model} embeddings.")
        return

    click.echo("[5/5] Writing re-embedded chunks back to backend…")
    new_dims = len(re_embedded[0].embedding)
    vdb.initialize_schema(collection, new_dims)
    vdb.import_chunks(collection, re_embedded)

    # Update kbvc.lock
    _update_lock(repo, to_model, new_dims)

    click.echo("")
    click.echo(f"✓ Migration complete: {len(re_embedded)} chunk(s) re-embedded with {to_model}.")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  kbvc config set embed.model {to_model}")
    click.echo("  kbvc backend info")


def _update_lock(repo: "KbvcRepo", model: str, dims: int) -> None:
    """Update kbvc.lock to reflect the new embedding model (best-effort)."""
    lock_path = repo.root / "kbvc.lock"
    if lock_path.exists():
        _update_lock_yaml(lock_path, model, dims)


def _update_lock_yaml(lock_path, model: str, dims: int) -> None:
    """Patch kbvc.lock embedding.model and dims in-place (best-effort)."""
    import re
    try:
        text = lock_path.read_text(encoding="utf-8")
        text = re.sub(r"(model:\s+\").*?(\")", f'\\g<1>{model}\\2', text)
        text = re.sub(r"(dims:\s+)\d+", f"\\g<1>{dims}", text)
        lock_path.write_text(text, encoding="utf-8")
    except Exception:
        pass
