# kbvc/commands/commit.py
"""
kbvc commit — embed staged KOs and create a global commit object.

This module provides:
  _path_to_ko_id(src, frontmatter)  — derive the canonical KO ID for a file
  run_commit(repo, message, dry_run) — full commit pipeline

The commit pipeline:
  1. Load the staging index and verify something is staged
  2. Load embed + vectordb backends
  3. For each staged file: parse, chunk, diff-embed (only changed chunks)
  4. Upsert changed vectors; delete removed vectors
  5. Update KOStore (version, chunk_hashes, vector_ids)
  6. Write KOVersionSnapshot for each KO touched
  7. Snapshot graph / prompt / retrieval stores
  8. Create and persist CommitObject
  9. Advance HEAD, rewrite kbvc.lock, clear staging index
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


# ---------------------------------------------------------------------------
# _path_to_ko_id — canonical KO identifier
# ---------------------------------------------------------------------------

def _path_to_ko_id(src: Path, frontmatter: dict) -> str:
    """
    Return the canonical KO ID for a source file.

    Priority:
      1. The ``id`` field in the frontmatter YAML (explicit, stable)
      2. The file stem (e.g. ``caching-policy`` for ``caching-policy.md``)

    The stem fallback ensures that files without a proper frontmatter block
    still receive a deterministic, human-readable KO ID derived from their
    filename — consistent with how kbvc checkout and kbvc history reference
    KOs when frontmatter is absent.
    """
    return str(frontmatter.get("id", "") or src.stem)


# ---------------------------------------------------------------------------
# run_commit — full embed + commit pipeline
# ---------------------------------------------------------------------------

def run_commit(
    repo: "KbvcRepo",
    message: str,
    dry_run: bool = False,
) -> None:
    """
    Embed staged KOs and persist a CommitObject.

    Reads the staging index, embeds only changed/new chunks (diff-embed),
    updates all stores, and advances HEAD.  Exits with a ClickException on
    any unrecoverable error so the CLI always shows a clean message.
    """
    import click

    from kbvc.backends import get_embed_backend, get_vectordb_backend
    from kbvc.core.chunker import (
        parse_frontmatter,
        split_into_chunks,
        compute_changed_chunks,
        compute_deleted_chunks,
    )
    from kbvc.core.commit import CommitObject, KOChange
    from kbvc.core.graph import RelationGraph
    from kbvc.core.index import StagingIndex
    from kbvc.core.ko import KOStore, KnowledgeObject
    from kbvc.core.prompt_store import PromptVersionStore
    from kbvc.core.retrieval_store import RetrievalConfigStore
    from kbvc.core.versioner import KOVersioner, KOVersionSnapshot
    from kbvc.utils.lock import write_lock_file
    from datetime import datetime, timezone

    # ── 1. Load staging index ─────────────────────────────────────────────────
    index = StagingIndex.load(repo.index_path)
    if index.is_empty:
        raise click.ClickException(
            "Nothing staged. Use: kbvc add <file>   or   kbvc add ."
        )

    config = repo.config()
    branch = repo.current_branch()
    collection = config.get("vectordb.collection", "kbvc")

    # ── 2. Load backends ──────────────────────────────────────────────────────
    try:
        embed = get_embed_backend(config)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        vdb = get_vectordb_backend(config)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)
    versioner = KOVersioner(repo.ko_versions_dir)

    changed_kos: Dict[str, KOChange] = {}

    # Load graph early so depends_on relations can be registered during the file loop
    graph = RelationGraph(repo.graph_dir, branch)

    # ── 3–6. Embed staged files ───────────────────────────────────────────────
    for rel_str in list(index.staged_files):
        src = repo.root / rel_str
        if not src.exists():
            click.echo(f"  ⚠ Source file missing, skipping: {rel_str}", err=True)
            continue

        try:
            content = src.read_text(encoding="utf-8")
        except OSError as exc:
            click.echo(f"  ✗ Cannot read {rel_str}: {exc}", err=True)
            continue

        frontmatter, body = parse_frontmatter(content)
        ko_id = _path_to_ko_id(src, frontmatter)

        new_chunks = split_into_chunks(
            body, frontmatter,
            split_on=config.get("chunk.split_on", "##"),
            target_tokens=int(config.get("chunk.size", 400)),
        )

        existing_ko = ko_store.get(ko_id)
        old_hashes = list(existing_ko.chunk_hashes) if existing_ko else []
        old_version = existing_ko.version if existing_ko else 0

        changed_indices = compute_changed_chunks(new_chunks, old_hashes)
        deleted_indices = compute_deleted_chunks(new_chunks, old_hashes)

        if dry_run:
            click.echo(
                f"  [dry-run] {ko_id}  "
                f"+{len(changed_indices)} changed  -{len(deleted_indices)} deleted"
            )
            changed_kos[ko_id] = KOChange(
                from_version=old_version,
                to_version=old_version + 1,
                chunks_reembedded=list(changed_indices),
                reason=index.ko_reasons.get(ko_id, ""),
            )
            continue

        # Delete vectors for removed chunks
        for idx in deleted_indices:
            vid = f"{branch}__{ko_id}__chunk_{idx}"
            try:
                vdb.delete(collection, vid)
            except Exception:
                pass  # tolerate missing vectors on delete

        # Embed and upsert changed/new chunks
        embed_items = []
        for chunk in new_chunks:
            if chunk.index in changed_indices:
                try:
                    vec = embed.embed(chunk.text)
                except Exception as exc:
                    raise click.ClickException(
                        f"Embedding failed for {ko_id} chunk {chunk.index}: {exc}"
                    )
                vid = f"{branch}__{ko_id}__chunk_{chunk.index}"
                embed_items.append({
                    "id": vid,
                    "vector": vec,
                    "metadata": {
                        "ko_id": ko_id,
                        "chunk_index": chunk.index,
                        "chunk_hash": chunk.hash,
                        "section": chunk.section,
                        "branch": branch,
                        "ko_version": old_version + 1,
                        "source_path": rel_str,
                    },
                })

        if embed_items:
            vdb.upsert_batch(collection, embed_items)

        new_version = old_version + 1
        new_hashes = [c.hash for c in new_chunks]
        new_vector_ids = [f"{branch}__{ko_id}__chunk_{c.index}" for c in new_chunks]

        # Update or create KO in store
        now_iso = datetime.now(timezone.utc).date().isoformat()
        if existing_ko:
            from dataclasses import replace
            updated_ko = replace(
                existing_ko,
                version=new_version,
                last_updated=now_iso,
                chunk_hashes=new_hashes,
                vector_ids=new_vector_ids,
                branch=branch,
                tags=frontmatter.get("tags", existing_ko.tags),
                volatility=frontmatter.get("volatility", existing_ko.volatility),
            )
            ko_store.update(updated_ko)
        else:
            new_ko = KnowledgeObject(
                id=ko_id,
                source_type=frontmatter.get("source_type", "file"),
                path=rel_str,
                type=frontmatter.get("type", "doc"),
                tags=frontmatter.get("tags") or [],
                volatility=frontmatter.get("volatility", "slow"),
                version=new_version,
                last_updated=now_iso,
                chunk_hashes=new_hashes,
                vector_ids=new_vector_ids,
                branch=branch,
                valid_from=frontmatter.get("valid_from"),
                valid_to=frontmatter.get("valid_to"),
                depends_on=frontmatter.get("depends_on") or [],
            )
            ko_store.add(new_ko)

        # ── Auto-register frontmatter relations into the graph ───────────────
        depends_on = frontmatter.get("depends_on") or []
        if isinstance(depends_on, str):
            depends_on = [d.strip() for d in depends_on.split(",") if d.strip()]
        for dep_id in depends_on:
            # Avoid duplicate edges — check if this relation already exists
            existing = [
                r for r in graph.relations
                if r.from_id == ko_id and r.to_id == dep_id and r.type == "part_of"
            ]
            if not existing:
                rel = graph.add(
                    from_id=ko_id,
                    to_id=dep_id,
                    rel_type="part_of",
                    note=f"auto: {ko_id} depends on {dep_id} (from frontmatter)",
                )
                # Back-fill auto-discovery metadata
                rel.source = "auto"
                rel.confidence = 0.9
                graph._save_current()

        ko_change = KOChange(
            from_version=old_version,
            to_version=new_version,
            chunks_reembedded=list(changed_indices),
            reason=index.ko_reasons.get(ko_id, ""),
            chunks=[
                {"index": c.index, "section": c.section, "text": c.text, "hash": c.hash}
                for c in new_chunks
            ],
        )
        changed_kos[ko_id] = ko_change

        click.echo(
            f"  ✓ {ko_id}  v{new_version}"
            f"  ({len(embed_items)} chunks embedded, {len(deleted_indices)} deleted)"
        )

    if not changed_kos and not dry_run:
        raise click.ClickException("Nothing to commit after processing staged files.")

    if dry_run:
        click.echo("\n[dry-run] No changes written.")
        return

    # ── 7. Snapshot graph / prompt / retrieval ────────────────────────────────
    graph_version = graph.next_version()
    graph_snapshot = graph.snapshot(graph_version)

    prompt_store = PromptVersionStore(repo.prompts_dir)
    # commit_id is "pending" at this point; back-filled below
    prompt_snapshot = prompt_store.snapshot(commit_id="pending")

    retrieval_store = RetrievalConfigStore(repo.retrieval_dir)
    retrieval_snapshot = retrieval_store.snapshot(
        commit_id="pending",
        config=config,
        embed_backend=embed,
    )

    # ── 8. Create and persist CommitObject ────────────────────────────────────
    parent = repo.head_commit()
    commit_obj = CommitObject.create(
        parent=parent,
        branch=branch,
        message=message,
        changed_kos=changed_kos,
        graph_snapshot=graph_snapshot,
        prompt_snapshot=prompt_snapshot,
        retrieval_snapshot=retrieval_snapshot,
    )
    commit_obj.save(repo.commits_dir)

    # Back-fill commit_id into KO version snapshots
    for ko_id, ko_change in changed_kos.items():
        snap = KOVersionSnapshot(
            version=ko_change.to_version,
            commit_id=commit_obj.commit_id,
            reason=ko_change.reason,
            frontmatter={},
            chunks=ko_change.chunks,
            vector_ids=[
                f"{branch}__{ko_id}__chunk_{c['index']}"
                for c in ko_change.chunks
            ],
            changed_chunks=ko_change.chunks_reembedded,
            deleted_chunks=[],
        )
        # Only save if not already saved (idempotency guard)
        if not versioner.exists(ko_id, ko_change.to_version):
            versioner.save_version(ko_id, snap)

    # ── 9. Advance HEAD, write lock, clear index ──────────────────────────────
    repo.advance_head(commit_obj.commit_id)
    write_lock_file(repo.lock_path, config, embed)
    index.clear()
    index.save(repo.index_path)

    click.echo(
        f"\n✓ Committed {commit_obj.display_id}  \"{message}\""
        f"  ({len(changed_kos)} KO(s))"
    )
