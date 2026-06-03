# kbvc/commands/commit.py
"""
run_commit() — the most critical path in KBVC.

Steps (from §8.12):
  1.  Resolve embed + vectordb backends
  2.  Process each staged file: diff chunks, embed changed, upsert, delete removed
  3.  Snapshot the graph
  4.  Snapshot current prompt (auto-create default if never set)
  5.  Snapshot active retrieval config
  6.  Create the global CommitObject (SHA-256 DAG)
  7.  Back-fill commit_id into prompt/retrieval snapshots + vector metadata
  8.  Persist per-KO version snapshots
  9.  Save commit object + advance HEAD
  10. Update kbvc.lock
  11. Clear staging index
  12. Print output
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import click

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


def run_commit(repo: "KbvcRepo", message: str, dry_run: bool = False) -> None:
    from kbvc.backends import get_embed_backend, get_vectordb_backend
    from kbvc.core.chunker import (
        compute_changed_chunks,
        compute_deleted_chunks,
        parse_frontmatter,
        split_into_chunks,
    )
    from kbvc.core.commit import CommitObject, KOChange
    from kbvc.core.graph import RelationGraph
    from kbvc.core.index import StagingIndex
    from kbvc.core.ko import KnowledgeObject, KOStore
    from kbvc.core.prompt_store import PromptVersionStore
    from kbvc.core.retrieval_store import RetrievalConfigStore
    from kbvc.core.versioner import KOVersionSnapshot, KOVersioner
    from kbvc.utils.display import print_commit_summary, print_warning
    from kbvc.utils.lock import write_lock_file

    config = repo.config()
    index = StagingIndex.load(repo.index_path)
    ko_store = KOStore(repo.ko_store_path)
    branch = repo.current_branch()
    collection = config.get("vectordb.collection", "kbvc")

    # Guard: nothing staged
    if index.is_empty:
        raise click.ClickException(
            "Nothing to commit. Stage files with: kbvc add <file>"
        )

    # ── Step 1: resolve backends ──────────────────────────────────────────────
    try:
        embed = get_embed_backend(config)
        vdb = get_vectordb_backend(config)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # ── Step 2: process each staged file ─────────────────────────────────────
    changed_kos: Dict[str, KOChange] = {}
    versioner = KOVersioner(repo.ko_versions_dir)

    for file_path_str in index.staged_files:
        src = repo.root / file_path_str
        if not src.exists():
            raise click.ClickException(
                f"Staged file not found: {file_path_str}\n"
                "File may have been deleted after staging. "
                "Run 'kbvc status' to inspect."
            )

        content = src.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)

        # P2: consistent ko_id derivation (single helper mirrors §8.15)
        ko_id = _path_to_ko_id(src, frontmatter)

        new_chunks = split_into_chunks(
            body,
            frontmatter,
            split_on=config.get("chunk.split_on", "##"),
            target_tokens=int(config.get("chunk.size", 400)),
        )

        existing_ko = ko_store.get(ko_id)
        stored_hashes = existing_ko.chunk_hashes if existing_ko else []
        old_version = existing_ko.version if existing_ko else 0

        changed_idxs = compute_changed_chunks(new_chunks, stored_hashes)
        deleted_idxs = compute_deleted_chunks(new_chunks, stored_hashes)

        if not changed_idxs and not deleted_idxs:
            click.echo(f"  [unchanged] {file_path_str}")
            continue   # truly unchanged — skip

        if dry_run:
            n_embed = len(changed_idxs)
            n_del = len(deleted_idxs)
            click.echo(
                f"  {ko_id}@{old_version} → @{old_version + 1}  "
                f"({n_embed} re-embed, {n_del} delete)"
            )
            continue

        # Volatility guard: frozen KOs cannot be re-embedded
        if existing_ko and existing_ko.volatility == "frozen":
            print_warning(
                f"Skipping {ko_id} — volatility=frozen. "
                "Use 'kbvc patch' to update metadata without re-embedding."
            )
            continue

        # Embed changed chunks and collect for batch upsert
        embed_items = []
        for chunk in new_chunks:
            if chunk.index in changed_idxs:
                try:
                    vector = embed.embed(chunk.text)
                except Exception as exc:
                    raise click.ClickException(
                        f"Embedding failed for {ko_id} chunk {chunk.index}: {exc}"
                    )
                vid = f"{branch}__{ko_id}__chunk_{chunk.index}"
                embed_items.append({
                    "id": vid,
                    "vector": vector,
                    "metadata": {
                        "ko_id": ko_id,
                        "chunk_index": chunk.index,
                        "chunk_hash": chunk.hash,
                        "section": chunk.section,
                        "branch": branch,
                        "ko_version": old_version + 1,
                        "commit_id": "pending",   # back-filled in Step 7
                        "source_path": file_path_str,
                    },
                })

        if embed_items:
            try:
                vdb.upsert_batch(collection, embed_items)
            except Exception as exc:
                raise click.ClickException(
                    f"Vector DB upsert failed for {ko_id}: {exc}"
                )

        # Delete vectors for removed chunks (shrunk document)
        for idx in deleted_idxs:
            vid = f"{branch}__{ko_id}__chunk_{idx}"
            try:
                vdb.delete(collection, vid)
            except Exception:
                pass   # best-effort delete; orphans cleaned by kbvc gc

        new_version = old_version + 1
        new_hashes = [c.hash for c in new_chunks]
        new_vids = [f"{branch}__{ko_id}__chunk_{i}" for i in range(len(new_chunks))]
        reason = index.ko_reasons.get(ko_id, "")

        # Update KO in store
        if existing_ko:
            updated_ko = replace(
                existing_ko,
                version=new_version,
                last_updated=date.today().isoformat(),
                chunk_hashes=new_hashes,
                vector_ids=new_vids,
            )
            ko_store.update(updated_ko)
        else:
            new_ko = KnowledgeObject(
                id=ko_id,
                source_type=frontmatter.get("source_type", "file"),
                path=file_path_str,
                type=frontmatter.get("type", "document"),
                tags=frontmatter.get("tags") or [],
                volatility=frontmatter.get("volatility", "slow"),
                version=1,
                last_updated=date.today().isoformat(),
                chunk_hashes=new_hashes,
                vector_ids=new_vids,
                branch=branch,
                valid_from=frontmatter.get("valid_from"),
                valid_to=frontmatter.get("valid_to"),
            )
            ko_store.add(new_ko)

        changed_kos[ko_id] = KOChange(
            from_version=old_version,
            to_version=new_version,
            chunks_reembedded=changed_idxs,
            reason=reason,
            chunks=[{"index": c.index, "section": c.section} for c in new_chunks],
        )

    if dry_run:
        return

    # After processing all staged files: check if anything actually changed
    if not changed_kos and not index.graph_dirty and not index.prompt_dirty:
        click.echo("Nothing to commit — all staged files are unchanged.")
        index.clear()
        index.save(repo.index_path)
        return

    # ── Step 3: snapshot graph ────────────────────────────────────────────────
    graph = RelationGraph(repo.graph_dir, branch)
    graph_version = graph.next_version()
    graph_snapshot = graph.snapshot(graph_version)

    # ── Step 4: snapshot prompt ───────────────────────────────────────────────
    prompt_store = PromptVersionStore(repo.prompts_dir)
    # Pass "pending" — back-filled after commit_id is known (Step 7)
    prompt_snapshot = prompt_store.snapshot(commit_id="pending")

    # ── Step 5: snapshot retrieval config ─────────────────────────────────────
    retrieval_store = RetrievalConfigStore(repo.retrieval_dir)
    retrieval_snapshot = retrieval_store.snapshot("pending", config, embed)

    # ── Step 6: create CommitObject (deterministic SHA-256) ───────────────────
    parent = repo.head_commit()
    commit = CommitObject.create(
        parent=parent,
        branch=branch,
        message=message,
        changed_kos=changed_kos,
        graph_snapshot=graph_snapshot,
        prompt_snapshot=prompt_snapshot,
        retrieval_snapshot=retrieval_snapshot,
    )

    # ── Step 7: back-fill commit_id into snapshots + vector metadata ──────────

    # Prompt snapshot: rewrite with real commit_id
    p_path = repo.prompts_dir / f"{prompt_snapshot}.json"
    p_data = json.loads(p_path.read_text(encoding="utf-8"))
    p_data["commit_id"] = commit.commit_id
    p_path.write_text(json.dumps(p_data, indent=2), encoding="utf-8")

    # Retrieval snapshot: same
    r_path = repo.retrieval_dir / f"{retrieval_snapshot}.json"
    r_data = json.loads(r_path.read_text(encoding="utf-8"))
    r_data["commit_id"] = commit.commit_id
    r_path.write_text(json.dumps(r_data, indent=2), encoding="utf-8")

    # Vector metadata: patch all newly upserted vectors with the real commit_id
    for ko_id_key in changed_kos:
        try:
            vdb.patch_metadata(
                collection,
                f"{branch}__{ko_id_key}__",
                {"commit_id": commit.commit_id},
            )
        except Exception:
            pass   # best-effort; metadata patch failures don't break the commit

    # ── Step 8: persist per-KO version snapshots ──────────────────────────────
    for ko_id_key, change in changed_kos.items():
        ko = ko_store.get(ko_id_key)
        versioner.save_version(
            ko_id_key,
            KOVersionSnapshot(
                version=change.to_version,
                commit_id=commit.commit_id,
                reason=change.reason,
                frontmatter={},
                chunks=change.chunks,        # {index, section} — enables kbvc explain section names
                vector_ids=ko.vector_ids if ko else [],
                changed_chunks=change.chunks_reembedded,
                deleted_chunks=[],
            ),
        )

    # ── Step 9: save commit + advance HEAD ────────────────────────────────────
    commit.save(repo.commits_dir)
    repo.advance_head(commit.commit_id)

    # ── Step 10: update kbvc.lock ─────────────────────────────────────────────
    try:
        write_lock_file(repo.lock_path, config, embed)
    except Exception:
        pass   # lock file failure is non-fatal

    # ── Step 11: clear staging index ──────────────────────────────────────────
    index.clear()
    index.save(repo.index_path)

    # ── Step 12: print commit summary ─────────────────────────────────────────
    print_commit_summary(commit, changed_kos)

    # P10: graph-only commit message
    if not changed_kos:
        click.echo("  (no KOs changed)")

    # Staleness detection: warn if any committed KO is listed as a dependency
    # of other KOs that haven't been re-committed (§3.3 depends_on)
    _warn_stale_dependents(ko_store, changed_kos)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_to_ko_id(path: Path, frontmatter: dict) -> str:
    """
    Derive a consistent KO id from a file path and its frontmatter.
    P2 from §8.15: this same function must be used in kbvc link, kbvc depends,
    kbvc checkout, etc. to avoid drift between how different commands resolve IDs.
    """
    return frontmatter.get("id") or path.stem.replace(" ", "-").lower()


def _warn_stale_dependents(ko_store: "KOStore", changed_kos: dict) -> None:
    """
    After a commit, scan all KOs to find dependents of just-changed KOs
    whose last_updated predates the dependency. Print a staleness warning.
    """
    if not changed_kos:
        return

    changed_ids = set(changed_kos.keys())
    stale: list[str] = []

    for ko in ko_store.all():
        # Skip KOs that were just committed
        if ko.id in changed_ids:
            continue
        # Check if any of this KO's dependencies were just changed
        if any(dep in changed_ids for dep in (ko.depends_on or [])):
            stale.append(ko.id)

    if stale:
        click.echo("")
        click.echo(
            "⚠  Dependency notice: the following KOs depend on just-changed KOs"
            " and may be stale:"
        )
        for ko_id in stale:
            click.echo(f"   - {ko_id}")
        click.echo("   Run: kbvc impact <file> to inspect")
