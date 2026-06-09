# kbvc/commands/push.py
"""
kbvc push — push committed knowledge state to a remote vector DB.

Design separation (key principle):
  kbvc commit  — creates the commit object + embeds + writes to the LOCAL vector DB
  kbvc push    — syncs the committed state to a REMOTE target

This mirrors git's commit vs push model:
  - You can commit many times locally before pushing
  - push is explicit, auditable, and reversible
  - Different branches can push to different remote collections

Remote target config lives in .kbvc/config under [remote.*] sections.

Usage:
    kbvc push                        # push to default remote
    kbvc push --remote staging       # push to named remote
    kbvc push --collection prod-v2   # override collection name
    kbvc push --dry-run              # show what would be pushed

v1: pushes by re-embedding (safe but expensive).
v2: vector export/import using native DB dump APIs (zero re-embed cost).

Push tracking: .kbvc/refs/remotes/<remote_name> stores the last pushed
commit_id so kbvc push knows what to push incrementally.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


# ---------------------------------------------------------------------------
# Push tracking
# ---------------------------------------------------------------------------

def get_remote_head(kbvc_dir: Path, remote_name: str) -> Optional[str]:
    """Return the last commit_id successfully pushed to this remote, or None."""
    ref = kbvc_dir / "refs" / "remotes" / remote_name
    if ref.exists():
        h = ref.read_text(encoding="utf-8").strip()
        return h if h else None
    return None


def set_remote_head(kbvc_dir: Path, remote_name: str, commit_id: str) -> None:
    """Record a successful push."""
    ref_dir = kbvc_dir / "refs" / "remotes"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / remote_name).write_text(commit_id, encoding="utf-8")


def get_remote_config(config: dict, remote_name: str) -> dict:
    """
    Extract remote-specific config from the flat config dict.

    ConfigParser collapses remote.NAME.KEY → section [remote], key NAME.KEY.
    So in the flat dict: key = "remote.NAME.KEY"
    We return a sub-dict with just KEY → value for the named remote.
    """
    prefix = f"remote.{remote_name}."
    remote_cfg: dict = {}
    for k, v in config.items():
        if k.startswith(prefix):
            short_key = k[len(prefix):]
            remote_cfg[short_key] = v
    # Also pull global embed config (embed model must match)
    for k in ("embed.backend", "embed.model", "embed.key", "embed.url"):
        if k in config:
            remote_cfg[k] = config[k]
    return remote_cfg


def run_push(
    repo: "KbvcRepo",
    remote_name: str = "origin",
    collection_override: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Push the current HEAD commit's vectors to the configured remote.

    Steps:
      1. Resolve remote config
      2. Determine which commits haven't been pushed yet (delta since remote HEAD)
      3. For each un-pushed commit: re-embed changed chunks and upsert to remote
      4. Patch metadata with collection-specific commit_id
      5. Advance the remote tracking ref
    """
    import click
    from kbvc.backends import get_embed_backend, get_vectordb_backend
    from kbvc.core.chunker import parse_frontmatter, split_into_chunks
    from kbvc.core.commit import walk_dag
    from kbvc.core.ko import KOStore
    from kbvc.core.versioner import KOVersioner

    local_head = repo.head_commit()
    if not local_head:
        raise click.ClickException("Nothing to push — no local commits yet.")

    remote_head = get_remote_head(repo.kbvc_dir, remote_name)
    if remote_head == local_head:
        click.echo(f"Remote '{remote_name}' is already up to date.")
        return

    # Resolve config
    local_config = repo.config()
    remote_raw = get_remote_config(local_config, remote_name)
    if not remote_raw.get("backend") and not remote_raw.get("vectordb.backend"):
        raise click.ClickException(
            f"Remote '{remote_name}' not configured.\n"
            f"Run: kbvc remote add {remote_name} --backend qdrant "
            f"--url https://... --collection kbvc-prod"
        )

    # Build a synthetic config dict for the remote backend
    remote_config = dict(local_config)
    if remote_raw.get("backend"):
        remote_config["vectordb.backend"] = remote_raw["backend"]
    if remote_raw.get("url"):
        remote_config["vectordb.url"] = remote_raw["url"]
    if remote_raw.get("key"):
        remote_config["vectordb.key"] = remote_raw["key"]
    collection = collection_override or remote_raw.get("collection", "kbvc-remote")
    remote_config["vectordb.collection"] = collection

    try:
        embed = get_embed_backend(remote_config)
        vdb = get_vectordb_backend(remote_config)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # Walk DAG from local HEAD, stop at remote HEAD
    all_commits = walk_dag(repo.commits_dir, local_head)
    to_push = []
    for c in all_commits:
        if c.commit_id == remote_head:
            break
        to_push.append(c)
    to_push.reverse()  # oldest first

    if not to_push:
        click.echo(f"Nothing to push to '{remote_name}'.")
        return

    branch = repo.current_branch()
    ko_store = KOStore(repo.ko_store_path)
    versioner = KOVersioner(repo.ko_versions_dir)

    click.echo(
        f"Pushing {len(to_push)} commit(s) → '{remote_name}' "
        f"collection='{collection}'"
        + (" [dry-run]" if dry_run else "")
    )

    total_upserts = 0
    for commit in to_push:
        click.echo(f"  {commit.display_id}  {commit.message}")
        for ko_id, change in commit.changed_kos.items():
            ko = ko_store.get(ko_id)
            if not ko:
                click.echo(f"    ⚠ KO '{ko_id}' not found in store — skipping")
                continue

            snap = versioner.load_version(ko_id, change.to_version)
            src = repo.root / ko.path
            if not src.exists():
                click.echo(f"    ⚠ Source file missing: {ko.path} — skipping")
                continue

            content = src.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content)
            chunks = split_into_chunks(body, fm)

            embed_items = []
            for chunk in chunks:
                if chunk.index in change.chunks_reembedded:
                    if dry_run:
                        click.echo(f"    [dry-run] would embed {ko_id} chunk {chunk.index}")
                        continue
                    vec = embed.embed(chunk.text)
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
                            "ko_version": change.to_version,
                            "commit_id": commit.commit_id,
                            "source_path": ko.path,
                            "pushed_from": "kbvc-push",
                        },
                    })

            if embed_items and not dry_run:
                vdb.upsert_batch(collection, embed_items)
                total_upserts += len(embed_items)
                click.echo(f"    ✓ {ko_id}@{change.to_version} "
                           f"({len(embed_items)} chunks)")

    if not dry_run:
        set_remote_head(repo.kbvc_dir, remote_name, local_head)
        click.echo(f"\nPushed {total_upserts} vectors to '{remote_name}'. "
                   f"Remote now at {local_head[:7]}.")
    else:
        click.echo(f"\n[dry-run] Would push {len(to_push)} commit(s) to '{remote_name}'.")
