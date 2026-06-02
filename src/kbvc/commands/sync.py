# kbvc/commands/sync.py
"""
kbvc sync — Volatility-aware auto-commit.

Phase 8 feature.

Scans all tracked KOs and stages + commits those whose source files have
changed since the last commit, filtered by volatility level:

  --volatility live     only `live` KOs
  --volatility slow     only `slow` KOs (default)
  --volatility all      all non-frozen KOs

Usage:
    kbvc sync                        # commit changed slow + live KOs
    kbvc sync --volatility live      # only live KOs
    kbvc sync --volatility all       # all non-frozen KOs
    kbvc sync --dry-run              # show what would be committed
    kbvc sync --message "nightly"    # custom commit message
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


def run_sync(
    repo: "KbvcRepo",
    volatility: str = "slow",  # "live" | "slow" | "all"
    dry_run: bool = False,
    message: str | None = None,
) -> None:
    import click
    from kbvc.core.ko import KOStore
    from kbvc.core.index import StagingIndex

    store = KOStore(repo.ko_store_path)
    index = StagingIndex.load(repo.index_path)

    # Determine which volatility levels to include
    if volatility == "live":
        include = {"live"}
    elif volatility == "slow":
        include = {"slow", "live"}
    else:  # "all"
        include = {"slow", "live"}  # never include frozen

    candidates = []
    for ko in store.all():
        if ko.volatility == "frozen":
            continue
        if ko.volatility not in include:
            continue
        src = repo.root / ko.path
        if not src.exists():
            continue
        # Compare current content hash against last committed chunk hashes
        src_text = src.read_text(encoding="utf-8")
        from kbvc.core.chunker import split_into_chunks, parse_frontmatter
        fm, body = parse_frontmatter(src_text)
        current_hashes = [c.hash for c in split_into_chunks(body, fm)]
        if current_hashes != list(ko.chunk_hashes):
            candidates.append(ko)

    if not candidates:
        click.echo("✓ Everything up to date — no changes detected.")
        return

    click.echo(f"Changed KOs ({len(candidates)}):")
    for ko in candidates:
        click.echo(f"  {ko.id}  [{ko.volatility}]  {ko.path}")

    if dry_run:
        click.echo("\nDry-run — nothing staged or committed.")
        return

    # Stage and commit
    for ko in candidates:
        index.stage(ko.path)

    commit_msg = message or f"kbvc sync: {len(candidates)} KO(s) updated"

    click.echo(f"\nCommitting: \"{commit_msg}\"")

    index.save(repo.index_path)

    from kbvc.commands.commit import run_commit

    run_commit(repo, commit_msg, dry_run=False)
    click.echo(f"✓ Synced {len(candidates)} KO(s).")
