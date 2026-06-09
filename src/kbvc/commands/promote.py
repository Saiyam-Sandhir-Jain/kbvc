# kbvc/commands/promote.py
"""
kbvc promote — promote a memory or observation into a staged Knowledge Object.

This bridges the gap between ephemeral agent memory and permanent, versioned
knowledge.  The promoted KO is written as a .md file in knowledge/promoted/
and staged for the next commit.

Usage:
    kbvc promote "Dragonfly fixed the Redis OOM issue in 2026-05" \\
        --id dragonfly-fix \\
        --type lesson \\
        --confidence 0.9 \\
        --source agent

The KO file is created with:
  - The memory text as body content
  - YAML frontmatter populated from flags
  - volatility: slow (default — these are curated facts, not live docs)

After running this, use `kbvc status` to see the staged file, review it,
then `kbvc commit -m "promote dragonfly observation"`.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


def run_promote(
    repo: "KbvcRepo",
    memory: str,
    ko_id: Optional[str],
    ko_type: str,
    confidence: float,
    source: str,
    tags: str,
) -> None:
    import click
    from kbvc.core.index import StagingIndex

    # Derive KO id from memory text if not provided
    if not ko_id:
        ko_id = _slugify(memory[:40])

    today = date.today().isoformat()
    tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    tags_yaml = "[" + ", ".join(tags_list) + "]" if tags_list else "[]"

    # Build .md content
    content = f"""---
id: {ko_id}
type: {ko_type}
tags: {tags_yaml}
volatility: slow
valid_from: "{today}"
valid_to: null
source_type: memory
promoted_from: memory
confidence: {confidence}
promoted_by: {source}
---

## Memory

{memory}
"""

    # Write to knowledge/promoted/
    promoted_dir = repo.root / "knowledge" / "promoted"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    out_path = promoted_dir / f"{ko_id}.md"

    if out_path.exists():
        raise click.ClickException(
            f"File already exists: {out_path.relative_to(repo.root)}\n"
            "Use a different --id or edit the existing file."
        )

    out_path.write_text(content, encoding="utf-8")

    # Stage it
    index_path = repo.kbvc_dir / "index"
    idx = StagingIndex.load(index_path)
    rel_path = str(out_path.relative_to(repo.root))
    idx.stage(rel_path)
    idx.save(index_path)

    click.echo(f"✓ Created: {out_path.relative_to(repo.root)}")
    click.echo(f"  id:         {ko_id}")
    click.echo(f"  type:       {ko_type}")
    click.echo(f"  confidence: {confidence}")
    click.echo(f"  source:     {source}")
    click.echo("")
    click.echo("Staged and ready. Review the file, then:")
    click.echo(f'  kbvc commit -m "promote: {ko_id}"')


def _slugify(text: str) -> str:
    """Convert free-form text to a valid KO id slug."""
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:48] or "promoted-ko"
