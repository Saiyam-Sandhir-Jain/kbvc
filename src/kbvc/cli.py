# kbvc/cli.py
"""
KBVC CLI entry point — Click command group.

Phase 1 commands (working commit loop):
    kbvc init, kbvc config, kbvc add, kbvc commit, kbvc log

Phase 2 commands:
    kbvc status, kbvc checkout, kbvc diff, kbvc prompt

Phase 3 commands:
    kbvc link, kbvc unlink, kbvc graph, kbvc query

Phase 4 commands:
    kbvc branch, kbvc depends, kbvc impact, kbvc annotate,
    kbvc history, kbvc trace, kbvc doctor

Phase 5 commands:
    kbvc ingest, kbvc push, kbvc remote, kbvc analyze,
    kbvc extract, kbvc stale, kbvc stats

Phase 6 commands:
    kbvc backend init/info, kbvc migrate backend, kbvc explain,
    kbvc promote

Phase 7 commands:
    kbvc gc, kbvc migrate embeddings, kbvc migrate schema

Phase 8 commands:
    kbvc sync

Phase 9 commands:
    kbvc contradict list/resolve

Future (planned):
    kbvc ask (AI-assisted knowledge management)
"""

from __future__ import annotations

from pathlib import Path

import click


# ── Main group ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="kbvc")
def main():
    """KBVC — Git-native Knowledge Infrastructure Layer for AI systems."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Working commit loop
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--name", default=None, help="Repository name (default: directory name)")
@click.option("--no-git", is_flag=True, default=False, help="Skip git init")
def init(name, no_git):
    """Initialise a KBVC repository in the current directory."""
    from kbvc.core.repo import KbvcRepo
    try:
        repo = KbvcRepo.init(Path("."), name=name, no_git=no_git)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))

    click.echo(f"Initialised KBVC repository: {repo.root.name}")
    info = repo.repo_info()
    click.echo(f"  repo_id:  {info['repo_id']}")
    click.echo(f"  format:   v{info['format_version']}")
    if not no_git:
        click.echo(f"  git:      initialised ✓")
    click.echo("")
    click.echo("Next steps:")
    click.echo("  kbvc config set embed.backend openai   # or gemini / ollama / huggingface")
    click.echo("  kbvc config set embed.key sk-...")
    click.echo("  kbvc config set vectordb.backend lancedb  # or qdrant / chroma / pgvector")
    click.echo("  kbvc config set vectordb.url ./kbvc_lance")
    click.echo("  kbvc backend init")
    click.echo("  kbvc doctor")


# ── config group ──────────────────────────────────────────────────────────────

@main.group()
def config():
    """Get and set KBVC configuration."""
    pass


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a config value. Key: section.name (e.g. embed.backend)."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.utils.config import write_config_key
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        write_config_key(repo.kbvc_dir / "config", key, value)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"  {key} = {value}")


@config.command(name="get")
@click.argument("key")
def config_get(key):
    """Get a config value."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    cfg = repo.config()
    val = cfg.get(key)
    if val is None:
        raise click.ClickException(f"Config key not found: {key}")
    click.echo(val)


@config.command(name="list")
def config_list():
    """List all config values."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    cfg = repo.config()
    for key, val in sorted(cfg.items()):
        # Redact sensitive keys
        if any(s in key for s in ("key", "password", "token", "secret")):
            val = "***" if val else "(not set)"
        click.echo(f"  {key} = {val}")


# ── add ───────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("paths", nargs=-1, required=True)
def add(paths):
    """Stage files or directories for the next commit.

    Examples:
        kbvc add projects/manifestai.md
        kbvc add .
        kbvc add knowledge/
    """
    from kbvc.core.chunker import parse_frontmatter, split_into_chunks
    from kbvc.core.index import StagingIndex
    from kbvc.core.ko import KOStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    index = StagingIndex.load(repo.index_path)
    ko_store = KOStore(repo.ko_store_path)
    config = repo.config()

    staged_count = 0
    skipped_count = 0

    # Expand paths: resolve globs and directories
    all_files: list[Path] = []
    for path_str in paths:
        p = Path(path_str)
        if path_str == ".":
            # P6: sorted for deterministic staging order and commit hashes
            all_files.extend(sorted(repo.root.rglob("*.md")))
        elif p.is_dir():
            all_files.extend(sorted(p.rglob("*.md")))
        elif p.exists():
            all_files.append(p)
        else:
            # Try glob
            matches = sorted(repo.root.glob(path_str))
            if matches:
                all_files.extend(matches)
            else:
                click.echo(f"  ✗ Not found: {path_str}", err=True)

    for src in all_files:
        # Skip hidden files and .kbvc directory
        try:
            rel = src.relative_to(repo.root)
        except ValueError:
            rel = src
        parts = rel.parts
        if any(p.startswith(".") for p in parts):
            continue
        if not src.is_file():
            continue

        rel_str = str(rel)

        # Compute whether this file has changed vs stored KO
        try:
            content = src.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(content)
            from kbvc.commands.commit import _path_to_ko_id
            ko_id = _path_to_ko_id(src, frontmatter)
            new_chunks = split_into_chunks(
                body, frontmatter,
                split_on=config.get("chunk.split_on", "##"),
                target_tokens=int(config.get("chunk.size", 400)),
            )
            existing_ko = ko_store.get(ko_id)
            stored_hashes = existing_ko.chunk_hashes if existing_ko else []

            from kbvc.core.chunker import compute_changed_chunks, compute_deleted_chunks
            changed = compute_changed_chunks(new_chunks, stored_hashes)
            deleted = compute_deleted_chunks(new_chunks, stored_hashes)

            if not changed and not deleted and existing_ko:
                status = "unchanged"
                skipped_count += 1
            elif existing_ko:
                status = f"modified (+{len(changed)} -{len(deleted)} chunks)"
                index.stage(rel_str)
                staged_count += 1
            else:
                status = f"new ({len(new_chunks)} chunks)"
                index.stage(rel_str)
                staged_count += 1

            click.echo(f"  {status:30s} {rel_str}")

        except Exception as exc:
            click.echo(f"  ✗ Error reading {rel_str}: {exc}", err=True)

    index.save(repo.index_path)
    click.echo("")
    click.echo(f"Staged {staged_count} file(s). {skipped_count} unchanged.")
    if staged_count:
        click.echo("Run: kbvc commit -m \"<message>\" to embed and commit.")


# ── commit ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("-m", "--message", required=True, help="Commit message")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be embedded without writing anything")
def commit(message, dry_run):
    """Embed staged KOs and create a global commit object."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.commit import run_commit
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    run_commit(repo, message, dry_run=dry_run)


# ── log ───────────────────────────────────────────────────────────────────────

@main.command(name="log")
@click.option("--oneline", is_flag=True, help="Compact one-line format")
@click.argument("file", required=False)
def log_cmd(oneline, file):
    """Show commit history.

    Optionally filter to commits that touched a specific file:
        kbvc log -- projects/manifestai.md
    """
    from kbvc.core.commit import walk_dag
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    # P3: walk the DAG from HEAD instead of sorting by mtime
    commits = walk_dag(repo.commits_dir, repo.head_commit())

    if not commits:
        click.echo("No commits yet. Run: kbvc add <file> && kbvc commit -m \"...\"")
        return

    for c in commits:
        # Optional filter: only show commits that touched the given file
        if file:
            from kbvc.commands.commit import _path_to_ko_id
            from kbvc.core.chunker import parse_frontmatter
            target_path = Path(file)
            if target_path.exists():
                fm, _ = parse_frontmatter(target_path.read_text(encoding="utf-8"))
                target_id = _path_to_ko_id(target_path, fm)
            else:
                target_id = target_path.stem
            if target_id not in c.changed_kos:
                continue

        if oneline:
            click.echo(f"{c.display_id}  {c.message}")
        else:
            click.echo(f"commit {c.display_id}  ({c.branch})")
            click.echo(f"  {c.timestamp}  {c.message}")
            for ko_id, ch in c.changed_kos.items():
                click.echo(f"    {ko_id}@{ch.from_version} → @{ch.to_version}"
                           + (f"  [{ch.reason}]" if ch.reason else ""))
            if c.changed_kos:
                click.echo(f"    Graph: {c.graph_snapshot}  "
                           f"Prompt: {c.prompt_snapshot}  "
                           f"Retrieval: {c.retrieval_snapshot}")
            click.echo()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Status, checkout, diff, prompt
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
def status():
    """Show staged files, unstaged changes, and graph/prompt state."""
    from kbvc.core.chunker import compute_changed_chunks, compute_deleted_chunks, parse_frontmatter, split_into_chunks
    from kbvc.core.commit import walk_dag
    from kbvc.core.index import StagingIndex
    from kbvc.core.ko import KOStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    branch = repo.current_branch()
    head = repo.head_commit()
    commits = walk_dag(repo.commits_dir, head)

    click.echo(f"Branch: {branch}  HEAD: {commits[0].display_id if commits else '(no commits)'}")
    click.echo("")

    index = StagingIndex.load(repo.index_path)
    config = repo.config()
    ko_store = KOStore(repo.ko_store_path)

    if index.staged_files:
        click.echo("Staged for next commit:")
        for f in index.staged_files:
            click.echo(f"  + {f}")
    else:
        click.echo("Nothing staged. Use: kbvc add <file>")

    if index.graph_dirty:
        click.echo("\n  Graph: dirty (kbvc link called — commit to snapshot)")
    if index.prompt_dirty:
        click.echo("\n  Prompt: dirty (kbvc prompt set called — commit to snapshot)")

    # Check for unstaged modifications
    click.echo("\nUnstaged modifications (tracked KOs):")
    found_unstaged = False
    for ko in ko_store.all():
        if ko.source_type != "file":
            continue
        src = repo.root / ko.path
        if not src.exists():
            click.echo(f"  ! deleted (on disk): {ko.path}")
            found_unstaged = True
            continue
        try:
            content = src.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content)
            chunks = split_into_chunks(
                body, fm,
                split_on=config.get("chunk.split_on", "##"),
                target_tokens=int(config.get("chunk.size", 400)),
            )
            changed = compute_changed_chunks(chunks, ko.chunk_hashes)
            deleted = compute_deleted_chunks(chunks, ko.chunk_hashes)
            if changed or deleted:
                n_ch = len(changed)
                n_del = len(deleted)
                click.echo(f"  M {ko.path}"
                           + (f"  +{n_ch} chunk(s) changed" if n_ch else "")
                           + (f"  -{n_del} chunk(s) removed" if n_del else ""))
                click.echo(f"    → Run: kbvc add {ko.path}")
                found_unstaged = True
        except Exception as exc:
            click.echo(f"  ? {ko.path}  (could not diff: {exc})")
            found_unstaged = True
    if not found_unstaged:
        click.echo("  (none)")


@main.command()
@click.argument("commit_hash")
@click.argument("file", required=False)
@click.option("--ko-version", type=int, default=None,
              help="Restore a KO to a specific version number")
def checkout(commit_hash, file, ko_version):
    """Restore KOs, graph, prompt, and retrieval config to a past commit.

    Examples:
        kbvc checkout c7f3a91
        kbvc checkout c7f3a91 -- projects/manifestai.md
        kbvc checkout projects/manifestai.md --ko-version 2
    """
    from kbvc.core.commit import CommitObject
    from kbvc.core.graph import RelationGraph
    from kbvc.core.ko import KOStore
    from kbvc.core.prompt_store import PromptVersionStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.core.versioner import KOVersioner
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    # P9: handle empty repo
    if not repo.head_commit():
        raise click.ClickException("No commits yet.")

    try:
        target_commit = CommitObject.load(repo.commits_dir, commit_hash)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc))

    branch = repo.current_branch()
    ko_store = KOStore(repo.ko_store_path)
    versioner = KOVersioner(repo.ko_versions_dir)

    if file:
        # Single-KO checkout
        from kbvc.commands.commit import _path_to_ko_id
        from kbvc.core.chunker import parse_frontmatter
        src = Path(file)
        if src.exists():
            fm, _ = parse_frontmatter(src.read_text(encoding="utf-8"))
            ko_id = _path_to_ko_id(src, fm)
        else:
            ko_id = src.stem

        if ko_id not in target_commit.changed_kos:
            raise click.ClickException(
                f"KO '{ko_id}' was not changed in commit {target_commit.display_id}"
            )
        change = target_commit.changed_kos[ko_id]
        snap = versioner.load_version(ko_id, change.to_version)
        # Update ko_store to point to this version's vector_ids
        ko = ko_store.get(ko_id)
        if ko:
            from dataclasses import replace
            ko_store.update(replace(ko, vector_ids=snap.vector_ids, version=snap.version))
        click.echo(f"Restored {ko_id}@{snap.version} (from commit {target_commit.display_id})")
        return

    # Full checkout — restore everything
    if ko_version is not None:
        raise click.ClickException(
            "--ko-version requires specifying a file: kbvc checkout <file> --ko-version N"
        )

    # Restore all KOs changed by this commit
    for ko_id, change in target_commit.changed_kos.items():
        snap = versioner.load_version(ko_id, change.to_version)
        ko = ko_store.get(ko_id)
        if ko:
            from dataclasses import replace
            ko_store.update(replace(ko, vector_ids=snap.vector_ids, version=snap.version))
            click.echo(f"  Restored {ko_id}@{snap.version}")

    # Restore graph
    graph = RelationGraph(repo.graph_dir, branch)
    graph.restore_snapshot(target_commit.graph_snapshot)
    click.echo(f"  Graph: {target_commit.graph_snapshot}")

    # Restore prompt
    prompt_store = PromptVersionStore(repo.prompts_dir)
    prompt_store.restore(target_commit.prompt_snapshot)
    click.echo(f"  Prompt: {target_commit.prompt_snapshot}")

    click.echo(f"\nChecked out commit {target_commit.display_id}: {target_commit.message}")
    click.echo("Note: vectors from this commit are still live in the DB (no re-embedding needed).")
    click.echo("Run 'kbvc commit -m \"rollback\"' to record this as a new commit.")


@main.command(name="diff")
@click.argument("path_or_commit_a", required=False)
@click.argument("commit_b", required=False)
@click.argument("file", required=False)
def diff_cmd(path_or_commit_a, commit_b, file):
    """Show chunk-level diff between current state and last commit, or two commits."""
    from kbvc.core.chunker import parse_frontmatter, split_into_chunks, compute_changed_chunks, compute_deleted_chunks
    from kbvc.core.commit import CommitObject, walk_dag
    from kbvc.core.ko import KOStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    config = repo.config()
    ko_store = KOStore(repo.ko_store_path)

    if path_or_commit_a and not commit_b:
        # kbvc diff <file> — current vs last commit
        src = Path(path_or_commit_a)
        if src.exists():
            from kbvc.commands.commit import _path_to_ko_id
            content = src.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content)
            ko_id = _path_to_ko_id(src, fm)
            new_chunks = split_into_chunks(body, fm,
                split_on=config.get("chunk.split_on", "##"),
                target_tokens=int(config.get("chunk.size", 400)))
            ko = ko_store.get(ko_id)
            stored = ko.chunk_hashes if ko else []
            changed = compute_changed_chunks(new_chunks, stored)
            deleted = compute_deleted_chunks(new_chunks, stored)
            if not changed and not deleted:
                click.echo(f"{path_or_commit_a}: no changes")
            else:
                click.echo(f"{path_or_commit_a}  ({ko_id})")
                for idx in changed:
                    click.echo(f"  ~ chunk[{idx}] section='{new_chunks[idx].section}'")
                for idx in deleted:
                    click.echo(f"  - chunk[{idx}] (deleted)")
        else:
            click.echo(f"File not found: {path_or_commit_a}")
    else:
        click.echo("Use: kbvc diff <file>  to see changes vs last commit")


# ── prompt group ──────────────────────────────────────────────────────────────

@main.group()
def prompt():
    """Version and manage retrieval prompts."""
    pass


@prompt.command(name="set")
@click.argument("text")
@click.option("--system", default="", help="Optional system prompt")
@click.option("--notes", default="", help="Change notes")
def prompt_set(text, system, notes):
    """Set a new retrieval prompt (staged for next commit)."""
    from kbvc.core.index import StagingIndex
    from kbvc.core.prompt_store import PromptVersionStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    store = PromptVersionStore(repo.prompts_dir)
    pv = store.set(text, system_prompt=system, notes=notes)
    # Mark prompt dirty in the staging index
    index = StagingIndex.load(repo.index_path)
    index.mark_prompt_dirty()
    index.save(repo.index_path)
    click.echo(f"Prompt set (v{pv.version}, pending commit).")


@prompt.command(name="get")
def prompt_get():
    """Show the current prompt."""
    from kbvc.core.prompt_store import PromptVersionStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    store = PromptVersionStore(repo.prompts_dir)
    pv = store.get_current()
    if pv is None:
        click.echo("No prompt set. Use: kbvc prompt set \"<text>\"")
        return
    click.echo(f"Version:  {pv.version}")
    click.echo(f"Retrieval: {pv.retrieval_prompt}")
    if pv.system_prompt:
        click.echo(f"System:   {pv.system_prompt}")


@prompt.command(name="log")
def prompt_log():
    """Show prompt version history."""
    from kbvc.core.prompt_store import PromptVersionStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    store = PromptVersionStore(repo.prompts_dir)
    versions = store.log()
    if not versions:
        click.echo("No prompt snapshots yet.")
        return
    for pv in reversed(versions):
        cid = pv.commit_id[:7] if pv.commit_id else "pending"
        click.echo(f"p-v{pv.version}  {cid}  {pv.created_at[:10]}  "
                   f"{pv.retrieval_prompt[:60]}...")


@prompt.command(name="checkout")
@click.argument("snapshot_name")
def prompt_checkout(snapshot_name):
    """Restore a prior prompt version (e.g. p-v2)."""
    from kbvc.core.prompt_store import PromptVersionStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    store = PromptVersionStore(repo.prompts_dir)
    try:
        store.restore(snapshot_name)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Restored prompt {snapshot_name}. Run 'kbvc commit -m \"..\"' to record.")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Relations, graph, query
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.argument("file_a")
@click.argument("file_b")
@click.option("--type", "rel_type", required=True,
              help="Relation type (e.g. informed_by, created_at, part_of)")
@click.option("--note", default="", help="Human-readable description")
@click.option("--valid-from", default=None, help="Validity start date (YYYY-MM-DD)")
@click.option("--valid-to", default=None, help="Validity end date (YYYY-MM-DD)")
def link(file_a, file_b, rel_type, note, valid_from, valid_to):
    """Add a relation between two KOs (no re-embedding required)."""
    from kbvc.core.chunker import parse_frontmatter
    from kbvc.core.graph import RelationGraph
    from kbvc.core.index import StagingIndex
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.commit import _path_to_ko_id
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    branch = repo.current_branch()

    def resolve_id(path_str: str) -> str:
        p = Path(path_str)
        if p.exists():
            fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
            return _path_to_ko_id(p, fm)
        return p.stem

    from_id = resolve_id(file_a)
    to_id = resolve_id(file_b)

    graph = RelationGraph(repo.graph_dir, branch)
    rel = graph.add(
        from_id=from_id,
        to_id=to_id,
        rel_type=rel_type,
        note=note,
        valid_from=valid_from,
        valid_to=valid_to,
    )

    # Mark graph dirty in staging index
    index = StagingIndex.load(repo.index_path)
    index.mark_graph_dirty()
    index.save(repo.index_path)

    click.echo(f"Added relation: {from_id} --[{rel_type}]--> {to_id}  (id: {rel.id})")
    click.echo("Run: kbvc commit -m \"...\" to snapshot the graph.")


@main.command()
@click.argument("rel_id")
def unlink(rel_id):
    """Remove a relation by its ID."""
    from kbvc.core.graph import RelationGraph
    from kbvc.core.index import StagingIndex
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    branch = repo.current_branch()
    graph = RelationGraph(repo.graph_dir, branch)
    if graph.remove(rel_id):
        index = StagingIndex.load(repo.index_path)
        index.mark_graph_dirty()
        index.save(repo.index_path)
        click.echo(f"Removed relation: {rel_id}")
    else:
        click.echo(f"Relation not found: {rel_id}")


@main.command(name="graph")
@click.argument("file", required=False)
@click.option("--depth", default=1, show_default=True, help="Traversal depth")
@click.option("--type", "rel_type", default=None, help="Filter by relation type")
@click.option("--all", "show_all", is_flag=True, help="Print the full graph")
def graph_cmd(file, depth, rel_type, show_all):
    """Show the knowledge relation graph."""
    from kbvc.core.graph import RelationGraph
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.commit import _path_to_ko_id
    from kbvc.core.chunker import parse_frontmatter
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    branch = repo.current_branch()
    graph = RelationGraph(repo.graph_dir, branch)

    if show_all:
        if not graph.relations:
            click.echo("No relations yet. Use: kbvc link <file_a> <file_b> --type <type>")
            return
        for r in graph.relations:
            click.echo(f"  {r.from_id}  --[{r.type}]-->  {r.to_id}  "
                       + (f'"{r.note}"' if r.note else ""))
        return

    if not file:
        click.echo("Specify a file or use --all. Example: kbvc graph projects/manifestai.md")
        return

    src = Path(file)
    if src.exists():
        fm, _ = parse_frontmatter(src.read_text(encoding="utf-8"))
        ko_id = _path_to_ko_id(src, fm)
    else:
        ko_id = src.stem

    neighbors = graph.neighbors(ko_id, depth=depth, rel_type=rel_type)
    if not neighbors:
        click.echo(f"No relations found for '{ko_id}' (depth={depth})")
        return

    click.echo(f"{file}  ({ko_id})")
    for n in neighbors:
        r = n["relation"]
        direction_arrow = "→" if n["direction"] == "outgoing" else "←"
        other = r["to_id"] if n["direction"] == "outgoing" else r["from_id"]
        click.echo(f"  {direction_arrow} {other}  [{r['type']}]"
                   + (f'  "{r["note"]}"' if r.get("note") else ""))


@main.command()
@click.argument("query_text")
@click.option("--top-k", default=5, show_default=True, help="Number of results")
@click.option("--profile", default=None, type=click.Choice(["vector", "graphrag"]),
              help="Override retrieval profile for this query")
def query(query_text, top_k, profile):
    """Embed a query and retrieve the most relevant KO chunks."""
    from kbvc.backends import get_embed_backend, get_vectordb_backend
    from kbvc.core.graph import RelationGraph
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    config = repo.config()
    active_profile = profile or config.get("retrieval.profile", "vector")
    collection = config.get("vectordb.collection", "kbvc")

    try:
        embed = get_embed_backend(config)
        vdb = get_vectordb_backend(config)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        query_vec = embed.embed(query_text)
    except Exception as exc:
        raise click.ClickException(f"Embedding failed: {exc}")

    try:
        results = vdb.query(collection, query_vec, top_k=top_k)
    except Exception as exc:
        raise click.ClickException(f"Query failed: {exc}")

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Query: {query_text!r}  (profile={active_profile}, top_k={top_k})")
    click.echo("")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        ko_id = meta.get("ko_id", r.get("id", "unknown"))
        section = meta.get("section", "")
        score = r.get("score", 0.0)
        click.echo(f"  {i}. {ko_id}  [{section}]  score={score:.4f}")

    # GraphRAG expansion
    if active_profile == "graphrag" and results:
        branch = repo.current_branch()
        graph = RelationGraph(repo.graph_dir, branch)
        hop_depth = int(config.get("retrieval.hop_depth", 2))
        seen_ko_ids = {r.get("metadata", {}).get("ko_id") for r in results}
        click.echo(f"\nGraphRAG expansion (depth={hop_depth}):")
        expanded = set()
        for ko_id in list(seen_ko_ids):
            if ko_id:
                for n in graph.neighbors(ko_id, depth=hop_depth):
                    r_data = n["relation"]
                    other = r_data["to_id"] if n["direction"] == "outgoing" else r_data["from_id"]
                    if other not in seen_ko_ids and other not in expanded:
                        click.echo(f"  + {other}  (via [{r_data['type']}] from {ko_id})")
                        expanded.add(other)
        if not expanded:
            click.echo("  (no additional KOs via graph traversal)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Branches, dependencies, annotations, lineage, doctor
# ═══════════════════════════════════════════════════════════════════════════════

@main.group()
def branch():
    """Manage knowledge branches."""
    pass


@branch.command(name="create")
@click.argument("name")
def branch_create(name):
    """Create a new branch (copies current ko_store + graph)."""
    import shutil
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ref_path = repo.kbvc_dir / "refs" / "heads" / name
    if ref_path.exists():
        raise click.ClickException(f"Branch already exists: {name}")

    # Point new branch at current HEAD
    current_head = repo.head_commit() or ""
    ref_path.write_text(current_head, encoding="utf-8")
    click.echo(f"Created branch '{name}' from {current_head[:7] or '(empty)'}")
    click.echo(f"Run: kbvc branch switch {name}")


@branch.command(name="switch")
@click.argument("name")
def branch_switch(name):
    """Switch to a different branch."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ref_path = repo.kbvc_dir / "refs" / "heads" / name
    if not ref_path.exists():
        raise click.ClickException(
            f"Branch not found: '{name}'. "
            f"Create it first: kbvc branch create {name}"
        )

    (repo.kbvc_dir / "HEAD").write_text(f"ref: refs/heads/{name}\n", encoding="utf-8")
    head = ref_path.read_text(encoding="utf-8").strip()
    click.echo(f"Switched to branch '{name}' (HEAD: {head[:7] or 'empty'})")


@branch.command(name="list")
def branch_list():
    """List all branches."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    current = repo.current_branch()
    heads_dir = repo.kbvc_dir / "refs" / "heads"
    branches = sorted(heads_dir.iterdir()) if heads_dir.exists() else []
    for b in branches:
        prefix = "* " if b.name == current else "  "
        commit = b.read_text(encoding="utf-8").strip()
        click.echo(f"{prefix}{b.name}  {commit[:7] if commit else '(empty)'}")


@branch.command(name="show")
def branch_show():
    """Show current branch and HEAD commit."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    b = repo.current_branch()
    h = repo.head_commit()
    click.echo(f"Branch: {b}  HEAD: {h[:7] if h else '(no commits)'}")


@branch.command(name="delete")
@click.argument("name")
def branch_delete(name):
    """Delete a branch (does not delete vectors — use kbvc prune)."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    if repo.current_branch() == name:
        raise click.ClickException("Cannot delete the current branch.")
    ref_path = repo.kbvc_dir / "refs" / "heads" / name
    if not ref_path.exists():
        raise click.ClickException(f"Branch not found: {name}")
    ref_path.unlink()
    click.echo(f"Deleted branch '{name}'. Vectors remain in the DB (run kbvc gc when available).")


# ── depends group ─────────────────────────────────────────────────────────────

@main.group()
def depends():
    """Manage KO semantic dependencies (cache-invalidation graph)."""
    pass


@depends.command(name="add")
@click.argument("file")
@click.argument("dep_file")
def depends_add(file, dep_file):
    """Declare that <file> semantically depends on <dep_file>."""
    from dataclasses import replace
    from kbvc.core.chunker import parse_frontmatter
    from kbvc.core.ko import KOStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.commit import _path_to_ko_id
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)

    def resolve(path_str: str) -> str:
        p = Path(path_str)
        if p.exists():
            fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
            return _path_to_ko_id(p, fm)
        return p.stem

    ko_id = resolve(file)
    dep_id = resolve(dep_file)

    ko = ko_store.get(ko_id)
    if not ko:
        raise click.ClickException(f"KO not found: {ko_id}. Stage and commit it first.")

    if dep_id in (ko.depends_on or []):
        click.echo(f"  Already declared: {ko_id} depends on {dep_id}")
        return

    updated = replace(ko, depends_on=list(ko.depends_on or []) + [dep_id])
    ko_store.update(updated)
    click.echo(f"  {ko_id} now depends on {dep_id}")


@depends.command(name="remove")
@click.argument("file")
@click.argument("dep_file")
def depends_remove(file, dep_file):
    """Remove a dependency declaration."""
    from dataclasses import replace
    from kbvc.core.chunker import parse_frontmatter
    from kbvc.core.ko import KOStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.commit import _path_to_ko_id
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)

    def resolve(path_str: str) -> str:
        p = Path(path_str)
        if p.exists():
            fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
            return _path_to_ko_id(p, fm)
        return p.stem

    ko_id = resolve(file)
    dep_id = resolve(dep_file)
    ko = ko_store.get(ko_id)
    if not ko:
        raise click.ClickException(f"KO not found: {ko_id}")
    new_deps = [d for d in (ko.depends_on or []) if d != dep_id]
    ko_store.update(replace(ko, depends_on=new_deps))
    click.echo(f"  Removed dependency: {ko_id} no longer depends on {dep_id}")


@depends.command(name="list")
@click.argument("file", required=False)
def depends_list(file):
    """List dependencies for a KO (or all KOs)."""
    from kbvc.core.chunker import parse_frontmatter
    from kbvc.core.ko import KOStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.commit import _path_to_ko_id
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)

    if file:
        p = Path(file)
        if p.exists():
            fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
            ko_id = _path_to_ko_id(p, fm)
        else:
            ko_id = p.stem
        ko = ko_store.get(ko_id)
        if not ko:
            raise click.ClickException(f"KO not found: {ko_id}")
        click.echo(f"{ko_id} depends on:")
        for d in (ko.depends_on or []):
            click.echo(f"  - {d}")
    else:
        for ko in ko_store.all():
            if ko.depends_on:
                click.echo(f"{ko.id}:")
                for d in ko.depends_on:
                    click.echo(f"  - {d}")


@main.command()
@click.argument("file")
@click.option("--depth", default=1, show_default=True, help="Transitive depth")
def impact(file, depth):
    """Show which KOs depend on this file (reverse dependency lookup)."""
    from kbvc.core.chunker import parse_frontmatter
    from kbvc.core.ko import KOStore
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.commit import _path_to_ko_id
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)
    p = Path(file)
    if p.exists():
        fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
        target_id = _path_to_ko_id(p, fm)
    else:
        target_id = p.stem

    # BFS over depends_on
    affected: dict[str, str] = {}  # ko_id → "direct" | "depth N"
    frontier = {target_id}
    for level in range(1, depth + 1):
        next_frontier: set[str] = set()
        for ko in ko_store.all():
            if ko.id in affected or ko.id == target_id:
                continue
            if any(dep in frontier for dep in (ko.depends_on or [])):
                label = "direct" if level == 1 else f"depth {level}"
                affected[ko.id] = label
                next_frontier.add(ko.id)
        frontier = next_frontier

    if not affected:
        click.echo(f"No KOs depend on '{target_id}'")
        return

    click.echo(f"Affected KOs if '{target_id}' changes:")
    for ko_id, label in affected.items():
        click.echo(f"  - {ko_id}  ({label})")
    click.echo("\nRun: kbvc add <file> && kbvc commit -m \"...\" to re-sync stale KOs.")


@main.command()
@click.argument("file")
@click.option("--reason", required=True, help="Human-readable reason for this change")
def annotate(file, reason):
    """Annotate a staged KO with a change reason for the next commit."""
    from kbvc.core.chunker import parse_frontmatter
    from kbvc.core.index import StagingIndex
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.commit import _path_to_ko_id
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    p = Path(file)
    if p.exists():
        fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
        ko_id = _path_to_ko_id(p, fm)
    else:
        ko_id = p.stem

    index = StagingIndex.load(repo.index_path)
    index.set_reason(ko_id, reason)
    index.save(repo.index_path)
    click.echo(f"Annotated {ko_id}: {reason!r}")


@main.command()
@click.argument("file")
@click.option("--oneline", is_flag=True, help="Compact one-line per version")
def history(file, oneline):
    """Show per-KO version history with change reasons."""
    from kbvc.core.chunker import parse_frontmatter
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.core.versioner import KOVersioner
    from kbvc.commands.commit import _path_to_ko_id
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    p = Path(file)
    if p.exists():
        fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
        ko_id = _path_to_ko_id(p, fm)
    else:
        ko_id = p.stem

    versioner = KOVersioner(repo.ko_versions_dir)
    versions = versioner.all_versions(ko_id)

    if not versions:
        click.echo(f"No version history for '{ko_id}'.")
        return

    click.echo(f"{file}  ({ko_id})")
    for v in versions:
        cid = v.commit_id[:7] if v.commit_id else "unknown"
        if oneline:
            reason_str = f"  [{v.reason}]" if v.reason else ""
            click.echo(f"  v{v.version}  {cid}{reason_str}")
        else:
            click.echo(f"  v{v.version}  commit {cid}")
            if v.reason:
                click.echo(f"         reason: {v.reason}")
            click.echo(f"         chunks changed: {v.changed_chunks}")


@main.command()
@click.argument("vector_id")
def trace(vector_id):
    """Trace a vector ID back to its origin (full lineage audit trail)."""
    from kbvc.core.lineage import ChainTracer
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    tracer = ChainTracer(repo.kbvc_dir)
    try:
        record = tracer.trace(vector_id)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc))

    click.echo(f"Chunk        {record.vector_id}")
    click.echo(f"KO           {record.ko_id}  (version {record.ko_version})")
    click.echo(f"Section      {record.section!r}")
    click.echo(f"Committed    {record.commit_id[:7]}  {record.commit_timestamp[:10]}  "
               f"{record.commit_message!r}")
    click.echo(f"Embedded     {record.embed_model}  ({record.embed_dims} dims)")
    click.echo(f"Vector DB    {record.vectordb_backend}")


@main.command()
@click.option("--knowledge", is_flag=True,
              help="Run extended knowledge health checks (stale, orphans, missing files)")
def doctor(knowledge):
    """Diagnose repository health (reads only — no writes).

    \b
    Basic checks (always run):
      repo structure, git, backends, KO store, commit history, lock file

    \b
    Extended checks (--knowledge):
      stale KOs, orphan relations, missing source files, frozen KO count,
      KOs with no chunks embedded, entity extraction coverage
    """
    from kbvc.core.ko import KOStore
    from kbvc.core.repo import KbvcRepo

    # doctor does NOT raise NotKBVCRepositoryError — it diagnoses instead
    click.echo("KBVC Doctor\n")

    try:
        repo = KbvcRepo.require()
        info = repo.repo_info()
        click.echo(f"✓  KBVC Repository  (format v{info['format_version']}, "
                   f"repo_id {info['repo_id'][:8]}...)")
    except Exception as exc:
        click.echo(f"✗  KBVC Repository — {exc}")
        click.echo("   Run: kbvc init")
        return

    # Git
    import subprocess
    try:
        # --git-dir works even on a brand-new repo with zero commits.
        # --abbrev-ref HEAD fails with exit 128 when there are no commits yet.
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repo.root,
            capture_output=True, text=True, check=True,
        )
        # Git repo exists — now try to get the branch (may be unborn on empty repo)
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo.root,
            capture_output=True, text=True,
        )
        if branch_result.returncode == 0:
            git_branch = branch_result.stdout.strip()
            click.echo(f"✓  Git Repository  (branch: {git_branch})")
        else:
            # Repo initialised but no commits yet — HEAD is unborn, that's fine
            click.echo("✓  Git Repository  (initialised, no commits yet)")
    except FileNotFoundError:
        click.echo("✗  Git Repository — git not found on PATH")
    except Exception:
        click.echo("✗  Git Repository — not initialised (run: git init)")

    # Config
    config = repo.config()
    embed_backend = config.get("embed.backend", "")
    vdb_backend = config.get("vectordb.backend", "")
    click.echo(f"{'✓' if embed_backend else '✗'}  Embed backend    {embed_backend or '(not configured)'}")
    click.echo(f"{'✓' if vdb_backend else '✗'}  VectorDB backend {vdb_backend or '(not configured)'}")

    # KO store
    ko_store = KOStore(repo.ko_store_path)
    click.echo(f"✓  KO store         {len(ko_store)} KO(s) tracked")

    # Commits
    from kbvc.core.commit import walk_dag
    commits = walk_dag(repo.commits_dir, repo.head_commit())
    click.echo(f"✓  Commit history   {len(commits)} commit(s)")

    # Lock file
    if repo.lock_path.exists():
        click.echo("✓  kbvc.lock        present")
    else:
        click.echo("✗  kbvc.lock        missing (run kbvc commit to generate)")

    # Quick stale summary (always shown)
    stale_ids = []
    for ko in ko_store.all():
        for dep_id in (ko.depends_on or []):
            dep = ko_store.get(dep_id)
            if dep and dep.last_updated > ko.last_updated:
                stale_ids.append(ko.id)
                break
    if stale_ids:
        click.echo(f"⚠  Stale KOs        {len(stale_ids)}  "
                   f"(run: kbvc stale for details)")
    else:
        click.echo("✓  Stale KOs         0")

    # Remotes
    heads_dir = repo.kbvc_dir / "refs" / "remotes"
    remote_count = len(list(heads_dir.iterdir())) if heads_dir.exists() else 0
    click.echo(f"✓  Remotes           {remote_count}  "
               + ("(run: kbvc remote list)" if remote_count else "(none configured — kbvc push unavailable)"))

    # ── Extended --knowledge checks ───────────────────────────────────────────
    if not knowledge:
        click.echo("\nRun with --knowledge for extended health checks.")
        return

    from kbvc.core.graph import RelationGraph
    from kbvc.commands.stale import compute_staleness

    click.echo("\n─── Knowledge Health ───────────────────────────────")

    branch = repo.current_branch()
    graph = RelationGraph(repo.graph_dir, branch)
    report = compute_staleness(ko_store, graph)

    # Stale detail
    if report.stale:
        click.echo(f"⚠  Stale KOs ({len(report.stale)}):")
        for e in report.stale:
            click.echo(f"     {e.ko_id}  depends on: "
                       + ", ".join(d['dep_id'] for d in e.stale_deps))
        click.echo("   Fix: kbvc stale --fix && kbvc commit -m \"refresh\"")
    else:
        click.echo("✓  No stale KOs")

    # Orphan relations
    if report.orphan_relations:
        click.echo(f"⚠  Orphan relations ({len(report.orphan_relations)}):")
        for o in report.orphan_relations:
            click.echo(f"     {o.rel_id}: {o.from_id} → {o.to_id}  (missing: {o.missing_side})")
        click.echo("   Fix: kbvc unlink <rel_id>  or  kbvc gc (v2)")
    else:
        click.echo("✓  No orphan relations")

    # Missing source files
    missing_files = []
    for ko in ko_store.all():
        if ko.source_type == "file":
            src = repo.root / ko.path
            if not src.exists():
                missing_files.append(ko.id)
    if missing_files:
        click.echo(f"⚠  Missing source files ({len(missing_files)}):")
        for kid in missing_files:
            click.echo(f"     {kid}")
        click.echo("   Fix: restore the file and re-commit, "
                   "or remove the KO from ko_store.json")
    else:
        click.echo("✓  All source files present")

    # Frozen KOs
    frozen = report.frozen
    if frozen:
        click.echo(f"❄  Frozen KOs ({len(frozen)}):  "
                   + ", ".join(e.ko_id for e in frozen))
    else:
        click.echo("✓  No frozen KOs")

    # KOs with no chunks (never embedded or zero-length)
    unchunked = [ko.id for ko in ko_store.all() if not ko.chunk_hashes]
    if unchunked:
        click.echo(f"⚠  KOs with no chunks ({len(unchunked)}):")
        for kid in unchunked:
            click.echo(f"     {kid}  (re-add and commit)")
    else:
        click.echo("✓  All KOs have chunks")

    # Entity coverage
    total_kos = len(ko_store.all())
    kos_with_entities = sum(1 for ko in ko_store.all() if ko.entities)
    pct = (kos_with_entities / total_kos * 100) if total_kos else 0
    click.echo(f"{'✓' if pct > 50 else '·'}  Entity coverage  "
               f"{kos_with_entities}/{total_kos} KOs ({pct:.0f}%)  "
               f"{'(run: kbvc extract --all)' if pct < 50 else ''}")

    # Relations per KO ratio
    rel_per_ko = (len(graph.relations) / total_kos) if total_kos else 0
    click.echo(f"{'✓' if rel_per_ko >= 0.5 else '·'}  Relation density "
               f"{rel_per_ko:.1f} relations/KO  "
               f"{'(run: kbvc analyze)' if rel_per_ko < 0.5 and total_kos > 1 else ''}")

    # Branch divergence
    heads_dir = repo.kbvc_dir / "refs" / "heads"
    branches = list(heads_dir.iterdir()) if heads_dir.exists() else []
    if len(branches) > 1:
        click.echo(f"·  {len(branches)} branches — consider merging or pruning stale branches")

    click.echo("─────────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7 — kbvc gc
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--dry-run", is_flag=True, help="Preview without deleting")
@click.option("--snapshots", is_flag=True,
              help="Also prune unreachable graph/prompt/retrieval snapshot files")
def gc(dry_run, snapshots):
    """Garbage-collect orphaned vectors and unreachable snapshots.

    \b
    1. Compares live vector IDs (from KO store) against all vectors
       present in the configured local vector backend.
    2. Deletes vectors that are no longer referenced by any KO.
    3. Optionally prunes graph/prompt/retrieval snapshot files that
       are not reachable from any commit in the current branch (--snapshots).

    \b
    Example:
        kbvc gc              # remove orphaned vectors
        kbvc gc --dry-run    # preview what would be deleted
        kbvc gc --snapshots  # also prune unreachable snapshots
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.gc import run_gc
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        run_gc(repo, dry_run=dry_run, prune_snapshots=snapshots)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc ingest — separate download pipeline (website/github/pdf/notion)
# ═══════════════════════════════════════════════════════════════════════════════

@main.group()
def ingest():
    """Pull external knowledge into the repo as KO Markdown files.

    \b
    Ingest is SEPARATE from commit and push:
        kbvc ingest website https://...   # download → .md file
        kbvc add ingested/                 # stage the .md files
        kbvc commit -m "ingest docs"       # embed + commit locally
        kbvc push                          # push vectors to remote

    This lets you review, edit, or reject ingested content before
    it ever touches your vector database.
    """
    pass


@ingest.command(name="website")
@click.argument("url")
@click.option("--output-dir", default="ingested/web", show_default=True)
@click.option("--force", is_flag=True, help="Overwrite existing file")
def ingest_website_cmd(url, output_dir, force):
    """Fetch a web page and convert it to a versioned KO Markdown file."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.ingest import ingest_website
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    out = repo.root / output_dir
    out.mkdir(parents=True, exist_ok=True)
    click.echo(f"Fetching {url} ...")
    try:
        result = ingest_website(url, out, force=force)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    for f in result.output_files:
        click.echo(f"  ✓ {f.relative_to(repo.root)}")
    for w in result.warnings:
        click.echo(f"  ⚠ {w}")
    click.echo(f"\n{result.bytes_fetched:,} bytes fetched.")
    click.echo(f"Review and commit:\n"
               f"  kbvc add {output_dir}/\n"
               f"  kbvc commit -m \"ingest: {url[:50]}\"")


@ingest.command(name="github")
@click.argument("repo_url")
@click.option("--branch", default="main", show_default=True)
@click.option("--output-dir", default="ingested/github", show_default=True)
@click.option("--force", is_flag=True)
def ingest_github_cmd(repo_url, branch, output_dir, force):
    """Clone a GitHub repo and import its Markdown documentation files."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.ingest import ingest_github
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    out = repo.root / output_dir
    out.mkdir(parents=True, exist_ok=True)
    click.echo(f"Cloning {repo_url} (branch: {branch}) ...")
    try:
        result = ingest_github(repo_url, out, branch=branch, force=force)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"  {len(result.output_files)} file(s)  "
               f"{result.bytes_fetched:,} bytes")
    for w in result.warnings:
        click.echo(f"  ⚠ {w}")
    click.echo(f"\nReview and commit:\n"
               f"  kbvc add {output_dir}/\n"
               f"  kbvc commit -m \"import: {repo_url}\"")


@ingest.command(name="pdf")
@click.argument("pdf_path")
@click.option("--output-dir", default="ingested/pdf", show_default=True)
@click.option("--force", is_flag=True)
def ingest_pdf_cmd(pdf_path, output_dir, force):
    """Convert a PDF to a KO Markdown file.

    Requires: pip install pypdf
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.ingest import ingest_pdf
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    src = Path(pdf_path)
    if not src.exists():
        raise click.ClickException(f"PDF not found: {pdf_path}")
    out = repo.root / output_dir
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = ingest_pdf(src, out, force=force)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    for f in result.output_files:
        click.echo(f"  ✓ {f.relative_to(repo.root)}")
    for w in result.warnings:
        click.echo(f"  ⚠ {w}")
    click.echo(f"\nReview and commit:\n"
               f"  kbvc add {output_dir}/\n"
               f"  kbvc commit -m \"ingest: {src.name}\"")


@ingest.command(name="notion")
@click.argument("page_id")
@click.option("--token", envvar="NOTION_TOKEN",
              help="Notion API token (or set NOTION_TOKEN env var)")
@click.option("--output-dir", default="ingested/notion", show_default=True)
@click.option("--force", is_flag=True)
def ingest_notion_cmd(page_id, token, output_dir, force):
    """Import a Notion page as a KO Markdown file.

    Requires: pip install notion-client
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.ingest import ingest_notion
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    if not token:
        raise click.ClickException(
            "Notion API token required.\n"
            "  export NOTION_TOKEN=secret_..."
        )
    out = repo.root / output_dir
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = ingest_notion(page_id, out, notion_token=token, force=force)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    for f in result.output_files:
        click.echo(f"  ✓ {f.relative_to(repo.root)}")
    for w in result.warnings:
        click.echo(f"  ⚠ {w}")
    click.echo(f"\nReview and commit:\n"
               f"  kbvc add {output_dir}/\n"
               f"  kbvc commit -m \"import notion {page_id[:8]}\"")


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc push — push vectors to remote (separate from kbvc commit)
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.argument("remote_name", default="origin", metavar="REMOTE")
@click.option("--collection", default=None, help="Override collection name for this push")
@click.option("--dry-run", is_flag=True)
def push(remote_name, collection, dry_run):
    """Push committed vectors to a remote vector DB.

    \b
    kbvc commit  embeds + writes to your LOCAL vector DB.
    kbvc push    syncs committed knowledge to a REMOTE target.

    \b
    First configure a remote:
        kbvc remote add origin --backend qdrant \\
            --url https://prod.qdrant.io --collection kbvc-prod

    Then push:
        kbvc push             # push HEAD to origin
        kbvc push staging     # push to staging remote
        kbvc push --dry-run   # preview without writing
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.push import run_push
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        run_push(repo, remote_name=remote_name,
                 collection_override=collection, dry_run=dry_run)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Push failed: {exc}")


# ── remote management ─────────────────────────────────────────────────────────

@main.group()
def remote():
    """Manage remote vector DB targets (for kbvc push)."""
    pass


@remote.command(name="add")
@click.argument("name")
@click.option("--backend", required=True,
              type=click.Choice(["qdrant", "pgvector", "pinecone", "chroma"]))
@click.option("--url", required=True)
@click.option("--key", default="")
@click.option("--collection", default="kbvc", show_default=True)
def remote_add(name, backend, url, key, collection):
    """Register a remote vector DB for kbvc push."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.utils.config import write_config_key
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    cfg = repo.kbvc_dir / "config"
    write_config_key(cfg, f"remote.{name}.backend", backend)
    write_config_key(cfg, f"remote.{name}.url", url)
    write_config_key(cfg, f"remote.{name}.collection", collection)
    if key:
        secrets = repo.kbvc_dir / "secrets"
        with open(secrets, "a", encoding="utf-8") as f:
            f.write(f"[remote.{name}]\n  key = {key}\n")
        click.echo("  ⚠ API key written to .kbvc/secrets (gitignored)")
    click.echo(f"Remote '{name}' configured: {backend} → {url}/{collection}")
    click.echo(f"Push with: kbvc push {name}")


@remote.command(name="list")
def remote_list():
    """List configured remotes."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    cfg = repo.config()
    # Keys are stored as "remote.NAME.KEY" in the flat dict
    remotes: dict = {}
    for k, v in cfg.items():
        if k.startswith("remote."):
            rest = k[len("remote."):]          # e.g. "tmp.backend"
            dot = rest.find(".")
            if dot != -1:
                rname = rest[:dot]             # "tmp"
                rkey = rest[dot + 1:]          # "backend"
                remotes.setdefault(rname, {})[rkey] = v
    if not remotes:
        click.echo("No remotes. Run: kbvc remote add origin --backend qdrant --url ...")
        return
    for rname, rdata in remotes.items():
        ref = repo.kbvc_dir / "refs" / "remotes" / rname
        last = ref.read_text().strip()[:7] if ref.exists() and ref.read_text().strip() else "never"
        click.echo(f"  {rname:<12} {rdata.get('backend','?'):<10} "
                   f"{rdata.get('url','?')} / {rdata.get('collection','?')}"
                   f"  [last push: {last}]")


@remote.command(name="remove")
@click.argument("name")
def remote_remove(name):
    """Remove a remote."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from configparser import ConfigParser
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    cfg_path = repo.kbvc_dir / "config"
    parser = ConfigParser()
    parser.read(cfg_path)
    # Keys are stored as "name.backend", "name.url" etc. under [remote] section
    prefix = f"{name}."
    removed = False
    if parser.has_section("remote"):
        keys_to_remove = [k for k in parser.options("remote") if k.startswith(prefix)]
        for k in keys_to_remove:
            parser.remove_option("remote", k)
            removed = True
    if removed:
        with open(cfg_path, "w", encoding="utf-8") as f:
            parser.write(f)
        click.echo(f"Removed remote '{name}'.")
    else:
        click.echo(f"Remote '{name}' not found.")


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc analyze — auto relation discovery
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--min-confidence", default=0.40, show_default=True, type=float,
              help="Minimum confidence score (0–1)")
@click.option("--max-suggestions", default=10, show_default=True, type=int,
              help="Maximum suggestions (prevents graph explosion)")
@click.option("--apply", "auto_apply", is_flag=True,
              help="Automatically create all suggested relations")
@click.option("--use-vectors", is_flag=True,
              help="Use vector similarity (requires configured backends)")
def analyze(min_confidence, max_suggestions, auto_apply, use_vectors):
    """Discover potential relations between KOs automatically.

    \b
    Uses two strategies:
      1. Heuristic: cross-references, shared tags, relation keywords
      2. Vector: cosine similarity via your vector DB (--use-vectors)

    Suggestions are ranked by confidence and capped at --max-suggestions
    to prevent graph explosion.

    \b
    Examples:
        kbvc analyze
        kbvc analyze --use-vectors --min-confidence 0.6
        kbvc analyze --apply    # auto-link all suggestions
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.core.ko import KOStore
    from kbvc.core.graph import RelationGraph
    from kbvc.core.index import StagingIndex
    from kbvc.core.chunker import parse_frontmatter
    from kbvc.commands.analyze import suggest_relations
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)
    branch = repo.current_branch()
    graph = RelationGraph(repo.graph_dir, branch)
    all_kos = ko_store.all()

    if len(all_kos) < 2:
        click.echo("Need at least 2 committed KOs to analyze.")
        return

    existing = {(r.from_id, r.to_id) for r in graph.relations}

    source_texts: dict[str, str] = {}
    for ko in all_kos:
        src = repo.root / ko.path
        if src.exists():
            _, body = parse_frontmatter(src.read_text(encoding="utf-8"))
            source_texts[ko.id] = body

    embed_b = vdb_b = None
    if use_vectors:
        from kbvc.backends import get_embed_backend, get_vectordb_backend
        config = repo.config()
        try:
            embed_b = get_embed_backend(config)
            vdb_b = get_vectordb_backend(config)
        except ValueError as exc:
            raise click.ClickException(str(exc))

    suggestions = suggest_relations(
        kos=all_kos,
        existing_relations=existing,
        source_texts=source_texts,
        max_suggestions=max_suggestions,
        min_confidence=min_confidence,
        embed_backend=embed_b,
        vdb_backend=vdb_b,
        collection=repo.config().get("vectordb.collection", "kbvc"),
    )

    if not suggestions:
        click.echo(f"No suggestions above confidence={min_confidence:.0%} "
                   f"({len(all_kos)} KOs analyzed).")
        return

    mode = "vector+heuristic" if use_vectors else "heuristic"
    click.echo(f"Relation suggestions [{mode}]  "
               f"min_confidence={min_confidence:.0%}:\n")
    for i, s in enumerate(suggestions, 1):
        bar = "█" * int(s.confidence * 10) + "░" * (10 - int(s.confidence * 10))
        click.echo(f"  {i:2}. {bar} {s.confidence:.0%}  "
                   f"{s.from_id} --[{s.suggested_type}]--> {s.to_id}")
        click.echo(f"       {s.reasoning}")

    if auto_apply:
        click.echo("")
        index = StagingIndex.load(repo.index_path)
        for s in suggestions:
            graph.add(s.from_id, s.to_id, s.suggested_type,
                      note=f"auto (confidence={s.confidence:.2f})")
            click.echo(f"  ✓ {s.from_id} --[{s.suggested_type}]--> {s.to_id}")
        index.mark_graph_dirty()
        index.save(repo.index_path)
        click.echo(f"\n{len(suggestions)} relation(s) applied. "
                   "Run: kbvc commit -m \"auto-link\"")
    else:
        click.echo("\nApply individually:")
        for s in suggestions[:3]:
            click.echo(f"  kbvc link {s.from_id} {s.to_id} --type {s.suggested_type}")
        if len(suggestions) > 3:
            click.echo(f"  ... ({len(suggestions) - 3} more)")
        click.echo("\nOr: kbvc analyze --apply")


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc extract — auto entity extraction
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.argument("file", required=False)
@click.option("--all", "extract_all", is_flag=True, help="Extract from all tracked KOs")
@click.option("--min-confidence", default=0.65, show_default=True, type=float)
@click.option("--apply", "auto_apply", is_flag=True,
              help="Write entities to ko_store (commit to persist)")
def extract(file, extract_all, min_confidence, auto_apply):
    """Extract named entities from KOs automatically.

    \b
    Detects: AI models, tools, institutions, concepts.

    \b
    Examples:
        kbvc extract projects/manifestai.md
        kbvc extract --all
        kbvc extract --all --min-confidence 0.8 --apply
    """
    import uuid
    from dataclasses import replace
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.core.ko import KOStore, Entity
    from kbvc.core.chunker import parse_frontmatter, split_into_chunks
    from kbvc.commands.analyze import extract_entities
    from kbvc.commands.commit import _path_to_ko_id
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)
    config = repo.config()

    targets: list[tuple[str, Path]] = []
    if extract_all:
        for ko in ko_store.all():
            if ko.source_type == "file":
                targets.append((ko.id, repo.root / ko.path))
    elif file:
        src = Path(file)
        if not src.exists():
            raise click.ClickException(f"File not found: {file}")
        fm, _ = parse_frontmatter(src.read_text(encoding="utf-8"))
        ko_id = _path_to_ko_id(src, fm)
        targets.append((ko_id, src))
    else:
        raise click.ClickException("Specify a file or use --all")

    total_found = 0
    for ko_id, src in targets:
        content = src.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content)
        chunks = split_into_chunks(
            body, fm,
            split_on=config.get("chunk.split_on", "##"),
            target_tokens=int(config.get("chunk.size", 400)),
        )
        candidates = extract_entities(ko_id, body, chunks, min_confidence=min_confidence)

        if not candidates:
            click.echo(f"  {ko_id}: no entities found")
            continue

        click.echo(f"\n  {ko_id}  ({len(candidates)} entities)")
        for c in candidates:
            bar = "█" * int(c.confidence * 10) + "░" * (10 - int(c.confidence * 10))
            click.echo(f"    {bar} {c.confidence:.0%}  "
                       f"{c.name:<28}  type={c.type}  chunk={c.chunk_index}")
        total_found += len(candidates)

        if auto_apply:
            ko = ko_store.get(ko_id)
            if ko:
                existing_names = {e.name.lower() for e in ko.entities}
                new_ents = [
                    Entity(
                        id=f"ent-{uuid.uuid4().hex[:8]}",
                        name=c.name, type=c.type,
                        ko_id=ko_id, chunk_index=c.chunk_index,
                    )
                    for c in candidates if c.name.lower() not in existing_names
                ]
                ko_store.update(replace(ko, entities=list(ko.entities) + new_ents))
                click.echo(f"    ✓ {len(new_ents)} entity/entities written")

    click.echo(f"\n{total_found} entities across {len(targets)} KO(s)")
    if auto_apply and total_found:
        click.echo("Run: kbvc commit -m \"extract entities\" to persist.")
    elif not auto_apply and total_found:
        click.echo("Add --apply to write to ko_store.")


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc stale — knowledge freshness report
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--show-fresh", is_flag=True, help="Include fresh KOs")
@click.option("--show-frozen", is_flag=True, help="Include frozen KOs")
@click.option("--fix", "auto_stage", is_flag=True,
              help="Stage all stale KOs for re-commit")
def stale(show_fresh, show_frozen, auto_stage):
    """Report which KOs are stale due to updated dependencies.

    \b
    A KO is STALE when a dependency (kbvc depends add) has been committed
    more recently than the KO itself.

    \b
    Examples:
        kbvc stale                   # show stale KOs only
        kbvc stale --show-fresh      # full freshness dashboard
        kbvc stale --fix             # stage stale KOs for re-commit
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.core.ko import KOStore
    from kbvc.core.graph import RelationGraph
    from kbvc.core.index import StagingIndex
    from kbvc.commands.stale import compute_staleness
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)
    branch = repo.current_branch()
    graph = RelationGraph(repo.graph_dir, branch)
    report = compute_staleness(ko_store, graph)

    if report.stale:
        click.echo(f"STALE  ({len(report.stale)}):\n")
        for e in report.stale:
            click.echo(f"  ✗  {e.ko_id}  [last updated: {e.ko_last_updated}]")
            for d in e.stale_deps:
                click.echo(f"       ↑ {d['dep_id']}  updated: {d['dep_updated']}")
        click.echo("")
    else:
        click.echo("✓  No stale KOs.\n")

    if report.orphan_relations:
        click.echo(f"ORPHAN RELATIONS  ({len(report.orphan_relations)}):\n")
        for o in report.orphan_relations:
            click.echo(f"  ⚠  {o.rel_id}  {o.from_id} → {o.to_id}  "
                       f"(missing: {o.missing_side})")
        click.echo(f"\n  Clean up with: kbvc unlink <rel_id>  or  kbvc gc (v2)\n")

    if show_frozen and report.frozen:
        click.echo(f"FROZEN  ({len(report.frozen)}):")
        for e in report.frozen:
            click.echo(f"  ❄  {e.ko_id}")
        click.echo("")

    if show_fresh and report.fresh:
        click.echo(f"FRESH  ({len(report.fresh)}):")
        for e in report.fresh:
            click.echo(f"  ✓  {e.ko_id}")
        click.echo("")

    click.echo(
        f"Summary: {len(report.stale)} stale  /  {len(report.fresh)} fresh  /  "
        f"{len(report.frozen)} frozen  /  {len(report.orphan_relations)} orphan relations"
    )

    if auto_stage and report.stale:
        index = StagingIndex.load(repo.index_path)
        n = 0
        for e in report.stale:
            src = repo.root / e.ko_path
            if src.exists():
                index.stage(e.ko_path)
                click.echo(f"  + staged: {e.ko_path}")
                n += 1
        index.save(repo.index_path)
        click.echo(f"\nStaged {n} KO(s). Run: kbvc commit -m \"refresh stale KOs\"")


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc stats — knowledge evolution analytics
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.option("--json-out", is_flag=True, help="Output as JSON")
def stats(json_out):
    """Show knowledge base evolution analytics.

    \b
    Includes: KO/relation/commit counts, breakdowns by type and volatility,
    most-changed and most-connected KOs, growth this month, embed cost estimate.
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.core.ko import KOStore
    from kbvc.core.graph import RelationGraph
    from kbvc.core.commit import walk_dag
    from kbvc.commands.stats import compute_stats
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    ko_store = KOStore(repo.ko_store_path)
    branch = repo.current_branch()
    graph = RelationGraph(repo.graph_dir, branch)
    commits = walk_dag(repo.commits_dir, repo.head_commit())
    report = compute_stats(ko_store, graph, commits, repo.kbvc_dir)

    if json_out:
        import json, dataclasses
        click.echo(json.dumps(dataclasses.asdict(report), indent=2))
        return

    W = 46
    click.echo(f"\n{'━' * W}")
    click.echo(f"  KBVC Knowledge Base  ·  {repo.current_branch()}")
    click.echo(f"{'━' * W}")
    click.echo(f"  KOs              {report.total_kos:>10,}")
    click.echo(f"  Relations        {report.total_relations:>10,}")
    click.echo(f"  Commits          {report.total_commits:>10,}")
    click.echo(f"  Chunks embedded  {report.total_chunks_embedded:>10,}")
    click.echo(f"  Branches         {report.branch_count:>10,}  "
               f"({', '.join(report.branch_names)})")

    if report.kos_by_type:
        click.echo(f"\n  By type:")
        for t, n in sorted(report.kos_by_type.items(), key=lambda x: -x[1]):
            bar = "▓" * min(n, 20)
            click.echo(f"    {t:<18} {n:>4}  {bar}")

    if report.kos_by_volatility:
        click.echo(f"\n  By volatility:")
        for v, n in sorted(report.kos_by_volatility.items(), key=lambda x: -x[1]):
            click.echo(f"    {v:<18} {n:>4}")

    if report.relations_by_type:
        click.echo(f"\n  Relations by type:")
        for t, n in sorted(report.relations_by_type.items(), key=lambda x: -x[1]):
            click.echo(f"    {t:<18} {n:>4}")

    click.echo(f"\n  This month:  +{report.kos_added_this_month} KOs  "
               f"+{report.commits_this_month} commits")

    if report.most_changed_ko:
        click.echo(f"\n  Most changed:    {report.most_changed_ko} "
                   f"({report.most_changed_ko_commits} commits)")
    if report.most_connected_ko:
        click.echo(f"  Most connected:  {report.most_connected_ko} "
                   f"({report.most_connected_ko_relations} relations)")
    click.echo(f"\n{'━' * W}\n")


@main.command()
@click.argument("url")
@click.argument("directory", required=False)
def clone(url, directory):
    """Clone a KBVC repository from a Git remote."""
    import subprocess
    target = directory or url.split("/")[-1].replace(".git", "")
    click.echo(f"Cloning {url} into {target}/ ...")
    try:
        subprocess.run(["git", "clone", url, target], check=True)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"git clone failed: {exc}")
    except FileNotFoundError:
        raise click.ClickException("git not found on PATH.")

    repo_json = Path(target) / ".kbvc" / "repo.json"
    if not repo_json.exists():
        raise click.ClickException(
            f"Repository cloned but is NOT a KBVC repository.\n"
            "Expected: .kbvc/repo.json\n\n"
            "This appears to be a standard Git repository.\n"
            "To use it as a KBVC repository, run:\n"
            "  kbvc init"
        )
    import json
    info = json.loads(repo_json.read_text())
    click.echo(f"\nCloned KBVC repository: {info.get('name', target)}")
    click.echo(f"  Format: v{info.get('format_version')}")

    # Read kbvc.lock for configuration hints
    lock_path = Path(target) / "kbvc.lock"
    embed_provider = embed_model = vdb_provider = collection = ""
    if lock_path.exists():
        try:
            import yaml as _yaml
            lock = _yaml.safe_load(lock_path.read_text())
            embed_provider = lock.get("embedding", {}).get("provider", "")
            embed_model = lock.get("embedding", {}).get("model", "")
            vdb_provider = lock.get("vector_store", {}).get("provider", "")
            collection = lock.get("vector_store", {}).get("collection", "kbvc")
            click.echo(f"\nLock file detected:")
            click.echo(f"  embedding:    {embed_provider} / {embed_model}")
            click.echo(f"  vector store: {vdb_provider} / {collection}")
        except Exception:
            pass

    click.echo(f"\nSuccessfully cloned KBVC repository into {target}/")
    click.echo("Next steps to restore the knowledge state:")
    click.echo(f"  cd {target}")
    ep = embed_provider or "openai  # or gemini/ollama/huggingface"
    vp = vdb_provider or "qdrant   # or pgvector/chroma/lancedb"
    click.echo(f"  kbvc config set embed.backend {ep}")
    click.echo(f"  kbvc config set embed.key <your-api-key>")
    click.echo(f"  kbvc config set vectordb.backend {vp}")
    click.echo(f"  kbvc config set vectordb.url <url>")
    click.echo(f"  kbvc backend init          # create vector DB schema")
    click.echo(f"  kbvc push                  # re-embed and push knowledge state")


@main.group()
def migrate():
    """VSAL backend migration — move knowledge between vector stores."""
    pass


@migrate.command(name="backend")
@click.option("--from", "from_backend", required=True,
              help="Source backend (qdrant | pgvector | chroma | pinecone)")
@click.option("--to", "to_backend", required=True,
              help="Target backend (qdrant | pgvector | chroma | pinecone)")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
def migrate_backend(from_backend, to_backend, dry_run):
    """Migrate all ChunkRecords from one vector backend to another.

    \b
    Flow:
        Source Backend
              ↓  export_chunks()
        List[ChunkRecord]       ← universal KBVC schema
              ↓  import_chunks()
        Target Backend

    \b
    Example:
        kbvc migrate backend --from qdrant --to pgvector
        kbvc migrate backend --from qdrant --to pgvector --dry-run
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.migrate import run_migrate
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        run_migrate(repo, from_backend, to_backend, dry_run=dry_run)
    except ValueError as exc:
        raise click.ClickException(str(exc))


@migrate.command(name="embeddings")
@click.option("--from", "from_model", required=True,
              help="Current embedding model (informational, for verification)")
@click.option("--to", "to_model", required=True,
              help="New embedding model to re-embed with")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
def migrate_embeddings(from_model, to_model, dry_run):
    """Re-embed the entire knowledge base with a new embedding model.

    \b
    Exports all ChunkRecords, re-embeds each chunk text with the new
    model, and writes the new vectors back under the same vector IDs
    (in-place overwrite).  Updates kbvc.lock on completion.

    \b
    Example:
        kbvc migrate embeddings \\
            --from text-embedding-3-small \\
            --to gemini-embedding-001
        kbvc migrate embeddings \\
            --from text-embedding-3-small \\
            --to gemini-embedding-001 \\
            --dry-run
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.migrate_embeddings import run_migrate_embeddings
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        run_migrate_embeddings(repo, from_model, to_model, dry_run=dry_run)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc))


@migrate.command(name="schema")
@click.option("--dry-run", is_flag=True, help="Show what would change without writing")
def migrate_schema(dry_run):
    """Upgrade the kbvc.lock schema_version to the latest format.

    \b
    Phase 7: bumps the schema_version field in kbvc.lock from 1 → 2,
    which signals to other tools (kbvc push --rebuild, CI) that the
    lock file has been audited for the new format.

    \b
    Example:
        kbvc migrate schema
        kbvc migrate schema --dry-run
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    lock_path = repo.root / "kbvc.lock"
    if not lock_path.exists():
        raise click.ClickException(
            "kbvc.lock not found. Run `kbvc commit` first to generate it."
        )

    import re
    text = lock_path.read_text(encoding="utf-8")
    match = re.search(r"schema_version:\s*(\d+)", text)
    current_ver = int(match.group(1)) if match else 1
    target_ver = current_ver + 1

    click.echo(f"kbvc.lock schema_version: {current_ver} → {target_ver}")
    if dry_run:
        click.echo("Dry-run — no changes written.")
        return

    new_text = re.sub(
        r"(schema_version:\s*)\d+",
        f"\\g<1>{target_ver}",
        text,
    )
    lock_path.write_text(new_text, encoding="utf-8")
    click.echo(f"✓ kbvc.lock updated to schema_version: {target_ver}")


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc backend — VSAL schema management
# ═══════════════════════════════════════════════════════════════════════════════

@main.group()
def backend():
    """Vector Storage Abstraction Layer — schema init and info."""
    pass


@backend.command(name="init")
def backend_init():
    """Idempotently create the KBVC schema on the configured vector backend.

    \b
    Reads backend type and embedding dimensions from config, then calls
    initialize_schema() on the backend adapter.

    \b
    Example output:
        Detected:

          Backend:    pgvector
          Collection: kbvc
          Model:      gemini-embedding-001
          Dimensions: 3072

        Creating schema…
          ✓ kbvc_chunks
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.backend import run_backend_init
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        run_backend_init(repo)
    except (ValueError, click.ClickException):
        raise
    except Exception as exc:
        raise click.ClickException(str(exc))


@backend.command(name="info")
def backend_info():
    """Show current vector backend configuration (no credentials)."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.backend import run_backend_info
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    run_backend_info(repo)


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc explain — knowledge provenance chain
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.argument("vector_id")
def explain(vector_id):
    """Trace a vector ID to its full knowledge provenance chain.

    \b
    Shows: chunk → KO → commit → embedding model → relations + confidence.

    \b
    Example:
        kbvc explain main__caching-strategy__chunk_2
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.explain import run_explain
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    run_explain(repo, vector_id)


# ═══════════════════════════════════════════════════════════════════════════════
# kbvc promote — memory → Knowledge Object promotion
# ═══════════════════════════════════════════════════════════════════════════════

@main.command()
@click.argument("memory")
@click.option("--id", "ko_id", default=None,
              help="KO id (default: slugified from memory text)")
@click.option("--type", "ko_type", default="lesson",
              type=click.Choice(["lesson", "observation", "concept", "doc", "project"]),
              help="KO type (default: lesson)")
@click.option("--confidence", default=0.8, type=float,
              help="Confidence score 0.0–1.0 (default: 0.8 for promoted memories)")
@click.option("--source", default="agent",
              type=click.Choice(["human", "agent", "auto"]),
              help="Knowledge source (default: agent)")
@click.option("--tags", default="",
              help="Comma-separated tags")
def promote(memory, ko_id, ko_type, confidence, source, tags):
    """Promote a memory or observation into a staged Knowledge Object.

    \b
    Bridges ephemeral agent memory and permanent versioned knowledge.
    Creates a .md file in knowledge/promoted/ and stages it for commit.

    \b
    Example:
        kbvc promote "Dragonfly fixed the Redis OOM issue in May 2026" \\
            --id dragonfly-fix --type lesson --confidence 0.9 --source agent

        kbvc status          # review the staged file
        kbvc commit -m "promote: dragonfly observation"
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.promote import run_promote
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        run_promote(repo, memory, ko_id, ko_type, confidence, source, tags)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc))


@main.command()
@click.option("--volatility", default="slow",
              type=click.Choice(["live", "slow", "all"]),
              help="Which volatility tier to sync (default: slow)")
@click.option("--dry-run", is_flag=True, help="Show what would be committed without writing")
@click.option("--message", "-m", default=None, help="Custom commit message")
def sync(volatility, dry_run, message):
    """Stage + commit all modified KOs matching a volatility filter.

    \b
    Compares current file hashes against the last committed chunk hashes.
    Any KO whose content has changed is staged and committed automatically.
    Frozen KOs are always excluded.

    \b
    Volatility filter:
        live   — only `live` KOs
        slow   — `slow` and `live` KOs (default)
        all    — all non-frozen KOs

    \b
    Example:
        kbvc sync                        # commit changed slow + live KOs
        kbvc sync --volatility live      # only live KOs
        kbvc sync --dry-run              # preview without committing
        kbvc sync -m "nightly refresh"   # custom commit message
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.sync import run_sync
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        run_sync(repo, volatility=volatility, dry_run=dry_run, message=message)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 9 — kbvc contradict
# ═══════════════════════════════════════════════════════════════════════════════

@main.group()
def contradict():
    """Contradiction detection and resolution flow.

    \b
    Scans `contradicts` relations in the knowledge graph.
    A contradiction is marked resolved once a `supersedes` relation
    exists between the two KOs.

    \b
    Example workflow:
        kbvc contradict list
        kbvc link winner.md loser.md --type supersedes
        kbvc contradict resolve <rel_id>
    """
    pass


@contradict.command(name="list")
def contradict_list():
    """List all contradiction pairs and their resolution status."""
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.contradict import run_contradict_list
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    run_contradict_list(repo)


@contradict.command(name="resolve")
@click.argument("rel_id")
def contradict_resolve(rel_id):
    """Mark a contradiction as resolved (requires a supersedes relation).

    \b
    REL_ID is the relation ID (or prefix) of the `contradicts` relation
    to resolve.  Use `kbvc contradict list` to see relation IDs.
    """
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.commands.contradict import run_contradict_resolve
    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))
    try:
        run_contradict_resolve(repo, rel_id)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Phase 10: kbvc ask — AI-assisted knowledge Q&A with full provenance
# ---------------------------------------------------------------------------

@main.command()
@click.argument("question")
@click.option("--top-k", default=5, show_default=True,
              help="Number of KO chunks to retrieve.")
@click.option("--profile", default=None,
              help="Override the active retrieval profile.")
@click.option("--branch", "branch_override", default=None,
              help="Query a specific branch (default: current branch).")
@click.option("--show-ids", is_flag=True, default=False,
              help="Print raw vector IDs alongside results.")
def ask(question, top_k, profile, branch_override, show_ids):
    """Retrieve the most relevant knowledge chunks for a question.

    \b
    Embeds QUESTION, performs nearest-neighbour search across the active
    branch, and prints the top-K matching KO chunks with full provenance
    (KO id, version, commit SHA, similarity score).

    Use the output as grounded context for your LLM.  For a full audit
    trail of any returned chunk, run:

      kbvc explain <vector_id>

    \b
    Examples:
        kbvc ask "What is the caching strategy?"
        kbvc ask "Which projects use transformers?" --top-k 10
        kbvc ask "Summarise the auth flow" --branch feature/auth
    """
    from kbvc.backends import get_embed_backend, get_vectordb_backend
    from kbvc.core.repo import KbvcRepo, NotKBVCRepositoryError
    from kbvc.core.ko import KOStore
    from kbvc.core.chunker import parse_frontmatter, split_into_chunks

    try:
        repo = KbvcRepo.require()
    except NotKBVCRepositoryError as exc:
        raise click.ClickException(str(exc))

    config = repo.config()
    collection = config.get("vectordb.collection", "kbvc")
    branch = branch_override or repo.current_branch()

    try:
        embed = get_embed_backend(config)
        vdb = get_vectordb_backend(config)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # ── Embed the question ────────────────────────────────────────────────
    try:
        q_vec = embed.embed(question)
    except Exception as exc:
        raise click.ClickException(f"Embedding failed: {exc}")

    # ── Retrieve top-K chunks ─────────────────────────────────────────────
    results = vdb.query(
        collection, q_vec, top_k=top_k,
        filter={"branch": branch},
    )

    if not results:
        click.echo("No relevant knowledge found for that question.")
        click.echo(f"  branch: {branch}  |  collection: {collection}")
        return

    # ── Load KO source files to get chunk text ────────────────────────────
    ko_store = KOStore(repo.ko_store_path)

    context_parts = []
    for r in results:
        meta = r.get("metadata", {})
        ko_id = meta.get("ko_id", "")
        chunk_idx = int(meta.get("chunk_index", 0))
        ko = ko_store.get(ko_id)
        if not ko:
            continue
        src = repo.root / ko.path
        if not src.exists():
            continue
        try:
            content = src.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content)
            chunks = split_into_chunks(body, fm)
            chunk_text = chunks[chunk_idx].text if chunk_idx < len(chunks) else ""
        except Exception:
            chunk_text = ""
        context_parts.append({
            "vector_id": r.get("id", ""),
            "text": chunk_text,
            "ko_id": ko_id,
            "ko_version": meta.get("ko_version", ko.version),
            "commit_id": meta.get("commit_id", ""),
            "score": r.get("score", 0.0),
        })

    if not context_parts:
        click.echo("Found vector results but could not load chunk text.")
        click.echo("Make sure source files are present on disk.")
        return

    # ── Display ───────────────────────────────────────────────────────────
    W = 66
    click.echo(f"\n{'━' * W}")
    click.echo(f"  kbvc ask  ›  {question!r}")
    click.echo(f"  branch: {branch}  |  top-{len(context_parts)} chunks retrieved")
    click.echo(f"{'━' * W}\n")

    for i, part in enumerate(context_parts, 1):
        cid_short = part["commit_id"][:7] if part["commit_id"] else "unknown"
        score_pct = int(part["score"] * 100)
        header = (
            f"[{i}] {part['ko_id']}  "
            f"v{part['ko_version']}  "
            f"commit:{cid_short}  "
            f"score:{score_pct}%"
        )
        click.echo(click.style(header, bold=True))
        if show_ids:
            click.echo(f"    vector_id: {part['vector_id']}")
        text = part["text"]
        preview = text[:600] + ("…" if len(text) > 600 else "")
        click.echo(preview)
        click.echo()

    click.echo(f"{'━' * W}")
    click.echo("Use the context above with your LLM to generate a grounded answer.")
    click.echo("For full provenance of any chunk: kbvc explain <vector_id>")
    click.echo(f"{'━' * W}\n")
