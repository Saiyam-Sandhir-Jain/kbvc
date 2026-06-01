# kbvc/commands/explain.py
"""
kbvc explain <vector_id> — complete knowledge provenance chain.

Traces an answer back from vector ID all the way to the original commit,
including which KO it came from, what relations that KO has, and which
commit embedded it.

Output example:

  vector_id: main__caching-strategy__chunk_2
    ↓ chunk 2 of KO: caching-strategy @ v4
    ↓ committed: 2026-05-10  commit: a3f7c91
    ↓ message:   "update caching recommendations"
    ↓ embedded with: gemini-embedding-001 (3072 dims)
    ↓ stored in: pgvector

  Relations from caching-strategy:
    supported_by  benchmark-2026-05   [confidence: 1.0, source: human]
    supported_by  incident-17         [confidence: 1.0, source: human]
    supersedes    redis-config        [confidence: 1.0, source: human]

This is the provenance chain that makes KBVC auditable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kbvc.core.repo import KbvcRepo


def run_explain(repo: "KbvcRepo", vector_id: str) -> None:
    import click
    from kbvc.core.lineage import ChainTracer
    from kbvc.core.graph import RelationGraph

    # ── trace vector → commit ────────────────────────────────────────────────
    tracer = ChainTracer(repo.kbvc_dir)
    try:
        rec = tracer.trace(vector_id)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc))

    click.echo(f"vector_id:  {rec.vector_id}")
    click.echo(f"  ↓ chunk {rec.chunk_index} of KO: {rec.ko_id} @ v{rec.ko_version}")
    click.echo(f"  ↓ section: {rec.section}")
    click.echo(f"  ↓ committed:   {rec.commit_timestamp[:10]}  commit: {rec.commit_id[:7]}")
    click.echo(f"  ↓ message:     {rec.commit_message}")
    click.echo(f"  ↓ embedded with: {rec.embed_model} ({rec.embed_dims} dims)")
    click.echo(f"  ↓ stored in: {rec.vectordb_backend}")
    click.echo("")

    # ── relations from this KO ───────────────────────────────────────────────
    branch = repo.current_branch()
    graph = RelationGraph(repo.kbvc_dir / "graph", branch=branch)
    rels = [r for r in graph.relations if r.from_id == rec.ko_id or r.to_id == rec.ko_id]

    if rels:
        click.echo(f"Relations involving {rec.ko_id}:")
        for r in rels:
            direction = "→" if r.from_id == rec.ko_id else "←"
            other = r.to_id if r.from_id == rec.ko_id else r.from_id
            conf_str = f"confidence: {r.confidence:.2f}, source: {r.source}"
            click.echo(f"  {r.type:20s}  {direction} {other:30s}  [{conf_str}]")
    else:
        click.echo(f"No relations found for KO '{rec.ko_id}'.")
